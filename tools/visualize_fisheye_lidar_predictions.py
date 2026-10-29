#!/usr/bin/env python3
"""Render portable ROSA predictions over the corresponding LiDAR clouds."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rosa-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mmengine
import numpy as np


COLORS = {
    "Car": "#00e5ff",
    "Van": "#2979ff",
    "Truck": "#ffab00",
    "Bus": "#ff1744",
    "Pedestrian": "#76ff03",
    "Cyclist": "#d500f9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="portable predictions.json from checkpoint inference")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "all"),
                        default="val")
    parser.add_argument("--info", type=Path,
                        help="defaults to DATA_ROOT/rosa/fisheye3dod_infos_SPLIT.pkl")
    parser.add_argument("--score-threshold", type=float, default=0.10)
    parser.add_argument("--max-boxes", type=int, default=100)
    parser.add_argument("--bev-range", type=float, default=60.0)
    parser.add_argument("--point-stride", type=int, default=2)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--no-ground-truth", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--source-dataset", action="append", default=[], metavar="NAME=PATH",
        help=("override a source bag dataset from summary.json; repeat for "
              "16_09 and 30_09 when visualizing on another machine"))
    return parser.parse_args()


def parse_source_overrides(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=PATH, received {value!r}")
        name, raw_path = value.split("=", 1)
        result[name] = Path(raw_path).expanduser().resolve()
    return result


def source_datasets(data_root: Path,
                    overrides: dict[str, Path]) -> dict[str, Path]:
    summary_path = data_root / "summary.json"
    sources = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        for sequence in summary.get("sequences", []):
            sources[str(sequence["name"])] = Path(
                sequence["dataset"]).expanduser()
    sources.update(overrides)
    return sources


def resolve_lidar_path(data_root: Path, info_path: Path, sample: dict,
                       sources: dict[str, Path]) -> Path:
    raw_path = Path(str(sample["pc_info"]["pts_path"]))
    sequence = str(sample.get("metainfo", {}).get("sequence_id", ""))
    basename = raw_path.name
    candidates = [
        data_root / raw_path,
        info_path.parent / raw_path,
        data_root / "lidar_pcd" / basename,
    ]
    if sequence in sources:
        source = sources[sequence]
        candidates.extend((
            source / "lidar_pcd" / basename,
            source / "fisheye3dod_inference_2hz" / raw_path,
            source / raw_path,
        ))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    tried = "\n  ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(
        f"LiDAR cloud for sequence={sequence!r}, token={sample['token']} "
        f"was not found. Tried:\n  {tried}\n"
        "On another machine, pass --source-dataset NAME=/path/to/dataset.")


def load_points(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    marker = b"DATA binary\n"
    marker_index = payload.find(marker)
    if marker_index >= 0:
        payload = payload[marker_index + len(marker):]
    points = np.frombuffer(payload, dtype="<f4")
    if points.size % 4:
        raise ValueError(f"{path}: expected binary float32 XYZI point cloud")
    points = points.reshape(-1, 4)
    valid = np.isfinite(points).all(axis=1)
    valid &= np.linalg.norm(points[:, :3], axis=1) > 0.1
    return points[valid]


def box_polygon(box: np.ndarray) -> np.ndarray:
    length, width = box[3], box[4]
    local = np.asarray([
        [length / 2, width / 2],
        [length / 2, -width / 2],
        [-length / 2, -width / 2],
        [-length / 2, width / 2],
        [length / 2, width / 2],
    ])
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + box[:2]


def draw_box(axis, box: np.ndarray, color: str, line_style: str,
             line_width: float, label: str | None) -> None:
    polygon = box_polygon(box)
    axis.plot(polygon[:, 0], polygon[:, 1], color=color,
              linestyle=line_style, linewidth=line_width, alpha=0.95)
    heading = box[:2] + np.asarray([np.cos(box[6]), np.sin(box[6])]) * (
        box[3] / 2)
    axis.plot([box[0], heading[0]], [box[1], heading[1]], color=color,
              linestyle=line_style, linewidth=line_width + 0.4)
    if label:
        axis.text(box[0], box[1], label, color=color, fontsize=6,
                  bbox={"facecolor": "#05070a", "edgecolor": "none",
                        "alpha": 0.65, "pad": 0.8})


def selected_detections(frame: dict, threshold: float,
                        max_boxes: int) -> list[tuple[str, np.ndarray, float]]:
    detections = [
        (str(name), np.asarray(box, dtype=float), float(score))
        for name, box, score in zip(
            frame["names"], frame["boxes_3d"], frame["scores"])
        if float(score) >= threshold
    ]
    detections.sort(key=lambda item: item[2], reverse=True)
    return detections[:max_boxes]


def render_frame(output_path: Path, points: np.ndarray, sample: dict,
                 frame: dict, dataset_index: int, args: argparse.Namespace,
                 checkpoint_name: str) -> Counter:
    detections = selected_detections(
        frame, args.score_threshold, args.max_boxes)
    limit = args.bev_range
    visible = ((np.abs(points[:, 0]) <= limit) &
               (np.abs(points[:, 1]) <= limit) &
               (points[:, 2] >= -4.0) & (points[:, 2] <= 4.0))
    draw = points[visible][::max(1, args.point_stride)]

    figure, axis = plt.subplots(figsize=(10, 10), facecolor="#05070a")
    axis.set_facecolor("#05070a")
    if len(draw):
        axis.scatter(draw[:, 0], draw[:, 1], s=0.22, c=draw[:, 2],
                     cmap="gray", vmin=-2.5, vmax=1.5, alpha=0.72,
                     linewidths=0)

    if not args.no_ground_truth:
        annos = sample.get("annos", {})
        for name, box in zip(annos.get("gt_names", []),
                             annos.get("gt_boxes", [])):
            draw_box(axis, np.asarray(box, dtype=float), "#ffffff", "--",
                     0.9, None)

    counts = Counter()
    for name, box, score in detections:
        counts[name] += 1
        draw_box(axis, box, COLORS.get(name, "#ffee58"), "-", 1.7,
                 f"{name} {score:.2f}")

    axis.scatter([0], [0], marker="^", s=75, color="#00ff9d",
                 edgecolors="black", linewidths=0.7, zorder=10)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal")
    axis.set_xlabel("LiDAR X [m]", color="white")
    axis.set_ylabel("LiDAR Y [m]", color="white")
    axis.tick_params(colors="#b0bec5")
    axis.grid(color="white", alpha=0.08, linewidth=0.5)
    title = (
        f"{checkpoint_name} | {args.split} index {dataset_index} | "
        f"{sample['metainfo'].get('sequence_id', '')} frame "
        f"{sample['metainfo'].get('frame_id', '')} | token {sample['token']}\n"
        f"predictions: {len(detections)} at score >= "
        f"{args.score_threshold:.2f} (solid/color)"
    )
    if not args.no_ground_truth:
        title += " | ground truth: dashed/white"
    axis.set_title(title, color="white", fontsize=10)
    figure.tight_layout()
    figure.savefig(output_path, dpi=120, facecolor=figure.get_facecolor())
    plt.close(figure)
    return counts


def make_video(frames_dir: Path, output_dir: Path, fps: float) -> Path | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg is unavailable; skipping video")
        return None
    video = output_dir / "lidar_predictions.mp4"
    command = [
        ffmpeg, "-y", "-loglevel", "warning", "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.png"), "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-crf", "20", str(video),
    ]
    subprocess.run(command, check=True)
    return video


def main() -> None:
    args = parse_args()
    if args.frame_step < 1 or args.point_stride < 1 or args.max_boxes < 1:
        raise ValueError("frame-step, point-stride, and max-boxes must be >= 1")
    data_root = args.data_root.expanduser().resolve()
    predictions_path = args.predictions.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    info_path = (args.info or data_root / "rosa" /
                 f"fisheye3dod_infos_{args.split}.pkl").resolve()
    info = mmengine.load(info_path)
    samples = info["data_list"]
    prediction_payload = json.loads(predictions_path.read_text())
    frames = prediction_payload["frames"]
    if len(frames) != len(samples):
        raise RuntimeError(
            f"prediction/sample count mismatch: {len(frames)} != {len(samples)}")

    end = len(samples) - 1 if args.end_frame is None else args.end_frame
    if args.start_frame < 0 or end < args.start_frame or end >= len(samples):
        raise IndexError(
            f"require 0 <= start-frame <= end-frame < {len(samples)}")
    indices = list(range(args.start_frame, end + 1, args.frame_step))
    sources = source_datasets(
        data_root, parse_source_overrides(args.source_dataset))
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = Path(prediction_payload.get(
        "checkpoint", predictions_path.parent.name)).stem
    class_counts = Counter()
    for render_index, dataset_index in enumerate(
            mmengine.track_iter_progress(indices)):
        sample = samples[dataset_index]
        point_path = resolve_lidar_path(
            data_root, info_path, sample, sources)
        points = load_points(point_path)
        class_counts.update(render_frame(
            frames_dir / f"frame_{render_index:06d}.png", points, sample,
            frames[dataset_index], dataset_index, args, checkpoint_name))

    video = None if args.no_video else make_video(
        frames_dir, output_dir, args.video_fps)
    summary = {
        "checkpoint": checkpoint_name,
        "predictions": str(predictions_path),
        "annotation": str(info_path),
        "frames": len(indices),
        "dataset_indices": indices,
        "score_threshold": args.score_threshold,
        "max_boxes_per_frame": args.max_boxes,
        "bev_range_m": args.bev_range,
        "ground_truth_overlay": not args.no_ground_truth,
        "class_counts": dict(class_counts),
        "frames_dir": str(frames_dir),
        "video": None if video is None else str(video),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
