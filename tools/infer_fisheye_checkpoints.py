#!/usr/bin/env python3
"""Evaluate and export predictions from one or more ROSA checkpoints.

Each checkpoint runs in an isolated subprocess so CUDA memory, registries, and
MMEngine state cannot leak between evaluations.  The official Fisheye3DOD
metrics, raw MMEngine results, portable JSON predictions, and a comparison
summary are written below ``--output-dir``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import re
import pickle
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
ROSA_ROOT = REPO_ROOT / "projects" / "ROSA"
DEFAULT_CONFIG = (
    ROSA_ROOT / "configs" / "fisheye3dod" /
    "rosa_vov99_dual_export_24e.py"
)
DEFAULT_CHECKPOINTS = (
    ROSA_ROOT / "weights" / "epoch_24.pth",
)
CLASSES = ("Car", "Van", "Truck", "Bus", "Pedestrian", "Cyclist")
METRIC_PATTERN = re.compile(
    r"fisheye_bev/(?P<name>[A-Za-z0-9_@.]+):\s*(?P<value>[-+0-9.eE]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--checkpoint", type=Path, action="append",
        help="repeatable; defaults to the two epoch-16 best files and epoch 24")
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get(
            "ROSA_DATA_ROOT",
            REPO_ROOT / "data" / "Custom" / "combined_improved_detzero_rosa",
        )))
    parser.add_argument(
        "--official-root", type=Path,
        default=Path(os.environ.get(
            "FISHEYE3DOD_OFFICIAL_ROOT",
            REPO_ROOT / "projects" / "Fisheye3DOD",
        )))
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROSA_ROOT / "work_dirs" / "checkpoint_inference")
    parser.add_argument("--device", default="0", help="CUDA device index")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--score-threshold", type=float, default=0.05,
        help="portable JSON filter only; official metrics use unfiltered output")
    parser.add_argument(
        "--lidar-score-threshold", type=float, default=0.10,
        help="score threshold used only by LiDAR visualizations")
    parser.add_argument("--lidar-bev-range", type=float, default=60.0)
    parser.add_argument("--lidar-point-stride", type=int, default=2)
    parser.add_argument("--lidar-frame-step", type=int, default=1)
    parser.add_argument("--lidar-video-fps", type=float, default=2.0)
    parser.add_argument(
        "--lidar-source-dataset", action="append", default=[],
        metavar="NAME=PATH",
        help="portable source bag override forwarded to the LiDAR visualizer")
    parser.add_argument("--no-lidar-ground-truth", action="store_true")
    parser.add_argument("--no-lidar-video", action="store_true")
    parser.add_argument("--no-lidar-visualization", action="store_true")
    parser.add_argument("--allow-checkpoint-contract-mismatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} was not found: {path}")
    return path


def stream_command(command: list[str], environment: dict[str, str],
                   log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def parse_metrics(log_path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for match in METRIC_PATTERN.finditer(log_path.read_text(errors="replace")):
        metrics[match.group("name")] = float(match.group("value"))
    return metrics


def field(container: Any, name: str, default: Any = None) -> Any:
    if hasattr(container, name):
        return getattr(container, name)
    if isinstance(container, dict):
        return container.get(name, default)
    return default


def to_list(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "tensor"):
        value = value.tensor
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def load_cpu_results(path: Path):
    """Read our trusted local result pickle, including CUDA tensor storages."""
    import torch

    class CPUUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda data: torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
            return super().find_class(module, name)

    with path.open('rb') as stream:
        return CPUUnpickler(stream).load()


def export_portable_json(raw_path: Path, destination: Path,
                         checkpoint: Path,
                         score_threshold: float) -> tuple[int, int, dict[str, int]]:
    import numpy as np
    # The OpenMMLab environment can contain sklearn 0.19, which is imported
    # while unpickling MMDet3D samples and still uses removed NumPy aliases.
    if "float" not in np.__dict__:
        np.float = float  # type: ignore[attr-defined]
    if "int" not in np.__dict__:
        np.int = int  # type: ignore[attr-defined]
    if "bool" not in np.__dict__:
        np.bool = bool  # type: ignore[attr-defined]
    import mmengine

    results = load_cpu_results(raw_path)
    frames = []
    total_boxes = 0
    class_counts: Counter[str] = Counter()
    for index, sample in enumerate(results):
        prediction = field(sample, "pred_instances_3d")
        if prediction is None:
            prediction = field(sample, "pred_instances")
        if prediction is None:
            raise KeyError(f"prediction {index} has no pred_instances_3d")

        box_object = field(prediction, "bboxes_3d")
        if hasattr(box_object, 'gravity_center'):
            box_tensor = box_object.tensor.detach().cpu().clone()
            box_tensor[:, :3] = box_object.gravity_center.detach().cpu()
            boxes = box_tensor.tolist()
        else:
            raise TypeError('Expected 3D box objects with an explicit gravity_center')
        scores = to_list(field(prediction, "scores_3d"))
        labels = [int(value) for value in to_list(
            field(prediction, "labels_3d"))]
        if not (len(boxes) == len(scores) == len(labels)):
            raise ValueError(f"prediction {index} has inconsistent lengths")
        keep = [i for i, score in enumerate(scores)
                if float(score) >= score_threshold]
        metainfo = field(sample, "metainfo", {}) or {}
        if isinstance(sample, dict):
            metainfo = {**sample, **metainfo}
        frames.append({
            "index": index,
            "sample_idx": metainfo.get("sample_idx"),
            "token": str(metainfo.get("token", "")),
            "boxes_3d": [boxes[i] for i in keep],
            "scores": [float(scores[i]) for i in keep],
            "labels": [labels[i] for i in keep],
            "names": [
                CLASSES[labels[i]] if 0 <= labels[i] < len(CLASSES)
                else f"class_{labels[i]}"
                for i in keep
            ],
        })
        total_boxes += len(keep)
        class_counts.update(
            CLASSES[labels[i]] if 0 <= labels[i] < len(CLASSES)
            else f"class_{labels[i]}"
            for i in keep
        )
    payload = {
        "checkpoint": str(checkpoint),
        "classes": CLASSES,
        "format_version": 2,
        "box_origin": "gravity_center",
        "score_threshold": score_threshold,
        "frames": frames,
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return len(frames), total_boxes, dict(class_counts)


def main() -> None:
    args = parse_args()
    config = require_file(args.config, "config")
    data_root = args.data_root.expanduser().resolve()
    official_root = args.official_root.expanduser().resolve()
    checkpoints = tuple(args.checkpoint or DEFAULT_CHECKPOINTS)
    checkpoints = tuple(require_file(path, "checkpoint") for path in checkpoints)
    annotation = require_file(
        data_root / "rosa" / f"fisheye3dod_infos_{args.split}.pkl",
        f"{args.split} annotation")
    require_file(
        data_root / "rosa" / "anchors" / "kmeans300_seed3407.npy",
        "ROSA anchor")
    require_file(
        official_root / "dataset" / "fisheye3dod_metric.py",
        "official Fisheye3DOD metric")

    output_dir = args.output_dir.expanduser().resolve()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    python_path = os.pathsep.join((
        str(REPO_ROOT), str(ROSA_ROOT), str(official_root),
        os.environ.get("PYTHONPATH", ""),
    ))
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": args.device,
        "FISHEYE3DOD_OFFICIAL_ROOT": str(official_root),
        "PYTHONPATH": python_path,
        "ROSA_DATA_ROOT": str(data_root),
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
    })

    comparison = {
        "config": str(config),
        "data_root": str(data_root),
        "annotation": str(annotation),
        "split": args.split,
        "checkpoints": [],
    }
    for checkpoint in checkpoints:
        name = checkpoint.stem
        checkpoint_dir = output_dir / name
        raw_path = checkpoint_dir / "predictions.pkl"
        json_path = checkpoint_dir / "predictions.json"
        log_path = checkpoint_dir / "evaluation.log"
        work_dir = checkpoint_dir / "work"
        command = [
            sys.executable,
            str(ROSA_ROOT / "tools" / "run.py"),
            "evaluate",
            str(config),
            str(checkpoint),
            "--dump", str(raw_path),
            "--work-dir", str(work_dir),
        ]
        if args.allow_checkpoint_contract_mismatch:
            command.append("--allow-checkpoint-contract-mismatch")
        command.extend([
            "--cfg-options",
            f"test_dataloader.dataset.ann_file={annotation}",
            "test_dataloader.batch_size=1",
            f"test_dataloader.num_workers={args.num_workers}",
        ])
        print("\n$ " + " ".join(command), flush=True)
        if args.dry_run:
            continue
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stream_command(command, environment, log_path)
        metrics = parse_metrics(log_path)
        frame_count, total_boxes, predicted_class_counts = export_portable_json(
            raw_path, json_path, checkpoint, args.score_threshold)
        lidar_visualization = None
        if not args.no_lidar_visualization:
            lidar_dir = checkpoint_dir / "lidar_visualization"
            visualization_command = [
                sys.executable,
                str(ROSA_ROOT / "tools" /
                    "visualize_fisheye_lidar_predictions.py"),
                "--data-root", str(data_root),
                "--predictions", str(json_path),
                "--output-dir", str(lidar_dir),
                "--split", args.split,
                "--score-threshold", str(args.lidar_score_threshold),
                "--bev-range", str(args.lidar_bev_range),
                "--point-stride", str(args.lidar_point_stride),
                "--frame-step", str(args.lidar_frame_step),
                "--video-fps", str(args.lidar_video_fps),
            ]
            if args.no_lidar_ground_truth:
                visualization_command.append("--no-ground-truth")
            if args.no_lidar_video:
                visualization_command.append("--no-video")
            for source_dataset in args.lidar_source_dataset:
                visualization_command.extend(
                    ("--source-dataset", source_dataset))
            print("\n$ " + " ".join(visualization_command), flush=True)
            subprocess.run(
                visualization_command, cwd=REPO_ROOT,
                env=environment, check=True)
            lidar_visualization = {
                "frames": str(lidar_dir / "frames"),
                "video": (None if args.no_lidar_video else
                          str(lidar_dir / "lidar_predictions.mp4")),
                "summary": str(lidar_dir / "summary.json"),
            }

        comparison["checkpoints"].append({
            "name": name,
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "metrics": metrics,
            "prediction_frames": frame_count,
            "exported_boxes": total_boxes,
            "predicted_class_counts": predicted_class_counts,
            "raw_predictions": str(raw_path),
            "json_predictions": str(json_path),
            "evaluation_log": str(log_path),
            "lidar_visualization": lidar_visualization,
        })
        (output_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2) + "\n")

    if not args.dry_run:
        print("\nComparison written to", output_dir / "comparison.json")


if __name__ == "__main__":
    main()
