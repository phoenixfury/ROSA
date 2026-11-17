#!/usr/bin/env python3
"""Render dense chronological clips comparing four-bag epoch6 against E007.

Reuses checksum-verified native predictions; infers only missing selected frames.
Videos are qualitative diagnostic clips, not a validation-only benchmark.
"""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import subprocess
import sys

from compare_fourbag_training import ROOT, ROSA, dump, mmengine, np, sha
from render_rosa_retraining_comparison import writer
from render_three_model_fisheye_lidar_comparison import (
    OcamCamera, cv2, load_points, parse_sources, render_model, resolve_lidar,
)

MODELS = ('previous_best', 'new_best6')
SEQUENCES = ('16_09', '17_09', '25_09', '30_09')
LABELS = dict(previous_best='BASELINE E007 epoch1', new_best6='FOUR-BAG epoch6')


def dense_window(rows, count):
    """Rank contiguous windows by GT objects <=48m, independent of model output."""
    rows = sorted(rows, key=lambda r: int(r['metainfo']['acquisition_timestamp_ns']))
    if len(rows) < count:
        raise ValueError(f'Need {count} distinct frames; found {len(rows)}')
    ids = np.asarray([int(r['metainfo']['source_frame_id']) for r in rows])
    times = np.asarray([int(r['metainfo']['acquisition_timestamp_ns']) for r in rows])
    breaks = (np.diff(ids) != 1) | (np.diff(times) <= 0) | (np.diff(times) > 1_500_000_000)
    breaks_prefix = np.r_[0, np.cumsum(breaks)]
    density = np.asarray([sum(np.linalg.norm(b[:2]) <= 48 for b in r['annos']['gt_boxes']) for r in rows])
    prefix = np.r_[0, np.cumsum(density)]
    candidates = [(int(prefix[i+count]-prefix[i]), -i, i)
                  for i in range(len(rows)-count+1)
                  if breaks_prefix[i+count-1] == breaks_prefix[i]]
    if not candidates:
        raise ValueError(f'No contiguous {count}-frame acquisition-time window')
    _, _, first = max(candidates)
    return rows[first:first+count]


def cache_document(analysis, tag, plan):
    folder = analysis/tag
    done = json.loads((folder/'inference_complete.json').read_text())
    path = folder/'predictions.json'
    if (done['checkpoint_sha256'] != plan['models'][tag]['checkpoint_sha256'] or
            done['union_sha256'] != plan['union_sha256'] or
            done['predictions_sha256'] != sha(path)):
        raise ValueError(f'Invalid cached prediction provenance: {tag}')
    payload = json.loads(path.read_text())
    if payload['box_origin'] != 'gravity_center':
        raise ValueError('Expected explicit gravity-center predictions')
    frames = {str(r['token']): r for r in payload['frames']}
    if len(frames) != len(payload['frames']):
        raise ValueError('Duplicate cached tokens')
    return payload, frames


def prepare(args):
    original = json.loads((args.analysis/'plan.json').read_text())
    if sha(Path(original['union'])) != original['union_sha256']:
        raise ValueError('Original evaluation annotations changed')
    data = Path(original['data_root'])
    all_path = data/'rosa/fisheye3dod_infos_all.pkl'
    document = mmengine.load(all_path)
    groups = defaultdict(list)
    for row in document['data_list']:
        groups[str(row['metainfo']['sequence_id'])].append(row)
    clips = {s: dense_window(groups[s], args.frames_per_bag) for s in SEQUENCES}
    selected = [row for rows in clips.values() for row in rows]
    tokens = [str(r['token']) for r in selected]
    if len(set(tokens)) != len(tokens):
        raise ValueError('Duplicate selected frames')
    caches = [set(cache_document(args.analysis, t, original)[1]) for t in MODELS]
    cached_tokens = set.intersection(*caches)
    prior = {str(r['token']): r for r in mmengine.load(original['union'])['data_list']}
    # Reject cache reuse if geometry or image bindings have changed.
    for row in selected:
        token = str(row['token'])
        if token not in cached_tokens:
            continue
        old = prior[token]
        if not np.array_equal(row['annos']['gt_boxes'], old['annos']['gt_boxes']):
            raise ValueError(f'Annotation mismatch for {token}')
        for name, camera in row['cam_info']['cam_fisheye'].items():
            old_camera = old['cam_info']['cam_fisheye'][name]
            if camera['cam_path'] != old_camera['cam_path'] or not np.array_equal(camera['lidar2cam'], old_camera['lidar2cam']):
                raise ValueError(f'Calibration/image binding mismatch: {token}/{name}')
    missing = [r for r in selected if str(r['token']) not in cached_tokens]
    args.output.mkdir(parents=True, exist_ok=True)
    selected_path = args.output/'selected_infos.pkl'
    mmengine.dump(dict(metainfo=document.get('metainfo', {}), data_list=selected), selected_path)
    missing_root = args.output/'missing_inference'
    missing_root.mkdir(exist_ok=True)
    missing_path = missing_root/'evaluation_union.pkl'
    mmengine.dump(dict(metainfo=document.get('metainfo', {}), data_list=missing), missing_path)
    inference = dict(data_root=str(data), union=str(missing_path), union_sha256=sha(missing_path),
                     union_frames=len(missing), models={t: original['models'][t] for t in MODELS})
    dump(missing_root/'plan.json', inference)
    clip_metadata = {}
    for sequence, rows in clips.items():
        clip_metadata[sequence] = dict(
            frames=len(rows), source_frame_start=int(rows[0]['metainfo']['source_frame_id']),
            source_frame_end=int(rows[-1]['metainfo']['source_frame_id']),
            split_counts=dict(Counter(r['metainfo']['temporal_split'] for r in rows)),
            acquisition_duration_s=(int(rows[-1]['metainfo']['acquisition_timestamp_ns'])-
                                    int(rows[0]['metainfo']['acquisition_timestamp_ns']))/1e9,
            mean_gt_objects_within48m=float(np.mean([sum(np.linalg.norm(b[:2]) <= 48 for b in r['annos']['gt_boxes']) for r in rows])),
            tokens=[str(r['token']) for r in rows])
    manifest = dict(analysis=str(args.analysis), data_root=str(data), original_union_sha256=original['union_sha256'],
        all_info=str(all_path), all_info_sha256=sha(all_path), selected_info=str(selected_path),
        selected_info_sha256=sha(selected_path), frames=len(selected), frames_per_bag=args.frames_per_bag,
        missing_inference_root=str(missing_root), missing_frames=len(missing), cached_frames=len(selected)-len(missing),
        selection='Densest contiguous acquisition-time window per bag; max GT count within48m, not model score.',
        diagnostic_not_validation_only=True, models=inference['models'], model_labels=LABELS, clips=clip_metadata,
        score_threshold=args.score_threshold, additional_nms=False, fps=args.fps,
        bev_range=args.bev_range, point_stride=args.point_stride,
        box_origin='gravity_center', decoder='Same quality-ranked, class-wise rotated NMS0.5 decoder',
        layout='Left: baseline E007. Right: four-bag epoch6. Each side: LiDAR BEV and FRONT/LEFT/RIGHT/REAR.',
        ground_truth='White dashed DetZero pseudo-labels, not independent manual annotation.')
    dump(args.output/'manifest.json', manifest)
    print(json.dumps(dict(frames=len(selected), missing=len(missing),
        clips={s:{k:v for k,v in clip.items() if k != 'tokens'} for s,clip in clip_metadata.items()}), indent=2), flush=True)


def infer(args):
    manifest = json.loads((args.output/'manifest.json').read_text())
    if manifest['missing_frames'] == 0:
        return
    root = Path(manifest['missing_inference_root'])
    for tag in MODELS:
        log = root/f'{tag}.log'
        print(f'Inferring {tag}: {manifest["missing_frames"]} missing frames; log {log}', flush=True)
        with log.open('w') as stream:
            subprocess.run([sys.executable, str(ROSA/'tools/compare_fourbag_training.py'),
                '--stage', 'infer-one', '--model', tag, '--output', str(root)],
                stdout=stream, stderr=subprocess.STDOUT, check=True)


def render(args):
    cv2.setNumThreads(2)
    manifest = json.loads((args.output/'manifest.json').read_text())
    if sha(Path(manifest['selected_info'])) != manifest['selected_info_sha256']:
        raise ValueError('Selected annotations changed')
    plan = json.loads((args.analysis/'plan.json').read_text())
    data = Path(manifest['data_root'])
    info = Path(manifest['selected_info'])
    rows = mmengine.load(info)['data_list']
    wanted = {str(r['token']) for r in rows}
    predictions = {}
    for tag in MODELS:
        payload, frames = cache_document(args.analysis, tag, plan)
        if manifest['missing_frames']:
            missing_root = Path(manifest['missing_inference_root'])
            extra_plan = json.loads((missing_root/'plan.json').read_text())
            _, extra = cache_document(missing_root, tag, extra_plan)
            frames.update(extra)
        if not wanted <= set(frames):
            raise ValueError(f'Missing predictions for {tag}')
        selected = {k:copy.deepcopy(v) for k,v in payload.items() if k != 'frames'}
        selected['frames'] = [frames[str(r['token'])] for r in rows]
        dump(args.output/f'{tag}_selected_predictions.json', selected)
        predictions[tag] = {token: [(str(name), np.asarray(box, dtype=float)[:7], float(score))
            for name, box, score in zip(frames[token]['names'], frames[token]['boxes_3d'], frames[token]['scores'])]
            for token in wanted}
    args.score_threshold = manifest['score_threshold']
    args.bev_range = manifest.get('bev_range', args.bev_range)
    args.point_stride = manifest.get('point_stride', args.point_stride)
    args.no_ground_truth = False
    ocam = OcamCamera(filename=str(data/'calib_results.txt'), fov=220.)
    sources = parse_sources(data, [])
    metrics = json.loads((args.analysis/'comparison.json').read_text())['models']
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row['metainfo']['sequence_id'])].append(row)
    videos = []
    for sequence in SEQUENCES:
        destination = args.output/f'{sequence}_baseline_vs_epoch6_{manifest["frames_per_bag"]}frames.mp4'
        if destination.exists():
            raise FileExistsError(f'Refusing to replace completed clip: {destination}')
        temporary = destination.with_name(destination.stem+'.partial.mp4')
        process = writer(temporary, 3200, 960, manifest['fps'])
        try:
            for index, sample in enumerate(grouped[sequence]):
                points = load_points(resolve_lidar(data, info, sample, sources))
                panels = [render_model(sample, points, predictions[tag][str(sample['token'])], ocam,
                    data, args, LABELS[tag], metrics[tag]['combined_validation_674']['metrics']) for tag in MODELS]
                canvas = np.full((960, 3200, 3), 15, dtype=np.uint8)
                canvas[60:] = np.concatenate(panels, axis=1)
                split = sample['metainfo']['temporal_split'].upper()
                label = (f'{sequence} | {index+1}/{len(grouped[sequence])} | {split} example | '
                         f'nominal2Hz input, {manifest["fps"]:g}fps playback | score>={args.score_threshold:g} both models | '
                         'shown mAP/NDS: full674-val, NOT this clip | same decoder')
                cv2.putText(canvas, label, (25, 35), cv2.FONT_HERSHEY_SIMPLEX, .78, (240, 240, 240), 2, cv2.LINE_AA)
                process.stdin.write(canvas.tobytes())
                if index == 0:
                    cv2.imwrite(str(args.output/f'{sequence}_preview.jpg'), canvas)
                if (index+1) % 10 == 0:
                    print(f'Render {sequence}: {index+1}/{len(grouped[sequence])}', flush=True)
        finally:
            process.stdin.close()
            code = process.wait()
        if code:
            raise RuntimeError(f'ffmpeg failed for {sequence}: {code}')
        temporary.replace(destination)
        videos.append(destination)
        dump(args.output/'status.json', dict(completed_bags=[p.name for p in videos]))
    # Paths contain only fixed safe names; ffmpeg concat demuxer stream-copies.
    concat = args.output/'clips.ffconcat'
    concat.write_text('ffconcat version 1.0\n'+''.join(f"file '{p.name}'\n" for p in videos))
    combined = args.output/'all4bags_baseline_vs_epoch6.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '1',
        '-i', str(concat), '-c', 'copy', '-movflags', '+faststart', str(combined)], check=True)
    checks = {}
    for path in [*videos, combined]:
        probe = subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,nb_read_frames,r_frame_rate,duration', '-of', 'json', str(path)], text=True)
        row = json.loads(probe)['streams'][0]
        expected = manifest['frames'] if path == combined else manifest['frames_per_bag']
        if int(row['nb_read_frames']) != expected:
            raise RuntimeError(f'Unexpected decoded frame count: {path}: {row}')
        checks[path.name] = dict(row, bytes=path.stat().st_size)
    manifest.update(videos=[str(p) for p in videos], combined_video=str(combined), validation=checks)
    dump(args.output/'manifest.json', manifest)
    dump(args.output/'status.json', dict(phase='complete', combined_video=str(combined)))
    print(json.dumps(dict(videos=manifest['videos'], combined=str(combined), validation=checks), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=ROSA/'work_dirs/combined_4bags_analysis_20261107')
    parser.add_argument('--output', type=Path, default=ROSA/'work_dirs/combined_4bags_analysis_20261107/videos_100_per_bag')
    parser.add_argument('--stage', choices=('all', 'prepare', 'infer', 'render'), default='all')
    parser.add_argument('--frames-per-bag', type=int, default=100)
    parser.add_argument('--score-threshold', type=float, default=.1)
    parser.add_argument('--fps', type=float, default=2.)
    parser.add_argument('--bev-range', type=float, default=50.)
    parser.add_argument('--point-stride', type=int, default=3)
    args = parser.parse_args()
    args.analysis = args.analysis.resolve();args.output = args.output.resolve()
    if args.frames_per_bag <= 0 or args.fps <= 0 or not 0 <= args.score_threshold <= 1:
        parser.error('Invalid frame count, fps or score threshold')
    if args.stage in ('all', 'prepare'):prepare(args)
    if args.stage in ('all', 'infer'):infer(args)
    if args.stage in ('all', 'render'):render(args)


if __name__ == '__main__':
    main()
