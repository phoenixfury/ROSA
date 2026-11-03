#!/usr/bin/env python3
"""Fail-closed source sidecar and LiDAR-origin-only pseudo-motion audit.

Does not rewrite PKLs or source annotations. Every combined frame and every
ordered object is verified before any sidecar is published. LiDAR-origin
centers are still DetZero/Kalman pseudo-labels, not independent physical GT.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import asdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
AUDIT_PATH = ROOT/'projects/ROSA/tools/audit_temporal_motion_targets.py'
SPEC = importlib.util.spec_from_file_location('rosa_provenance_motion_audit', AUDIT_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_pickle(path):
    payload = path.read_bytes()
    return pickle.loads(payload)['data_list'], hashlib.sha256(payload).hexdigest()


def export_taxonomy():
    """Read literal export taxonomy without importing heavy converter modules."""
    path = ROOT/'tools/custom_data/build_improved_detzero_rosa_dataset.py'
    values = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in ('CLASSES', 'CLASS_MAP'):
                values[target.id] = ast.literal_eval(node.value)
    require(set(values) == {'CLASSES', 'CLASS_MAP'}, 'Exporter taxonomy cannot be verified')
    return values['CLASSES'], values['CLASS_MAP'], hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frame(sample, source, majority, forward, reverse):
    """Validate ordered object identity exactly; never infer geometric matches."""
    meta, annotations = sample['metainfo'], sample['annos']
    sequence, token = str(meta['sequence_id']), str(sample['token'])
    prefix = f'{sequence} token={token}'
    require(str(source['timestamp_ns']) == token, prefix+': frame token mismatch')
    require(int(source['frame_id']) == int(meta['source_frame_id']), prefix+': source frame ID mismatch')
    require(int(source['lidar_sensor_timestamp_ns']) == int(meta['acquisition_timestamp_ns']),
            prefix+': acquisition timestamp mismatch')
    objects = source['objects']
    boxes = np.asarray(annotations['gt_boxes'], dtype=np.float32).reshape(-1, 7)
    ids = np.asarray(annotations['gt_ids'], dtype=np.int64)
    names = np.asarray(annotations['gt_names']).astype(str)
    source_boxes = np.asarray([obj['box'] for obj in objects], dtype=np.float32).reshape(-1, 7)
    require(len(ids) == len(boxes) == len(names) == len(objects), prefix+': object count mismatch')
    require(len(set(ids.tolist())) == len(ids), prefix+': duplicate combined per-frame IDs')
    require(np.array_equal(source_boxes, boxes), prefix+': ordered float32 boxes differ; no fallback matching')
    source_ids = [int(obj['track_id']) for obj in objects]
    require(len(set(source_ids)) == len(source_ids), prefix+': duplicate source per-frame IDs')
    records = []
    for index, (obj, combined_id, name) in enumerate(zip(objects, ids.tolist(), names.tolist())):
        raw_id = int(obj['track_id'])
        require(name == majority[raw_id], prefix+f': mapped majority class mismatch at object {index}')
        require(raw_id not in forward or forward[raw_id] == combined_id,
                prefix+f': source track {raw_id} changes combined identity')
        require(combined_id not in reverse or reverse[combined_id] == raw_id,
                prefix+f': combined track {combined_id} merges source identities')
        forward[raw_id], reverse[combined_id] = combined_id, raw_id
        source_type = str(obj.get('source', 'lidar'))
        require(source_type in ('lidar', 'camera_track_hold'), prefix+': unsupported source type '+source_type)
        record = {key: value for key, value in obj.items() if key not in ('box', 'name', 'track_id')}
        record.update(source=source_type, source_track_id=raw_id, combined_gt_id=combined_id,
                      source_class=str(obj['name']), combined_class=name, object_index=index)
        records.append(record)
    return dict(sequence_id=sequence, scene=str(meta['scene_name']),
        split=str(meta['temporal_split']), source_frame_id=int(meta['source_frame_id']),
        acquisition_timestamp_ns=int(meta['acquisition_timestamp_ns']), objects=records)


def build_sidecar(root):
    summary_path = root/'summary.json'
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    classes, class_map, taxonomy_sha = export_taxonomy()
    category = {name: index for index, name in enumerate(classes)}
    all_samples, all_sha = load_pickle(root/'rosa/fisheye3dod_infos_all.pkl')
    require(len(all_samples) == int(summary['frames']), 'Combined frame count disagrees with summary')
    sources, majority, source_manifest = {}, {}, []
    source_counts = Counter()
    for entry in summary['sequences']:
        sequence = str(entry['name'])
        require(sequence not in sources, 'Duplicate sequence in summary: '+sequence)
        path = Path(entry['annotation_file'])
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        require(digest == entry['annotation_sha256'], 'Stale source annotation SHA-256: '+str(path))
        frames = json.loads(payload)
        require(len(frames) == int(entry['frames']), 'Source frame count mismatch: '+sequence)
        by_token, votes, scores = {}, defaultdict(Counter), defaultdict(Counter)
        for index, frame in enumerate(frames):
            token = str(frame['timestamp_ns'])
            require(token not in by_token, 'Duplicate source token: '+sequence+' '+token)
            require(int(frame['frame_id']) == index, 'Source frame numbering mismatch: '+sequence)
            by_token[token] = frame
            for obj in frame['objects']:
                name = class_map.get(str(obj['name']))
                require(name is not None, 'Unmapped source class: '+str(obj['name']))
                track_id = int(obj['track_id'])
                votes[track_id][name] += 1
                scores[track_id][name] += float(obj.get('score', 0.))
                source_counts[str(obj.get('source', 'lidar'))] += 1
        sources[sequence] = by_token
        majority[sequence] = {track_id: max(classes, key=lambda name: (
            counts[name], scores[track_id][name], -category[name]))
            for track_id, counts in votes.items()}
        source_manifest.append(dict(sequence=sequence, path=str(path.resolve()), sha256=digest,
                                    frames=len(frames), tracks=len(votes)))
    require(sum(len(frames) for frames in sources.values()) == len(all_samples),
            'Source and combined total frame counts differ')
    frames, forward, reverse, globally_owned = {}, defaultdict(dict), defaultdict(dict), {}
    seen_source_frames = defaultdict(set)
    for sample in all_samples:
        sequence, token = str(sample['metainfo']['sequence_id']), str(sample['token'])
        require(token not in frames, 'Duplicate global combined frame token: '+token)
        require(sequence in sources and token in sources[sequence], 'Missing source frame: '+sequence+' '+token)
        frames[token] = verify_frame(sample, sources[sequence][token], majority[sequence],
                                    forward[sequence], reverse[sequence])
        seen_source_frames[sequence].add(token)
        for combined_id, raw_id in reverse[sequence].items():
            owner = sequence, raw_id
            require(combined_id not in globally_owned or globally_owned[combined_id] == owner,
                    f'Combined ID {combined_id} overlaps different sequences')
            globally_owned[combined_id] = owner
    for sequence, source_frames in sources.items():
        require(seen_source_frames[sequence] == set(source_frames), 'Missing combined source frames: '+sequence)
        require(set(forward[sequence]) == set(majority[sequence]), 'Incomplete source-ID mapping: '+sequence)
    require(dict(source_counts) == summary['source_counts'], 'Source counts disagree with combined summary')
    require(len(globally_owned) == int(summary['instances']), 'Track count disagrees with summary')
    sidecar = dict(schema_version=1, verified_frames=len(frames),
        verified_objects=sum(len(frame['objects']) for frame in frames.values()),
        verified_tracks=len(globally_owned), source_counts=dict(source_counts),
        combined_all_sha256=all_sha, combined_summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),
        exporter_taxonomy_sha256=taxonomy_sha, source_annotations=source_manifest,
        raw_to_combined_ids={sequence: {str(raw): combined for raw, combined in mapping.items()}
                             for sequence, mapping in forward.items()}, frames=frames)
    return sidecar, all_samples


def verify_split(samples, all_samples, split, sidecar):
    canonical = {str(sample['token']): sample for sample in all_samples}
    seen = set()
    for sample in samples:
        token = str(sample['token'])
        require(token in canonical and token not in seen, split+': missing/duplicate frame '+token)
        seen.add(token)
        require(sidecar['frames'][token]['split'] == split, split+': split metadata mismatch '+token)
        expected = canonical[token]
        for field in ('gt_boxes', 'gt_names', 'gt_ids'):
            require(np.array_equal(sample['annos'][field], expected['annos'][field]),
                    split+': split/all annotations differ '+token+' '+field)
        for field in ('sequence_id', 'source_frame_id', 'acquisition_timestamp_ns', 'scene_name'):
            require(sample['metainfo'][field] == expected['metainfo'][field],
                    split+': split/all metadata differ '+token+' '+field)
    required = {token for token, frame in sidecar['frames'].items() if frame['split'] == split}
    require(seen == required, split+': not all expected split frames are present')


def forecast_context(samples, config):
    tracks = defaultdict(list)
    for sample in samples:
        meta = sample['metainfo']
        pose = np.asarray(meta['T_world_CLIDAR'], dtype=np.float64)
        for index, (box, track_id) in enumerate(zip(
                np.asarray(sample['annos']['gt_boxes']).reshape(-1, 7), sample['annos']['gt_ids'])):
            tracks[(str(meta['scene_name']), int(track_id))].append(dict(
                token=str(sample['token']), index=index, track_id=int(track_id),
                timestamp_ns=int(meta['acquisition_timestamp_ns']),
                world=pose[:3, :3]@box[:3]+pose[:3, 3], rotation=pose[:3, :3]))
    result = {}
    for observations in tracks.values():
        observations.sort(key=lambda row: row['timestamp_ns'])
        for current, future in zip(observations, observations[1:]):
            dt = (future['timestamp_ns']-current['timestamp_ns'])*1e-9
            if not 0 < dt <= config.max_gap_seconds:
                continue
            speed = np.linalg.norm(future['world'][:2]-current['world'][:2])/dt
            if not np.isfinite(speed) or speed > config.max_speed_mps:
                continue
            result[(current['token'], current['index'])] = dict(current=current, future=future, dt=dt)
    return result


def existing_forecasts(path, context):
    rows, seen = [], set()
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = str(row['token']), int(row['index'])
            require(key in context and key not in seen, 'Unexpected/duplicate primary forecast row '+str(key))
            require(int(row['track_id']) == context[key]['current']['track_id'], 'Primary row identity differs '+str(key))
            require(abs(float(row['dt_seconds'])-context[key]['dt']) < 1e-8, 'Primary row horizon differs '+str(key))
            for name in list(row):
                if name.endswith('_error_m') or name.endswith('_speed_mps'):
                    row[name] = float(row[name]) if row[name] else None
            row['index'] = int(row['index'])
            rows.append(row)
            seen.add(key)
    require(seen == set(context), 'Primary forecast CSV has incomplete candidate coverage')
    return rows


def cohort_summary(rows, variants):
    result = dict(candidates=len(rows), variants={})
    for variant in ['raw', *variants]:
        errors = [row[variant+'_error_m'] for row in rows if row[variant+'_error_m'] is not None]
        report = dict(forecast_coverage=len(errors)/max(len(rows), 1), all_eligible=AUDIT.summarize(errors))
        if variant != 'raw':
            report['matched_with_raw'] = AUDIT.paired_summary(rows, variant)
        result['variants'][variant] = report
    return result


def lidar_only_samples(samples, sidecar):
    output = []
    for sample in samples:
        token = str(sample['token'])
        keep = np.array([obj['source'] == 'lidar' for obj in sidecar['frames'][token]['objects']], dtype=bool)
        annotations = {key: np.asarray(value)[keep] for key, value in sample['annos'].items()}
        output.append(dict(sample, annos=annotations))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/Custom/combined_improved_detzero_rosa')
    parser.add_argument('--official-root', type=Path, default=ROOT/'projects/Fisheye3DOD')
    parser.add_argument('--audit-dir', type=Path, default=ROOT/'projects/ROSA/work_dirs/temporal_memory_ablation_20261031/motion_audit')
    parser.add_argument('--splits', nargs='+', choices=['train', 'val'], default=['train', 'val'])
    args = parser.parse_args()
    sys.path.insert(0, str(args.official_root.resolve()))
    primary = json.loads((args.audit_dir/'summary.json').read_text())
    config = AUDIT.MOTION.MotionTargetConfig(**primary['config'])
    variants = [f'robust{window}' for window in primary['windows']]
    sidecar, all_samples = build_sidecar(args.data_root)
    # Complete split/CSV verification and all calculations before publishing.
    report = dict(config=asdict(config), windows=primary['windows'], verified_frames=sidecar['verified_frames'],
        verified_objects=sidecar['verified_objects'], verified_tracks=sidecar['verified_tracks'],
        source_counts=sidecar['source_counts'], source_annotations=sidecar['source_annotations'], splits={},
        limitations=[
            'LiDAR-origin means the original postprocessor did not mark camera_track_hold; centers remain DetZero/Kalman pseudo-labels, not independent raw-point ground truth.',
            'Training thresholds are frozen from the primary audit; validation is not used for threshold selection.',
            'Existing-prediction cohorts retain all-source fitted histories but require an immediate next LiDAR-origin target.',
            'LiDAR-history cohorts remove held observations before fitting and before finding the next target, with the identical max-gap rule; their candidate denominator differs from existing-prediction cohorts.',
            'Source-only filtering can lose moving/occluded objects. Read paired errors together with coverage; never compare unrelated all-eligible averages.',
            'No source annotations, PKLs, training settings or GPU experiments are modified.'])
    csv_artifacts = {}
    for split in args.splits:
        samples, digest = load_pickle(args.data_root/'rosa'/f'fisheye3dod_infos_{split}.pkl')
        require(digest == primary['splits'][split]['sha256'], split+': primary audit used a different PKL')
        verify_split(samples, all_samples, split, sidecar)
        context = forecast_context(samples, config)
        rows = existing_forecasts(args.audit_dir/f'{split}_forecasts.csv', context)
        next_lidar, both_lidar = [], []
        for row in rows:
            pair = context[(row['token'], row['index'])]
            future = pair['future']
            if sidecar['frames'][future['token']]['objects'][future['index']]['source'] != 'lidar':
                continue
            next_lidar.append(row)
            if sidecar['frames'][row['token']]['objects'][row['index']]['source'] == 'lidar':
                both_lidar.append(row)
        filtered = lidar_only_samples(samples, sidecar)
        lidar_report, lidar_rows = AUDIT.audit(filtered, config, primary['windows'])
        # Also compare original all-source raw velocity against the exact same
        # genuine-LiDAR current/next pair, including pairs spanning a hold.
        original_raw, _ = AUDIT.raw_targets(samples, config)
        full_index = {(str(sample['token']), int(track_id)): index for sample in samples
                      for index, track_id in enumerate(sample['annos']['gt_ids'])}
        lidar_context = forecast_context(filtered, config)
        for row in lidar_rows:
            pair = lidar_context[(row['token'], row['index'])]
            index = full_index[(row['token'], int(row['track_id']))]
            local = original_raw[row['token']][index]
            error = None
            if np.isfinite(local).all():
                current, future, dt = pair['current'], pair['future'], pair['dt']
                velocity = (current['rotation']@np.r_[local, 0.])[:2]
                error = float(np.linalg.norm(current['world'][:2]+velocity*dt-future['world'][:2]))
            row['original_all_source_raw_error_m'] = error
        original_comparisons = {}
        for variant in ['raw', *variants]:
            pairs = [row for row in lidar_rows if row['original_all_source_raw_error_m'] is not None
                     and row[variant+'_error_m'] is not None]
            original_comparisons[variant] = dict(count=len(pairs),
                original_all_source_raw=AUDIT.summarize([row['original_all_source_raw_error_m'] for row in pairs]),
                lidar_history_method=AUDIT.summarize([row[variant+'_error_m'] for row in pairs]))
        report['splits'][split] = dict(frames=len(samples), primary_candidates=len(rows),
            existing_history_next_lidar=cohort_summary(next_lidar, variants),
            existing_history_current_and_next_lidar=cohort_summary(both_lidar, variants),
            lidar_history=lidar_report, lidar_history_vs_original_raw=original_comparisons)
        csv_artifacts[f'{split}_lidar_history_forecasts.csv'] = lidar_rows
        print(json.dumps(dict(split=split,
            next_lidar=report['splits'][split]['existing_history_next_lidar'],
            both_lidar=report['splits'][split]['existing_history_current_and_next_lidar'],
            lidar_history_vs_original_raw=original_comparisons), indent=2), flush=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = args.audit_dir/'provenance.json'
    provenance_path.write_text(json.dumps(sidecar, separators=(',', ':'), allow_nan=False)+'\n')
    report['provenance_sha256'] = hashlib.sha256(provenance_path.read_bytes()).hexdigest()
    (args.audit_dir/'provenance_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for name, rows in csv_artifacts.items():
        with (args.audit_dir/name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ['token'])
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# Source-aligned pseudo-motion audit', '',
        f'Fail-closed verification passed for {sidecar["verified_frames"]:,} frames, '
        f'{sidecar["verified_objects"]:,} objects, and {sidecar["verified_tracks"]:,} unique combined tracks.',
        'All original JSON hashes, ordered float32 boxes, acquisition timestamps, frame tokens, '
        'mapped track-majority labels, split annotation copies and bijective track IDs were checked.', '',
        '| Split/cohort/method | Forecast candidates | Method coverage | Matched raw mean (m) | Matched robust mean (m) |',
        '|---|---:|---:|---:|---:|']
    for split, split_report in report['splits'].items():
        for cohort in ('existing_history_next_lidar', 'existing_history_current_and_next_lidar'):
            values = split_report[cohort]
            for variant in variants:
                method = values['variants'][variant]
                paired = method['matched_with_raw']
                fmt = lambda value: f'{value:.4f}' if value is not None else 'n/a'
                lines.append(f'| {split}/{cohort}/{variant} | {values["candidates"]} | '
                    f'{method["forecast_coverage"]:.1%} | {fmt(paired["raw"]["mean_m"])} | '
                    f'{fmt(paired["robust"]["mean_m"])} |')
        for variant in variants:
            method = split_report['lidar_history']['variants'][variant]
            paired = method['matched_with_raw']
            lines.append(f'| {split}/LiDAR-only-history/{variant} | '
                f'{split_report["lidar_history"]["forecast_candidates"]} | {method["forecast_coverage"]:.1%} | '
                f'{fmt(paired["raw"]["mean_m"])} | {fmt(paired["robust"]["mean_m"])} |')
    lines += ['', '## Train-first interpretation', '']
    training = report['splits'].get('train')
    if training:
        paired_changes = []
        for cohort in ('existing_history_next_lidar', 'existing_history_current_and_next_lidar'):
            for variant in variants:
                pair = training[cohort]['variants'][variant]['matched_with_raw']
                if pair['raw']['mean_m'] is not None and pair['raw']['mean_m'] > 0:
                    change = (pair['robust']['mean_m']/pair['raw']['mean_m']-1)*100
                    paired_changes.append(change)
                    lines.append(f'- {cohort}/{variant}: matched train mean error {change:+.2f}% relative to raw.')
        for variant in variants:
            pair = training['lidar_history']['variants'][variant]['matched_with_raw']
            if pair['raw']['mean_m'] is not None and pair['raw']['mean_m'] > 0:
                paired_changes.append((pair['robust']['mean_m']/pair['raw']['mean_m']-1)*100)
        if paired_changes and min(paired_changes) >= 0:
            lines += ['', 'Every tested source-restricted robust variant has worse matched train mean error. '
                'Neither removing predicted future targets nor fitting only LiDAR-origin histories '
                'turns these robust-window candidates into a demonstrated broad motion-target improvement.']
        lines += ['', 'Read the source-restricted results together with the original all-source audit. '
            'Do not enable robust replacement or retraining from these consistency measurements alone. '
            'The LiDAR-only two-frame raw comparator changes its history/candidate horizon; its small '
            'differences versus original raw forecasts are reported separately on identical pairs.']
    lines += ['', '## Safeguards', '', *['- '+item for item in report['limitations']], '',
        'The source sidecar is provenance.json. Full paired errors, P90/coverage and subgroup statistics '
        'are in provenance_summary.json. Current-to-next LiDAR-only history forecast rows are saved separately. '
        'No automatic change to target generation or retraining is authorized by this consistency audit.', '']
    (args.audit_dir/'PROVENANCE_REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
