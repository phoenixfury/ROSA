#!/usr/bin/env python3
"""Audit causal robust pseudo-motion targets without changing source PKLs.

Forecasts use only observations through t and are scored against t+1 DetZero
pseudo-centers, NOT independent physical ground truth. Default thresholds are
fixed heuristics. Train-only is the default; --splits train val applies exactly
the same settings to validation without any validation-based tuning.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MOTION_PATH = ROOT/'projects/ROSA/rosa/integrations/fisheye3dod/motion_targets.py'
SPEC = importlib.util.spec_from_file_location('rosa_offline_motion_targets', MOTION_PATH)
MOTION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOTION
SPEC.loader.exec_module(MOTION)


def summarize(errors):
    values = np.asarray(errors, dtype=np.float64)
    if not len(values):
        return dict(count=0, mean_m=None, median_m=None, p90_m=None, over_1m_fraction=None)
    return dict(count=len(values), mean_m=float(np.mean(values)),
                median_m=float(np.median(values)), p90_m=float(np.percentile(values, 90)),
                over_1m_fraction=float(np.mean(values > 1.)))


def raw_targets(samples, config):
    """Exact existing two-frame rule, retained as the audit comparator."""
    previous, result, statuses = {}, {}, Counter()
    for sample in sorted(samples, key=lambda row: int(row['metainfo']['acquisition_timestamp_ns'])):
        meta = sample['metainfo']
        timestamp = int(meta['acquisition_timestamp_ns'])
        pose = np.asarray(meta['T_world_CLIDAR'], dtype=np.float64)
        boxes = np.asarray(sample['annos']['gt_boxes']).reshape(-1, 7)
        values = np.full((len(boxes), 2), np.nan, dtype=np.float32)
        for index, (box, track_id) in enumerate(zip(boxes, sample['annos']['gt_ids'])):
            world = pose[:3, :3]@box[:3]+pose[:3, 3]
            key = str(meta['scene_name']), int(track_id)
            status = 'insufficient_history'
            if key in previous:
                past_time, past_world = previous[key]
                dt = (timestamp-past_time)*1e-9
                status = 'time_discontinuity'
                if 0 < dt <= config.max_gap_seconds:
                    world_velocity = (world-past_world)/dt
                    status = 'excessive_speed'
                    if np.isfinite(world_velocity).all() and np.linalg.norm(world_velocity[:2]) <= config.max_speed_mps:
                        values[index] = (pose[:3, :3].T@world_velocity)[:2]
                        status = 'accepted'
            statuses[status] += 1
            previous[key] = timestamp, world
        result[str(sample['token'])] = values
    return result, dict(statuses)


def paired_summary(rows, variant):
    pairs = [row for row in rows if row.get('raw_error_m') is not None
             and row.get(variant+'_error_m') is not None]
    return dict(count=len(pairs), raw=summarize([row['raw_error_m'] for row in pairs]),
                robust=summarize([row[variant+'_error_m'] for row in pairs]),
                robust_fit_origin=summarize([row[variant+'_fit_origin_error_m'] for row in pairs]),
                robust_win_fraction=(float(np.mean([row[variant+'_error_m'] < row['raw_error_m']
                                                   for row in pairs])) if pairs else None))


def audit(samples, config, windows):
    raw, raw_statuses = raw_targets(samples, config)
    methods, diagnostics = {}, {}
    for window in windows:
        name = f'robust{window}'
        methods[name], diagnostics[name] = MOTION.robust_track_velocities(
            samples, replace(config, window=window), return_diagnostics=True)
    tracks = defaultdict(list)
    previous_pose = {}
    for sample in sorted(samples, key=lambda row: int(row['metainfo']['acquisition_timestamp_ns'])):
        meta, annotations = sample['metainfo'], sample['annos']
        scene = str(meta['scene_name'])
        timestamp = int(meta['acquisition_timestamp_ns'])
        pose = np.asarray(meta['T_world_CLIDAR'], dtype=np.float64)
        yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
        turn = 'unknown'
        if scene in previous_pose:
            previous_time, previous_yaw = previous_pose[scene]
            dt = (timestamp-previous_time)*1e-9
            if 0 < dt <= config.max_gap_seconds:
                change = np.arctan2(np.sin(yaw-previous_yaw), np.cos(yaw-previous_yaw))
                turn = 'ego_turn' if abs(np.degrees(change)/dt) >= 3. else 'ego_straight'
        previous_pose[scene] = timestamp, yaw
        for index, (box, track_id, name) in enumerate(zip(
                np.asarray(annotations['gt_boxes']).reshape(-1, 7),
                annotations['gt_ids'], annotations['gt_names'])):
            tracks[(scene, int(track_id))].append(dict(
                token=str(sample['token']), index=index, timestamp_ns=timestamp,
                scene=scene, bag=str(meta.get('sequence_id', scene)), class_name=str(name),
                track_id=int(track_id), world=pose[:3, :3]@box[:3]+pose[:3, 3],
                rotation=pose[:3, :3], ego_turn=turn))
    rows, exclusions = [], Counter()
    for observations in tracks.values():
        for step, (current, future) in enumerate(zip(observations, observations[1:])):
            dt = (future['timestamp_ns']-current['timestamp_ns'])*1e-9
            if not 0 < dt <= config.max_gap_seconds:
                exclusions['future_time_discontinuity'] += 1
                continue
            target = future['world'][:2]
            speed = np.linalg.norm(target-current['world'][:2])/dt
            if not np.isfinite(speed) or speed > config.max_speed_mps:
                exclusions['future_nonfinite_or_gross_jump'] += 1
                continue
            row = {key: current[key] for key in ('token', 'index', 'timestamp_ns', 'scene',
                                                'bag', 'class_name', 'track_id', 'ego_turn')}
            row.update(dt_seconds=dt,
                pseudo_motion='pseudo_static' if speed < .5 else 'pseudo_moving',
                pseudo_acceleration='unknown', pseudo_turn='unknown')
            if step:
                past = observations[step-1]
                past_dt = (current['timestamp_ns']-past['timestamp_ns'])*1e-9
                if 0 < past_dt <= config.max_gap_seconds:
                    before = (current['world'][:2]-past['world'][:2])/past_dt
                    after = (future['world'][:2]-current['world'][:2])/dt
                    acceleration = np.linalg.norm(after-before)/((past_dt+dt)/2)
                    row['pseudo_acceleration'] = 'accelerating' if acceleration >= 1.5 else 'steady'
                    if min(np.linalg.norm(before), np.linalg.norm(after)) >= 1.:
                        angle = np.arctan2(before[0]*after[1]-before[1]*after[0], before@after)
                        turn_rate = abs(np.degrees(angle))/((past_dt+dt)/2)
                        row['pseudo_turn'] = 'turning' if turn_rate >= 10. else 'straight'
            token, index = current['token'], current['index']
            for name, values in [('raw', raw), *methods.items()]:
                local = values[token][index]
                valid = bool(np.isfinite(local).all())
                row[name+'_error_m'] = None
                row[name+'_speed_mps'] = None
                if valid:
                    world_velocity = (current['rotation']@np.r_[local, 0.])[:2]
                    forecast = current['world'][:2]+world_velocity*dt
                    row[name+'_error_m'] = float(np.linalg.norm(forecast-target))
                    row[name+'_speed_mps'] = float(np.linalg.norm(world_velocity))
                if name != 'raw':
                    diagnostic = diagnostics[name][token][index]
                    row[name+'_status'] = diagnostic['status']
                    row[name+'_fit_origin_error_m'] = None
                    if valid:
                        # Same representable XY target as the primary
                        # forecast; only the current-position origin differs.
                        fitted = diagnostic['fitted_world_xy']+world_velocity*dt
                        row[name+'_fit_origin_error_m'] = float(np.linalg.norm(fitted-target))
            rows.append(row)
        exclusions['track_terminal_no_future'] += bool(observations)
    total_targets = sum(len(values) for values in raw.values())
    report = dict(frames=len(samples), targets=total_targets, tracks=len(tracks),
                  forecast_candidates=len(rows), exclusions=dict(exclusions), raw_statuses=raw_statuses,
                  variants={}, subgroup_definitions=dict(
                      pseudo_motion='Next pseudo-center displacement / true dt <0.5 m/s; diagnostic only, never fitted input.',
                      pseudo_acceleration='Adjacent pseudo-velocity change / mean dt >=1.5 m/s^2; diagnostic only, may include label noise.',
                      pseudo_turn='Adjacent pseudo-velocity direction change / mean dt >=10 degrees/s, both speeds >=1 m/s; diagnostic only.',
                      ego_turn='Acquisition-pose yaw rate >=3 degrees/s, diagnostic only.'))
    for name, values in [('raw', raw), *methods.items()]:
        errors = [row[name+'_error_m'] for row in rows if row[name+'_error_m'] is not None]
        known = sum(int(np.isfinite(value).all(axis=1).sum()) for value in values.values())
        speeds = np.concatenate([np.linalg.norm(value[np.isfinite(value).all(axis=1)], axis=1)
                                 for value in values.values()])
        summary = dict(known_targets=known, target_coverage=known/max(total_targets, 1),
                       forecast_coverage=len(errors)/max(len(rows), 1), all_eligible=summarize(errors),
                       target_xy_speed_mps=dict(
                           median=float(np.median(speeds)) if len(speeds) else None,
                           p90=float(np.percentile(speeds, 90)) if len(speeds) else None,
                           p99=float(np.percentile(speeds, 99)) if len(speeds) else None))
        if name != 'raw':
            summary['status_counts'] = dict(Counter(record['status'] for frame in diagnostics[name].values()
                                                   for record in frame))
            summary['matched_with_raw'] = paired_summary(rows, name)
            summary['subgroups_matched_with_raw'] = {}
            for group in ('class_name', 'bag', 'pseudo_motion', 'pseudo_acceleration', 'pseudo_turn', 'ego_turn'):
                summary['subgroups_matched_with_raw'][group] = {
                    key: paired_summary([row for row in rows if row[group] == key], name)
                    for key in sorted({row[group] for row in rows})}
        report['variants'][name] = summary
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/Custom/combined_improved_detzero_rosa')
    parser.add_argument('--official-root', type=Path, default=ROOT/'projects/Fisheye3DOD')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'projects/ROSA/work_dirs/temporal_memory_ablation_20261031/motion_audit')
    parser.add_argument('--splits', nargs='+', choices=['train', 'val'], default=['train'])
    parser.add_argument('--windows', nargs='+', type=int, default=[3, 5])
    parser.add_argument('--max-residual-m', type=float, default=.75)
    parser.add_argument('--max-endpoint-residual-m', type=float, default=1.25)
    parser.add_argument('--huber-delta-m', type=float, default=.35)
    args = parser.parse_args()
    sys.path.insert(0, str(args.official_root.resolve()))
    config = MOTION.MotionTargetConfig(huber_delta_m=args.huber_delta_m,
        max_residual_m=args.max_residual_m, max_endpoint_residual_m=args.max_endpoint_residual_m)
    for window in args.windows:
        replace(config, window=window)  # Validate before unpickling.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overall = dict(config=asdict(config), windows=args.windows,
        design='Frozen heuristic thresholds; train primary audit, optional val with identical settings.',
        limitations=[
            'All centers/IDs are DetZero/Kalman pseudo-labels, not independent motion ground truth.',
            'Forecast inputs include only observations through t; t+1 is scoring-only.',
            'Matched-cohort errors and coverage must be read together; all-eligible errors are not directly comparable across methods.',
            'Current-anchor and fitted-origin forecasts use the identical current-LiDAR-XY target rotated back into world XY; only the forecast origin differs.',
            'Aligned per-object camera-hold/LiDAR provenance is absent in combined PKLs; no source weighting is invented.',
            'No source PKL or annotation is modified.'], splits={})
    lines = ['# Offline causal motion-target audit', '',
        'Primary metric: XY next-pseudo-center forecast error, with each velocity anchored at the same current pseudo-center.',
        'These are consistency diagnostics against DetZero/Kalman, not physical ground truth. Lower error alone does not authorize retraining.', '',
        '| Split/method | Target coverage | Forecast coverage | Matched raw mean (m) | Matched robust mean (m) | Matched raw P90 (m) | Matched robust P90 (m) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for split in args.splits:
        path = args.data_root/'rosa'/f'fisheye3dod_infos_{split}.pkl'
        payload = path.read_bytes()
        package = pickle.loads(payload)
        samples = package['data_list']
        report, rows = audit(samples, config, args.windows)
        report.update(path=str(path.resolve()), sha256=hashlib.sha256(payload).hexdigest())
        overall['splits'][split] = report
        with (args.output_dir/f'{split}_forecasts.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ['token'])
            writer.writeheader()
            writer.writerows(rows)
        for name, summary in report['variants'].items():
            if name == 'raw':
                continue
            paired = summary['matched_with_raw']
            fmt = lambda value: f'{value:.4f}' if value is not None else 'n/a'
            lines.append(f'| {split}/{name} | {summary["target_coverage"]:.1%} | {summary["forecast_coverage"]:.1%} | '
                f'{fmt(paired["raw"]["mean_m"])} | {fmt(paired["robust"]["mean_m"])} | '
                f'{fmt(paired["raw"]["p90_m"])} | {fmt(paired["robust"]["p90_m"])} |')
        print(json.dumps(dict(split=split, variants=report['variants']), indent=2), flush=True)
    lines += ['', '## Train-first findings', '']
    training = overall['splits'].get('train')
    if training:
        any_mean_improvement = False
        for name, summary in training['variants'].items():
            if name == 'raw':
                continue
            paired = summary['matched_with_raw']
            raw_mean, robust_mean = paired['raw']['mean_m'], paired['robust']['mean_m']
            if raw_mean is not None:
                change = (robust_mean/raw_mean-1)*100 if raw_mean else 0.
                lines.append(f'- {name}: matched mean forecast error changes {change:+.2f}%; '
                    f'target coverage {summary["target_coverage"]:.1%} versus raw '
                    f'{training["variants"]["raw"]["target_coverage"]:.1%}.')
                any_mean_improvement |= robust_mean < raw_mean*.99
        if not any_mean_improvement:
            lines += ['', 'Neither robust window improves matched train mean error by 1%. '
                'Do not enable broad robust-motion replacement or launch retraining from this audit. '
                'Lower all-eligible errors mainly reflect different, easier retained cohorts; '
                'inspect the paired table and lost coverage instead. The 1% reporting tolerance '
                'is descriptive, not a tuned statistical significance test.']
        lines += ['', 'Class-conditional differences and ego/object-turn/acceleration diagnostics '
            'are in summary.json. They are exploratory, not independently validated class policies. '
            'Validation, if requested, uses unchanged train-first settings and must not be used to retune them.']
    lines += ['', '## Interpretation safeguards', '', *['- '+item for item in overall['limitations']], '',
        'Thresholds are diagnostic uncertainty heuristics. See summary.json for the complete frozen configuration, exclusions, class/bag/motion/ego-turn subgroups and masked-target counts.', '']
    (args.output_dir/'summary.json').write_text(json.dumps(overall, indent=2, allow_nan=False)+'\n')
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
