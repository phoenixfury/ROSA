#!/usr/bin/env python3
"""Frozen-checkpoint temporal-memory experiments; never starts training.

Run in openmmlab from the repository root::

    python projects/ROSA/tools/test_temporal_memory.py --epoch 4
    python projects/ROSA/tools/test_temporal_memory.py --epoch 7 \
        --variant current --variant quality_nms --variant quality_nms_ego_only

The default epoch-4 sequence tests the legacy cache, current-evidence gating,
quality ranking, NMS, their combination, ego-only propagation and a 50-slot
raw-score control. Quality threshold 0.05 is an initial hypothesis, NOT a
calibrated equivalent of the legacy class threshold 0.15. Per-frame occupancy
is retained to expose that confound. All runs share the output decoder and
validation data, use one GPU serially, and produce no checkpoint updates.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
ROSA = ROOT / 'projects/ROSA'
TEMPORAL_RUN = ROSA / 'work_dirs/rosa_vov99_temporal_12e'
LEGACY = ROSA / 'work_dirs/temporal_plateau_comparison_20261031'
CONFIG = ROSA / 'configs/fisheye3dod/rosa_vov99_temporal_12e.py'
CHECKPOINTS = {4: TEMPORAL_RUN / 'best_fisheye_bev_NDS_epoch_4.pth',
               7: TEMPORAL_RUN / 'best_fisheye_bev_mAP_epoch_7.pth'}
VARIANTS = {
    'current': dict(cache_score_mode='class', cache_nms_iou_threshold=None,
                    motion_mode='predicted', cache_require_current=False,
                    min_confidence=.15, num_temp_instances=100),
    'current_gated': dict(cache_score_mode='class', cache_nms_iou_threshold=None,
                    motion_mode='predicted', cache_require_current=True,
                    min_confidence=.15, num_temp_instances=100),
    'quality': dict(cache_score_mode='quality', cache_nms_iou_threshold=None,
                    motion_mode='predicted', cache_require_current=True,
                    min_confidence=.05, num_temp_instances=100),
    'class_nms': dict(cache_score_mode='class', cache_nms_iou_threshold=.5,
                    motion_mode='predicted', cache_require_current=True,
                    min_confidence=.15, num_temp_instances=100),
    'quality_nms': dict(cache_score_mode='quality', cache_nms_iou_threshold=.5,
                    motion_mode='predicted', cache_require_current=True,
                    min_confidence=.05, num_temp_instances=100),
    'quality_nms_ego_only': dict(cache_score_mode='quality', cache_nms_iou_threshold=.5,
                    motion_mode='ego_only', cache_require_current=True,
                    min_confidence=.05, num_temp_instances=100),
    'raw50': dict(cache_score_mode='class', cache_nms_iou_threshold=None,
                    motion_mode='predicted', cache_require_current=True,
                    min_confidence=.15, num_temp_instances=50),
    'quality_nms50': dict(cache_score_mode='quality', cache_nms_iou_threshold=.5,
                    motion_mode='predicted', cache_require_current=True,
                    min_confidence=.05, num_temp_instances=50),
}
DEFAULT_VARIANTS = tuple(name for name in VARIANTS if name != 'quality_nms50')
IMPORTS = [
    'projects.mmdet3d_plugin', 'rosa.integrations.sparse4d_components',
    'rosa.integrations.optim', 'rosa.integrations.fisheye3dod.dataset',
    'rosa.integrations.fisheye3dod.model', 'rosa.integrations.fisheye3dod.metric',
    'rosa.integrations.fisheye_ray_sparse4d', 'rosa.integrations.fisheye_box_sparse4d',
    'rosa.integrations.kitti360.backbone',
    'rosa.integrations.fisheye3dod.balanced_training',
    'rosa.integrations.fisheye3dod.temporal',
    'projects.ROSA.tools.test_temporal_memory',
]
METRIC = re.compile(r'fisheye_bev/([A-Za-z0-9_@.]+):\s*([-+0-9.eE]+)')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def source_fingerprint():
    """Hash explicit inference dependencies and the full local config family."""
    paths = [Path(__file__).resolve(), ROSA / 'tools/run.py',
        ROSA / 'rosa/integrations/fisheye3dod/temporal.py',
        ROSA / 'rosa/integrations/fisheye3dod/dataset.py',
        ROSA / 'rosa/integrations/fisheye3dod/model.py',
        ROSA / 'rosa/integrations/fisheye3dod/metric.py',
        ROSA / 'rosa/integrations/fisheye_box_sparse4d.py',
        ROSA / 'rosa/integrations/fisheye_ray_sparse4d.py',
        ROOT / 'projects/mmdet3d_plugin/models/instance_bank.py',
        ROOT / 'projects/mmdet3d_plugin/models/sparse4d_head.py',
        ROOT / 'projects/mmdet3d_plugin/models/detection3d/decoder.py',
        ROOT / 'projects/mmdet3d_plugin/datasets/transforms/pack_sparse4d.py']
    paths += sorted(CONFIG.parent.glob('*.py'))
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths if path.is_file()}


def _register_hook():
    from mmengine.hooks import Hook
    from mmengine.registry import HOOKS

    @HOOKS.register_module(force=True)
    class MemoryAblationDiagnosticsHook(Hook):
        """Record detached per-lane cache/propagation stats, not predictions."""
        def __init__(self, out_file):
            self.out_file = Path(out_file)
            self.stream = None

        def before_test_epoch(self, runner):
            self.out_file.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.out_file.open('w')

        def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
            model = runner.model.module if hasattr(runner.model, 'module') else runner.model
            bank = model.head.instance_bank
            cache = getattr(bank, 'last_cache_stats', None) or []
            propagation = getattr(bank, 'last_get_stats', None) or []
            samples = outputs or []
            if isinstance(cache, dict):
                cache = [cache]
            if isinstance(propagation, dict):
                propagation = [propagation]
            for lane, sample in enumerate(samples):
                meta = sample.metainfo if hasattr(sample, 'metainfo') else sample
                record = dict(batch_index=batch_idx, lane=lane,
                    token=str(meta.get('token', '')),
                    scene_token=str(meta.get('scene_token', '')),
                    temporal_is_padding=bool(meta.get('temporal_is_padding', False)),
                    cache=cache[lane] if lane < len(cache) else {},
                    propagation=propagation[lane] if lane < len(propagation) else {})
                self.stream.write(json.dumps(record, allow_nan=False) + '\n')
            self.stream.flush()

        def after_test_epoch(self, runner, metrics=None):
            if self.stream is not None:
                self.stream.close()
                self.stream = None


def config_text(directory, variant, workers):
    hooks = [dict(type='SyncBuffersHook', priority='NORMAL'),
             dict(type='TemporalCacheResetHook', priority='HIGH'),
             dict(type='MemoryAblationDiagnosticsHook', priority='LOW',
                  out_file=str(directory / 'cache_stats.jsonl'))]
    bank = dict(VARIANTS[variant], collect_cache_stats=True)
    return ('# Generated frozen-weight inference experiment.\n'
            f'_base_ = [{str(CONFIG)!r}]\n'
            f'custom_imports = {dict(imports=IMPORTS, allow_failed_imports=False)!r}\n'
            f'custom_hooks = {hooks!r}\n'
            f'model = {dict(head=dict(instance_bank=bank))!r}\n'
            f'test_dataloader = {dict(batch_size=1, num_workers=workers)!r}\n')


def summarize_cache(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    observed = [row for row in rows if not row['temporal_is_padding']]
    result = dict(frames=len(observed), unique_tokens=len({r['token'] for r in observed}))
    for group in ('cache', 'propagation'):
        values = defaultdict(list)
        for row in observed:
            for key, value in row[group].items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values[key].append(value)
        result[group] = {key: dict(mean=sum(v) / len(v), min=min(v), max=max(v),
            p50=sorted(v)[len(v) // 2], samples=len(v)) for key, v in values.items()}
    return result


def operating_diagnostics(directory, infos):
    sys.path.insert(0, str(ROSA / 'tools'))
    from compare_retrained_fisheye import (
        CLASSES, load_frames, matched_recall_points, operating_counts, select)
    frames, _ = load_frames(directory, infos, with_iou=False)
    result = dict(samples=len(frames),
        matched_recall_precision=matched_recall_points(frames), score_thresholds={})
    for threshold in (.1, .2, .3, .5):
        result['score_thresholds'][str(threshold)] = operating_counts(
            frames, [select(frame, threshold=threshold) for frame in frames])
    cars = []
    for frame in frames:
        pred, truth = frame['labels'] == 0, frame['gt_names'] == CLASSES[0]
        cars.append(dict(frame, boxes=frame['boxes'][pred], scores=frame['scores'][pred],
            labels=frame['labels'][pred], gt_boxes=frame['gt_boxes'][truth],
            gt_names=frame['gt_names'][truth]))
    result['Car_matched_recall_precision'] = matched_recall_points(cars)
    bags = defaultdict(list)
    for frame, info in zip(frames, infos):
        bags[str(info['metainfo']['sequence_id'])].append(frame)
    result['per_bag_score_0_1'] = {key: operating_counts(subset,
        [select(frame, threshold=.1) for frame in subset]) for key, subset in bags.items()}
    return result


def legacy_canary(directory, epoch, validation_hash, checkpoint_hash):
    """The unmodified configuration must reproduce old numerical predictions."""
    import numpy as np
    for name, value in (('float', float), ('int', int), ('bool', bool)):
        if name not in np.__dict__:
            setattr(np, name, value)
    sys.path.insert(0, str(ROSA / 'tools'))
    from infer_fisheye_checkpoints import field, load_cpu_results
    previous = LEGACY / f'temporal{epoch}_memory_on'
    record = json.loads((previous / 'evaluation.json').read_text())
    if (record.get('validation_sha256') != validation_hash or
            record.get('checkpoint_sha256') != checkpoint_hash):
        raise ValueError('Legacy canary annotations/checkpoint do not match')
    old = load_cpu_results(previous / 'predictions.pkl')
    new = load_cpu_results(directory / 'predictions.pkl')
    exact, close, max_difference = len(old) == len(new), len(old) == len(new), 0.
    for a, b in zip(old, new):
        exact &= str(field(a, 'token')) == str(field(b, 'token'))
        close &= str(field(a, 'token')) == str(field(b, 'token'))
        ap, bp = field(a, 'pred_instances_3d'), field(b, 'pred_instances_3d')
        for name in ('bboxes_3d', 'scores_3d', 'labels_3d'):
            av, bv = field(ap, name), field(bp, name)
            av = getattr(av, 'tensor', av).detach().cpu().numpy()
            bv = getattr(bv, 'tensor', bv).detach().cpu().numpy()
            if av.shape != bv.shape:
                exact, close = False, False
                continue
            exact &= bool(np.array_equal(av, bv))
            close &= bool(np.allclose(av, bv, rtol=1.e-6, atol=1.e-6))
            if av.size:
                max_difference = max(max_difference, float(np.abs(av - bv).max()))
    current = json.loads((directory / 'evaluation.json').read_text())
    delta = {key: current['metrics'].get(key, float('inf')) - value
             for key, value in record['metrics'].items()}
    metrics_equal = all(abs(value) <= .000051 for value in delta.values())
    result = dict(previous_directory=str(previous), predictions_exact=bool(exact),
        predictions_close=bool(close), maximum_absolute_difference=max_difference,
        metric_deltas=delta, rounded_metrics_equal=metrics_equal,
        passed=bool(exact and metrics_equal))
    write_json(directory / 'legacy_canary.json', result)
    if not result['passed']:
        raise RuntimeError(f'Legacy-current canary changed; inspect {directory}/legacy_canary.json')
    return result


def legacy_memory_off_reference(epoch, validation_hash, checkpoint_hash):
    """Attach old controls honestly: hashes verified, old code not fingerprinted."""
    directory = LEGACY / f'temporal{epoch}_memory_off'
    record_path = directory / 'evaluation.json'
    prediction_path = directory / 'predictions.pkl'
    if not record_path.is_file() or not prediction_path.is_file():
        return None
    record = json.loads(record_path.read_text())
    if (record.get('validation_sha256') != validation_hash or
            record.get('checkpoint_sha256') != checkpoint_hash or
            record.get('return_code') != 0):
        raise ValueError('Legacy memory-off reference has incompatible data/checkpoint')
    result = dict(directory=str(directory), metrics=record['metrics'],
        checkpoint_sha256=checkpoint_hash, validation_sha256=validation_hash,
        predictions_sha256=sha256(prediction_path),
        provenance='Verified checkpoint/data hashes; historical source revision unavailable.',
        legacy_reference=True, new_code_fingerprinted=False)
    diagnostics = LEGACY / 'operating_points.json'
    if diagnostics.is_file():
        stored = json.loads(diagnostics.read_text())
        previous = stored.get('models', {}).get(f'temporal{epoch}_memory_off')
        if (stored.get('validation_sha256') == validation_hash and previous and
                previous.get('predictions_sha256') == result['predictions_sha256']):
            result['operating_points'] = previous
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', action='append', type=int, choices=(4, 7))
    parser.add_argument('--variant', action='append', choices=tuple(VARIANTS))
    parser.add_argument('--data-root', type=Path, default=Path(os.environ.get(
        'ROSA_DATA_ROOT', ROOT / 'data/Custom/combined_improved_detzero_rosa')))
    parser.add_argument('--official-root', type=Path, default=ROOT / 'projects/Fisheye3DOD')
    parser.add_argument('--output-dir', type=Path,
        default=ROSA / 'work_dirs/temporal_memory_ablation_20261031')
    parser.add_argument('--device', default='0')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--skip-legacy-canary', action='store_true',
        help='Explicitly allow current-cache changes; recorded in the manifest')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.data_root, args.output_dir = args.data_root.resolve(), args.output_dir.resolve()
    for source in (ROOT, ROSA, args.official_root.resolve()):
        sys.path.insert(0, str(source))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotation = args.data_root / 'rosa/fisheye3dod_infos_val.pkl'
    with annotation.open('rb') as stream:
        infos = pickle.load(stream)
    infos = infos['data_list'] if isinstance(infos, dict) else infos
    tokens = [str(info['token']) for info in infos]
    if len(tokens) != len(set(tokens)):
        raise ValueError('Validation tokens must be unique')
    fingerprint = dict(validation_sha256=sha256(annotation), source_sha256=source_fingerprint(),
        validation_tokens_sha256=hashlib.sha256('\n'.join(tokens).encode()).hexdigest(),
        validation_samples=len(tokens))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.device,
        PYTHONPATH=os.pathsep.join(str(ROOT / name) for name in
            ('', 'projects/ROSA', 'projects/Fisheye3DOD')),
        ROSA_DATA_ROOT=str(args.data_root),
        FISHEYE3DOD_OFFICIAL_ROOT=str(args.official_root.resolve()),
        MPLCONFIGDIR='/tmp/rosa-memory-ablation-matplotlib')
    summary = dict(**fingerprint, models={}, legacy_memory_off_references={},
        diagnostic_matching='Class-aware score-ordered BEV center <2 m; not official AP.',
        quality_threshold_note='0.05 initial hypothesis; inspect occupancy against raw50/current_gated.',
        training_started=False, output_decoder_changed=False,
        skip_legacy_canary=args.skip_legacy_canary)
    summary_path = args.output_dir / 'evaluations.json'
    if summary_path.is_file():
        previous = json.loads(summary_path.read_text())
        if any(previous.get(key) != value for key, value in fingerprint.items()):
            raise ValueError('Existing output uses different code/data; choose a new --output-dir')
        summary['models'] = previous.get('models', {})
        summary['legacy_memory_off_references'] = previous.get('legacy_memory_off_references', {})
    for epoch in args.epoch or [4]:
        checkpoint = CHECKPOINTS[epoch]
        checkpoint_hash = sha256(checkpoint)
        reference = legacy_memory_off_reference(epoch,
            fingerprint['validation_sha256'], checkpoint_hash)
        if reference is not None:
            summary['legacy_memory_off_references'][str(epoch)] = reference
        for variant in args.variant or DEFAULT_VARIANTS:
            if source_fingerprint() != fingerprint['source_sha256'] or sha256(annotation) != fingerprint['validation_sha256']:
                raise RuntimeError('Code/validation changed during experiment; do not mix runs')
            name = f'epoch{epoch}_{variant}'
            directory = args.output_dir / name
            directory.mkdir(exist_ok=True)
            config = directory / 'inference_config.py'
            text = config_text(directory, variant, args.workers)
            config.write_text(text)
            command = [sys.executable, str(ROSA / 'tools/run.py'), 'evaluate',
                str(config), str(checkpoint), '--work-dir', str(directory / 'work'),
                '--dump', str(directory / 'predictions.pkl')]
            identity = dict(**fingerprint, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                epoch=epoch, variant=variant, cache_options=VARIANTS[variant], command=command,
                config_sha256=hashlib.sha256(text.encode()).hexdigest(), device=args.device)
            record_path = directory / 'evaluation.json'
            cached = json.loads(record_path.read_text()) if record_path.is_file() else {}
            reusable = (all(cached.get(key) == value for key, value in identity.items())
                and cached.get('return_code') == 0 and bool(cached.get('metrics'))
                and (directory / 'predictions.pkl').is_file()
                and cached.get('predictions_sha256') == sha256(directory / 'predictions.pkl')
                and (directory / 'cache_stats.jsonl').is_file()
                and cached.get('cache_stats_sha256') == sha256(directory / 'cache_stats.jsonl'))
            if args.dry_run:
                print(name, 'REUSE' if reusable else 'EVALUATE', ' '.join(command), flush=True)
                continue
            if reusable:
                print('Reusing', name, flush=True)
                record = cached
            else:
                print('Evaluating', name, flush=True)
                started = time.monotonic()
                with (directory / 'evaluation.log').open('w') as log:
                    process = subprocess.run(command, cwd=ROOT, env=environment,
                        stdout=log, stderr=subprocess.STDOUT)
                metrics = {match.group(1): float(match.group(2)) for match in
                    METRIC.finditer((directory / 'evaluation.log').read_text(errors='replace'))}
                record = dict(identity, return_code=process.returncode,
                    elapsed_seconds=time.monotonic() - started, metrics=metrics)
                write_json(record_path, record)
                if process.returncode or not metrics:
                    raise RuntimeError(f'{name} failed; see {directory}/evaluation.log')
                if source_fingerprint() != fingerprint['source_sha256'] or sha256(annotation) != fingerprint['validation_sha256']:
                    raise RuntimeError('Code/validation changed during inference; result is not reusable')
                record['predictions_sha256'] = sha256(directory / 'predictions.pkl')
                record['cache_stats_sha256'] = sha256(directory / 'cache_stats.jsonl')
                record['cache_summary'] = summarize_cache(directory / 'cache_stats.jsonl')
                if (record['cache_summary']['frames'] != len(tokens) or
                        record['cache_summary']['unique_tokens'] != len(tokens)):
                    raise RuntimeError('Cache diagnostics did not cover every validation token')
                write_json(record_path, record)
            if variant == 'current' and not args.skip_legacy_canary:
                record['legacy_canary'] = legacy_canary(directory, epoch,
                    fingerprint['validation_sha256'], checkpoint_hash)
            diagnostic_path = directory / 'operating_points.json'
            diagnostic = json.loads(diagnostic_path.read_text()) if diagnostic_path.is_file() else {}
            if (diagnostic.get('predictions_sha256') != record['predictions_sha256'] or
                    diagnostic.get('validation_sha256') != fingerprint['validation_sha256']):
                diagnostic = dict(operating_diagnostics(directory, infos),
                    predictions_sha256=record['predictions_sha256'],
                    validation_sha256=fingerprint['validation_sha256'])
                write_json(diagnostic_path, diagnostic)
            record['operating_points'] = diagnostic
            summary['models'][name] = record
            write_json(summary_path, summary)
            print(name, {key: metrics for key, metrics in record['metrics'].items()
                if key in ('mAP', 'NDS', 'Car_AP')}, flush=True)
    if not args.dry_run:
        write_json(summary_path, summary)
        print(summary_path, flush=True)


if __name__ == '__main__':
    main()
else:
    _register_hook()
