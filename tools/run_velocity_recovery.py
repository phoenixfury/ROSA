#!/usr/bin/env python3
"""Small, paired XY-only temporal recovery pilot, NOT a full-network retrain.

Both arms use identical clips, fresh Adam state and frozen BN/dropout/DN.
Only recovery resets the learned velocity-gate projection rows to zero logits.
All other persistent state is hash-checked unchanged after optimization.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
ROSA = ROOT / 'projects/ROSA'
CONFIG = ROSA / 'configs/fisheye3dod/rosa_vov99_temporal_12e.py'
DEFAULT_CHECKPOINT = ROSA / 'work_dirs/rosa_vov99_temporal_12e/best_fisheye_bev_NDS_epoch_4.pth'
OFFICIAL = Path(os.environ.get('FISHEYE3DOD_OFFICIAL_ROOT', ROOT / 'projects/Fisheye3DOD')).resolve()
sys.path[:0] = [str(ROOT), str(ROSA), str(OFFICIAL), str(ROSA / 'tools')]
os.environ['FISHEYE3DOD_OFFICIAL_ROOT'] = str(OFFICIAL)
os.environ['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROSA), str(OFFICIAL),
                                         os.environ.get('PYTHONPATH', '')])
os.environ.setdefault('MPLCONFIGDIR', '/tmp/rosa-velocity-recovery-matplotlib')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def setup(args):
    import numpy as np
    for name, value in [('float', float), ('int', int), ('bool', bool)]:
        if name not in np.__dict__:
            setattr(np, name, value)
    os.environ['ROSA_DATA_ROOT'] = str(args.data_root)
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    config = Config.fromfile(str(CONFIG))
    init_default_scope(config.get('default_scope', 'mmdet3d'))
    return config


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def plan(args, config):
    from mmdet3d.registry import DATASETS
    from rosa.integrations.fisheye3dod.velocity_recovery import choose_chronological_clips
    from test_temporal_memory import source_fingerprint
    dataset = DATASETS.build(config.train_dataloader.dataset)
    selected = choose_chronological_clips(dataset.temporal_records, args.clips_per_scene, 32)
    by_scene = defaultdict(int)
    finite, moving = 0, 0
    import numpy as np
    for row in selected:
        by_scene[row['scene']] += 1
        assert 'train' in str(dataset.get_data_info(row['index'])['metainfo']['temporal_split'])
        velocity = dataset._temporal_velocities[row['token']]
        finite += int(np.isfinite(velocity).all(-1).sum())
        moving += int((np.linalg.norm(velocity, axis=-1) > 2.).sum())
    assert len(selected) % args.accumulate == 0
    files = source_fingerprint()
    for path in [Path(__file__).resolve(), ROSA / 'rosa/integrations/fisheye3dod/velocity_recovery.py',
                 ROSA / 'rosa/hybrid_box_decoder.py',
                 ROOT / 'projects/mmdet3d_plugin/models/detection3d/target.py']:
        files[str(path.relative_to(ROOT))] = sha256(path)
    result = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
        train_sha256=sha256(args.data_root / 'rosa/fisheye3dod_infos_train.pkl'),
        val_sha256=sha256(args.data_root / 'rosa/fisheye3dod_infos_val.pkl'),
        source_sha256=files, selected=selected, train_frames=len(selected),
        frames_by_scene=dict(by_scene), finite_target_objects_before_range_filter=finite,
        moving_target_objects_before_range_filter=moving, seed=args.seed,
        lr=args.lr, epochs=args.epochs, accumulate=args.accumulate,
        objective='existing free-query finite-XY L1, equally averaged over decoder stages',
        trainable='XY residual rows and velocity-gate row only; no weight decay; fresh optimizer',
        model_mode='eval: frozen BN/dropout; no GT-seeded denoising queries',
        cache='original class-ranked memory, 100 slots, predicted XY, no cache NMS',
        targets='unchanged causal two-observation DetZero pseudo-motion',
        promotion_checks=dict(moving_epe_relative_gain=.10, matched_moving_coverage=.95,
            car_ap_max_drop=.01, map_max_drop=.005, nds_max_drop=.005,
            duplicate_ratio_max=1.05),
        notes=['Pilot clip selection uses train scene/time only, no validation or class selection.',
               'Detection weights are frozen, but velocity feedback can still change geometry outputs.',
               'Pseudo-motion targets are not independent physical ground truth.'])
    return result


def verify_plan(args, manifest):
    assert sha256(args.checkpoint) == manifest['checkpoint_sha256'], 'Initialization changed'
    for relative, expected in manifest['source_sha256'].items():
        assert sha256(ROOT / relative) == expected, f'Source changed: {relative}'
    for split in ('train', 'val'):
        assert sha256(args.data_root / f'rosa/fisheye3dod_infos_{split}.pkl') == manifest[f'{split}_sha256']


def train_arm(args, config, manifest):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from mmengine.dataset import default_collate, DefaultSampler
    from mmengine.runner.checkpoint import load_checkpoint
    from mmdet3d.registry import DATASETS, MODELS
    from rosa.integrations.fisheye3dod.temporal import TemporalSequenceBatchSampler
    from rosa.integrations.fisheye3dod.velocity_recovery import (
        configure_velocity_recovery, protected_state_sha256, velocity_only_loss)
    seed_all(args.seed)
    directory = args.output_dir / args.arm
    directory.mkdir(exist_ok=True)
    if (directory / 'training.json').exists():
        raise FileExistsError(f'Refusing to overwrite completed training: {directory}')
    dataset = DATASETS.build(config.train_dataloader.dataset)
    # Keep the full causal target history; restrict only what the sampler visits.
    dataset.temporal_records = manifest['selected']
    sampler = DefaultSampler(dataset, shuffle=False, seed=args.seed)
    batches = TemporalSequenceBatchSampler(sampler, batch_size=1, clip_length=32,
        shuffle=True, seed=args.seed)
    loader = DataLoader(dataset, batch_sampler=batches, collate_fn=default_collate,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed))
    model = MODELS.build(config.model)
    model.init_weights()
    original = load_checkpoint(model, str(args.checkpoint), map_location='cpu', strict=True)
    metadata = dict(original.get('meta', {}))
    del original
    # No BN updates, dropout or GT-seeded DN: recover motion outputs in the
    # deployed free-query pathway, with otherwise frozen representation.
    model.cuda().eval()
    records = configure_velocity_recovery(model, reset_gates=args.arm == 'recovery')
    protected = protected_state_sha256(model, records)
    optimizer = torch.optim.AdamW([record.parameter for record in records], lr=args.lr,
        weight_decay=0.)
    scaler = torch.cuda.amp.GradScaler(init_scale=128., growth_interval=2000)
    optimizer.zero_grad(set_to_none=True)
    gate_values = {}
    handles = []
    for index, layer in enumerate(model.head.fisheye_box_decoder.layers):
        def capture(module, inputs, output, index=index):
            gate_values[index] = output[..., 3].detach()
        handles.append(layer.refinement.update_gate.register_forward_hook(capture))
    initial_path = directory / 'reset_only.pth'
    if args.arm == 'recovery':
        torch.save(dict(state_dict=model.state_dict(), meta=dict(metadata,
            velocity_recovery=dict(arm='reset_only', optimizer_steps=0))), initial_path)
    total_steps = args.epochs * len(loader)
    steps, updates, skipped, max_gradient = 0, 0, 0, 0.
    moving_seen = 0
    began = time.monotonic()
    log = (directory / 'steps.jsonl').open('w', buffering=1)
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.head.instance_bank.reset()
        for batch in loader:
            data = model.data_preprocessor(batch, training=False)
            metas = model._build_metas(data['inputs'], data['data_samples'])
            with torch.autocast('cuda', dtype=torch.float16):
                outputs = model._forward(**data)
            loss, diagnostics = velocity_only_loss(model.head, outputs, metas)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite velocity loss')
            scaler.scale(loss / args.accumulate).backward()
            steps += 1
            gradient_norm = None
            if steps % args.accumulate == 0:
                scaler.unscale_(optimizer)
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                    [record.parameter for record in records], 5.))
                if not np.isfinite(gradient_norm):
                    raise FloatingPointError('Nonfinite pilot gradient')
                max_gradient = max(max_gradient, gradient_norm)
                before_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < before_scale:
                    skipped += 1
                else:
                    updates += 1
                optimizer.zero_grad(set_to_none=True)
            pred, truth = diagnostics['prediction'], diagnostics['target']
            speed = truth.float().norm(dim=-1)
            moving = speed > 2.
            moving_seen += int(moving.sum())
            valid_mask = diagnostics['valid_mask']
            matched_gate = gate_values[max(gate_values)][valid_mask]
            row = dict(epoch=epoch + 1, step=steps, token=str(metas['img_metas'][0]['token']),
                loss=float(loss.detach()), valid_objects=len(truth), moving_objects=int(moving.sum()),
                prediction_speed_mean=float(pred.float().norm(dim=-1).mean()) if len(pred) else None,
                target_speed_mean=float(speed.mean()) if len(speed) else None,
                moving_velocity_epe=float((pred[moving]-truth[moving]).float().norm(dim=-1).mean())
                    if moving.any() else None,
                gate_mean=float(matched_gate.float().sigmoid().mean()) if len(matched_gate) else None,
                gate_fp16_zero_fraction=float((matched_gate.sigmoid() == 0).float().mean())
                    if len(matched_gate) else None,
                gradient_norm=gradient_norm, optimizer_steps=updates,
                elapsed_seconds=time.monotonic()-began)
            log.write(json.dumps(row, allow_nan=False) + '\n')
            if steps == 1 or steps % 25 == 0 or steps == total_steps:
                eta = (time.monotonic()-began) / steps * (total_steps-steps)
                print(f'{args.arm} [{steps}/{total_steps}] loss={row["loss"]:.4f} '
                    f'gate={row["gate_mean"]} updates={updates} ETA={eta/60:.1f}min', flush=True)
    log.close()
    for handle in handles:
        handle.remove()
    if args.arm == 'recovery' and max_gradient <= 1e-6:
        raise RuntimeError('Recovery produced no nonzero gradients')
    assert protected_state_sha256(model, records) == protected, 'Non-velocity state changed'
    verify_plan(args, manifest)
    model.head.instance_bank.reset()
    metadata['velocity_recovery'] = dict(arm=args.arm, steps=steps,
        optimizer_steps=updates, init_sha256=manifest['checkpoint_sha256'])
    checkpoint = directory / 'pilot.pth'
    torch.save(dict(state_dict=model.state_dict(), meta=metadata), checkpoint)
    result = dict(arm=args.arm, frames=steps, optimizer_steps=updates, skipped_steps=skipped,
        elapsed_seconds=time.monotonic()-began, checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint), protected_state_sha256=protected,
        non_velocity_state_unchanged=True, init_checkpoint_unchanged=True,
        moving_observations= moving_seen, maximum_gradient_norm=max_gradient,
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
        allowed_rows=[dict(name=r.name, rows=r.rows, kind=r.kind) for r in records])
    write_json(directory / 'training.json', result)
    print(json.dumps(result, indent=2), flush=True)


def evaluation(args, name, checkpoint):
    directory = args.output_dir / name
    directory.mkdir(exist_ok=True)
    config = directory / 'inference_config.py'
    config.write_text(f'_base_ = [{str(CONFIG)!r}]\n'
        f'test_dataloader = {dict(batch_size=1, num_workers=args.workers)!r}\n')
    command = [sys.executable, str(ROSA / 'tools/run.py'), 'evaluate', str(config),
        str(checkpoint), '--dump', str(directory / 'predictions.pkl'),
        '--work-dir', str(directory / 'work')]
    run_command(command, directory / 'evaluation.log')
    metrics = {match.group(1): float(match.group(2)) for match in re.finditer(
        r'fisheye_bev/([A-Za-z0-9_@.]+):\s*([-+0-9.eE]+)',
        (directory / 'evaluation.log').read_text(errors='replace'))}
    assert all(key in metrics for key in ('mAP', 'NDS', 'Car_AP'))
    write_json(directory / 'evaluation.json', dict(metrics=metrics,
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        predictions_sha256=sha256(directory / 'predictions.pkl')))
    print(name, {key: metrics[key] for key in ('mAP', 'NDS', 'Car_AP')}, flush=True)


def run_command(command, log):
    print('Starting:', ' '.join(map(str, command)), '\nLog:', log, flush=True)
    with Path(log).open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
            stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Experiment subprocess failed: {log}\n' +
            '\n'.join(Path(log).read_text(errors='replace').splitlines()[-15:]))


def summarize(args, config, manifest):
    import numpy as np
    # Reuse the canonical registered dataset module, avoiding duplicate aliases.
    import importlib
    package = importlib.import_module('Fisheye3DOD.dataset')
    sys.modules['dataset'] = package
    for key, module in list(sys.modules.items()):
        if key.startswith('Fisheye3DOD.dataset.'):
            sys.modules[key.replace('Fisheye3DOD.', '', 1)] = module
    from compare_retrained_fisheye import CLASSES, RANGES, load_frames, read_infos, select
    from test_temporal_memory import operating_diagnostics
    from rosa.integrations.fisheye3dod.temporal import track_velocities
    infos = read_infos(args.data_root / 'rosa/fisheye3dod_infos_val.pkl')
    targets = track_velocities(infos)
    cases = dict(original=ROSA / 'work_dirs/temporal_memory_ablation_20261031/epoch4_current',
        reset_only=args.output_dir / 'reset_only',
        control=args.output_dir / 'control_eval', recovery=args.output_dir / 'recovery_eval')
    report, matches = {}, {}
    for name, directory in cases.items():
        record = json.loads((directory / 'evaluation.json').read_text())
        assert sha256(directory / 'predictions.pkl') == record['predictions_sha256']
        if name == 'original':
            assert record['checkpoint_sha256'] == manifest['checkpoint_sha256']
            assert record['validation_sha256'] == manifest['val_sha256']
        frames, _ = load_frames(directory, infos, with_iou=False)
        diagnostic = operating_diagnostics(directory, infos)
        matched = {}
        for frame, info in zip(frames, infos):
            names = np.asarray(info['annos']['gt_names'], str)
            boxes = np.asarray(info['annos']['gt_boxes']).reshape(-1, 7)
            keep = np.asarray([np.linalg.norm(box[:2]) < RANGES[name]
                               for box, name in zip(boxes, names)], dtype=bool)
            velocity = targets[str(info['token'])][keep]
            rotation = np.asarray(info['metainfo']['T_world_CLIDAR'])[:3, :3]
            used = set()
            for index in select(frame, threshold=.1):
                distance = np.linalg.norm(frame['gt_boxes'][:, :2]-frame['boxes'][index, :2], axis=1)
                candidates = [j for j in np.flatnonzero(
                    (frame['gt_names'] == CLASSES[frame['labels'][index]]) & (distance < 2.))
                    if j not in used]
                if not candidates:
                    continue
                j = min(candidates, key=lambda j: distance[j]); used.add(j)
                if not np.isfinite(velocity[j]).all():
                    continue
                prediction = (rotation @ np.r_[frame['boxes'][index, 7:9], 0.])[:2]
                target = (rotation @ np.r_[velocity[j], 0.])[:2]
                matched[(frame['token'], j)] = dict(epe=float(np.linalg.norm(prediction-target)),
                    predicted_speed=float(np.linalg.norm(prediction)),
                    target_speed=float(np.linalg.norm(target)))
        matches[name] = matched
        report[name] = dict(metrics=record['metrics'], diagnostic=diagnostic,
            matched_finite_velocity=len(matched),
            matched_moving=sum(row['target_speed'] > 2. for row in matched.values()))
    common = set.intersection(*(set(rows) for rows in matches.values()))
    common = {key for key in common if matches['original'][key]['target_speed'] > 2.}
    for name, rows in matches.items():
        report[name]['common_moving'] = dict(count=len(common),
            mean_epe=float(np.mean([rows[key]['epe'] for key in common])) if common else None,
            median_predicted_speed=float(np.median([rows[key]['predicted_speed'] for key in common]))
                if common else None)
    c, r = report['control'], report['recovery']
    limits = manifest['promotion_checks']
    raw_epe, new_epe = c['common_moving']['mean_epe'], r['common_moving']['mean_epe']
    checks = dict(nonempty_common_cohort=len(common) >= 100,
        moving_epe_improved=bool(raw_epe and new_epe <= raw_epe*(1-limits['moving_epe_relative_gain'])),
        matched_moving_coverage=r['matched_moving'] >= limits['matched_moving_coverage']*report['original']['matched_moving'],
        car_ap_preserved=r['metrics']['Car_AP'] >= c['metrics']['Car_AP']-limits['car_ap_max_drop'],
        map_preserved=r['metrics']['mAP'] >= c['metrics']['mAP']-limits['map_max_drop'],
        nds_preserved=r['metrics']['NDS'] >= c['metrics']['NDS']-limits['nds_max_drop'],
        duplicates_preserved=r['diagnostic']['score_thresholds']['0.1']['total']['duplicate_fp'] <=
            limits['duplicate_ratio_max']*c['diagnostic']['score_thresholds']['0.1']['total']['duplicate_fp'])
    result = dict(models=report, checks=checks, passes_pilot=all(checks.values()),
        next_step='A passed pilot permits a larger controlled experiment, not automatic deployment/full retraining.',
        no_automatic_training_extension=True)
    write_json(args.output_dir / 'results.json', result)
    lines = ['# Controlled velocity-gate recovery pilot', '',
        'Matched-cohort velocity metrics use causal DetZero pseudo-motion, not independent ground truth.', '',
        '| Model | mAP % | NDS | Car AP % | Shared moving EPE m/s | Shared predicted speed median m/s |',
        '|---|---:|---:|---:|---:|---:|']
    for name, row in report.items():
        metric, motion = row['metrics'], row['common_moving']
        lines.append(f'| {name} | {metric["mAP"]*100:.2f} | {metric["NDS"]:.4f} | '
            f'{metric["Car_AP"]*100:.2f} | {motion["mean_epe"]} | {motion["median_predicted_speed"]} |')
    lines += ['', f'Shared moving cohort: {len(common)} objects.', '',
        f'Predeclared pilot checks: {checks}', '',
        f'Passes pilot: {result["passes_pilot"]}. No automatic training extension.', '',
        'See plan.json, per-arm training.json/steps.jsonl and results.json for provenance, '
        'frozen-state checks, ETA/learning curves and detection operating points.']
    (args.output_dir / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(checks=checks, passes_pilot=result['passes_pilot']), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
        default=ROSA / 'work_dirs/velocity_gate_recovery_20261031')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--data-root', type=Path, default=Path(os.environ.get('ROSA_DATA_ROOT',
        ROOT / 'data/Custom/combined_improved_detzero_rosa')))
    parser.add_argument('--clips-per-scene', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--accumulate', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--phase', choices=('trial', 'train', 'analyze'), default='trial')
    parser.add_argument('--arm', choices=('control', 'recovery'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    assert args.epochs > 0 and args.accumulate > 0 and args.lr > 0
    args.output_dir = args.output_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.data_root = args.data_root.resolve()
    config = setup(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / 'plan.json'
    if args.phase == 'trial':
        if manifest_path.exists():
            raise FileExistsError('Use a new output directory for a new paired trial')
        manifest = plan(args, config)
        if args.dry_run:
            print(json.dumps({k: v for k, v in manifest.items() if k not in ('selected', 'source_sha256')}, indent=2))
            return
        write_json(manifest_path, manifest)
    else:
        manifest = json.loads(manifest_path.read_text())
    verify_plan(args, manifest)
    if args.phase == 'train':
        assert args.arm
        train_arm(args, config, manifest)
        return
    if args.phase == 'analyze':
        summarize(args, config, manifest)
        return
    shared = ['--output-dir', str(args.output_dir), '--checkpoint', str(args.checkpoint),
        '--data-root', str(args.data_root), '--epochs', str(args.epochs), '--lr', str(args.lr),
        '--accumulate', str(args.accumulate), '--workers', str(args.workers), '--seed', str(args.seed)]
    for arm in ('control', 'recovery'):
        run_command([sys.executable, str(Path(__file__).resolve()), *shared,
            '--phase', 'train', '--arm', arm], args.output_dir / f'{arm}_train.log')
        if arm == 'recovery':
            evaluation(args, 'reset_only', args.output_dir / arm / 'reset_only.pth')
        evaluation(args, f'{arm}_eval', args.output_dir / arm / 'pilot.pth')
        verify_plan(args, manifest)
        gc.collect()
    summarize(args, config, manifest)


if __name__ == '__main__':
    main()
