#!/usr/bin/env python3
"""Paired joint-decoder pilot: inherited versus reset velocity gates.

Reuses the frozen-representation pilot's data selection, metric implementation
and inference path. No prior experiment implementation is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import run_velocity_recovery as base

ROOT, ROSA = base.ROOT, base.ROSA
PRIOR = ROSA / 'work_dirs/velocity_gate_recovery_20261031'


def make_plan(args, config):
    result = base.plan(args, config)
    for path in [Path(__file__).resolve(), ROSA / 'rosa/integrations/fisheye3dod/joint_velocity_recovery.py']:
        result['source_sha256'][str(path.relative_to(ROOT))] = base.sha256(path)
    # Fail before any GPU training if the saved comparison references do not fit.
    original = ROSA / 'work_dirs/temporal_memory_ablation_20261031/epoch4_current'
    record = json.loads((original / 'evaluation.json').read_text())
    assert record['checkpoint_sha256'] == result['checkpoint_sha256']
    assert record['validation_sha256'] == result['val_sha256']
    assert base.sha256(original / 'predictions.pkl') == record['predictions_sha256']
    prior = json.loads((args.reset_reference.parent / 'plan.json').read_text())
    assert prior['checkpoint_sha256'] == result['checkpoint_sha256']
    assert prior['val_sha256'] == result['val_sha256']
    reset = json.loads((args.reset_reference / 'evaluation.json').read_text())
    assert base.sha256(args.reset_reference / 'predictions.pkl') == reset['predictions_sha256']
    result.update(experiment_id='E006_joint_decoder_recovery', parent_experiment='E005_velocity_rows',
        objective='unchanged native free-query class/box/centerness/yawness/ray losses summed over stages',
        trainable='entire existing fisheye decoder, excluding Z residual rows; backbone/FPN/priors frozen',
        model_mode='eval with autograd; frozen BN/dropout; no GT-seeded DN',
        reset_reference=str(args.reset_reference), optimizer='fresh AdamW, no decay, clip norm 5',
        paired_input_check='SHA256 of each complete augmented CPU image tensor',
        additional_detection_guard='recovery also must preserve original mAP/NDS/Car AP within the same tolerances',
        notes=['Train clips and targets match E005; both new arms share all settings except gate initialization.',
               'Normal detection losses and decoder features now adapt together; frozen representation E005 is a separate experiment.',
               'Frozen backbone/FPN/BN/priors and Z output rows are verified, but geometry/classification weights now change.',
               'Pseudo-motion targets are not independent physical ground truth.'])
    return result


def train_arm(args, config, manifest):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from mmengine.dataset import default_collate, DefaultSampler
    from mmengine.runner.checkpoint import load_checkpoint
    from mmdet3d.registry import DATASETS, MODELS
    from rosa.integrations.fisheye3dod.temporal import TemporalSequenceBatchSampler
    from rosa.integrations.fisheye3dod.velocity_recovery import velocity_only_loss
    from rosa.integrations.fisheye3dod.joint_velocity_recovery import (
        configure_joint_decoder, protected_joint_state_sha256)
    for key in ('lr', 'epochs', 'accumulate', 'seed'):
        assert getattr(args, key) == manifest[key], f'Plan override forbidden: {key}'
    base.seed_all(args.seed)
    directory = args.output_dir / args.arm
    directory.mkdir(exist_ok=True)
    if (directory / 'training.json').exists():
        raise FileExistsError(f'Refusing to overwrite {directory}')
    dataset = DATASETS.build(config.train_dataloader.dataset)
    dataset.temporal_records = manifest['selected']
    sampler = DefaultSampler(dataset, shuffle=False, seed=args.seed)
    batches = TemporalSequenceBatchSampler(sampler, batch_size=1, clip_length=32,
        shuffle=True, seed=args.seed)
    loader = DataLoader(dataset, batch_sampler=batches, collate_fn=default_collate,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed))
    model = MODELS.build(config.model)
    model.init_weights()
    checkpoint = load_checkpoint(model, str(args.checkpoint), map_location='cpu', strict=True)
    metadata = dict(checkpoint.get('meta', {})); del checkpoint
    model.cuda().eval()
    trainable, protected_rows = configure_joint_decoder(model, reset_gates=args.arm == 'recovery')
    protected = protected_joint_state_sha256(model, trainable, protected_rows)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.)
    scaler = torch.cuda.amp.GradScaler(init_scale=1., growth_interval=2000)
    optimizer.zero_grad(set_to_none=True)
    gates = {}
    handles = []
    for index, layer in enumerate(model.head.fisheye_box_decoder.layers):
        def capture(module, inputs, output, index=index):
            gates[index] = output[..., 3].detach()
        handles.append(layer.refinement.update_gate.register_forward_hook(capture))
    steps = updates = skipped = moving_seen = 0
    maximum_gradient = 0.
    total = len(loader) * args.epochs
    began = time.monotonic()
    with (directory / 'steps.jsonl').open('w', buffering=1) as log:
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            model.head.instance_bank.reset()
            for batch in loader:
                image = batch['inputs']['img'].contiguous()
                image_hash = hashlib.sha256(image.numpy().tobytes()).hexdigest()
                data = model.data_preprocessor(batch, training=False)
                metas = model._build_metas(data['inputs'], data['data_samples'])
                with torch.autocast('cuda', dtype=torch.float16):
                    outputs = model._forward(**data)
                    losses = model.head.loss(outputs, metas)
                loss = sum(value for key, value in losses.items() if 'loss' in key)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite joint loss')
                with torch.no_grad():
                    _, diagnostic = velocity_only_loss(model.head,
                        dict(classification=[outputs['classification'][-1]],
                             prediction=[outputs['prediction'][-1]]), metas)
                scaler.scale(loss / args.accumulate).backward()
                steps += 1
                gradient_norm = None
                if steps % args.accumulate == 0:
                    scaler.unscale_(optimizer)
                    gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 5.))
                    before_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() < before_scale:
                        skipped += 1
                    else:
                        assert np.isfinite(gradient_norm)
                        updates += 1
                        maximum_gradient = max(maximum_gradient, gradient_norm)
                    optimizer.zero_grad(set_to_none=True)
                pred, truth = diagnostic['prediction'], diagnostic['target']
                moving = truth.float().norm(dim=-1) > 2.
                moving_seen += int(moving.sum())
                gate = gates[max(gates)][diagnostic['valid_mask']]
                row = dict(epoch=epoch+1, step=steps, token=str(metas['img_metas'][0]['token']),
                    input_sha256=image_hash, loss=float(loss.detach()),
                    classification_loss=float(sum(value.detach() for key, value in losses.items()
                        if key.startswith('loss_cls'))),
                    box_loss=float(sum(value.detach() for key, value in losses.items()
                        if key.startswith('loss_box'))),
                    valid_objects=len(truth), moving_objects=int(moving.sum()),
                    prediction_speed_mean=float(pred.float().norm(dim=-1).mean()) if len(pred) else None,
                    target_speed_mean=float(truth.float().norm(dim=-1).mean()) if len(truth) else None,
                    moving_velocity_epe=float((pred[moving]-truth[moving]).float().norm(dim=-1).mean())
                        if moving.any() else None,
                    gate_mean=float(gate.float().sigmoid().mean()) if len(gate) else None,
                    gate_fp16_zero_fraction=float((gate.sigmoid()==0).float().mean()) if len(gate) else None,
                    gradient_norm=gradient_norm if gradient_norm is None or np.isfinite(gradient_norm) else None,
                    optimizer_steps=updates, skipped_steps=skipped, elapsed_seconds=time.monotonic()-began)
                log.write(json.dumps(row, allow_nan=False)+'\n')
                if steps == 1 or steps % 25 == 0 or steps == total:
                    eta = (time.monotonic()-began)/steps*(total-steps)/60
                    print(f'joint {args.arm} [{steps}/{total}] loss={row["loss"]:.4f} '
                        f'gate={row["gate_mean"]} updates={updates} skips={skipped} ETA={eta:.1f}min', flush=True)
    for handle in handles:
        handle.remove()
    assert maximum_gradient > 1e-6
    assert protected_joint_state_sha256(model, trainable, protected_rows) == protected
    base.verify_plan(args, manifest)
    model.head.instance_bank.reset()
    metadata['joint_velocity_recovery'] = dict(arm=args.arm, optimizer_steps=updates,
        initialization_sha256=manifest['checkpoint_sha256'])
    path = directory / 'pilot.pth'
    torch.save(dict(state_dict=model.state_dict(), meta=metadata), path)
    base.write_json(directory / 'training.json', dict(arm=args.arm, frames=steps,
        optimizer_steps=updates, skipped_steps=skipped, elapsed_seconds=time.monotonic()-began,
        checkpoint=str(path), checkpoint_sha256=base.sha256(path),
        trainable_names=sorted(trainable), trainable_parameter_count=sum(p.numel() for p in parameters),
        protected_rows=protected_rows, protected_state_sha256=protected,
        backbone_fpn_priors_buffers_z_unchanged=True, moving_observations=moving_seen,
        maximum_gradient_norm=maximum_gradient, peak_cuda_memory_bytes=torch.cuda.max_memory_allocated()))


def analyze(args, config, manifest):
    base.summarize(args, config, manifest)
    rows = {}
    for arm in ('control', 'recovery'):
        rows[arm] = [json.loads(line) for line in (args.output_dir/arm/'steps.jsonl').read_text().splitlines()]
    paired = [(row['token'], row['input_sha256']) for row in rows['control']] == [
        (row['token'], row['input_sha256']) for row in rows['recovery']]
    c = json.loads((args.output_dir/'control/training.json').read_text())
    r = json.loads((args.output_dir/'recovery/training.json').read_text())
    result = json.loads((args.output_dir/'results.json').read_text())
    original, recovery = result['models']['original'], result['models']['recovery']
    checks = result['checks']
    checks['identical_augmented_inputs'] = paired
    checks['equal_optimizer_updates_no_skips'] = c['optimizer_steps']==r['optimizer_steps'] and c['skipped_steps']==r['skipped_steps']==0
    checks['identical_protected_state'] = c['protected_state_sha256']==r['protected_state_sha256']
    for key, tolerance in [('Car_AP', .01), ('mAP', .005), ('NDS', .005)]:
        checks[f'original_{key}_preserved'] = recovery['metrics'][key] >= original['metrics'][key]-tolerance
    result.update(experiment_id=manifest['experiment_id'], passes_pilot=all(checks.values()),
        protocol='joint decoder, normal detection losses, frozen backbone/FPN/BN/priors/Z rows')
    base.write_json(args.output_dir/'results.json', result)
    report = args.output_dir/'REPORT.md'
    content = report.read_text()
    content = content.replace('# Controlled velocity-gate recovery pilot', '# Joint decoder velocity-gate adaptation pilot', 1)
    report.write_text(content + '\n## Joint protocol checks\n\n' + json.dumps(checks, indent=2) +
        f'\n\nFinal joint-pilot decision: {result["passes_pilot"]}. No automatic extension/promotion.\n')
    print('Final joint checks:', checks, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROSA/'work_dirs/joint_decoder_recovery_20261031')
    parser.add_argument('--checkpoint', type=Path, default=base.DEFAULT_CHECKPOINT)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/Custom/combined_improved_detzero_rosa')
    parser.add_argument('--reset-reference', type=Path, default=PRIOR/'reset_only')
    parser.add_argument('--clips-per-scene', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--accumulate', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--phase', choices=('trial','train','analyze'), default='trial')
    parser.add_argument('--arm', choices=('control','recovery'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    assert args.epochs > 0 and args.accumulate > 0 and args.lr > 0 and args.workers >= 0
    for key in ('output_dir','checkpoint','data_root','reset_reference'):
        setattr(args,key,getattr(args,key).resolve())
    config = base.setup(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/'plan.json'
    if args.phase == 'trial':
        if path.exists():
            raise FileExistsError('Use a fresh trial directory')
        manifest = make_plan(args, config)
        if args.dry_run:
            print(json.dumps({k:v for k,v in manifest.items() if k not in ('selected','source_sha256')}, indent=2))
            return
        base.write_json(path,manifest)
        (args.output_dir/'reset_only').symlink_to(args.reset_reference, target_is_directory=True)
    else:
        manifest = json.loads(path.read_text())
    base.verify_plan(args,manifest)
    if args.phase == 'train':
        assert args.arm
        train_arm(args,config,manifest)
        return
    if args.phase == 'analyze':
        analyze(args,config,manifest)
        return
    shared = ['--output-dir',str(args.output_dir),'--checkpoint',str(args.checkpoint),
        '--data-root',str(args.data_root),'--reset-reference',str(args.reset_reference),
        '--epochs',str(args.epochs),'--lr',str(args.lr),'--accumulate',str(args.accumulate),
        '--workers',str(args.workers),'--seed',str(args.seed)]
    for arm in ('control','recovery'):
        base.run_command([sys.executable,str(Path(__file__).resolve()),*shared,
            '--phase','train','--arm',arm],args.output_dir/f'{arm}_train.log')
        base.evaluation(args,f'{arm}_eval',args.output_dir/arm/'pilot.pth')
        base.verify_plan(args,manifest)
    analyze(args,config,manifest)


if __name__ == '__main__':
    main()
