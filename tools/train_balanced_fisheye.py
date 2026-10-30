#!/usr/bin/env python3
"""Validate the portable dataset and launch corrected ROSA on one or more GPUs."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
ROSA = ROOT / 'projects/ROSA'
OFFICIAL = Path(os.environ.get('FISHEYE3DOD_OFFICIAL_ROOT', ROOT / 'projects/Fisheye3DOD')).resolve()
sys.path[:0] = [str(ROOT), str(ROSA), str(OFFICIAL)]
os.environ['FISHEYE3DOD_OFFICIAL_ROOT'] = str(OFFICIAL)
os.environ['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROSA), str(OFFICIAL),
                                         os.environ.get('PYTHONPATH', '')])
os.environ.setdefault('MPLCONFIGDIR', '/tmp/rosa-train-matplotlib')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
for name, value in [('float', float), ('int', int), ('bool', bool)]:
    if name not in np.__dict__:
        setattr(np, name, value)
import torch
from mmengine.config import Config
from mmengine.fileio import load
from mmengine.registry import init_default_scope
from mmdet3d.registry import DATASETS, MODELS, TASK_UTILS

CLASSES = ('Car', 'Van', 'Truck', 'Bus', 'Pedestrian', 'Cyclist')
RECIPES = {
    'finetune': 'rosa_vov99_balanced_finetune_8e.py',
    'retrain': 'rosa_vov99_balanced_retrain_24e.py',
    'baseline-adapt': 'rosa_vov99_baseline_adapt_8e.py',
}


def batch_plan(gpus, batch_size, effective_batch):
    if min(gpus, batch_size, effective_batch) < 1:
        raise ValueError('GPU count and batch sizes must be positive')
    micro_batch = gpus * batch_size
    if effective_batch % micro_batch:
        raise ValueError(f'Effective batch {effective_batch} must be divisible by '
                         f'{gpus} GPUs x {batch_size} samples/GPU')
    return dict(per_gpu_batch=batch_size, accumulation=effective_batch // micro_batch,
                effective_batch=effective_batch)


def runtime_options(config, plan, workers):
    original_batch = config.train_dataloader.batch_size
    options = [f'train_dataloader.batch_size={plan["per_gpu_batch"]}',
               f'optim_wrapper.accumulative_counts={plan["accumulation"]}',
               f'train_dataloader.num_workers={workers}',
               f'train_dataloader.persistent_workers={workers > 0}',
               'train_dataloader.pin_memory=True',
               'model.data_preprocessor.non_blocking=True',
               f'experiment_contract.per_gpu_batch={plan["per_gpu_batch"]}',
               f'experiment_contract.effective_batch={plan["effective_batch"]}']
    if workers:
        options.append('train_dataloader.prefetch_factor=2')
    # Preserve warmup exposure/optimizer-update count when moving from
    # accumulation-heavy batch 1 to larger physical batches.
    ratio = original_batch / plan['per_gpu_batch']
    for index, schedule in enumerate(config.param_scheduler):
        if schedule.get('type') == 'LinearLR' and not schedule.get('by_epoch', False):
            begin = round(schedule.get('begin', 0) * ratio)
            end = max(begin + 1, round(schedule['end'] * ratio))
            options += [f'param_scheduler.{index}.begin={begin}',
                        f'param_scheduler.{index}.end={end}']
    return options


def merge_options(config, values):
    from mmengine.config import DictAction
    overrides = dict(item.split('=', 1) for item in values)
    config.merge_from_dict({key: DictAction._parse_iterable(value)
                            for key, value in overrides.items()})


def training_command(args, config_path, work_dir, options):
    command = [sys.executable]
    if args.gpus > 1:
        command += ['-m', 'torch.distributed.run', '--nnodes=1',
                    f'--nproc_per_node={args.gpus}', f'--master_port={args.master_port}',
                    '--master_addr=127.0.0.1']
    command += [str(ROSA / 'tools/run.py'), 'train', str(config_path), '--amp',
                '--launcher', 'pytorch' if args.gpus > 1 else 'none',
                '--work-dir', str(work_dir)]
    if args.fast:
        command.append('--fast')
    if options:
        command += ['--cfg-options', *options]
    return command


def select_probe_result(rows, memory_margin):
    acceptable = [row for row in rows if row.get('status') == 'ok' and all(
        rank['peak_allocated_bytes'] <= (1. - memory_margin) * rank['total_memory_bytes']
        for rank in row['ranks'])]
    if not acceptable:
        raise RuntimeError('No batch passed the throughput/memory probe; inspect batch_probe logs')
    return max(acceptable, key=lambda row: row['samples_per_second'])


def auto_batch(args, config, config_path, work_dir, dense_indices):
    candidates = sorted(set(args.batch_candidates))
    candidates = [batch for batch in candidates if batch > 0 and
                  args.effective_batch % (batch * args.gpus) == 0]
    if not candidates:
        raise ValueError('No batch candidate divides the requested effective batch')
    rows = []
    probe_root = work_dir / 'batch_probe'
    probe_root.mkdir(parents=True, exist_ok=True)
    for batch in candidates:
        plan = batch_plan(args.gpus, batch, args.effective_batch)
        probe_dir = probe_root / f'batch_{batch}'
        probe_dir.mkdir(parents=True, exist_ok=True)
        # An earlier probe must never be mistaken for a successful new run.
        report = probe_dir / 'throughput_probe.json'
        if report.exists():
            report.rename(probe_dir / f'throughput_probe.previous.{time.time_ns()}.json')
        options = [*args.cfg_options, *runtime_options(config, plan, args.workers),
                   'train_dataloader.dataset.dataset.indices=' + repr(dense_indices)]
        command = training_command(args, config_path, probe_dir, options)
        accumulation = plan['accumulation']
        warmup = ((args.probe_warmup + accumulation - 1) // accumulation) * accumulation
        measured = ((args.probe_iterations - args.probe_warmup + accumulation - 1) // accumulation) * accumulation
        command[command.index('--cfg-options'):command.index('--cfg-options')] = [
            '--throughput-probe', str(warmup + measured), '--probe-warmup', str(warmup)]
        log_path = probe_dir / 'console.log'
        print(f'Probing {batch} samples/GPU, accumulation {plan["accumulation"]}; log: {log_path}', flush=True)
        with log_path.open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            log_text = log_path.read_text(errors='replace')
            if 'out of memory' in log_text.lower() or 'cublas_status_alloc_failed' in log_text.lower():
                row = dict(status='oom', **plan)
                print(f'Batch {batch}: CUDA out of memory; trying the next candidate.', flush=True)
            else:
                raise RuntimeError(f'Training probe failed (not an OOM); inspect {log_path}\n' +
                                   '\n'.join(log_text.splitlines()[-12:]))
        else:
            row = dict(json.loads(report.read_text()), status='ok', **plan)
            peak_gib = max(rank['peak_allocated_bytes'] for rank in row['ranks']) / 2**30
            print(f'Batch {batch}: {row["samples_per_second"]:.2f} frames/s, '
                  f'peak allocated {peak_gib:.2f} GiB/GPU', flush=True)
        rows.append(row)
        (probe_root / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    selected = select_probe_result(rows, args.memory_margin)
    (probe_root / 'selected.json').write_text(json.dumps(selected, indent=2) + '\n')
    print(f'Selected batch {selected["per_gpu_batch"]}/GPU, accumulation '
          f'{selected["accumulation"]}, effective batch {args.effective_batch}', flush=True)
    return batch_plan(args.gpus, selected['per_gpu_batch'], args.effective_batch)


def preflight(config, work_dir):
    if (config.model.img_backbone.type == 'VoVNetCP' and
            not config.model.img_backbone.norm_eval):
        backbone_class = MODELS.get('VoVNetCP')
        source = sys.modules[backbone_class.__module__]
        if not hasattr(source, '_private_batchnorm_recompute_buffers'):
            raise RuntimeError('Live BatchNorm needs the checkpoint-recompute fix. '
                'Commit/pull projects/Fisheye3DOD as well as projects/ROSA before training.')
    checkpoint = Path(config.load_from)
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Initialization checkpoint missing: {checkpoint}; use --init-checkpoint')
    anchor = Path(config.model.head.instance_bank.anchor)
    if not anchor.is_file() or np.load(anchor).shape != (300, 11):
        raise ValueError(f'Missing/invalid 300x11 anchor file: {anchor}')
    init_default_scope(config.default_scope)
    dataset = DATASETS.build(config.train_dataloader.dataset)
    counts, review, empty = Counter(), [], 0
    dn_counts = Counter()
    sampler = TASK_UTILS.build(config.model.head.sampler)
    torch.manual_seed(config.randomness.seed)
    missing_images = []
    base = dataset.dataset
    density = []
    for index in range(len(base)):
        sample = base.get_data_info(index)
        ann = sample['annos']
        names = np.asarray(ann['gt_names'], dtype=str)
        boxes = np.asarray(ann['gt_boxes']).reshape(-1, 7)
        if not np.isfinite(boxes).all() or (boxes[:, 3:6] <= 0).any():
            raise ValueError(f'Invalid GT geometry: {sample["token"]}')
        counts.update(names.tolist())
        density.append(len(names))
        limits = np.asarray([48., 48., 48., 48., 36., 42.])
        labels = np.asarray([CLASSES.index(name) for name in names], dtype=np.int64)
        in_range = np.linalg.norm(boxes[:, :2], axis=1) < limits[labels]
        effective_labels = torch.as_tensor(labels[in_range], dtype=torch.long)
        selected_labels = effective_labels[sampler._select_dn_gt(effective_labels)]
        dn_counts.update(CLASSES[label] for label in selected_labels.tolist())
        empty += len(names) == 0
        cameras = sample['cam_info']['cam_fisheye']
        for camera in cameras.values():
            if not Path(camera['cam_path']).is_file():
                missing_images.append(camera['cam_path'])
        for box, name, track in zip(boxes, names, ann['gt_ids']):
            if name in ('Van', 'Cyclist') or (name in ('Truck', 'Bus') and box[5] < 2.):
                review.append(dict(token=sample['token'], sequence=sample['metainfo']['scene_name'],
                    track_id=int(track), label=name,
                    length=float(box[3]), width=float(box[4]), height=float(box[5]),
                    front_image=cameras['fisheye_camera_front']['cam_path'],
                    reason='rare class' if name in ('Van', 'Cyclist') else 'low vehicle height; review subtype'))
    if missing_images:
        raise FileNotFoundError(f'{len(missing_images)} camera images missing, including {missing_images[:3]}')
    for loader_name in ['val_dataloader', 'test_dataloader']:
        annotation = Path(config[loader_name].dataset.ann_file)
        if not annotation.is_file():
            raise FileNotFoundError(annotation)
        validation = load(annotation)
        for sample in validation['data_list']:
            for camera in sample['cam_info']['cam_fisheye'].values():
                path = Path(camera['cam_path'])
                if not path.is_absolute():
                    path = Path(config.data_root) / path
                if not path.is_file():
                    raise FileNotFoundError(path)
    work_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(recipe=config.experiment_contract.training_recipe,
        data_root=config.data_root, initialization=str(checkpoint.resolve()),
        original_train_frames=len(base), balanced_train_frames=len(dataset),
        empty_train_frames=empty, raw_class_counts=dict(counts),
        class_weights=config.model.head.loss_cls.class_weights,
        max_frame_repeats=max(Counter(dataset.repeat_indices).values()),
        example_stratified_dn_counts=dict(dn_counts),
        per_gpu_batch=config.train_dataloader.batch_size,
        accumulation=config.optim_wrapper.accumulative_counts,
        effective_batch=config.experiment_contract.get('effective_batch'),
        epochs=config.train_cfg.max_epochs,
        backbone_frozen_stages=config.model.img_backbone.frozen_stages,
        backbone_norm_eval=config.model.img_backbone.norm_eval,
        head_peak_lr=config.optim_wrapper.optimizer.lr,
        backbone_peak_lr=(config.optim_wrapper.optimizer.lr *
            config.optim_wrapper.paramwise_cfg.custom_keys.img_backbone.lr_mult),
        decoder=dict(config.model.head.decoder),
        annotation_review_rows=len(review),
        note='Review rows flag candidates; no labels were automatically changed.')
    (work_dir / 'balanced_preflight.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (work_dir / 'rare_class_review.csv').open('w', newline='') as stream:
        fields = ['token', 'sequence', 'track_id', 'label', 'length', 'width', 'height', 'front_image', 'reason']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review)
    print(json.dumps(summary, indent=2), flush=True)
    # Stress all candidates with crowded frames (the largest DN query count),
    # rather than letting a random empty batch suggest an unsafe batch size.
    return np.argsort(-np.asarray(density), kind='stable')[:64].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', choices=list(RECIPES), default='finetune')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2, help='Samples per GPU; default 2')
    parser.add_argument('--effective-batch', type=int, default=8,
                        help='Global effective batch; accumulation is calculated automatically')
    parser.add_argument('--workers', type=int, default=4, help='Dataloader workers per GPU')
    parser.add_argument('--fast', action='store_true', help='Enable cuDNN tuning and TF32')
    parser.add_argument('--auto-batch', action='store_true', help='Benchmark candidates, then train with the fastest safe one')
    parser.add_argument('--batch-candidates', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--probe-only', action='store_true', help='Benchmark batches and exit without full training')
    parser.add_argument('--probe-iterations', type=int, default=12)
    parser.add_argument('--probe-warmup', type=int, default=4)
    parser.add_argument('--memory-margin', type=float, default=.08,
                        help='Minimum fraction of GPU memory not allocated by PyTorch during the probe')
    parser.add_argument('--master-port', type=int, default=29503)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--smoke-test', action='store_true', help='Four real AMP training iterations, no validation')
    parser.add_argument('--resume', type=Path, help='Resume a corrected-recipe checkpoint, not the old epoch-16 optimizer')
    parser.add_argument('--cfg-options', nargs='*', default=[])
    args = parser.parse_args()
    if args.gpus < 1:
        parser.error('--gpus must be positive')
    if args.workers < 0 or not 0 <= args.memory_margin < 1:
        parser.error('workers must be nonnegative and memory-margin must be in [0,1)')
    if args.probe_warmup < 0 or args.probe_iterations <= args.probe_warmup:
        parser.error('probe-iterations must exceed probe-warmup >= 0')
    if args.probe_only and not args.auto_batch:
        parser.error('--probe-only requires --auto-batch')
    if args.auto_batch and (args.resume or args.smoke_test):
        parser.error('--auto-batch needs a fresh run; cannot combine with --resume or --smoke-test')
    if args.data_root:
        os.environ['ROSA_DATA_ROOT'] = str(args.data_root.resolve())
    if args.init_checkpoint:
        os.environ['ROSA_INIT_CKPT'] = str(args.init_checkpoint.resolve())
    config_path = ROSA / 'configs/fisheye3dod' / RECIPES[args.recipe]
    config = Config.fromfile(config_path)
    reserved = {'train_dataloader.batch_size', 'optim_wrapper.accumulative_counts',
                'train_dataloader.num_workers'}
    if any(item.split('=', 1)[0] in reserved for item in args.cfg_options):
        parser.error('Use --batch-size, --effective-batch and --workers for batch/worker settings')
    merge_options(config, args.cfg_options)
    original_config = copy.deepcopy(config)
    plan = batch_plan(args.gpus, args.batch_size, args.effective_batch)
    merge_options(config, runtime_options(original_config, plan, args.workers))
    work_dir = (args.work_dir or Path(config.work_dir)).resolve()
    if args.smoke_test and args.work_dir is None:
        work_dir = work_dir.with_name(work_dir.name + '_smoke')
    if not args.preflight_only and not args.resume and (work_dir / 'last_checkpoint').exists():
        raise FileExistsError(f'{work_dir} already has checkpoints; use --resume or a new --work-dir')
    dense_indices = preflight(config, work_dir)
    if args.preflight_only:
        return
    if torch.cuda.device_count() < args.gpus:
        raise RuntimeError(f'{args.gpus} GPUs requested, but {torch.cuda.device_count()} visible; set --gpus appropriately')
    if args.auto_batch:
        plan = auto_batch(args, original_config, config_path, work_dir, dense_indices)
        merge_options(config, runtime_options(original_config, plan, args.workers))
        preflight(config, work_dir)
        if args.probe_only:
            return
    overrides = [*args.cfg_options, *runtime_options(original_config, plan, args.workers)]
    if args.smoke_test:
        overrides += [
            'train_cfg._delete_=True', 'train_cfg.type=IterBasedTrainLoop',
            'train_cfg.max_iters=4', 'train_cfg.val_interval=1000',
            'val_cfg=None', 'val_dataloader=None', 'val_evaluator=None',
            'param_scheduler=[]', 'custom_hooks=[]',
            'train_dataloader.num_workers=0', 'train_dataloader.persistent_workers=False',
            'optim_wrapper.accumulative_counts=1',
            'default_hooks.logger.interval=1', 'default_hooks.checkpoint.save_best=None',
            'default_hooks.checkpoint.rule=None',
            'default_hooks.checkpoint.interval=10000', 'default_hooks.checkpoint.by_epoch=False',
        ]
    command = training_command(args, config_path, work_dir, overrides)
    if args.resume:
        position = command.index('--cfg-options')
        command[position:position] = ['--resume', str(args.resume.resolve())]
    print('$ ' + shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)


if __name__ == '__main__':
    main()
