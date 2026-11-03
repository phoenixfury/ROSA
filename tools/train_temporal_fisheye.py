#!/usr/bin/env python3
"""Validate chronological data and launch temporal ROSA on one or more GPUs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

# Reuse the portable environment, AMP launcher and batch/accumulation contract.
from train_balanced_fisheye import (
    ROOT, ROSA, Config, DATASETS, init_default_scope, np, torch,
    batch_plan, merge_options, runtime_options, training_command,
)
from mmengine.dataset import DefaultSampler
from mmdet3d.registry import TRANSFORMS
from rosa.integrations.fisheye3dod.dataset import Fisheye3DODEvaluatorRangeFilter
from rosa.integrations.fisheye3dod.temporal import TemporalSequenceBatchSampler


def preflight(config, work_dir, gpus):
    init_default_scope(config.default_scope)
    if not Path(config.load_from).is_file():
        raise FileNotFoundError(f'Missing initialization checkpoint: {config.load_from}; use --init-checkpoint')
    if np.load(config.model.head.instance_bank.anchor).shape != (300, 11):
        raise ValueError('Temporal ROSA requires the 300x11 training anchor file')
    report = dict(data_root=config.data_root, initialization=config.load_from,
                  slots=config.model.head.instance_bank.num_temp_instances,
                  effective_batch=config.experiment_contract.effective_batch,
                  per_gpu_batch=config.train_dataloader.batch_size,
                  gpus=gpus, epochs=config.train_cfg.max_epochs,
                  gradients_through_history=False)
    token_sets = {}
    for split, loader in (('train', config.train_dataloader), ('val', config.val_dataloader)):
        dataset = DATASETS.build(loader.dataset)
        token_sets[split] = {row['token'] for row in dataset.temporal_records}
        frames = len(dataset)
        original = DefaultSampler(dataset, shuffle=False)
        batching = TemporalSequenceBatchSampler(original, loader.batch_size,
            clip_length=loader.batch_sampler.clip_length,
            shuffle=loader.batch_sampler.shuffle, rank=0, world_size=gpus)
        known = sum(int(np.isfinite(v).all(axis=1).sum()) for v in dataset._temporal_velocities.values())
        unknown = sum(len(v) for v in dataset._temporal_velocities.values()) - known
        labels = Counter()
        in_range_labels = Counter()
        annotation_loader = TRANSFORMS.build(dict(type='LoadAnnotations3D',
            with_bbox=False, with_bbox_3d=True, with_label=False, with_label_3d=True))
        for index in range(frames):
            item = dataset.get_data_info(index)
            labels.update(str(name) for name in item['annos']['gt_names'])
            for camera in item['cam_info']['cam_fisheye'].values():
                if not Path(camera['cam_path']).is_file():
                    raise FileNotFoundError(camera['cam_path'])
            if split == 'train':
                filtered = Fisheye3DODEvaluatorRangeFilter()(annotation_loader(item))
                annotation = filtered['ann_info']
                count = len(filtered['gt_bboxes_3d'])
                if count != len(annotation['instance_inds']) or count != len(filtered['gt_labels_3d']):
                    raise ValueError(f'Filtered track-ID/box mismatch: {item["token"]}')
                in_range_labels.update(dataset.metainfo['classes'][int(label)]
                                       for label in filtered['gt_labels_3d'])
        report[split] = dict(frames=frames,
            scenes=len({row['scene'] for row in dataset.temporal_records}),
            steps_per_rank=len(batching), tail_padding_frames=batching.padding_frames,
            class_counts=dict(labels), known_velocity_targets=known,
            unknown_velocity_targets=unknown)
        if split == 'train':
            report[split]['in_range_training_classes'] = dict(in_range_labels)
    if token_sets['train'] & token_sets['val']:
        raise ValueError('Train/validation token overlap')
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir/'temporal_preflight.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/Custom/combined_improved_detzero_rosa')
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1, help='Independent chronological lanes per GPU')
    parser.add_argument('--effective-batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--master-port', type=int, default=29504)
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--smoke-test', action='store_true', help='Six real chronological AMP iterations')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--cfg-options', nargs='*', default=[])
    args = parser.parse_args()
    if args.gpus < 1 or args.workers < 0:
        parser.error('GPU count must be positive and workers nonnegative')
    os.environ['ROSA_DATA_ROOT'] = str(args.data_root.resolve())
    if args.init_checkpoint:
        os.environ['ROSA_INIT_CKPT'] = str(args.init_checkpoint.resolve())
    config_path = ROSA/'configs/fisheye3dod/rosa_vov99_temporal_12e.py'
    config = Config.fromfile(str(config_path))
    merge_options(config, args.cfg_options)
    plan = batch_plan(args.gpus, args.batch_size, args.effective_batch)
    overrides = runtime_options(config, plan, args.workers)
    merge_options(config, overrides)
    work_dir = (args.work_dir or Path(config.work_dir)).resolve()
    if args.smoke_test and args.work_dir is None:
        work_dir = work_dir.with_name(work_dir.name+'_smoke')
    if not args.preflight_only and not args.resume and (work_dir/'last_checkpoint').exists():
        raise FileExistsError(f'{work_dir} already has checkpoints; use --resume or a new directory')
    preflight(config, work_dir, args.gpus)
    if args.preflight_only:
        return
    if torch.cuda.device_count() < args.gpus:
        raise RuntimeError(f'{args.gpus} GPUs requested, {torch.cuda.device_count()} available')
    options = [*args.cfg_options, *overrides]
    if args.smoke_test:
        options += ['train_cfg._delete_=True', 'train_cfg.type=IterBasedTrainLoop',
            'train_cfg.max_iters=6', 'train_cfg.val_interval=1000',
            'val_cfg=None', 'val_dataloader=None', 'val_evaluator=None',
            'param_scheduler=[]', 'train_dataloader.num_workers=0',
            'train_dataloader.persistent_workers=False',
            'optim_wrapper.accumulative_counts=1', 'default_hooks.logger.interval=1',
            'default_hooks.checkpoint.save_best=None', 'default_hooks.checkpoint.rule=None',
            'default_hooks.checkpoint.interval=10000', 'default_hooks.checkpoint.by_epoch=False']
    command = training_command(args, config_path, work_dir, options)
    if args.resume:
        position = command.index('--cfg-options')
        command[position:position] = ['--resume', str(args.resume.resolve())]
    print('$ '+shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)


if __name__ == '__main__':
    main()
