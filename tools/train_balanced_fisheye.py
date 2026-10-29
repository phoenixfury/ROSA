#!/usr/bin/env python3
"""Validate the portable dataset and launch corrected ROSA on one or more GPUs."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

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
from mmdet3d.registry import DATASETS, TASK_UTILS

CLASSES = ('Car', 'Van', 'Truck', 'Bus', 'Pedestrian', 'Cyclist')


def preflight(config, work_dir):
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
    for index in range(len(base)):
        sample = base.get_data_info(index)
        ann = sample['annos']
        names = np.asarray(ann['gt_names'], dtype=str)
        boxes = np.asarray(ann['gt_boxes']).reshape(-1, 7)
        if not np.isfinite(boxes).all() or (boxes[:, 3:6] <= 0).any():
            raise ValueError(f'Invalid GT geometry: {sample["token"]}')
        counts.update(names.tolist())
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
        epochs=config.train_cfg.max_epochs,
        annotation_review_rows=len(review),
        note='Review rows flag candidates; no labels were automatically changed.')
    (work_dir / 'balanced_preflight.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (work_dir / 'rare_class_review.csv').open('w', newline='') as stream:
        fields = ['token', 'sequence', 'track_id', 'label', 'length', 'width', 'height', 'front_image', 'reason']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', choices=['finetune', 'retrain'], default='finetune')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--master-port', type=int, default=29503)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--smoke-test', action='store_true', help='Four real AMP training iterations, no validation')
    parser.add_argument('--resume', type=Path, help='Resume a corrected-recipe checkpoint, not the old epoch-16 optimizer')
    parser.add_argument('--cfg-options', nargs='*', default=[])
    args = parser.parse_args()
    if args.gpus < 1:
        parser.error('--gpus must be positive')
    if args.data_root:
        os.environ['ROSA_DATA_ROOT'] = str(args.data_root.resolve())
    if args.init_checkpoint:
        os.environ['ROSA_INIT_CKPT'] = str(args.init_checkpoint.resolve())
    name = 'rosa_vov99_balanced_finetune_8e.py' if args.recipe == 'finetune' else 'rosa_vov99_balanced_retrain_24e.py'
    config_path = ROSA / 'configs/fisheye3dod' / name
    config = Config.fromfile(config_path)
    if args.cfg_options:
        from mmengine.config import DictAction
        overrides = dict(item.split('=', 1) for item in args.cfg_options)
        config.merge_from_dict({key: DictAction._parse_iterable(value) for key, value in overrides.items()})
    work_dir = (args.work_dir or Path(config.work_dir)).resolve()
    if args.smoke_test and args.work_dir is None:
        work_dir = work_dir.with_name(work_dir.name + '_smoke')
    if not args.preflight_only and not args.resume and (work_dir / 'last_checkpoint').exists():
        raise FileExistsError(f'{work_dir} already has checkpoints; use --resume or a new --work-dir')
    preflight(config, work_dir)
    if args.preflight_only:
        return
    if torch.cuda.device_count() < args.gpus:
        raise RuntimeError(f'{args.gpus} GPUs requested, but {torch.cuda.device_count()} visible; set --gpus appropriately')
    command = [sys.executable]
    if args.gpus > 1:
        command += ['-m', 'torch.distributed.run', '--nnodes=1',
                    f'--nproc_per_node={args.gpus}', f'--master_port={args.master_port}']
        # A fixed rendezvous endpoint gives this independent training job its
        # own port even when another torchrun job is active on the server.
        command += ['--master_addr=127.0.0.1']
    command += [str(ROSA / 'tools/run.py'), 'train', str(config_path), '--amp',
                '--launcher', 'pytorch' if args.gpus > 1 else 'none',
                '--work-dir', str(work_dir)]
    if args.resume:
        command += ['--resume', str(args.resume.resolve())]
    overrides = list(args.cfg_options)
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
    if overrides:
        command += ['--cfg-options', *overrides]
    print('$ ' + shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)


if __name__ == '__main__':
    main()
