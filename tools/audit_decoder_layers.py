#!/usr/bin/env python3
"""Capture six decoder stages in one read-only GPU validation pass.

The resulting NPZ allows CPU evaluation of stage choice, confidence ranking,
heading quality, or mixed-field ablations without rerunning the image backbone.
No weights, training settings, or dataset annotations are changed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'projects/ROSA'), str(ROOT / 'projects/Fisheye3DOD')]
os.environ.setdefault('MPLCONFIGDIR', '/tmp/rosa-decoder-audit-mpl')
os.environ.setdefault('FISHEYE3DOD_OFFICIAL_ROOT', str(ROOT / 'projects/Fisheye3DOD'))
import numpy as np
for name, value in [('float', float), ('int', int), ('bool', bool)]:
    if name not in np.__dict__:
        setattr(np, name, value)
import torch
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from infer_fisheye_checkpoints import field
from run import evaluate


@HOOKS.register_module()
class DecoderLayerCaptureHook(Hook):
    def __init__(self, output_dir):
        self.output = Path(output_dir)
        self.frames = []
        self.tokens = []

    def before_test(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        head = model.head
        architecture = dict(
            model=type(model).__name__, head=type(head).__name__,
            backbone=type(model.img_backbone).__name__,
            parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            modules={name: dict(parameters=sum(p.numel() for p in module.parameters()),
                trainable_parameters=sum(p.numel() for p in module.parameters() if p.requires_grad))
                for name, module in model.named_children()},
            num_queries=head.instance_bank.num_anchor,
            temporal_slots=head.instance_bank.num_temp_instances,
            instance_feature_trainable=head.instance_bank.instance_feature.requires_grad,
            anchor_trainable=head.instance_bank.anchor.requires_grad,
            decoder_config=vars(head.fisheye_box_decoder.config),
        )
        (self.output / 'architecture.json').write_text(json.dumps(architecture, indent=2) + '\n')
        self.handle = head.register_forward_hook(self.capture)

    def capture(self, module, inputs, outputs):
        del module, inputs
        self.pending = {name: torch.stack(outputs[name]).detach().float().cpu().numpy()
                        for name in ('classification', 'prediction', 'quality')}

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        del runner, batch_idx, data_batch
        for index, sample in enumerate(outputs):
            metadata = field(sample, 'metainfo', {})
            token = metadata.get('token', field(sample, 'token', None))
            if token is None:
                raise ValueError('Missing sample token in validation result')
            self.tokens.append(str(token))
            self.frames.append({key: value[:, index] for key, value in self.pending.items()})

    def after_test(self, runner):
        del runner
        self.handle.remove()
        if len(self.tokens) != len(set(self.tokens)):
            raise ValueError('Duplicate validation tokens')
        arrays = {key: np.stack([f[key] for f in self.frames]) for key in self.frames[0]}
        np.savez_compressed(self.output / 'decoder_layers.npz', tokens=np.asarray(self.tokens), **arrays)
        print(f'Captured {len(self.frames)} frames: {self.output / "decoder_layers.npz"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'projects/ROSA/configs/fisheye3dod/rosa_vov99_baseline_adapt_8e.py')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    os.environ['ROSA_DATA_ROOT'] = str(args.data_root.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = Config.fromfile(args.config)
    config.work_dir = str(args.output_dir.resolve() / 'work')
    config.launcher = 'none'
    config.test_dataloader.batch_size = 1
    config.test_dataloader.num_workers = args.workers
    config.test_dataloader.dataset.ann_file = str(args.data_root.resolve() / 'rosa/fisheye3dod_infos_val.pkl')
    config.custom_hooks = [dict(type='DecoderLayerCaptureHook', output_dir=str(args.output_dir.resolve()))]
    evaluate(SimpleNamespace(checkpoint=args.checkpoint,
        allow_checkpoint_contract_mismatch=False, dump=None), config)


if __name__ == '__main__':
    main()
