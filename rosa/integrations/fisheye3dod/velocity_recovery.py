"""Opt-in, row-isolated velocity recovery helpers; no registration/default changes."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
from torch import nn


@dataclass
class RecoveryParameter:
    name: str
    parameter: nn.Parameter
    rows: tuple
    kind: str


def configure_velocity_recovery(model, reset_gates=False):
    """Freeze everything except XY residual/gate rows; never train velocity Z.

    Optimizers must use zero weight decay and fresh state: gradient masking
    alone does not protect inactive rows from decay or restored momentum.
    """
    refinements = [layer.refinement for layer in model.head.fisheye_box_decoder.layers]
    # Validate the complete layout before mutating any parameter.
    for refinement in refinements:
        assert refinement.yaw_update_mode in ('rotation', 'vector_residual')
        projection, gate = refinement.trunk[-1], refinement.update_gate
        assert isinstance(projection, nn.Linear) and isinstance(gate, nn.Linear)
        assert gate.out_features == 4 and gate.bias is not None
        expected = 10 if refinement.yaw_update_mode == 'rotation' else 11
        assert projection.out_features == expected and projection.bias is not None
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    model.requires_grad_(False)
    records = []
    for refinement in refinements:
        first = 7 if refinement.yaw_update_mode == 'rotation' else 8
        for module, rows, kind in ((refinement.trunk[-1], (first, first + 1), 'xy_residual'),
                                   (refinement.update_gate, (3,), 'velocity_gate')):
            for parameter in (module.weight, module.bias):
                if reset_gates and kind == 'velocity_gate':
                    with torch.no_grad():
                        parameter[list(rows)] = 0.
                parameter.requires_grad_(True)
                mask = torch.zeros_like(parameter)
                mask[list(rows)] = 1.
                parameter.register_hook(lambda gradient, mask=mask: gradient * mask)
                records.append(RecoveryParameter(names[id(parameter)], parameter, rows, kind))
    return records


def protected_state_sha256(model, records):
    """Hash every persistent tensor, masking ONLY permitted update rows."""
    allowed = {record.name: record.rows for record in records}
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        if name in allowed:
            value = value.clone()
            value[list(allowed[name])] = 0.
        digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def velocity_only_loss(head, outputs, metas):
    """Existing L1 regression/assignment, free queries, finite causal XY only.

    No denoising queries (the pilot runs the model in eval mode), no new target
    smoothing, no future labels, no artificial velocity bounds or class weights.
    Equal auxiliary stage weighting avoids changing scale with decoder depth.
    """
    losses, final = [], None
    for classification, prediction in zip(outputs['classification'], outputs['prediction']):
        prediction = prediction[..., :10].float()
        labels, target, weights = head.sampler.sample(classification.float(), prediction,
            metas['gt_labels_3d'], metas['gt_bboxes_3d'])
        positive = (labels >= 0) & (labels < head.num_classes)
        valid = positive & torch.isfinite(target[..., 8:10]).all(-1)
        valid &= (weights[..., 8:10] > 0).all(-1)
        selected, truth = prediction[valid], torch.nan_to_num(target[valid])
        if valid.any():
            weight = weights[valid] * prediction.new_tensor(head.reg_weights)
            weight[:, :8] = 0.
            loss = head.loss_reg(selected, truth, weight=weight,
                avg_factor=int(valid.sum()), suffix='_recovery', quality=None)['loss_box_recovery']
        else:
            loss = prediction[..., 8:10].sum() * 0.
        losses.append(loss)
        final = dict(valid_mask=valid, prediction=selected[:, 8:10].detach(),
                     target=truth[:, 8:10].detach())
    return torch.stack(losses).mean(), final


def choose_chronological_clips(records, clips_per_scene=8, clip_length=32):
    """Uniform non-overlapping train-only clips; no labels/validation selection."""
    from collections import defaultdict
    import numpy as np
    if clips_per_scene < 1 or clip_length < 1:
        raise ValueError('Positive clip count/length required')
    scenes = defaultdict(list)
    for record in records:
        scenes[record['scene']].append(record)
    chosen = []
    for scene in sorted(scenes):
        ordered = sorted(scenes[scene], key=lambda row: row['timestamp_ns'])
        clips = [ordered[start:start + clip_length]
                 for start in range(0, len(ordered) - clip_length + 1, clip_length)]
        if not clips:
            raise ValueError(f'Scene {scene} has no full {clip_length}-frame clip')
        for index in np.linspace(0, len(clips) - 1, min(clips_per_scene, len(clips)), dtype=int):
            chosen.extend(clips[index])
    assert len({row['token'] for row in chosen}) == len(chosen)
    return chosen
