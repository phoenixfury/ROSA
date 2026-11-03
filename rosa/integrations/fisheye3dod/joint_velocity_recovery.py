"""Opt-in joint decoder adaptation with a paired velocity-gate initialization."""
from __future__ import annotations

import hashlib
import torch
from torch import nn


def configure_joint_decoder(model, reset_gates=False):
    """Train the existing decoder, never backbone/FPN/priors or Z output rows.

    Call after moving the model to its final device. Use fresh optimizer state
    and zero weight decay so masked Z rows cannot change through AdamW decay.
    Running in eval mode freezes BN/dropout and excludes GT-seeded DN queries;
    autograd is still enabled and normal free-query detection losses are used.
    """
    decoder = model.head.fisheye_box_decoder
    for layer in decoder.layers:
        refinement = layer.refinement
        assert refinement.yaw_update_mode in ('rotation', 'vector_residual')
        assert isinstance(refinement.trunk[-1], nn.Linear)
        assert isinstance(refinement.update_gate, nn.Linear)
        assert refinement.update_gate.out_features == 4
    model.requires_grad_(False)
    decoder.requires_grad_(True)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    protected_rows = {}
    for layer in decoder.layers:
        refinement = layer.refinement
        if reset_gates:
            with torch.no_grad():
                refinement.update_gate.weight[3] = 0.
                refinement.update_gate.bias[3] = 0.
        projection = refinement.trunk[-1]
        z_row = 9 if refinement.yaw_update_mode == 'rotation' else 10
        assert projection.out_features == z_row + 1
        for parameter in (projection.weight, projection.bias):
            mask = torch.ones_like(parameter)
            mask[z_row] = 0.
            parameter.register_hook(lambda gradient, mask=mask: gradient * mask)
            protected_rows[names[id(parameter)]] = [z_row]
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable and all(name.startswith('head.fisheye_box_decoder.') for name in trainable)
    return trainable, protected_rows


def protected_joint_state_sha256(model, trainable, protected_rows):
    """Hash all frozen tensors/buffers and the frozen Z rows inside trainable tensors."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name in trainable and name not in protected_rows:
            continue
        value = tensor.detach().cpu().contiguous()
        if name in protected_rows:
            value = value[protected_rows[name]].contiguous()
        digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
