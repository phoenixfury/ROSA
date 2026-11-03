"""Row isolation, optimizer invariants and label-independent clip selection."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

PATH = Path(__file__).resolve().parents[1] / 'rosa/integrations/fisheye3dod/velocity_recovery.py'
import sys
SPEC = importlib.util.spec_from_file_location('velocity_recovery_under_test', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def tiny_model(mode='rotation'):
    model = nn.Module()
    model.other = nn.Linear(3, 3)
    model.head = nn.Module()
    model.head.fisheye_box_decoder = nn.Module()
    layer = nn.Module()
    layer.refinement = nn.Module()
    layer.refinement.yaw_update_mode = mode
    layer.refinement.trunk = nn.Sequential(nn.Linear(3, 10 if mode == 'rotation' else 11))
    layer.refinement.update_gate = nn.Linear(3, 4)
    model.head.fisheye_box_decoder.layers = nn.ModuleList([layer])
    model.register_buffer('bn_running', torch.ones(3))
    return model


def test_gate_reset_touches_only_velocity_row():
    model = tiny_model()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    records = MODULE.configure_velocity_recovery(model, reset_gates=True)
    permitted = {record.name for record in records if record.kind == 'velocity_gate'}
    for key, value in model.state_dict().items():
        if key in permitted:
            torch.testing.assert_close(value[:3], before[key][:3], rtol=0, atol=0)
            assert not value[3].count_nonzero()
        else:
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_multiple_adam_steps_preserve_all_inactive_rows_and_z():
    for mode in ('rotation', 'vector_residual'):
        model = tiny_model(mode)
        records = MODULE.configure_velocity_recovery(model, reset_gates=True)
        baseline = MODULE.protected_state_sha256(model, records)
        optimizer = torch.optim.AdamW([row.parameter for row in records], lr=.01, weight_decay=0.)
        for _ in range(4):
            optimizer.zero_grad()
            sum(row.parameter.sum() for row in records).backward()
            for row in records:
                mask = torch.ones_like(row.parameter, dtype=torch.bool)
                mask[list(row.rows)] = False
                assert not row.parameter.grad[mask].count_nonzero()
            optimizer.step()
        assert MODULE.protected_state_sha256(model, records) == baseline
        assert not model.other.weight.requires_grad


def test_control_initialization_does_not_reset_any_state():
    model = tiny_model()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    MODULE.configure_velocity_recovery(model, reset_gates=False)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_clips_are_disjoint_chronological_and_uniform_by_scene():
    records = [dict(index=i, scene=f'train_{i//160}', token=str(i), timestamp_ns=i)
               for i in range(480)]
    chosen = MODULE.choose_chronological_clips(list(reversed(records)), 2, 32)
    assert len(chosen) == len({row['token'] for row in chosen}) == 192
    for offset in range(0, len(chosen), 32):
        clip = chosen[offset:offset+32]
        assert len({row['scene'] for row in clip}) == 1
        assert [row['index'] for row in clip] == list(range(clip[0]['index'], clip[0]['index']+32))


def test_unknown_velocity_never_becomes_zero_target_supervision():
    class Head:
        num_classes = 1
        reg_weights = [1.]*10
        def __init__(self):
            self.sampler = SimpleNamespace(sample=self.sample)
        def sample(self, classification, prediction, labels, boxes):
            target = torch.zeros_like(prediction)
            target[0, 0, 8:] = torch.tensor([3., 4.])
            target[0, 1, 8:] = float('nan')
            return torch.zeros(1, 2, dtype=torch.long), target, torch.ones_like(prediction)
        def loss_reg(self, selected, truth, weight, avg_factor, **kwargs):
            assert len(selected) == 1 and torch.isfinite(truth).all()
            return {'loss_box_recovery': ((selected-truth).abs()*weight).sum()/avg_factor}
    prediction = torch.zeros(1, 2, 10, requires_grad=True)
    loss, diagnostic = MODULE.velocity_only_loss(Head(),
        dict(classification=[torch.zeros(1, 2, 1)], prediction=[prediction]),
        dict(gt_labels_3d=[], gt_bboxes_3d=[]))
    loss.backward()
    assert float(loss) == 7.
    assert diagnostic['valid_mask'].tolist() == [[True, False]]
    assert not prediction.grad[0, 1].count_nonzero()
    assert not prediction.grad[..., :8].count_nonzero()


def test_all_unknown_targets_give_connected_zero_loss():
    prediction = torch.zeros(1, 2, 10, requires_grad=True)
    sampler = SimpleNamespace(sample=lambda *args: (
        torch.zeros(1, 2, dtype=torch.long), torch.full_like(prediction, float('nan')),
        torch.ones_like(prediction)))
    loss, diagnostic = MODULE.velocity_only_loss(SimpleNamespace(num_classes=1, sampler=sampler),
        dict(classification=[torch.zeros(1, 2, 1)], prediction=[prediction]),
        dict(gt_labels_3d=[], gt_bboxes_3d=[]))
    loss.backward()
    assert float(loss) == 0.
    assert not prediction.grad.count_nonzero()
    assert len(diagnostic['prediction']) == 0
