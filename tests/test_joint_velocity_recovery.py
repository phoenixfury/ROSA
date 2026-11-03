"""Joint adaptation must enable geometry/classification while protecting Z/backbone."""
import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn
from test_velocity_recovery import tiny_model

PATH = Path(__file__).resolve().parents[1]/'rosa/integrations/fisheye3dod/joint_velocity_recovery.py'
SPEC = importlib.util.spec_from_file_location('joint_velocity_under_test',PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_joint_optimization_updates_geometry_and_classifier_but_not_z():
    model = tiny_model()
    decoder = model.head.fisheye_box_decoder
    decoder.classifier = nn.Linear(3, 2)
    names, rows = MODULE.configure_joint_decoder(model,reset_gates=True)
    protected = MODULE.protected_joint_state_sha256(model,names,rows)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01,weight_decay=0.)
    for _ in range(3):
        optimizer.zero_grad()
        sum(p.sum() for p in model.parameters() if p.requires_grad).backward()
        optimizer.step()
    assert MODULE.protected_joint_state_sha256(model,names,rows)==protected
    assert not torch.equal(decoder.classifier.weight,before['head.fisheye_box_decoder.classifier.weight'])
    final = decoder.layers[0].refinement.trunk[-1]
    assert not torch.equal(final.weight[0],before['head.fisheye_box_decoder.layers.0.refinement.trunk.0.weight'][0])
    torch.testing.assert_close(final.weight[9],before['head.fisheye_box_decoder.layers.0.refinement.trunk.0.weight'][9],rtol=0,atol=0)
    assert not model.other.weight.requires_grad


def test_reset_changes_only_velocity_gate_initialization():
    model = tiny_model()
    original = {k:v.clone() for k,v in model.state_dict().items()}
    MODULE.configure_joint_decoder(model,reset_gates=True)
    for name,value in model.state_dict().items():
        if '.update_gate.' in name:
            torch.testing.assert_close(value[:3],original[name][:3],rtol=0,atol=0)
            assert not value[3].count_nonzero()
        else:
            torch.testing.assert_close(value,original[name],rtol=0,atol=0)


def test_inherited_control_does_not_change_initialization():
    model = tiny_model()
    original = {k:v.clone() for k,v in model.state_dict().items()}
    MODULE.configure_joint_decoder(model,reset_gates=False)
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,original[name],rtol=0,atol=0)


def test_hash_detects_buffer_or_protected_z_changes():
    model = tiny_model('vector_residual')
    names,rows = MODULE.configure_joint_decoder(model)
    baseline = MODULE.protected_joint_state_sha256(model,names,rows)
    with torch.no_grad():
        model.head.fisheye_box_decoder.layers[0].refinement.trunk[-1].weight[10,0] += 1
    assert MODULE.protected_joint_state_sha256(model,names,rows)!=baseline
