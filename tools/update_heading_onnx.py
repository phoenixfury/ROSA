#!/usr/bin/env python3
"""Memory-bounded ONNX weight update for E008's three existing yaw tensors.

Reject every non-yaw checkpoint change. Locate each initializer by EXACT parent
parameter value (or its transpose), require a unique match, and preserve all
nodes/operators. This is not a general substitute for architecture re-export.
Fresh independent PyTorch/ORT validation remains required after this update.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROSA = Path(__file__).resolve().parents[1]
if sys.version_info[:2] == (3, 8):
    sys.path.insert(0, str(ROSA/'work_dirs/onnx_deps_py38'))
import numpy as np
import onnx
from onnx import numpy_helper
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def find_initializer(graph, value):
    matches = []
    for index, tensor in enumerate(graph.graph.initializer):
        if tuple(tensor.dims) not in (value.shape, value.T.shape):
            continue
        array = numpy_helper.to_array(tensor)
        if array.shape == value.shape and np.array_equal(array, value):
            matches.append((index, False))
        elif array.shape == value.T.shape and np.array_equal(array, value.T):
            matches.append((index, True))
    if len(matches) != 1:
        raise ValueError(f'Expected one exact initializer match, found {len(matches)}')
    return matches[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-onnx', type=Path, required=True)
    p.add_argument('--parent-checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('Select a new output; existing deployment artifacts are protected')
    manifest = json.loads(args.base_onnx.with_suffix('.json').read_text())
    if sha(args.base_onnx) != manifest['onnx_sha256'] or sha(args.parent_checkpoint) != manifest['checkpoint_sha256']:
        raise ValueError('Base ONNX/parent checkpoint provenance mismatch')
    parent = torch.load(args.parent_checkpoint, map_location='cpu', weights_only=False)['state_dict']
    updated = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['state_dict']
    prefix = 'head.fisheye_box_decoder.layers.5.refinement.'
    allowed = {prefix+'trunk.3.weight':6, prefix+'trunk.3.bias':6, prefix+'face_refinement.weight':3}
    if set(parent) != set(updated):
        raise ValueError('Checkpoint key layout changed; full export is required')
    changed = [key for key in parent if not torch.equal(parent[key], updated[key])]
    if set(changed) != set(allowed):
        raise ValueError('Checkpoint changes are not exactly the three supported heading tensors')
    for key, row in allowed.items():
        mask = torch.ones(parent[key].shape[0], dtype=torch.bool)
        mask[row] = False
        if not torch.equal(parent[key][mask], updated[key][mask]):
            raise ValueError('Non-yaw output rows changed; full export is required')
    graph = onnx.load(str(args.base_onnx))
    replacements = []
    for key in allowed:
        index, transpose = find_initializer(graph, parent[key].numpy())
        old = graph.graph.initializer[index]
        value = updated[key].numpy()
        value = value.T.copy() if transpose else value.copy()
        replacement = numpy_helper.from_array(value, name=old.name)
        old.CopyFrom(replacement)
        replacements.append(dict(parameter=key, initializer=old.name, transpose=transpose))
    onnx.checker.check_model(graph)
    if any(node.domain not in ('', 'ai.onnx') for node in graph.graph.node):
        raise ValueError('Unexpected custom operator domain')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(graph, str(args.output))
    manifest.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
        onnx_sha256=sha(args.output), references=[], checked=True,
        reference_note='Parent FP32 references do not apply after adaptation. See fresh native AMP/ORT parity report.',
        weight_update=dict(base_onnx=str(args.base_onnx.resolve()),
            base_onnx_sha256=sha(args.base_onnx), parent_checkpoint_sha256=sha(args.parent_checkpoint),
            exactly_matched_initializers=replacements, node_graph_unchanged=True,
            source_sha256=sha(Path(__file__))))
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Updated/ONNX-checked:', args.output, json.dumps(replacements), flush=True)


if __name__ == '__main__':
    main()
