#!/usr/bin/env python3
"""Validate real-frame ORT/PyTorch references; optional native AMP diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROSA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROSA))
if sys.version_info[:2] == (3, 8) and (ROSA / 'work_dirs/onnx_deps_py38').is_dir():
    sys.path.insert(0, str(ROSA / 'work_dirs/onnx_deps_py38'))

import numpy as np
from deployment.runtime import CAMERAS, COORDINATES, compare_outputs, create_session, load_npz, postprocess


def native_diagnostics(actual, native):
    rows = [(np.asarray(box)[:7], name, float(score)) for box, name, score in
            zip(native['boxes_3d'], native['names'], native['scores']) if score >= .1]
    used, errors = set(), []
    for box, name, score in zip(actual['boxes_3d'], actual['names'], actual['scores']):
        candidates = [(float(np.linalg.norm(np.asarray(box)[:3] - candidate[0][:3])), index)
                      for index, candidate in enumerate(rows) if index not in used and candidate[1] == name]
        if not candidates:
            continue
        distance, index = min(candidates)
        if distance > 2:
            continue
        used.add(index)
        other, _, other_score = rows[index]
        yaw_error = abs((box[6]-other[6]+np.pi)%(2*np.pi)-np.pi)
        errors.append(dict(center_m=distance, dimensions_m=float(np.abs(np.asarray(box)[3:6]-other[3:6]).max()),
                           yaw_rad=float(yaw_error), score=abs(score-other_score)))
    return dict(native_detections=len(rows), onnx_detections=len(actual['scores']), matched=len(errors),
                maximum_errors={key: max(row[key] for row in errors) for key in errors[0]} if errors else {},
                note='Different precision/kernel paths: native fused CUDA AMP vs standard-operator ONNX FP32. Class-aware nearest-center diagnostic, not exact query matching.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--provider', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--indices', type=int, nargs='+', default=[0, 200, 400])
    parser.add_argument('--native-predictions', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.model.with_suffix('.json').read_text())
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if digest != manifest['onnx_sha256']:
        raise ValueError('ONNX file differs from export manifest')
    native = json.loads(args.native_predictions.read_text()) if args.native_predictions else None
    if native and native.get('box_origin') != 'gravity_center':
        raise ValueError('Native diagnostic predictions must explicitly use gravity centers')
    session = create_session(args.model, args.provider)
    result = dict(onnx_sha256=digest, checkpoint_sha256=manifest['checkpoint_sha256'],
        providers=session.get_providers(), coordinates=COORDINATES, box_origin='gravity_center',
        reference_precision='FP32 PyTorch, learned camera RFL intact; portable tensor reader', cases=[])
    for index in args.indices:
        reference = load_npz(args.model.parent / f'example_{index}_reference.npz')
        feed = load_npz(args.model.parent / f'example_{index}_input.npz')
        started = time.perf_counter()
        raw = session.run(None, feed)
        row = dict(index=index, network_seconds=time.perf_counter()-started,
                   pytorch_parity=compare_outputs(raw, reference))
        # Calibrated optical forward is -Z in the internal ROSA OCam basis.
        row['optical_directions_in_CLIDAR'] = {name: (feed['lidar2cam'][0, i, :3, :3].T @
            np.array([0., 0., -1.])).tolist() for i, name in enumerate(CAMERAS)}
        if native:
            expected_token = next(item['token'] for item in manifest['references'] if item['index'] == index)
            if str(native['frames'][index]['token']) != str(expected_token):
                raise ValueError('Native predictions and exported reference tokens differ')
            row['native_amp_diagnostic'] = native_diagnostics(postprocess(*raw), native['frames'][index])
        result['cases'].append(row)
        print(index, row['pytorch_parity'], flush=True)
    result['passed'] = all(row['pytorch_parity']['passed'] for row in result['cases'])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if not result['passed']:
        raise RuntimeError('ONNX/PyTorch numerical validation failed')


if __name__ == '__main__':
    main()
