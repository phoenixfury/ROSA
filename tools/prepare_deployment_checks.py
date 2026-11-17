#!/usr/bin/env python3
"""Prepare checkpoint-bound, untuned operating checks from an ONNX bundle."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from benchmark_fisheye_backends import sha
from deployment.runtime import CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sequence', action='append', default=[])
    parser.add_argument('--score-threshold', type=float, default=.1)
    args = parser.parse_args()
    if not 0 <= args.score_threshold <= 1:
        parser.error('Score threshold must be in [0,1]')
    if args.output.exists():
        raise FileExistsError(args.output)
    metadata = json.loads(args.model.with_suffix('.json').read_text())
    if sha(args.model) != metadata['onnx_sha256']:
        raise ValueError('ONNX checksum mismatch')
    source = json.loads(args.samples.read_text())
    root = Path(source['data_root'])
    groups = defaultdict(list)
    selected = args.sequence or ['16_09', '17_09', '30_09']
    for original in source['samples']:
        sequence = str(original['sequence'])
        if sequence not in selected:
            continue
        row = dict(original)
        row['image_paths'] = [str((root/p).resolve()) for p in row['image_relative_paths']]
        row['calibration'] = str((args.samples.parent/row['calibration']).resolve())
        groups[sequence].append(row)
    if set(groups) != set(selected):
        raise ValueError('Missing requested sequences')
    args.output.mkdir(parents=True)
    for name in selected:
        (args.output/(name+'.json')).write_text(json.dumps(dict(
            samples=groups[name], data_root=str(root), source_manifest_sha256=sha(args.samples)), indent=2)+'\n')
    profile = dict(format_version=1, checkpoint_sha256=metadata['checkpoint_sha256'],
        nms_iou=.5, score_thresholds={name:args.score_threshold for name in CLASSES},
        note='Untuned uniform operating-point profile for numerical/speed comparisons. Not inherited E007 thresholds; not a precision/recall optimization.')
    (args.output/'uniform_profile.json').write_text(json.dumps(profile, indent=2)+'\n')
    print(json.dumps(dict(sequences={name:len(groups[name]) for name in selected},
        checkpoint_sha256=metadata['checkpoint_sha256'], threshold=profile), indent=2))


if __name__ == '__main__':
    main()
