#!/usr/bin/env python3
"""CPU-only decoder-stage and yaw-quality ablations on captured logits/boxes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import json
from pathlib import Path

from audit_fisheye_false_positives import CLASSES, np, read_infos, annotation, Fisheye3DODEval, torch
import projects.mmdet3d_plugin
from projects.mmdet3d_plugin.models.detection3d.decoder import SparseBox3DDecoder


def proposal_coverage(arrays, lookup):
    result = {}
    predictions = arrays['prediction']
    classifications = arrays['classification']
    qualities = arrays['quality']
    tokens = arrays['tokens'].tolist()
    for stage in range(6):
        counts = defaultdict(Counter)
        for index, token in enumerate(tokens):
            xyz = predictions[index, stage, :, :2]
            cls = torch.from_numpy(classifications[index, stage]).float().sigmoid().numpy()
            quality = torch.from_numpy(qualities[index, stage, :, 0]).float().sigmoid().numpy()
            labels = cls.argmax(-1)
            scores = cls.max(-1) * quality
            annos = lookup[token]['annos']
            for box, name in zip(annos['gt_boxes'], annos['gt_names']):
                label = CLASSES.index(str(name))
                if np.linalg.norm(box[:2]) >= (48,48,48,48,36,42)[label]:
                    continue
                close = np.linalg.norm(xyz - np.asarray(box[:2]), axis=-1) < 2.
                c = counts[str(name)]
                c['gt'] += 1
                c['any_query_within_2m'] += int(close.any())
                c['correct_top_class_within_2m'] += int((close & (labels == label)).any())
                c['correct_top_class_score_0_1_within_2m'] += int((close & (labels == label) & (scores >= .1)).any())
        result[str(stage)] = {k: dict(v) for k,v in counts.items()}
    return dict(stages=result, note='Pre-NMS, non-one-to-one GT neighborhood coverage across all 300 free queries; not detection recall/AP')


def decode(arrays, stage, mode):
    cls = torch.from_numpy(arrays['classification'][:, stage]).float()
    box = torch.from_numpy(arrays['prediction'][:, stage]).float().clone()
    quality = torch.from_numpy(arrays['quality'][:, stage]).float()
    # V49 classifiers/quality heads run under CUDA autocast. Preserve their
    # half-precision sigmoid products for the standard stage comparison.
    cls, quality = cls.half(), quality.half()
    if mode == 'yaw_flip':
        reverse = quality[..., 1].sigmoid() < .5
        box[..., 6:8] *= torch.where(reverse[..., None], -1., 1.)
    if mode == 'yaw_rank':
        centerness = quality[..., 0].sigmoid() * quality[..., 1].sigmoid()
        quality[..., 0] = torch.logit(centerness.float().clamp(1e-6, 1. - 1e-6))
    decoder = SparseBox3DDecoder(single_label=True, quality_before_topk=True,
                                 nms_iou_threshold=.5, nms_class_groups=())
    return decoder.decode([cls], [box], quality=None if mode == 'class_only' else [quality])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = np.load(args.probe)
    infos = read_infos(args.data_root / 'rosa/fisheye3dod_infos_val.pkl')
    tokens = arrays['tokens'].tolist()
    lookup = {str(info['token']): info for info in infos}
    if len(tokens) != len(lookup) or set(tokens) != set(lookup):
        raise ValueError('Probe and validation tokens differ')
    (args.output_dir / 'proposal_coverage.json').write_text(
        json.dumps(proposal_coverage(arrays, lookup), indent=2) + '\n')
    gt = {}
    for token in tokens:
        annos = lookup[token]['annos']
        gt[token] = [annotation(token, np.asarray(box), str(name), -1.)
                     for box, name in zip(annos['gt_boxes'], annos['gt_names'])]
    result = {'probe': str(args.probe),
              'note': 'Read-only CPU decoding; tiny numeric/tie differences from CUDA NMS are possible',
              'variants': {}}
    variants = [(f'stage_{stage}', stage, 'standard') for stage in range(6)]
    variants += [(mode, 5, mode) for mode in ('class_only', 'yaw_rank', 'yaw_flip')]
    for name, stage, mode in variants:
        decoded = decode(arrays, stage, mode)
        predictions = {}
        for token, output in zip(tokens, decoded):
            boxes = output['boxes_3d'].numpy()
            scores = output['scores_3d'].float().numpy()
            labels = output['labels_3d'].numpy()
            predictions[token] = [annotation(token, box, CLASSES[int(label)], float(score))
                                  for box, score, label in zip(boxes, scores, labels)]
        with (args.output_dir / f'{name}.log').open('w') as stream, redirect_stdout(stream):
            metrics = Fisheye3DODEval(predictions, gt).main()[2]
        result['variants'][name] = metrics
        (args.output_dir / 'layer_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        print(name, {k: round(metrics[k], 5) for k in ('mAP', 'NDS', 'mAOE', 'Car_AP', 'Truck_AP', 'Pedestrian_AP')}, flush=True)


if __name__ == '__main__':
    main()
