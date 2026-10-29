#!/usr/bin/env python3
"""Read-only model/data audit and offline suppression ablations on saved results."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import io
import json
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

os.environ.setdefault('MPLCONFIGDIR', '/tmp/rosa-audit-matplotlib')
import cv2
import numpy as np
for name, value in [('float', float), ('int', int), ('bool', bool)]:
    if name not in np.__dict__:
        setattr(np, name, value)
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'projects' / 'Fisheye3DOD'))
from dataset.fisheye3dod_eval import Fisheye3DODEval

CLASSES = ('Car', 'Van', 'Truck', 'Bus', 'Pedestrian', 'Cyclist')
RANGES = dict(zip(CLASSES, (48, 48, 48, 48, 36, 42)))


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda data: torch.load(io.BytesIO(data), map_location='cpu')
        return super().find_class(module, name)


def get(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def read_infos(path):
    with path.open('rb') as stream:
        result = pickle.load(stream)
    return result['data_list'] if isinstance(result, dict) else result


def annotation(token, box, name, score):
    return dict(sample_token=str(token), translation=box[:3].tolist(),
                ego_translation=box[:3].tolist(), size=box[[4, 3, 5]].tolist(),
                rotation=[float(np.cos(box[6] / 2)), 0., 0., float(np.sin(box[6] / 2))],
                velocity=[0., 0.], detection_name=name,
                detection_score=float(score), attribute_name='')


def polygon(box):
    return cv2.boxPoints(((float(box[0]), float(box[1])),
                         (float(box[3]), float(box[4])), float(np.degrees(box[6]))))


def overlap_matrix(boxes):
    count = len(boxes)
    overlaps = np.eye(count)
    polygons = [polygon(box) for box in boxes]
    areas = boxes[:, 3] * boxes[:, 4]
    radii = np.linalg.norm(boxes[:, 3:5], axis=1) / 2
    for i in range(count):
        nearby = np.linalg.norm(boxes[i + 1:, :2] - boxes[i, :2], axis=1)
        for j in np.flatnonzero(nearby < radii[i] + radii[i + 1:]) + i + 1:
            intersection = cv2.intersectConvexConvex(polygons[i], polygons[j])[0]
            overlaps[i, j] = overlaps[j, i] = intersection / max(
                areas[i] + areas[j] - intersection, 1.e-8)
    return overlaps


def keep_indices(frame, threshold=0., nms=None, vehicle_group=False, exact=False):
    keep = []
    for i in np.argsort(-frame['scores'], kind='stable'):
        if frame['scores'][i] < threshold:
            continue
        reject = False
        for j in keep:
            same = frame['labels'][i] == frame['labels'][j]
            if vehicle_group:
                same |= frame['labels'][i] < 4 and frame['labels'][j] < 4
            if exact and np.array_equal(frame['boxes'][i], frame['boxes'][j]):
                reject = True
                break
            if nms is not None and same and frame['iou'][i, j] > nms:
                reject = True
                break
        if not reject:
            keep.append(int(i))
    return keep


def operating_counts(frames, selected):
    total = Counter()
    per_class = defaultdict(Counter)
    for frame, indices in zip(frames, selected):
        used = set()
        for name in frame['gt_names']:
            total['gt'] += 1
            per_class[name]['gt'] += 1
        for i in indices:
            name = CLASSES[frame['labels'][i]]
            counts = per_class[name]
            total['predictions'] += 1
            counts['predictions'] += 1
            distance = np.linalg.norm(frame['gt_boxes'][:, :2] - frame['boxes'][i, :2], axis=1)
            compatible = np.flatnonzero((frame['gt_names'] == name) & (distance < 2.))
            available = [j for j in compatible if j not in used]
            if available:
                j = min(available, key=lambda j: distance[j])
                used.add(j)
                kind = 'tp'
            elif len(compatible):
                kind = 'duplicate_fp'
            elif np.any(distance < 2.):
                kind = 'wrong_class_or_nearby_fp'
            else:
                kind = 'unmatched_fp'
            total[kind] += 1
            counts[kind] += 1
    def finish(c):
        d = dict(c)
        d['precision'] = c['tp'] / max(c['predictions'], 1)
        d['recall'] = c['tp'] / max(c['gt'], 1)
        return d
    return dict(total=finish(total), per_class={k: finish(v) for k, v in per_class.items()})


def dataset_audit(train, val):
    result = {}
    track_sets = {}
    for split, infos in [('train', train), ('val', val)]:
        counts, effective, dn, occurrences, tracks = Counter(), Counter(), Counter(), Counter(), defaultdict(set)
        all_tracks = set()
        empty, empty_after_range = 0, 0
        dimensions = defaultdict(list)
        for sample in infos:
            names = np.asarray(sample['annos']['gt_names'], dtype=str)
            boxes = np.asarray(sample['annos']['gt_boxes']).reshape(-1, 7)
            ids = sample['annos']['gt_ids']
            scene = sample['metainfo'].get('sequence_id', sample['metainfo']['scene_name'])
            counts.update(str(n) for n in names)
            empty += len(names) == 0
            valid = [i for i, (n, b) in enumerate(zip(names, boxes)) if np.linalg.norm(b[:2]) < RANGES[str(n)]]
            effective.update(str(names[i]) for i in valid)
            dn.update(str(names[i]) for i in valid[:4])
            occurrences.update({str(names[i]) for i in valid})
            empty_after_range += len(valid) == 0
            for i in valid:
                key = (scene, str(ids[i]))
                tracks[str(names[i])].add(key)
                all_tracks.add(key)
                dimensions[str(names[i])].append(boxes[i, 3:6])
        result[split] = dict(frames=len(infos), raw_counts=dict(counts),
                            evaluator_range_counts=dict(effective),
                            first4_dn_counts=dict(dn), frame_occurrences=dict(occurrences),
                            unique_tracks={k: len(v) for k, v in tracks.items()},
                            empty_frames=empty, empty_after_range=empty_after_range,
                            dimensions_p05_median_p95={k: np.percentile(v, [5, 50, 95], axis=0).tolist() for k, v in dimensions.items()})
        track_sets[split] = all_tracks
    result['shared_track_ids_train_val'] = len(track_sets['train'] & track_sets['val'])
    return result


def denoising_contract_audit():
    """Execute the current target generator and loss mask without building a model."""
    path = ROOT / 'projects/mmdet3d_plugin/models/detection3d/target.py'
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'SparseBox3DTarget')
    node.decorator_list = []
    scope = dict(torch=torch, np=np, F=torch.nn.functional,
                 X=0, Y=1, Z=2, W=3, L=4, H=5, SIN_YAW=6, COS_YAW=7)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    sampler = scope['SparseBox3DTarget'](num_dn_groups=5, max_dn_gt=4,
        dn_noise_scale=[2., 2., 2., .5, .5, .5, .5, .5, .5, .5])
    boxes = torch.tensor([[10., 3., -.3, 4., 2., 1.5, 0., 0., 0.],
                          [20., 3., -.3, 4., 2., 1.5, 0., 0., 0.]])
    anchors, targets, labels, mask, valid, _ = sampler.get_dn_anchors([torch.tensor([0, 4])], [boxes])
    labels = torch.where(labels == -3, 6, labels)
    path = ROOT / 'projects/mmdet3d_plugin/models/sparse4d_head.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'Sparse4DHead')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_for_dn_loss')
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), dict(reduce_mean=lambda x: x), scope)
    output = scope['prepare_for_dn_loss'](SimpleNamespace(num_classes=6, reg_weights=[1.] * 10),
        dict(dn_valid_mask=valid, dn_cls_target=labels, dn_reg_target=targets))
    _, cls_targets, reg_targets, positives, _, count = output
    return dict(valid_copies=int(valid.sum()), genuine_positive_copies=int((labels < 6).sum()),
                background_copies=int((labels == 6).sum()),
                background_copies_selected_for_regression=int(((cls_targets == 6) & positives).sum()),
                all_zero_regression_targets=int((reg_targets == 0).all(-1).sum()),
                actual_positive_normalizer=float(count),
                group_class_labels=[row.tolist() for row in labels[0].split(4)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = read_infos(args.data_root / 'rosa/fisheye3dod_infos_train.pkl')
    val = read_infos(args.data_root / 'rosa/fisheye3dod_infos_val.pkl')
    gt_map = {str(info['token']): info for info in val}
    with args.predictions.open('rb') as stream:
        raw = CPUUnpickler(stream).load()
    frames, gt_annos = [], {}
    exact_stats = Counter()
    for sample in raw:
        token = str(sample['token'] if isinstance(sample, dict) else sample.metainfo['token'])
        pred = get(sample, 'pred_instances_3d')
        boxes_obj = get(pred, 'bboxes_3d')
        boxes = boxes_obj.tensor.detach().cpu().numpy().copy()
        boxes[:, :3] = boxes_obj.gravity_center.detach().cpu().numpy()
        scores = get(pred, 'scores_3d').detach().cpu().numpy()
        labels = get(pred, 'labels_3d').detach().cpu().numpy()
        valid = np.asarray([np.linalg.norm(b[:2]) < RANGES[CLASSES[c]] for b, c in zip(boxes, labels)])
        boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
        info = gt_map[token]
        gn = np.asarray(info['annos']['gt_names'], dtype=str)
        gb = np.asarray(info['annos']['gt_boxes']).reshape(-1, 7)
        valid_gt = np.asarray([np.linalg.norm(b[:2]) < RANGES[str(n)] for b, n in zip(gb, gn)], dtype=bool)
        gn, gb = gn[valid_gt], gb[valid_gt]
        gt_annos[token] = [annotation(token, b, str(n), -1.) for b, n in zip(gb, gn)]
        frame = dict(token=token, boxes=boxes, scores=scores, labels=labels,
                     gt_boxes=gb, gt_names=gn, iou=overlap_matrix(boxes))
        for threshold in [0., .05, .1, .2, .3]:
            indices = np.flatnonzero(scores >= threshold)
            signatures = [boxes[i].tobytes() for i in indices]
            exact_stats[str(threshold)] += len(signatures) - len(set(signatures))
        frames.append(frame)
    results = dict(dataset=dataset_audit(train, val), denoising_contract=denoising_contract_audit(), exact_repeated_boxes=dict(exact_stats),
                   matching='class-aware greedy BEV center matching at 2 m, evaluator class ranges; FP categories are diagnostic proxies',
                   threshold_sweep={}, ablations={})
    for threshold in [0., .05, .1, .15, .2, .25, .3, .4, .5]:
        indices = [keep_indices(f, threshold) for f in frames]
        results['threshold_sweep'][str(threshold)] = operating_counts(frames, indices)
    variants = [('raw', {}), ('one_class_per_identical_box', dict(exact=True))]
    for threshold in [.25, .5, .7]:
        variants.append((f'class_nms_{threshold}', dict(nms=threshold)))
    variants.extend([
        ('vehicle_group_nms_0.5', dict(nms=.5, vehicle_group=True)),
        ('score_0.2', dict(threshold=.2)),
        ('score_0.3', dict(threshold=.3)),
        ('score_0.1_class_nms_0.5', dict(threshold=.1, nms=.5)),
        ('score_0.2_class_nms_0.5', dict(threshold=.2, nms=.5)),
    ])
    for name, options in variants:
        selected = [keep_indices(frame, **options) for frame in frames]
        predictions = {f['token']: [annotation(f['token'], f['boxes'][i], CLASSES[f['labels'][i]], f['scores'][i]) for i in idx] for f, idx in zip(frames, selected)}
        _, _, metrics = Fisheye3DODEval(predictions, gt_annos).main()
        operating = operating_counts(frames, [[i for i in idx if f['scores'][i] >= .1] for f, idx in zip(frames, selected)])
        results['ablations'][name] = dict(metrics=metrics, at_display_threshold_0_1=operating,
                                          predictions=sum(map(len, selected)))
        print(name, json.dumps({k: round(metrics[k], 5) for k in ['mAP', 'NDS', 'Car_AP', 'Truck_AP', 'Bus_AP', 'Pedestrian_AP']}), flush=True)
    (args.output_dir / 'audit.json').write_text(json.dumps(results, indent=2))
    print('Audit:', args.output_dir / 'audit.json')


if __name__ == '__main__':
    main()
