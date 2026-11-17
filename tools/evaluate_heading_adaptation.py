#!/usr/bin/env python3
"""Old-benchmark regression check and fresh-PyTorch held-out angle check."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from train_balanced_fisheye import ROOT, ROSA, np, torch
import mmengine
from mmcv.ops import nms_rotated
from audit_fisheye_false_positives import CLASSES, RANGES, annotation, Fisheye3DODEval
from finetune_heading_rows import angular_metrics, design


def selected(row, boxes):
    scores, labels = row['scores'], row['labels']
    retained = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        b = torch.as_tensor(boxes[indices][:, [0, 1, 3, 4, 6]].copy(), device='cuda', dtype=torch.float32)
        s = torch.as_tensor(scores[indices].copy(), device='cuda', dtype=torch.float32)
        _, keep = nms_rotated(b, s, .5, clockwise=False)
        retained.extend(indices[keep.cpu().numpy()].tolist())
    retained = np.asarray(retained, dtype=int)
    return retained[np.argsort(-scores[retained], kind='stable')]


def cached_predictions(records, delta, thresholds=None, residual_function=None):
    frames = []
    for record in records:
        with np.load(record['path'], allow_pickle=False) as row:
            boxes = row['boxes'].copy()
            indices=np.arange(len(boxes))
            boxes[:, 6] += (design(row,indices) @ delta if residual_function is None
                            else residual_function(row,indices,record))
            boxes[:, 6] = (boxes[:, 6]+np.pi)%(2*np.pi)-np.pi
            keep = selected(row, boxes)
            if thresholds is not None:
                keep = np.asarray([i for i in keep if float(row['scores'][i]) >= thresholds[CLASSES[int(row['labels'][i])]]], dtype=int)
            frames.append(dict(token=record['token'], boxes_3d=boxes[keep].tolist(),
                scores=row['scores'][keep].tolist(), labels=row['labels'][keep].tolist(),
                names=[CLASSES[int(i)] for i in row['labels'][keep]]))
    return dict(frames=frames, box_origin='gravity_center', format_version=2, classes=CLASSES,
                score_threshold=None, class_score_thresholds=thresholds, nms_iou=.5)


def official(payload, infos):
    gt, pred = {}, {}
    for frame, info in zip(payload['frames'], infos):
        token = str(info['token'])
        if frame['token'] != token:
            raise ValueError('Official metric token mismatch')
        gt[token] = [annotation(token, np.asarray(box), str(name), -1.)
                     for box, name in zip(info['annos']['gt_boxes'], info['annos']['gt_names'])
                     if np.linalg.norm(box[:2]) < RANGES[str(name)]]
        pred[token] = [annotation(token, np.asarray(box), str(name), score)
                      for box, name, score in zip(frame['boxes_3d'], frame['names'], frame['scores'])
                      if np.linalg.norm(box[:2]) < RANGES[str(name)]]
    return Fisheye3DODEval(pred, gt).main()[2]


def angle_pairs(payload, targets):
    from scipy.optimize import linear_sum_assignment
    pairs = {}
    if [f['token'] for f in payload['frames']] != [f['token'] for f in targets['frames_data']]:
        raise ValueError('Heading metric token mismatch')
    for frame, gt in zip(payload['frames'], targets['frames_data']):
        if gt['split'] != 'val' or not gt['targets']:
            continue
        vehicle_indices = [i for i,n in enumerate(frame['names']) if n in CLASSES[:4]]
        if not vehicle_indices:
            continue
        boxes = np.asarray(frame['boxes_3d'])[vehicle_indices, :7]
        gb = np.asarray([t['box'] for t in gt['targets']])
        cost = np.linalg.norm(boxes[:, None, :2]-gb[None, :, :2], axis=-1)
        ia, ib = linear_sum_assignment(cost)
        for i,j in zip(ia,ib):
            if cost[i,j] < 1.5:
                pairs[(gt['frame'], gt['targets'][j]['track'])] = dict(
                    prediction=float(boxes[i,6]), target=float(gb[j,6]),
                    moving_ego=gt['ego_speed_mps']>.15 or abs(gt['ego_yaw_rate_rps'])>.01)
    return pairs


def common_angles(before, after):
    keys = sorted(set(before)&set(after))
    output = dict(before_matches=len(before), after_matches=len(after), common_matches=len(keys))
    for mode in ('all', 'ego_stopped', 'ego_moving'):
        selected_keys = [k for k in keys if mode == 'all' or before[k]['moving_ego'] == (mode == 'ego_moving')]
        output[mode] = {name: angular_metrics(
            np.asarray([rows[k]['prediction'] for k in selected_keys]),
            np.asarray([rows[k]['target'] for k in selected_keys]))
            for name, rows in (('before', before), ('after', after))} if selected_keys else {}
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--targets', type=Path, required=True)
    p.add_argument('--variant', action='append', required=True, metavar='NAME=TRAIN_DIR')
    p.add_argument('--native-before', type=Path)
    p.add_argument('--native-after', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.cache/'cache_manifest.json').read_text())
    targets = json.loads(args.targets.read_text())
    thresholds = json.loads((ROSA/'work_dirs/threshold_tuning_e1_20261103/balanced_profile.json').read_text())['score_thresholds']
    old_records = manifest['datasets']['old_val']['records']
    old_infos = mmengine.load(manifest['datasets']['old_val']['selected_info'])['data_list']
    result = dict(source_checkpoint_sha256=manifest['checkpoint_sha256'], models={},
                  matching='Class-aware old official benchmark; vehicle-family center <1.5m for new partial heading targets')
    variants = [('baseline', np.zeros(1025, dtype=np.float32))]
    for value in args.variant:
        name, directory = value.split('=', 1)
        with np.load(Path(directory)/'heading_delta.npz', allow_pickle=False) as file:
            variants.append((name, file['delta'].copy()))
    for name, delta in variants:
        print('Evaluating old benchmark:', name, flush=True)
        old = cached_predictions(old_records, delta)
        with (args.output/f'{name}_official_metrics.log').open('w') as log, redirect_stdout(log):
            metrics = official(old, old_infos)
        result['models'][name] = dict(old_validation_metrics=metrics)
        new = cached_predictions(manifest['datasets']['new']['records'], delta, thresholds)
        (args.output/f'{name}_new_predictions.json').write_text(json.dumps(new, indent=2)+'\n')
        if name == 'baseline':
            baseline_pairs = angle_pairs(new, targets)
        else:
            result['models'][name]['cached_new_heading_comparison'] = common_angles(baseline_pairs, angle_pairs(new, targets))
        print(name, {k:metrics[k] for k in ('mAP', 'NDS', 'mAOE')}, flush=True)
        (args.output/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    if args.native_before and args.native_after:
        before, after = json.loads(args.native_before.read_text()), json.loads(args.native_after.read_text())
        for frame in before['frames']:
            keep = [i for i,(n,s) in enumerate(zip(frame['names'],frame['scores'])) if s >= thresholds[n]]
            for key in ('boxes_3d', 'scores', 'labels', 'names'):
                frame[key] = [frame[key][i] for i in keep]
        result['fresh_native_new_heading_comparison'] = common_angles(angle_pairs(before, targets), angle_pairs(after, targets))
        result['native_matching_note'] = 'Fresh independent native PyTorch inference, same inherited per-class thresholds; common GT-track/frame matches.'
    (args.output/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(args.output/'comparison.json', flush=True)


if __name__ == '__main__':
    main()
