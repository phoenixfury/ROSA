#!/usr/bin/env python3
"""Compare saved ROSA predictions on identical validation tokens (CPU only).

Example: --model baseline=path/to/evaluation --model retrained=another/path
Each directory must contain predictions.pkl. Official metrics are recomputed
without a display threshold; diagnostic P/R uses class-aware 2 m BEV matching.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path

from audit_fisheye_false_positives import (
    CLASSES, RANGES, Fisheye3DODEval, annotation, dataset_audit,
    get, keep_indices, np, operating_counts, overlap_matrix, read_infos,
)
from infer_fisheye_checkpoints import load_cpu_results


def load_frames(directory, infos, with_iou=True):
    raw = load_cpu_results(directory / 'predictions.pkl')
    by_token = {str(get(row, 'token')): row for row in raw}
    tokens = [str(info['token']) for info in infos]
    if len(by_token) != len(raw) or set(by_token) != set(tokens):
        raise ValueError(f'{directory}: predictions and validation tokens differ')
    frames, gt_annos = [], {}
    for token, info in zip(tokens, infos):
        pred = get(by_token[token], 'pred_instances_3d')
        obj = get(pred, 'bboxes_3d')
        boxes = obj.tensor.detach().cpu().numpy().copy()
        boxes[:, :3] = obj.gravity_center.detach().cpu().numpy()
        scores = get(pred, 'scores_3d').detach().cpu().numpy()
        labels = get(pred, 'labels_3d').detach().cpu().numpy()
        valid = np.asarray([np.linalg.norm(b[:2]) < RANGES[CLASSES[c]]
                            for b, c in zip(boxes, labels)], dtype=bool)
        boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
        gn = np.asarray(info['annos']['gt_names'], dtype=str)
        gb = np.asarray(info['annos']['gt_boxes']).reshape(-1, 7)
        mask = np.asarray([np.linalg.norm(b[:2]) < RANGES[str(n)]
                           for b, n in zip(gb, gn)], dtype=bool)
        gn, gb = gn[mask], gb[mask]
        frames.append(dict(token=token, boxes=boxes, scores=scores,
                           labels=labels, gt_boxes=gb, gt_names=gn,
                           iou=overlap_matrix(boxes) if with_iou else None))
        gt_annos[token] = [annotation(token, b, str(n), -1.)
                           for b, n in zip(gb, gn)]
    return frames, gt_annos


def official_metrics(frames, selected, gt_annos):
    predictions = {f['token']: [annotation(f['token'], f['boxes'][i],
                    CLASSES[f['labels'][i]], f['scores'][i]) for i in indices]
                   for f, indices in zip(frames, selected)}
    return Fisheye3DODEval(predictions, gt_annos).main()[2]


def select(frame, threshold=0., **options):
    if not options:
        order = np.argsort(-frame['scores'], kind='stable')
        return order[frame['scores'][order] >= threshold].tolist()
    return keep_indices(frame, threshold=threshold, **options)


def matched_recall_points(frames):
    rows = []
    gt_count = sum(len(f['gt_names']) for f in frames)
    for f in frames:
        used = set()
        for i in select(f):
            distances = np.linalg.norm(f['gt_boxes'][:, :2] - f['boxes'][i, :2], axis=1)
            candidates = [j for j in np.flatnonzero(
                (f['gt_names'] == CLASSES[f['labels'][i]]) & (distances < 2.))
                if j not in used]
            if candidates:
                used.add(min(candidates, key=lambda j: distances[j]))
            rows.append((float(f['scores'][i]), bool(candidates)))
    scores, positives = np.asarray(sorted(rows, reverse=True)).T
    cumulative = positives.cumsum()
    # Only complete equal-score groups represent realizable score thresholds.
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    result = {}
    for target in (.3, .4, .5, .6):
        eligible = ends[cumulative[ends] / max(gt_count, 1) >= target]
        if len(eligible):
            i = eligible[0]
            result[str(target)] = dict(threshold=float(scores[i]),
                precision=float(cumulative[i] / (i + 1)),
                recall=float(cumulative[i] / max(gt_count, 1)))
        else:
            result[str(target)] = None
    return result


def gt_neighborhoods(frames):
    result = defaultdict(Counter)
    for f in frames:
        indices = np.flatnonzero(f['scores'] >= .1)
        for box, name in zip(f['gt_boxes'], f['gt_names']):
            distances = np.linalg.norm(f['boxes'][indices, :2] - box[:2], axis=1)
            if len(distances) and distances.min() < 2.:
                nearest = indices[distances.argmin()]
                result[str(name)][CLASSES[f['labels'][nearest]]] += 1
            else:
                result[str(name)]['no_prediction_within_2m'] += 1
    return {k: dict(v) for k, v in result.items()}


def audit(directory, infos, log):
    frames, gt_annos = load_frames(directory, infos)
    result = dict(directory=str(directory.resolve()), samples=len(frames),
                  matched_recall=matched_recall_points(frames),
                  gt_neighborhood_score_0_1=gt_neighborhoods(frames),
                  threshold_sweep={}, ablations={}, exact_duplicate_boxes={})
    for threshold in (0., .05, .1, .15, .2, .25, .3, .4, .5):
        selected = [select(f, threshold=threshold) for f in frames]
        counts = operating_counts(frames, selected)
        total = counts['total']
        total['f1'] = 2 * total['precision'] * total['recall'] / max(
            total['precision'] + total['recall'], 1.e-12)
        result['threshold_sweep'][str(threshold)] = counts
        result['exact_duplicate_boxes'][str(threshold)] = sum(
            len(indices) - len({f['boxes'][i].tobytes() for i in indices})
            for f, indices in zip(frames, selected))
    variants = [('raw', {}), ('class_nms_0.5', dict(nms=.5)),
                ('vehicle_group_nms_0.5', dict(nms=.5, vehicle_group=True))]
    with redirect_stdout(log):
        for name, options in variants:
            selected = [select(f, **options) for f in frames]
            metrics = official_metrics(frames, selected, gt_annos)
            display = [[i for i in indices if f['scores'][i] >= .1]
                       for f, indices in zip(frames, selected)]
            result['ablations'][name] = dict(
                metrics=metrics, at_score_0_1=operating_counts(frames, display))
    result['metrics'] = result['ablations']['raw']['metrics']
    result['best_swept_f1_threshold'] = max(result['threshold_sweep'],
        key=lambda t: result['threshold_sweep'][t]['total']['f1'])
    return result


def training_history(run):
    paths = sorted(run.glob('*/vis_data/scalars.json'))
    if (run / 'vis_data/scalars.json').is_file():
        paths.append(run / 'vis_data/scalars.json')
    if len(paths) != 1:
        raise ValueError(f'Expected one training scalar file below {run}')
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    validation = [dict(epoch=r['step'], **{
        k.split('/', 1)[1]: v for k, v in r.items()
        if k.startswith('fisheye_bev/')}) for r in rows if 'fisheye_bev/mAP' in r]
    training = {}
    for epoch in sorted({r['epoch'] for r in rows if 'loss' in r}):
        selected = [r for r in rows if r.get('epoch') == epoch and 'loss' in r]
        training[epoch] = {k: float(np.mean([r[k] for r in selected if k in r]))
                           for k in ('loss', 'loss_cls_5', 'loss_box_5',
                                     'loss_cls_dn_5', 'loss_box_dn_5', 'time',
                                     'grad_norm', 'rosa_grad_norm')}
    return dict(validation=validation, training_means=training,
                best_mAP_epoch=max(validation, key=lambda r:r['mAP'])['epoch'],
                best_NDS_epoch=max(validation, key=lambda r:r['NDS'])['epoch'])


def plots(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for key, label in [('baseline_history', 'Original baseline'),
                       ('training_history', 'Balanced retrain')]:
        if key not in result:
            continue
        rows = result[key]['validation']
        for axis, metric in zip(axes[0], ('mAP', 'NDS')):
            axis.plot([r['epoch'] for r in rows], [r[metric] * 100 for r in rows],
                      marker='o', markersize=3, label=label)
            axis.set(xlabel='Epoch', ylabel=f'{metric} (%)')
            axis.grid(alpha=.25)
            axis.legend()
    for name, record in result['models'].items():
        rows = list(record['threshold_sweep'].values())
        axes[1, 0].plot([r['total']['recall'] for r in rows],
                        [r['total']['precision'] for r in rows], marker='.', label=name)
    axes[1, 0].set(xlabel='Recall', ylabel='Precision',
                   title='2 m diagnostic matching; score sweep (no NMS)')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=.25)
    rows = result.get('training_history', {}).get('validation', [])
    for name in CLASSES[1:]:
        axes[1, 1].plot([r['epoch'] for r in rows], [r[f'{name}_AP'] * 100 for r in rows],
                       label=name)
    axes[1, 1].set(xlabel='Epoch', ylabel='AP (%)', title='Retrained rare-class validation AP')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=.25)
    figure.tight_layout()
    figure.savefig(output / 'learning_and_precision_recall.png', dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--model', action='append', required=True, metavar='NAME=DIR')
    parser.add_argument('--training-run', type=Path)
    parser.add_argument('--baseline-run', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    info_path = args.data_root / 'rosa/fisheye3dod_infos_val.pkl'
    infos = read_infos(info_path)
    destination = args.output_dir / 'comparison.json'
    result = json.loads(destination.read_text()) if destination.exists() else {}
    digest = hashlib.sha256(info_path.read_bytes()).hexdigest()
    if result and result.get('validation_sha256') != digest:
        raise ValueError('Existing comparison uses different validation data')
    result.update(validation_sha256=digest, validation_samples=len(infos),
                  matching='Diagnostic class-aware greedy BEV center <2 m; '
                  'unmatched predictions can include localization errors or missing labels',
                  dataset=dataset_audit(read_infos(args.data_root /
                      'rosa/fisheye3dod_infos_train.pkl'), infos))
    if args.training_run:
        result['training_history'] = training_history(args.training_run)
    if args.baseline_run:
        result['baseline_history'] = training_history(args.baseline_run)
    models = result.setdefault('models', {})
    for value in args.model:
        name, path = value.split('=', 1)
        directory = Path(path).resolve()
        if name in models and models[name]['directory'] == str(directory):
            if 'matched_recall' not in models[name] or 'gt_neighborhood_score_0_1' not in models[name]:
                frames, _ = load_frames(directory, infos, with_iou=False)
                models[name]['matched_recall'] = matched_recall_points(frames)
                models[name]['gt_neighborhood_score_0_1'] = gt_neighborhoods(frames)
                destination.write_text(json.dumps(result, indent=2) + '\n')
            print(f'Reusing {name}', flush=True)
            continue
        print(f'Analyzing {name}', flush=True)
        with (args.output_dir / f'{name}_metric_details.log').open('w') as log:
            models[name] = audit(directory, infos, log)
        destination.write_text(json.dumps(result, indent=2) + '\n')
        print(name, {k: round(models[name]['metrics'][k], 5)
                     for k in ('mAP', 'NDS')}, flush=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    plots(result, args.output_dir)
    print(destination)


if __name__ == '__main__':
    main()
