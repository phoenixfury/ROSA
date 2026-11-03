#!/usr/bin/env python3
"""Exact score sweep of saved, already-NMSed ROSA validation predictions.

No inference/retraining or decoder changes. Matching is class-aware greedy BEV
center distance, NOT official AP or manual truth. Equal-score groups stay intact.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CLASSES = ('Car', 'Van', 'Truck', 'Bus', 'Pedestrian', 'Cyclist')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def match_events(frames, radius=2.):
    """Columns: score, class, kind (TP/duplicate/wrong-class/unmatched), frame.

    A threshold keeps a score-ordered prefix within each class, so TP assignments
    do not change when it increases. Match once, then sweep exact score groups.
    """
    rows = []
    gt = np.zeros((len(frames), len(CLASSES)), dtype=np.int64)
    for index, frame in enumerate(frames):
        used = set()
        for label, name in enumerate(CLASSES):
            gt[index, label] = np.sum(frame['gt_names'] == name)
        for i in np.argsort(-frame['scores'], kind='stable'):
            label = int(frame['labels'][i])
            distance = np.linalg.norm(frame['gt_boxes'][:, :2] - frame['boxes'][i, :2], axis=1)
            compatible = np.flatnonzero((frame['gt_names'] == CLASSES[label]) & (distance < radius))
            available = [j for j in compatible if j not in used]
            if available:
                used.add(min(available, key=lambda j: distance[j]))
                kind = 0
            else:
                kind = 1 if len(compatible) else (2 if np.any(distance < radius) else 3)
            rows.append((float(frame['scores'][i]), label, kind, index))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4), gt


def counts(events, gt_count):
    n = len(events)
    tp = int(np.sum(events[:, 2] == 0))
    fp = n - tp
    return dict(gt=int(gt_count), predictions=n, tp=tp, fp=fp, fn=int(gt_count)-tp,
        precision=tp/max(n, 1), recall=tp/max(int(gt_count), 1),
        f1=2*tp/max(n+int(gt_count), 1), f2=5*tp/max(n+4*int(gt_count), 1),
        duplicate_fp=int(np.sum(events[:, 2] == 1)),
        wrong_class_or_nearby_fp=int(np.sum(events[:, 2] == 2)),
        unmatched_fp=int(np.sum(events[:, 2] == 3)))


def evaluate(events, gt, thresholds):
    limits = np.asarray([thresholds[name] for name in CLASSES])
    selected = events[events[:, 0] >= limits[events[:, 1].astype(int)]]
    return dict(total=counts(selected, gt.sum()), per_class={
        name: counts(selected[selected[:, 1] == label], gt[:, label].sum())
        for label, name in enumerate(CLASSES)})


def curve(events, gt_count, baseline=.1):
    selected = events[events[:, 0] >= baseline]
    selected = selected[np.argsort(-selected[:, 0], kind='stable')]
    if not len(selected):
        return dict(threshold=np.array([baseline]), tp=np.array([0]), fp=np.array([0]),
                    f1=np.array([0.]), f2=np.array([0.]))
    ends = np.r_[np.flatnonzero(selected[:-1, 0] != selected[1:, 0]), len(selected)-1]
    tp = np.cumsum(selected[:, 2] == 0)[ends]
    n = ends + 1
    return dict(threshold=selected[ends, 0], tp=tp, fp=n-tp,
        f1=2*tp/np.maximum(n+gt_count, 1), f2=5*tp/np.maximum(n+4*gt_count, 1))


def choose(events, gt_count, baseline=.1, retention=.95, objective='retention'):
    values = curve(events, gt_count, baseline)
    base_tp = int(np.sum((events[:, 0] >= baseline) & (events[:, 2] == 0)))
    if not base_tp:
        return baseline  # No positive evidence: never tune a class away.
    if objective == 'retention':
        eligible = np.flatnonzero(values['tp'] >= int(np.ceil(base_tp * retention)))
        index = int(eligible[0])  # Highest threshold = least FP under TP constraint.
    else:
        index = int(np.argmax(values[objective]))
    # Round DOWN so a copied decimal never drops the boundary TP.
    return max(baseline, float(np.floor(values['threshold'][index]*1e6)/1e6))


def policies(events, gt, baseline=.1, retention=.95, min_gt=50, min_tp=10):
    result = {'baseline_0.1': dict.fromkeys(CLASSES, baseline)}
    for fraction in (.99, .95, .90, .85, .80):
        value = choose(events, gt.sum(), baseline, fraction)
        result[f'global_keep_{int(fraction*100)}pct_tp'] = dict.fromkeys(CLASSES, value)
    for objective in ('f1', 'f2'):
        value = choose(events, gt.sum(), baseline=0., objective=objective)
        result['global_max_' + objective] = dict.fromkeys(CLASSES, value)
    values, balanced, support = {}, {}, {}
    for label, name in enumerate(CLASSES):
        subset = events[events[:, 1] == label]
        n_gt = int(gt[:, label].sum())
        n_tp = int(np.sum((subset[:, 0] >= baseline) & (subset[:, 2] == 0)))
        supported = n_gt >= min_gt and n_tp >= min_tp
        values[name] = choose(subset, n_gt, baseline, retention) if supported else baseline
        balanced[name] = choose(subset, n_gt, baseline, .85) if supported else baseline
        support[name] = dict(gt=n_gt, baseline_tp=n_tp, tuned=supported,
            reason='retention-constrained' if supported else 'insufficient support; retain baseline')
    result['classwise_recall_first'] = values
    result['classwise_balanced'] = balanced
    return result, support


def subset(events, gt, indices):
    return events[np.isin(events[:, 3].astype(int), indices)], gt[indices]


def report(result, destination):
    rows = result['policies']
    lines = ['# E007 epoch-1 score-threshold tuning', '',
        'Best detection checkpoint: `best_fisheye_bev_mAP_epoch_1.pth`. '
        '601 cached validation frames; no weights, NMS, or training changed.', '',
        'Recommended for preserving most existing TP while rejecting most FP: '
        '**balanced_profile.json**, retaining at least 85% of each supported class\'s baseline TP. '
        'For stricter TP preservation use **recall_first_profile.json** (95% by default). '
        'The maximum-F1 global cutoff is %.6f, but loses roughly one quarter of baseline TP. '
        'There is no cutoff that removes every false positive without losing true positives.'
        % rows['global_max_f1']['thresholds']['Car'], '',
        '| Policy | Global cutoff | TP | FP | Precision | Recall | Existing TP retained | FP removed |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, row in rows.items():
        r = row['total']
        cutoff = 'per class' if name.startswith('classwise') else str(row['thresholds']['Car'])
        lines.append('| %s | %s | %d | %d | %.2f%% | %.2f%% | %.2f%% | %.2f%% |' %
            (name, cutoff, r['tp'], r['fp'], 100*r['precision'], 100*r['recall'],
             100*r['baseline_tp_retention'], 100*r['baseline_fp_removed_fraction']))
    lines += ['', 'TP retention and FP removal are relative to score >=0.10, '
              'not relative to all real objects. Recall uses all 5,635 in-range GT boxes.', '',
              '## Recommended balanced per-class profile', '',
              'At least 85% of each supported class\'s baseline TP are retained. '
              'Unsupported classes stay at 0.10. This protects minority classes compared '
              'with a car-dominated global objective.', '',
              '| Class | Cutoff | GT | TP before / after | FP before / after |',
              '|---|---:|---:|---:|---:|']
    for name in CLASSES:
        base = rows['baseline_0.1']['per_class'][name]
        tuned = rows['classwise_balanced']['per_class'][name]
        lines.append('| %s | %.6f | %d | %d / %d | %d / %d |' % (name,
            rows['classwise_balanced']['thresholds'][name], base['gt'],
            base['tp'], tuned['tp'], base['fp'], tuned['fp']))
    lines += ['', 'Van has no validation GT; Cyclist has only 12 boxes. Neither cutoff was '
              'optimized: both retain 0.10. Truck/Bus evidence is also limited and correlated.', '',
              '## Chronological audit', '',
              'Fit on the earliest 70% of each bag\'s validation frames; assess on the latest '
              '30%. This is a drift check, not an independent, track-separated test.', '',
              '| Policy fitted on early frames | Late TP | Late FP | Late precision | Late recall |',
              '|---|---:|---:|---:|---:|']
    for name in ('baseline_0.1', 'global_max_f1', 'classwise_balanced', 'classwise_recall_first'):
        r = result['chronological_70_30_audit']['policies'][name]['late']['total']
        lines.append('| %s | %d | %d | %.2f%% | %.2f%% |' %
                     (name, r['tp'], r['fp'], 100*r['precision'], 100*r['recall']))
    lines += ['', '## Scope and limitations', '',
              '- Matching: class-aware greedy one-to-one BEV center distance <2m. '
              'Fixed-policy sensitivity at 1m/4m and per-bag counts are in the JSON report.',
              '- Ranges: vehicles <48m, pedestrians <36m, cyclists <42m. '
              'No claim is made about thresholds beyond those ranges.',
              '- GT is the DetZero/Kalman/camera-hold pseudo-label set, not independent manual truth. '
              'An unlabeled real object is counted as FP.',
              '- These are validation-tuned operating points, not an official AP/NDS improvement '
              'or an unbiased test. High-confidence duplicates can survive score filtering.',
              '- Scores are promoted to float64 for exact threshold comparisons. The old NumPy '
              'audit compared float16 AMP scores to a float16-rounded cutoff: its 0.10 included '
              '15 scores actually equal to 0.0999755859375 (1 TP, 14 FP). This report excludes them.',
              '- No retraining or ONNX re-export is needed. Apply the cutoff after decoding, '
              'keeping class-wise rotated BEV NMS at IoU 0.5.', '',
              'These measured counts use cached native AMP predictions. ONNX FP32 score '
              'differences can move borderline boxes across the cutoffs; full ONNX operating-point '
              'metrics have not been recomputed.', '',
              'Full precision/recall counts, support checks, source SHA256s, and matching sensitivity: '
              '`threshold_analysis.json`. Profiles are tied to this checkpoint SHA256.', '']
    destination.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--retention', type=float, default=.95)
    args = parser.parse_args()
    if not 0 < args.retention <= 1:
        parser.error('--retention must be in (0,1]')
    # Heavy framework imports needed only to read our trusted saved predictions.
    from compare_retrained_fisheye import load_frames, read_infos
    annotation = args.data_root / 'rosa/fisheye3dod_infos_val.pkl'
    infos = read_infos(annotation)
    frames, _ = load_frames(args.predictions_dir, infos, with_iou=False)
    if not all(np.isfinite(f['scores']).all() for f in frames):
        raise ValueError('Non-finite prediction scores')
    source = json.loads((args.predictions_dir.parent / 'comparison.json').read_text())
    record = next(row for row in source['checkpoints']
                  if Path(row['raw_predictions']).name == 'predictions.pkl' and
                  Path(row['raw_predictions']).parent.name == args.predictions_dir.name)
    checkpoint_sha = sha(args.checkpoint)
    if checkpoint_sha != record['sha256']:
        raise ValueError('Checkpoint does not match the cached inference provenance')
    events, gt = match_events(frames)
    choices, support = policies(events, gt, retention=args.retention)
    per_bag = {}
    for index, info in enumerate(infos):
        per_bag.setdefault(str(info['metainfo']['sequence_id']), []).append(index)
    results = {name: dict(thresholds=limits, **evaluate(events, gt, limits), per_bag={
        bag: evaluate(*subset(events, gt, indices), limits)['total']
        for bag, indices in per_bag.items()}) for name, limits in choices.items()}
    baseline = results['baseline_0.1']['total']
    for row in results.values():
        row['total']['baseline_tp_retention'] = row['total']['tp']/max(baseline['tp'], 1)
        row['total']['baseline_fp_removed_fraction'] = 1-row['total']['fp']/max(baseline['fp'], 1)
    sensitivity = {}
    for radius in (1., 4.):
        alternate, alternate_gt = match_events(frames, radius)
        sensitivity[str(radius)] = {name: evaluate(alternate, alternate_gt, limits)['total']
                                   for name, limits in choices.items()}
    # Supplementary chronological fit/audit, not an independent/track-separated test.
    early, late = [], []
    for indices in per_bag.values():
        ordered = sorted(indices, key=lambda i: int(infos[i]['metainfo']['acquisition_timestamp_ns']))
        boundary = max(1, int(len(ordered)*.7))
        early.extend(ordered[:boundary])
        late.extend(ordered[boundary:])
    fitted, _ = policies(*subset(events, gt, early), retention=args.retention)
    audit = {name: dict(thresholds=limits,
        early=evaluate(*subset(events, gt, early), limits),
        late=evaluate(*subset(events, gt, late), limits)) for name, limits in fitted.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_sha,
        annotation=str(annotation.resolve()), annotation_sha256=sha(annotation),
        predictions_sha256=sha(args.predictions_dir/'predictions.pkl'), frames=len(frames),
        matching='Class-aware greedy BEV center distance <2 m, score ordered, one-to-one',
        range_m=dict(zip(CLASSES, (48, 48, 48, 48, 36, 42))),
        decoder='Existing single-label quality-weighted score + class-wise rotated BEV NMS IoU 0.5; unchanged',
        retention_target=args.retention, class_support=support, policies=results,
        fixed_policy_matching_sensitivity=sensitivity,
        chronological_70_30_audit=dict(early_frames=len(early), late_frames=len(late), policies=audit),
        caveats=['Validation-tuned; not an unbiased test or an official AP/NDS improvement.',
                 'GT is DetZero/Kalman/camera-hold pseudo-labels; missing labels count as FP.',
                 'Frame counts are temporally correlated; Van absent, Cyclist extremely small.',
                 'Threshold tuning applies only inside the evaluator class ranges.'])
    (args.output_dir/'threshold_analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    profile = dict(format_version=1, checkpoint_sha256=checkpoint_sha, nms_iou=.5,
        score_thresholds=choices['classwise_recall_first'], retention_target=args.retention,
        matching=result['matching'], annotation_sha256=result['annotation_sha256'],
        note='Validation-tuned recall-first profile; weakly supported classes retain 0.1.')
    (args.output_dir/'recall_first_profile.json').write_text(json.dumps(profile, indent=2)+'\n')
    balanced = dict(profile, score_thresholds=choices['classwise_balanced'], retention_target=.85,
        note='Validation-tuned: retain >=85% of each supported class baseline TP; unsupported stay 0.1.')
    (args.output_dir/'balanced_profile.json').write_text(json.dumps(balanced, indent=2)+'\n')
    f1_profile = dict(profile, score_thresholds=choices['global_max_f1'], retention_target=None,
        note='Maximum validation micro F1; loses roughly one quarter of baseline TP.')
    (args.output_dir/'max_f1_profile.json').write_text(json.dumps(f1_profile, indent=2)+'\n')
    report(result, args.output_dir/'REPORT.md')
    with (args.output_dir/'global_threshold_sweep.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['threshold', *counts(events[:0], gt.sum())])
        writer.writeheader()
        for threshold in np.arange(.01, .901, .01):
            writer.writerow(dict(threshold=round(float(threshold), 2),
                **evaluate(events, gt, dict.fromkeys(CLASSES, float(threshold)))['total']))
    for name, row in results.items():
        print(name, json.dumps(dict(thresholds=row['thresholds'], **row['total'])), flush=True)
    print(args.output_dir/'threshold_analysis.json', flush=True)


if __name__ == '__main__':
    main()
