#!/usr/bin/env python3
"""Audit the completed E007 run and its same-protocol checkpoint inference.

Does not change weights, decoding, labels or training source. Inference is
exported separately with infer_fisheye_checkpoints.py; videos use the existing
four-camera/LiDAR comparison renderer. All AP/NDS comparisons retain one split.
"""
import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean

from train_balanced_fisheye import ROOT, ROSA, Config, torch
from compare_retrained_fisheye import (
    CLASSES, RANGES, dataset_audit, gt_neighborhoods, load_frames,
    matched_recall_points, operating_counts, read_infos, select)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def history(run):
    paths = list(run.glob('*/vis_data/scalars.json'))
    if len(paths) != 1:
        raise ValueError('Expected exactly one scalar history for this run')
    training, validation, nonfinite = defaultdict(list), [], []
    for line in paths[0].read_text().splitlines():
        row = json.loads(line)
        if any(isinstance(value, (int, float)) and not math.isfinite(value)
               for value in row.values()):
            nonfinite.append(row.get('step'))
        if 'fisheye_bev/mAP' in row:
            validation.append(dict(epoch=int(row['step']), metrics={
                key.split('/', 1)[1]: value for key, value in row.items()
                if key.startswith('fisheye_bev/')}))
        elif 'loss' in row:
            training[int(row['epoch'])].append(row)
    for row in validation:
        train = training[row['epoch']]
        row['mean_logged_training'] = {
            key: mean(item[key] for item in train if key in item)
            for key in ('loss', 'loss_cls_5', 'loss_box_5', 'loss_cls_dn_5',
                        'loss_box_dn_5', 'grad_norm', 'rosa_grad_norm', 'lr', 'time')}
        row['training_log_points'] = len(train)
    return dict(source=str(paths[0]), source_sha256=sha(paths[0]), epochs=validation,
                nonfinite_log_steps=nonfinite,
                note='Training means summarize logged rolling averages, not exact sample-weighted epoch losses.')


def group(name):
    if '.running_' in name or name.endswith('.num_batches_tracked'):
        return 'batchnorm_buffers'
    if name.startswith('img_backbone.'):
        return 'backbone_parameters'
    if name.startswith('img_neck.'):
        return 'fpn_parameters'
    if name.startswith('head.fisheye_box_decoder.'):
        return 'rosa_decoder_parameters'
    return 'other_state'


def weight_inventory(run, parent, destination):
    if destination.exists():
        cached = json.loads(destination.read_text())
        paths = [parent, *sorted(run.glob('*.pth'))]
        if all(str(path) in cached['checkpoints'] and
               cached['checkpoints'][str(path)]['sha256'] == sha(path) for path in paths):
            return cached
    # These are the user's own local training checkpoints; load on CPU only.
    original = torch.load(parent, map_location='cpu', weights_only=False)
    baseline = original['state_dict']
    original_meta = original.get('meta', {})
    del original
    result = dict(parent=str(parent), parent_epoch=original_meta.get('epoch'), checkpoints={})
    for path in [parent, *sorted(run.glob('*.pth'))]:
        print('Auditing checkpoint', path.name, flush=True)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state = checkpoint['state_dict']
        digest = hashlib.sha256()
        drift = defaultdict(lambda: dict(delta_squared=0., reference_squared=0., elements=0))
        invalid, mismatched = [], []
        for name, value in sorted(state.items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
            if not torch.isfinite(tensor).all():
                invalid.append(name)
            if name not in baseline or tensor.shape != baseline[name].shape:
                mismatched.append(name)
                continue
            if tensor.is_floating_point():
                reference = baseline[name].float()
                difference = tensor.float() - reference
                row = drift[group(name)]
                row['delta_squared'] += float(difference.square().sum())
                row['reference_squared'] += float(reference.square().sum())
                row['elements'] += tensor.numel()
        for row in drift.values():
            row['relative_l2_change'] = math.sqrt(row['delta_squared'] /
                                                 max(row['reference_squared'], 1e-30))
        meta = checkpoint.get('meta', {})
        result['checkpoints'][str(path)] = dict(
            sha256=sha(path), model_state_sha256=digest.hexdigest(), size_bytes=path.stat().st_size,
            epoch=meta.get('epoch'), iteration=meta.get('iter'), state_keys=len(state),
            state_elements=sum(value.numel() for value in state.values()),
            nonfinite_state_keys=invalid, missing_parent_keys=sorted(set(baseline) - set(state)),
            mismatched_keys=mismatched, change_from_initializer=dict(drift),
            optimizer_present='optimizer' in checkpoint,
            experiment_contract=meta.get('experiment_contract'))
        del checkpoint, state
        gc.collect()
    result['epoch7_final_model_identical'] = (
        result['checkpoints'][str(run / 'epoch_7.pth')]['model_state_sha256'] ==
        result['checkpoints'][str(run / 'dual_metric_plateau_epoch_7.pth')]['model_state_sha256'])
    write_json(destination, result)
    return result


def complete_counts(payload):
    def finish(row):
        row = dict(row)
        for key in ('gt', 'predictions', 'tp', 'duplicate_fp',
                    'wrong_class_or_nearby_fp', 'unmatched_fp'):
            row.setdefault(key, 0)
        row.update(fp=row['predictions'] - row['tp'], fn=row['gt'] - row['tp'])
        row['f1'] = 2 * row['precision'] * row['recall'] / max(row['precision'] + row['recall'], 1e-12)
        return row
    return dict(total=finish(payload['total']), per_class={
        name: finish(payload['per_class'].get(name, dict(precision=0., recall=0.)))
        for name in CLASSES})


def diagnostics(directory, infos):
    frames, _ = load_frames(directory, infos, with_iou=False)
    rows = {}
    for threshold in (.1, .2, .3, .5):
        rows[str(threshold)] = complete_counts(operating_counts(
            frames, [select(frame, threshold=threshold) for frame in frames]))
    per_bag = defaultdict(list)
    for frame, info in zip(frames, infos):
        per_bag[str(info['metainfo']['sequence_id'])].append(frame)
    result = dict(score_thresholds=rows, matched_recall=matched_recall_points(frames),
                  gt_neighborhood_at_score_0_1=gt_neighborhoods(frames), per_bag={})
    for sequence, subset in sorted(per_bag.items()):
        result['per_bag'][sequence] = dict(frames=len(subset), score_thresholds={
            str(threshold): complete_counts(operating_counts(
                subset, [select(frame, threshold=threshold) for frame in subset]))
            for threshold in (.1, .2)})
    return result


def nds_decomposition(parent, current):
    # Released Fisheye3DOD: mean_ap_weight=3, three TP metrics, six classes.
    contributions = dict(mAP=.5 * (current['mAP'] - parent['mAP']))
    for key in ('mATE', 'mASE', 'mAOE'):
        contributions[key] = (max(0., 1. - current[key]) - max(0., 1. - parent[key])) / 6.
    per_class = {}
    for name in CLASSES:
        values = dict(AP=(current[name + '_AP'] - parent[name + '_AP']) / 12.)
        for key in ('ATE', 'ASE', 'AOE'):
            # All compared macro-errors are <1, so the macro clip is inactive.
            values[key] = (parent[name + '_' + key] - current[name + '_' + key]) / 36.
        values['total_contribution'] = sum(values.values())
        per_class[name] = values
    return dict(total=current['NDS'] - parent['NDS'], macro_contributions=contributions,
                per_class=per_class, precision='Rounded inference metrics; small rounding residuals expected.')


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = result['training_history']['epochs']
    epochs = [row['epoch'] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for key in ('mAP', 'NDS'):
        axes[0, 0].plot(epochs, [row['metrics'][key] * 100 for row in history], marker='o', label=key)
        if result.get('models'):
            reference = result['models']['parent_adapt4']['metrics'][key]
            axes[0, 0].axhline(reference * 100, linestyle='--', alpha=.5, label='Parent ' + key)
    axes[0, 0].set(title='Same 601-frame validation', xlabel='Epoch', ylabel='Score (%)')
    axes[0, 1].plot(epochs, [row['mean_logged_training']['loss'] for row in history], marker='o')
    axes[0, 1].set(title='Training objective', xlabel='Epoch', ylabel='Mean logged loss')
    for name in ('Car', 'Truck', 'Bus', 'Pedestrian', 'Cyclist'):
        axes[1, 0].plot(epochs, [row['metrics'][name + '_AP'] * 100 for row in history], marker='.', label=name)
    axes[1, 0].set(title='Class AP; Van unsupported in validation', xlabel='Epoch', ylabel='AP (%)')
    if result.get('models'):
        for name, record in result['models'].items():
            rows = list(record['diagnostics']['score_thresholds'].values())
            axes[1, 1].plot([row['total']['recall'] for row in rows],
                            [row['total']['precision'] for row in rows], marker='.', label=name)
        axes[1, 1].set(title='Diagnostic 2m matching; score sweep', xlabel='Recall', ylabel='Precision')
    else:
        axes[1, 1].plot(epochs, [row['mean_logged_training']['lr'] for row in history], marker='o')
        axes[1, 1].set(title='Learning rate remains near its peak', xlabel='Epoch', ylabel='Learning rate')
    for axis in axes.flat:
        axis.grid(alpha=.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
    fig.suptitle('E007 single-frame refresh: best mAP e1, best NDS e3, plateau stop e7')
    fig.tight_layout()
    fig.savefig(output / 'analysis.png', dpi=150)
    plt.close(fig)


def report(result, output):
    models = result['models']
    order = ('parent_adapt4', 'refresh_map_e1', 'refresh_nds_e3', 'refresh_final_e7')
    labels = dict(parent_adapt4='Starting adapted model (e4)', refresh_map_e1='Refresh e1 / best mAP',
                  refresh_nds_e3='Refresh e3 / best NDS', refresh_final_e7='Refresh e7 / final')
    parent, best, nds = [models[key]['metrics'] for key in order[:3]]
    data = result['dataset']
    history = result['training_history']['epochs']
    lines = ['# E007 single-frame refresh — checkpoint analysis', '',
        '## Decision', '',
        '**Prefer the epoch-1 mAP checkpoint as the general detection candidate.** '
        'Epoch 3 is the NDS specialist, not an unqualified better detector; '
        'the final epoch 7 regresses and should not be selected merely because it is newest.', '',
        f'Epoch 1 versus its initializer: mAP {parent["mAP"]*100:.2f}% → {best["mAP"]*100:.2f}% '
        f'({(best["mAP"]-parent["mAP"])*100:+.2f} percentage points), '
        f'NDS {parent["NDS"]:.4f} → {best["NDS"]:.4f}. '
        f'Car AP nevertheless changes {parent["Car_AP"]*100:.2f}% → {best["Car_AP"]*100:.2f}%. '
        'Keep the parent available for car-focused comparison; no deployment weights were replaced.', '',
        '## Same-protocol evaluation', '',
        f'- {result["validation_samples"]} validation frames; SHA256 `{result["validation_sha256"]}`.',
        '- All four checkpoints freshly inferred with the same single-frame config and decoder: '
        'single-label, quality before top-k, class-wise rotated BEV NMS IoU 0.5. No temporal slots.',
        '- Official AP/NDS use all saved detector outputs, not the video display threshold. '
        'Evaluator XY ranges: Car/Van/Truck/Bus 48m, Pedestrian 36m, Cyclist 42m.',
        '- Video display: score >=0.10, 60m BEV viewport, no additional NMS. '
        'Portable JSON export floor is 0.05; raw predictions remain unfiltered by that floor.',
        '- Ground truth is DetZero/Kalman/camera-hold pseudo-labels, not independent manual truth. '
        'The initializer may have earlier-split exposure. This is adaptation validation, not an unbiased test.',
        f'- Validation in-range counts: `{data["val"]["evaluator_range_counts"]}`; '
        f'unique pseudo-tracks by class: `{data["val"]["unique_tracks"]}`. '
        'No Van validation labels: Van AP=0 is not evidence that Van quality was measured.', '',
        '| Model | mAP (%) | NDS | Car AP (%) | mATE (m) | mASE | mAOE (rad) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in order:
        m = models[name]['metrics']
        lines.append(f'| {labels[name]} | {m["mAP"]*100:.2f} | {m["NDS"]:.4f} | '
                     f'{m["Car_AP"]*100:.2f} | {m["mATE"]:.4f} | {m["mASE"]:.4f} | {m["mAOE"]:.4f} |')
    lines += ['', '### Class AP (%)', '', '| Class | Parent | Epoch 1 | Epoch 3 | Epoch 7 |',
              '|---|---:|---:|---:|---:|']
    for cls in CLASSES:
        values = ' | '.join(f'{models[name]["metrics"][cls+"_AP"]*100:.2f}' for name in order)
        lines.append(f'| {cls} | {values} |')
    lines += ['', '## False positives and missed objects', '',
        'These are diagnostic class-aware, score-ordered, one-to-one center matches '
        'within 2m in BEV, restricted to evaluator class ranges. They are **not** '
        'official AP and not an IoU-based physical duplicate count. Pseudo-label '
        'omissions can also count real objects as false positives.', '',
        '| Model (score >=0.10) | Predictions | TP | FP | FN | Precision | Recall | Duplicate-like FP |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in order:
        row = models[name]['diagnostics']['score_thresholds']['0.1']['total']
        lines.append(f'| {labels[name]} | {row["predictions"]} | {row["tp"]} | {row["fp"]} | '
                     f'{row["fn"]} | {row["precision"]*100:.2f}% | {row["recall"]*100:.2f}% | {row["duplicate_fp"]} |')
    old_counts = models['parent_adapt4']['diagnostics']['score_thresholds']['0.1']['total']
    new_counts = models['refresh_map_e1']['diagnostics']['score_thresholds']['0.1']['total']
    lines += ['', f'At score 0.10, epoch 1 reduces diagnostic FP by '
              f'{old_counts["fp"]-new_counts["fp"]} '
              f'({100*(old_counts["fp"]-new_counts["fp"])/old_counts["fp"]:.1f}%), '
              f'but recall changes {old_counts["recall"]*100:.2f}% → {new_counts["recall"]*100:.2f}%. '
              f'**Duplicate-like FP do not improve: {old_counts["duplicate_fp"]} → '
              f'{new_counts["duplicate_fp"]}.** Retraining did not solve duplicate detections.', '',
              '### Precision at matched recall', '',
              'Use complete equal-score groups; achieved recall may exceed the target slightly. '
              'This aggregate diagnostic is car-dominated, unlike six-class macro mAP.', '',
              '| Target recall | Parent precision | Epoch 1 | Epoch 3 | Epoch 7 |',
              '|---|---:|---:|---:|---:|']
    for target in ('0.3', '0.4', '0.5', '0.6'):
        values = []
        for name in order:
            point = models[name]['diagnostics']['matched_recall'][target]
            values.append(f'{point["precision"]*100:.2f}%' if point else 'not reached')
        lines.append(f'| {float(target)*100:.0f}% | ' + ' | '.join(values) + ' |')
    lines += ['', '### Parent versus epoch 1 by source bag (score >=0.10)', '',
              '| Bag | Frames | Parent precision / recall | Epoch-1 precision / recall |',
              '|---|---:|---:|---:|']
    for bag, before in models['parent_adapt4']['diagnostics']['per_bag'].items():
        after = models['refresh_map_e1']['diagnostics']['per_bag'][bag]
        b, a = [row['score_thresholds']['0.1']['total'] for row in (before, after)]
        lines.append(f'| {bag} | {before["frames"]} | {b["precision"]*100:.2f}% / '
                     f'{b["recall"]*100:.2f}% | {a["precision"]*100:.2f}% / {a["recall"]*100:.2f}% |')
    lines += ['', 'The full score sweeps (0.1/0.2/0.3/0.5), per-bag and per-class '
              'counts, and equal-recall precision comparisons are in `comparison.json`. '
              'Do not infer a calibration-independent precision gain from a single shared score threshold.', '',
              '## Why best NDS is not best mAP', '',
              'The released evaluator uses `NDS = (3*mAP + max(0,1-mATE) + '
              'max(0,1-mASE) + max(0,1-mAOE))/6`. It does **not** score velocity '
              'or tracking identity. This is not the five-error standard nuScenes NDS.', '',
              '| Epoch-3 NDS change vs parent | NDS percentage-point contribution |', '|---|---:|']
    decomposition = models['refresh_nds_e3']['NDS_decomposition_vs_parent']
    for key, value in decomposition['macro_contributions'].items():
        lines.append(f'| {key} | {value*100:+.3f} |')
    cyclist = decomposition['per_class']['Cyclist']['total_contribution']
    lines += ['', f'Epoch 3 gains {(nds["NDS"]-parent["NDS"])*100:.3f} NDS points overall. '
              f'The Cyclist AP/error terms alone contribute approximately {cyclist*100:+.3f} points '
              '(rounded-metric decomposition). That tiny class can dominate this aggregate: '
              f'its ATE changes {parent["Cyclist_ATE"]:.4f}m → {nds["Cyclist_ATE"]:.4f}m, '
              f'while its AP changes {parent["Cyclist_AP"]*100:.2f}% → {nds["Cyclist_AP"]*100:.2f}%. '
              'This is a true-positive error statistic over a small, changing matched subset, '
              'not proof of broadly improved cyclist detection.', '',
              '## Training and stopping', '',
              'The controller correctly stopped after epoch 7: neither metric improved '
              'meaningfully in epochs 4–7. Its last reset was the NDS improvement at epoch 3. '
              'Training did not fail or run to the 24-epoch cap.', '',
              '| Epoch | Mean logged loss | Mean logged LR | mAP (%) | NDS |',
              '|---|---:|---:|---:|---:|']
    for row in history:
        train, metrics = row['mean_logged_training'], row['metrics']
        lines.append(f'| {row["epoch"]} | {train["loss"]:.4f} | {train["lr"]:.3g} | '
                     f'{metrics["mAP"]*100:.2f} | {metrics["NDS"]:.4f} |')
    lines += ['', 'The means above summarize logged rolling loss averages, not exact sample-weighted losses. '
              'Falling training objective alongside worsening validation is consistent with '
              'over-adaptation/overfitting to correlated noisy pseudo-labels. This is an inference, '
              'not an isolated causal experiment. Rare-class sampling and live BN may contribute; '
              'their individual effects have not been separated.', '',
              'The 24-epoch cosine schedule remains relatively high-LR during the first seven epochs; '
              'early stopping prevents further damage but does not recover an earlier solution. '
              'No new training or architecture changes were started for this analysis.', '',
              '## Checkpoint and architecture audit', '',
              'Architecture is unchanged: VoVNet-99 backbone, four FPN levels, four 800x800 fisheye inputs, '
              '300 queries, 256-dimensional embeddings and six hybrid V4 decoder layers; '
              'sparse image evidence uses fixed/learned keypoints, separate size/yaw updates and quality scores. '
              'Zero temporal slots and baseline zero-velocity target ABI remain in use. '
              'The backbone and decoder trained; BN statistics were live; effective batch was 8 '
              '(physical batch 1, accumulation 8), with class-balanced DN and photometric augmentation.', '',
              'All retained checkpoint model tensors are finite with matching parent key shapes. '
              f'Epoch 7 regular/final model tensor hashes agree: **{result["checkpoint_audit"]["epoch7_final_model_identical"]}**. '
              'Checkpoint epochs/iterations, hashes and optimizer presence are in `checkpoint_audit.json`.', '',
              '| Checkpoint | Backbone relative L2 change | FPN | ROSA decoder | BN buffers |',
              '|---|---:|---:|---:|---:|']
    for name in order[1:]:
        audit = result['checkpoint_audit']['checkpoints'][models[name]['checkpoint']]
        changes = audit['change_from_initializer']
        values = ' | '.join(f'{changes[key]["relative_l2_change"]*100:.3f}%'
            for key in ('backbone_parameters', 'fpn_parameters', 'rosa_decoder_parameters', 'batchnorm_buffers'))
        lines.append(f'| {labels[name]} | {values} |')
    lines += ['', 'Relative L2 changes are descriptive, not layer importance or proof of causality. '
              'The evidence supports parameter adaptation plus generalization loss, not a corrupt '
              'checkpoint or accidentally enabled temporal cache.', '',
              '## Recommended next steps (not executed)', '',
              '1. Inspect the epoch-1 video as the general-purpose candidate; compare to the '
              'parent for cars and to epoch 3 for localization. Do not automatically promote epoch 3 on NDS alone.',
              '2. Keep class-wise NMS. If operational precision is insufficient, tune class-specific '
              'score thresholds on a separate tuning set; report precision at equal recall, '
              'not just at a shared score. No thresholds were changed here.',
              '3. If training again, compare a short 1–3 epoch, lower-LR refresh (e.g. 1e-5) '
              'with a short cosine schedule, and a matched frozen-BN control. Change one factor at a time.',
              '4. Review rare-class pseudo-labels and use a manually reviewed, track-separated '
              'holdout with Van examples and more Cyclist tracks before claiming minority-class gains.', '',
              '## Videos and reproducibility', '',
              'All videos contain four synchronized fisheye projections plus LiDAR BEV boxes. '
              'Predictions are solid/class-colored; GT is dashed white pseudo-labels. '
              'The complete 601-frame validation order is shown at 2fps (300.5 seconds), '
              'chronological within scenes with hard cuts between scenes/bags.', '',
              '- [Epoch 1 / best mAP](videos/refresh_map_e1.mp4)',
              '- [Epoch 3 / best NDS](videos/refresh_nds_e3.mp4)',
              '- [Starting model](videos/parent_adapt4.mp4)',
              '- [Parent vs epoch 1](videos/parent_adapt4_vs_refresh_map_e1.mp4)',
              '- [Parent vs epoch 3](videos/parent_adapt4_vs_refresh_nds_e3.mp4)', '',
              'Inference config/checkpoint hashes are in `evaluations/comparison.json`, '
              'and exact executed commands are in `inference_console.log`; '
              'training inputs/source hashes are in `training_analysis.json`; video frame tokens, '
              'decoder/thresholds and labels are in `videos/manifest.json`. '
              'The saved geometry is explicitly gravity-center boxes, not bottom-centered boxes.', '',
              '![Learning curves and operating points](analysis.png)', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))


def archive(result, output):
    """Append a post-training analysis record; never rewrite training patches."""
    import shutil
    import subprocess
    from archive_rosa_experiments import addition_patch, check_patch, immutable

    destination = ROSA / 'experiments/E007_singleframe_refresh'
    for source, target in [('comparison.json', 'checkpoint_comparison.json'),
                           ('training_analysis.json', 'training_analysis.json'),
                           ('checkpoint_audit.json', 'checkpoint_audit.json'),
                           ('analysis.png', 'checkpoint_analysis.png')]:
        shutil.copy2(output / source, destination / target)
    prefix = '../../work_dirs/' + output.relative_to(ROSA / 'work_dirs').as_posix() + '/'
    body = (output / 'REPORT.md').read_text()
    body = body.replace('(videos/', '(' + prefix + 'videos/')
    body = body.replace('(analysis.png)', '(checkpoint_analysis.png)')
    body = body.replace('`comparison.json`', '`checkpoint_comparison.json`')
    (destination / 'checkpoint_comparison.md').write_text(body)
    files = ['tools/analyze_singleframe_refresh.py', 'tests/test_singleframe_analysis.py']
    patch = destination / 'checkpoint_analysis.patch'
    immutable(patch, b''.join(addition_patch(name, (ROSA / name).read_text()) for name in files))
    base = json.loads((destination / 'record.json').read_text())['patch_base_commit']
    check_patch(base, [destination / 'rosa.patch', destination / 'archive_tooling.patch', patch])
    videos = {}
    for path in sorted((output / 'videos').glob('*.mp4')):
        video = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,width,height,nb_frames,r_frame_rate:format=duration',
            '-of', 'json', str(path)], check=True, capture_output=True, text=True).stdout)
        if int(video['streams'][0]['nb_frames']) != 601 or abs(float(video['format']['duration']) - 300.5) > .01:
            raise ValueError('Video does not cover the complete expected validation timeline')
        videos[path.name] = dict(path=str(path), sha256=sha(path), **video)
    write_json(destination / 'checkpoint_comparison_record.json', dict(
        status='complete', output=str(output), validation_sha256=result['validation_sha256'],
        analysis_source_sha256={name: sha(ROSA / name) for name in files},
        source_patch_sha256=sha(patch), source_patch_base=base,
        patch_order=['rosa.patch', 'archive_tooling.patch', 'checkpoint_analysis.patch'],
        patch_verification='git apply --check on clean temporary pinned base',
        checkpoint_sha256={name: model['checkpoint_sha256'] for name, model in result['models'].items()},
        artifacts={name: sha(destination / name) for name in ('checkpoint_comparison.md',
                   'checkpoint_comparison.json', 'checkpoint_audit.json', 'checkpoint_analysis.png')},
        videos=videos, note='Post-training analysis only. Training patch, metrics and original report remain unchanged.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROSA / 'work_dirs/singleframe_refresh_20261031')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/Custom/combined_improved_detzero_rosa')
    parser.add_argument('--output-dir', type=Path, default=ROSA / 'work_dirs/singleframe_refresh_analysis_20261102')
    parser.add_argument('--training-only', action='store_true')
    parser.add_argument('--video-manifest-only', action='store_true',
                        help='Prepare the parent/e1/e3 renderer input once those inferences finish')
    parser.add_argument('--archive', action='store_true', help='Append checked analysis patch/results to E007')
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = args.run.resolve()
    plan = json.loads((run / 'plan.json').read_text())
    parent = Path(plan['initialization'])
    val_file = args.data_root / 'rosa/fisheye3dod_infos_val.pkl'
    train_file = args.data_root / 'rosa/fisheye3dod_infos_train.pkl'
    if sha(val_file) != plan['validation_sha256'] or sha(train_file) != plan['train_sha256']:
        raise ValueError('Dataset no longer matches the captured training protocol')
    if args.video_manifest_only:
        inference = json.loads((args.output_dir / 'evaluations/comparison.json').read_text())
        aliases = {parent.stem: 'parent_adapt4',
            'best_fisheye_bev_mAP_epoch_1': 'refresh_map_e1',
            'best_fisheye_bev_NDS_epoch_3': 'refresh_nds_e3'}
        models = {aliases[row['name']]: dict(directory=str(Path(row['raw_predictions']).parent),
                  metrics=row['metrics'], checkpoint=row['path'], checkpoint_sha256=row['sha256'])
                  for row in inference['checkpoints'] if row['name'] in aliases}
        if set(models) != set(aliases.values()):
            raise ValueError('Parent/e1/e3 inference must finish before video rendering')
        write_json(args.output_dir / 'video_comparison.json', dict(
            validation_sha256=sha(val_file), models=models,
            same_decoder='single-label, quality-first, class-wise NMS IoU 0.5; temporal slots=0'))
        return
    # Inspect the resolved data-only config; do not re-import plugin aliases
    # into the CPU evaluator's registry a second time.
    config = Config.fromfile(run / 'rosa_vov99_singleframe_refresh_24e.py',
                             import_custom_modules=False)
    architecture = dict(backbone=dict(config.model.img_backbone), fpn=dict(config.model.img_neck),
        head_type=config.model.head.type, instance_bank=dict(config.model.head.instance_bank),
        decoder=dict(config.model.head.v4_decoder_cfg), output_decoder=dict(config.model.head.decoder),
        temporal_enabled=config.model.head.get('temporal_enabled', False),
        reg_weights=config.model.head.reg_weights,
        class_weights=config.model.head.loss_cls.class_weights,
        optimizer=dict(config.optim_wrapper), schedules=config.param_scheduler)
    result = dict(validation_sha256=sha(val_file), validation_samples=601,
        plan=plan, stop=json.loads((run / 'results.json').read_text()), architecture=architecture,
        training_history=history(run), dataset=dataset_audit(read_infos(train_file), read_infos(val_file)),
        checkpoint_audit=weight_inventory(run, parent, args.output_dir / 'checkpoint_audit.json'),
        source_sha256={str(Path(__file__).resolve().relative_to(ROOT)): sha(__file__)}, models={})
    write_json(args.output_dir / 'training_analysis.json', result)
    if not args.training_only:
        inference = json.loads((args.output_dir / 'evaluations/comparison.json').read_text())
        aliases = {parent.stem: 'parent_adapt4',
            'best_fisheye_bev_mAP_epoch_1': 'refresh_map_e1',
            'best_fisheye_bev_NDS_epoch_3': 'refresh_nds_e3',
            'dual_metric_plateau_epoch_7': 'refresh_final_e7'}
        infos = read_infos(val_file)
        for record in inference['checkpoints']:
            name = aliases[record['name']]
            if record['prediction_frames'] != len(infos) or sha(record['path']) != record['sha256']:
                raise ValueError('Inference/checkpoint mismatch')
            directory = Path(record['raw_predictions']).parent
            print('Analyzing predictions', name, flush=True)
            cache = directory / 'diagnostics.json'
            fingerprint = sha(directory / 'predictions.pkl')
            previous = json.loads(cache.read_text()) if cache.exists() else {}
            if previous.get('predictions_sha256') != fingerprint or previous.get('validation_sha256') != sha(val_file):
                previous = dict(predictions_sha256=fingerprint, validation_sha256=sha(val_file),
                                diagnostics=diagnostics(directory, infos))
                write_json(cache, previous)
            result['models'][name] = dict(directory=str(directory), checkpoint=record['path'],
                checkpoint_sha256=record['sha256'], metrics=record['metrics'],
                diagnostics=previous['diagnostics'])
        baseline = result['models']['parent_adapt4']['metrics']
        for name, record in result['models'].items():
            record['NDS_decomposition_vs_parent'] = nds_decomposition(baseline, record['metrics'])
        result['same_decoder'] = 'single-label; quality-first; class-wise rotated BEV NMS IoU 0.5; no temporal slots'
        result['diagnostic_matching'] = 'Class-aware, score-ordered greedy BEV center distance <2m; evaluator class ranges; not official AP or physical duplicate counting.'
        result['video_protocol'] = dict(score_threshold=.1, fps=2., additional_nms=False,
            box_origin='gravity_center', ground_truth='white dashed DetZero pseudo-labels')
        write_json(args.output_dir / 'comparison.json', result)
        report(result, args.output_dir)
    plot(result, args.output_dir)
    if args.archive:
        if args.training_only:
            raise ValueError('--archive requires completed checkpoint inference')
        archive(result, args.output_dir)
    print('Analysis written to', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
