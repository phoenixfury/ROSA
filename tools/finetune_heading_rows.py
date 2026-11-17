#!/usr/bin/env python3
"""Fine-tune 1,025 existing final-yaw coefficients from frozen model features.

Exact architecture is retained. Backbone/BN/classification/centers/sizes and
the first five decoder stages are byte-identical to the parent checkpoint.
Replay is drawn only from the old TRAIN set. Old validation is a selection
guard, never a training loss. New-bag targets have a purged chronological split.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

ROSA = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def design(row, indices):
    return np.concatenate((row['hidden'][indices].astype(np.float32)*row['yaw_gate'][indices, None],
        row['yaw_gate'][indices, None],
        row['faces'][indices].astype(np.float32)*row['face_gate'][indices, None]), axis=1)


def match_queries(row, targets):
    candidates = np.flatnonzero((row['scores'] >= .05) & (row['labels'] < 4))
    if not len(candidates) or not targets:
        return np.array([], dtype=int), np.array([], dtype=int)
    gt = np.asarray([t['box'] for t in targets])
    costs = np.linalg.norm(row['boxes'][candidates, None, :2]-gt[None, :, :2], axis=-1)
    ia, ib = linear_sum_assignment(costs)
    keep = costs[ia, ib] < 1.5
    return candidates[ia[keep]], ib[keep]


def collect_supervision(records, frames, split, feature_function=None):
    x, y, directed, weights, base, frame_ids = [], [], [], [], [], []
    for record in records:
        frame = frames[record['source_index']]
        if frame['split'] != split:
            continue
        with np.load(record['path'], allow_pickle=False) as row:
            iq, ig = match_queries(row, frame['targets'])
            if not len(iq):
                continue
            target = [frame['targets'][i] for i in ig]
            x.append(design(row, iq) if feature_function is None else feature_function(row,iq,record))
            base.extend(row['boxes'][iq, 6].tolist())
            y.extend([t['box'][6] for t in target])
            directed.extend([t['directed_confidence'] >= .7 for t in target])
            weights.extend([t['score'] for t in target])
            frame_ids.extend([record['source_index']]*len(iq))
    if not x:
        raise ValueError('No independently supervised matched headings in '+split)
    return dict(x=np.concatenate(x), target=np.asarray(y, dtype=np.float32),
        base=np.asarray(base, dtype=np.float32), directed=np.asarray(directed),
        weight=np.asarray(weights, dtype=np.float32), frames=np.asarray(frame_ids))


def collect_teacher(records):
    values = []
    for record in records:
        with np.load(record['path'], allow_pickle=False) as row:
            indices = np.flatnonzero(row['scores'] >= .1)
            if len(indices):
                values.append(design(row, indices))
    if not values:
        raise ValueError('Empty replay/validation teacher')
    return np.concatenate(values)


def angular_metrics(prediction, target):
    d = prediction-target
    directed = np.degrees(np.abs((d+np.pi)%(2*np.pi)-np.pi))
    axis = np.degrees(np.abs((d+np.pi/2)%np.pi-np.pi/2))
    return dict(matches=len(d), axis_mean_deg=float(axis.mean()), axis_median_deg=float(np.median(axis)),
        axis_p95_deg=float(np.percentile(axis, 95)), directed_mean_deg=float(directed.mean()),
        directed_median_deg=float(np.median(directed)), axis_over_45deg_fraction=float(np.mean(axis > 45)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--targets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--replay-weight', type=float, default=2.)
    p.add_argument('--guard-mean-deg', type=float, default=3.)
    p.add_argument('--guard-p95-deg', type=float, default=10.)
    args = p.parse_args()
    torch.manual_seed(3407)
    np.random.seed(3407)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.cache/'cache_manifest.json').read_text())
    targets = json.loads(args.targets.read_text())
    train = collect_supervision(manifest['datasets']['new']['records'], targets['frames_data'], 'train')
    val = collect_supervision(manifest['datasets']['new']['records'], targets['frames_data'], 'val')
    replay = collect_teacher(manifest['datasets']['replay']['records'])
    guard = collect_teacher(manifest['datasets']['old_val']['records'])
    device = torch.device('cuda')
    tx = torch.from_numpy(train['x']).to(device)
    base = torch.from_numpy(train['base']).to(device)
    target = torch.from_numpy(train['target']).to(device)
    directed = torch.from_numpy(train['directed']).to(device)
    weight = torch.from_numpy(train['weight']).to(device)
    rx = torch.from_numpy(replay).to(device)
    vx = torch.from_numpy(val['x']).to(device)
    gx = torch.from_numpy(guard).to(device)
    delta = torch.nn.Parameter(torch.zeros(1025, device=device))
    optimizer = torch.optim.Adam([delta], lr=args.lr)
    initial = angular_metrics(val['base'], val['target'])
    result = dict(parent=manifest['checkpoint'], parent_sha256=manifest['checkpoint_sha256'],
        feature_manifest_sha256=sha(args.cache/'cache_manifest.json'), targets_sha256=sha(args.targets),
        optimized_scalar_coefficients=1025, method='final existing yaw trunk row + signed-face yaw row only',
        train_matches=len(tx), val_matches=len(vx), replay_queries=len(rx), guard_queries=len(gx),
        moving_ego_frames_in_training=targets['split_summary']['train']['moving_ego_frames'],
        split=targets['chronological_split'], baseline_new_val=initial, history=[],
        caveats=targets['caveats'], guard=dict(mean_degrees=args.guard_mean_deg, p95_degrees=args.guard_p95_deg))
    best_metric, best, best_epoch = initial['axis_mean_deg'], delta.detach().cpu().clone(), 0
    wait = 0
    for epoch in range(1, args.epochs+1):
        permutation = torch.randperm(len(tx), device=device)
        losses = []
        for indices in permutation.split(256):
            prediction = base[indices]+tx[indices]@delta
            difference = prediction-target[indices]
            # Directed supervision only when independent CP front/back voting
            # is consistent; otherwise supervise its unambiguous physical axis.
            angular = torch.where(directed[indices], 1-torch.cos(difference),
                                  .5*(1-torch.cos(2*difference)))
            new_loss = (angular*weight[indices]).sum()/weight[indices].sum()
            old_indices = torch.randint(len(rx), (len(indices),), device=device)
            replay_loss = (1-torch.cos(rx[old_indices]@delta)).mean()
            loss = new_loss+args.replay_weight*replay_loss+.01*delta.square().sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([delta], 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            val_prediction = val['base']+(vx@delta).cpu().numpy()
            drift = (gx@delta).cpu().numpy()
        metrics = angular_metrics(val_prediction, val['target'])
        wrapped_drift = np.degrees(np.abs((drift+np.pi)%(2*np.pi)-np.pi))
        metrics.update(epoch=epoch, mean_train_loss=float(np.mean(losses)),
            old_guard_mean_yaw_change_deg=float(wrapped_drift.mean()),
            old_guard_p95_yaw_change_deg=float(np.percentile(wrapped_drift, 95)))
        acceptable = (metrics['old_guard_mean_yaw_change_deg'] <= args.guard_mean_deg and
                      metrics['old_guard_p95_yaw_change_deg'] <= args.guard_p95_deg)
        metrics['guard_passed'] = bool(acceptable)
        result['history'].append(metrics)
        if acceptable and metrics['axis_mean_deg'] < best_metric-.02:
            best_metric, best, best_epoch = metrics['axis_mean_deg'], delta.detach().cpu().clone(), epoch
            wait = 0
        else:
            wait += 1
        print(json.dumps(metrics), flush=True)
        if wait >= 15:
            break
    result.update(best_epoch=best_epoch, best_new_val_axis_mean_deg=best_metric,
                  stopped_epoch=epoch, stop_reason='15 validations without guarded improvement' if wait >= 15 else 'epoch cap')
    np.savez(args.output/'heading_delta.npz', delta=best.numpy())
    checkpoint = torch.load(manifest['checkpoint'], map_location='cpu', weights_only=False)
    if sha(manifest['checkpoint']) != manifest['checkpoint_sha256']:
        raise ValueError('Parent checkpoint changed during feature caching')
    state = checkpoint['state_dict']
    prefix = 'head.fisheye_box_decoder.layers.5.refinement.'
    keys = (prefix+'trunk.3.weight', prefix+'trunk.3.bias', prefix+'face_refinement.weight')
    original = {key:state[key].clone() for key in keys}
    state[keys[0]][6] += best[:256].to(state[keys[0]].dtype)
    state[keys[1]][6] += best[256].to(state[keys[1]].dtype)
    state[keys[2]][3] += best[257:].to(state[keys[2]].dtype)
    for key, row in ((keys[0], 6), (keys[1], 6), (keys[2], 3)):
        mask = torch.ones(state[key].shape[0], dtype=torch.bool)
        mask[row] = False
        if not torch.equal(state[key][mask], original[key][mask]):
            raise RuntimeError('A non-yaw output row changed')
    for key in ('optimizer', 'param_schedulers', 'message_hub'):
        checkpoint.pop(key, None)
    checkpoint.setdefault('meta', {})['heading_adaptation'] = dict(experiment='E008_static_heading',
        parent_sha256=manifest['checkpoint_sha256'], best_epoch=best_epoch,
        only_changed_keys=list(keys), optimized_scalar_coefficients=1025,
        cached_train_frames=targets['split_summary']['train']['frames'],
        source_bag='2026_09_25_13_28_18', temporal_slots=0)
    destination = args.output/'best_heading_only.pth'
    torch.save(checkpoint, destination)
    result.update(checkpoint=str(destination), checkpoint_sha256=sha(destination), only_changed_keys=list(keys),
        frozen_architecture_and_non_yaw_rows_preserved=True,
        note='Cached-feature screening; independently rerun PyTorch and old validation before promoting weights.')
    (args.output/'training_report.json').write_text(json.dumps(result, indent=2)+'\n')
    profile = json.loads((ROSA/'work_dirs/threshold_tuning_e1_20261103/balanced_profile.json').read_text())
    profile.update(checkpoint_sha256=result['checkpoint_sha256'],
        note='Inherited E007 thresholds; raw scores/classes are unchanged by final-yaw-only adaptation; not retuned.')
    (args.output/'inherited_threshold_profile.json').write_text(json.dumps(profile, indent=2)+'\n')
    print('CHECKPOINT', destination, 'best_epoch', best_epoch, flush=True)


if __name__ == '__main__':
    main()
