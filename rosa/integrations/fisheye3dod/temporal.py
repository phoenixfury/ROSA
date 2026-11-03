"""Chronological query memory and track-derived motion for custom fisheye bags.

Memory is detached between frames (one-frame truncated backpropagation). A
batch lane owns one chronological stream; scene/prev-token/time checks make
cross-bag, shuffled-frame and train/validation reuse impossible.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Sampler
from mmengine.dist import get_dist_info
from mmengine.hooks import Hook
from mmdet3d.registry import DATASETS, DATA_SAMPLERS, HOOKS, METRICS, MODELS

from projects.mmdet3d_plugin.models.instance_bank import InstanceBank
from projects.mmdet3d_plugin.core.box3d import CNS
from projects.mmdet3d_plugin.models.detection3d.decoder import SparseBox3DDecoder
from .dataset import Fisheye3DODSparse4DDataset
from .metric import Fisheye3DODReleasedMetricBridge


def track_velocities(samples, max_gap_seconds=1.5, max_speed_mps=45.):
    """Causal world displacement, rotated into the current LiDAR XY axes.

    Track IDs only identify the same pseudo-labelled object. Unknown motion
    remains NaN so it is excluded from regression, rather than labelled static.
    No observations from another scene/split or a future frame are used.
    """
    previous = {}
    velocities = {}
    for sample in sorted(samples, key=lambda s: int(s['metainfo']['acquisition_timestamp_ns'])):
        meta = sample['metainfo']
        scene = str(meta['scene_name'])
        timestamp = int(meta['acquisition_timestamp_ns'])
        pose = np.asarray(meta['T_world_CLIDAR'], dtype=np.float64)
        if (pose.shape != (4, 4) or not np.isfinite(pose).all() or
                not np.allclose(pose[3], [0, 0, 0, 1])):
            raise ValueError('Temporal training requires a valid T_world_CLIDAR')
        boxes = np.asarray(sample['annos']['gt_boxes']).reshape(-1, 7)
        ids = np.asarray(sample['annos']['gt_ids'], dtype=np.int64)
        if len(ids) != len(boxes) or len(set(ids.tolist())) != len(ids):
            raise ValueError('Temporal training requires aligned, unique per-frame track IDs')
        values = np.full((len(boxes), 2), np.nan, dtype=np.float32)
        for index, (box, track_id) in enumerate(zip(boxes, ids)):
            world = pose[:3, :3] @ box[:3] + pose[:3, 3]
            key = scene, int(track_id)
            if key in previous:
                past_time, past_world = previous[key]
                dt = (timestamp - past_time) * 1e-9
                if 0 < dt <= max_gap_seconds:
                    world_velocity = (world - past_world) / dt
                    if np.isfinite(world_velocity).all() and np.linalg.norm(world_velocity[:2]) <= max_speed_mps:
                        values[index] = (pose[:3, :3].T @ world_velocity)[:2]
            previous[key] = timestamp, world
        velocities[str(sample['token'])] = values
    return velocities


@DATASETS.register_module()
class TemporalFisheye3DODDataset(Fisheye3DODSparse4DDataset):
    """Read temporal PKLs, preserving IDs and adding weak velocity targets."""

    def __init__(self, *args, velocity_max_gap_seconds=1.5,
                 velocity_max_speed_mps=45., **kwargs):
        self._temporal_velocities = {}
        kwargs['lazy_init'] = False
        kwargs['with_velocity'] = True
        if kwargs.get('filter_empty_gt', False):
            raise ValueError('Temporal datasets must retain empty frames')
        kwargs['filter_empty_gt'] = False
        super().__init__(*args, **kwargs)
        samples = [super(TemporalFisheye3DODDataset, self).get_data_info(i)
                   for i in range(len(self))]
        self.temporal_records = []
        for index, sample in enumerate(samples):
            meta = sample['metainfo']
            required = ('acquisition_timestamp_ns', 'T_world_CLIDAR',
                        'temporal_split', 'scene_name', 'prev_id', 'next_id')
            if any(key not in meta for key in required):
                raise ValueError('Use the acquisition-time chronological combined dataset for temporal training')
            self.temporal_records.append(dict(index=index,
                scene=str(meta['scene_name']), token=str(sample['token']),
                timestamp_ns=int(meta['acquisition_timestamp_ns'])))
        self._temporal_velocities = track_velocities(
            samples, velocity_max_gap_seconds, velocity_max_speed_mps)

    def get_data_info(self, index):
        info = super().get_data_info(index)
        velocity = self._temporal_velocities.get(str(info['token']))
        if velocity is not None:
            # The released evaluator does not support unknown velocity. Keep
            # its legacy zero fallback there; training masks unknown targets.
            target = np.nan_to_num(velocity) if self.test_mode else velocity
            for key in ('ann_info', 'eval_ann_info'):
                if key in info:
                    boxes = info[key]['gt_bboxes_3d']
                    if len(boxes) != len(target):
                        raise ValueError('Track velocity and GT order differ')
                    boxes.tensor[:, 7:9] = boxes.tensor.new_tensor(target)
        return info

    def __getitem__(self, index):
        padding = False
        if isinstance(index, tuple):
            index, padding = index
        result = super().__getitem__(index)
        result['data_samples'].set_metainfo({'temporal_is_padding': bool(padding)})
        return result


@DATA_SAMPLERS.register_module()
class TemporalSequenceBatchSampler(Sampler):
    """Fixed chronological lanes on every DDP rank; shuffle whole clips only.

    Rank/lane assignment is deterministic and disjoint. Short lanes pad only
    at their tails, explicitly marked so validation never counts duplicates.
    Training tail repetitions retain their real GT, with memory reset.
    """

    def __init__(self, sampler, batch_size, clip_length=32, shuffle=True,
                 seed=3407, rank=None, world_size=None):
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.clip_length = int(clip_length)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        default_rank, default_world = get_dist_info()
        self.rank = default_rank if rank is None else int(rank)
        self.world_size = default_world if world_size is None else int(world_size)
        if self.batch_size < 1 or self.clip_length < 0 or not 0 <= self.rank < self.world_size:
            raise ValueError('Invalid temporal lane configuration')
        dataset = sampler.dataset
        if not hasattr(dataset, 'temporal_records'):
            raise TypeError('Temporal batching requires TemporalFisheye3DODDataset, without a repeat wrapper')
        grouped = defaultdict(list)
        self._timestamps = {}
        for record in dataset.temporal_records:
            grouped[record['scene']].append(record)
            self._timestamps[record['index']] = record['timestamp_ns']
        clips = []
        for records in grouped.values():
            records.sort(key=lambda row: row['timestamp_ns'])
            width = self.clip_length or len(records)
            clips.extend([row['index'] for row in records[start:start+width]]
                         for start in range(0, len(records), width))
        if not clips:
            raise ValueError('Empty temporal dataset')
        self.lanes = [[] for _ in range(self.world_size * self.batch_size)]
        lengths = [0] * len(self.lanes)
        for clip in sorted(clips, key=len, reverse=True):
            lane = min(range(len(lengths)), key=lambda i: (lengths[i], i))
            self.lanes[lane].append(clip)
            lengths[lane] += len(clip)
        self.steps = max(lengths)
        self.padding_frames = len(self.lanes) * self.steps - sum(lengths)
        self.fallback = clips[0][0]
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps

    def __iter__(self):
        epoch = getattr(self.sampler, 'epoch', self.epoch)
        streams = []
        for lane in range(self.rank*self.batch_size, (self.rank+1)*self.batch_size):
            clips = list(self.lanes[lane])
            if self.shuffle:
                generator = torch.Generator().manual_seed(self.seed + int(epoch)*997 + lane)
                clips = [clips[i] for i in torch.randperm(len(clips), generator=generator).tolist()]
            else:
                clips.sort(key=lambda clip: self._timestamps[clip[0]])
            streams.append([index for clip in clips for index in clip])
        for step in range(self.steps):
            yield [(stream[step], False) if step < len(stream)
                   else (stream[-1] if stream else self.fallback, True)
                   for stream in streams]


@MODELS.register_module()
class TemporalInstanceBank(InstanceBank):
    """Propagate detached high-confidence query slots into the next frame."""

    def __init__(self, *args, min_confidence=.15, cache_score_mode='class',
                 cache_nms_iou_threshold=None, cache_require_current=False,
                 motion_mode='predicted', collect_cache_stats=False, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < self.num_temp_instances < self.num_anchor:
            raise ValueError('Temporal memory must leave fresh detection queries')
        if not 0 <= min_confidence < 1 or self.max_time_interval <= 0:
            raise ValueError('Invalid temporal confidence/time limit')
        if self.anchor_handler is None:
            raise ValueError('Temporal memory requires an anchor projection handler')
        self.min_confidence = float(min_confidence)
        if cache_score_mode not in ('class', 'quality'):
            raise ValueError('cache_score_mode must be class or quality')
        if motion_mode not in ('predicted', 'ego_only'):
            raise ValueError('motion_mode must be predicted or ego_only')
        if cache_nms_iou_threshold is not None and not 0 < cache_nms_iou_threshold <= 1:
            raise ValueError('Cache NMS IoU must be in (0, 1]')
        self.cache_score_mode = cache_score_mode
        self.cache_nms_iou_threshold = cache_nms_iou_threshold
        self.cache_require_current = bool(cache_require_current)
        self.motion_mode = motion_mode
        self.collect_cache_stats = bool(collect_cache_stats)

    def reset(self):
        super().reset()
        self.valid_slots = None
        self.cached_slots_valid = None
        self.cached_pose = None
        self.cached_timestamp = None
        self.cached_scenes = None
        self.cached_tokens = None
        self.cached_padding = None
        self.last_cache_stats = []
        self.last_get_stats = []

    def train(self, mode=True):
        if self.training != mode:
            self.reset()
        return super().train(mode)

    def get(self, batch_size, metas=None, dn_metas=None):
        del dn_metas
        if metas is None or any(key not in metas for key in ('timestamp', 'T_global', 'T_global_inv', 'img_metas')):
            raise ValueError('Temporal memory requires timestamps, ego poses and scene metadata')
        records = metas['img_metas']
        if len(records) != batch_size or any(not row.get('scene_token') for row in records):
            raise ValueError('Temporal memory requires one scene token per batch lane')
        feature = self.instance_feature.unsqueeze(0).expand(batch_size, -1, -1)
        anchor = self.anchor.unsqueeze(0).expand(batch_size, -1, -1)
        interval = anchor.new_full((batch_size,), self.default_time_interval)
        self.mask = torch.zeros(batch_size, dtype=torch.bool, device=anchor.device)
        self.valid_slots = self.mask[:, None].expand(-1, self.num_temp_instances)
        if self.collect_cache_stats:
            self.last_get_stats = [dict(token=str(row['token']), used_slots=0,
                continuity=False, dt_seconds=None) for row in records]
        if self.cached_anchor is None or self.cached_anchor.shape[0] != batch_size:
            return feature, anchor, None, None, interval
        timestamp = torch.as_tensor(metas['timestamp'], device=anchor.device, dtype=torch.float64).reshape(-1)
        dt = timestamp - self.cached_timestamp
        continuity = [str(row['scene_token']) == self.cached_scenes[i]
                      and str(row.get('prev', '')) == self.cached_tokens[i]
                      and not row.get('temporal_is_padding', False)
                      and not self.cached_padding[i]
                      for i, row in enumerate(records)]
        self.mask = (torch.tensor(continuity, device=anchor.device) &
                     torch.isfinite(dt) & (dt > 0) & (dt <= self.max_time_interval))
        interval = torch.where(self.mask, dt.to(anchor.dtype), interval)
        # Subtracting large world translations in FP16/BF16 can erase the
        # entire ego displacement. Geometry stays FP32 even under AMP.
        with torch.autocast(device_type=anchor.device.type, enabled=False):
            transform = (torch.as_tensor(metas['T_global_inv'], device=anchor.device, dtype=torch.float32)
                         @ self.cached_pose.float())
            cached = self.cached_anchor.float().clone()
            # Only XY velocity is supervised. Never extrapolate box height/Z.
            cached[..., 10] = 0.
            safe_dt = torch.where(self.mask, dt, torch.zeros_like(dt)).float()
            motion_dt = torch.zeros_like(safe_dt) if self.motion_mode == 'ego_only' else safe_dt
            propagated = self.anchor_handler.anchor_projection(
                cached, [transform], time_intervals=[-motion_dt])[0].to(anchor.dtype)
        finite = torch.isfinite(propagated).all(dim=-1)
        self.valid_slots = self.mask[:, None] & self.cached_slots_valid & finite
        if self.collect_cache_stats:
            self.last_get_stats = [dict(token=str(row['token']),
                used_slots=int(self.valid_slots[i].sum().item()),
                continuity=bool(self.mask[i].item()), dt_seconds=float(dt[i].item()))
                for i, row in enumerate(records)]
        n = self.num_temp_instances
        selected_feature = torch.where(self.valid_slots[..., None], self.cached_feature, feature[:, :n])
        selected_anchor = torch.where(self.valid_slots[..., None], propagated, anchor[:, :n])
        feature = torch.cat((selected_feature, feature[:, n:]), dim=1)
        anchor = torch.cat((selected_anchor, anchor[:, n:]), dim=1)
        return feature, anchor, selected_feature, selected_anchor, interval

    @torch.no_grad()
    def cache(self, instance_feature, anchor, confidence, metas=None, feature_maps=None,
              quality=None):
        del feature_maps
        class_score, labels = confidence.detach().float().sigmoid().max(dim=-1)
        score = class_score
        quality_score = None
        if self.cache_score_mode == 'quality':
            if quality is None or quality.shape[:2] != score.shape:
                raise ValueError('Quality-ranked cache requires aligned per-query quality logits')
            quality_score = quality.detach().float()[..., CNS].sigmoid()
            score = score * quality_score
        finite = (torch.isfinite(anchor).all(dim=-1) &
                  torch.isfinite(instance_feature).all(dim=-1) & torch.isfinite(score))
        current_eligible = finite & (score >= self.min_confidence)
        score = torch.where(finite, score, torch.zeros_like(score))
        if self.confidence is not None and self.confidence.shape[0] == score.shape[0]:
            n = self.num_temp_instances
            history = self.confidence * self.confidence_decay
            score[:, :n] = torch.where(self.valid_slots, torch.maximum(score[:, :n], history), score[:, :n])
        eligible = current_eligible if self.cache_require_current else finite & (score >= self.min_confidence)
        before_nms = eligible.clone()
        geometry_rejected = torch.zeros_like(eligible)
        if self.cache_nms_iou_threshold is not None:
            from mmcv.ops import nms_rotated
            survivors = torch.zeros_like(eligible)
            for batch_index in range(score.shape[0]):
                # Decode log dimensions/yaw exactly as the output decoder does.
                boxes = SparseBox3DDecoder.decode_box(anchor[batch_index].detach().float())
                geometry_valid = torch.isfinite(boxes).all(dim=-1) & (boxes[:, 3:6] > 0).all(dim=-1)
                geometry_rejected[batch_index] = eligible[batch_index] & ~geometry_valid
                eligible[batch_index] &= geometry_valid
                before_nms[batch_index] = eligible[batch_index]
                for label in labels[batch_index, eligible[batch_index]].unique():
                    members = torch.nonzero(eligible[batch_index] &
                        (labels[batch_index] == label), as_tuple=True)[0]
                    _, keep = nms_rotated(boxes[members][:, [0, 1, 3, 4, 6]].contiguous(),
                        score[batch_index, members].contiguous(),
                        float(self.cache_nms_iou_threshold), clockwise=False)
                    survivors[batch_index, members[keep]] = True
            eligible = survivors
        # Invalid/suppressed rows are padding, never additional cached objects.
        legacy_selection = (self.cache_score_mode == 'class' and
                            self.cache_nms_iou_threshold is None and
                            not self.cache_require_current)
        ranked = (score if legacy_selection else
                  torch.where(eligible, score, score.new_full((), -float('inf'))))
        values, indices = ranked.topk(self.num_temp_instances, dim=1)
        selected_valid = eligible.gather(1, indices)
        feature = instance_feature.detach().gather(
            1, indices[..., None].expand(-1, -1, instance_feature.shape[-1]))
        selected_anchor = anchor.detach().gather(
            1, indices[..., None].expand(-1, -1, anchor.shape[-1]))
        self.cached_feature = (feature if legacy_selection else
            torch.where(selected_valid[..., None], feature, torch.zeros_like(feature)))
        self.cached_anchor = (selected_anchor if legacy_selection else
            torch.where(selected_valid[..., None], selected_anchor, torch.zeros_like(selected_anchor)))
        self.confidence = (values if legacy_selection else
            torch.where(selected_valid, values, torch.zeros_like(values)))
        self.cached_slots_valid = selected_valid
        self.cached_pose = torch.as_tensor(metas['T_global'], device=anchor.device, dtype=torch.float32).detach().clone()
        self.cached_timestamp = torch.as_tensor(metas['timestamp'], device=anchor.device, dtype=torch.float64).detach().reshape(-1).clone()
        self.cached_scenes = [str(row['scene_token']) for row in metas['img_metas']]
        self.cached_tokens = [str(row['token']) for row in metas['img_metas']]
        self.cached_padding = [bool(row.get('temporal_is_padding', False)) for row in metas['img_metas']]
        if self.collect_cache_stats:
            self.last_cache_stats = [dict(token=str(row['token']),
                query_count=int(score.shape[1]), memory_capacity=self.num_temp_instances,
                candidates_current=int(current_eligible[i].sum().item()),
                candidates_before_nms=int(before_nms[i].sum().item()),
                candidates_after_nms=int(eligible[i].sum().item()),
                suppressed_nms=int((before_nms[i] & ~eligible[i]).sum().item()),
                geometry_rejected=int(geometry_rejected[i].sum().item()),
                cached_valid=int(selected_valid[i].sum().item()),
                mean_class_score=float(class_score[i, indices[i, selected_valid[i]]].mean().item())
                    if selected_valid[i].any() else None,
                mean_quality_score=float(quality_score[i, indices[i, selected_valid[i]]].mean().item())
                    if quality_score is not None and selected_valid[i].any() else None)
                for i, row in enumerate(metas['img_metas'])]


@HOOKS.register_module()
class TemporalCacheResetHook(Hook):
    """Do not carry training/validation/test memory across loop boundaries."""

    @staticmethod
    def _reset(runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        model.head.instance_bank.reset()

    def before_train_epoch(self, runner):
        self._reset(runner)

    def before_val_epoch(self, runner):
        self._reset(runner)

    def after_val_epoch(self, runner, metrics=None):
        self._reset(runner)

    def before_test_epoch(self, runner):
        self._reset(runner)


@METRICS.register_module()
class TemporalFisheye3DODMetric(Fisheye3DODReleasedMetricBridge):
    """Exclude explicit DDP lane padding before distributed metric collection."""

    def process(self, data_batch, data_samples):
        super().process(data_batch, [sample for sample in data_samples
                                    if not sample.get('temporal_is_padding', False)])
