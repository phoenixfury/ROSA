"""Temporal motion, stream isolation, DDP padding and unknown-GT regression."""
from types import SimpleNamespace
import unittest

from test_balanced_training import ROOT, np, torch, SparseBox3DTarget, Sparse4DHead
from mmengine.config import Config
from rosa.integrations.fisheye3dod.temporal import (
    TemporalInstanceBank, TemporalSequenceBatchSampler,
    TemporalFisheye3DODMetric, track_velocities,
)
from rosa.integrations.fisheye3dod.dataset import Fisheye3DODEvaluatorRangeFilter
from mmdet3d.structures import LiDARInstance3DBoxes


def metadata(token='f0', previous='', scene='bag_train_000', timestamp=1_789_000_000., pose=None):
    pose = torch.eye(4) if pose is None else pose
    return dict(timestamp=torch.tensor([timestamp], dtype=torch.float64),
        T_global=pose[None], T_global_inv=torch.linalg.inv(pose)[None],
        img_metas=[dict(token=token, prev=previous, scene_token=scene,
                        temporal_is_padding=False)])


def bank():
    anchors = np.zeros((4, 11), dtype=np.float32)
    anchors[:, 0] = np.arange(4) + 5.
    anchors[:, 7] = 1.
    return TemporalInstanceBank(num_anchor=4, embed_dims=8, anchor=anchors,
        num_temp_instances=2, min_confidence=.1, max_time_interval=1.5,
        anchor_handler=dict(type='SparseBox3DKeyPointsGenerator'))


class TemporalTrainingTests(unittest.TestCase):
    def test_memory_compensates_ego_turn_motion_and_preserves_z(self):
        memory = bank()
        memory.get(1, metadata())
        feature = torch.arange(32.).reshape(1, 4, 8).requires_grad_()
        anchors = memory.anchor[None].clone()
        anchors[0, 0, :3] = torch.tensor([10., 0., 1.])
        anchors[0, 0, 8:11] = torch.tensor([2., 0., 99.])
        logits = torch.tensor([[[6.], [5.], [-9.], [-9.]]])
        memory.cache(feature, anchors, logits, metadata())
        self.assertIsNone(memory.cached_feature.grad_fn)
        pose = torch.eye(4)
        pose[:2, :2] = torch.tensor([[0., -1.], [1., 0.]])
        pose[0, 3] = 1.
        _, propagated, _, _, dt = memory.get(1, metadata(
            'f1', 'f0', timestamp=1_789_000_000.55, pose=pose))
        self.assertTrue(memory.valid_slots.all())
        torch.testing.assert_close(propagated[0, 0, :3], torch.tensor([0., -10.1, 1.]), atol=1e-5, rtol=0)
        torch.testing.assert_close(propagated[0, 0, 6:8], torch.tensor([-1., 0.]))
        torch.testing.assert_close(propagated[0, 0, 8:11], torch.tensor([0., -2., 0.]))
        self.assertAlmostEqual(float(dt[0]), .55, places=5)

    def test_memory_rejects_scene_predecessor_time_and_padding_mismatches(self):
        for changes in (dict(scene='other'), dict(previous='not_f0'),
                        dict(timestamp=1_789_000_003.),
                        dict(timestamp=1_788_999_999.5),
                        dict(timestamp=1_789_000_000.)):
            memory = bank()
            memory.get(1, metadata())
            memory.cache(torch.ones(1, 4, 8), memory.anchor[None],
                         torch.ones(1, 4, 1)*5., metadata())
            options = dict(token='f1', previous='f0', timestamp=1_789_000_000.5)
            options.update(changes)
            feature, anchors, _, _, _ = memory.get(1, metadata(**options))
            self.assertFalse(memory.valid_slots.any())
            torch.testing.assert_close(anchors, memory.anchor[None])
        padded = metadata('f1', 'f0', timestamp=1_789_000_000.5)
        padded['img_metas'][0]['temporal_is_padding'] = True
        memory.get(1, padded)
        self.assertFalse(memory.valid_slots.any())
        memory.eval()
        self.assertIsNone(memory.cached_feature)

    def test_batch_lanes_reset_independently(self):
        memory = bank()
        first = metadata()
        for key in ('timestamp', 'T_global', 'T_global_inv'):
            first[key] = first[key].repeat(2, *([1]*(first[key].ndim-1)))
        first['img_metas'] = [first['img_metas'][0],
                              dict(token='g0', prev='', scene_token='other', temporal_is_padding=False)]
        memory.get(2, first)
        memory.cache(torch.ones(2, 4, 8), memory.anchor[None].repeat(2, 1, 1),
                     torch.ones(2, 4, 1)*5., first)
        second = dict(first)
        second['timestamp'] = first['timestamp']+.5
        second['img_metas'] = [dict(token='f1', prev='f0', scene_token='bag_train_000'),
                               dict(token='h0', prev='', scene_token='new_bag')]
        memory.get(2, second)
        self.assertEqual(memory.mask.tolist(), [True, False])
        self.assertTrue(memory.valid_slots[0].all())
        self.assertFalse(memory.valid_slots[1].any())

    def test_amp_does_not_erase_ego_translation_at_large_world_coordinates(self):
        memory = bank()
        first_pose = torch.eye(4)
        first_pose[0, 3] = 10_000.
        first = metadata(pose=first_pose)
        memory.get(1, first)
        memory.cache(torch.ones(1, 4, 8), memory.anchor[None],
                     torch.ones(1, 4, 1)*5., first)
        current_pose = first_pose.clone()
        current_pose[0, 3] += 1.
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            _, anchors, _, _, _ = memory.get(1, metadata(
                'f1', 'f0', timestamp=1_789_000_000.5, pose=current_pose))
        # Equal scores do not prescribe top-k order, so use the cached center.
        expected = memory.cached_anchor[0, :, 0] - 1.
        torch.testing.assert_close(anchors[0, :2, 0], expected)

    def test_parked_object_is_zero_world_velocity_while_ego_turns(self):
        samples = []
        for index in range(2):
            pose = np.eye(4)
            if index:
                pose[:2, :2] = [[0., -1.], [1., 0.]]
                pose[0, 3] = 1.
            world = np.array([10. + index, 0., 0.])
            parked = pose[:3, :3].T @ (np.array([10., 0., 0.])-pose[:3, 3])
            moving = pose[:3, :3].T @ (world-pose[:3, 3])
            samples.append(dict(token=str(index), metainfo=dict(scene_name='train',
                acquisition_timestamp_ns=1_000_000_000 + index*500_000_000,
                T_world_CLIDAR=pose), annos=dict(gt_ids=[1, 2],
                    gt_boxes=[[*parked, 4., 2., 1.5, 0.], [*moving, 4., 2., 1.5, 0.]])))
        velocities = track_velocities(samples)
        self.assertTrue(np.isnan(velocities['0']).all())
        np.testing.assert_allclose(velocities['1'][0], [0., 0.])
        np.testing.assert_allclose(velocities['1'][1], [0., -2.])
        samples[1]['metainfo']['scene_name'] = 'val'
        self.assertTrue(np.isnan(track_velocities(samples)['1']).all())

    def test_ddp_streams_visit_real_frames_once_and_mark_tail_padding(self):
        records = [dict(index=i, scene=f'scene_{i//12}', timestamp_ns=i*500_000_000)
                   for i in range(36)]
        source = SimpleNamespace(dataset=SimpleNamespace(temporal_records=records), epoch=0)
        samplers = [TemporalSequenceBatchSampler(source, batch_size=2,
                    clip_length=4, rank=rank, world_size=2) for rank in range(2)]
        actual = []
        self.assertEqual(len(samplers[0]), len(samplers[1]))
        for sampler in samplers:
            batches = list(sampler)
            self.assertEqual(len(batches), len(sampler))
            for lane in range(2):
                padded = False
                for batch in batches:
                    index, padding = batch[lane]
                    if padding:
                        padded = True
                    else:
                        self.assertFalse(padded)
                        actual.append(index)
        self.assertEqual(sorted(actual), list(range(36)))

    def test_unknown_velocity_does_not_poison_assignment_or_dn(self):
        sampler = SparseBox3DTarget(num_dn_groups=1, max_dn_gt=1,
                                    dn_ignore_gt_overlap=False)
        boxes = torch.tensor([[10., 0., 0., 4., 2., 1.5, 0., float('nan'), float('nan')]])
        encoded = sampler.encode_reg_target([boxes])[0]
        prediction = torch.nan_to_num(encoded)[None].repeat(1, 2, 1)
        prediction[0, 0, 0] += 20.
        labels, targets, weight = sampler.sample(torch.zeros(1, 2, 6), prediction,
                                                 [torch.tensor([0])], [boxes])
        self.assertEqual(int(labels[0, 1]), 0)
        self.assertTrue((weight[0, 1, 8:] == 0).all())
        anchors, target, cls, _, valid, _ = sampler.get_dn_anchors([torch.tensor([0])], [boxes])
        self.assertTrue(torch.isfinite(anchors).all())
        cls = torch.where(cls == -3, 6, cls)
        fake = SimpleNamespace(num_classes=6, reg_weights=[1.]*10)
        output = Sparse4DHead.prepare_for_dn_loss(fake, dict(
            dn_valid_mask=valid, dn_cls_target=cls, dn_reg_target=target))
        self.assertTrue(torch.isfinite(output[2]).all())
        self.assertTrue((output[4][:, 8:] == 0).all())

    def test_metric_does_not_count_padding(self):
        metric = TemporalFisheye3DODMetric(prefix='fisheye_bev')
        sample = dict(token='a', sample_idx='a', eval_ann_info={}, pred_instances_3d={})
        metric.process({}, [sample, dict(sample, temporal_is_padding=True)])
        self.assertEqual(len(metric.results), 1)

    def test_range_filter_aligns_loaded_boxes_labels_and_track_ids(self):
        boxes = LiDARInstance3DBoxes(torch.tensor([
            [10., 0., 0., 4., 2., 1.5, 0., 0., 0.],
            [55., 0., 0., 4., 2., 1.5, 0., 0., 0.]]), box_dim=9)
        annotation = dict(gt_bboxes_3d=boxes, gt_labels_3d=np.array([0, 0]),
                          instance_inds=np.array([77, 88]))
        results = dict(ann_info=annotation, **annotation)
        output = Fisheye3DODEvaluatorRangeFilter().transform(results)
        self.assertEqual(len(output['gt_bboxes_3d']), 1)
        self.assertEqual(output['gt_labels_3d'].tolist(), [0])
        self.assertEqual(output['ann_info']['instance_inds'].tolist(), [77])
        self.assertEqual(output['instance_inds'].tolist(), [77])

    def test_config_uses_temporal_data_and_not_repeated_frame_sampling(self):
        config = Config.fromfile(str(ROOT/'projects/ROSA/configs/fisheye3dod/rosa_vov99_temporal_12e.py'))
        self.assertTrue(config.model.head.temporal_enabled)
        self.assertEqual(config.model.head.instance_bank.num_temp_instances, 100)
        self.assertEqual(config.train_dataloader.dataset.type, 'TemporalFisheye3DODDataset')
        self.assertEqual(config.train_dataloader.batch_sampler.clip_length, 32)
        self.assertFalse(config.train_dataloader.sampler.shuffle)
        self.assertEqual(config.model.data_preprocessor.type, 'Sparse4DDataPreprocessor')
        self.assertEqual(config.test_evaluator.type, 'TemporalFisheye3DODMetric')


if __name__ == '__main__':
    unittest.main()
