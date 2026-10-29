"""Regression checks for DN supervision, rare-class training and decoding."""
from pathlib import Path
from types import SimpleNamespace
import os
import pickle
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'projects/ROSA'), str(ROOT / 'projects/Fisheye3DOD')]
os.environ.setdefault('FISHEYE3DOD_OFFICIAL_ROOT', str(ROOT / 'projects/Fisheye3DOD'))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/rosa-tests-matplotlib')
import numpy as np
for name, value in [('float', float), ('int', int), ('bool', bool)]:
    if name not in np.__dict__:
        setattr(np, name, value)
import torch
from mmengine.dataset import BaseDataset
from mmdet.models.losses import FocalLoss
from mmdet3d.registry import MODELS
import projects.mmdet3d_plugin
from projects.mmdet3d_plugin.models.detection3d.target import SparseBox3DTarget
from projects.mmdet3d_plugin.models.detection3d.decoder import SparseBox3DDecoder
from projects.mmdet3d_plugin.models.sparse4d_head import Sparse4DHead
from rosa.integrations.sparse4d_components import EvaluatorAlignedSparseBox3DLoss
from rosa.integrations.fisheye3dod.balanced_training import (
    CappedClassBalancedDataset, FisheyePhotometricAugmentation, ForegroundBalancedFocalLoss)


def boxes(count):
    result = torch.tensor([[10., 2., -.5, 4., 2., 1.5, 0., 0., 0.]]).repeat(count, 1)
    result[:, 0] += torch.arange(count) * 8
    return result


class BalancedTrainingTests(unittest.TestCase):
    def test_background_is_not_a_dn_regression_target(self):
        head = SimpleNamespace(num_classes=6, reg_weights=[1.] * 10)
        result = Sparse4DHead.prepare_for_dn_loss(head, dict(
            dn_valid_mask=torch.tensor([[True, True, False]]),
            dn_cls_target=torch.tensor([[1, 6, -1]]),
            dn_reg_target=torch.cat((torch.ones(1, 1, 10), torch.zeros(1, 2, 10)), 1)))
        self.assertEqual(result[3].tolist(), [True, False])
        self.assertEqual(result[2].shape, (1, 10))
        self.assertEqual(float(result[-1]), 1.)

    def test_dn_groups_pair_positives_and_negatives(self):
        sampler = SparseBox3DTarget(num_dn_groups=5, max_dn_gt=4,
                                   dn_ignore_gt_overlap=False)
        anchors, targets, labels, mask, valid, _ = sampler.get_dn_anchors(
            [torch.tensor([0, 4]), torch.empty(0, dtype=torch.long)], [boxes(2), boxes(0)])
        for group in range(5):
            start = group * 4
            self.assertTrue((labels[0, start:start + 2] >= 0).all())
            self.assertTrue((labels[0, start + 2:start + 4] == -3).all())
            self.assertFalse(mask[start:start + 4, start:start + 4].any())
            if group < 4:
                self.assertTrue(mask[start:start + 4, start + 4:start + 8].all())
        self.assertFalse(valid[1].any())
        self.assertEqual(anchors.shape, targets.shape)

    def test_van_gets_selected_despite_annotation_order(self):
        sampler = SparseBox3DTarget(max_dn_gt=4, dn_selection='class_balanced',
                                   dn_class_weights=[1., 3., 2., 2., 1.5, 3.])
        labels = torch.tensor([0] * 30 + [1, 4])
        for _ in range(30):
            chosen = labels[sampler._select_dn_gt(labels)].tolist()
            self.assertIn(1, chosen)
            self.assertIn(4, chosen)
        self.assertEqual(len(sampler._select_dn_gt(labels[:0])), 0)

    def test_negative_inside_other_gt_is_ignored(self):
        gt = SparseBox3DTarget.encode_reg_target([boxes(2)])[0]
        inside = SparseBox3DTarget._inside_gt(gt[1:2], gt)
        self.assertTrue(inside.item())
        self.assertFalse(SparseBox3DTarget._inside_gt(gt[1:2] + 100., gt).item())

    def test_foreground_weight_does_not_multiply_background(self):
        logits = torch.tensor([[.5, -.5], [.5, -.5]], requires_grad=True)
        labels = torch.tensor([1, 2])
        ordinary = FocalLoss()(logits, labels, reduction_override='none')
        balanced = ForegroundBalancedFocalLoss([1., 3.])(
            logits, labels, reduction_override='none')
        torch.testing.assert_close(balanced[1], ordinary[1])
        torch.testing.assert_close(balanced[0, 0], ordinary[0, 0])
        torch.testing.assert_close(balanced[0, 1], ordinary[0, 1] * 3)
        balanced.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_empty_frame_has_finite_losses_and_backward(self):
        logits = torch.randn(1, 5, 6, requires_grad=True)
        prediction = torch.randn(1, 5, 10, requires_grad=True)
        quality = torch.randn(1, 5, 2, requires_grad=True)
        head = SimpleNamespace(
            reg_weights=[1.] * 10, sampler=SparseBox3DTarget(),
            gt_cls_key='labels', gt_reg_key='boxes', cls_threshold_to_reg=-1.,
            loss_cls=ForegroundBalancedFocalLoss([1., 3., 2., 2., 1.5, 3.]),
            loss_reg=EvaluatorAlignedSparseBox3DLoss(
                loss_box=dict(type='mmdet.L1Loss'),
                loss_centerness=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True),
                loss_yawness=dict(type='mmdet.GaussianFocalLoss')))
        losses = Sparse4DHead._loss_fp32(head,
            dict(classification=[logits], prediction=[prediction], quality=[quality]),
            dict(labels=[torch.empty(0, dtype=torch.long)], boxes=[boxes(0)]))
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        sum(losses.values()).backward()
        for tensor in [logits, prediction, quality]:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_augmentation_preserves_calibration_and_black_pixels(self):
        images = [np.full((12, 12, 3), 100., dtype=np.float32) for _ in range(4)]
        for image in images:
            image[:2] = 0
        calibration = np.eye(4)
        sample = dict(img=images, lidar2cam=calibration)
        result = FisheyePhotometricAugmentation(probability=1.)(sample)
        self.assertIs(result['lidar2cam'], calibration)
        self.assertTrue(all(np.all(image[:2] == 0) for image in result['img']))
        self.assertTrue(all(image.dtype == np.float32 for image in result['img']))

    def test_decoder_uses_one_class_and_ranks_quality_first(self):
        anchors = torch.zeros(1, 2, 10)
        anchors[0, 1, 0] = 8.
        anchors[..., 7] = 1.
        decoder = SparseBox3DDecoder(num_output=1, single_label=True, quality_before_topk=True)
        result = decoder.decode([torch.tensor([[[10., 9.], [2., 1.]]])], [anchors],
            quality=[torch.tensor([[[-10., 0.], [10., 0.]]])])[0]
        self.assertEqual(float(result['boxes_3d'][0, 0]), 8.)
        self.assertEqual(len(result['scores_3d']), 1)

    def test_vehicle_nms_preserves_pedestrian(self):
        anchors = torch.zeros(1, 3, 10)
        anchors[..., 7] = 1.
        logits = torch.tensor([[[5., -5., -5.], [-5., 4., -5.], [-5., -5., 3.]]])
        decoder = SparseBox3DDecoder(single_label=True, nms_iou_threshold=.5,
                                    nms_class_groups=((0, 1),))
        result = decoder.decode([logits], [anchors])[0]
        self.assertEqual(result['labels_3d'].tolist(), [0, 2])

    def test_instance_ids_stay_aligned_after_threshold(self):
        anchors = torch.zeros(1, 2, 10)
        anchors[..., 7] = 1.
        result = SparseBox3DDecoder(score_threshold=.9).decode(
            [torch.tensor([[[5.], [-5.]]])], [anchors],
            instance_id=torch.tensor([[17, 99]]))[0]
        self.assertEqual(result['instance_ids'].tolist(), [17])

    def test_repeat_cap_retains_every_original_frame(self):
        class ToyDataset(BaseDataset):
            def load_data_list(self):
                return [dict(labels=[0])] * 9 + [dict(labels=[1]), dict(labels=[])]
            def get_cat_ids(self, idx):
                return self.get_data_info(idx)['labels']
        wrapped = CappedClassBalancedDataset(ToyDataset(), oversample_thr=.5, max_repeat=2.)
        counts = np.bincount(wrapped.repeat_indices, minlength=11)
        self.assertTrue((counts >= 1).all())
        self.assertTrue((counts <= 2).all())
        self.assertEqual(counts[-1], 1)

    def test_export_uses_gravity_center_and_dict_token(self):
        import runpy
        import json
        from mmdet3d.structures import LiDARInstance3DBoxes
        exporter = runpy.run_path(str(ROOT / 'projects/ROSA/tools/infer_fisheye_checkpoints.py'))
        original = LiDARInstance3DBoxes(torch.tensor([[2., 3., -1., 4., 2., 2., 0.]]))
        sample = dict(token='123456789', sample_idx=7, pred_instances_3d=dict(
            bboxes_3d=original, scores_3d=torch.tensor([.8]), labels_3d=torch.tensor([0])))
        with tempfile.TemporaryDirectory() as directory:
            raw, output = Path(directory) / 'raw.pkl', Path(directory) / 'predictions.json'
            with raw.open('wb') as stream:
                pickle.dump([sample], stream)
            exporter['export_portable_json'](raw, output, Path('test.pth'), .1)
            payload = json.loads(output.read_text())
        self.assertEqual(payload['box_origin'], 'gravity_center')
        self.assertEqual(payload['frames'][0]['token'], '123456789')
        self.assertEqual(payload['frames'][0]['boxes_3d'][0][2], 0.)
        self.assertEqual(float(original.tensor[0, 2]), -1.)


if __name__ == '__main__':
    unittest.main()
