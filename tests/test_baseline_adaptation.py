"""CPU checks for baseline adaptation, class-wise NMS and live checkpointed BN."""
import copy
import sys
import unittest

from test_balanced_training import ROOT, SparseBox3DDecoder, torch
from mmengine.config import Config
from mmdet3d.registry import MODELS
import rosa.integrations.kitti360.backbone

VOVNET = sys.modules['fisheye3dod.models.petr.vovnetcp']


class BaselineAdaptationTests(unittest.TestCase):
    def test_portable_config_and_training_contract(self):
        config = Config.fromfile(str(ROOT / 'projects/ROSA/configs/fisheye3dod/'
                                     'rosa_vov99_baseline_adapt_8e.py'))
        self.assertEqual(config.model.img_backbone.frozen_stages, -1)
        self.assertFalse(config.model.img_backbone.norm_eval)
        self.assertTrue(config.load_from.endswith('best_fisheye_bev_NDS_epoch_16.pth'))
        self.assertFalse(config.resume)
        self.assertFalse(config.find_unused_parameters)
        self.assertEqual(config.train_cfg.max_epochs, 8)
        lr = config.optim_wrapper.optimizer.lr
        self.assertEqual(lr, 1e-4)
        self.assertEqual(lr * config.optim_wrapper.paramwise_cfg.custom_keys.img_backbone.lr_mult, 1e-4)
        self.assertEqual(config.param_scheduler[1].T_max, 8)
        decoder = config.model.head.decoder
        self.assertTrue(decoder.single_label)
        self.assertTrue(decoder.quality_before_topk)
        self.assertEqual(decoder.nms_iou_threshold, .5)
        self.assertEqual(decoder.nms_class_groups, ())
        self.assertIsNone(decoder.score_threshold)
        self.assertEqual(config.train_dataloader.dataset.dataset.pipeline,
                         config.train_pipeline)
        backbone = MODELS.build(config.model.img_backbone)
        backbone.train()
        bn = [m for m in backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        self.assertTrue(bn)
        self.assertTrue(all(m.training for m in bn))
        self.assertTrue(all(p.requires_grad for p in backbone.parameters()))
        backbone.eval()
        self.assertTrue(all(not m.training for m in bn))

    def test_class_nms_suppresses_only_same_class(self):
        anchors = torch.zeros(1, 3, 10)
        anchors[..., 7] = 1.
        logits = torch.tensor([[[5., -5.], [4., -5.], [-5., 3.]]])
        decoder = SparseBox3DDecoder(single_label=True, quality_before_topk=True,
            nms_iou_threshold=.5, nms_class_groups=())
        result = decoder.decode([logits], [anchors])[0]
        self.assertEqual(result['labels_3d'].tolist(), [0, 1])

    def test_bn_updates_once_and_gradients_match_without_checkpointing(self):
        torch.set_num_threads(1)
        for momentum in (.1, None):
            plain = VOVNET._OSA_module(8, 4, 8, 2, 'bn_probe', SE=True, with_cp=False)
            for module in plain.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.momentum = momentum
            checked = copy.deepcopy(plain)
            checked.use_checkpoint = True
            original_buffers = {n: b for n, b in checked.named_buffers()}
            for step in range(1, 4):
                image = torch.randn(2, 8, 8, 8)
                first = image.clone().requires_grad_()
                second = image.clone().requires_grad_()
                plain.zero_grad()
                checked.zero_grad()
                expected = plain(first)
                actual = checked(second)
                torch.testing.assert_close(actual, expected)
                expected.square().mean().backward()
                actual.square().mean().backward()
                torch.testing.assert_close(first.grad, second.grad)
                for (name, p), (_, q) in zip(plain.named_parameters(), checked.named_parameters()):
                    torch.testing.assert_close(p.grad, q.grad, msg=name)
                for name, buffer in checked.named_buffers():
                    self.assertIs(buffer, original_buffers[name])
                    torch.testing.assert_close(buffer, dict(plain.named_buffers())[name])
                    if name.endswith('num_batches_tracked'):
                        self.assertEqual(int(buffer), step)
            checked.eval()
            before = {n: b.clone() for n, b in checked.named_buffers()}
            with torch.no_grad():
                checked(image)
            for name, buffer in checked.named_buffers():
                torch.testing.assert_close(buffer, before[name])

    def test_recompute_buffer_restore_on_exception(self):
        bn = torch.nn.BatchNorm2d(4)
        originals = dict(bn.named_buffers())
        with self.assertRaisesRegex(RuntimeError, 'test exception'):
            with VOVNET._private_batchnorm_recompute_buffers(bn):
                bn(torch.randn(2, 4, 8, 8))
                raise RuntimeError('test exception')
        for name, buffer in bn.named_buffers():
            self.assertIs(buffer, originals[name])
        self.assertEqual(int(bn.num_batches_tracked), 0)


if __name__ == '__main__':
    unittest.main()
