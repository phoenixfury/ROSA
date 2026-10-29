"""Two CPU ranks exercise checkpointing plus empty/positive DN loss branches."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile

from test_balanced_training import ROOT, boxes
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from projects.mmdet3d_plugin.models.sparse4d_head import Sparse4DHead
from projects.mmdet3d_plugin.models.detection3d.target import SparseBox3DTarget
from rosa.integrations.sparse4d_components import EvaluatorAlignedSparseBox3DLoss
from rosa.integrations.fisheye3dod.balanced_training import ForegroundBalancedFocalLoss
import rosa.integrations.kitti360.backbone


class CheckpointedTrainingProbe(nn.Module):
    def __init__(self):
        super().__init__()
        osa = sys.modules['fisheye3dod.models.petr.vovnetcp']._OSA_module
        self.backbone = osa(8, 4, 8, 2, 'ddp_probe', SE=True)
        self.classifier = nn.Linear(8, 4 * 6)
        self.regressor = nn.Linear(8, 4 * 10)
        self.quality = nn.Linear(8, 4 * 2)
        self.loss_cls = ForegroundBalancedFocalLoss([1., 3., 2., 2., 1.5, 3.])
        self.loss_reg = EvaluatorAlignedSparseBox3DLoss(
            loss_box=dict(type='mmdet.L1Loss'),
            loss_centerness=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True),
            loss_yawness=dict(type='mmdet.GaussianFocalLoss'))
        self.sampler = SparseBox3DTarget(num_dn_groups=1, max_dn_gt=2,
                                       dn_selection='class_balanced')
        self.reg_weights = [1.] * 10
        self.num_classes = 6
        self.gt_cls_key, self.gt_reg_key = 'labels', 'boxes'
        self.cls_threshold_to_reg = -1.
        self.dn_loss_weight = 1.

    def prepare_for_dn_loss(self, model_outs):
        return Sparse4DHead.prepare_for_dn_loss(self, model_outs)

    def forward(self, image, rank):
        feature = self.backbone(image).mean((2, 3))
        cls = self.classifier(feature).reshape(1, 4, 6)
        reg = self.regressor(feature).reshape(1, 4, 10)
        quality = self.quality(feature).reshape(1, 4, 2)
        labels = [torch.tensor([1]) if rank else torch.empty(0, dtype=torch.long)]
        gt = [boxes(1) if rank else boxes(0)]
        outputs = dict(classification=[cls], prediction=[reg], quality=[quality])
        dn = self.sampler.get_dn_anchors(labels, gt)
        if dn is not None:
            _, target, dn_labels, _, valid, _ = dn
            outputs.update(dn_classification=[cls[:, :2]], dn_prediction=[reg[:, :2]],
                dn_reg_target=target, dn_cls_target=torch.where(dn_labels == -3, 6, dn_labels),
                dn_valid_mask=valid)
        losses = Sparse4DHead._loss_fp32(self, outputs, dict(labels=labels, boxes=gt))
        keys = [None, None]
        dist.all_gather_object(keys, list(losses))
        assert keys[0] == keys[1], keys
        return sum(losses.values())


def worker(rank, rendezvous, output_dir):
    torch.set_num_threads(1)
    torch.manual_seed(3407)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    model = DistributedDataParallel(CheckpointedTrainingProbe(), find_unused_parameters=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        loss = model(torch.randn(1, 8, 8, 8), rank)
        assert torch.isfinite(loss), loss
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        optimizer.step()
        losses.append(float(loss))
    Path(output_dir, f'rank{rank}.json').write_text(json.dumps(losses))
    dist.destroy_process_group()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(worker, args=(str(Path(directory, 'rendezvous')), directory), nprocs=2, join=True)
        print('DDP passed: checkpointed backbone, find_unused_parameters=False, '
              'rank 0 empty GT, rank 1 rare-class GT + positive/negative DN, 3 steps/rank.')
