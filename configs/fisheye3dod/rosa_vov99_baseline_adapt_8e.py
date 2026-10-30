"""Adapt baseline epoch 16 with live BN statistics and restored backbone LR.

Fresh optimizer/schedule, NOT a resume of the old epoch-16 optimizer. The
existing corrected DN / balanced supervision / photometric recipe is retained.
Validation and deployment both use class-wise rotated BEV NMS at IoU 0.50.
"""
import os

_base_ = ['./rosa_vov99_baseline_class_nms.py']
repo_root = os.path.abspath(os.path.join('{{fileDirname}}', '../../../..'))
load_from = os.environ.get('ROSA_INIT_CKPT', os.path.join(
    repo_root, 'projects', 'ROSA', 'weights', 'best_fisheye_bev_NDS_epoch_16.pth'))
resume = False

# Unfreeze every stage: frozen_stages=1 would still freeze early-stage BN.
model = dict(img_backbone=dict(frozen_stages=-1, norm_eval=False))
optim_wrapper = dict(
    optimizer=dict(lr=1e-4),
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=1.)}),
)
train_cfg = dict(max_epochs=8, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=.1, begin=0, end=100, by_epoch=False),
    dict(type='CosineAnnealingLR', T_max=8, begin=0, end=8, by_epoch=True,
         convert_to_iter_based=True, eta_min_ratio=.05),
]
work_dir = os.path.join(repo_root, 'work_dirs/rosa_vov99_baseline_adapt_8e')
experiment_contract = dict(
    training_recipe='baseline_epoch16_bn_lr_adaptation_v1',
    initialization=load_from,
    backbone_norm_eval=False, backbone_frozen_stages=-1,
    head_peak_lr=1e-4, backbone_peak_lr=1e-4,
    batchnorm_checkpoint_recompute='private buffers; one running-stat update per forward',
    decoder='single-label, quality before top-k, class-wise rotated BEV NMS',
    nms_iou_threshold=.5, nms_class_groups=(),
    validation_uses_nms=True,
)
