"""Refresh the best adapted single-frame weights on the latest combined data."""
import os

_base_ = ['./rosa_vov99_baseline_adapt_8e.py']
repo_root = os.path.abspath(os.path.join('{{fileDirname}}', '../../../..'))
load_from = os.environ.get('ROSA_INIT_CKPT', os.path.join(
    repo_root, 'projects/ROSA/work_dirs/server/rosa_vov99_baseline_adapt_3080ti_2gpu',
    'best_fisheye_bev_NDS_epoch_4.pth'))
resume = False
custom_imports = dict(allow_failed_imports=False, imports=[
    'projects.mmdet3d_plugin', 'rosa.integrations.sparse4d_components',
    'rosa.integrations.optim', 'rosa.integrations.fisheye3dod.dataset',
    'rosa.integrations.fisheye3dod.model', 'rosa.integrations.fisheye3dod.metric',
    'rosa.integrations.fisheye_ray_sparse4d', 'rosa.integrations.fisheye_box_sparse4d',
    'rosa.integrations.kitti360.backbone',
    'rosa.integrations.fisheye3dod.balanced_training',
    'rosa.integrations.fisheye3dod.singleframe_plateau',
])
# These are baseline settings, not temporal weak-velocity regression or memory.
model = dict(
    img_backbone=dict(frozen_stages=-1, norm_eval=False),
    head=dict(instance_bank=dict(type='InstanceBank', num_temp_instances=0),
              reg_weights=[2., 2., 2., 1., 1., 1., 1., 1., 1., 1.]))
optim_wrapper = dict(accumulative_counts=8, optimizer=dict(lr=5e-5),
                     paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=1.)}))
train_dataloader = dict(batch_size=1, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=2)
test_dataloader = dict(batch_size=1, num_workers=2)
train_cfg = dict(max_epochs=24, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=.1, begin=0, end=100, by_epoch=False),
    dict(type='CosineAnnealingLR', T_max=24, begin=0, end=24, by_epoch=True,
         convert_to_iter_based=True, eta_min_ratio=.05),
]
custom_hooks = [dict(type='SyncBuffersHook', priority='NORMAL'),
                dict(type='SingleFramePlateauHook', patience=4, min_delta=1e-4)]
default_hooks = dict(logger=dict(interval=50), checkpoint=dict(
    interval=1, max_keep_ckpts=3,
    save_best=['fisheye_bev/mAP', 'fisheye_bev/NDS'], rule=['greater', 'greater']))
work_dir = os.path.join(repo_root, 'projects/ROSA/work_dirs/singleframe_refresh_20261031')
experiment_contract = dict(
    training_recipe='E007_singleframe_refresh_v1', initialization=load_from,
    temporal=False, temporal_slots=0, motion_targets='baseline zero-velocity ABI only',
    dataset='latest combined dataset; existing chronological/purged split unchanged',
    head_peak_lr=5e-5, backbone_peak_lr=5e-5,
    per_gpu_batch=1, effective_batch=8,
    plateau='neither mAP nor NDS improves >1e-4 for four validations',
    max_epochs=24,
)
