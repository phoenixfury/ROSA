"""Temporal adaptation on acquisition-time, track-labelled custom bags.

100 detached query slots are propagated into the next observation, leaving
200 fresh queries. Shared decoder self-attention/image evidence refines both.
This is recurrent query seeding, not the retired Sparse4D temp-GNN path.
"""
import os

_base_ = ['./rosa_vov99_baseline_adapt_8e.py']

repo_root = os.path.abspath(os.path.join('{{fileDirname}}', '../../../..'))
data_root = os.environ.get('ROSA_DATA_ROOT', os.path.join(
    repo_root, 'data/Custom/combined_improved_detzero_rosa'))
default_checkpoint = os.path.join(repo_root, 'projects/ROSA/work_dirs/server',
    'rosa_vov99_baseline_adapt_3080ti_2gpu/best_fisheye_bev_NDS_epoch_4.pth')
load_from = os.environ.get('ROSA_INIT_CKPT', default_checkpoint)
resume = False

custom_imports = dict(allow_failed_imports=False, imports=[
    'projects.mmdet3d_plugin', 'rosa.integrations.sparse4d_components',
    'rosa.integrations.optim', 'rosa.integrations.fisheye3dod.dataset',
    'rosa.integrations.fisheye3dod.model', 'rosa.integrations.fisheye3dod.metric',
    'rosa.integrations.fisheye_ray_sparse4d', 'rosa.integrations.fisheye_box_sparse4d',
    'rosa.integrations.kitti360.backbone',
    'rosa.integrations.fisheye3dod.balanced_training',
    'rosa.integrations.fisheye3dod.temporal',
])

model = dict(
    data_preprocessor=dict(type='Sparse4DDataPreprocessor'),
    head=dict(temporal_enabled=True,
        # Motion is weak supervision derived from DetZero pseudo tracks.
        reg_weights=[2., 2., 2., 1., 1., 1., 1., 1., .2, .2],
        instance_bank=dict(
        type='TemporalInstanceBank', num_temp_instances=100,
        min_confidence=.15, max_time_interval=1.5, confidence_decay=.6,
        anchor=os.path.join(data_root, 'rosa/anchors/kmeans300_seed3407.npy'))),
)

# No frame oversampling or frame shuffle: each lane walks a chronological
# clip. Rare classes retain weighted classification and stratified DN losses.
train_pipeline = [
    dict(type='LoadOfficialFisheye3DODForSparse4D', to_float32=True),
    dict(type='FisheyePhotometricAugmentation'),
    dict(type='LoadAnnotations3D', with_bbox=False, with_bbox_3d=True,
         with_label=False, with_label_3d=True),
    dict(type='Fisheye3DODEvaluatorRangeFilter'),
    dict(type='PackSparse4DInputs', ocam_lens_fov_deg=220.),
]
train_dataloader = dict(
    batch_size=1, num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=dict(type='TemporalSequenceBatchSampler', clip_length=32,
                       shuffle=True, seed=3407),
    dataset=dict(_delete_=True, type='TemporalFisheye3DODDataset',
        ann_file=os.path.join(data_root, 'rosa/fisheye3dod_infos_train.pkl'),
        data_root=data_root, filter_empty_gt=False, test_mode=False,
        with_velocity=True, pipeline=train_pipeline,
        metainfo=dict(classes=('Car', 'Van', 'Truck', 'Bus', 'Pedestrian', 'Cyclist'))),
)
val_dataloader = dict(
    batch_size=1, sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=dict(type='TemporalSequenceBatchSampler', clip_length=0,
                       shuffle=False, seed=3407),
    dataset=dict(type='TemporalFisheye3DODDataset', data_root=data_root,
        ann_file=os.path.join(data_root, 'rosa/fisheye3dod_infos_val.pkl')),
)
test_dataloader = val_dataloader
val_evaluator = dict(type='TemporalFisheye3DODMetric', prefix='fisheye_bev')
test_evaluator = val_evaluator
custom_hooks = [dict(type='SyncBuffersHook', priority='NORMAL'),
                dict(type='TemporalCacheResetHook', priority='HIGH')]
optim_wrapper = dict(accumulative_counts=8,
    optimizer=dict(lr=5e-5),
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=1.)}))
train_cfg = dict(max_epochs=12, val_interval=1)
default_hooks = dict(logger=dict(interval=10))
param_scheduler = [
    dict(type='LinearLR', start_factor=.1, begin=0, end=200, by_epoch=False),
    dict(type='CosineAnnealingLR', T_max=12, begin=0, end=12, by_epoch=True,
         convert_to_iter_based=True, eta_min_ratio=.05),
]
find_unused_parameters = False
work_dir = os.path.join(repo_root, 'projects/ROSA/work_dirs/rosa_vov99_temporal_12e')
experiment_contract = dict(
    training_recipe='temporal_query_memory_v1', initialization=load_from,
    dataset='16_09 + 17_09 + 30_09 acquisition-time improved DetZero',
    temporal=True, temporal_slots=100, temporal_max_interval_seconds=1.5,
    temporal_update='detached recurrent query seeding; existing shared decoder attention',
    train_sampling='chronological 32-frame clips; shuffle clips, not frames',
    split='per-bag chronological time holdout with purge gap; split-local scene links',
    velocity_supervision='causal track-derived world displacement in current LiDAR axes',
    weak_motion_targets=True, num_temp_dn_groups=0,
    head_peak_lr=5e-5, backbone_peak_lr=5e-5,
)
