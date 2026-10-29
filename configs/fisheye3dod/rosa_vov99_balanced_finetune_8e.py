"""Corrected, balanced eight-epoch fine-tune from our epoch-16 detector."""
import os

_base_ = ['./rosa_vov99_dual_export_24e.py']

repo_root = os.path.abspath(os.path.join('{{fileDirname}}', '../../../..'))
default_data_root = os.path.join(repo_root, 'data', 'combined_improved_detzero_rosa')
if not os.path.isdir(default_data_root):
    default_data_root = os.path.join(repo_root, 'data', 'Custom', 'combined_improved_detzero_rosa')
data_root = os.environ.get('ROSA_DATA_ROOT', default_data_root)
load_from = os.environ.get('ROSA_INIT_CKPT', os.path.join(
    repo_root, 'projects', 'ROSA', 'weights', 'best_fisheye_bev_NDS_epoch_16.pth'))
resume = False  # Fresh optimizer/schedule for the corrected objective.

custom_imports = dict(allow_failed_imports=False, imports=[
    'projects.mmdet3d_plugin', 'rosa.integrations.sparse4d_components',
    'rosa.integrations.optim', 'rosa.integrations.fisheye3dod.dataset',
    'rosa.integrations.fisheye3dod.model', 'rosa.integrations.fisheye3dod.metric',
    'rosa.integrations.fisheye_ray_sparse4d', 'rosa.integrations.fisheye_box_sparse4d',
    'rosa.integrations.kitti360.backbone',
    'rosa.integrations.fisheye3dod.balanced_training',
])

class_weights = [1., 3., 2., 2., 1.5, 3.]  # Car, Van, Truck, Bus, Pedestrian, Cyclist
model = dict(
    img_backbone=dict(frozen_stages=1, norm_eval=True),
    head=dict(
        cls_threshold_to_reg=-1.,  # Every matched rare-class box gets regression.
        dn_loss_weight=1.,
        decoder=dict(single_label=True, quality_before_topk=True),
        instance_bank=dict(anchor=os.path.join(data_root, 'rosa/anchors/kmeans300_seed3407.npy')),
        loss_cls=dict(_delete_=True, type='ForegroundBalancedFocalLoss',
                      class_weights=class_weights, use_sigmoid=True,
                      alpha=.25, gamma=2., loss_weight=2.),
        sampler=dict(max_dn_gt=4, dn_selection='class_balanced',
                     dn_class_weights=class_weights, dn_box_relative_noise=True,
                     dn_ignore_gt_overlap=True,
                     dn_noise_scale=[.5, .5, .25, .2, .2, .2, .2, .2, 0., 0.]),
    ),
)

train_pipeline = [
    dict(type='LoadOfficialFisheye3DODForSparse4D', to_float32=True),
    dict(type='FisheyePhotometricAugmentation'),
    dict(type='LoadAnnotations3D', with_bbox=False, with_bbox_3d=True,
         with_label=False, with_label_3d=True),
    dict(type='Fisheye3DODEvaluatorRangeFilter'),
    dict(type='PackSparse4DInputs', ocam_lens_fov_deg=220.),
]
train_dataloader = dict(
    batch_size=1, num_workers=4,
    dataset=dict(_delete_=True, type='CappedClassBalancedDataset',
        oversample_thr=.2, max_repeat=3., seed=3407,
        dataset=dict(type='Fisheye3DODSparse4DDataset',
            ann_file=os.path.join(data_root, 'rosa/fisheye3dod_infos_train.pkl'),
            data_root=data_root, filter_empty_gt=False, test_mode=False,
            with_velocity=True, pipeline=train_pipeline,
            metainfo=dict(classes=('Car', 'Van', 'Truck', 'Bus', 'Pedestrian', 'Cyclist')))),
)
val_dataloader = dict(batch_size=1, dataset=dict(
    data_root=data_root, ann_file=os.path.join(data_root, 'rosa/fisheye3dod_infos_val.pkl')))
test_dataloader = dict(batch_size=1, dataset=dict(
    data_root=data_root, ann_file=os.path.join(data_root, 'rosa/fisheye3dod_infos_val.pkl')))

optim_wrapper = dict(
    accumulative_counts=4,  # 1 frame/GPU * 2 GPUs * 4 accumulation = 8 frames.
    optimizer=dict(lr=5e-5),
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=.1)}),
)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=8, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=.1, begin=0, end=100, by_epoch=False),
    dict(type='CosineAnnealingLR', T_max=8, begin=0, end=8, by_epoch=True,
         convert_to_iter_based=True, eta_min_ratio=.05),
]
find_unused_parameters = False
custom_hooks = [dict(type='SyncBuffersHook', priority='NORMAL')]
default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=3,
    save_best=['fisheye_bev/mAP', 'fisheye_bev/NDS'], rule=['greater', 'greater']))
work_dir = os.path.join(repo_root, 'work_dirs/rosa_vov99_balanced_finetune_8e')
experiment_contract = dict(
    _delete_=True,
    training_recipe='balanced_v1', initialization=load_from,
    dataset='16_09 + 30_09 combined improved DetZero', backbone='VoVNet-99',
    classes=('Car', 'Van', 'Truck', 'Bus', 'Pedestrian', 'Cyclist'),
    split='existing temporal-block train/validation split',
    corrected_dn_background_regression=True, corrected_dn_group_mask=True,
    dn_selection='randomized class-stratified four GT',
    class_weights=class_weights, retain_empty_frames=True,
    augmentation='photometric only; fisheye geometry unchanged',
    temporal=False,
)
