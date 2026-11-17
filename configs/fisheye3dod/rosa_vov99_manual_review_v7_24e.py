"""Single-frame ROSA: native v7 images, resized to 800x800 in the loader."""
import os

_base_ = ['./rosa_vov99_singleframe_refresh_24e.py']
repo_root = os.path.abspath(os.path.join('{{fileDirname}}', '../../../..'))
default_data_root = os.path.join(repo_root, 'data/combined_bags_camera_ray_dedup_v2')
if not os.path.isdir(default_data_root):
    default_data_root = os.path.join(repo_root, 'data/Custom/combined_bags_camera_ray_dedup_v2')
data_root = os.environ.get('ROSA_DATA_ROOT', default_data_root)
annotation_root = os.environ.get('ROSA_ANN_ROOT', os.path.join(
    repo_root, 'projects/raybev_lite/datasets/combined_bags_camera_ray_manual_review_v7'))
calibration = os.path.join(repo_root, 'projects/raybev_lite/configs/calibration_baselink.json')
custom_imports = dict(allow_failed_imports=False, imports=[
    'projects.mmdet3d_plugin', 'rosa.integrations.sparse4d_components',
    'rosa.integrations.optim', 'rosa.integrations.fisheye3dod.dataset',
    'rosa.integrations.fisheye3dod.model', 'rosa.integrations.fisheye3dod.metric',
    'rosa.integrations.fisheye_ray_sparse4d', 'rosa.integrations.fisheye_box_sparse4d',
    'rosa.integrations.kitti360.backbone',
    'rosa.integrations.fisheye3dod.balanced_training',
    'rosa.integrations.fisheye3dod.singleframe_plateau',
    'rosa.integrations.fisheye3dod.native_resize'])
load_from = os.environ.get('ROSA_INIT_CKPT', os.path.join(
    repo_root, 'projects/ROSA/work_dirs/combined_4bags/best_fisheye_bev_mAP_epoch_6.pth'))
resume = False
model = dict(head=dict(instance_bank=dict(anchor=os.path.join(repo_root,
    'projects/ROSA/datasets/combined_bags_camera_manual_review_v7/rosa/anchors/train_centers300_seed3407.npy'))))
native_loader = dict(type='LoadNativeResizedFisheyeForSparse4D', calibration=calibration)
train_pipeline = [native_loader, dict(type='FisheyePhotometricAugmentation'),
    dict(type='LoadAnnotations3D', with_bbox=False, with_bbox_3d=True, with_label=False, with_label_3d=True),
    dict(type='Fisheye3DODEvaluatorRangeFilter'),
    dict(type='PackSparse4DInputs', ocam_lens_fov_deg=220.)]
val_pipeline = [native_loader, dict(type='PackSparse4DInputs', ocam_lens_fov_deg=220.)]
train_dataloader = dict(dataset=dict(dataset=dict(
    data_root=data_root, ann_file=os.path.join(annotation_root, 'fisheye3dod_infos_train.pkl'), pipeline=train_pipeline)))
val_dataloader = dict(dataset=dict(
    data_root=data_root, ann_file=os.path.join(annotation_root, 'fisheye3dod_infos_val.pkl'), pipeline=val_pipeline))
test_dataloader = dict(dataset=dict(
    data_root=data_root, ann_file=os.path.join(annotation_root, 'fisheye3dod_infos_val.pkl'), pipeline=val_pipeline))
optim_wrapper = dict(accumulative_counts=4)
work_dir = os.path.join(repo_root, 'projects/ROSA/work_dirs/rosa_vov99_manual_review_v7_native_resize')
experiment_contract = dict(
    training_recipe='reviewed_v7_native_runtime_resize_singleframe',
    dataset='native reviewed v7; exact labels, tracks, splits; in-memory 800x800 resize and exact native lens polynomial',
    initialization=load_from, per_gpu_batch=1, effective_batch=8,
    anchors='TRAIN-only constructor anchors; full checkpoint warm-start restores learned anchors',
    comparison_caveat='Audit old checkpoint training overlap before claiming unseen-track generalization')
