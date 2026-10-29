"""Clean controlled retrain from the class-remapped KITTI-360 weights."""
import os

_base_ = ['./rosa_vov99_balanced_finetune_8e.py']
repo_root = os.path.abspath(os.path.join('{{fileDirname}}', '../../../..'))
load_from = os.environ.get('ROSA_INIT_CKPT', os.path.join(repo_root, 'projects',
    'ROSA', 'weights', 'rosa_kitti360_vov99_24e_improved_detzero_classes.pth'))
model = dict(img_backbone=dict(frozen_stages=-1, norm_eval=True))
optim_wrapper = dict(optimizer=dict(lr=1e-4))
train_cfg = dict(max_epochs=24, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=.1, begin=0, end=500, by_epoch=False),
    dict(type='CosineAnnealingLR', T_max=24, begin=0, end=24, by_epoch=True,
         convert_to_iter_based=True, eta_min_ratio=.01),
]
work_dir = os.path.join(repo_root, 'work_dirs/rosa_vov99_balanced_retrain_24e')
experiment_contract = dict(initialization=load_from, training_recipe='balanced_v1_clean_retrain')
