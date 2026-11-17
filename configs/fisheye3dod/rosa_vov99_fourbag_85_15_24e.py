"""Full-model four-bag refresh, keeping the original single-frame architecture."""
import os

_base_=['./rosa_vov99_singleframe_refresh_24e.py']
repo_root=os.path.abspath(os.path.join('{{fileDirname}}','../../../..'))
data_root=os.environ.get('ROSA_DATA_ROOT',os.path.join(repo_root,
    'data/Custom/combined_improved_detzero_4bags_20261105'))
ann_train=os.path.join(data_root,'rosa/fisheye3dod_infos_train.pkl')
ann_val=os.path.join(data_root,'rosa/fisheye3dod_infos_val.pkl')
anchor_file=os.path.join(data_root,'rosa/anchors/kmeans300_seed3407.npy')
model=dict(head=dict(instance_bank=dict(anchor=anchor_file)))
train_dataloader=dict(dataset=dict(dataset=dict(data_root=data_root,ann_file=ann_train)))
val_dataloader=dict(dataset=dict(data_root=data_root,ann_file=ann_val))
test_dataloader=dict(dataset=dict(data_root=data_root,ann_file=ann_val))
load_from=os.environ.get('ROSA_INIT_CKPT',os.path.join(repo_root,
    'projects/ROSA/work_dirs/singleframe_refresh_20261031/best_fisheye_bev_mAP_epoch_1.pth'))
resume=False
find_unused_parameters=False
work_dir=os.path.join(repo_root,'projects/ROSA/work_dirs/fourbag_85_15')
experiment_contract=dict(training_recipe='E011_fourbag_full_model_85_15',initialization=load_from,
    dataset='four bags; preserved old601 + new73 validation; acquisition-time chronological holdouts',
    full_model_training=True,heading_only=False,temporal=False,temporal_slots=0,
    head_peak_lr=5e-5,backbone_peak_lr=5e-5,per_gpu_batch=1,effective_batch=8,
    caveat='Same-scene pseudo-label validation, no Van positives in holdout; not leave-one-bag-out.')
