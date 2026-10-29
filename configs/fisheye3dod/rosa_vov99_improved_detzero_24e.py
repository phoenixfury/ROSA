"""Fine-tune ROSA VoVNet-99 on the combined improved-DetZero dataset."""

import os

_base_ = ["./rosa_r18_fisheye3dod_20e.py"]

data_root = os.environ.get(
    "ROSA_DATA_ROOT",
    "/home/mostafa/mmdetection3d/data/Custom/combined_improved_detzero_rosa",
)
ann_root = os.path.join(data_root, "ImageSets-improved-detzero")
ann_train = os.path.join(ann_root, "fisheye3dod_infos_train.pkl")
ann_val = os.path.join(ann_root, "fisheye3dod_infos_val.pkl")
anchor_file = os.path.join(
    data_root, "improved_detzero_kmeans300_seed3407.npy")
load_from = os.environ.get(
    "ROSA_INIT_CKPT",
    "/home/mostafa/mmdetection3d/projects/ROSA/weights/"
    "rosa_kitti360_vov99_24e_improved_detzero_classes.pth",
)

custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        "projects.mmdet3d_plugin",
        "rosa.integrations.sparse4d_components",
        "rosa.integrations.optim",
        "rosa.integrations.fisheye3dod.dataset",
        "rosa.integrations.fisheye3dod.model",
        "rosa.integrations.fisheye3dod.metric",
        "rosa.integrations.fisheye_ray_sparse4d",
        "rosa.integrations.fisheye_box_sparse4d",
        "rosa.integrations.kitti360.backbone",
    ],
)

model = dict(
    img_backbone=dict(
        _delete_=True,
        frozen_stages=-1,
        init_cfg=None,
        input_ch=3,
        norm_eval=False,
        out_features=("stage2", "stage3", "stage4", "stage5"),
        spec_name="V-99-eSE",
        type="VoVNetCP",
    ),
    img_neck=dict(in_channels=[256, 512, 768, 1024]),
    head=dict(
        instance_bank=dict(
            anchor=anchor_file,
            anchor_handler=dict(
                anchor_z_is_bottom=False,
                type="NormalizedSparseBoxKeyPointsGenerator",
            ),
        ),
        loss_reg=dict(type="EvaluatorAlignedSparseBox3DLoss"),
        # Positive and negative DN copies mean dense traffic can otherwise add
        # 32 GT x 5 groups x 2 = 320 auxiliary queries on top of the 300 free
        # detector queries. Keep all five positive/negative groups but sample
        # at most four GTs for the auxiliary DN task (40 DN queries). Ordinary
        # Hungarian detection supervision still uses every GT box.
        sampler=dict(max_dn_gt=4),
        v4_decoder_cfg=dict(
            checkpoint_point_chunks=True,
            streaming_camera_rfl_cuda=True,
            use_fused_daf=True,
        ),
    ),
)

# The available RTX 3060 Laptop GPU has 6 GiB.  Gradient accumulation keeps
# the effective single-GPU batch at four while each forward uses one sample.
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(ann_file=ann_train, data_root=data_root),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(ann_file=ann_val, data_root=data_root),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(ann_file=ann_val, data_root=data_root),
)
optim_wrapper = dict(accumulative_counts=4)

train_cfg = dict(max_epochs=24, type="EpochBasedTrainLoop", val_interval=4)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=1.0 / 3.0,
        type="LinearLR"),
    dict(
        T_max=24, begin=0, by_epoch=True, convert_to_iter_based=True,
        end=24, eta_min_ratio=0.001, type="CosineAnnealingLR"),
]
find_unused_parameters = True
work_dir = "work_dirs/rosa_vov99_improved_detzero_24e"

experiment_contract = dict(
    dataset="16_09 + 30_09 improved DetZero camera-validated annotations",
    cameras="four canonical 800x800 OCam fisheye views",
    initialization=load_from,
    backbone="VoVNet-99",
    classes=("Car", "Van", "Truck", "Bus", "Pedestrian", "Cyclist"),
    construction_vehicle_mapping="Van",
    split="per-sequence 40-second temporal blocks, 80/20",
)
