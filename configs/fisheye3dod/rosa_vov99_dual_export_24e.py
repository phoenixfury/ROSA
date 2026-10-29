"""Train ROSA on the exact samples exported for the Fisheye3DOD benchmark."""

import os

_base_ = ["./rosa_vov99_improved_detzero_24e.py"]

data_root = os.environ.get(
    "ROSA_DATA_ROOT",
    "/home/mostafa/mmdetection3d/data/Custom/combined_improved_detzero_rosa",
)
ann_train = os.path.join(
    data_root, "rosa", "fisheye3dod_infos_train.pkl")
ann_val = os.path.join(data_root, "rosa", "fisheye3dod_infos_val.pkl")
anchor_file = os.path.join(
    data_root, "rosa", "anchors", "kmeans300_seed3407.npy")

model = dict(head=dict(instance_bank=dict(anchor=anchor_file)))
train_dataloader = dict(dataset=dict(
    ann_file=ann_train,
    data_root=data_root,
))
val_dataloader = dict(dataset=dict(
    ann_file=ann_val,
    data_root=data_root,
))
test_dataloader = dict(dataset=dict(
    ann_file=ann_val,
    data_root=data_root,
))

work_dir = "work_dirs/rosa_vov99_dual_export_24e"
