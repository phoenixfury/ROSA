# Corrected ROSA training on the combined custom dataset

Use the new launcher from the mmdetection3d checkout. It sets both project import paths and the Fisheye3DOD root, validates both splits/images/anchors, and writes `balanced_preflight.json` plus `rare_class_review.csv` into the work directory. Existing PKLs/images do not need rewriting.

## Install the code update on the server

The archive `work_dirs/three_model_benchmark/rosa_balanced_server_patch.tar.gz` contains only the changed/new code, configs, tests, and this guide. Copy it to the server and extract from the repository root:

```bash
cd /home/mustafa/mmdetection3d
tar -xzf /path/to/rosa_balanced_server_patch.tar.gz
```

The update includes `projects/mmdet3d_plugin` and the Fisheye3DOD VoVNet checkpointing change. Copying only `projects/ROSA` would miss critical fixes. No weights, datasets or environments are included in the archive.

## Recommended: clean retraining, 24 epochs

This starts with the original class-remapped KITTI-360 pretrained weights, so the model is not already adapted to the faulty denoising task. It starts a fresh optimizer/scheduler and saves best mAP/NDS checkpoints after validation every epoch.

```bash
cd /home/mustafa/mmdetection3d
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n fisheye3d \
  python projects/ROSA/tools/train_balanced_fisheye.py \
  --recipe retrain --gpus 2 \
  --data-root /home/mustafa/mmdetection3d/data/combined_improved_detzero_rosa
```

Default initialization:
`projects/ROSA/weights/rosa_kitti360_vov99_24e_improved_detzero_classes.pth`.

To validate files without training, append `--preflight-only`. If the initialization checkpoint lives elsewhere, add `--init-checkpoint /path/to/checkpoint.pth`.

## Faster option: eight-epoch fine-tune

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n fisheye3d \
  python projects/ROSA/tools/train_balanced_fisheye.py \
  --recipe finetune --gpus 2 \
  --data-root /home/mustafa/mmdetection3d/data/combined_improved_detzero_rosa \
  --init-checkpoint projects/ROSA/weights/best_fisheye_bev_NDS_epoch_16.pth
```

Use initialization, not `--resume`, for this transition from the old recipe. The optimizer, denoising objective, sampling and LR schedule changed. `--resume` is only for interrupting/resuming a run already trained with the corrected recipe; pass the matching `--recipe` and checkpoint path.

## Training changes

| Area | Corrected behavior |
|---|---|
| DN background | Class 6 remains a classification-only negative, never a box-regression target |
| DN normalization | Count actual foreground DN copies across ranks; clamp to one for empty batches |
| DN groups | Every attention group contains its own positive and negative copies |
| DN selection | Randomized class-stratified selection; keep the four-GT memory cap |
| DN noise | XYZ scaled by box size and rotated into the object's axes; unit yaw vectors; zero velocity noise |
| Nearby GT | Ignore DN negatives centered inside any real GT box, including GT excluded from DN selection |
| DN weighting | Auxiliary multiplier reduced from 5 to 1 |
| Rare-class regression | Every Hungarian-matched query gets regression, regardless of initial class confidence |
| Classification | Positive-only class multipliers `[1,3,2,2,1.5,3]` for Car/Van/Truck/Bus/Pedestrian/Cyclist |
| Frame sampling | Seeded fractional class-balanced repeats, maximum three per original frame |
| Empty frames | Retained as background supervision, with finite empty-box losses |
| Mixed-rank DDP | Empty/nonempty ranks execute the same DN reduction and emit identical loss keys |
| Images | Mild exposure/color/noise/blur; calibration and image coordinates unchanged |
| Backbone | Non-reentrant activation checkpointing; BN statistics fixed during these small-batch runs |
| Fine-tune only | Freeze stem/stage2 and lower backbone LR to 10% of head LR |
| Validation | Every epoch; retain best mAP and NDS checkpoints |
| Provenance | Save recipe and implementation hashes in checkpoints |

The current 2,023 training frames become 2,264 training entries after balancing. All 96 raw empty training frames remain. A seeded pass through the actual filtered training labels selects all 53 Van boxes for DN, compared with zero using the original first-four rule. No labels are automatically reclassified or deleted. The generated 221-row review list prioritizes Van/Cyclist labels and low-height Truck/Bus candidates. Those are candidates for human verification, not verified errors.

Both configurations use batch size 1 per GPU and accumulation 4. On two GPUs the effective batch is eight frames. The clean retrain uses head LR 1e-4 and backbone LR 1e-5. The fine-tune uses head LR 5e-5 and backbone LR 5e-6. AMP is enabled by the launcher. `find_unused_parameters=False` is retained with corrected non-reentrant checkpointing.

To change routine settings, append e.g.:

```bash
--cfg-options train_dataloader.num_workers=4 train_cfg.max_epochs=8
```

The configured cosine schedule must also be adjusted when changing epoch count; use the provided eight-epoch config for a short run. Override the work directory with `--work-dir /path/to/new_run`. Existing checkpoints are never silently resumed by the launcher.

## Decoder and visualization

New training configs validate with one label per query, with class/centerness combined before top-k selection. They do not impose a global confidence cutoff or NMS during training validation. This avoids removing rare-class candidates prematurely. Consequently their decoding protocol differs from the historical flattened-class baseline; retain that distinction when comparing scores.

`configs/fisheye3dod/rosa_vov99_balanced_deploy.py` enables optional rotated BEV NMS at IoU 0.5, grouping Car/Van/Truck/Bus while keeping Pedestrian/Cyclist separate. Test suppressed and raw results separately. This config is for evaluating/deploying a selected trained checkpoint, not a second training recommendation.

The portable JSON exporter now emits `box_origin=gravity_center`, correctly converts bottom-centered MMDet3D box storage, preserves tokens from dictionary results, and reads result pickles on CPU. The combined comparison renderer also handles old bottom-centered JSON exports. The original epoch-16 corrected export is available at:

`projects/ROSA/work_dirs/checkpoint_inference/best_fisheye_bev_NDS_epoch_16/predictions_gravity_center.json`.

## Verification

```bash
conda run --no-capture-output -n fisheye3d \
  python projects/ROSA/tests/test_balanced_training.py -v
conda run --no-capture-output -n fisheye3d \
  python projects/ROSA/tests/test_balanced_ddp.py
```

The unit checks cover DN masks/targets, Van selection, empty-frame backward, positive-only class weighting, photometric calibration invariance, quality-aware decoding, grouped suppression, instance-ID filtering and gravity-center export. The two-process Gloo test exercises the real checkpointed OSA block with empty GT on one rank and rare-class/DN supervision on the other using `find_unused_parameters=False`.

Four real 800x800 AMP training iterations were verified on an RTX 3060 Laptop GPU for both initializations. Logged peak allocation was approximately 4,429 MiB for fine-tuning and 4,921 MiB for the clean restart. These are smoke tests, not trained models or evidence of higher validation AP. They do not substitute for a full two-GPU NCCL run on the server.

For a short server GPU check before starting the complete run:

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n fisheye3d \
  python projects/ROSA/tools/train_balanced_fisheye.py \
  --recipe retrain --gpus 2 --smoke-test \
  --data-root /home/mustafa/mmdetection3d/data/combined_improved_detzero_rosa
```

This writes to a separate `_smoke` work directory by default. Ignore its `iter_4.pth` for model selection.

Temporal model/tracker additions, automatic relabeling of ambiguous objects, and a manually verified track-disjoint holdout are separate work. The current model remains single-frame; no fabricated motion compensation is enabled.
