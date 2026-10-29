# Corrected ROSA training on the combined custom dataset

Use the new launcher from the mmdetection3d checkout. It sets both project import paths and the Fisheye3DOD root, validates both splits/images/anchors, and writes `balanced_preflight.json` plus `rare_class_review.csv` into the work directory. Existing PKLs/images do not need rewriting.

## Install the code update on the server

The latest archive `work_dirs/three_model_benchmark/rosa_3080ti_throughput_patch.tar.gz` contains only the changed/new code, configs, tests, and this guide. Copy it to the server and extract from the repository root:

```bash
cd /home/mustafa/mmdetection3d
tar -xzf /path/to/rosa_3080ti_throughput_patch.tar.gz
```

The update includes `projects/mmdet3d_plugin` and the Fisheye3DOD VoVNet checkpointing change. Copying only `projects/ROSA` would miss critical fixes. No weights, datasets or environments are included in the archive.

## Two RTX 3080 Ti GPUs: benchmark batch size, then train

The launcher now defaults to **two samples per GPU**, with accumulation calculated automatically to preserve an effective batch of eight. The bare configs retain the original batch-1 settings for smaller GPUs; use the launcher for the server settings.

| Samples per GPU | GPUs | Accumulation | Effective batch |
|---:|---:|---:|---:|
| 1 | 2 | 4 | 8 |
| 2 | 2 | 2 | 8 |
| 4 | 2 | 1 | 8 |

For the fastest tested choice on the actual server:

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n fisheye3d \
  python projects/ROSA/tools/train_balanced_fisheye.py \
  --recipe retrain --gpus 2 --auto-batch --fast --workers 6 \
  --data-root /home/mustafa/mmdetection3d/data/combined_improved_detzero_rosa \
  --work-dir work_dirs/rosa_vov99_balanced_3080ti_2gpu
```

This benchmarks batch sizes 1, 2 and 4 in separate training processes using both GPUs, AMP, real backward/optimizer steps and the 64 most crowded training frames. Default warmup is four iterations, followed by eight measured iterations; boundaries are aligned to complete gradient accumulation cycles. The score is aggregate frames/second including dataloader wait. Warmup and steady-state memory are reported separately. Each candidate must leave at least 8% of each GPU's memory unallocated by PyTorch during measured training. OOM candidates are rejected; unrelated errors stop the probe with a log path. Batch 4 is a candidate, not a claim that it fits 12 GB.

The fastest safe candidate is selected automatically and a **fresh full run** starts from the configured initial checkpoint using the complete training split. Probes do not save model checkpoints. Results/logs are retained under `WORK_DIR/batch_probe/`, including `results.json` and `selected.json`. Add `--probe-only` to benchmark and exit before full training. `--auto-batch` is for new runs, not resumed runs.

For an immediate run with the starting choice, replace `--auto-batch` with `--batch-size 2`. A larger physical batch reduces gradient accumulation; the learning rate remains unchanged because the effective batch stays eight. Iteration-based warmup is scaled to preserve the same training exposure. Batch 3 does not divide effective batch eight and is rejected; exploring it requires explicitly choosing another effective batch and reconsidering the optimization settings.

`--fast` enables cuDNN algorithm tuning and TF32 and disables deterministic algorithms. These speed settings can change exact numerical reproducibility. All server launches enable pinned-memory loading and asynchronous GPU transfers. `--workers 6` means six workers per GPU; adjust it to the server CPU/RAM/storage bandwidth if data loading is the bottleneck. See the [PyTorch performance guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

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

The launcher defaults to batch size 2 per GPU and accumulation 2 on two GPUs (effective batch eight). The clean retrain uses head LR 1e-4 and backbone LR 1e-5. The fine-tune uses head LR 5e-5 and backbone LR 5e-6. AMP is enabled by the launcher. `find_unused_parameters=False` is retained with corrected non-reentrant checkpointing.

To change routine settings, append e.g.:

```bash
--workers 4 --cfg-options train_cfg.max_epochs=8
```

The configured cosine schedule must also be adjusted when changing epoch count; use the provided eight-epoch config for a short run. Override the work directory with `--work-dir /path/to/new_run`. Existing checkpoints are never silently resumed by the launcher.

When resuming, pass the same batch size/effective batch used by that run. Changing batches midway also changes dataloader iteration counts and scheduler state; use a fresh initialization run when changing this recipe.

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

The automatic-probe workflow is also checked locally on one GPU. No claim is made about which batch wins on the server before it benchmarks the two 3080 Ti GPUs. For a 6 GB local GPU use `--gpus 1 --batch-size 1` rather than the server's batch-2 default.

For a short server GPU check before starting the complete run:

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n fisheye3d \
  python projects/ROSA/tools/train_balanced_fisheye.py \
  --recipe retrain --gpus 2 --smoke-test \
  --data-root /home/mustafa/mmdetection3d/data/combined_improved_detzero_rosa
```

This writes to a separate `_smoke` work directory by default. Ignore its `iter_4.pth` for model selection.

Temporal model/tracker additions, automatic relabeling of ambiguous objects, and a manually verified track-disjoint holdout are separate work. The current model remains single-frame; no fabricated motion compensation is enabled.
