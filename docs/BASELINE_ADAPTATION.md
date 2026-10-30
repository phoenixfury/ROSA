# Baseline epoch-16 adaptation on two RTX 3080 Ti GPUs

This is a **fresh eight-epoch fine-tune from the original baseline epoch-16 weights**, not a restart from KITTI-360 and not a resume of the balanced retrain. It makes no automatic changes to annotations or PKLs.

## What this recipe does

| Setting | Value |
|---|---|
| Initialization | `projects/ROSA/weights/best_fisheye_bev_NDS_epoch_16.pth` |
| Backbone | All stages trainable, including stem; `frozen_stages=-1` |
| BatchNorm | `norm_eval=False`; running statistics adapt on training images only |
| Checkpointed BN | Exactly one running-stat update per real forward, not a second update during backward recomputation |
| Peak head LR | `1e-4` |
| Peak backbone LR | `1e-4` (10× the failed clean retrain; original baseline's backbone LR) |
| Schedule | Eight epochs, short warmup then cosine decay; fresh optimizer |
| Batch | Default 2 frames/GPU × 2 GPUs × accumulation 2 = effective batch 8 |
| Decoder | One class/query; multiply class score by centerness before top-k |
| Suppression | Class-wise rotated BEV NMS, IoU 0.50; **no vehicle-class grouping** |
| Score cutoff | None during validation/export metrics; choose display thresholds separately |
| Model selection | Validate with NMS every epoch; save best mAP and NDS checkpoints |

The existing corrected DN targets/noise/grouping, positive-only class weights `[1,3,2,2,1.5,3]`, capped balanced sampling, empty-frame supervision, and photometric augmentation are retained from the fine-tuning recipe. Thus this is the requested deployment-oriented adaptation, not a one-factor BN-only scientific ablation. The old recipe files are unchanged so their results remain reproducible.

BatchNorm uses ordinary per-rank statistics (not SyncBatchNorm). Existing DDP buffer synchronization is retained. Activation checkpointing remains non-reentrant, with `find_unused_parameters=False`. During recomputation only, BN gets private temporary running-stat buffers while staying in training mode: outputs/gradients still use batch statistics, but the real running mean/variance and batch counter update only once. Evaluation uses fixed learned statistics and never updates them on validation images.

## Commit/pull both projects

`projects/ROSA` and `projects/Fisheye3DOD` are separate nested Git repositories in this workspace. Commit/push and pull **both**; an outer `mmdetection3d` commit alone will not include their internal changes.

ROSA changes for this recipe:

- `configs/fisheye3dod/rosa_vov99_baseline_adapt_8e.py`
- `configs/fisheye3dod/rosa_vov99_baseline_class_nms.py`
- `tools/train_balanced_fisheye.py`
- `tests/test_baseline_adaptation.py`
- `tests/test_balanced_ddp.py`
- `docs/BASELINE_ADAPTATION.md`
- `docs/BALANCED_FISHEYE_TRAINING.md`

Fisheye3DOD change:

- `models/petr/vovnetcp.py`

The launcher checks for the BN recomputation fix before starting live-BN training. No new dependencies are required. Keep the existing `projects/mmdet3d_plugin` implementation on the server up to date as before.

## Server preflight (no GPU training)

From `/home/mustafa/mmdetection3d` after pulling both projects:

```bash
conda run --no-capture-output -n fisheye3d \
python projects/ROSA/tools/train_balanced_fisheye.py \
  --recipe baseline-adapt --gpus 2 --batch-size 2 --workers 6 \
  --data-root /home/mustafa/mmdetection3d/data/combined_improved_detzero_rosa \
  --init-checkpoint projects/ROSA/weights/best_fisheye_bev_NDS_epoch_16.pth \
  --preflight-only
```

This verifies the baseline checkpoint, anchors, both splits and camera images. Inspect `work_dirs/rosa_vov99_baseline_adapt_8e/balanced_preflight.json`: both peak LRs must be `0.0001`, `backbone_norm_eval` must be `false`, frozen stages `-1`, and `decoder.nms_class_groups` must be empty.

Paths are derived from the checkout and `--data-root`; no `/home/mostafa` path editing is needed. If the original baseline checkpoint is stored elsewhere, append `--init-checkpoint /absolute/path/to/best_fisheye_bev_NDS_epoch_16.pth`. Do **not** substitute a checkpoint from `rosa_vov99_balanced_3080ti_2gpu` for this experiment.

## Launch training

Recommended: re-probe physical batches with live BN, retain effective batch eight, then start a fresh full run:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
conda run --no-capture-output -n fisheye3d \
python projects/ROSA/tools/train_balanced_fisheye.py \
  --recipe baseline-adapt \
  --gpus 2 --auto-batch --effective-batch 8 --fast --workers 6 \
  --data-root /home/mustafa/mmdetection3d/data/combined_improved_detzero_rosa \
  --init-checkpoint projects/ROSA/weights/best_fisheye_bev_NDS_epoch_16.pth \
  --master-port 29504 \
  --work-dir work_dirs/rosa_vov99_baseline_adapt_3080ti_2gpu
```

The crowded-frame probe tests batches 1, 2 and 4 per GPU, rejects OOM candidates, and selects the fastest safe result. Probes run independently from baseline initialization; their BN updates do not leak into the full training run. The learning rates do not scale when the physical batch changes because effective batch remains eight. `--fast` enables TF32/cuDNN tuning, so exact numerical reproducibility may differ.

To skip probing and use the previous server's batch choice, replace `--auto-batch` with `--batch-size 2`. This still uses accumulation 2. A fresh probe is recommended because BN is now live.

For a four-iteration server check, replace `--auto-batch` with `--batch-size 2 --smoke-test`, and use a separate `--work-dir work_dirs/rosa_baseline_adapt_smoke`. Smoke checkpoints are not usable trained models. Do not mix smoke and full-run directories.

Do not use `--resume` when transitioning from baseline epoch 16. To resume an interrupted run of **this exact recipe**, use its own checkpoint and the same selected physical/effective batch, omit `--auto-batch`, and add `--resume /path/to/this/run/epoch_N.pth`.

## Inference before or after adaptation

To use the **original unchanged baseline** with the new decoder plus class-wise NMS immediately:

```bash
CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n fisheye3d \
python projects/ROSA/tools/infer_fisheye_checkpoints.py \
  --config projects/ROSA/configs/fisheye3dod/rosa_vov99_baseline_class_nms.py \
  --checkpoint projects/ROSA/weights/best_fisheye_bev_NDS_epoch_16.pth \
  --data-root /home/mustafa/mmdetection3d/data/combined_improved_detzero_rosa \
  --output-dir work_dirs/rosa_baseline_class_nms \
  --no-lidar-visualization
```

After adaptation, use the same inference config and replace `--checkpoint` with the selected new best-mAP checkpoint, writing to a different output directory. The helper sets project import paths and Fisheye3DOD root automatically. Class-wise NMS occurs once, inside the decoder. Do not apply it a second time in an exporter.

For raw-decoder ablations only, the historical `rosa_vov99_balanced_finetune_8e.py` config uses the same new decoder without NMS. Keep raw and NMS metrics labeled separately. NMS is not part of the differentiable training loss; it affects validation/inference and checkpoint selection.

## Verification

```bash
conda run --no-capture-output -n fisheye3d \
  python projects/ROSA/tests/test_baseline_adaptation.py -v
conda run --no-capture-output -n fisheye3d \
  python projects/ROSA/tests/test_balanced_ddp.py
```

The new tests check the actual merged config/backbone, all-stage BN unfreezing, no cross-class NMS, checkpointed versus ordinary outputs/gradients/running statistics (including cumulative BN momentum), and restoration on recomputation-context exceptions. The two-process CPU DDP test checks one BN counter increment per step with checkpointing and different empty/rare-class supervision on the two ranks. A full two-GPU NCCL run still needs to be verified on the server; no accuracy improvement is guaranteed before validation.

Local verification: 18 CPU unit tests passed, the two-process CPU DDP test passed, and four real AMP training iterations completed from baseline epoch 16 on an RTX 3060 Laptop GPU. All 99 backbone BN counters increased by exactly four, all 99 running means changed, all saved state tensors are finite, and optimizer LR was `1e-4`. Logged peak allocation was 4,921 MiB for batch one. This does not establish the safe server batch size or final accuracy; use the two-GPU auto-probe and validation results.
