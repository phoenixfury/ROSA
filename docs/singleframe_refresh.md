# E007: non-temporal refresh on the latest combined dataset

Initialize from `work_dirs/server/rosa_vov99_baseline_adapt_3080ti_2gpu/best_fisheye_bev_NDS_epoch_4.pth`,
not a temporal checkpoint. Use a **fresh optimizer**, not that checkpoint's old
optimizer/scheduler. All backbone/decoder parameters train, and BN statistics
update. Peak backbone/head LR is 5e-5 with warmup and a 24-epoch cosine cap.
The original single-frame zero-velocity ABI and regression weights are retained;
there are no derived temporal motion targets, temporal slots, sequence sampler,
gate resets or row-only adaptation. Physically archived temporal source is not
deleted or imported by this recipe. Existing class balancing/photometric/DN
training and class-wise IoU-0.5 rotated BEV NMS remain enabled.

The current data have 3278 base training frames and 601 validation frames.
Class balancing repeats some training frames. The current chronological/purged
per-bag split is unchanged; only independent training frames are shuffled.
Labels are DetZero/Kalman/camera-hold pseudo-labels, not untouched manual truth.
The initializer may have seen some frames under earlier splits, so validation
results are adaptation diagnostics rather than an unbiased benchmark.

From the mmdetection3d root:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n openmmlab python \
  projects/ROSA/tools/train_singleframe_refresh.py \
  --data-root data/Custom/combined_improved_detzero_rosa \
  --gpus 1 --batch-size 1 --effective-batch 8 --workers 2 --background
```

The launcher checks all training/validation image paths, finite GT and anchors,
then measures real AMP optimizer updates on the 64 most crowded training frames.
The local 6-GB GPU uses physical batch 1 and accumulation 8. For a two-GPU server,
use `--gpus 2 --batch-size 2 --effective-batch 8`, with paths supplied through
`--data-root` / `--init-checkpoint`. Every launch stress-tests its physical batch;
there is no untested automatic batch enlargement. `--probe-only` runs only the
stress test; `--preflight-only` checks the dataset without training.

Validate every epoch, retaining separate best raw mAP and best raw NDS weights.
Reset the plateau counter if **either** metric improves by more than 0.0001
against its last meaningful best. Stop when **neither** improves for four
consecutive validations, or at the 24-epoch safety cap. This is one joint
criterion, not two independent early-stop hooks. The first validation establishes
the reference; the older parent metric does not prematurely stop this run.

Live outputs in `projects/ROSA/work_dirs/singleframe_refresh_20261031`:

- `console.log`: training loss, ETA to the epoch cap and validation metrics.
- `progress.json`: latest iteration, ETA, best meaningful metrics, plateau wait.
- `plateau.json`: complete validation history/stopping state, atomically saved.
- `best_fisheye_bev_mAP_epoch_*.pth`, `best_fisheye_bev_NDS_epoch_*.pth`:
  native best metric checkpoints; these may correspond to different epochs.
- `results.json`, `REPORT.md`: written on successful stop/cap, not before.
- `last_checkpoint`: final resumable checkpoint with explicit completed-epoch
  metadata. Resume via `--resume /absolute/path/from/last_checkpoint`, same
  work directory. Stopping state also resumes; a completed plateau cannot be
  bypassed just by restarting.

```bash
tail -f projects/ROSA/work_dirs/singleframe_refresh_20261031/console.log
python -m json.tool projects/ROSA/work_dirs/singleframe_refresh_20261031/progress.json
```

At completion the launcher automatically refreshes the E007 experiment archive.
The archive includes a checked independent patch, run hashes, metrics and report,
not duplicated checkpoint/data binaries. Keep the captured source fixed while
the run is active; subsequent source changes require a new experiment ID.
