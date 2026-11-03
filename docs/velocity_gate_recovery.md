# Controlled velocity-gate recovery pilot

This is an opt-in motion-only experiment, not a replacement training recipe.
It starts from temporal epoch 4 and tests whether restoring velocity-update
capacity allows the existing representation to learn useful causal XY motion.
No original checkpoint, annotation, PKL, default model setting or decoder is
overwritten.

## What is controlled

- **Control:** unchanged gates, fresh optimizer, XY-only optimization.
- **Recovery:** same settings and optimizer, but initialize only the velocity
  gate's projection row/bias to zero logits (sigmoid 0.5) before optimization.
- **Reset-only:** evaluate that initialization without optimizer updates.
- **Original:** hash-verified saved epoch-4 predictions from the preceding study.

Only the two XY rows of each final residual projection and the velocity row of
each gate projection can change. Z velocity, center/size/yaw/classification,
backbone and all persistent buffers are protected. Gradient masks, fresh AdamW
state and zero weight decay enforce row isolation; a hash checks every protected
tensor/row afterward. Optimizer state from older training is never resumed.

The pilot uses eval-mode forward passes with gradients enabled: BatchNorm and
dropout are frozen and GT-seeded denoising queries are disabled. This deliberately
tests learning in the deployed free-query path. Six decoder stages receive the
existing finite-XY L1 supervision, averaged across stages. Class/geometry losses
are not optimized. The existing assignment, causal two-observation targets,
original 100-slot memory and output NMS are unchanged. Unknown targets stay masked;
no Z/height extrapolation or robust target smoothing is added.

Although geometry weights are frozen, learned velocity can change later decoder
features and propagated centers. Geometry output is **not** assumed unchanged;
full detection validation is required.

## Run

From the repository root in the OpenMMLab environment:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n openmmlab python \
  projects/ROSA/tools/run_velocity_recovery.py \
  --output-dir projects/ROSA/work_dirs/velocity_gate_recovery_20261031
```

Use `--dry-run` first to inspect the plan. Use a new output directory for each
trial. The default initialization/reference is the local temporal epoch-4 study;
the original prediction artifact must be retained for its comparison. This
single-GPU pilot is not a DDP/full-training launcher.

Defaults are one pass over up to eight uniformly time-spaced, non-overlapping
32-frame clips per train scene, learning rate 0.001, accumulation 8, seed 3407.
The current split selects 992 frames from four train scene segments. Clip
selection does not inspect classes or validation performance. Full train history
is retained when deriving causal motion targets; restricting sampler visits does
not regenerate or truncate those targets. Both arms use the same clip order,
photometric augmentation settings, seeds and number of optimizer opportunities.

The launcher executes serial GPU jobs and then compares all 601 chronological
validation frames. It does **not** automatically extend training or deploy a
winner. Output includes:

- `plan.json`: data/checkpoint/source fingerprints, selected train tokens and
  predeclared pilot criteria.
- `control_train.log`, `recovery_train.log`: loss, gate values, update counts and ETA.
- `<arm>/steps.jsonl`: per-frame loss, velocity diagnostics, saturation and gradients.
- `<arm>/training.json`: protected-state verification, allowed rows, timings and memory.
- `<arm>/pilot.pth`: separate trained checkpoints; `recovery/reset_only.pth` is
  an initialization ablation, not a trained model.
- `*_eval/predictions.pkl`, `reset_only/predictions.pkl`: ordinary ROSA outputs.
- `results.json`, `REPORT.md`: detection metrics, operating points and velocity
  error on a shared matched-moving-object cohort.

While running, inspect progress with:

```bash
tail -n 8 projects/ROSA/work_dirs/velocity_gate_recovery_20261031/control_train.log
tail -n 8 projects/ROSA/work_dirs/velocity_gate_recovery_20261031/recovery_train.log
```

## Interpreting the result

Before training, the plan records requirements: at least 10% lower shared-cohort
moving-object velocity error than the unchanged-gate control, retained matched
coverage, and bounded degradation in Car AP/mAP/NDS/duplicates. These are practical
pilot gates, not statistical significance guarantees. A pass permits a larger
controlled experiment, not automatic promotion.

The shared cohort fixes object membership to reduce recall-selection bias.
Full matched coverage and detection operating points must still be inspected.
Motion targets are DetZero/Kalman-derived pseudo-labels, including camera holds;
better agreement is not an independent measurement of physical motion. This
development validation split also has no Vans and very few Cyclists. Failure
of the isolated pilot would not prove a properly supervised full temporal
architecture cannot learn motion.

## Completed 2026-10-31 pilot

The paired 992-frame/124-update trial completed, with full 601-frame validation.
Recovery removed sampled gate saturation but improved shared-cohort moving
velocity error by only 0.29% versus control, missing the predeclared 10% gate.
Detection was better than control but slightly below the untouched checkpoint.
No weights were promoted and no further training was automatically launched.
See [the full report](../work_dirs/velocity_gate_recovery_20261031/REPORT.md),
including reset-only results and the audit showing saturation in the parent model.
