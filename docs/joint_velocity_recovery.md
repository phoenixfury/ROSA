# E006: joint decoder adaptation with controlled velocity-gate recovery

E005 restored velocity gradients but its frozen-representation pilot did not
learn useful motion. E006 tests whether existing decoder features can adapt
jointly with motion while normal detection losses protect class/geometry output.

Both arms start from the untouched temporal epoch-4 checkpoint, not E005 weights.
They share 992 chronological training frames, one pass, 124 accumulated optimizer
updates, LR 5e-5, fresh AdamW with no decay, clip norm 5, and seed 3407.
Only recovery resets the velocity-gate projection rows/biases to zero logits.

The entire existing fisheye decoder (including classifiers/quality heads) is
trainable. Backbone, FPN, query priors, buffers and final Z-velocity residual
rows are protected by hashes/masks. Forward mode remains eval with autograd:
BN/dropout stay frozen and GT-seeded DN queries remain absent. Native free-query
classification, box, centerness, yawness and ray losses are used unchanged.
This is decoder adaptation, not whole-backbone fine-tuning or a new architecture.
Targets, cache selection/propagation and output decoder remain unchanged.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n openmmlab python \
  projects/ROSA/tools/run_joint_velocity_recovery.py \
  --output-dir projects/ROSA/work_dirs/joint_decoder_recovery_20261031
```

Use `--dry-run` to inspect the plan. Use a fresh output directory for another
trial. The local E004 original prediction and E005 reset-only reference artifacts
must be retained: initialization/data/prediction hashes are checked before GPU
training. Reset-only is reused rather than recomputed because its weights and
inference/data protocol are unchanged; the new artifact directory contains a
reference symlink, not a second copy.

Full augmented-image tensor hashes and frame tokens verify paired inputs.
The plan freezes promotion criteria before optimization: at least 10% lower
shared-moving-object velocity error versus control, preserved matched coverage,
bounded detection/duplicate degradation, equal update budgets/no skips, identical
protected state, and detection guardrails against the untouched original too.
Passing permits a larger controlled test, not automatic deployment or training
extension. No full-backbone training starts automatically.

Artifacts include `plan.json`, arm training logs/`steps.jsonl`/`training.json`,
separate `pilot.pth` weights, full 601-frame inference outputs, `results.json`
and `REPORT.md`. Motion agreement is measured against causal DetZero/Kalman
pseudo-targets, not independent physical ground truth. This development split
has no Vans, few Cyclists, and potential earlier-initializer exposure.

The versioned [experiment ledger](../EXPERIMENTS.md) retains this protocol,
its patch, metrics and analysis outside ignored `work_dirs/`.
