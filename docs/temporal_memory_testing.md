# Frozen-weight temporal-memory tests

These experiments change ROSA query-memory selection, not DetZero annotations,
the output decoder, calibration, or checkpoint parameters. Training is never
started by the commands below. Existing temporal defaults remain unchanged.

From the repository root, with the OpenMMLab environment and CUDA available:

```bash
export PYTHONPATH="$PWD:$PWD/projects/ROSA:$PWD/projects/Fisheye3DOD"
export FISHEYE3DOD_OFFICIAL_ROOT="$PWD/projects/Fisheye3DOD"
export ROSA_DATA_ROOT="$PWD/data/Custom/combined_improved_detzero_rosa"

conda run --no-capture-output -n openmmlab python \
  projects/ROSA/tools/test_temporal_memory.py --epoch 4
```

The default sequence runs `current`, `current_gated`, `quality`, `class_nms`,
`quality_nms`, `quality_nms_ego_only`, and `raw50` on all validation frames.
Use repeatable `--variant` and `--epoch` flags to select confirmation tests,
for example `--epoch 7 --variant quality_nms`. `--dry-run` prints commands
without inference. Results resume only when checkpoint, data, code, config,
prediction and cache-stat fingerprints agree. Do not edit inference source
files while a sequence is running; the runner rejects mixed-code results.

The `current` run must numerically reproduce its existing saved predictions.
Its canary comparison is not bypassed by default. Source-independent legacy
memory-off references are labelled explicitly, not represented as fresh runs.

## Opt-in instance-bank settings

| Setting | Legacy default | Meaning |
|---|---|---|
| `cache_score_mode` | `class` | `quality` multiplies max class probability by final-stage centerness quality |
| `cache_nms_iou_threshold` | `None` | Optional class-wise rotated BEV NMS before cache top-k |
| `cache_require_current` | `False` | Require current evidence before historical confidence can be used |
| `motion_mode` | `predicted` | `ego_only` disables center extrapolation, preserving ego compensation and rotated velocity seed |
| `collect_cache_stats` | `False` | Record per-frame active slots, candidate/suppression counts and scores |

Quality-ranked tests start at cache confidence 0.05, versus the original raw
class threshold 0.15. These thresholds are not calibrated equivalents.
Occupancy and a reduced-capacity raw-score control must accompany results;
otherwise apparently better selection may just mean less memory. Total query
count remains 300. Suppressed/invalid cache slots fall back to fresh queries.
No additional learned parameters are introduced.

All variants keep the same final-stage feature/anchor/class/quality alignment,
the normal single-label quality-ranked output decoder and class-wise output
NMS. Cache NMS is an additional upstream operation, not another display filter.
Cache resets and acquisition-time/scene/predecessor checks remain active.

## Motion-target audit, before retraining

```bash
conda run --no-capture-output -n openmmlab python \
  projects/ROSA/tools/audit_temporal_motion_targets.py --help
```

The pure NumPy `motion_targets.py` prototype estimates causal world-XY motion
using three/five past observations and robust fitting. It preserves unknown
velocities as NaNs and classifies static objects only from sustained near-zero
global motion. It is **not enabled in the training dataset**.

Audit raw versus robust forecasts on identical retained examples, together
with supervision coverage and class/ego-turn subsets. Do not infer improvement
from lower acceleration or unpaired averages. Next-frame annotations are for
scoring only and must not enter the fit. Pseudo-label agreement is not an
independent measurement of physical motion.

Changing targets cannot change a frozen checkpoint. Runtime ego-only testing
instead isolates whether the existing model's XY extrapolation helps. Neither
test extrapolates Z/height; legitimate ego-coordinate transformations remain.

Current study artifacts are under
`projects/ROSA/work_dirs/temporal_memory_ablation_20261031/`. Inspect AP/NDS,
Car AP, precision at matched recall, duplicate counts, slot occupancy and
temporal residuals before selecting a policy. The chronological validation set
is a development benchmark with DetZero pseudo-GT, not an untouched manual test
set. No Van GT and few Cyclist tracks make macro-AP selection particularly
fragile.

## Completed study and training gate

The 2026-10-31 study completed nine frozen inference variants/confirmations and
two matched-budget controls on 601 validation frames. See
[the report](../work_dirs/temporal_memory_ablation_20261031/REPORT.md) for results.
Quality ranking gives small macro-metric gains but no clear overall duplicate/
Car-AP improvement; robust motion fitting loses coverage without improving
matched forecast error. Neither change is promoted to a default.

The study also found nearly zero predicted motion and demonstrated a saturated
velocity-update gate blocking residual gradients in a real moving-object AMP
probe. Temporarily neutralizing velocity gates restores gradients, but this is
not an inference fix or evidence of better trained accuracy. A controlled
velocity-gate recovery trial with explicit motion diagnostics should precede
further full temporal training. Original checkpoints and training targets remain
unchanged; diagnostic backward passes did not perform optimizer steps.

The subsequent opt-in [controlled velocity-gate recovery pilot](velocity_gate_recovery.md)
trains only XY residual/gate rows, with an unchanged-gate control and reset-only
validation. It is separate from the default temporal training recipe.
