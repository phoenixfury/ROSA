# ROSA experiment ledger

This is the persistent, Git-trackable record of the local ROSA experiments.
Each experiment folder contains a source patch, machine-readable metrics,
analysis, change notes, protocol/provenance and reproduction entry points.
Large weights, datasets, predictions and videos stay in ignored `work_dirs/`;
the ledger records their locations and hashes where available.
The backfill covers retained ROSA reports and recipes; it does not invent results
for unrecorded ad-hoc runs or cover DetZero/Fisheye3DOD-only training experiments.

## Experiments and decisions

| ID | Change / question | Evaluation protocol | Main finding | Record |
|---|---|---|---|---|
| E000 | Original epoch-16 baseline; corrected single-label/quality decoder and NMS references | Old 376 + separate current 601 | Decoder correction was more promising than replacing weights with the balanced retrain | [Baseline](experiments/E000_baseline/README.md) |
| E001 | Clean balanced retrain: corrected denoising selection, balancing and augmentation | Old 376 | Best retrain e12 mAP 5.28%, NDS 0.1340; baseline with the same decoder 5.64%, 0.1654. Not promoted | [Balanced retrain](experiments/E001_balanced_retrain/README.md) |
| E002 | Retain baseline e16; live BN, restored backbone LR, class-wise NMS | Old 376; separate current 601 parent reference | Old split: adaptation e4 mAP 5.40%, NDS 0.1655. Current parent: 10.21%, 0.2322, Car AP 40.01%. Not a direct cross-split gain | [BN/LR adaptation](experiments/E002_baseline_adaptation/README.md) |
| E003 | Chronological acquisition-time clips; 100 detached temporal slots; causal global-motion targets | Current 601 | e4/on 11.18% mAP, 0.2438 NDS, Car AP 33.25%; e7/off 13.21%, 0.2365, Car 36.79%. Memory worsened duplicates; stopped after NDS plateau | [Temporal](experiments/E003_temporal/README.md) |
| E004 | Frozen quality ranking/cache NMS, matched-budget controls, ego-only propagation; causal robust-target audit | Current 601; offline train-first pseudo-motion forecasts | No broad policy improvement; robust targets lost coverage without better paired error. No defaults changed or training launched | [Memory/motion audits](experiments/E004_memory_motion_audit/README.md) |
| E005 | Reset inherited velocity gates; optimize only XY/gate rows against unchanged targets | Same 992 train / 601 val | Recovery 11.04% mAP, 0.2391 NDS; moving error 7.508 vs control 7.530m/s (0.29%, below 10% requirement). Not promoted | [Row-only recovery](experiments/E005_velocity_rows/README.md) |
| E006 | Joint decoder/normal detection-loss adaptation; inherited versus reset gates | Same 992 train / 601 val | Motion error improves 10.55% vs control, but original mAP/NDS and coverage guards fail. Recovery: 10.21% mAP, 0.2315 NDS, Car 33.25%. Not promoted | [Joint recovery](experiments/E006_joint_decoder_recovery/README.md) |
| E007 | Restart from best adapted single-frame e4; full-model refresh without temporal slots | Current 3278 base train / 601 val | Stopped automatically e7. Best mAP e1: 11.23%, NDS 0.2482; best NDS e3: 0.2530, mAP 9.65%. Parent: 10.21%, 0.2322. e1 Car AP drops 40.01%→39.01%; duplicate-like FP at score .1 increase 741→772. Prefer e1 as a macro-detection candidate, not an unconditional replacement | [Checkpoint report and videos](experiments/E007_singleframe_refresh/checkpoint_comparison.md) |

All percentages in the table are mAP/Car-AP percentages; NDS is a fraction.
Do not directly rank results from different validation hashes/decoding protocols.
The 376-frame set and the current 601-frame chronological split are different
datasets. Current targets are DetZero/Kalman/camera-hold pseudo-labels, not an
untouched manually reviewed test set. Current validation has no Vans and only
12 in-range Cyclist boxes from four tracks; old initializers may have seen some
frames under previous splits. AP/NDS do not measure velocity or track identity.

## Per-experiment contents

- `README.md`: question, status, patch base/caveats, reproduction entry point.
- `rosa.patch`: pinned-base source patch; application checked on a clean temporary
  export without modifying this worktree/index or creating Git commits.
- `changes.json`: logical change, affected files and any relative patch.
- `metrics.json`: original metric/protocol data, never fabricated pending results.
- `analysis.md`: archived report copied from the actual experiment artifact.
- `record.json`: patch/source/artifact hashes, pinned commits, run-source matches
  where captured, and historical reconstruction limitations.
- `plan.json`, arm training summaries or extra audits where applicable.

E004 additionally has 11 `variants/<run>/` records with metrics and a
`policy.patch` adding a named configuration **after** its family source patch.
The fixed-budget controls have their original wrapper archived alongside them.
E004/E005/E006 have `changes_from_E00N.patch` additions relative to the preceding
archived compatible snapshot. E006's new source additions are an exact delta
against the E005 snapshot; older reconstructed stages are not exact historical
source transitions.

## Patch provenance and application

There are two deliberately distinct kinds of patch:

1. E000–E002 are exact committed recipe changes with their individual parent
   commits recorded. A recipe commit is **not proof** that a checkpoint trained
   from a clean version of it; older server provenance included dirty state.
2. E003–E006 are independent **cumulative compatible snapshots** against ROSA
   `f305bfc117ac9c5f85ee41cbb3580dfbfe9eccd0`. Older uncommitted revisions cannot
   be recovered exactly. E003 includes later backward-compatible cache knobs;
   its legacy defaults have a verified exact-prediction canary. Captured run
   source hashes are checked where available.

Do **not** apply every cumulative patch sequentially. On a separate clean
checkout at the pinned base, choose one cumulative patch, or apply the explicit
relative additions to the preceding archived snapshot:

E007 is a separate non-temporal recipe on the same pinned base. It does not
require the temporal/memory/recovery patches E003–E006; those experiments stay
archived but are not enabled in the new run.

```bash
# In the separate ROSA checkout at the base recorded in record.json:
git apply --check /path/to/E006_joint_decoder_recovery/rosa.patch
git apply /path/to/E006_joint_decoder_recovery/rosa.patch
```

The patches do not bundle checkpoints, environment wheels, datasets or external
repositories. ROSA depends on host-side changes too: see
[dependencies.json](experiments/dependencies.json) and
[external_mmdetection3d.patch](experiments/external_mmdetection3d.patch), applied
separately in the pinned mmdetection3d parent repository. Fisheye3DOD's commit and
dirty status are recorded. No claim of bitwise retraining reproducibility is made.

## Refresh after an experiment finishes

From the mmdetection3d root:

```bash
conda run --no-capture-output -n openmmlab python \
  projects/ROSA/tools/archive_rosa_experiments.py --verify-patches
```

The exporter refreshes completed results/status and verifies source against the
captured run manifests. It refuses to replace an existing patch with different
content: source changes require a new experiment record, not rewriting history.
Keep the source fixed while a paired trial is running. For the next experiment,
add a new protocol/spec and patch before training, then archive the final metrics
and decision. The archive is about a few MB, not a duplicate training dataset.
Use `--only E006_joint_decoder_recovery` to refresh one record without
regenerating older snapshots. For future experiments, register a new ID and use
`--only` when changed source no longer matches a completed experiment's snapshot.
