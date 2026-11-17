#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(realpath "$(dirname "$0")/../../..")"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/projects/ROSA${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${FISHEYE3DOD_OFFICIAL_ROOT:-}" ]] && [[ -f "$REPO_ROOT/projects/Fisheye3DOD/dataset/fisheye3dod_metric.py" ]]; then
  export FISHEYE3DOD_OFFICIAL_ROOT="$REPO_ROOT/projects/Fisheye3DOD"
fi
DEFAULT_NATIVE="$REPO_ROOT/data/combined_bags_camera_ray_dedup_v2"
if [[ ! -d "$DEFAULT_NATIVE" ]]; then DEFAULT_NATIVE="$REPO_ROOT/data/Custom/combined_bags_camera_ray_dedup_v2";fi
export ROSA_DATA_ROOT="${ROSA_DATA_ROOT:-${NATIVE_ROOT:-$DEFAULT_NATIVE}}"
export ROSA_ANN_ROOT="${ROSA_ANN_ROOT:-$REPO_ROOT/projects/raybev_lite/datasets/combined_bags_camera_ray_manual_review_v7}"
export ROSA_INIT_CKPT="${ROSA_INIT_CKPT:-$REPO_ROOT/projects/ROSA/work_dirs/combined_4bags/best_fisheye_bev_mAP_epoch_6.pth}"
OUT="${OUT:-$REPO_ROOT/projects/ROSA/work_dirs/rosa_vov99_manual_review_v7_native_resize}"
"${PYTHON:-python}" projects/ROSA/tools/verify_native_review_v7.py \
  --annotations "$ROSA_ANN_ROOT" --root "$ROSA_DATA_ROOT"
if [[ ! -f "$ROSA_INIT_CKPT" ]]; then echo "Missing ROSA_INIT_CKPT: $ROSA_INIT_CKPT" >&2;exit 1;fi
if [[ -d "$OUT" ]] && [[ -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Choose a fresh OUT; existing run directory is not empty: $OUT" >&2;exit 1
fi
COMMAND=("${PYTHON:-python}" -m torch.distributed.run --standalone --nnodes=1
  --nproc_per_node="${NPROC_PER_NODE:-2}" projects/ROSA/tools/run.py train
  projects/ROSA/configs/fisheye3dod/rosa_vov99_manual_review_v7_24e.py
  --amp --launcher pytorch --work-dir "$OUT")
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${COMMAND[@]}";printf '\n';else exec "${COMMAND[@]}";fi
