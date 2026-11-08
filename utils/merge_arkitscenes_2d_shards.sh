#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/arkitscenes_coco}"
SPLIT="${SPLIT:-all}"

if [[ "${SPLIT}" != "train" && "${SPLIT}" != "val" && "${SPLIT}" != "all" ]]; then
  echo "SPLIT must be train, val, or all" >&2
  exit 2
fi

cd "${REPO_ROOT}"

merge_split() {
  local scene_dir="$1"
  local output_json="$2"
  local rejection_json="$3"
  "${PYTHON_BIN}" utils/merge_arkitscenes_2d_coco_shards.py \
    --input-dir "${scene_dir}" \
    --output "${output_json}" \
    --rejections-output "${rejection_json}"
}

if [[ "${SPLIT}" == "train" || "${SPLIT}" == "all" ]]; then
  merge_split \
    "${OUTPUT_ROOT}/bbox_train_scenes" \
    "${OUTPUT_ROOT}/bbox_train.json" \
    "${OUTPUT_ROOT}/bbox_train_rejections.json"
fi

if [[ "${SPLIT}" == "val" || "${SPLIT}" == "all" ]]; then
  merge_split \
    "${OUTPUT_ROOT}/bbox_val_scenes" \
    "${OUTPUT_ROOT}/bbox_val.json" \
    "${OUTPUT_ROOT}/bbox_val_rejections.json"
fi
