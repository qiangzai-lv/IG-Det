#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-data/ARKitScenes_processed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/arkitscenes_coco}"
SPLIT="train"
ANN_FILE="${DATA_ROOT}/arkit_infos_${SPLIT}_with_ids.pkl"
"${PYTHON_BIN}" utils/add_arkitscenes_3d_instance_metadata.py \
  --input "${DATA_ROOT}/arkit_infos_${SPLIT}.pkl" --output "${ANN_FILE}"
extra=()
if [[ "${VISUALIZE:-0}" == "1" ]]; then
  extra+=(--visualize --visualization-max-images "${VISUALIZATION_MAX_IMAGES:--1}")
fi
exec "${PYTHON_BIN}" utils/arkitscenes_3d_to_coco_bbox.py \
  --data-root "${DATA_ROOT}" --ann-file "${ANN_FILE}" \
  --output-dir "${SCENE_OUTPUT_DIR:-${OUTPUT_ROOT}/bbox_${SPLIT}_scenes}" \
  --num-shards "${NUM_SHARDS:-1}" --shard-id "${SHARD_ID:-0}" \
  --num-views "${NUM_VIEWS:--1}" --sampling uniform \
  --min-visible-ratio 0.05 "${extra[@]}" "$@"
