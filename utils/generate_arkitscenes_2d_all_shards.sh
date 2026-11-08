#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-data/ARKitScenes_processed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/arkitscenes_coco}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
SHARD_START="${SHARD_START:-0}"
SHARD_END="${SHARD_END:-${MAX_PARALLEL}}"
SPLIT="${SPLIT:-all}"

if (( MAX_PARALLEL <= 0 )); then
  echo "MAX_PARALLEL must be positive" >&2
  exit 2
fi
if (( SHARD_START < 0 || SHARD_START >= MAX_PARALLEL )); then
  echo "SHARD_START must be in [0, MAX_PARALLEL)" >&2
  exit 2
fi
if (( SHARD_END <= SHARD_START || SHARD_END > MAX_PARALLEL )); then
  echo "SHARD_END must be in (SHARD_START, MAX_PARALLEL]" >&2
  exit 2
fi
if [[ "${SPLIT}" != "train" && "${SPLIT}" != "val" && "${SPLIT}" != "all" ]]; then
  echo "SPLIT must be train, val, or all" >&2
  exit 2
fi

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
export PYTHONUNBUFFERED=1
echo "[$(date '+%F %T')] Starting ARKitScenes generation: split=${SPLIT} shards=[${SHARD_START},${SHARD_END})/${MAX_PARALLEL}"
echo "Data: ${DATA_ROOT}; output: ${OUTPUT_ROOT}; python: ${PYTHON_BIN}"

running_pids=()
cleanup() {
  local pid
  for pid in "${running_pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

ensure_metadata() {
  local source_file="$1"
  local output_file="$2"
  echo "[$(date '+%F %T')] Preparing metadata: ${source_file}"
  "${PYTHON_BIN}" -u utils/add_arkitscenes_3d_instance_metadata.py \
    --input "${source_file}" --output "${output_file}"
}

run_split() {
  local split_name="$1"
  local worker_script="$2"
  local scene_dir="$3"
  local shard_id
  local pid

  mkdir -p "${scene_dir}"
  running_pids=()
  for ((shard_id = SHARD_START; shard_id < SHARD_END; shard_id++)); do
    echo "[$(date '+%F %T')] Launching ${split_name} shard ${shard_id}/${MAX_PARALLEL}"
    env \
      ARKIT_METADATA_PREPARED=1 \
      PYTHON_BIN="${PYTHON_BIN}" \
      DATA_ROOT="${DATA_ROOT}" \
      OUTPUT_ROOT="${OUTPUT_ROOT}" \
      NUM_SHARDS="${MAX_PARALLEL}" \
      SHARD_ID="${shard_id}" \
      SCENE_OUTPUT_DIR="${scene_dir}" \
      NUM_VIEWS="${NUM_VIEWS:-}" \
      VISUALIZE="${VISUALIZE:-0}" \
      VISUALIZATION_MAX_IMAGES="${VISUALIZATION_MAX_IMAGES:--1}" \
      bash "${worker_script}" &
    running_pids+=("$!")
  done
  for pid in "${running_pids[@]}"; do
    wait "${pid}"
  done
  running_pids=()
  echo "Completed ${split_name} shards [${SHARD_START}, ${SHARD_END})"
}

if [[ "${SPLIT}" == "train" || "${SPLIT}" == "all" ]]; then
  ensure_metadata \
    "${DATA_ROOT}/arkit_infos_train.pkl" \
    "${DATA_ROOT}/arkit_infos_train_with_ids.pkl"
  run_split \
    train \
    utils/generate_arkitscenes_2d_train.sh \
    "${OUTPUT_ROOT}/bbox_train_scenes"
fi

if [[ "${SPLIT}" == "val" || "${SPLIT}" == "all" ]]; then
  ensure_metadata \
    "${DATA_ROOT}/arkit_infos_val.pkl" \
    "${DATA_ROOT}/arkit_infos_val_with_ids.pkl"
  run_split \
    val \
    utils/generate_arkitscenes_2d_val.sh \
    "${OUTPUT_ROOT}/bbox_val_scenes"
fi
