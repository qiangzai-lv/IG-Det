#!/usr/bin/env bash

CONFIG=$1
shift

# MA preset envs
MASTER_HOST="$VC_WORKER_HOSTS"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
NNODES="$MA_NUM_HOSTS"
NODE_RANK="$VC_TASK_INDEX"
NGPUS_PER_NODE="$MA_NUM_GPUS"

# 自定义端口
MASTER_PORT=38888

# Python 脚本
PYTHON_SCRIPT="$(dirname "$0")/train.py"

export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=180
export PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH}"

python -m torch.distributed.run \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --nproc_per_node="$NGPUS_PER_NODE" \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$PYTHON_SCRIPT" \
    "$CONFIG" \
    --launcher pytorch \
    "$@"