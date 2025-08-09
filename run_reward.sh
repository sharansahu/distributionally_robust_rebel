#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd -P )"

# === config ===
: "${REWARD_FILE:=$REPO_ROOT/scripts/train_reward_model.py}"

# How many GPUs to use
: "${N_GPUS:=1}"

: "${NCCL_DEBUG:=WARN}"
export NCCL_DEBUG

echo "[Reward] Entry: $REWARD_FILE | GPUs: $N_GPUS"

if (( N_GPUS > 1 )); then
  torchrun --standalone --nproc_per_node="$N_GPUS" "$REWARD_FILE"
else
  python "$REWARD_FILE"
fi