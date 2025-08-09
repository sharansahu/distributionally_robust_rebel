#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd -P )"

# === config ===
: "${REBEL_FILE:=$REPO_ROOT/experiments/run_emotion_rebel_experiments.py}"

# Path to DS config
: "${DS_CONFIG:=$REPO_ROOT/ds_zero2.json}"

# GPU count for local node
: "${N_GPUS:=8}"

# Optional: NCCL quality-of-life flags
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "$DS_CONFIG" ]]; then
  echo "DeepSpeed config not found at $DS_CONFIG; generating default..."
  DS_CONFIG="$DS_CONFIG" bash "$(dirname "$0")/make_ds_config.sh"
fi

echo "[REBEL/DPO] Entry: $REBEL_FILE | GPUs: $N_GPUS | DS: $DS_CONFIG"

deepspeed --num_gpus="$N_GPUS" "$REBEL_FILE" --deepspeed_config "$DS_CONFIG"
