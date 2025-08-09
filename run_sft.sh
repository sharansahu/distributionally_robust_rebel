#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd -P )"

# === config ===
: "${SFT_FILE:=$REPO_ROOT/scripts/train_sft_model.py}"

# How many GPUs to use (>=2 will use torchrun/DDP)
: "${N_GPUS:=1}"

# NCCL nic tuning (optional)
: "${NCCL_DEBUG:=WARN}"
export NCCL_DEBUG

echo "[SFT] Entry: $SFT_FILE | GPUs: $N_GPUS"

if (( N_GPUS > 1 )); then
  torchrun --standalone --nproc_per_node="$N_GPUS" "$SFT_FILE"
else
  python "$SFT_FILE"
fi