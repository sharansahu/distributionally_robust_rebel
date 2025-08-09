#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd -P )"


: "${SFT_FILE:=$REPO_ROOT/scripts/train_sft_model.py}"
: "${REWARD_FILE:=$REPO_ROOT/scripts/train_reward_model.py}"
: "${REBEL_FILE:=$REPO_ROOT/experiments/run_emotion_rebel_experiments.py}"

: "${SFT_GPUS:=1}"
: "${REWARD_GPUS:=1}"
: "${REBEL_GPUS:=8}"
: "${DS_CONFIG:=$REPO_ROOT/ds_zero2.json}"

bash "$(dirname "$0")/run_sft.sh"    SFT_FILE="$SFT_FILE"       N_GPUS="$SFT_GPUS"
bash "$(dirname "$0")/run_reward.sh" REWARD_FILE="$REWARD_FILE" N_GPUS="$REWARD_GPUS"
bash "$(dirname "$0")/run_rebel_local.sh" REBEL_FILE="$REBEL_FILE" DS_CONFIG="$DS_CONFIG" N_GPUS="$REBEL_GPUS"
