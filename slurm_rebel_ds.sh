#!/usr/bin/env bash
#SBATCH -J rebel_ds
#SBATCH -A your_account            # <-- change to your allocation
#SBATCH -p your_partition          # <-- e.g., gpu, a100, h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 48:00:00
#SBATCH -o logs/rebel_%j.out
#SBATCH -e logs/rebel_%j.err

set -euo pipefail

# --- locate repo relative to this script ---
SCRIPT_DIR="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd -P )"

mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/configs"

# --- (optional) cluster modules ---
module purge
# module load cuda/12.1
# module load anaconda/2024.06

# --- activate conda env: dro-rebel ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dro-rebel

# --- files ---
REBEL_FILE="${REBEL_FILE:=$REPO_ROOT/experiments/run_emotion_rebel_experiments.py}"
DS_CONFIG="${DS_CONFIG:-$REPO_ROOT/ds_zero2.json}"

# --- niceties ---
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1

echo "JOB $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Running: $REBEL_FILE"
echo "DS config: $DS_CONFIG"

# --- launch DeepSpeed (single node) ---
deepspeed \
  --num_nodes=1 \
  --num_gpus="${SLURM_GPUS_ON_NODE:-8}" \
  "$REBEL_FILE" \
  --deepspeed_config "$DS_CONFIG"

