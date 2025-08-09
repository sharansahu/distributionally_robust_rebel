#!/usr/bin/env bash
set -euo pipefail

#############################################
# Usage:
#   bash scripts/run_armorm.sh MODE
#   MODE ∈ {score, align, both}
#############################################

# --- resolve repo root ---
SCRIPT_DIR="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd -P )"

MODE="${1:-align}"  # score | align | both

# --- conda env ---
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-dro-rebel}"

# --- paths to python entrypoints ---
SCORE_PY="${SCORE_PY:-$REPO_ROOT/scripts/score_armorm_distributed.py}"
ALIGN_PY="${ALIGN_PY:-$REPO_ROOT/experiments/run_armorm_alignment.py}"

# --- models/datasets ---
ARMORM_MODEL="${ARMORM_MODEL:-RLHFlow/ArmoRM-Llama3-8B-v0.1}"
BASE_POLICY="${BASE_POLICY:-meta-llama/Llama-3.2-1B-Instruct}"
DATASET_ID="${DATASET_ID:-HuggingFaceH4/helpsteer2}"   # scorer can also take a local folder

# --- deepspeed config for alignment ---
DS_CONFIG="${DS_CONFIG:-$REPO_ROOT/configs/ds_zero2.json}"

# --- scorer settings (torchrun DDP) ---
SCORE_GPUS="${SCORE_GPUS:-8}"
SCORE_SPLIT="${SCORE_SPLIT:-train}"
SCORE_INPUT_KEY="${SCORE_INPUT_KEY:-prompt}"
SCORE_OUTPUT_KEY="${SCORE_OUTPUT_KEY:-response}"
SCORE_MAX_LEN="${SCORE_MAX_LEN:-4096}"
SCORE_MICRO_BS="${SCORE_MICRO_BS:-2}"
SCORE_OUTDIR="${SCORE_OUTDIR:-$REPO_ROOT/armorm_scores}"
SCORE_FLASH_ATTN="${SCORE_FLASH_ATTN:-0}"        # 1 to enable flash-attn2
SCORE_DTYPE="${SCORE_DTYPE:-bf16}"               # bf16|fp16|fp32
SCORE_USE_DS_INFER="${SCORE_USE_DS_INFER:-0}"    # 1 to enable DS kernel injection

# --- alignment settings (DeepSpeed ZeRO-2) ---
ALIGN_GPUS="${ALIGN_GPUS:-8}"
VARIANT="${VARIANT:-DPO}"   # DPO | WDPO | KL-DPO | REBEL | W-REBEL | KL-REBEL | CHI-REBEL
SEEN_OBJECTIVES="${SEEN_OBJECTIVES:-helpsteer-helpfulness,helpsteer-correctness}"
EVAL_OBJECTIVES="${EVAL_OBJECTIVES:-helpsteer-helpfulness,helpsteer-correctness,helpsteer-coherence,ultrafeedback-truthfulness,ArmoRM}"
TRAIN_PROMPTS="${TRAIN_PROMPTS:-512}"
TEST_PROMPTS="${TEST_PROMPTS:-128}"
ITERATIONS="${ITERATIONS:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-5}"
ETA="${ETA:-0.01}"
RHO0="${RHO0:-0.5}"
TAU="${TAU:-0.1}"
ALPHA0="${ALPHA0:-0.15}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
SEED="${SEED:-1234}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output_armorm}"
POLICY_FLASH_ATTN="${POLICY_FLASH_ATTN:-0}"      # 1 to enable flash-attn2 in policy
ARMORM_FLASH_ATTN="${ARMORM_FLASH_ATTN:-0}"      # 1 to enable flash-attn2 in ArmoRM scorer

# --- niceties ---
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1

# --- activate conda ---
if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
else
  echo "[warn] conda activation script not found at $CONDA_SH; continuing without activation."
fi

# --- ensure DS config exists ---
if [[ ! -f "$DS_CONFIG" ]]; then
  mkdir -p "$(dirname "$DS_CONFIG")"
  cat > "$DS_CONFIG" <<'JSON'
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": 4,
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_scatter": true
  },
  "bf16": { "enabled": true },
  "fp16": { "enabled": false, "loss_scale": 512 },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-5,
      "betas": [0.9, 0.98],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0.0,
      "warmup_max_lr": 1e-5,
      "warmup_num_steps": 1000,
      "total_num_steps": 100000
    }
  },
  "gradient_clipping": 1.0,
  "wall_clock_breakdown": false
}
JSON
  echo "[info] Wrote default DeepSpeed config to $DS_CONFIG"
fi

run_score() {
  mkdir -p "$SCORE_OUTDIR"

  torchrun --standalone --nproc_per_node="$SCORE_GPUS" "$SCORE_PY" \
    --armorm_model "$ARMORM_MODEL" \
    --dataset "$DATASET_ID" \
    --dataset_split "$SCORE_SPLIT" \
    --input_key "$SCORE_INPUT_KEY" \
    --output_key "$SCORE_OUTPUT_KEY" \
    --output_path "$SCORE_OUTDIR" \
    --micro_batch_size "$SCORE_MICRO_BS" \
    --max_len "$SCORE_MAX_LEN" \
    --dtype "$SCORE_DTYPE" \
    $( [[ "$SCORE_FLASH_ATTN" == "1" ]] && echo --flash_attn ) \
    $( [[ "$SCORE_USE_DS_INFER" == "1" ]] && echo --use_deepspeed_inference )
}

run_align() {
  mkdir -p "$OUTPUT_DIR"

  deepspeed --num_gpus="$ALIGN_GPUS" "$ALIGN_PY" \
    --deepspeed_config "$DS_CONFIG" \
    --base_policy "$BASE_POLICY" \
    --dataset_id "$DATASET_ID" \
    --armorm_id "$ARMORM_MODEL" \
    --variant "$VARIANT" \
    --seen_objectives "$SEEN_OBJECTIVES" \
    --eval_objectives "$EVAL_OBJECTIVES" \
    --train_prompts "$TRAIN_PROMPTS" \
    --test_prompts "$TEST_PROMPTS" \
    --iterations "$ITERATIONS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --eta "$ETA" \
    --rho0 "$RHO0" \
    --tau "$TAU" \
    --alpha0 "$ALPHA0" \
    --max_seq_length "$MAX_SEQ_LEN" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR" \
    $( [[ "$POLICY_FLASH_ATTN" == "1" ]] && echo --policy_flash_attn ) \
    $( [[ "$ARMORM_FLASH_ATTN" == "1" ]] && echo --armorm_flash_attn )
}

case "$MODE" in
  score)
    echo "[run] scoring with ArmoRM on $DATASET_ID ($SCORE_SPLIT)"
    run_score
    ;;
  align)
    echo "[run] alignment ($VARIANT) with online ArmoRM scoring"
    run_align
    ;;
  both)
    echo "[run] scoring first, then alignment"
    run_score
    run_align
    ;;
  *)
    echo "Unknown MODE: $MODE (expected: score | align | both)"
    exit 1
    ;;
esac

echo "[done] $MODE complete."
