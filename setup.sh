#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# DRO–REBEL bootstrap script
# Creates conda env "dro-rebel", installs Torch + deps, and
# writes a default DeepSpeed ZeRO-2 config.
#
# Usage:
#   bash scripts/setup.sh
#
# Customization via env vars:
#   CONDA_SH=~/miniconda3/etc/profile.d/conda.sh
#   CONDA_ENV=dro-rebel
#   PYTHON_VERSION=3.10
#   TORCH_CHANNEL=auto    # auto | cu121 | cu124 | cpu
#   INSTALL_FLASH_ATTN=0  # 1 to install flash-attn
#   INSTALL_BITSANDBYTES=0# 1 to install bitsandbytes
# ------------------------------------------------------------

SCRIPT_DIR="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd -P )"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-dro-rebel}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_CHANNEL="${TORCH_CHANNEL:-auto}"   # auto-detect GPU and default to cu121 if present
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
INSTALL_BITSANDBYTES="${INSTALL_BITSANDBYTES:-0}"

echo "[info] repo: $REPO_ROOT"
echo "[info] conda: $CONDA_SH"
echo "[info] env:   $CONDA_ENV (python $PYTHON_VERSION)"
echo "[info] torch channel: $TORCH_CHANNEL"

# ----- ensure conda is available -----
if [[ ! -f "$CONDA_SH" ]]; then
  echo "[error] Could not find conda activation script at: $CONDA_SH"
  echo "        Set CONDA_SH to your Conda path and re-run."
  exit 1
fi
# shellcheck disable=SC1090
source "$CONDA_SH"

# ----- create/activate env -----
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "[info] conda env '$CONDA_ENV' already exists."
else
  echo "[info] creating conda env '$CONDA_ENV'..."
  conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION"
fi
conda activate "$CONDA_ENV"

# ----- decide torch channel if auto -----
if [[ "$TORCH_CHANNEL" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[info] NVIDIA GPU detected via nvidia-smi — choosing cu121 by default."
    TORCH_CHANNEL="cu121"
  else
    echo "[info] No NVIDIA GPU detected — choosing cpu build."
    TORCH_CHANNEL="cpu"
  fi
fi

# ----- install torch per channel -----
echo "[info] installing torch for channel: $TORCH_CHANNEL"
pip install --upgrade pip wheel setuptools

case "$TORCH_CHANNEL" in
  cu121)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ;;
  cu124)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ;;
  cpu)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ;;
  *)
    echo "[error] Unknown TORCH_CHANNEL: $TORCH_CHANNEL (expected auto|cu121|cu124|cpu)"
    exit 1
    ;;
esac

# ----- project deps -----
echo "[info] installing project dependencies..."
pip install \
  transformers \
  datasets \
  accelerate \
  deepspeed \
  sentencepiece \
  numpy \
  pandas \
  scikit-learn \
  tqdm \
  pyarrow

# Optional extras
if [[ "$INSTALL_FLASH_ATTN" == "1" ]]; then
  echo "[info] installing flash-attn (requires matching CUDA/SM); this may take a while..."
  pip install flash-attn --no-build-isolation
fi

if [[ "$INSTALL_BITSANDBYTES" == "1" ]]; then
  echo "[info] installing bitsandbytes..."
  pip install bitsandbytes
fi

# ----- write default DeepSpeed ZeRO-2 config if missing -----
mkdir -p "$REPO_ROOT/configs"
DS_CFG="$REPO_ROOT/configs/ds_zero2.json"
if [[ ! -f "$DS_CFG" ]]; then
  cat > "$DS_CFG" <<'JSON'
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
  echo "[info] wrote default DeepSpeed config to $DS_CFG"
else
  echo "[info] found existing DeepSpeed config at $DS_CFG"
fi

# ----- sanity check -----
python - <<'PY'
import torch, transformers, datasets, deepspeed, sys
print("[ok] torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("[ok] cuda available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
print("[ok] transformers:", transformers.__version__)
print("[ok] datasets:", datasets.__version__)
print("[ok] deepspeed:", deepspeed.__version__)
PY

echo
echo "[done] setup complete."
echo "       Env:    $CONDA_ENV"
echo "       Torch:  $TORCH_CHANNEL"
echo "       Config: $DS_CFG"
echo
echo "Next steps:"
echo "  - (Optional) huggingface-cli login"
echo "  - Run alignment:  bash scripts/run_armorm.sh align"
echo "  - Or score-only:  bash scripts/run_armorm.sh score"
