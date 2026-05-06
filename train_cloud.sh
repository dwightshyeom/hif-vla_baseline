#!/usr/bin/env bash
# train_cloud.sh
#
# Cloud-ready training script for HiF-VLA on Swap-T.
# - Downloads the pretrained VLA from HuggingFace Hub (openvla/openvla-7b)
# - Downloads the training dataset from HF Hub (MemoryManip/Memory-T-Bench)
# - Logs to W&B
# - Runs rollout every 1000 gradient steps and uploads videos to W&B
#
# Usage on cloud server:
#   bash train_cloud.sh [--wandb_entity YOUR_ENTITY] [extra finetune flags]
#
# Prerequisites:
#   pip install -r requirements_cloud.txt
#   wandb login          # paste your API key once
#   huggingface-cli login  # if using gated repos (not needed for public ones)

set -euo pipefail

# -----------------------------------------------------------------------
# Configurable defaults (override via env vars or extra CLI args below)
# -----------------------------------------------------------------------
WANDB_ENTITY="${WANDB_ENTITY:-your-wandb-entity}"
WANDB_PROJECT="${WANDB_PROJECT:-hifvla-pusht}"

# Pretrained checkpoint to fine-tune from (HF Hub repo ID).
VLA_PATH="${VLA_PATH:-openvla/openvla-7b}"

# Training hyper-params optimised for a single 80 GB A100 or similar.
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_STEPS="${MAX_STEPS:-50005}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
LR="${LR:-5e-4}"
LORA_RANK="${LORA_RANK:-32}"
HISTORY="${HISTORY:-8}"

# Rollout: evaluate every SAVE_FREQ steps (set to 0 to disable)
ROLLOUT_FREQ="${ROLLOUT_FREQ:-$SAVE_FREQ}"
N_ROLLOUT_EPS="${N_ROLLOUT_EPS:-3}"

# Output directory
RUN_ROOT="${RUN_ROOT:-runs/pusht_hifvla}"

# -----------------------------------------------------------------------
# Navigate to HiF-VLA directory (training script expects this)
# -----------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/HiF-VLA"

# -----------------------------------------------------------------------
# Memory / CUDA optimisations
# -----------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# -----------------------------------------------------------------------
# Launch training
# -----------------------------------------------------------------------
echo "========================================================"
echo " HiF-VLA Cloud Training"
echo "  VLA checkpoint : ${VLA_PATH}"
echo "  Dataset        : MemoryManip/Memory-T-Bench (HF Hub)"
echo "  W&B entity     : ${WANDB_ENTITY}"
echo "  Run root       : ${RUN_ROOT}"
echo "  Rollout every  : ${ROLLOUT_FREQ} steps"
echo "========================================================"

accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    ../baseline/training/finetune_pusht.py \
    --vla_path          "${VLA_PATH}" \
    --use_hf_dataset    True \
    --hf_dataset_repo   "MemoryManip/Memory-T-Bench" \
    --hf_dataset_filename "swapt-shuffle/data.zarr.zip" \
    --run_root_dir      "../${RUN_ROOT}" \
    --batch_size        "${BATCH_SIZE}" \
    --grad_accumulation_steps "${GRAD_ACCUM}" \
    --max_steps         "${MAX_STEPS}" \
    --save_freq         "${SAVE_FREQ}" \
    --learning_rate     "${LR}" \
    --lora_rank         "${LORA_RANK}" \
    --history_length    "${HISTORY}" \
    --rollout_freq      "${ROLLOUT_FREQ}" \
    --n_rollout_episodes "${N_ROLLOUT_EPS}" \
    --wandb_entity      "${WANDB_ENTITY}" \
    --wandb_project     "${WANDB_PROJECT}" \
    "$@"
