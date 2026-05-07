#!/usr/bin/env bash
# train_cloud.sh
#
# Cloud-ready training script for HiF-VLA on Swap-T.
#
# Dataset options (pick ONE):
#   0. Multi-task manifest (each task: zarr + language + dataset_name for stats):
#        TASK_MANIFEST=configs/pusht_tasks_example.json bash train_cloud.sh
#   1. Upload data.zarr.zip to the server and set ZARR_PATH:
#        ZARR_PATH=/workspace/data.zarr.zip bash train_cloud.sh
#   2. Download from HF Hub (requires HF_TOKEN for gated repos):
#        ZARR_PATH="" HF_TOKEN=hf_xxx bash train_cloud.sh
#
# Usage:
#   WANDB_ENTITY=seansyeom3 ZARR_PATH=/workspace/data.zarr.zip bash train_cloud.sh

set -euo pipefail

# -----------------------------------------------------------------------
# Configurable defaults (override via env vars before calling this script)
# -----------------------------------------------------------------------
WANDB_ENTITY="${WANDB_ENTITY:-your-wandb-entity}"
WANDB_PROJECT="${WANDB_PROJECT:-hifvla-pusht}"

# Pretrained VLA checkpoint (HF Hub repo ID or local path).
VLA_PATH="${VLA_PATH:-openvla/openvla-7b}"

# Dataset: set ZARR_PATH to a local file (directory or .zip) to skip HF Hub.
# Leave empty to download from HF Hub instead.
ZARR_PATH="${ZARR_PATH:-}"

# JSON list of tasks (zarr_path, dataset_name, language). Overrides ZARR_PATH / HF mode.
TASK_MANIFEST="${TASK_MANIFEST:-}"

# Training hyper-params
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_STEPS="${MAX_STEPS:-50005}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
LR="${LR:-5e-4}"
LORA_RANK="${LORA_RANK:-32}"
HISTORY="${HISTORY:-8}"

# Rollout: run evaluation every ROLLOUT_FREQ gradient steps (0 = disabled)
ROLLOUT_FREQ="${ROLLOUT_FREQ:-$SAVE_FREQ}"
N_ROLLOUT_EPS="${N_ROLLOUT_EPS:-3}"

# Output directory (relative to project root)
RUN_ROOT="${RUN_ROOT:-runs/pusht_hifvla}"

# -----------------------------------------------------------------------
# Navigate to HiF-VLA directory (required for prismatic imports)
# -----------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/HiF-VLA"

# -----------------------------------------------------------------------
# Memory / CUDA optimisations
# -----------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# -----------------------------------------------------------------------
# Build dataset flags
# -----------------------------------------------------------------------
if [ -n "${TASK_MANIFEST}" ]; then
    DATASET_FLAGS="--task_manifest ../${TASK_MANIFEST}"
    DATASET_LABEL="multi-task ${TASK_MANIFEST}"
elif [ -n "${ZARR_PATH}" ]; then
    # Local dataset (uploaded .zip or directory)
    DATASET_FLAGS="--zarr_path ${ZARR_PATH}"
    DATASET_LABEL="${ZARR_PATH}"
else
    # HF Hub dataset — needs HF_TOKEN for gated repos
    if [ -z "${HF_TOKEN:-}" ]; then
        HF_TOKEN_FILE="${HF_HOME:-$HOME/.cache/huggingface}/token"
        if [ -f "${HF_TOKEN_FILE}" ]; then
            export HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"
            echo "[auth] Loaded HF token from ${HF_TOKEN_FILE}"
        else
            echo "WARNING: HF_TOKEN not set. Download may fail for private/gated repos."
            echo "  Run: huggingface-cli login  OR  export HF_TOKEN=hf_xxxxxxxxxxxx"
        fi
    fi
    DATASET_FLAGS="--use_hf_dataset True --hf_dataset_repo MemoryManip/Memory-T-Bench --hf_dataset_filename swapt-shuffle/data.zarr.zip"
    DATASET_LABEL="MemoryManip/Memory-T-Bench (HF Hub)"
fi

# -----------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------
echo "========================================================"
echo " HiF-VLA Cloud Training"
echo "  VLA            : ${VLA_PATH}"
echo "  Dataset        : ${DATASET_LABEL}"
echo "  W&B entity     : ${WANDB_ENTITY}"
echo "  Run root       : ${RUN_ROOT}"
echo "  Rollout every  : ${ROLLOUT_FREQ} steps"
echo "========================================================"

accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    ../baseline/training/finetune_pusht.py \
    --vla_path          "${VLA_PATH}" \
    ${DATASET_FLAGS} \
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
