#!/usr/bin/env bash
# Run from anywhere inside the repo; the script resolves its own location.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── tuneable defaults (override via environment or edit here) ─────────────────
VLA_PATH="${VLA_PATH:-$REPO_ROOT/openvla-oft/openvla_clean_weights}"
ZARR_PATH="${ZARR_PATH:-$REPO_ROOT/memory_diffusion_policy/swap_3t_dataset_320}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs/pusht_hifvla}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
WANDB_ENTITY="${WANDB_ENTITY:-seansyeom3}"
# ─────────────────────────────────────────────────────────────────────────────

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO_ROOT/HiF-VLA"

accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  "$REPO_ROOT/baseline/training/finetune_pusht.py" \
  --vla_path      "$VLA_PATH" \
  --zarr_path     "$ZARR_PATH" \
  --run_root_dir  "$RUN_ROOT" \
  --batch_size    "$BATCH_SIZE" \
  --grad_accumulation_steps "$GRAD_ACCUM" \
  --wandb_entity  "$WANDB_ENTITY" \
  "$@"
