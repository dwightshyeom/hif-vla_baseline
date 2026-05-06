#!/usr/bin/env bash
# Evaluate a HiF-VLA Push-T checkpoint.
# Run from anywhere; the script resolves its own location.
#
# Quickstart:
#   ./eval_pusht.sh --pretrained_checkpoint runs/pusht_hifvla/<ckpt_dir>
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── tuneable defaults (override via environment or edit here) ─────────────────
ZARR_PATH="${ZARR_PATH:-$REPO_ROOT/memory_diffusion_policy/swap_3t_dataset_320}"
NUM_EPISODES="${NUM_EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-500}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/experiments/logs/pusht}"
# ─────────────────────────────────────────────────────────────────────────────

cd "$REPO_ROOT/HiF-VLA"

python "$REPO_ROOT/baseline/eval/run_pusht_eval.py" \
  --zarr_path     "$ZARR_PATH" \
  --num_episodes  "$NUM_EPISODES" \
  --max_steps     "$MAX_STEPS" \
  --log_dir       "$LOG_DIR" \
  "$@"
