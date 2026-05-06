#!/usr/bin/env bash
# =============================================================================
#  docker_train.sh — Memory_DP training launcher
#
#  Usage:
#    ./docker_train.sh build
#         Build (or rebuild) the Docker image.
#
#    ./docker_train.sh shell
#         Open an interactive bash shell inside the dev container.
#
#    ./docker_train.sh train <hydra args...>
#         Run `python train.py ...` inside the train container.  Examples:
#
#         ./docker_train.sh train --config-name=train_obs_action_chunk_lstm_image_workspace \
#             lstm_pretrain.zarr_path=data/pusht_2d_friction_demos_320.zarr \
#             lstm_pretrain.num_epochs=2
#
#         ./docker_train.sh train --config-name=train_diffusion_unet_hybrid_lstm_workspace \
#             task=pusht_image_friction_three_tracks_obs_action_chunk_lstm \
#             policy.lstm_checkpoint_path=outputs/<run>/best_model.pt
#
#  Environment variables (optional):
#    WANDB_API_KEY   — pass your W&B key for experiment tracking
#    GPU             — which GPU to use (e.g. GPU=1; default: all)
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'

usage() {
    echo -e "${CYAN}Usage:${NC} $0 {build|train|shell} [extra args...]"
    echo ""
    echo "  build              Build the Docker image"
    echo "  train  <args...>   Run python train.py inside the train service with"
    echo "                     the given hydra arguments"
    echo "  shell              Open an interactive bash shell inside the dev service"
    echo ""
    echo "Environment variables:"
    echo "  WANDB_API_KEY      Your Weights & Biases API key"
    echo "  GPU                GPU index (default: all). e.g. GPU=0"
    exit 1
}

[[ $# -lt 1 ]] && usage

ACTION="$1"; shift

if [[ -n "${GPU:-}" ]]; then
    export NVIDIA_VISIBLE_DEVICES="$GPU"
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"

case "$ACTION" in
    build)
        echo -e "${GREEN}[*] Building Docker image...${NC}"
        docker compose build "$@"
        echo -e "${GREEN}[*] Build complete.${NC}"
        ;;
    train)
        echo -e "${GREEN}[*] Starting training: python train.py $*${NC}"
        docker compose run --rm train python train.py "$@"
        ;;
    shell)
        echo -e "${GREEN}[*] Opening interactive shell...${NC}"
        docker compose run --rm dev "$@"
        ;;
    *)
        echo -e "${RED}Unknown action: $ACTION${NC}"
        usage
        ;;
esac
