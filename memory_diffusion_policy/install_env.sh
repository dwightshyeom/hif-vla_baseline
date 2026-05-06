#!/usr/bin/env bash
# =============================================================================
# install_env.sh — Set up Memory_DP locally (without Docker)
# =============================================================================
#
#   1. fetch the bundled upstream diffusion_policy submodule
#   2. install micromamba (idempotent)
#   3. create the `robodiff` conda environment
#   4. pip install -e . — exposes both `memory_diffusion_policy` and the
#      bundled upstream `diffusion_policy` on sys.path
#
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

MAMBA_ROOT_PREFIX="${HOME}/micromamba"
MAMBA_BIN_DIR="${HOME}/.local/bin"
MAMBA_EXE="${MAMBA_BIN_DIR}/micromamba"

export MAMBA_ROOT_PREFIX
export PATH="${MAMBA_BIN_DIR}:${PATH}"

# ---------- 0. Pull the upstream diffusion_policy submodule ----------
echo ">>> Initializing third_party/diffusion_policy submodule..."
git submodule update --init --recursive

# ---------- 1. Install micromamba (idempotent) ----------
if [ ! -x "${MAMBA_EXE}" ]; then
    echo ">>> Installing micromamba..."
    mkdir -p "${MAMBA_BIN_DIR}"
    ARCH="$(uname -m)"
    case "${ARCH}" in
        x86_64)  MM_ARCH="64" ;;
        aarch64) MM_ARCH="aarch64" ;;
        arm64)   MM_ARCH="aarch64" ;;
        *) echo "Unsupported arch: ${ARCH}"; exit 1 ;;
    esac
    curl -Ls "https://micro.mamba.pm/api/micromamba/linux-${MM_ARCH}/latest" \
        | tar -xvj -C "${MAMBA_BIN_DIR}" --strip-components=1 bin/micromamba
    chmod +x "${MAMBA_EXE}"
else
    echo ">>> micromamba already installed at ${MAMBA_EXE}, skipping."
fi

# Shell hook for this script + bashrc init for future sessions (idempotent)
eval "$("${MAMBA_EXE}" shell hook --shell bash --root-prefix "${MAMBA_ROOT_PREFIX}")"
"${MAMBA_EXE}" shell init --shell bash --root-prefix "${MAMBA_ROOT_PREFIX}" || true

# ---------- 2. Create the environment ----------
echo ">>> Creating robodiff environment from conda_environment.yaml..."
"${MAMBA_EXE}" env create -f conda_environment.yaml -y --channel-priority flexible || \
    echo ">>> robodiff environment already exists, skipping."

# ---------- 3. Activate and pip install -e . ----------
"${MAMBA_EXE}" activate robodiff
echo ">>> Installing memory_diffusion_policy + bundled upstream submodule..."
pip install -e . --no-deps

echo ""
echo ">>> Done! Activate with:"
echo "    source ~/.bashrc && micromamba activate robodiff"
echo ""
echo ">>> Quick smoke test:"
echo '    python -c "import memory_diffusion_policy, diffusion_policy; print(\"Memory_DP install OK\")"'
