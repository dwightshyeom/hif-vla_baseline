# =============================================================================
# Dockerfile for Memory_DP (memory_diffusion_policy)
# Base: NVIDIA CUDA 11.8 + Ubuntu 20.04 + Mambaforge (fast solver)
# =============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# ---------- System packages ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    build-essential cmake pkg-config \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    libosmesa6-dev libgl1-mesa-dev libglfw3 patchelf \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---------- Install Mambaforge (mamba = 10-100x faster than conda solver) ----------
ENV CONDA_DIR=/opt/conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/download/23.3.1-1/Mambaforge-23.3.1-1-Linux-x86_64.sh -O /tmp/mambaforge.sh \
    && bash /tmp/mambaforge.sh -b -p $CONDA_DIR \
    && rm /tmp/mambaforge.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# ---------- Create conda environment (using mamba for fast solving) ----------
WORKDIR /workspace
COPY conda_environment_docker.yaml /workspace/conda_environment_docker.yaml
RUN mamba env create -f conda_environment_docker.yaml \
    && mamba clean -afy \
    && find $CONDA_DIR -name '*.pyc' -delete

# Activate the conda env by default
ENV CONDA_DEFAULT_ENV=robodiff
ENV PATH=$CONDA_DIR/envs/robodiff/bin:$PATH

# Trust /workspace for git (W&B runs git rev-parse at startup)
RUN git config --global --add safe.directory /workspace \
    && git config --global --add safe.directory /workspace/third_party/diffusion_policy

# ---------- Copy project code ----------
# memory_diffusion_policy = our package; third_party/diffusion_policy = upstream
# fork (real-stanford/diffusion_policy) consumed via git submodule.
COPY setup.py /workspace/setup.py
COPY memory_diffusion_policy/ /workspace/memory_diffusion_policy/
COPY third_party/ /workspace/third_party/

# A single editable install of memory_diffusion_policy also exposes the
# bundled upstream `diffusion_policy` package (see setup.py).
RUN pip install -e . --no-deps

# Copy remaining project files (top-level entry points, demo scripts, configs)
COPY train.py eval.py /workspace/
COPY rollout_checkpoint.py /workspace/
COPY demo_*.py diagnose_*.py *.py /workspace/

# ---------- Default command ----------
CMD ["bash"]
