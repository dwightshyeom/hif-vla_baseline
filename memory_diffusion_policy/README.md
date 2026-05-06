# Memory_DP — Memory-augmented Diffusion Policy

Memory_DP studies whether a long-horizon **LSTM memory module** improves
[Diffusion Policy (Chi et al., 2024)](https://diffusion-policy.cs.columbia.edu/) on
multi-stage manipulation tasks where the underlying state-only / image-only
observation stream loses crucial history (which goal we already hit, which
sub-task we're in, what the friction profile was up to now, etc.).

The codebase is structured so that:

- Upstream **Diffusion Policy** is consumed verbatim as a git submodule under
  [`third_party/diffusion_policy/`](third_party/diffusion_policy) — we **do
  not** vendor or fork it.
- Our additions live in the [`memory_diffusion_policy/`](memory_diffusion_policy)
  Python package.  A single `pip install -e .` installs both packages so any
  caller can `import memory_diffusion_policy` and `import diffusion_policy`.
- Six policy/modality combinations are first-class:

  |                        | state policy (lowdim)                                                       | vision policy (image)                                              |
  | ---------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------ |
  | **Vanilla DP**         | `train_diffusion_unet_lowdim_workspace.yaml`                                | `train_diffusion_unet_hybrid_workspace.yaml`                       |
  | **Frozen LSTM + DP**   | `train_diffusion_unet_lowdim_obs_action_chunk_lstm_workspace.yaml`          | `train_diffusion_unet_hybrid_lstm_workspace.yaml`                  |
  | **Finetune LSTM + DP** | `train_diffusion_unet_lowdim_obs_action_chunk_lstm_finetune_workspace.yaml` | `train_diffusion_unet_hybrid_lstm_finetune_workspace.yaml`         |

  Plus two LSTM-pretraining workspaces: `train_obs_action_chunk_lstm_workspace.yaml`
  and `train_obs_action_chunk_lstm_image_workspace.yaml`.

- Eight Push-T task variants are maintained: `pusht`, `pusht_asym`,
  `pusht_friction`, `pusht_friction_three_tracks`, `pusht_three_goals`,
  `pusht_three_goals_swap`, `pusht_two_swap`, `pusht_two_goals`.

---

## Repo layout

```
memory_diffusion_policy/      # this package
├── config/                  # hydra workspace and task YAML configs
│   ├── task/                # one YAML per (task × modality × policy variant)
│   └── train_*_workspace.yaml
├── dataset/                 # PushT* + obs_action_chunk_lstm_* datasets
├── env/pusht/               # PushT environment variants (sim)
├── env_runner/              # rollout drivers per task / modality
├── model/
│   ├── obs_action_chunk_lstm.py        # state LSTM
│   └── obs_action_chunk_lstm_image.py  # image LSTM
├── policy/
│   ├── _baseline_template.py           # skeleton for new VLA-style baselines
│   ├── diffusion_unet_lowdim_policy.py
│   ├── diffusion_unet_hybrid_image_policy.py
│   ├── diffusion_unet_lowdim_policy_with_obs_action_chunk_lstm{,_finetune}.py
│   └── diffusion_unet_hybrid_image_policy_with_obs_action_chunk_lstm{,_finetune}.py
├── scripts/
│   └── eval_obs_action_chunk_lstm{,_image}.py   # standalone LSTM-pretrain eval/viz
└── workspace/
    ├── train_diffusion_unet_lowdim_workspace.py
    ├── train_diffusion_unet_hybrid_workspace.py
    ├── train_diffusion_unet_lowdim_obs_action_chunk_lstm{,_finetune}_workspace.py
    ├── train_diffusion_unet_hybrid_lstm{,_finetune}_workspace.py
    └── train_obs_action_chunk_lstm{,_image}_workspace.py   # LSTM pretraining
third_party/
└── diffusion_policy/        # git submodule of real-stanford/diffusion_policy
tests/                       # pytest smoke + config-compose suite
train.py                     # hydra entry point — drives all workspaces
eval.py                      # rollout from a checkpoint
Dockerfile, docker-compose.yaml, docker_train.sh
install_env.sh               # micromamba + submodule + pip install -e .
```

---

## Install

### Option 1: Docker (recommended for training)

```bash
git clone --recurse-submodules <this repo>
cd memory_diffusion_policy
./docker_train.sh build       # builds memory-diffusion-policy:latest
./docker_train.sh shell       # interactive shell with /workspace live-mounted
```

The dev container live-mounts `.` into `/workspace`, so editing source on the
host is immediately visible inside the container.

### Option 2: Local micromamba (one-shot)

```bash
git clone --recurse-submodules <this repo>
cd memory_diffusion_policy
./install_env.sh              # installs micromamba + creates robodiff env + pip install -e .
source ~/.bashrc && micromamba activate robodiff
python -c "import memory_diffusion_policy, diffusion_policy; print('Memory_DP install OK')"
```

`install_env.sh` is idempotent — re-running it on an existing checkout will
just re-install `memory_diffusion_policy` editable.

### Option 3: Local micromamba (manual / step-by-step)

If you'd rather understand each step or `install_env.sh` doesn't fit your
shell, here is the equivalent procedure spelled out:

```bash
# 0. Pull the upstream submodule (needed only if you cloned without --recurse-submodules)
git submodule update --init --recursive

# 1. Install micromamba (linux x86_64 — see https://mamba.readthedocs.io for other archs)
mkdir -p ~/.local/bin
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C ~/.local/bin --strip-components=1 bin/micromamba
export PATH=~/.local/bin:$PATH
export MAMBA_ROOT_PREFIX=~/micromamba
eval "$(micromamba shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba shell init --shell bash --root-prefix $MAMBA_ROOT_PREFIX   # makes the activate command persistent

# 2. Create the robodiff env from the bundled spec
micromamba env create -f conda_environment.yaml -y --channel-priority flexible
micromamba activate robodiff

# 3. Editable-install both packages (memory_diffusion_policy + bundled upstream diffusion_policy)
pip install -e . --no-deps

# 4. Smoke-check
python -c "import memory_diffusion_policy, diffusion_policy; print('Memory_DP install OK')"
```

Use `conda_environment_macos.yaml` instead at step 2 on macOS development
machines (no GPU support — useful for editing only).

### Already cloned without `--recurse-submodules`?

```bash
git submodule update --init --recursive
```

---

## Train recipes

Every training run goes through `train.py`, which is a thin Hydra dispatcher:

```bash
python train.py --config-name=<workspace> [<key>=<value> ...]
```

### LSTM pretraining (do this first if you want frozen / finetune variants)

```bash
# state LSTM (lowdim)
docker compose run --rm train python train.py \
  --config-name=train_obs_action_chunk_lstm_workspace \
  lstm_pretrain.zarr_path=data/pusht_three_goals_demo_with_indicator.zarr \
  lstm_pretrain.num_epochs=1000 lstm_pretrain.use_wandb=true

# image LSTM
docker compose run --rm train python train.py \
  --config-name=train_obs_action_chunk_lstm_image_workspace \
  lstm_pretrain.zarr_path=data/pusht_2d_friction_demos_320.zarr \
  lstm_pretrain.dataset_class=memory_diffusion_policy.dataset.pusht_friction_lstm_image_dataset.PushTFrictionLSTMImageDataset \
  lstm_pretrain.num_epochs=1000 lstm_pretrain.use_wandb=true
```

The pretrain run writes `best_model.pt` and `checkpoint_epoch_NNNN.pt` under
its hydra output directory.

### DP variants (state)

```bash
# vanilla DP, state
python train.py --config-name=train_diffusion_unet_lowdim_workspace \
  task=pusht_lowdim training.seed=42

# frozen-LSTM + DP, state
python train.py --config-name=train_diffusion_unet_lowdim_obs_action_chunk_lstm_workspace \
  task=pusht_three_goals_lowdim_obs_action_chunk_lstm \
  policy.lstm_checkpoint_path=outputs/<lstm_pretrain_run>/best_model.pt

# finetune-LSTM + DP, state
python train.py --config-name=train_diffusion_unet_lowdim_obs_action_chunk_lstm_finetune_workspace \
  task=pusht_three_goals_lowdim_obs_action_chunk_lstm_finetune \
  policy.lstm_checkpoint_path=outputs/<lstm_pretrain_run>/best_model.pt
```

### DP variants (vision)

```bash
# vanilla DP, vision
python train.py --config-name=train_diffusion_unet_hybrid_workspace \
  task=pusht_image_friction_three_tracks

# frozen-LSTM + DP, vision
python train.py --config-name=train_diffusion_unet_hybrid_lstm_workspace \
  task=pusht_image_friction_obs_action_chunk_lstm \
  policy.lstm_checkpoint_path=outputs/<lstm_pretrain_image_run>/best_model.pt

# finetune-LSTM + DP, vision
python train.py --config-name=train_diffusion_unet_hybrid_lstm_finetune_workspace \
  task=pusht_image_three_goals_obs_action_chunk_lstm_finetune \
  policy.lstm_checkpoint_path=outputs/<lstm_pretrain_image_run>/best_model.pt
```

For each task variant we maintain (`pusht_image_friction_three_tracks`,
`pusht_image_three_goals_swap`, etc.), the corresponding LSTM frozen / finetune
config follows the convention `<task>_obs_action_chunk_lstm{,_finetune}.yaml`.
List them with:

```bash
ls memory_diffusion_policy/config/task/
```

---

## Eval

```bash
python eval.py --checkpoint=outputs/<run>/checkpoints/best_model.ckpt \
               --output_dir=outputs/<run>/eval
```

`eval.py` re-instantiates the workspace from the checkpoint's embedded config
and runs the env runner the workspace is paired with — same code path as
training-time validation rollouts.

For real-robot deployment use `eval_real_robot.py` and `demo_real_robot.py`
(both depend on `pyrealsense2` and the UR5 RTDE library, which are not
installed in the default training environment).

---

## Adding a new task

1. **Env** — add `memory_diffusion_policy/env/pusht/pusht_<your_task>_env.py`
   (and an `_image` / `_keypoints` sibling if needed).
2. **Env runner** — add
   `memory_diffusion_policy/env_runner/pusht_<your_task>_{image,keypoints}_runner.py`.
3. **Dataset** — add
   `memory_diffusion_policy/dataset/pusht_<your_task>_{image,lowdim}_dataset.py`.
4. **Task config** — add `memory_diffusion_policy/config/task/<your_task>.yaml`
   wiring the runner + dataset together via Hydra `_target_` strings.
5. **(For LSTM variants)** add the matching
   `<your_task>_obs_action_chunk_lstm{,_finetune}.yaml` task config that swaps
   the dataset to one of the generic
   `obs_action_chunk_lstm{,_image}_{chunked_latent,finetune}_dataset` classes.

Then any of the six DP workspaces will pick the new task up:

```bash
python train.py --config-name=train_diffusion_unet_hybrid_workspace \
  task=<your_task>
```

---

## Adding a new baseline (e.g. a VLA model)

Copy [`memory_diffusion_policy/policy/_baseline_template.py`](memory_diffusion_policy/policy/_baseline_template.py)
to a descriptive name and implement the four override points
(`__init__`, `set_normalizer`, `predict_action`, `compute_loss`).  The two
abstract bases (`BaseImagePolicy`, `BaseLowdimPolicy`) impose only the
`predict_action` / `set_normalizer` contract — your model and training loop
are otherwise free.

Then:

1. Write a workspace at `memory_diffusion_policy/workspace/train_<your_baseline>_workspace.py`
   that subclasses `BaseWorkspace`, instantiates your policy via
   `hydra.utils.instantiate(cfg.policy)`, and runs your training loop.
2. Drop a YAML at `memory_diffusion_policy/config/train_<your_baseline>_workspace.yaml`
   pointing `_target_` at your workspace class.

For inspiration, the LSTM-pretraining workspaces show the pattern of wrapping
a non-DP training loop in a hydra-managed BaseWorkspace
(`memory_diffusion_policy/workspace/train_obs_action_chunk_lstm_workspace.py`).

---

## Tests

```bash
docker compose run --rm dev pytest tests/ -q
```

Runs in a few seconds.  Covers:

- `test_imports.py` — every public module under `memory_diffusion_policy/`
  imports without error, and the upstream submodule resolves.
- `test_configs.py` — every workspace YAML composes against a representative
  task override; every kept `pusht_*` task config loads under its matching
  workspace.

---

## Known limitations / deferred work

- The four LSTM-conditioned DP workspaces still each carry a near-duplicate
  copy of the vanilla-DP training loop.  Factoring out a shared mixin is
  intentionally deferred so the refactor stays bit-for-bit equivalent in
  training behavior.
- `memory_diffusion_policy/scripts/eval_obs_action_chunk_lstm{,_image}.py`
  are still standalone argparse scripts (run them as
  `python -m memory_diffusion_policy.scripts.eval_obs_action_chunk_lstm \
        --checkpoint <path>`).  Folding them into the `eval.py` flow
  is a follow-up.

---

## Acknowledgments

Built on top of [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
by Cheng Chi et al. (Columbia + TRI + MIT).  All upstream code is consumed
verbatim from that submodule; only files that needed substantive
modification live in `memory_diffusion_policy/`.
