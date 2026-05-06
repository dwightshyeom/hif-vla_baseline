"""End-to-end smoke tests: instantiate each kept workspace from its YAML and
run a single forward+loss+backward pass.

Two test families:
  * test_dp_workspace_forward_loss — the six DP workspaces from the
    original layout (vanilla / frozen-LSTM / finetune-LSTM × {state, image}),
    parametrised over every kept push-T task variant we have data for.
    Where the task config's default zarr does not exist on disk we
    substitute a structurally-compatible zarr (same keypoint / state / img
    columns) so the dataset class still loads.
  * test_lstm_pretrain_workspace — both LSTM-pretraining workspaces
    (state and image) at the smallest practical hyperparams.

Tests run on GPU when available (cuda:0) — the box has plenty of free
VRAM as long as the parent training job stays under control.  Set
MEMORY_DP_SMOKE_DEVICE=cpu to force the CPU path.

Each test asserts a finite loss and a successful backward pass; the
training loop's correctness is upstream's responsibility.
"""
import os
import pathlib
from typing import Optional

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "memory_diffusion_policy" / "config"

# Pick GPU when available, override via env var.
DEVICE = os.environ.get(
    "MEMORY_DP_SMOKE_DEVICE",
    "cuda:0" if torch.cuda.is_available() else "cpu",
)


def _zarr_exists(rel: str) -> bool:
    return (ROOT / rel).exists()


# --------------------------------------------------------------------------
# Substitutions: every kept task config's intended zarr_path, plus a
# structurally-compatible fallback we have on disk.  Used when the original
# zarr is missing.  None means "don't substitute, skip if missing."
# --------------------------------------------------------------------------
ZARR_SUBSTITUTES = {
    # vanilla pusht_image references a 1-goal demo we don't have on disk;
    # the 3-goal vision zarr has the same image / keypoint structure.
    "data/pusht_one_goal_demo_vision.zarr":
        "data/pusht_three_goals_demo_vision.zarr",
    # friction_three_tracks has its own dataset class but shares the
    # friction column schema with the 320_96 zarr.
    "data/pusht_friction_three_tracks_demo.zarr":
        "data/pusht_2d_friction_demos_320_96.zarr",
    # three_goals_swap reads from data/swap_3t_dataset_320 (without the
    # .zarr suffix) on disk; the YAMLs declare swap_3t_dataset_320.zarr
    # for the vanilla DP variant and three_goals_swap_demo.zarr for the
    # LSTM-conditioned variant.  Map both to the real path.
    "data/three_goals_swap_demo.zarr": "data/swap_3t_dataset_320",
    "data/swap_3t_dataset_320.zarr": "data/swap_3t_dataset_320",
    # lowdim three_goals — same goal_X_keypoint schema lives in the vision
    # zarr, which the lowdim dataset class will read just fine.
    "data/pusht_three_goals_demo_with_indicator.zarr":
        "data/pusht_three_goals_demo_vision.zarr",
    "data/pusht_three_goals_demo_pos_30_rot_0.3.zarr":
        "data/pusht_three_goals_demo_vision.zarr",
    # vanilla pusht_lowdim — needs state + action.  asym has both at the
    # same shape (action=2, state≥2) so it's a safe stand-in.
    "data/my_pusht_demo.zarr": "data/pusht_asym_demos_320_96.zarr",
    # lowdim two-swap config points at swap_t_dataset_160 (zarr or dir);
    # we have swap_t_dataset_320 with the same schema.
    "data/swap_t_dataset_160": "data/swap_t_dataset_320.zarr",
    # lowdim friction — friction zarr is right schema.
    "data/pusht_friction_demo.zarr": "data/pusht_2d_friction_demos_320_96.zarr",
    # two_goals lowdim points at a fixed_two_goals zarr we don't have;
    # vanilla PushTLowdimDataset only needs state+action so any zarr works.
    "data/pusht_fixed_two_goals_demo.zarr":
        "data/pusht_asym_demos_320_96.zarr",
}


def _resolve_zarr_override(task_yaml: pathlib.Path) -> Optional[str]:
    """Read the task YAML's zarr_path; if missing on disk, return a
    Hydra-style override using the substitution table.  None means no
    override needed."""
    import re
    text = task_yaml.read_text()
    m = re.search(r"^\s*zarr_path:\s*(\S+)", text, flags=re.MULTILINE)
    if not m:
        return None
    declared = m.group(1).strip().strip("'\"")
    if _zarr_exists(declared):
        return None
    sub = ZARR_SUBSTITUTES.get(declared)
    if sub is None:
        # No substitute available → skip the test below.
        return "<MISSING>"
    return f"task.dataset.zarr_path={sub}"


# --------------------------------------------------------------------------
# (workspace, task) combos — one per (DP variant × push-T task).
# --------------------------------------------------------------------------
DP_COMBOS = [
    # ---- vanilla DP, image ----
    ("train_diffusion_unet_hybrid_workspace", "pusht_image"),
    ("train_diffusion_unet_hybrid_workspace", "pusht_image_asym"),
    ("train_diffusion_unet_hybrid_workspace", "pusht_image_friction"),
    ("train_diffusion_unet_hybrid_workspace", "pusht_image_friction_three_tracks"),
    ("train_diffusion_unet_hybrid_workspace", "pusht_image_three_goals"),
    ("train_diffusion_unet_hybrid_workspace", "pusht_image_three_goals_swap"),
    ("train_diffusion_unet_hybrid_workspace", "pusht_image_two_swap"),
    # ---- frozen-LSTM + DP, image ----
    ("train_diffusion_unet_hybrid_lstm_workspace",
     "pusht_image_asym_obs_action_chunk_lstm"),
    ("train_diffusion_unet_hybrid_lstm_workspace",
     "pusht_image_friction_obs_action_chunk_lstm"),
    ("train_diffusion_unet_hybrid_lstm_workspace",
     "pusht_image_friction_three_tracks_obs_action_chunk_lstm"),
    ("train_diffusion_unet_hybrid_lstm_workspace",
     "pusht_image_three_goals_obs_action_chunk_lstm"),
    ("train_diffusion_unet_hybrid_lstm_workspace",
     "pusht_image_three_goals_swap_obs_action_chunk_lstm"),
    ("train_diffusion_unet_hybrid_lstm_workspace",
     "pusht_image_two_swap_obs_action_chunk_lstm"),
    # ---- finetune-LSTM + DP, image ----
    ("train_diffusion_unet_hybrid_lstm_finetune_workspace",
     "pusht_image_asym_obs_action_chunk_lstm_finetune"),
    ("train_diffusion_unet_hybrid_lstm_finetune_workspace",
     "pusht_image_friction_obs_action_chunk_lstm_finetune"),
    ("train_diffusion_unet_hybrid_lstm_finetune_workspace",
     "pusht_image_three_goals_obs_action_chunk_lstm_finetune"),
    ("train_diffusion_unet_hybrid_lstm_finetune_workspace",
     "pusht_image_two_swap_obs_action_chunk_lstm_finetune"),
    # ---- vanilla DP, lowdim ----
    ("train_diffusion_unet_lowdim_workspace", "pusht_lowdim"),
    ("train_diffusion_unet_lowdim_workspace", "pusht_lowdim_friction"),
    ("train_diffusion_unet_lowdim_workspace", "pusht_lowdim_two_swap"),
    ("train_diffusion_unet_lowdim_workspace", "pusht_three_goals_lowdim"),
    ("train_diffusion_unet_lowdim_workspace", "pusht_two_goals_lowdim"),
    # ---- frozen / finetune LSTM + DP, lowdim ----
    ("train_diffusion_unet_lowdim_obs_action_chunk_lstm_workspace",
     "pusht_three_goals_lowdim_obs_action_chunk_lstm"),
    ("train_diffusion_unet_lowdim_obs_action_chunk_lstm_finetune_workspace",
     "pusht_three_goals_lowdim_obs_action_chunk_lstm_finetune"),
]


@pytest.fixture(scope="module")
def lstm_pretrain_image_ckpt(tmp_path_factory):
    """Yield the image-LSTM pretrain checkpoint, training one if missing.

    Reuses outputs/_smoke_lstm_image_train/best_model.pt when it exists so
    repeated runs don't pay the pretrain cost.
    """
    cached = ROOT / "outputs" / "_smoke_lstm_image_train" / "best_model.pt"
    if cached.exists():
        return cached

    import hydra
    from hydra import compose, initialize_config_dir

    out_dir = tmp_path_factory.mktemp("lstm_pretrain_image_ckpt")
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="train_obs_action_chunk_lstm_image_workspace",
            overrides=[
                "lstm_pretrain.zarr_path=data/pusht_2d_friction_demos_320_96.zarr",
                "lstm_pretrain.dataset_class=memory_diffusion_policy.dataset."
                "pusht_friction_lstm_image_dataset.PushTFrictionLSTMImageDataset",
                "lstm_pretrain.batch_size=2",
                "lstm_pretrain.num_epochs=1",
                "lstm_pretrain.num_workers=0",
                "lstm_pretrain.use_wandb=false",
                "lstm_pretrain.log_mode=disabled",
                f"lstm_pretrain.device={DEVICE}",
                f"lstm_pretrain.output_dir={out_dir}",
            ],
        )
        cls = hydra.utils.get_class(cfg._target_)
        ws = cls(cfg, output_dir=str(out_dir))
        ws.run()
    ckpt = out_dir / "best_model.pt"
    assert ckpt.exists()
    return ckpt


@pytest.fixture(scope="module")
def lstm_pretrain_lowdim_ckpt(tmp_path_factory):
    """Train the state-LSTM pretrain workspace once and yield its checkpoint."""
    cached = ROOT / "outputs" / "_smoke_lstm_lowdim_train" / "best_model.pt"
    if cached.exists():
        return cached
    sub_zarr = "data/pusht_three_goals_demo_vision.zarr"
    if not _zarr_exists(sub_zarr):
        pytest.skip(f"missing {sub_zarr}")

    import hydra
    from hydra import compose, initialize_config_dir

    out_dir = ROOT / "outputs" / "_smoke_lstm_lowdim_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="train_obs_action_chunk_lstm_workspace",
            overrides=[
                f"lstm_pretrain.zarr_path={sub_zarr}",
                "lstm_pretrain.batch_size=2",
                "lstm_pretrain.num_epochs=1",
                "lstm_pretrain.num_workers=0",
                "lstm_pretrain.use_wandb=false",
                f"lstm_pretrain.device={DEVICE}",
                f"lstm_pretrain.output_dir={out_dir}",
            ],
        )
        cls = hydra.utils.get_class(cfg._target_)
        ws = cls(cfg, output_dir=str(out_dir))
        ws.run()
    return out_dir / "best_model.pt"


def _maybe_skip_for_data(task: str):
    """Skip if the task references a zarr we have no fallback for."""
    yaml_path = CONFIG_DIR / "task" / f"{task}.yaml"
    if not yaml_path.exists():
        pytest.skip(f"task config not found: {yaml_path}")
    sub = _resolve_zarr_override(yaml_path)
    if sub == "<MISSING>":
        pytest.skip(f"no fallback zarr for {task}")
    return sub  # str override or None


@pytest.mark.parametrize("config_name,task", DP_COMBOS)
def test_dp_workspace_forward_loss(
    config_name, task, lstm_pretrain_image_ckpt, lstm_pretrain_lowdim_ckpt, tmp_path
):
    """For each DP combo: instantiate, run one forward+loss+backward pass."""
    zarr_override = _maybe_skip_for_data(task)

    is_lowdim = "lowdim" in config_name
    is_lstm = "lstm" in config_name
    ckpt = lstm_pretrain_lowdim_ckpt if (is_lstm and is_lowdim) else lstm_pretrain_image_ckpt

    overrides = [
        f"task={task}",
        f"training.device={DEVICE}",
        "training.use_ema=false",
        "dataloader.batch_size=2",
        "dataloader.num_workers=0",
        "dataloader.persistent_workers=false",
        "val_dataloader.batch_size=2",
        "val_dataloader.num_workers=0",
        "val_dataloader.persistent_workers=false",
        "logging.mode=disabled",
    ]
    if zarr_override:
        overrides.append(zarr_override)
    if is_lstm:
        overrides.append(f"policy.lstm_checkpoint_path={ckpt}")
        # The lowdim chunked-latent dataset also references an LSTM
        # checkpoint inside the task YAML.  Repoint it to the freshly-
        # trained one rather than the stale absolute path.
        if "obs_action_chunk_lstm_workspace" in config_name and "finetune" not in config_name:
            overrides.append(f"task.dataset.lstm_checkpoint_path={ckpt}")
    # pusht_lowdim_friction has memory_flags=True by default but the
    # friction zarr we have on disk doesn't carry that array.  Disable
    # consistently on dataset + env_runner to match older demos.
    if task == "pusht_lowdim_friction":
        overrides.append("task.dataset.include_memory_flags=False")
        overrides.append("task.env_runner.include_memory_flags=False")
        overrides.append("task.obs_dim=20")

    import hydra
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=config_name, overrides=overrides)

    cls = hydra.utils.get_class(cfg._target_)
    ws = cls(cfg, output_dir=str(tmp_path))
    policy = getattr(ws, "model", None) or getattr(ws, "policy", None)
    assert policy is not None, f"workspace exposes neither .model nor .policy"

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)

    use_cuda = torch.cuda.is_available() and DEVICE.startswith("cuda")
    if use_cuda:
        # Move to device AFTER set_normalizer so normalizer's stat buffers
        # (loaded via state_dict) end up on the same device as the model.
        policy = policy.to(DEVICE)

    loader_kwargs = dict(batch_size=2, num_workers=0, shuffle=True)
    custom_collate = getattr(dataset, "collate_fn", None)
    if custom_collate is not None:
        loader_kwargs["collate_fn"] = custom_collate
    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
    batch = next(iter(loader))

    if use_cuda:
        from diffusion_policy.common.pytorch_util import dict_apply
        batch = dict_apply(batch, lambda x: x.to(DEVICE) if torch.is_tensor(x) else x)

    policy.train()
    loss = policy.compute_loss(batch)
    assert torch.isfinite(loss), f"non-finite loss: {loss.item()}"
    loss.backward()

    # Free GPU memory between tests.
    del policy, loss, batch, ws, loader, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_lstm_pretrain_image_workspace(lstm_pretrain_image_ckpt):
    """The image LSTM pretrain workspace ran end-to-end and wrote a checkpoint."""
    assert lstm_pretrain_image_ckpt.exists()
    state = torch.load(lstm_pretrain_image_ckpt, map_location="cpu")
    assert "model_state" in state and "epoch" in state


def test_lstm_pretrain_lowdim_workspace(lstm_pretrain_lowdim_ckpt):
    """The state LSTM pretrain workspace ran end-to-end and wrote a checkpoint."""
    assert lstm_pretrain_lowdim_ckpt.exists()
    state = torch.load(lstm_pretrain_lowdim_ckpt, map_location="cpu")
    assert "model_state" in state and "epoch" in state
