"""Hydra-compose every canonical workspace YAML against representative tasks.

Verifies that:
  * every kept workspace config can be composed,
  * its `_target_` resolves to an instantiable class,
  * task overrides for every kept push-T variant work end-to-end.

This catches mismatches between workspace and task configs (e.g. a task
referencing a deleted dataset class), missing keys, or stale `_target_`
strings — without paying for actually running training.
"""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "memory_diffusion_policy" / "config"


# (workspace_config_name, task_override) — covers all 6 DP combos and both
# LSTM-pretrain workspaces, against a representative push-T task each.
WORKSPACE_TASK_PAIRS = [
    # vanilla DP
    ("train_diffusion_unet_lowdim_workspace", "task=pusht_lowdim"),
    ("train_diffusion_unet_hybrid_workspace", "task=pusht_image_friction_three_tracks"),
    # frozen LSTM + DP
    ("train_diffusion_unet_lowdim_obs_action_chunk_lstm_workspace",
     "task=pusht_three_goals_lowdim_obs_action_chunk_lstm"),
    ("train_diffusion_unet_hybrid_lstm_workspace",
     "task=pusht_image_friction_obs_action_chunk_lstm"),
    # finetune LSTM + DP
    ("train_diffusion_unet_lowdim_obs_action_chunk_lstm_finetune_workspace",
     "task=pusht_three_goals_lowdim_obs_action_chunk_lstm_finetune"),
    ("train_diffusion_unet_hybrid_lstm_finetune_workspace",
     "task=pusht_image_three_goals_obs_action_chunk_lstm_finetune"),
    # LSTM pretraining (no task override needed; configs are self-contained)
    ("train_obs_action_chunk_lstm_workspace", None),
    ("train_obs_action_chunk_lstm_image_workspace", None),
]


@pytest.mark.parametrize("config_name,override", WORKSPACE_TASK_PAIRS)
def test_workspace_config_composes(config_name, override):
    import hydra
    from hydra import compose, initialize_config_dir

    overrides = [override] if override else []
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=config_name, overrides=overrides)
        assert cfg._target_.startswith("memory_diffusion_policy."), cfg._target_
        cls = hydra.utils.get_class(cfg._target_)
        assert callable(cls), f"_target_ {cfg._target_} is not callable"


# Every task config we kept must still be parseable under the appropriate
# DP workspace.  This catches per-task regressions even if the workspace
# config above passed.
ALL_KEPT_TASK_CONFIGS = sorted(
    p.stem for p in (CONFIG_DIR / "task").glob("pusht_*.yaml")
)


@pytest.mark.parametrize("task", ALL_KEPT_TASK_CONFIGS)
def test_each_kept_task_config_loads(task):
    """Each task config loads under whichever DP workspace matches its name."""
    if "obs_action_chunk_lstm_finetune" in task:
        ws = (
            "train_diffusion_unet_hybrid_lstm_finetune_workspace"
            if "image" in task
            else "train_diffusion_unet_lowdim_obs_action_chunk_lstm_finetune_workspace"
        )
    elif "obs_action_chunk_lstm" in task:
        ws = (
            "train_diffusion_unet_hybrid_lstm_workspace"
            if "image" in task
            else "train_diffusion_unet_lowdim_obs_action_chunk_lstm_workspace"
        )
    elif "image" in task:
        ws = "train_diffusion_unet_hybrid_workspace"
    else:
        ws = "train_diffusion_unet_lowdim_workspace"

    import hydra
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=ws, overrides=[f"task={task}"])
        # Sanity: the task block exists and carries at least a dataset target.
        assert "dataset" in cfg.task, f"task {task} missing dataset block"
        assert cfg.task.dataset.get("_target_"), \
            f"task {task} dataset._target_ unset"
