"""
Standalone rollout / eval script for diffusion-policy checkpoints.

Mirrors the in-training rollout block found in every workspace's run() method:

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval()
    runner_log = env_runner.run(policy, epoch=current_epoch)

so the metrics produced here are bit-comparable with the metrics that would
have been logged at that epoch had `rollout_every` been set low enough.

Use this when you trained with `rollout_every` very high (or larger than
`num_epochs`) to skip mid-training rollouts and want to evaluate at the
end (or at any saved checkpoint) without re-launching training.

Usage:
    # Plain eval, EMA model, default config from the checkpoint
    python rollout_checkpoint.py \\
        --checkpoint data/outputs/.../checkpoints/latest.ckpt \\
        --output_dir data/eval_runs/run_001

    # Override env-runner knobs without editing yaml
    python rollout_checkpoint.py -c <ckpt> -o <out> \\
        --override task.env_runner.n_test=200 \\
        --override task.env_runner.test_start_seed=200000 \\
        --override task.env_runner.max_steps=800

    # Skip wandb video upload, save videos to disk under <out>/epoch_*_rollout/
    python rollout_checkpoint.py -c <ckpt> -o <out> --rollout_mode local

    # Use the online (non-EMA) model
    python rollout_checkpoint.py -c <ckpt> -o <out> --no-ema

    # Tag this rollout with a fake "epoch" so past-action PDFs / video
    # filenames don't collide if you eval the same ckpt multiple times
    python rollout_checkpoint.py -c <ckpt> -o <out> --epoch 9999

What's saved in <output_dir>:
    eval_log.json                    runner_log with wandb.Video objects
                                     replaced by their underlying file paths.
    eval_summary.json                aggregated metrics: train_mean_score,
                                     test_mean_score, success counts, plus
                                     the resolved CLI args / overrides.
    media/*.mp4                      env-rendered videos (wandb mode).
    epoch_<E>_rollout/*.mp4          env-rendered videos (local mode).
    past_action_viz/*.pdf            past-action PDFs (LSTM-aware policies).
"""
import sys
# Line-buffered stdout/stderr so progress lines flush in real time.
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import json
import pathlib
import time
import collections

import click
import dill
import hydra
import torch
import wandb
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace


# ---------------------------------------------------------------------------
# Override parsing
# ---------------------------------------------------------------------------
def _parse_value(v: str):
    """Light type coercion for CLI overrides: ints, floats, bools, None, str."""
    if v in ("True", "true"):
        return True
    if v in ("False", "false"):
        return False
    if v in ("None", "null", "~"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _apply_override(cfg, dotted_key: str, value_str: str):
    """OmegaConf dotted-key assignment, e.g. cfg.task.env_runner.n_test = 200."""
    OmegaConf.update(cfg, dotted_key, _parse_value(value_str), merge=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _summarize_runner_log(runner_log: dict) -> dict:
    """Aggregate a runner_log into headline numbers per prefix."""
    prefixes = collections.defaultdict(lambda: {
        "n_envs": 0,
        "n_success": 0,
        "n_videos": 0,
        "max_rewards": [],
    })
    for k, v in runner_log.items():
        if "_mean_score" in k:
            continue
        if k.startswith("train/sim_success_") or k.startswith("test/sim_success_"):
            prefix = k.split("/", 1)[0] + "/"
            prefixes[prefix]["n_envs"] += 1
            prefixes[prefix]["n_success"] += int(float(v) >= 1.0)
        elif k.startswith("train/sim_max_reward_") or k.startswith("test/sim_max_reward_"):
            prefix = k.split("/", 1)[0] + "/"
            prefixes[prefix]["max_rewards"].append(float(v))
        elif "sim_video_" in k:
            prefix = k.split("/", 1)[0] + "/"
            prefixes[prefix]["n_videos"] += 1

    summary = {}
    for prefix, agg in prefixes.items():
        rewards = agg["max_rewards"]
        if rewards:
            summary[f"{prefix}mean_max_reward"] = sum(rewards) / len(rewards)
            summary[f"{prefix}min_max_reward"] = min(rewards)
            summary[f"{prefix}max_max_reward"] = max(rewards)
        if agg["n_envs"] > 0:
            summary[f"{prefix}success_rate"] = agg["n_success"] / agg["n_envs"]
            summary[f"{prefix}n_envs"] = agg["n_envs"]
            summary[f"{prefix}n_success"] = agg["n_success"]
            summary[f"{prefix}n_videos"] = agg["n_videos"]
    # Mean-score keys produced directly by the runner.
    for k, v in runner_log.items():
        if k.endswith("/mean_score"):
            summary[k] = float(v)
    return summary


def _serialize_runner_log_for_json(runner_log: dict) -> dict:
    """Convert wandb.Video values to their on-disk paths so the dict is JSONable."""
    out = {}
    for k, v in runner_log.items():
        try:
            if isinstance(v, wandb.sdk.data_types.video.Video):
                out[k] = getattr(v, "_path", str(v))
            else:
                # Probe with json.dumps; fall back to str() for anything exotic.
                json.dumps(v)
                out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command(context_settings=dict(show_default=True))
@click.option('-c', '--checkpoint', required=True, type=click.Path(exists=True),
              help='Path to the .ckpt file produced by training.')
@click.option('-o', '--output_dir', required=True, type=click.Path(),
              help='Directory for eval_log.json + videos. Must NOT already '
                   'exist (or pass --force-output to overwrite).')
@click.option('-d', '--device', default='cuda:0',
              help='Torch device (e.g. cuda:0, cpu).')
@click.option('--ema/--no-ema', default=None,
              help='Force EMA (online) weights regardless of cfg.training.use_ema. '
                   'Default: follow the value baked into the checkpoint.')
@click.option('--epoch', default=0, type=int,
              help='Pseudo-epoch passed to env_runner.run(epoch=...). Affects '
                   'past-action-viz / per-epoch rollout folder names so '
                   'consecutive runs against the same checkpoint do not '
                   'overwrite each other.')
@click.option('--rollout_mode', default=None, type=click.Choice(['wandb', 'local']),
              help='Force rollout_mode=wandb|local (overrides cfg.task.env_runner.rollout_mode).')
@click.option('--n_envs', default=None, type=int,
              help='Override task.env_runner.n_envs (parallel envs per chunk).')
@click.option('--override', '-O', 'overrides', multiple=True,
              help='Hydra-style dotted override, e.g. '
                   '`-O task.env_runner.n_test=200`. May be repeated.')
@click.option('--force-output', is_flag=True, default=False,
              help='Overwrite output_dir if it already exists.')
@click.option('--dry-run', is_flag=True, default=False,
              help='Build everything but skip the actual env_runner.run() call.')
def main(checkpoint, output_dir, device, ema, epoch, rollout_mode, n_envs,
         overrides, force_output, dry_run):
    t0 = time.time()
    out = pathlib.Path(output_dir)
    if out.exists():
        if not force_output:
            raise click.ClickException(
                f"Output path {output_dir!r} already exists. "
                f"Pass --force-output to overwrite, or pick a fresh path.")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[rollout] output_dir={out}")

    # ------------------------------------------------------------------
    # Load checkpoint payload — gives us cfg + state_dicts + pickles.
    # ------------------------------------------------------------------
    print(f"[rollout] loading checkpoint: {checkpoint}")
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill,
                         map_location='cpu')
    cfg = payload['cfg']
    cfg_target = cfg._target_
    print(f"[rollout]   workspace target: {cfg_target}")
    print(f"[rollout]   task name:        {cfg.task.name}")

    # ------------------------------------------------------------------
    # Apply CLI overrides BEFORE we instantiate (so the env_runner sees
    # them at construction time — n_train/n_test/seeds/etc.).
    # ------------------------------------------------------------------
    applied = []
    if rollout_mode is not None:
        OmegaConf.update(cfg, 'task.env_runner.rollout_mode', rollout_mode, merge=True)
        applied.append(f"task.env_runner.rollout_mode={rollout_mode}")
    if n_envs is not None:
        OmegaConf.update(cfg, 'task.env_runner.n_envs', n_envs, merge=True)
        applied.append(f"task.env_runner.n_envs={n_envs}")
    for ov in overrides:
        if '=' not in ov:
            raise click.ClickException(
                f"Override must be `key=value`, got: {ov!r}")
        key, _, value = ov.partition('=')
        _apply_override(cfg, key.strip(), value.strip())
        applied.append(f"{key}={value}")
    if applied:
        print(f"[rollout] applied {len(applied)} override(s):")
        for line in applied:
            print(f"    {line}")

    # ------------------------------------------------------------------
    # Reconstruct the workspace with the (possibly overridden) cfg, then
    # load the saved state_dicts back into it. This yields the SAME
    # in-memory workspace the training run had at checkpoint time.
    # ------------------------------------------------------------------
    cls = hydra.utils.get_class(cfg_target)
    workspace: BaseWorkspace = cls(cfg, output_dir=str(out))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # ------------------------------------------------------------------
    # Pick policy (EMA vs online) and move to device.
    # ------------------------------------------------------------------
    if ema is None:
        # Default: follow checkpoint's training-time choice.
        use_ema = bool(cfg.training.use_ema)
    else:
        use_ema = bool(ema)

    if use_ema and getattr(workspace, 'ema_model', None) is None:
        print("[rollout] WARNING: --ema requested (or use_ema=True in cfg) but "
              "workspace has no ema_model; falling back to online model.")
        use_ema = False

    policy = workspace.ema_model if use_ema else workspace.model
    print(f"[rollout] policy: {'EMA' if use_ema else 'online'} weights "
          f"({type(policy).__name__})")

    dev = torch.device(device)
    policy.to(dev)
    policy.eval()

    # ------------------------------------------------------------------
    # Instantiate env runner — output_dir kw is the same plumbing the
    # training-time workspace uses, so videos / past-action PDFs land
    # under the rollout output_dir, not the original training run's dir.
    # ------------------------------------------------------------------
    print(f"[rollout] instantiating env_runner from cfg.task.env_runner ...")
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner, output_dir=str(out))
    print(f"[rollout]   {type(env_runner).__name__}: "
          f"n_inits={len(env_runner.env_init_fn_dills)}, "
          f"n_envs={len(env_runner.env_fns)}, "
          f"max_steps={env_runner.max_steps}")

    if dry_run:
        print("[rollout] --dry-run: skipping env_runner.run()")
        runner_log = {}
    else:
        # ------------------------------------------------------------------
        # Run rollouts. The image runners accept `epoch=` (used for
        # past-action PDF / per-epoch local-rollout folder naming); the
        # lowdim runners take only `policy` and ignore extra kwargs.
        # ------------------------------------------------------------------
        print(f"[rollout] running rollouts (epoch={epoch}) ...")
        rollout_t0 = time.time()
        try:
            runner_log = env_runner.run(policy, epoch=epoch)
        except TypeError:
            # Older / lowdim runners: signature is run(self, policy).
            runner_log = env_runner.run(policy)
        print(f"[rollout] env_runner.run finished in "
              f"{time.time() - rollout_t0:.1f} s")

    # ------------------------------------------------------------------
    # Persist results.
    # ------------------------------------------------------------------
    json_log = _serialize_runner_log_for_json(runner_log)
    log_path = out / "eval_log.json"
    with open(log_path, "w") as fh:
        json.dump(json_log, fh, indent=2, sort_keys=True)
    print(f"[rollout] wrote {log_path} ({len(json_log)} entries)")

    summary = _summarize_runner_log(runner_log)
    summary["_meta"] = {
        "checkpoint": str(checkpoint),
        "workspace_target": cfg_target,
        "task_name": cfg.task.name,
        "use_ema": use_ema,
        "device": str(device),
        "epoch": epoch,
        "overrides": applied,
        "wall_time_sec": round(time.time() - t0, 2),
    }
    summary_path = out / "eval_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(f"[rollout] wrote {summary_path}")

    # ------------------------------------------------------------------
    # Pretty-print the headlines.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Rollout summary")
    print("=" * 60)
    for k, v in sorted(summary.items()):
        if k == "_meta":
            continue
        if isinstance(v, float):
            print(f"  {k:35s} {v:.4f}")
        else:
            print(f"  {k:35s} {v}")
    print("=" * 60)
    print(f"  total wall-time: {time.time() - t0:.1f} s")


if __name__ == '__main__':
    main()
