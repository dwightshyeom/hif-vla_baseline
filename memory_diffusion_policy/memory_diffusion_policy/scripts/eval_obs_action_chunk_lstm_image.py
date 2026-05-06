#!/usr/bin/env python3
"""
Evaluation and visualisation script for Vision-based ObsActionChunkLSTM.

For each sampled episode:
  1. Runs the model through the full episode to get future/past chunk predictions.
  2. Denormalises predictions back to action space.
  3. For sampled timesteps, renders:
     - The input image at t
     - Overlaid GT future trajectory (pink) and predicted (purple)
     - Overlaid GT past trajectory (orange) and predicted (sienna)

Usage
-----
NVIDIA_VISIBLE_DEVICES=0 docker compose run --rm dev python eval_obs_action_chunk_lstm_image.py --checkpoint outputs/obs_action_chunk_lstm_image_2d_friction_vision_320_step_1_hidden_64_past_consistent_strong_scale_5/best_model.pt --output_dir outputs/obs_action_chunk_lstm_image_2d_friction_vision_320_step_1_hidden_64_past_consistent_strong_scale_5/eval_viz --n_episodes 10 --n_timesteps 20
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import yaml

import importlib

from memory_diffusion_policy.dataset.pusht_three_goals_lstm_image_dataset import (
    PushTThreeGoalsLSTMImageDataset,
)
from memory_diffusion_policy.model.obs_action_chunk_lstm_image import ObsActionChunkLSTMImage


def _resolve_dataset_class(dotted_path):
    if not dotted_path:
        return PushTThreeGoalsLSTMImageDataset
    module_path, cls_name = dotted_path.rsplit('.', 1)
    return getattr(importlib.import_module(module_path), cls_name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_C_TRAJ_GT    = "#ff69b4"     # pink
_C_TRAJ_PRED  = "#9b59b6"     # purple
_C_PAST_GT    = "#fb8500"     # orange
_C_PAST_PRED  = "#bc6c25"     # sienna
_C_AGENT      = "#4169e1"     # royal blue

CANVAS_SIZE = 96  # image resolution


def _load_torch_artifact(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _normalize_seq(normalizer, x: torch.Tensor, key: str) -> torch.Tensor:
    B, T, D = x.shape
    x_flat = x.reshape(B * T, D).unsqueeze(1)
    return normalizer[key].normalize(x_flat).squeeze(1).reshape(B, T, D).to(x.device)


def _denormalize_seq(normalizer, x: torch.Tensor, key: str) -> torch.Tensor:
    B, T, D = x.shape
    x_flat = x.reshape(B * T, D).unsqueeze(1)
    return normalizer[key].unnormalize(x_flat).squeeze(1).reshape(B, T, D)


# ---------------------------------------------------------------------------
# Build chunk targets (duplicated from training script for standalone use)
# ---------------------------------------------------------------------------

def build_future_chunk_targets(action_norm_np, lengths_np, chunk_H):
    B, T, D = action_norm_np.shape
    targets = np.zeros((B, T, chunk_H, D), dtype=np.float32)
    mask = np.zeros((B, T), dtype=np.float32)
    offsets = np.arange(chunk_H)
    for b in range(B):
        T_b = int(lengths_np[b])
        n = max(0, T_b - chunk_H + 1)
        if n > 0:
            rows = np.arange(n)[:, None]
            targets[b, :n] = action_norm_np[b, rows + offsets]
            mask[b, :n] = 1.0
    return targets, mask


def build_past_chunk_targets(action_norm_np, lengths_np, past_chunk_H):
    """past_target[b, t, k] = action_norm_np[b, t-1-k]  if t-1-k >= 0 else 0.
    Slot k=0 is a_{t-1} — same convention as the training script."""
    B, T, D = action_norm_np.shape
    targets = np.zeros((B, T, past_chunk_H, D), dtype=np.float32)
    mask = np.zeros((B, T, past_chunk_H), dtype=np.float32)
    offsets = np.arange(1, past_chunk_H + 1)
    for b in range(B):
        T_b = int(lengths_np[b])
        if T_b == 0:
            continue
        t_vals = np.arange(T_b)
        idx = t_vals[:, None] - offsets[None, :]
        valid = idx >= 0
        idx_clamped = np.clip(idx, 0, T_b - 1)
        gathered = action_norm_np[b, idx_clamped]
        gathered[~valid] = 0.0
        targets[b, :T_b] = gathered
        mask[b, :T_b] = valid.astype(np.float32)
    return targets, mask


# ---------------------------------------------------------------------------
# Quantitative metrics
# ---------------------------------------------------------------------------

def compute_trajectory_errors(model, normalizer, dataset, device, n_episodes=10):
    """Compute future trajectory MAE/RMSE in unnormalised action space."""
    all_mae, all_rmse = [], []

    for i in range(min(n_episodes, len(dataset))):
        s = dataset[i]
        image = s['image'].unsqueeze(0).to(device)           # (1, T, 3, H, W)
        agent_pos = s['agent_pos'].unsqueeze(0).to(device)   # (1, T, 2)
        action = s['action'].unsqueeze(0).to(device)         # (1, T, 2)
        T = image.shape[1]
        lengths = torch.tensor([T])

        agent_pos_norm = _normalize_seq(normalizer, agent_pos, 'agent_pos')
        action_norm = _normalize_seq(normalizer, action, 'action')

        with torch.no_grad():
            (_, future_pred, _, _, _, _, _, _, _) = model(
                image, agent_pos_norm, action_norm, lengths=lengths)

        if future_pred is None:
            continue

        # future_pred: (1, T, H, 2) normalised
        action_norm_np = action_norm.cpu().numpy()
        lengths_np = lengths.numpy()
        future_gt, future_mask = build_future_chunk_targets(
            action_norm_np, lengths_np, model.chunk_H)

        # Denormalise both
        fp = future_pred[0].cpu()   # (T, H, 2)
        gt = torch.from_numpy(future_gt[0])  # (T, H, 2)
        fm = future_mask[0]  # (T,)

        # Unnormalize
        Tv, H, D = fp.shape
        fp_flat = fp.reshape(Tv * H, D).unsqueeze(1)
        gt_flat = gt.reshape(Tv * H, D).unsqueeze(1)
        fp_un = normalizer['action'].unnormalize(fp_flat).squeeze(1).reshape(Tv, H, D).numpy()
        gt_un = normalizer['action'].unnormalize(gt_flat).squeeze(1).reshape(Tv, H, D).numpy()

        valid = fm > 0.5
        if valid.sum() == 0:
            continue

        err = np.abs(fp_un[valid] - gt_un[valid])
        all_mae.append(float(err.mean()))
        all_rmse.append(float(np.sqrt((err ** 2).mean())))

    return {
        'traj_mae_mean': float(np.mean(all_mae)) if all_mae else 0.0,
        'traj_mae_std': float(np.std(all_mae)) if all_mae else 0.0,
        'traj_rmse_mean': float(np.mean(all_rmse)) if all_rmse else 0.0,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise_episode(
    model, normalizer, image_np, agent_pos_np, action_np,
    device, n_timesteps=8, episode_label="", output_path=None,
):
    """
    Render a grid of panels. Each panel shows the input image with overlaid
    trajectory predictions (future in purple/pink, past in sienna/orange).
    """
    T = len(image_np)
    image_t = torch.from_numpy(image_np).unsqueeze(0).to(device)      # (1, T, 3, 96, 96)
    agent_t = torch.from_numpy(agent_pos_np).unsqueeze(0).to(device)  # (1, T, 2)
    action_t = torch.from_numpy(action_np).unsqueeze(0).to(device)    # (1, T, 2)
    lengths = torch.tensor([T])

    agent_norm = _normalize_seq(normalizer, agent_t, 'agent_pos')
    action_norm = _normalize_seq(normalizer, action_t, 'action')

    with torch.no_grad():
        (_, future_pred, past_pred, _, _, _, _, _, _) = model(
            image_t, agent_norm, action_norm, lengths=lengths)

    # Build GT targets
    action_norm_np = action_norm.cpu().numpy()
    lengths_np = lengths.numpy()

    future_gt_norm = past_gt_norm = None
    future_mask = past_mask = None
    if future_pred is not None:
        future_gt_norm, future_mask = build_future_chunk_targets(
            action_norm_np, lengths_np, model.chunk_H)
    if past_pred is not None:
        past_gt_norm, past_mask = build_past_chunk_targets(
            action_norm_np, lengths_np, model.past_chunk_H)

    # Denormalise
    def _denorm_chunk(x_norm, key='action'):
        """(1, T, H, D) -> numpy"""
        x = x_norm[0].cpu()
        Tv, H, D = x.shape
        flat = x.reshape(Tv * H, D).unsqueeze(1)
        return normalizer[key].unnormalize(flat).squeeze(1).reshape(Tv, H, D).numpy()

    fp_un = _denorm_chunk(future_pred) if future_pred is not None else None
    fg_un = _denorm_chunk(torch.from_numpy(future_gt_norm)) if future_gt_norm is not None else None
    pp_un = _denorm_chunk(past_pred) if past_pred is not None else None
    pg_un = _denorm_chunk(torch.from_numpy(past_gt_norm)) if past_gt_norm is not None else None

    # Pick timesteps
    sampled_t = np.round(np.linspace(
        min(5, T - 1), T - 2, n_timesteps)).astype(int)
    sampled_t = np.unique(np.clip(sampled_t, 0, T - 1))

    ncols = min(4, len(sampled_t))
    nrows = (len(sampled_t) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    # Agent position normalizer stats (for trajectory coords)
    # The trajectories are in action space so we use action unnormalization
    # to convert them to pixel-like coords.

    for panel_idx, t in enumerate(sampled_t):
        row, col = divmod(panel_idx, ncols)
        ax = axes[row, col]

        # Show input image (CHW -> HWC)
        img = image_np[t].transpose(1, 2, 0)  # (96, 96, 3)
        ax.imshow(img, extent=[0, CANVAS_SIZE, CANVAS_SIZE, 0])

        # Scale actions to image coords: the actions are in raw action space
        # (roughly 0-512 for PushT), need to map to 0-96 image coords.
        scale = CANVAS_SIZE / 512.0

        # Draw agent position
        ap = agent_pos_np[t]
        ax.plot(ap[0] * scale, ap[1] * scale, 'o', color=_C_AGENT,
                markersize=4, zorder=10)

        # GT future trajectory
        if fg_un is not None and future_mask is not None and future_mask[0, t] > 0.5:
            gt_pts = fg_un[t] * scale  # (H, 2)
            ax.plot(gt_pts[:, 0], gt_pts[:, 1], '-o', color=_C_TRAJ_GT,
                    markersize=2, linewidth=1.0, alpha=0.7, zorder=8)
            ax.plot(gt_pts[0, 0], gt_pts[0, 1], 's', color=_C_TRAJ_GT,
                    markersize=5, zorder=11)          # start = square
            ax.plot(gt_pts[-1, 0], gt_pts[-1, 1], '*', color=_C_TRAJ_GT,
                    markersize=7, zorder=11)          # end = star

        # Predicted future trajectory
        if fp_un is not None and future_mask is not None and future_mask[0, t] > 0.5:
            pred_pts = fp_un[t] * scale  # (H, 2)
            ax.plot(pred_pts[:, 0], pred_pts[:, 1], '-o', color=_C_TRAJ_PRED,
                    markersize=2, linewidth=1.0, alpha=0.7, zorder=9)
            ax.plot(pred_pts[0, 0], pred_pts[0, 1], 's', color=_C_TRAJ_PRED,
                    markersize=5, zorder=12)
            ax.plot(pred_pts[-1, 0], pred_pts[-1, 1], '*', color=_C_TRAJ_PRED,
                    markersize=7, zorder=12)

        # GT past trajectory
        if pg_un is not None and past_mask is not None:
            valid_past = past_mask[0, t] > 0.5
            if valid_past.any():
                gt_past = pg_un[t][valid_past] * scale
                ax.plot(gt_past[:, 0], gt_past[:, 1], '-o', color=_C_PAST_GT,
                        markersize=2, linewidth=1.0, alpha=0.5, zorder=6)
                ax.plot(gt_past[0, 0], gt_past[0, 1], 's', color=_C_PAST_GT,
                        markersize=5, zorder=11)
                ax.plot(gt_past[-1, 0], gt_past[-1, 1], '*', color=_C_PAST_GT,
                        markersize=7, zorder=11)

        if pp_un is not None and past_mask is not None:
            valid_past = past_mask[0, t] > 0.5
            if valid_past.any():
                pred_past = pp_un[t][valid_past] * scale
                ax.plot(pred_past[:, 0], pred_past[:, 1], '-o', color=_C_PAST_PRED,
                        markersize=2, linewidth=1.0, alpha=0.5, zorder=7)
                ax.plot(pred_past[0, 0], pred_past[0, 1], 's', color=_C_PAST_PRED,
                        markersize=5, zorder=12)
                ax.plot(pred_past[-1, 0], pred_past[-1, 1], '*', color=_C_PAST_PRED,
                        markersize=7, zorder=12)

        ax.set_xlim(0, CANVAS_SIZE)
        ax.set_ylim(CANVAS_SIZE, 0)
        ax.set_title(f"t={t}", fontsize=9)
        ax.set_aspect('equal')

    # Hide unused axes
    for panel_idx in range(len(sampled_t), nrows * ncols):
        row, col = divmod(panel_idx, ncols)
        axes[row, col].set_visible(False)

    # ---- Figure-level legend at bottom ----------------------------------
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=_C_TRAJ_GT,   linestyle='-', marker='o', markersize=4, label='GT future'),
        Line2D([0], [0], color=_C_TRAJ_PRED,  linestyle='-', marker='o', markersize=4, label='Pred future'),
        Line2D([0], [0], color=_C_PAST_GT,    linestyle='-', marker='o', markersize=4, label='GT past'),
        Line2D([0], [0], color=_C_PAST_PRED,  linestyle='-', marker='o', markersize=4, label='Pred past'),
        Line2D([0], [0], color='gray', linestyle='none', marker='s', markersize=5, label='Start'),
        Line2D([0], [0], color='gray', linestyle='none', marker='*', markersize=7, label='End'),
        Line2D([0], [0], color=_C_AGENT, linestyle='none', marker='o', markersize=5, label='Agent pos'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=len(legend_handles), fontsize=7, frameon=True,
               bbox_to_anchor=(0.5, -0.02))
    axes[row, col].set_visible(False)

    fig.suptitle(episode_label, fontsize=11)
    fig.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and visualise Vision ObsActionChunkLSTM")
    parser.add_argument("--checkpoint", type=str,
                        default="./outputs/obs_action_chunk_lstm_image_three_goals/best_model.pt")
    parser.add_argument("--output_dir", type=str,
                        default="./outputs/obs_action_chunk_lstm_image_three_goals/eval_viz")
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--n_timesteps", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # ---- Load checkpoint ------------------------------------------------
    ckpt_path = Path(args.checkpoint)
    assert ckpt_path.exists(), f"Not found: {ckpt_path}"
    ckpt = _load_torch_artifact(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    shape_meta = ckpt.get("shape_meta", {
        'obs': {
            'image': {'shape': [3, 96, 96], 'type': 'rgb'},
            'agent_pos': {'shape': [2], 'type': 'low_dim'},
        },
        'action': {'shape': [2]},
    })

    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*65}")
    print("Vision ObsActionChunkLSTM — evaluation & visualisation")
    print(f"{'='*65}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Device     : {device}")

    # ---- Output directory -----------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train").mkdir(exist_ok=True)
    (output_dir / "val").mkdir(exist_ok=True)

    # ---- Normalizer -----------------------------------------------------
    norm_path = ckpt_path.parent / "normalizer.pt"
    if norm_path.exists():
        normalizer = _load_torch_artifact(norm_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Normalizer not found at {norm_path}")

    # ---- Dataset --------------------------------------------------------
    zarr_path = ckpt_args.get('zarr_path', 'data/pusht_three_goals_demo_vision.zarr')
    val_ratio = ckpt_args.get('val_ratio', 0.15)
    seed = ckpt_args.get('seed', 42)
    subsample = ckpt_args.get('action_step_subsample', 1)

    DatasetCls = _resolve_dataset_class(ckpt_args.get('dataset_class'))
    train_ds = DatasetCls(
        zarr_path=zarr_path, val_ratio=val_ratio, seed=seed,
        action_step_subsample=subsample)
    val_ds = train_ds.get_validation_dataset()
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}  "
          f"|  subsample: {subsample}")

    # ---- Model ----------------------------------------------------------
    crop_shape = tuple(ckpt_args.get('crop_shape', [76, 76]))
    past_chunk_H = ckpt_args.get('past_chunk_H', ckpt_args.get('chunk_H', 16))

    model = ObsActionChunkLSTMImage(
        shape_meta=shape_meta,
        action_dim=2,
        hidden_size=ckpt_args.get('hidden_size', 256),
        num_layers=ckpt_args.get('num_layers', 2),
        dropout=ckpt_args.get('dropout', 0.1),
        chunk_H=ckpt_args.get('chunk_H', 16),
        use_obs_head=False,
        use_future_head=bool(ckpt_args.get('use_future_head', True)),
        num_future_modes=int(ckpt_args.get('num_future_modes', 1)),
        future_abstraction=ckpt_args.get('future_abstraction', 'raw'),
        future_n_bases=int(ckpt_args.get('future_n_bases', 32)),
        use_past_head=bool(ckpt_args.get('use_past_head', True)),
        past_chunk_H=past_chunk_H,
        past_abstraction=ckpt_args.get('past_abstraction', 'raw'),
        past_n_bases=int(ckpt_args.get('past_n_bases', 32)),
        use_vq=bool(ckpt_args.get('use_vq', False)),
        vq_n_codes=int(ckpt_args.get('vq_n_codes', 512)),
        vq_commitment_weight=float(ckpt_args.get('vq_commitment_weight', 0.25)),
        crop_shape=crop_shape,
        obs_encoder_group_norm=bool(ckpt_args.get('obs_encoder_group_norm', True)),
        eval_fixed_crop=True,
        freeze_encoder=False,
        num_kp=int(ckpt_args.get('num_kp', 32)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', float('nan')):.5f})")
    print(f"Visual encoder dim: {model.obs_feature_dim}")

    # ---- Quantitative evaluation ----------------------------------------
    if model.use_future_head:
        print(f"\n{'='*65}")
        print("Quantitative trajectory prediction errors")
        print(f"{'='*65}")
        for split_name, ds in [("train", train_ds), ("val", val_ds)]:
            n_ep = min(args.n_episodes * 2, len(ds))
            err = compute_trajectory_errors(model, normalizer, ds, device, n_ep)
            print(
                f"  {split_name:5s} ({n_ep} eps) | "
                f"traj MAE={err['traj_mae_mean']:.2f}\u00b1{err['traj_mae_std']:.2f}  "
                f"traj RMSE={err['traj_rmse_mean']:.2f}")

    # ---- Visualise episodes ---------------------------------------------
    for split_name, dataset in [("train", train_ds), ("val", val_ds)]:
        n_ep = min(args.n_episodes, len(dataset))
        ep_indices = np.round(np.linspace(0, len(dataset) - 1, n_ep)).astype(int)
        print(f"\n--- {split_name} split ({n_ep} episodes) ---")

        for rank, ep_idx in enumerate(ep_indices):
            s = dataset[int(ep_idx)]
            image_np = s['image'].numpy()          # (T, 3, 96, 96)
            agent_pos_np = s['agent_pos'].numpy()  # (T, 2)
            action_np = s['action'].numpy()        # (T, 2)
            T = len(image_np)
            label = f"{split_name} ep {ep_idx} (T={T})"
            out_path = output_dir / split_name / f"ep_{ep_idx:04d}.pdf"
            print(f"  [{rank+1}/{n_ep}] {label} -> {out_path.name}")

            visualise_episode(
                model=model,
                normalizer=normalizer,
                image_np=image_np,
                agent_pos_np=agent_pos_np,
                action_np=action_np,
                device=device,
                n_timesteps=args.n_timesteps,
                episode_label=label,
                output_path=out_path,
            )

    print(f"\nAll saved to: {output_dir}")


if __name__ == "__main__":
    main()
