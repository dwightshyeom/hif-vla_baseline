#!/usr/bin/env python3
"""
Evaluation and visualisation script for ObsActionChunkLSTM.

The model directly predicts raw (LP-normalised) action chunks — no tokenizer
or DCT transform is involved.

For each sampled episode this script:
  1. Runs the model through the full episode to get:
       - Predicted observations          (obs head, if use_obs_head)
       - Predicted future action chunk   norm(a_{t:t+H})          (future head)
       - Predicted past action chunk     norm(reversed past pastH) (past head)
  2. Denormalises the chunk predictions back to pixel space.
  3. Renders a panel per sampled timestep t:

     Spatial panel (top)
     ────────────────────
       • T-block + agent at t-1          faded blue    (input context)
       • Red arrow at t-1                (input action)
       • T-block + agent at t  (GT)      solid slate-gray
       • T-block at t (obs pred)         amber overlay
       • GT future trajectory            pink path
       • Predicted future trajectory     purple path
       • GT past trajectory              orange path (if use_past_head)
       • Predicted past trajectory       sienna path (if use_past_head)

     Optional chunk comparison panels (one row per action dim, show_chunk=True)
     ──────────────────────────────────────────────────────────────────────────
       • Line plot: predicted vs GT per-step action values for future chunk
       • Line plot: predicted vs GT per-step action values for past chunk

Usage
-----
python eval_obs_action_chunk_lstm.py --checkpoint outputs/obs_action_chunk_lstm_varying_goal_multi_future/best_model.pt --output_dir  outputs/obs_action_chunk_lstm_varying_goal_multi_future/eval_viz_multi_future --n_episodes 10 --n_timesteps 40 --show_chunk
"""

import argparse
import json
import os
import sys
from math import cos, sin, atan2
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow, Circle, Polygon
import numpy as np
import torch
import yaml

from memory_diffusion_policy.dataset.pusht_three_goals_lstm_dataset import (
    PushTThreeGoalsLSTMDatasetWithIndicator,
)
from memory_diffusion_policy.model.obs_action_chunk_lstm import ObsActionChunkLSTM, dct_forward, dct_inverse

# ---------------------------------------------------------------------------
# PushT geometry constants
# ---------------------------------------------------------------------------
CANVAS_SIZE = 512
_T_SCALE    = 30
_T_LENGTH   = 4

_T_VERTS_CROSSBAR = np.array([
    [-_T_LENGTH * _T_SCALE / 2,  0],
    [ _T_LENGTH * _T_SCALE / 2,  0],
    [ _T_LENGTH * _T_SCALE / 2,  _T_SCALE],
    [-_T_LENGTH * _T_SCALE / 2,  _T_SCALE],
], dtype=float)

_T_VERTS_STEM = np.array([
    [-_T_SCALE / 2,  _T_SCALE],
    [-_T_SCALE / 2,  _T_LENGTH * _T_SCALE],
    [ _T_SCALE / 2,  _T_LENGTH * _T_SCALE],
    [ _T_SCALE / 2,  _T_SCALE],
], dtype=float)

_AGENT_RADIUS = 15.0

_GOAL_POSES = [
    np.array([160, 360, np.pi / 4]),
    np.array([256, 152, 0.0]),
    np.array([352, 360, -np.pi / 4]),
]

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_C_GT_BLOCK    = "#778da9"
_C_GT_AGENT    = "#4169e1"
_C_GT_EDGE     = "#3a4a5c"
_C_PREV_BLOCK  = "#778da9"
_C_PREV_AGENT  = "#4169e1"
_C_PREV_EDGE   = "#445566"
_C_PRED_BLOCK  = "#ff9f1c"
_C_PRED_AGENT  = "#ff6b00"
_C_PRED_EDGE   = "#c94d00"
_C_ACTION_IN   = "#e63946"
_C_TRAJ_GT     = "#ff69b4"     # pink  – GT future trajectory
_C_TRAJ_PRED   = "#9b59b6"     # purple – predicted future trajectory
_C_PAST_GT     = "#fb8500"     # orange – GT past trajectory
_C_PAST_PRED   = "#bc6c25"     # sienna – predicted past trajectory
_C_GOAL        = "#90ee90"
_C_GOAL_EDGE   = "#228b22"

# Distinct colours for multi-modal future trajectory modes
_MODE_COLORS = [
    '#9b59b6',   # purple  (mode 1, same as single-mode pred)
    '#e74c3c',   # red
    '#2ecc71',   # green
    '#f39c12',   # amber
    '#3498db',   # blue
    '#1abc9c',   # teal
    '#e67e22',   # orange
    '#8e44ad',   # dark purple
]


# ---------------------------------------------------------------------------
# Torch-load helper
# ---------------------------------------------------------------------------

def _load_torch_artifact(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _pose_from_keypoints(local_kps, global_kps):
    mu_l = local_kps.mean(axis=0)
    mu_g = global_kps.mean(axis=0)
    H_mat = (local_kps - mu_l).T @ (global_kps - mu_g)
    U, _, Vt = np.linalg.svd(H_mat)
    d  = np.linalg.det(Vt.T @ U.T)
    R  = Vt.T @ np.diag([1.0, d]) @ U.T
    angle = float(atan2(R[1, 0], R[0, 0]))
    trans = mu_g - R @ mu_l
    return float(trans[0]), float(trans[1]), angle


def _tblock_world_verts(x, y, angle):
    c, s = cos(angle), sin(angle)
    R = np.array([[c, -s], [s, c]])
    return (_T_VERTS_CROSSBAR @ R.T + [x, y],
            _T_VERTS_STEM     @ R.T + [x, y])


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_tblock(ax, x, y, angle, fc, ec="none", alpha=1.0, lw=1.2, zorder=3, label=None):
    cb, st = _tblock_world_verts(x, y, angle)
    for i, v in enumerate([cb, st]):
        ax.add_patch(Polygon(v, closed=True, facecolor=fc, edgecolor=ec,
                             linewidth=lw, alpha=alpha, zorder=zorder,
                             label=(label if i == 0 else None)))

def _draw_agent(ax, pos, fc, ec="none", alpha=1.0, lw=1.0, zorder=4, label=None):
    ax.add_patch(Circle(pos, _AGENT_RADIUS, facecolor=fc, edgecolor=ec,
                        linewidth=lw, alpha=alpha, zorder=zorder, label=label))

def _draw_arrow(ax, src, dst, color, alpha=0.9, zorder=7, label=None):
    dx, dy = float(dst[0] - src[0]), float(dst[1] - src[1])
    L = np.hypot(dx, dy)
    if L < 1.0:
        return
    hw = max(8.0, L * 0.15)
    hl = max(12.0, L * 0.25)
    ax.add_patch(FancyArrow(float(src[0]), float(src[1]), dx, dy,
                            width=3, head_width=hw, head_length=hl,
                            color=color, alpha=alpha,
                            length_includes_head=True,
                            zorder=zorder, label=label))

def _draw_trajectory(ax, points, base_color, alpha_range=(0.35, 0.95),
                     zorder=8, label=None):
    H = len(points)
    if H < 1:
        return
    alphas    = np.linspace(alpha_range[0], alpha_range[1], H)
    base_rgba = np.array(matplotlib.colors.to_rgba(base_color))
    for h in range(H):
        colour = (*base_rgba[:3], float(alphas[h]))
        ax.plot(points[h, 0], points[h, 1], 'o', color=colour,
                markersize=5, zorder=zorder, label=(label if h == 0 else None))
        if h > 0:
            ax.plot([points[h-1, 0], points[h, 0]],
                    [points[h-1, 1], points[h, 1]],
                    '-', color=colour, linewidth=1.5, zorder=zorder)
    start_col = (*base_rgba[:3], min(1.0, float(alphas[0])  + 0.3))
    end_col   = (*base_rgba[:3], min(1.0, float(alphas[-1]) + 0.2))
    ax.plot(points[0, 0],  points[0, 1],  '*', color=start_col, markersize=14,
            zorder=zorder+2, markeredgecolor='white', markeredgewidth=0.8)
    ax.plot(points[-1, 0], points[-1, 1], 'D', color=end_col,   markersize=7,
            zorder=zorder+2, markeredgecolor='white', markeredgewidth=0.8)

def _draw_goal_regions(ax, goal_poses=None):
    poses = goal_poses if goal_poses is not None else _GOAL_POSES
    for gp in poses:
        x, y, a = float(gp[0]), float(gp[1]), float(gp[2])
        for v in _tblock_world_verts(x, y, a):
            ax.add_patch(Polygon(v, closed=True, facecolor=_C_GOAL,
                                 edgecolor=_C_GOAL_EDGE, linewidth=0.8,
                                 alpha=0.35, zorder=1))

def _setup_spatial_ax(ax):
    ax.set_xlim(0, CANVAS_SIZE)
    ax.set_ylim(CANVAS_SIZE, 0)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_visible(False)


# ---------------------------------------------------------------------------
# Normaliser helpers
# ---------------------------------------------------------------------------

def _normalize_seq(normalizer, x, key):
    B, T, D = x.shape
    xf = x.reshape(B * T, D).unsqueeze(1)
    return normalizer[key].normalize(xf).squeeze(1).reshape(B, T, D)

def _denormalize_seq(normalizer, x, key):
    B, T, D = x.shape
    xf = x.reshape(B * T, D).unsqueeze(1)
    return normalizer[key].unnormalize(xf).squeeze(1).reshape(B, T, D)

def _denormalize_chunk_np(normalizer, chunk_norm_np, key="action"):
    """
    Denormalise a chunk array of shape (..., action_dim) using the LP normalizer.
    Works for any leading shape (e.g. (chunk_H, D) or (T, chunk_H, D)).
    """
    shape = chunk_norm_np.shape
    flat  = chunk_norm_np.reshape(-1, shape[-1])
    t_in  = torch.from_numpy(flat).float().unsqueeze(1)
    t_out = normalizer[key].unnormalize(t_in).squeeze(1)
    return t_out.numpy().reshape(shape)


def _normalize_chunk_np(normalizer, chunk_px_np, key="action"):
    """Normalise a pixel-space chunk array of shape (..., action_dim)."""
    shape = chunk_px_np.shape
    flat  = chunk_px_np.reshape(-1, shape[-1])
    t_in  = torch.from_numpy(flat.astype(np.float32)).unsqueeze(1)
    t_out = normalizer[key].normalize(t_in).squeeze(1)
    return t_out.numpy().reshape(shape)


# ---------------------------------------------------------------------------
# Chunk comparison panels
# ---------------------------------------------------------------------------

def _draw_chunk_panel(axes_row, chunk_gt, chunk_pred, action_dim,
                      mae_pixels=None,
                      gt_color=None, pred_color=None,
                      gt_label="GT", pred_label="Pred",
                      title_prefix="Chunk"):
    """
    Draw per-dimension line plots comparing predicted vs GT action chunks.

    Args:
        axes_row:   list of `action_dim` Axes.
        chunk_gt:   (H, action_dim)  ground-truth chunk in pixel space.
        chunk_pred: (H, action_dim)  predicted chunk in pixel space.
        mae_pixels: (action_dim,) per-dim MAE in pixel units or None.
    """
    gt_color   = gt_color   or _C_TRAJ_GT
    pred_color = pred_color or _C_TRAJ_PRED
    dim_labels = ['x', 'y', 'z']
    H = len(chunk_gt)
    xs = np.arange(H)

    for d, ax in enumerate(axes_row):
        ax.plot(xs, chunk_gt[:, d],   color=gt_color,   linewidth=1.8,
                label=gt_label, marker='o', markersize=3)
        ax.plot(xs, chunk_pred[:, d], color=pred_color, linewidth=1.8,
                linestyle='--', label=pred_label, marker='o', markersize=3)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
        dl  = dim_labels[d] if d < len(dim_labels) else str(d)
        ttl = f"{title_prefix} dim {dl}"
        if mae_pixels is not None:
            ttl += f"  (MAE={mae_pixels[d]:.1f}px)"
        ax.set_title(ttl, fontsize=10)
        ax.tick_params(labelsize=9)
        ax.set_xlabel("step", fontsize=9)
        ax.set_ylabel("action (px)", fontsize=9)
        if d == 0:
            ax.legend(fontsize=9, loc="upper right")


# ---------------------------------------------------------------------------
# DCT coefficient comparison PDF
# ---------------------------------------------------------------------------

def _save_dct_comparison_pdf(
    dct_coeffs_gt:   np.ndarray,   # (T, K, D)
    dct_coeffs_pred: np.ndarray,   # (T, K, D)
    past_valid_mask: np.ndarray,   # (T, H_past) bool
    sampled_t:       np.ndarray,
    action_dim:      int,
    episode_label:   str,
    output_path:     Path,
):
    """Save DCT coefficient comparison (GT vs pred) for each sampled timestep."""
    K        = dct_coeffs_gt.shape[1]
    n_panels = len(sampled_t)
    COLS_MAX = 4
    ncols    = min(n_panels, COLS_MAX)
    nrows    = (n_panels + ncols - 1) // ncols
    fig_rows = nrows * action_dim

    fig, axes = plt.subplots(
        fig_rows, ncols,
        figsize=(4.5 * ncols, 3.2 * fig_rows),
        squeeze=False,
    )

    xs     = np.arange(K)
    bar_w  = 0.38
    dim_labels = ['x', 'y', 'z']

    # ---- Compute global y-axis limits per action dim (consistent scale) ----
    y_min = np.full(action_dim, np.inf)
    y_max = np.full(action_dim, -np.inf)
    for t in sampled_t:
        for d in range(action_dim):
            vals = np.concatenate([dct_coeffs_gt[t, :, d], dct_coeffs_pred[t, :, d]])
            y_min[d] = min(y_min[d], float(vals.min()))
            y_max[d] = max(y_max[d], float(vals.max()))
    for d in range(action_dim):
        span = max(y_max[d] - y_min[d], 1e-6)
        y_min[d] -= 0.10 * span
        y_max[d] += 0.10 * span

    for panel_idx, t in enumerate(sampled_t):
        col     = panel_idx % ncols
        n_valid = int(past_valid_mask[t].sum())

        for d in range(action_dim):
            row = (panel_idx // ncols) * action_dim + d
            ax  = axes[row, col]

            gt_c   = dct_coeffs_gt[t,   :, d]   # (K,)
            pred_c = dct_coeffs_pred[t, :, d]   # (K,)
            mae    = float(np.abs(gt_c - pred_c).mean())

            ax.bar(xs - bar_w / 2, gt_c,   width=bar_w, color=_C_PAST_GT,
                   alpha=0.85, label='GT')
            ax.bar(xs + bar_w / 2, pred_c, width=bar_w, color=_C_PAST_PRED,
                   alpha=0.85, label='Pred')
            ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
            ax.set_ylim(y_min[d], y_max[d])

            dl  = dim_labels[d] if d < len(dim_labels) else str(d)
            ax.set_title(
                f"t={t}  dim {dl}  |  {n_valid} valid steps  MAE={mae:.3f}",
                fontsize=9)
            ax.set_xlabel("basis k", fontsize=8)
            ax.set_ylabel("coeff (norm)", fontsize=8)
            ax.tick_params(labelsize=8)
            if d == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper right")

    # Hide unused axes
    for panel_idx in range(n_panels, nrows * ncols):
        col = panel_idx % ncols
        for d in range(action_dim):
            row = (panel_idx // ncols) * action_dim + d
            axes[row, col].set_visible(False)

    dct_path = output_path.parent / (output_path.stem + '_dct' + output_path.suffix)
    fig.suptitle(f"DCT coefficients – {episode_label}", fontsize=11, y=1.01)
    fig.tight_layout()
    plt.savefig(dct_path, bbox_inches='tight', dpi=110, format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core visualisation
# ---------------------------------------------------------------------------

def visualise_episode(
    model:        ObsActionChunkLSTM,
    normalizer:   dict,
    local_kps:    np.ndarray,
    obs_np:       np.ndarray,
    action_np:    np.ndarray,
    device:       str,
    keypoint_only: bool,
    keypoint_dim:  int,
    n_timesteps:   int,
    episode_label: str,
    output_path:   Path,
    show_chunk:    bool = True,
    goal_poses:    list = None,
):
    """Run forward pass, sample timesteps, render spatial + chunk panels."""
    T = len(obs_np)
    assert T >= 2

    H      = model.chunk_H
    H_past = model.past_chunk_H
    action_dim = action_np.shape[-1]
    has_future = model.use_future_head
    has_obs    = model.use_obs_head

    # ---- Forward pass ---------------------------------------------------
    obs_t    = torch.from_numpy(obs_np).float().unsqueeze(0)
    action_t = torch.from_numpy(action_np).float().unsqueeze(0)
    lengths  = torch.tensor([T])

    obs_norm    = _normalize_seq(normalizer, obs_t,    "obs")
    action_norm = _normalize_seq(normalizer, action_t, "action")
    obs_model   = obs_norm[:, :, :keypoint_dim] if keypoint_only else obs_norm

    model.eval()
    with torch.no_grad():
        obs_pred_norm, future_chunk_pred, past_chunk_pred, _, future_mode_logits = model.predict(
            obs_model.to(device), action_norm.to(device),
            lengths=lengths, hidden_state=None,
        )
        if obs_pred_norm is not None:
            obs_pred_norm = obs_pred_norm.cpu()
        if future_chunk_pred is not None:
            future_chunk_pred = future_chunk_pred.cpu()
        if past_chunk_pred is not None:
            past_chunk_pred = past_chunk_pred.cpu()
        if future_mode_logits is not None:
            future_mode_logits = future_mode_logits.cpu()

    # ---- Denormalise obs predictions ------------------------------------
    pred_obs_np = None
    if has_obs and obs_pred_norm is not None:
        pred_obs_full = obs_norm.clone()
        pred_obs_full[:, :, :obs_pred_norm.shape[-1]] = obs_pred_norm
        pred_obs_np   = _denormalize_seq(normalizer, pred_obs_full, "obs").squeeze(0).numpy()

    # ---- Denormalise future chunk predictions → pixel space -------------
    # future_chunk_pred: (1, T, [M,] H, D)
    fut_pred_px  = None
    traj_gt_all  = None
    mode_probs   = None      # (T, M) softmax probabilities, or None if single-mode
    n_modes      = getattr(model, 'num_future_modes', 1)
    if has_future and future_chunk_pred is not None:
        if future_chunk_pred.dim() == 5:
            # Multi-modal: (1, T, M, H, D)
            M = future_chunk_pred.shape[2]
            fut_pred_np = future_chunk_pred.squeeze(0).numpy()   # (T, M, H, D)
            fut_pred_px = np.zeros_like(fut_pred_np)
            for t in range(T):
                for m in range(M):
                    fut_pred_px[t, m] = _denormalize_chunk_np(normalizer, fut_pred_np[t, m])
            if future_mode_logits is not None:
                mode_probs = torch.softmax(
                    future_mode_logits.squeeze(0), dim=-1).numpy()  # (T, M)
        else:
            # Single mode: (1, T, H, D)
            fut_pred_np = future_chunk_pred.squeeze(0).numpy()   # (T, H, D)
            fut_pred_px = np.zeros_like(fut_pred_np)
            for t in range(T):
                fut_pred_px[t] = _denormalize_chunk_np(normalizer, fut_pred_np[t])  # (H, D)

        # GT future trajectories in pixel space
        traj_gt_all = np.zeros((T, H, action_dim), dtype=np.float32)
        for t in range(T - H + 1):
            traj_gt_all[t] = action_np[t:t + H]

    # ---- Denormalise past chunk predictions → pixel space ---------------
    traj_past_gt_all    = None
    traj_past_pred_all  = None
    past_valid_mask_all = None   # (T, H_past) bool — True for real data
    dct_coeffs_gt_all   = None   # (T, K, D)  – set in DCT mode
    dct_coeffs_pred_all = None   # (T, K, D)  – set in DCT mode
    if model.use_past_head and past_chunk_pred is not None:
        past_pred_np = past_chunk_pred.squeeze(0).numpy()   # (T, H_past, D) normalised
        past_pred_px = np.zeros_like(past_pred_np)
        for t in range(T):
            past_pred_px[t] = _denormalize_chunk_np(normalizer, past_pred_np[t])

        traj_past_gt_all    = np.zeros((T, H_past, action_dim), dtype=np.float32)
        traj_past_pred_all  = np.zeros((T, H_past, action_dim), dtype=np.float32)
        past_valid_mask_all = np.zeros((T, H_past), dtype=bool)
        for t in range(T):
            indices = np.arange(t - 1, t - H_past - 1, -1)  # may contain negatives
            valid   = indices >= 0
            past_valid_mask_all[t] = valid
            if valid.any():
                traj_past_gt_all[t, valid]   = action_np[indices[valid]]
                traj_past_pred_all[t, valid] = past_pred_px[t, valid]

        # ---- Basis-abstracted mode (DCT / segment_pool / DFT / DWT): replace
        #      raw GT with basis-filtered GT; compute coefficients for comparison --
        if getattr(model, 'past_abstraction', 'raw') in ('dct', 'segment_pool', 'dft', 'dwt') and \
                hasattr(model, '_past_dct_basis'):
            basis = model._past_dct_basis.cpu()   # (H_past, K)

            # Build full normalized GT matrix (T, H_past, D), zero-padded where invalid
            gt_norm_full = np.zeros((T, H_past, action_dim), dtype=np.float32)
            for t in range(T):
                valid = past_valid_mask_all[t]
                if valid.any():
                    idx_t = np.arange(t - 1, t - H_past - 1, -1)
                    gt_norm_full[t, valid] = _normalize_chunk_np(
                        normalizer, action_np[idx_t[valid]])

            gt_t   = torch.from_numpy(gt_norm_full)   # (T, H_past, D)
            pred_t = torch.from_numpy(past_pred_np)   # (T, H_past, D)

            coeffs_gt   = dct_forward(gt_t,   basis)  # (T, K, D)
            coeffs_pred = dct_forward(pred_t, basis)  # (T, K, D)

            dct_coeffs_gt_all   = coeffs_gt.numpy()
            dct_coeffs_pred_all = coeffs_pred.numpy()

            # Replace raw GT spatial trajectory with DCT-filtered version
            gt_filtered_norm = dct_inverse(coeffs_gt, basis).numpy()  # (T, H_past, D)
            for t in range(T):
                traj_past_gt_all[t] = _denormalize_chunk_np(
                    normalizer, gt_filtered_norm[t])

    # ---- Sample timesteps -----------------------------------------------
    if has_future:
        candidate_t = np.arange(1, T - H + 1) if T > H else np.arange(1, T)
    else:
        candidate_t = np.arange(1, T)
    if len(candidate_t) > n_timesteps:
        idx       = np.round(np.linspace(0, len(candidate_t) - 1, n_timesteps)).astype(int)
        sampled_t = candidate_t[idx]
    else:
        sampled_t = candidate_t

    n_panels = len(sampled_t)
    COLS_MAX = 4
    ncols    = min(n_panels, COLS_MAX)
    nrows    = (n_panels + ncols - 1) // ncols

    show_past   = show_chunk and (traj_past_gt_all is not None)
    show_future = show_chunk and has_future
    n_future_rows = action_dim if show_future else 0
    n_past_rows = action_dim if show_past else 0
    n_spec_rows = 1 + n_future_rows + n_past_rows
    spec_h_ratios = [7.0] + ([3.2] * n_future_rows) + ([3.2] * n_past_rows)
    fig_h         = (10.0 + 3.2 * n_future_rows + 3.2 * n_past_rows) * nrows
    top_margin_in = 0.90
    bot_margin_in = 3.00
    top_frac = 1.0 - top_margin_in / fig_h
    bot_frac = bot_margin_in / fig_h

    fig = plt.figure(figsize=(10.5 * ncols, fig_h))
    outer = gridspec.GridSpec(nrows, ncols, figure=fig,
                              hspace=0.50, wspace=0.25,
                              left=0.04, right=0.97,
                              top=top_frac, bottom=bot_frac)

    for panel_idx, t in enumerate(sampled_t):
        row, col = divmod(panel_idx, ncols)
        inner = gridspec.GridSpecFromSubplotSpec(
            n_spec_rows, 1, subplot_spec=outer[row, col],
            hspace=0.50, height_ratios=spec_h_ratios)

        # ---- Spatial subplot ----------------------------------------
        ax_sp = fig.add_subplot(inner[0])
        _setup_spatial_ax(ax_sp)
        _draw_goal_regions(ax_sp, goal_poses=goal_poses)

        kp_prev     = obs_np[t - 1, :18].reshape(9, 2)
        kp_curr     = obs_np[t,     :18].reshape(9, 2)
        agent_prev  = obs_np[t - 1, 18:20]
        agent_curr  = obs_np[t,     18:20]
        action_prev = action_np[t - 1]

        xp, yp, ap = _pose_from_keypoints(local_kps, kp_prev)
        xc, yc, ac = _pose_from_keypoints(local_kps, kp_curr)

        if has_obs and pred_obs_np is not None:
            kp_pred = pred_obs_np[t, :18].reshape(9, 2)
            xd, yd, ad = _pose_from_keypoints(local_kps, kp_pred)
            if not keypoint_only:
                agent_pred = pred_obs_np[t, 18:20]

        _draw_tblock(ax_sp, xp, yp, ap, fc=_C_PREV_BLOCK, ec=_C_PREV_EDGE,
                     alpha=0.25, lw=0.7, zorder=2,
                     label="T/Agent t-1 (ctx)" if col == 0 and row == 0 else None)
        if not keypoint_only:
            _draw_agent(ax_sp, agent_prev, fc=_C_PREV_AGENT, ec=_C_PREV_EDGE,
                        alpha=0.25, lw=0.7, zorder=2)

        _draw_arrow(ax_sp, agent_prev, action_prev, color=_C_ACTION_IN, alpha=0.88,
                    label="Action t-1 (input)" if col == 0 and row == 0 else None)

        _draw_tblock(ax_sp, xc, yc, ac, fc=_C_GT_BLOCK, ec=_C_GT_EDGE,
                     alpha=0.93, lw=1.2, zorder=4,
                     label="T/Agent t (GT)" if col == 0 and row == 0 else None)
        if not keypoint_only:
            _draw_agent(ax_sp, agent_curr, fc=_C_GT_AGENT, ec=_C_GT_EDGE,
                        alpha=0.93, lw=1.2, zorder=5)

        if has_obs and pred_obs_np is not None:
            _draw_tblock(ax_sp, xd, yd, ad, fc=_C_PRED_BLOCK, ec=_C_PRED_EDGE,
                         alpha=0.45, lw=1.5, zorder=6,
                         label="T t (obs pred)" if col == 0 and row == 0 else None)
            if not keypoint_only:
                _draw_agent(ax_sp, agent_pred, fc=_C_PRED_AGENT, ec=_C_PRED_EDGE,
                            alpha=0.45, lw=1.5, zorder=6)

        # GT future trajectory (pink)
        if traj_gt_all is not None:
            _draw_trajectory(ax_sp, traj_gt_all[t], _C_TRAJ_GT, alpha_range=(0.40, 0.95),
                             zorder=9, label="Traj GT" if col == 0 and row == 0 else None)

        # Predicted future trajectory (purple / multi-modal)
        if fut_pred_px is not None:
            if fut_pred_px.ndim == 4:
                # Multi-modal: fut_pred_px is (T, M, H, D)
                M = fut_pred_px.shape[1]

                # Determine winner mode: highest predicted probability
                if mode_probs is not None:
                    winner_idx = int(np.argmax(mode_probs[t]))
                else:
                    winner_idx = 0

                for m in range(M):
                    prob = mode_probs[t, m] if mode_probs is not None else 1.0 / M
                    is_winner = (m == winner_idx)
                    alpha_base = max(0.30, float(prob)) if is_winner else max(0.15, float(prob) * 0.7)
                    color = _MODE_COLORS[m % len(_MODE_COLORS)]
                    _draw_trajectory(
                        ax_sp, fut_pred_px[t, m], color,
                        alpha_range=(alpha_base * 0.5, alpha_base),
                        zorder=(15 if is_winner else 10 + m), label=None)
                    # Draw thicker outline for winner
                    if is_winner:
                        pts = fut_pred_px[t, m]
                        ax_sp.plot(pts[:, 0], pts[:, 1], '-', color=color,
                                   linewidth=3.5, alpha=0.35, zorder=14)

                # Per-subplot legend for modes
                mode_handles = []
                for m in range(M):
                    prob = mode_probs[t, m] if mode_probs is not None else 1.0 / M
                    is_winner = (m == winner_idx)
                    c = _MODE_COLORS[m % len(_MODE_COLORS)]
                    tag = f"M{m+1} {prob:.0%}" + (" \u2605" if is_winner else "")
                    mode_handles.append(
                        Line2D([0], [0], marker='o', color=c, markersize=5,
                               linewidth=2.0 if is_winner else 1.2,
                               label=tag))
                ax_sp.legend(handles=mode_handles, fontsize=9, loc='upper left',
                             framealpha=0.6, handlelength=1.2, labelspacing=0.25)
            else:
                _draw_trajectory(ax_sp, fut_pred_px[t], _C_TRAJ_PRED,
                                 alpha_range=(0.40, 0.95), zorder=10,
                                 label="Traj pred" if col == 0 and row == 0 else None)

        # GT past trajectory (orange) — only valid (non-padded) points
        if traj_past_gt_all is not None:
            vmask = past_valid_mask_all[t]          # (H_past,)
            if vmask.any():
                _draw_trajectory(ax_sp, traj_past_gt_all[t][vmask], _C_PAST_GT,
                                 alpha_range=(0.35, 0.85), zorder=8,
                                 label="Past GT" if col == 0 and row == 0 else None)

        # Predicted past trajectory (sienna) — only valid points
        if traj_past_pred_all is not None:
            vmask = past_valid_mask_all[t]
            if vmask.any():
                _draw_trajectory(ax_sp, traj_past_pred_all[t][vmask], _C_PAST_PRED,
                                 alpha_range=(0.35, 0.85), zorder=9,
                                 label="Past pred" if col == 0 and row == 0 else None)

        ax_sp.set_title(f"t = {t}", fontsize=11)

        # ---- Future chunk comparison panels -------------------------
        if show_future:
            gt_chunk = traj_gt_all[t]    # (H, D) pixel
            if fut_pred_px.ndim == 4:
                # Multi-modal: overlay all modes in the chunk panels
                M = fut_pred_px.shape[1]
                # Winner = highest predicted probability
                if mode_probs is not None:
                    _winner = int(np.argmax(mode_probs[t]))
                else:
                    _winner = 0
                for d in range(action_dim):
                    ax_c = fig.add_subplot(inner[1 + d])
                    xs = np.arange(H)
                    ax_c.plot(xs, gt_chunk[:, d], '-o', color=_C_TRAJ_GT,
                              markersize=3, label='GT')
                    for m in range(M):
                        prob = mode_probs[t, m] if mode_probs is not None else 1.0 / M
                        col_m = _MODE_COLORS[m % len(_MODE_COLORS)]
                        is_w = (m == _winner)
                        alpha = max(0.35, float(prob)) if is_w else max(0.20, float(prob) * 0.7)
                        lw = 2.0 if is_w else 1.0
                        tag = f'M{m+1} {prob:.0%}' + (' \u2605' if is_w else '')
                        ax_c.plot(xs, fut_pred_px[t, m, :, d], '-s', color=col_m,
                                  markersize=3 if is_w else 2, alpha=alpha,
                                  linewidth=lw, label=tag)
                    dim_labels = ['x', 'y', 'z']
                    dl = dim_labels[d] if d < len(dim_labels) else f'd{d}'
                    ax_c.set_ylabel(dl, fontsize=9)
                    ax_c.legend(fontsize=9, loc='upper right')
                    if d == 0:
                        ax_c.set_title('Future chunk (multi-modal)', fontsize=9)
            else:
                pred_chunk = fut_pred_px[t]    # (H, D) pixel
                if t + H <= T:
                    mae_px = np.abs(pred_chunk - gt_chunk).mean(axis=0)
                else:
                    mae_px = None
                spec_axes = [fig.add_subplot(inner[1 + d]) for d in range(action_dim)]
                _draw_chunk_panel(spec_axes, gt_chunk, pred_chunk, action_dim,
                                  mae_pixels=mae_px,
                                  gt_color=_C_TRAJ_GT, pred_color=_C_TRAJ_PRED,
                                  gt_label="Future GT", pred_label="Future pred",
                                  title_prefix="Future chunk")

        # ---- Past chunk comparison panels ---------------------------
        if show_past:
            past_row_offset = 1 + n_future_rows
            vmask = past_valid_mask_all[t]      # (H_past,)
            if vmask.any():
                gt_past   = traj_past_gt_all[t][vmask]    # (n_valid, D)
                pred_past = traj_past_pred_all[t][vmask]  # (n_valid, D)
                mae_px_past = np.abs(pred_past - gt_past).mean(axis=0)
                past_spec_axes = [
                    fig.add_subplot(inner[past_row_offset + d])
                    for d in range(action_dim)
                ]
                _draw_chunk_panel(past_spec_axes, gt_past, pred_past, action_dim,
                                  mae_pixels=mae_px_past,
                                  gt_color=_C_PAST_GT, pred_color=_C_PAST_PRED,
                                  gt_label="Past GT", pred_label="Past pred",
                                  title_prefix="Past chunk")
            else:
                for d in range(action_dim):
                    ax_empty = fig.add_subplot(inner[past_row_offset + d])
                    ax_empty.set_visible(False)

    # ---- Legend ---------------------------------------------------------
    def _p(fc, al, lbl):
        return mpatches.Patch(facecolor=fc, alpha=al, label=lbl)

    legend_handles = [
        _p(_C_GOAL,       0.55, "Goal regions"),
        _p(_C_PREV_BLOCK, 0.40, "T/Agent t-1 (input context)"),
        mpatches.Patch(facecolor=_C_ACTION_IN,  label="Action t-1 (red arrow, input)"),
        _p(_C_GT_BLOCK,   0.93, "T/Agent t   (GT)"),
    ]
    if has_obs:
        legend_handles.append(
            _p(_C_PRED_BLOCK, 0.55, "T t         (obs predicted, amber)"),
        )
    if has_future:
        if n_modes > 1:
            for m in range(n_modes):
                legend_handles.append(
                    mpatches.Patch(
                        facecolor=_MODE_COLORS[m % len(_MODE_COLORS)],
                        label=f"Future mode {m+1} pred (★=start ◆=end)"),
                )
            legend_handles.append(
                mpatches.Patch(facecolor='white', edgecolor='black',
                               label="★ in legend = winner (highest probability)"),
            )
            legend_handles.append(
                mpatches.Patch(facecolor=_C_TRAJ_GT,
                               label=f"Future traj t:t+{H} GT  (★=start ◆=end, pink)"),
            )
        else:
            legend_handles += [
                mpatches.Patch(facecolor=_C_TRAJ_GT,   label=f"Future traj t:t+{H} GT  (★=start ◆=end, pink)"),
                mpatches.Patch(facecolor=_C_TRAJ_PRED, label=f"Future traj t:t+{H} pred (★=start ◆=end, purple)"),
            ]
    if model.use_past_head:
        legend_handles += [
            mpatches.Patch(facecolor=_C_PAST_GT,
                           label=f"Past traj [t-1\u2026t-H_past] GT  (orange, H_past={H_past})"),
            mpatches.Patch(facecolor=_C_PAST_PRED,
                           label=f"Past traj [t-1\u2026t-H_past] pred (sienna, H_past={H_past})"),
        ]

    mode_str = (f"keypoint-only ({keypoint_dim}d)" if keypoint_only else f"full obs ({obs_np.shape[-1]}d)")
    mode_count_str = f"  modes={n_modes}" if n_modes > 1 else ""
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, bot_frac * 0.55))
    fig.suptitle(
        f"{episode_label}   [{mode_str}  |  chunk H={H}  past_H={H_past}{mode_count_str}]",
        fontsize=12, y=(top_frac + 1.0) / 2,
    )
    plt.savefig(output_path, dpi=110, format="pdf")
    plt.close(fig)

    # ---- DCT coefficient comparison PDF (only in DCT mode) ----------------
    if dct_coeffs_gt_all is not None:
        _save_dct_comparison_pdf(
            dct_coeffs_gt_all, dct_coeffs_pred_all,
            past_valid_mask_all, sampled_t,
            action_dim, episode_label, output_path,
        )


# ---------------------------------------------------------------------------
# Quantitative evaluation
# ---------------------------------------------------------------------------

def compute_trajectory_errors(
    model:         ObsActionChunkLSTM,
    normalizer:    dict,
    dataset,
    device:        str,
    keypoint_only: bool,
    keypoint_dim:  int,
    n_episodes:    int,
) -> dict:
    """
    Compute MAE and RMSE of the reconstructed H-step future action trajectory
    vs ground truth over a set of episodes and all valid timesteps.

    Returns empty metrics if the future head is disabled.
    """
    if not model.use_future_head:
        return {
            "traj_mae_mean":  float('nan'),
            "traj_mae_std":   float('nan'),
            "traj_rmse_mean": float('nan'),
        }

    H = model.chunk_H
    all_mae  = []
    all_rmse = []

    indices = np.round(np.linspace(0, len(dataset) - 1, n_episodes)).astype(int)

    model.eval()
    with torch.no_grad():
        for ep_idx in indices:
            sample    = dataset[int(ep_idx)]
            obs_np    = np.array(sample["obs"])
            action_np = np.array(sample["action"])
            T = len(obs_np)
            if T < H + 1:
                continue

            obs_t    = torch.from_numpy(obs_np).float().unsqueeze(0)
            action_t = torch.from_numpy(action_np).float().unsqueeze(0)
            lengths  = torch.tensor([T])

            obs_norm    = _normalize_seq(normalizer, obs_t,    "obs")
            action_norm = _normalize_seq(normalizer, action_t, "action")
            obs_model   = obs_norm[:, :, :keypoint_dim] if keypoint_only else obs_norm

            _, future_pred, _, _, _ = model.predict(
                obs_model.to(device), action_norm.to(device),
                lengths=lengths, hidden_state=None,
            )
            if future_pred.dim() == 4:
                # Single mode: (1, T, H, D)
                fut_pred_np = future_pred.cpu().squeeze(0).numpy()   # (T, H, D)
            else:
                # Multi-modal: (1, T, M, H, D) — use best mode per timestep (lowest error)
                fut_all = future_pred.cpu().squeeze(0).numpy()       # (T, M, H, D)
                # Select mode closest to GT at each valid t
                M = fut_all.shape[1]
                action_dim = action_np.shape[-1]
                fut_pred_np = np.zeros((T, H, action_dim), dtype=np.float32)
                for t in range(T - H + 1):
                    gt_px = action_np[t:t + H]
                    best_err = float('inf')
                    for m in range(M):
                        pred_px_m = _denormalize_chunk_np(normalizer, fut_all[t, m])
                        err_m = np.abs(pred_px_m - gt_px).mean()
                        if err_m < best_err:
                            best_err = err_m
                            fut_pred_np[t] = fut_all[t, m]  # keep in norm space

            ep_mae  = []
            ep_rmse = []
            for t in range(T - H + 1):
                pred_px = _denormalize_chunk_np(normalizer, fut_pred_np[t])
                gt_px   = action_np[t:t + H]
                err     = np.abs(pred_px - gt_px)
                ep_mae.append(err.mean())
                ep_rmse.append(np.sqrt((err ** 2).mean()))

            all_mae.append(np.mean(ep_mae))
            all_rmse.append(np.mean(ep_rmse))

    return {
        "traj_mae_mean":  float(np.mean(all_mae)),
        "traj_mae_std":   float(np.std(all_mae)),
        "traj_rmse_mean": float(np.mean(all_rmse)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and visualise ObsActionChunkLSTM")
    parser.add_argument("--checkpoint", type=str,
                        default="./outputs/obs_action_chunk_lstm/best_model.pt")
    parser.add_argument("--config",      type=str, default=None)
    parser.add_argument("--zarr_path",   type=str,
                        default="data/pusht_three_goals_demo_with_indicator.zarr")
    parser.add_argument("--output_dir",  type=str,
                        default="./outputs/obs_action_chunk_lstm/eval_viz")
    parser.add_argument("--n_episodes",  type=int,   default=5)
    parser.add_argument("--n_timesteps", type=int,   default=8)
    parser.add_argument("--val_ratio",   type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--device",      type=str,   default="cuda")
    parser.add_argument("--keypoint_only", action="store_true")
    parser.add_argument("--keypoint_dim",  type=int, default=18)
    parser.add_argument("--show_chunk",    action="store_true",
                        help="Show per-dimension action value comparison panels")
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k) and k not in ("output_dir",):
                setattr(args, k, v)

    # ---- Load checkpoint ------------------------------------------------
    ckpt_path = Path(args.checkpoint)
    assert ckpt_path.exists(), f"Not found: {ckpt_path}"
    ckpt      = _load_torch_artifact(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    for k in ("hidden_size", "num_layers", "dropout",
              "keypoint_only", "keypoint_dim",
              "chunk_H", "num_future_modes",
              "future_abstraction", "future_n_bases",
              "use_future_head", "use_obs_head", "use_past_head", "past_chunk_H",
              "past_abstraction", "past_n_bases",
              "action_only",
              "use_vq", "vq_n_codes", "vq_commitment_weight",
              "include_goal_keypoints"):
        if k in ckpt_args:
            setattr(args, k, ckpt_args[k])
    for k in ("zarr_path", "val_ratio", "seed"):
        if k in ckpt_args:
            setattr(args, k, ckpt_args[k])

    device        = args.device if torch.cuda.is_available() else "cpu"
    keypoint_only = bool(args.keypoint_only)
    keypoint_dim  = int(args.keypoint_dim)
    include_goal_keypoints = bool(getattr(args, "include_goal_keypoints", False))
    chunk_H       = int(getattr(args, "chunk_H",      16))
    past_chunk_H  = int(getattr(args, "past_chunk_H", chunk_H))

    print(f"\n{'='*65}")
    print("ObsActionChunkLSTM \u2013 evaluation & visualisation")
    print(f"{'='*65}")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Device       : {device}")
    print(f"chunk_H      : {chunk_H}  |  past_chunk_H: {past_chunk_H}")
    print(f"use_future_head: {bool(getattr(args, 'use_future_head', True))}")
    print(f"use_obs_head   : {bool(getattr(args, 'use_obs_head', True))}")
    print(f"keypoint_only: {keypoint_only}  (dim={keypoint_dim})")
    print(f"include_goal_keypoints: {include_goal_keypoints}")

    # ---- Output directories ---------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train").mkdir(exist_ok=True)
    (output_dir / "val").mkdir(exist_ok=True)

    # ---- Local block keypoints ------------------------------------------
    from memory_diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
    from memory_diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager
    kp_kwargs  = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    kp_manager = PymunkKeypointManager(**kp_kwargs)
    local_kps  = kp_manager.local_keypoint_map["block"]

    # ---- LP Normalizer --------------------------------------------------
    norm_path = ckpt_path.parent / "normalizer.pt"
    if norm_path.exists():
        normalizer = _load_torch_artifact(norm_path, map_location="cpu")
    else:
        print("Normalizer not found \u2013 recomputing ...")
        tmp_ds = PushTThreeGoalsLSTMDatasetWithIndicator(
            zarr_path=args.zarr_path, val_ratio=0.0, seed=args.seed,
            include_goal_keypoints=include_goal_keypoints)
        normalizer = tmp_ds.get_normalizer()
        del tmp_ds

    # ---- Dataset --------------------------------------------------------
    train_ds = PushTThreeGoalsLSTMDatasetWithIndicator(
        zarr_path=args.zarr_path, val_ratio=args.val_ratio, seed=args.seed,
        include_goal_keypoints=include_goal_keypoints)
    val_ds = train_ds.get_validation_dataset()
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # ---- Open zarr for per-episode goal keypoints -----------------------
    import zarr
    zarr_root = zarr.open(args.zarr_path, mode='r')
    zarr_episode_ends = np.array(zarr_root['meta/episode_ends'])  # (n_episodes,)
    zarr_goal_kps = {
        g: zarr_root[f'data/{g}']
        for g in ('goal_1_keypoint', 'goal_2_keypoint', 'goal_3_keypoint')
    }

    # ---- Model ----------------------------------------------------------
    obs_dim       = 74 if include_goal_keypoints else 20
    action_dim    = 2
    model_obs_dim = keypoint_dim if keypoint_only else obs_dim

    model = ObsActionChunkLSTM(
        obs_dim=model_obs_dim,
        action_dim=action_dim,
        hidden_size=getattr(args, "hidden_size",  256),
        num_layers=getattr(args,  "num_layers",   2),
        dropout=getattr(args,     "dropout",      0.1),
        chunk_H=chunk_H,
        use_future_head=bool(getattr(args, "use_future_head", True)),
        num_future_modes=int(getattr(args, "num_future_modes", 1)),
        future_abstraction=getattr(args, "future_abstraction", "raw"),
        future_n_bases=int(getattr(args, "future_n_bases", 32)),
        use_obs_head=bool(getattr(args, "use_obs_head", True)),
        use_past_head=bool(getattr(args, "use_past_head", False)),
        past_chunk_H=past_chunk_H,
        past_abstraction=getattr(args, "past_abstraction", "raw"),
        past_n_bases=int(getattr(args, "past_n_bases", 32)),
        action_only=bool(getattr(args, "action_only", False)),
        use_vq=bool(getattr(args, "use_vq", False)),
        vq_n_codes=int(getattr(args, "vq_n_codes", 512)),
        vq_commitment_weight=float(getattr(args, "vq_commitment_weight", 0.25)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model loaded  (epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', float('nan')):.5f})")

    # ---- Quantitative evaluation ----------------------------------------
    if model.use_future_head:
        print(f"\n{'='*65}")
        print("Quantitative trajectory prediction errors")
        print(f"{'='*65}")
        for split_name, ds in [("train", train_ds), ("val", val_ds)]:
            n_ep = min(args.n_episodes * 2, len(ds))
            err  = compute_trajectory_errors(
                model, normalizer, ds, device, keypoint_only, keypoint_dim, n_ep)
            print(
                f"  {split_name:5s} ({n_ep} eps) | "
                f"traj MAE={err['traj_mae_mean']:.2f}\u00b1{err['traj_mae_std']:.2f}px  "
                f"traj RMSE={err['traj_rmse_mean']:.2f}px"
            )
    else:
        print(f"\n{'='*65}")
        print("Future head disabled — skipping future trajectory error metrics")
        print(f"{'='*65}")

    # ---- Visualise episodes ---------------------------------------------
    for split_name, dataset in [("train", train_ds), ("val", val_ds)]:
        n_ep       = min(args.n_episodes, len(dataset))
        ep_indices = np.round(np.linspace(0, len(dataset) - 1, n_ep)).astype(int)
        print(f"\n--- {split_name} split ({n_ep} episodes) ---")

        for rank, ep_idx in enumerate(ep_indices):
            s         = dataset[int(ep_idx)]
            obs_np    = np.array(s["obs"])
            action_np = np.array(s["action"])
            T         = len(obs_np)
            label     = f"{split_name} ep {ep_idx}  (T={T})"
            out_path  = output_dir / split_name / f"ep_{ep_idx:04d}.pdf"
            print(f"  [{rank+1}/{n_ep}] {label}  -> {out_path.name}")

            # Read per-episode goal keypoints directly from the zarr file
            actual_ep_idx = dataset.episode_indices[int(ep_idx)]
            ep_start = int(zarr_episode_ends[actual_ep_idx - 1]) if actual_ep_idx > 0 else 0
            ep_goal_poses = []
            for gk in ('goal_1_keypoint', 'goal_2_keypoint', 'goal_3_keypoint'):
                goal_kps = np.array(zarr_goal_kps[gk][ep_start])  # (9, 2) at first timestep
                gx, gy, ga = _pose_from_keypoints(local_kps, goal_kps)
                ep_goal_poses.append(np.array([gx, gy, ga]))

            visualise_episode(
                model=model,
                normalizer=normalizer,
                local_kps=local_kps,
                obs_np=obs_np,
                action_np=action_np,
                device=device,
                keypoint_only=keypoint_only,
                keypoint_dim=keypoint_dim,
                n_timesteps=args.n_timesteps,
                episode_label=label,
                output_path=out_path,
                show_chunk=args.show_chunk,
                goal_poses=ep_goal_poses,
            )

    print(f"\nAll saved to: {output_dir}")


if __name__ == "__main__":
    main()
