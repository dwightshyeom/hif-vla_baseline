#!/usr/bin/env python3
"""
Training script for LSTM with Observation Prediction Head + Raw Action Chunk Heads.

The model (ObsActionChunkLSTM) shares a single LSTM encoder and decodes the
hidden state through up to three heads:

    obs_head          : hidden → obs_t                         (MSE loss)
    future_chunk_head : hidden → norm(a_{t:t+H})               (MSE loss, H steps, optional)
    past_chunk_head   : hidden → norm(rev past pastH actions)  (MSE loss, optional)

Unlike the FAST variant, there is NO tokenizer or DCT transform.  The chunk
heads directly predict LP-normalised action values in [−1, 1] ≈ space.

Usage
-----
    python train_obs_action_chunk_lstm.py --config config/obs_action_chunk_lstm_config.yaml
"""

import argparse
import json
import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed.")

from memory_diffusion_policy.dataset.pusht_three_goals_lstm_dataset import (
    PushTThreeGoalsLSTMDatasetWithIndicator,
    collate_fn_lstm_with_indicator,
)
from memory_diffusion_policy.model.obs_action_chunk_lstm import (
    ObsActionChunkLSTM,
    compute_obs_action_chunk_loss,
    greedy_spatial_grouping,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_seq(normalizer, x: torch.Tensor, key: str) -> torch.Tensor:
    """Normalise (B, T, D) via the LP normalizer (flattens/restores T axis)."""
    B, T, D = x.shape
    x_flat = x.reshape(B * T, D).unsqueeze(1)
    return normalizer[key].normalize(x_flat).squeeze(1).reshape(B, T, D).to(x.device)


def build_future_chunk_targets(
    action_norm_np: np.ndarray,   # (B, T, action_dim)  LP-normalised
    lengths_np:     np.ndarray,   # (B,)
    chunk_H:        int,
) -> tuple:
    """
    Build LP-normalised future action chunk targets.

    For each (b, t) where t + chunk_H <= T_b:
        future_target[b, t] = action_norm_np[b, t : t+chunk_H]

    Returns
    -------
    targets    : (B, T, chunk_H, action_dim)  float32
    future_mask: (B, T)                       float32 (1 = valid)
    """
    B, T, D = action_norm_np.shape
    targets = np.zeros((B, T, chunk_H, D), dtype=np.float32)
    mask    = np.zeros((B, T),             dtype=np.float32)

    step_offsets = np.arange(chunk_H)           # (H,)
    for b in range(B):
        T_b = int(lengths_np[b])
        n   = max(0, T_b - chunk_H + 1)         # number of valid starting steps
        if n > 0:
            row_idx = np.arange(n)[:, None]      # (n, 1)
            col_idx = row_idx + step_offsets     # (n, H)
            targets[b, :n] = action_norm_np[b, col_idx]   # (n, H, D)
            mask[b, :n]    = 1.0

    return targets, mask


def build_past_chunk_targets(
    action_norm_np: np.ndarray,   # (B, T, action_dim)  LP-normalised
    lengths_np:     np.ndarray,   # (B,)
    past_chunk_H:   int,
) -> tuple:
    """
    Build LP-normalised reversed past action chunk targets with zero-padding
    for early timesteps.

    For each (b, t) where 0 <= t < T_b:
        past_target[b, t, k] = action_norm_np[b, t-1-k]  if t-1-k >= 0
                                0.0                        otherwise
        (k = 0, 1, ..., past_chunk_H-1)

    Note: slot k=0 corresponds to a_{t-1}, which is ALSO the action input to
    LSTM step t (the LSTM sees {obs_{t-1}, a_{t-1}} and is asked to emit
    a_{t-1}, a_{t-2}, …, a_{t-past_chunk_H} as past-chunk output).

    Early timesteps where some or all past indices would be negative are
    zero-padded so the model still receives full-length chunks.  The returned
    per-position mask indicates which slots hold real data so the loss is
    only computed on actual past actions (not the padded zeros).

    Returns
    -------
    targets   : (B, T, past_chunk_H, action_dim)  float32
    past_mask : (B, T, past_chunk_H)              float32  per-position validity
    """
    B, T, D = action_norm_np.shape
    targets = np.zeros((B, T, past_chunk_H, D), dtype=np.float32)
    mask    = np.zeros((B, T, past_chunk_H),    dtype=np.float32)

    # offsets: 1, 2, ..., past_chunk_H  (distance back from t; includes t-1)
    offsets = np.arange(1, past_chunk_H + 1)   # (past_chunk_H,)

    for b in range(B):
        T_b = int(lengths_np[b])
        if T_b == 0:
            continue
        t_vals = np.arange(T_b)                            # (T_b,)
        idx    = t_vals[:, None] - offsets[None, :]        # (T_b, past_chunk_H)
        valid  = idx >= 0                                  # (T_b, past_chunk_H)
        idx_clamped = np.clip(idx, 0, T_b - 1)
        gathered = action_norm_np[b, idx_clamped]          # (T_b, past_chunk_H, D)
        gathered[~valid] = 0.0                             # zero-pad negative indices
        targets[b, :T_b] = gathered
        mask[b, :T_b]    = valid.astype(np.float32)        # 1 only for real data

    return targets, mask


def build_spatial_keypoint_targets(
    action_norm_np: np.ndarray,   # (B, T, action_dim)  LP-normalised
    lengths_np:     np.ndarray,   # (B,)
    past_chunk_H:   int,
    n_groups:       int,
    spatial_threshold: float = 0.1,
) -> tuple:
    """
    Build spatial-keypoint targets for the past trajectory.

    For each (b, t) the valid past actions (reversed: recent→older) are fed
    to ``greedy_spatial_grouping`` which returns K centroids and normalised
    durations.  These are packed into target tensors suitable for the
    spatial_keypoint loss path.

    Returns
    -------
    centroid_dur_targets : (B, T, K, action_dim + 1)  centroids + duration
    kp_mask              : (B, T, K)                  per-group validity
    """
    B, T, D = action_norm_np.shape
    K = n_groups

    targets = np.zeros((B, T, K, D + 1), dtype=np.float32)
    kp_mask = np.zeros((B, T, K),        dtype=np.float32)

    for b in range(B):
        T_b = int(lengths_np[b])
        for t in range(T_b):
            # Reversed past indices: t-1, t-2, ... (includes t-1)
            n_avail = t   # number of valid past actions (indices 0..t-1)
            if n_avail < 1:
                continue
            n_past = min(n_avail, past_chunk_H)
            indices = np.arange(t - 1, t - 1 - n_past, -1)  # all >= 0
            past_actions = action_norm_np[b, indices]         # (n_past, D)

            centroids, durations = greedy_spatial_grouping(
                past_actions, K, spatial_threshold)

            targets[b, t, :, :D] = centroids
            targets[b, t, :, D]  = durations
            n_real = min(n_past, K)
            kp_mask[b, t, :n_real] = 1.0

    return targets, kp_mask


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model:              nn.Module,
    dataloader:         DataLoader,
    optimizer:          torch.optim.Optimizer,
    normalizer:         dict,
    device:             str,
    epoch:              int,
    writer:             SummaryWriter = None,
    l1_on_hidden_state: bool  = False,
    l1_weight:          float = 0.0,
    obs_loss_weight:    float = 1.0,
    future_loss_weight: float = 1.0,
    keypoint_only:      bool  = False,
    keypoint_dim:       int   = 18,
    use_obs_head:       bool  = True,
    use_future_head:    bool  = True,
    use_past_head:      bool  = False,
    past_loss_weight:   float = 1.0,
    past_temporal_weight: str   = 'uniform',
    past_temporal_alpha:  float = 2.0,
    past_dct_basis:       Optional[torch.Tensor] = None,
    past_dct_freq_decay:  float = 0.0,
    past_consistency_weight: float = 0.0,
    past_consistency_n_steps: list = None,
    past_abstraction:     str   = 'raw',
    spatial_threshold:    float = 0.1,
    action_noise_scale:  float = 0.0,
    action_dropout_prob: float = 0.0,
    noise_warmup_ratio:  float = 1.0,
    future_mode_cls_weight: float = 0.0,
    future_entropy_weight:  float = 0.0,
    future_dct_basis:       Optional[torch.Tensor] = None,
    future_dct_freq_decay:  float = 0.0,
) -> dict:
    model.train()

    total_loss       = 0.0
    total_obs_mse    = 0.0
    total_future_mse = 0.0
    total_past_mse   = 0.0
    total_l1         = 0.0
    total_consistency = 0.0
    total_vq         = 0.0
    total_mode_cls   = 0.0
    total_entropy    = 0.0
    num_batches      = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        obs     = batch['obs'].to(device)       # (B, T, 20)
        actions = batch['action'].to(device)    # (B, T, 2)
        mask    = batch['mask'].to(device)      # (B, T)
        lengths = batch['lengths']              # CPU

        obs_norm    = _normalize_seq(normalizer, obs,     'obs')
        action_norm = _normalize_seq(normalizer, actions, 'action')

        # ---- Action noise augmentation (training only) -------------------
        if action_noise_scale > 0.0 or action_dropout_prob > 0.0:
            _scale = action_noise_scale * noise_warmup_ratio
            _drop  = action_dropout_prob * noise_warmup_ratio
            # Gaussian noise → "noisy trajectory" (for LSTM input + past targets)
            action_norm_noisy = action_norm
            if _scale > 0.0:
                action_norm_noisy = action_norm + torch.randn_like(action_norm) * _scale
            # Extra dropout on LSTM input only (not past targets)
            action_norm_input = action_norm_noisy
            if _drop > 0.0:
                keep = (torch.rand(action_norm.shape[0], action_norm.shape[1], 1,
                                   device=action_norm.device) >= _drop).float()
                action_norm_input = action_norm_noisy * keep
        else:
            action_norm_noisy = action_norm
            action_norm_input = action_norm

        obs_model  = obs_norm[:, :, :keypoint_dim] if keypoint_only else obs_norm
        obs_target = obs_model if use_obs_head else None

        # ---- Build chunk targets (from normalised actions) ---------------
        # Past targets use noisy actions (LSTM reconstructs what it saw).
        # Future targets use clean actions (predict the true intended trajectory).
        action_noisy_np = action_norm_noisy.cpu().numpy()
        action_clean_np = action_norm.cpu().numpy()
        lengths_np      = lengths.numpy()

        future_target = future_mask_t = None
        if use_future_head:
            future_np, future_mask_np = build_future_chunk_targets(
                action_clean_np, lengths_np, model.chunk_H)
            future_target = torch.from_numpy(future_np).to(device)
            future_mask_t = torch.from_numpy(future_mask_np).to(device)

        past_target = past_mask_t = None
        if use_past_head:
            if past_abstraction == 'spatial_keypoint':
                kp_np, kp_mask_np = build_spatial_keypoint_targets(
                    action_noisy_np, lengths_np, model.past_chunk_H,
                    model.past_n_bases, spatial_threshold)
                past_target = torch.from_numpy(kp_np).to(device)
                past_mask_t = torch.from_numpy(kp_mask_np).to(device)
            else:
                past_np, past_mask_np = build_past_chunk_targets(
                    action_noisy_np, lengths_np, model.past_chunk_H)
                past_target = torch.from_numpy(past_np).to(device)
                past_mask_t = torch.from_numpy(past_mask_np).to(device)

        # ---- Forward pass -----------------------------------------------
        (obs_pred, future_pred, past_pred, _, lstm_output,
         vq_loss, past_pred_coeffs, future_mode_logits, future_pred_coeffs) = model(
            obs_model, action_norm_input, lengths=lengths, hidden_state=None)

        loss_dict = compute_obs_action_chunk_loss(
            obs_pred, obs_target, mask,
            future_pred, future_target, future_mask_t,
            past_pred, past_target, past_mask_t,
            lstm_output, l1_on_hidden_state, l1_weight,
            obs_loss_weight, future_loss_weight, past_loss_weight,
            past_temporal_weight=past_temporal_weight,
            past_temporal_alpha=past_temporal_alpha,
            past_dct_basis=past_dct_basis,
            past_dct_freq_decay=past_dct_freq_decay,
            past_pred_coeffs=past_pred_coeffs,
            past_consistency_weight=past_consistency_weight,
            past_consistency_n_steps=past_consistency_n_steps,
            vq_loss=vq_loss,
            future_mode_logits=future_mode_logits,
            future_mode_cls_weight=future_mode_cls_weight,
            future_entropy_weight=future_entropy_weight,
            future_dct_basis=future_dct_basis,
            future_dct_freq_decay=future_dct_freq_decay,
            future_pred_coeffs=future_pred_coeffs,
        )
        loss = loss_dict['loss']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss       += loss.item()
        total_obs_mse    += loss_dict['obs_mse_loss'].item()
        total_future_mse += loss_dict['future_mse_loss'].item()
        total_past_mse   += loss_dict['past_mse_loss'].item()
        total_l1         += loss_dict['l1_loss'].item()
        total_consistency += loss_dict['consistency_loss'].item()
        total_vq         += loss_dict['vq_loss'].item()
        total_mode_cls   += loss_dict['future_mode_cls_loss'].item()
        total_entropy    += loss_dict['future_entropy_loss'].item()
        num_batches      += 1

        pbar.set_postfix({
            'loss':   f"{loss.item():.4f}",
            'obs':    f"{loss_dict['obs_mse_loss'].item():.4f}",
            'fut':    f"{loss_dict['future_mse_loss'].item():.4f}",
            'past':   f"{loss_dict['past_mse_loss'].item():.4f}",
            'cons':   f"{loss_dict['consistency_loss'].item():.4f}",
            'vq':     f"{loss_dict['vq_loss'].item():.4f}",
        })

        if writer is not None:
            gs = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/loss_step',       loss.item(),                             gs)
            writer.add_scalar('train/obs_mse_step',    loss_dict['obs_mse_loss'].item(),        gs)
            writer.add_scalar('train/future_mse_step', loss_dict['future_mse_loss'].item(),     gs)
            writer.add_scalar('train/past_mse_step',   loss_dict['past_mse_loss'].item(),       gs)

    return {
        'loss':             total_loss        / num_batches,
        'obs_mse':          total_obs_mse     / num_batches,
        'future_mse':       total_future_mse  / num_batches,
        'past_mse':         total_past_mse    / num_batches,
        'l1_loss':          total_l1          / num_batches,
        'consistency_loss': total_consistency / num_batches,
        'vq_loss':          total_vq          / num_batches,
        'mode_cls_loss':    total_mode_cls    / num_batches,
        'entropy_loss':     total_entropy     / num_batches,
    }


# ---------------------------------------------------------------------------
# One validation epoch
# ---------------------------------------------------------------------------

def validate_epoch(
    model:              nn.Module,
    dataloader:         DataLoader,
    normalizer:         dict,
    device:             str,
    epoch:              int,
    writer:             SummaryWriter = None,
    l1_on_hidden_state: bool  = False,
    l1_weight:          float = 0.0,
    obs_loss_weight:    float = 1.0,
    future_loss_weight: float = 1.0,
    keypoint_only:      bool  = False,
    keypoint_dim:       int   = 18,
    use_obs_head:       bool  = True,
    use_future_head:    bool  = True,
    use_past_head:      bool  = False,
    past_loss_weight:   float = 1.0,
    past_temporal_weight: str   = 'uniform',
    past_temporal_alpha:  float = 2.0,
    past_dct_basis:       Optional[torch.Tensor] = None,
    past_dct_freq_decay:  float = 0.0,
    past_consistency_weight: float = 0.0,
    past_consistency_n_steps: list = None,
    past_abstraction:     str   = 'raw',
    spatial_threshold:    float = 0.1,
    future_mode_cls_weight: float = 0.0,
    future_entropy_weight:  float = 0.0,
    future_dct_basis:       Optional[torch.Tensor] = None,
    future_dct_freq_decay:  float = 0.0,
) -> dict:
    model.eval()

    total_loss           = 0.0
    total_obs_mse        = 0.0
    total_future_mse     = 0.0
    total_past_mse       = 0.0
    total_l1             = 0.0
    total_consistency    = 0.0
    total_vq             = 0.0
    total_mode_cls       = 0.0
    total_entropy        = 0.0
    total_obs_per_dim    = None
    num_batches          = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            obs     = batch['obs'].to(device)
            actions = batch['action'].to(device)
            mask    = batch['mask'].to(device)
            lengths = batch['lengths']

            obs_norm    = _normalize_seq(normalizer, obs,     'obs')
            action_norm = _normalize_seq(normalizer, actions, 'action')

            obs_model  = obs_norm[:, :, :keypoint_dim] if keypoint_only else obs_norm
            obs_target = obs_model if use_obs_head else None

            action_norm_np = action_norm.cpu().numpy()
            lengths_np     = lengths.numpy()

            future_target = future_mask_t = None
            if use_future_head:
                future_np, future_mask_np = build_future_chunk_targets(
                    action_norm_np, lengths_np, model.chunk_H)
                future_target = torch.from_numpy(future_np).to(device)
                future_mask_t = torch.from_numpy(future_mask_np).to(device)

            past_target = past_mask_t = None
            if use_past_head:
                if past_abstraction == 'spatial_keypoint':
                    kp_np, kp_mask_np = build_spatial_keypoint_targets(
                        action_norm_np, lengths_np, model.past_chunk_H,
                        model.past_n_bases, spatial_threshold)
                    past_target = torch.from_numpy(kp_np).to(device)
                    past_mask_t = torch.from_numpy(kp_mask_np).to(device)
                else:
                    past_np, past_mask_np = build_past_chunk_targets(
                        action_norm_np, lengths_np, model.past_chunk_H)
                    past_target = torch.from_numpy(past_np).to(device)
                    past_mask_t = torch.from_numpy(past_mask_np).to(device)

            (obs_pred, future_pred, past_pred, _, lstm_output,
             vq_loss, past_pred_coeffs, future_mode_logits, future_pred_coeffs) = model(
                obs_model, action_norm, lengths=lengths, hidden_state=None)

            loss_dict = compute_obs_action_chunk_loss(
                obs_pred, obs_target, mask,
                future_pred, future_target, future_mask_t,
                past_pred, past_target, past_mask_t,
                lstm_output, l1_on_hidden_state, l1_weight,
                obs_loss_weight, future_loss_weight, past_loss_weight,
                past_temporal_weight=past_temporal_weight,
                past_temporal_alpha=past_temporal_alpha,
                past_dct_basis=past_dct_basis,
                past_dct_freq_decay=past_dct_freq_decay,
                past_pred_coeffs=past_pred_coeffs,
                past_consistency_weight=past_consistency_weight,
                past_consistency_n_steps=past_consistency_n_steps,
                vq_loss=vq_loss,
                future_mode_logits=future_mode_logits,
                future_mode_cls_weight=future_mode_cls_weight,
                future_entropy_weight=future_entropy_weight,
                future_dct_basis=future_dct_basis,
                future_dct_freq_decay=future_dct_freq_decay,
                future_pred_coeffs=future_pred_coeffs,
            )

            total_loss       += loss_dict['loss'].item()
            total_obs_mse    += loss_dict['obs_mse_loss'].item()
            total_future_mse += loss_dict['future_mse_loss'].item()
            total_past_mse   += loss_dict['past_mse_loss'].item()
            total_l1         += loss_dict['l1_loss'].item()
            total_consistency += loss_dict['consistency_loss'].item()
            total_vq         += loss_dict['vq_loss'].item()
            total_mode_cls   += loss_dict['future_mode_cls_loss'].item()
            total_entropy    += loss_dict['future_entropy_loss'].item()

            if total_obs_per_dim is None:
                total_obs_per_dim = loss_dict['obs_mse_per_dim'].cpu() if loss_dict['obs_mse_per_dim'] is not None else None
            else:
                if loss_dict['obs_mse_per_dim'] is not None:
                    total_obs_per_dim += loss_dict['obs_mse_per_dim'].cpu()

            num_batches += 1
            pbar.set_postfix({
                'loss': f"{loss_dict['loss'].item():.4f}",
                'obs':  f"{loss_dict['obs_mse_loss'].item():.4f}",
                'fut':  f"{loss_dict['future_mse_loss'].item():.4f}",
            })

    return {
        'loss':            total_loss        / num_batches,
        'obs_mse':         total_obs_mse     / num_batches,
        'future_mse':      total_future_mse  / num_batches,
        'past_mse':        total_past_mse    / num_batches,
        'l1_loss':         total_l1          / num_batches,
        'consistency_loss': total_consistency / num_batches,
        'vq_loss':         total_vq          / num_batches,
        'mode_cls_loss':   total_mode_cls    / num_batches,
        'entropy_loss':    total_entropy     / num_batches,
        'obs_mse_per_dim': total_obs_per_dim / num_batches if total_obs_per_dim is not None else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train LSTM with observation + raw action chunk heads'
    )
    parser.add_argument('--config',    type=str, default=None)
    parser.add_argument('--zarr_path', type=str,
                        default='/home/kuancheng/Desktop/memory_diffusion_policy/data/'
                                'pusht_three_goals_demo_with_indicator.zarr')
    parser.add_argument('--output_dir',  type=str,
                        default='./outputs/obs_action_chunk_lstm')
    parser.add_argument('--batch_size',  type=int,   default=16)
    parser.add_argument('--num_epochs',  type=int,   default=500)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--hidden_size', type=int,   default=256)
    parser.add_argument('--num_layers',  type=int,   default=2)
    parser.add_argument('--dropout',     type=float, default=0.1)
    parser.add_argument('--chunk_H',     type=int,   default=16,
                        help='Future action chunk length H')
    parser.add_argument('--obs_loss_weight',    type=float, default=1.0)
    parser.add_argument('--future_loss_weight', type=float, default=1.0)
    parser.add_argument('--l1_on_hidden_state', action='store_true')
    parser.add_argument('--l1_weight',          type=float, default=1e-3)
    parser.add_argument('--keypoint_only', action='store_true')
    parser.add_argument('--keypoint_dim', type=int, default=18)
    parser.add_argument('--val_ratio',    type=float, default=0.15)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--device',       type=str,   default='cuda')
    parser.add_argument('--num_workers',  type=int,   default=4)
    parser.add_argument('--use_wandb',    action='store_true')
    parser.add_argument('--wandb_project',  type=str, default='obs_action_chunk_lstm')
    parser.add_argument('--wandb_entity',   type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--use_past_head',   action='store_true',
                        help='Add a 3rd head predicting reversed past action chunk')
    parser.add_argument('--no_future_head', dest='use_future_head',
                        action='store_false',
                        help='Disable the future action chunk head (past-only + obs mode)')
    parser.set_defaults(use_future_head=True)
    parser.add_argument('--no_obs_head', dest='use_obs_head',
                        action='store_false',
                        help='Disable the observation prediction head (past-action-only mode)')
    parser.set_defaults(use_obs_head=True)
    parser.add_argument('--past_chunk_H',    type=int,   default=None,
                        help='Past chunk length (defaults to chunk_H when not set)')
    parser.add_argument('--past_loss_weight', type=float, default=1.0,
                        help='Weight for past chunk MSE loss')
    # Option 1: temporal reweighting
    parser.add_argument('--past_temporal_weight', type=str, default='uniform',
                        choices=['uniform', 'linear', 'exponential'],
                        help='Temporal weighting for past chunk loss')
    parser.add_argument('--past_temporal_alpha', type=float, default=2.0,
                        help='Strength of temporal reweighting (higher = more emphasis on older actions)')
    # Option 2: DCT abstraction
    parser.add_argument('--past_abstraction', type=str, default='raw',
                        choices=['raw', 'dct', 'segment_pool', 'dft', 'dwt',
                                 'spatial_keypoint'],
                        help=("'raw': predict all past actions; "
                              "'dct': predict DCT coefficients; "
                              "'segment_pool': predict piecewise-constant segment means; "
                              "'dft': predict real DFT coefficients (cos+sin per frequency); "
                              "'dwt': predict Haar wavelet coefficients (multi-scale); "
                              "'spatial_keypoint': group consecutive actions by spatial proximity"))
    parser.add_argument('--past_n_bases', type=int, default=32,
                        help='Number of DCT basis functions (only used when past_abstraction=dct)')
    parser.add_argument('--past_dct_freq_decay', type=float, default=0.0,
                        help=('Exponential frequency decay for DCT loss (>0 emphasises '
                              'low-freq coefficients; 0 = uniform). Only used with past_abstraction=dct.'))
    parser.add_argument('--past_consistency_weight', type=float, default=0.0,
                        help=('Weight for temporal consistency loss. Penalises disagreement '
                              'between N-step-offset predictions of the same past action. '
                              'Works in both raw and DCT mode.'))
    parser.add_argument('--past_consistency_n_steps', type=int, nargs='+', default=[1],
                        help=('List of N values for multi-step consistency loss. '
                              'E.g. --past_consistency_n_steps 1 2 4 8 enforces that '
                              'predictions at t and t+N agree on their overlapping region. '
                              'Default [1] matches the original adjacent-step behaviour.'))
    # VQ bottleneck on LSTM hidden states
    parser.add_argument('--use_vq', action='store_true',
                        help='Apply Vector Quantization bottleneck to LSTM hidden states')
    parser.add_argument('--vq_n_codes', type=int, default=512,
                        help='Number of VQ codebook entries')
    parser.add_argument('--vq_commitment_weight', type=float, default=0.25,
                        help='Weight for VQ commitment loss')
    # Spatial keypoint grouping threshold
    parser.add_argument('--spatial_threshold', type=float, default=0.1,
                        help='Euclidean distance threshold for grouping consecutive actions '
                             'into one spatial keypoint (only used with past_abstraction=spatial_keypoint)')
    # Action-only input mode
    parser.add_argument('--action_only', action='store_true',
                        help='Feed only action_{t-1} to LSTM (no observation input). '
                             'Useful for learning pure action-trajectory memory.')
    # Goal keypoints in observation
    parser.add_argument('--include_goal_keypoints', action='store_true',
                        help='Include 3 goal T keypoint locations in observation (adds 54 dims).'
                             ' Needed when goal positions vary across episodes.')
    # Action noise augmentation (robustness for DP co-training)
    parser.add_argument('--action_noise_scale', type=float, default=0.0,
                        help='Gaussian noise std in normalised action space (0 = disabled). '
                             'Recommended: 0.05-0.2.')
    parser.add_argument('--action_dropout_prob', type=float, default=0.0,
                        help='Per-timestep probability of zeroing action input to LSTM')
    parser.add_argument('--action_noise_warmup', type=int, default=0,
                        help='Epochs to linearly ramp noise from 0 to full scale (0 = instant)')
    # Multi-modal future prediction
    parser.add_argument('--num_future_modes', type=int, default=1,
                        help='Number of future prediction modes M. M=1: single-mode (standard MSE). '
                             'M>1: multi-modal with Winner-Takes-All loss.')
    parser.add_argument('--future_mode_cls_weight', type=float, default=0.1,
                        help='Cross-entropy weight for mode classification head (only when M>1)')
    parser.add_argument('--future_entropy_weight', type=float, default=0.01,
                        help='Entropy regularization weight to prevent mode collapse (only when M>1)')
    # Future abstraction (DCT)
    parser.add_argument('--future_abstraction', type=str, default='raw',
                        choices=['raw', 'dct'],
                        help="'raw': predict raw action chunks; 'dct': predict DCT coefficients")
    parser.add_argument('--future_n_bases', type=int, default=32,
                        help='Number of DCT basis functions for future abstraction (only when future_abstraction=dct)')
    parser.add_argument('--future_dct_freq_decay', type=float, default=0.0,
                        help='Exponential frequency decay for future DCT loss (>0 emphasises low-freq; 0=uniform)')
    # Action step subsampling (alignment with DP n_action_steps)
    parser.add_argument('--action_step_subsample', type=int, default=1,
                        help='Subsample episodes at this rate to match DP n_action_steps '
                             '(1 = no subsampling, 4 = every 4th step)')

    args = parser.parse_args()

    # ---- Load YAML config -----------------------------------------------
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    run_lstm_pretrain(args)


def run_lstm_pretrain(args):
    """Run LSTM pretraining given a populated argparse.Namespace.

    Split from ``main`` so the hydra workspace
    ``memory_diffusion_policy.workspace.train_obs_action_chunk_lstm_workspace``
    can drive the same training loop without re-parsing CLI args.
    """
    # ---- Reproducibility ------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device if torch.cuda.is_available() else 'cpu'

    past_chunk_H = args.past_chunk_H if args.past_chunk_H is not None else args.chunk_H

    print(f"\n{'='*70}")
    print("Training LSTM Obs + Raw Action Chunk Predictor")
    print(f"{'='*70}")
    print(f"Device        : {device}")
    print(f"Dataset       : {args.zarr_path}")
    print(f"Output        : {args.output_dir}")
    print(f"Hidden size   : {args.hidden_size}  |  Num layers: {args.num_layers}")
    print(f"chunk_H       : {args.chunk_H}  |  past_chunk_H: {past_chunk_H}")
    print(f"Obs weight    : {args.obs_loss_weight}  |  Future weight: {args.future_loss_weight}"
          f"  |  Past weight: {args.past_loss_weight}")
    print(f"use_future_head: {args.use_future_head}  |  use_past_head: {args.use_past_head}")
    print(f"use_obs_head   : {args.use_obs_head}")
    print(f"action_only   : {args.action_only}")
    if getattr(args, 'num_future_modes', 1) > 1:
        print(f"num_future_modes: {args.num_future_modes}  "
              f"mode_cls_w={args.future_mode_cls_weight}  "
              f"entropy_w={args.future_entropy_weight}")
    if getattr(args, 'action_noise_scale', 0) > 0 or getattr(args, 'action_dropout_prob', 0) > 0:
        print(f"Action noise  : scale={args.action_noise_scale}  "
              f"dropout={args.action_dropout_prob}  warmup={args.action_noise_warmup}ep")
    print(f"keypoint_only : {args.keypoint_only}  (keypoint_dim={args.keypoint_dim})")
    print(f"include_goal_keypoints: {getattr(args, 'include_goal_keypoints', False)}")
    print(f"action_step_subsample: {getattr(args, 'action_step_subsample', 1)}")

    # ---- Output directory -----------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(vars(args), f)

    # ---- Datasets -------------------------------------------------------
    print(f"\n{'='*70}")
    print("Loading dataset...")
    train_dataset = PushTThreeGoalsLSTMDatasetWithIndicator(
        zarr_path=args.zarr_path, val_ratio=args.val_ratio, seed=args.seed,
        include_goal_keypoints=bool(getattr(args, 'include_goal_keypoints', False)),
        action_step_subsample=int(getattr(args, 'action_step_subsample', 1)))
    val_dataset = train_dataset.get_validation_dataset()
    print(f"Train episodes: {len(train_dataset)}  |  Val episodes: {len(val_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn_lstm_with_indicator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_lstm_with_indicator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    # ---- LP normalizer --------------------------------------------------
    print("\nComputing LP normalizer...")
    normalizer = train_dataset.get_normalizer()
    torch.save(normalizer, output_dir / 'normalizer.pt')

    sample_batch = next(iter(train_dataloader))
    action_dim   = sample_batch['action'].shape[-1]
    obs_dim      = sample_batch['obs'].shape[-1]
    print(f"obs_dim={obs_dim}  action_dim={action_dim}")

    keypoint_only = args.keypoint_only
    keypoint_dim  = args.keypoint_dim
    model_obs_dim = keypoint_dim if keypoint_only else obs_dim

    # ---- Model ----------------------------------------------------------
    print(f"\n{'='*70}")
    print("Creating model...")
    model = ObsActionChunkLSTM(
        obs_dim=model_obs_dim,
        action_dim=action_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        chunk_H=args.chunk_H,
        use_obs_head=args.use_obs_head,
        use_future_head=args.use_future_head,
        num_future_modes=int(getattr(args, 'num_future_modes', 1)),
        future_abstraction=getattr(args, 'future_abstraction', 'raw'),
        future_n_bases=int(getattr(args, 'future_n_bases', 32)),
        use_past_head=args.use_past_head,
        past_chunk_H=past_chunk_H,
        past_abstraction=args.past_abstraction,
        past_n_bases=args.past_n_bases,
        action_only=bool(getattr(args, 'action_only', False)),
        use_vq=bool(getattr(args, 'use_vq', False)),
        vq_n_codes=int(getattr(args, 'vq_n_codes', 512)),
        vq_commitment_weight=float(getattr(args, 'vq_commitment_weight', 0.25)),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters   : {n_params:,}")
    print(f"LSTM input dim     : {model.input_dim}  ({'action only' if model.action_only else 'obs + action'})")
    if args.use_obs_head:
        print(f"Obs head out       : {model_obs_dim}")
    else:
        print(f"Obs head           : DISABLED")
    if args.use_future_head:
        M = int(getattr(args, 'num_future_modes', 1))
        fut_abst = getattr(args, 'future_abstraction', 'raw')
        modes_str = f"  modes={M}" if M > 1 else ""
        if fut_abst == 'dct':
            K_f = int(getattr(args, 'future_n_bases', 32))
            print(f"Future abstraction : DCT with {K_f} bases")
            print(f"Future chunk out   : {M * K_f * action_dim}  "
                  f"(M={M} × K={K_f} × D={action_dim}){modes_str}")
        else:
            print(f"Future abstraction : raw")
            print(f"Future chunk out   : {M * args.chunk_H * action_dim}  "
                  f"(M={M} × H={args.chunk_H} × D={action_dim}){modes_str}")
    else:
        print(f"Future chunk head  : DISABLED")
    if args.past_abstraction in ('dct', 'segment_pool', 'dft', 'dwt'):
        _labels = {'dct': 'DCT', 'segment_pool': 'Segment Pool',
                   'dft': 'DFT (real Fourier)', 'dwt': 'DWT (Haar wavelet)'}
        label = _labels[args.past_abstraction]
        from memory_diffusion_policy.model.obs_action_chunk_lstm import _basis_n_coeffs
        n_coeffs = _basis_n_coeffs(args.past_abstraction, args.past_n_bases)
        print(f"Past abstraction   : {label} with {args.past_n_bases} bases")
        print(f"Past head out      : {n_coeffs * action_dim}  "
              f"(n_coeffs={n_coeffs} × D={action_dim})")
    elif args.past_abstraction == 'spatial_keypoint':
        print(f"Past abstraction   : Spatial Keypoint with {args.past_n_bases} groups")
        print(f"Past head out      : {args.past_n_bases * (action_dim + 1)}  "
              f"(K={args.past_n_bases} × (D+1)={action_dim + 1})")
        print(f"Spatial threshold  : {args.spatial_threshold}")
    else:
        print(f"Past abstraction   : raw")
        print(f"Past chunk out     : {past_chunk_H * action_dim}  (H={past_chunk_H} × D={action_dim})")
    if args.past_temporal_weight != 'uniform':
        print(f"Past temporal wt   : {args.past_temporal_weight}  (alpha={args.past_temporal_alpha})")
    if getattr(args, 'use_vq', False):
        print(f"VQ bottleneck      : {args.vq_n_codes} codes, commitment_w={args.vq_commitment_weight}")

    # ---- Optimizer + scheduler ------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    # ---- Logging --------------------------------------------------------
    writer = SummaryWriter(log_dir=output_dir / 'logs')

    use_wandb = False
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed. Disabling.")
        else:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       name=args.wandb_run_name, config=vars(args))
            use_wandb = True

    # ---- Training loop --------------------------------------------------
    print(f"\n{'='*70}")
    print("Training...")
    print(f"{'='*70}\n")

    best_val_loss = float('inf')
    best_epoch    = 0

    l1_on    = args.l1_on_hidden_state
    l1_w     = args.l1_weight
    obs_w    = args.obs_loss_weight
    fut_w    = args.future_loss_weight
    past_w   = args.past_loss_weight
    use_obs    = args.use_obs_head
    use_future = args.use_future_head
    use_past = args.use_past_head
    past_tw  = args.past_temporal_weight
    past_ta  = args.past_temporal_alpha
    cons_w   = float(getattr(args, 'past_consistency_weight', 0.0))
    cons_n   = list(getattr(args, 'past_consistency_n_steps', [1]) or [1])

    # ---- Mutual exclusivity: Option 1 and Option 2 cannot be combined ------
    if args.past_abstraction in ('dct', 'segment_pool', 'dft', 'dwt') and past_tw != 'uniform':
        raise ValueError(
            "Option 1 (past_temporal_weight != 'uniform') and "
            f"Option 2 (past_abstraction='{args.past_abstraction}') cannot be used together. "
            f"Set past_temporal_weight='uniform' when using {args.past_abstraction} abstraction."
        )
    if args.past_abstraction == 'spatial_keypoint' and past_tw != 'uniform':
        raise ValueError(
            "past_temporal_weight != 'uniform' is incompatible with "
            "past_abstraction='spatial_keypoint'. Set past_temporal_weight='uniform'."
        )

    # ---- Basis for coefficient-space loss (DCT, segment pool, DFT, DWT) --------
    past_dct_basis_loss = None
    past_dct_freq_decay = float(getattr(args, 'past_dct_freq_decay', 0.0))
    if args.past_abstraction in ('dct', 'segment_pool', 'dft', 'dwt') and hasattr(model, '_past_dct_basis'):
        past_dct_basis_loss = model._past_dct_basis  # (H_past, K), stays on CPU until copied in loss

    future_dct_basis_loss = None
    future_dct_freq_decay = float(getattr(args, 'future_dct_freq_decay', 0.0))
    if getattr(args, 'future_abstraction', 'raw') == 'dct' and hasattr(model, '_future_dct_basis'):
        future_dct_basis_loss = model._future_dct_basis  # (H, K_f)

    past_abst = args.past_abstraction
    spat_thr  = float(getattr(args, 'spatial_threshold', 0.1))

    noise_scale  = float(getattr(args, 'action_noise_scale', 0.0))
    noise_drop   = float(getattr(args, 'action_dropout_prob', 0.0))
    noise_warmup = int(getattr(args, 'action_noise_warmup', 0))

    mode_cls_w  = float(getattr(args, 'future_mode_cls_weight', 0.0))
    ent_w       = float(getattr(args, 'future_entropy_weight', 0.0))

    for epoch in range(1, args.num_epochs + 1):
        warmup_ratio = min(1.0, epoch / noise_warmup) if noise_warmup > 0 else 1.0

        train_m = train_epoch(
            model, train_dataloader, optimizer, normalizer,
            device, epoch, writer, l1_on, l1_w, obs_w, fut_w,
            keypoint_only, keypoint_dim,
            use_obs_head=use_obs,
            use_future_head=use_future,
            use_past_head=use_past, past_loss_weight=past_w,
            past_temporal_weight=past_tw,
            past_temporal_alpha=past_ta,
            past_dct_basis=past_dct_basis_loss,
            past_dct_freq_decay=past_dct_freq_decay,
            past_consistency_weight=cons_w,
            past_consistency_n_steps=cons_n,
            past_abstraction=past_abst,
            spatial_threshold=spat_thr,
            action_noise_scale=noise_scale,
            action_dropout_prob=noise_drop,
            noise_warmup_ratio=warmup_ratio,
            future_mode_cls_weight=mode_cls_w,
            future_entropy_weight=ent_w,
            future_dct_basis=future_dct_basis_loss,
            future_dct_freq_decay=future_dct_freq_decay,
        )
        val_m = validate_epoch(
            model, val_dataloader, normalizer,
            device, epoch, writer, l1_on, l1_w, obs_w, fut_w,
            keypoint_only, keypoint_dim,
            use_obs_head=use_obs,
            use_future_head=use_future,
            use_past_head=use_past, past_loss_weight=past_w,
            past_temporal_weight=past_tw,
            past_temporal_alpha=past_ta,
            past_dct_basis=past_dct_basis_loss,
            past_dct_freq_decay=past_dct_freq_decay,
            past_consistency_weight=cons_w,
            past_consistency_n_steps=cons_n,
            past_abstraction=past_abst,
            spatial_threshold=spat_thr,
            future_mode_cls_weight=mode_cls_w,
            future_entropy_weight=ent_w,
            future_dct_basis=future_dct_basis_loss,
            future_dct_freq_decay=future_dct_freq_decay,
        )

        # -- TensorBoard --
        writer.add_scalar('train/loss_epoch',  train_m['loss'],       epoch)
        writer.add_scalar('train/obs_mse',     train_m['obs_mse'],    epoch)
        writer.add_scalar('train/future_mse',  train_m['future_mse'], epoch)
        writer.add_scalar('train/past_mse',    train_m['past_mse'],   epoch)
        writer.add_scalar('train/consistency',  train_m['consistency_loss'], epoch)
        writer.add_scalar('train/vq_loss',      train_m['vq_loss'],          epoch)
        writer.add_scalar('train/mode_cls_loss', train_m['mode_cls_loss'],   epoch)
        writer.add_scalar('train/entropy_loss',  train_m['entropy_loss'],    epoch)
        writer.add_scalar('val/loss_epoch',    val_m['loss'],         epoch)
        writer.add_scalar('val/obs_mse',       val_m['obs_mse'],      epoch)
        writer.add_scalar('val/future_mse',    val_m['future_mse'],   epoch)
        writer.add_scalar('val/past_mse',      val_m['past_mse'],     epoch)
        writer.add_scalar('val/consistency',   val_m['consistency_loss'],   epoch)
        writer.add_scalar('val/vq_loss',       val_m['vq_loss'],            epoch)
        writer.add_scalar('val/mode_cls_loss', val_m['mode_cls_loss'],      epoch)
        writer.add_scalar('val/entropy_loss',  val_m['entropy_loss'],       epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        if noise_scale > 0 or noise_drop > 0:
            writer.add_scalar('noise/effective_scale', noise_scale * warmup_ratio, epoch)
            writer.add_scalar('noise/effective_dropout', noise_drop * warmup_ratio, epoch)

        if val_m['obs_mse_per_dim'] is not None:
            for d, v in enumerate(val_m['obs_mse_per_dim'].tolist()):
                writer.add_scalar(f'val/obs_mse_dim_{d:02d}', v, epoch)

        # -- W&B --
        if use_wandb:
            wandb.log({
                'epoch':           epoch,
                'train/loss':      train_m['loss'],
                'train/obs_mse':   train_m['obs_mse'],
                'train/future_mse': train_m['future_mse'],
                'train/past_mse':  train_m['past_mse'],
                'train/consistency': train_m['consistency_loss'],
                'train/vq_loss':  train_m['vq_loss'],
                'train/mode_cls_loss': train_m['mode_cls_loss'],
                'train/entropy_loss':  train_m['entropy_loss'],
                'val/loss':        val_m['loss'],
                'val/obs_mse':     val_m['obs_mse'],
                'val/future_mse':  val_m['future_mse'],
                'val/past_mse':    val_m['past_mse'],
                'val/consistency': val_m['consistency_loss'],
                'val/vq_loss':   val_m['vq_loss'],
                'val/mode_cls_loss': val_m['mode_cls_loss'],
                'val/entropy_loss':  val_m['entropy_loss'],
                'lr': optimizer.param_groups[0]['lr'],
                **({'noise/effective_scale': noise_scale * warmup_ratio,
                    'noise/effective_dropout': noise_drop * warmup_ratio}
                   if noise_scale > 0 or noise_drop > 0 else {}),
            })

        print(
            f"\nEpoch {epoch}/{args.num_epochs}  "
            f"| Train  obs={train_m['obs_mse']:.5f}  fut={train_m['future_mse']:.5f}  "
            f"past={train_m['past_mse']:.5f}  cons={train_m['consistency_loss']:.5f}  "
            f"vq={train_m['vq_loss']:.5f}  mcls={train_m['mode_cls_loss']:.5f}  "
            f"ent={train_m['entropy_loss']:.5f}  "
            f"| Val    obs={val_m['obs_mse']:.5f}  fut={val_m['future_mse']:.5f}  "
            f"past={val_m['past_mse']:.5f}  cons={val_m['consistency_loss']:.5f}  "
            f"vq={val_m['vq_loss']:.5f}  mcls={val_m['mode_cls_loss']:.5f}  "
            f"ent={val_m['entropy_loss']:.5f}  "
            f"| LR={optimizer.param_groups[0]['lr']:.2e}"
        )

        scheduler.step(val_m['loss'])

        # -- save best --
        if val_m['loss'] < best_val_loss:
            best_val_loss = val_m['loss']
            best_epoch    = epoch
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss':    best_val_loss,
                'val_obs_mse': val_m['obs_mse'],
                'val_future_mse': val_m['future_mse'],
                'val_past_mse':   val_m['past_mse'],
                'args':        vars(args),
            }, output_dir / 'best_model.pt')
            print(f"  [*] Best model  val_loss={best_val_loss:.5f}  "
                  f"(obs={val_m['obs_mse']:.5f}  fut={val_m['future_mse']:.5f}  "
                  f"past={val_m['past_mse']:.5f})")

        if epoch % 50 == 0:
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss':    val_m['loss'],
                'args':        vars(args),
            }, output_dir / f'checkpoint_epoch_{epoch:04d}.pt')

    # ---- Summary --------------------------------------------------------
    print(f"\n{'='*70}")
    print("Training completed!")
    print(f"Best epoch   : {best_epoch}")
    print(f"Best val loss: {best_val_loss:.5f}")
    print(f"Output dir   : {output_dir}")
    print(f"TensorBoard  :  tensorboard --logdir {output_dir / 'logs'}")

    writer.close()
    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# Hydra workspace wrapper
# ---------------------------------------------------------------------------

from omegaconf import DictConfig, OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace


class TrainObsActionChunkLSTMWorkspace(BaseWorkspace):
    """BaseWorkspace adapter for the lowdim ObsActionChunkLSTM pretraining loop.

    Reads the ``lstm_pretrain`` block of the Hydra config, materializes it as
    an ``argparse.Namespace`` (matching the signature accepted by the legacy
    CLI script), and hands off to ``run_lstm_pretrain``.

    The workspace's hydra-managed output directory is used as ``output_dir``
    so each run writes ``best_model.pt`` / ``checkpoint_epoch_*.pt`` under
    a timestamped subtree alongside DP checkpoints.
    """

    include_keys = ['epoch', 'global_step']

    def __init__(self, cfg: DictConfig, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

    def run(self):
        params = OmegaConf.to_container(self.cfg.lstm_pretrain, resolve=True)
        if not params.get('output_dir'):
            params['output_dir'] = str(self.output_dir)
        params.setdefault('config', None)
        args = argparse.Namespace(**params)
        run_lstm_pretrain(args)


if __name__ == '__main__':
    main()
