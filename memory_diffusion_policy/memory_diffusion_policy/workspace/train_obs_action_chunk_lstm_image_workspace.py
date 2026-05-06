#!/usr/bin/env python3
"""
Training script for Vision-based LSTM with action chunk heads.

Uses the same visual encoder (robomimic-based CNN) as the vanilla Diffusion
Policy (DiffusionUnetHybridImagePolicy) so that the LSTM feature space is
aligned with DP during stitched rollouts.

Input per timestep:  {image (96×96×3), agent_pos (2,), action (2,)}
Output heads:        future_chunk, past_chunk (obs head disabled for vision)

Key alignment with DP:
  - action_step_subsample matches DP's n_action_steps
  - Same crop_shape, GroupNorm replacement, CropRandomizer
  - Same normalizer (limits for action/agent_pos, range for image)

Usage
-----
    NVIDIA_VISIBLE_DEVICES=0 docker compose run --rm dev python train_obs_action_chunk_lstm_image.py --config config/obs_action_chunk_lstm_image_config.yaml
"""
import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import argparse
import importlib
import json
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def _str2bool(v):
    """argparse-friendly bool coercer. The default ``type=bool`` parses any
    non-empty string as True, including ``"false"`` — surprising and the cause
    of historical wandb-on-by-accident bugs in this script."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("true", "yes", "y", "1", "on"):
        return True
    if s in ("false", "no", "n", "0", "off", ""):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


class _LogManager:
    """
    Encapsulates training-loop logging so the main script doesn't juggle
    wandb / TensorBoard / disabled paths everywhere.

    Modes (``log_mode``)
    --------------------
    ``"wandb"``    : TensorBoard scalars on disk **plus** wandb upload.
    ``"local"``    : TensorBoard scalars on disk only — no network traffic.
    ``"disabled"`` : nothing is recorded anywhere (debug / smoke tests).

    ``log_per_step`` : if False, skip the high-frequency per-step writes
                       (4 scalars × N batches × N epochs adds up on long
                       runs). Per-epoch writes are unaffected.
                       Default True for back-compat with the prior
                       tensorboard-everything style.

    Back-compat: legacy ``use_wandb=False`` callers can set ``log_mode=None``
    and pass ``use_wandb`` as a hint; ``main()`` resolves it via
    :func:`_resolve_log_mode` below.
    """

    _VALID = ("wandb", "local", "disabled")

    def __init__(self, log_mode, log_per_step, output_dir,
                 wandb_project=None, wandb_entity=None, wandb_run_name=None,
                 wandb_config=None):
        if log_mode not in self._VALID:
            raise ValueError(
                f"Unknown log_mode={log_mode!r}. "
                f"Choose from: {self._VALID}."
            )
        self.mode = log_mode
        self.per_step = bool(log_per_step)
        self._writer = None
        self._wandb_run = None

        if self.mode in ("wandb", "local"):
            self._writer = SummaryWriter(
                log_dir=str(Path(output_dir) / 'logs'))

        if self.mode == "wandb":
            if not WANDB_AVAILABLE:
                print("[log_mode=wandb] wandb is not installed — falling back "
                      "to log_mode='local' (TensorBoard only).")
                self.mode = "local"
            else:
                self._wandb_run = wandb.init(
                    project=wandb_project, entity=wandb_entity,
                    name=wandb_run_name, config=wandb_config or {})

    # Per-step writes — pass this into train_epoch as ``writer=…``.
    # When per-step logging is off (or mode is disabled), returns None and
    # the helper's existing ``if writer is not None`` guard is a no-op.
    @property
    def step_writer(self):
        if self.mode == "disabled" or not self.per_step:
            return None
        return self._writer

    # Per-epoch scalars (low-frequency).
    def scalar_epoch(self, tag, value, epoch):
        if self._writer is not None:
            self._writer.add_scalar(tag, value, epoch)

    # Per-epoch wandb dump. No-op unless mode == 'wandb'.
    def wandb_log(self, dct):
        if self._wandb_run is None:
            return
        wandb.log(dct)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._wandb_run is not None:
            wandb.finish()
            self._wandb_run = None


def _resolve_log_mode(args):
    """Pick the active log_mode, honoring the explicit ``log_mode`` arg first
    and falling back to legacy ``use_wandb`` if log_mode is unset."""
    explicit = getattr(args, 'log_mode', None)
    if explicit:
        return explicit
    return "wandb" if bool(getattr(args, 'use_wandb', False)) else "local"

from memory_diffusion_policy.dataset.pusht_three_goals_lstm_image_dataset import (
    PushTThreeGoalsLSTMImageDataset,
    collate_fn_lstm_image,
)


def _resolve_dataset_class(dotted_path: Optional[str]):
    """Resolve a 'pkg.mod.ClassName' path to the class object.
    Returns PushTThreeGoalsLSTMImageDataset when dotted_path is falsy so
    that existing configs without dataset_class continue to work.
    Every supported class must expose the same __init__ kwargs and return
    {image, agent_pos, action} per sample to be drop-in compatible with
    collate_fn_lstm_image."""
    if not dotted_path:
        return PushTThreeGoalsLSTMImageDataset
    module_path, cls_name = dotted_path.rsplit('.', 1)
    return getattr(importlib.import_module(module_path), cls_name)
from memory_diffusion_policy.model.obs_action_chunk_lstm_image import ObsActionChunkLSTMImage
from memory_diffusion_policy.model.obs_action_chunk_lstm import compute_obs_action_chunk_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_batch(normalizer, x: torch.Tensor, key: str) -> torch.Tensor:
    """Normalise (B, T, D) via the LP normalizer (flattens/restores T axis)."""
    B, T, D = x.shape
    x_flat = x.reshape(B * T, D).unsqueeze(1)
    return normalizer[key].normalize(x_flat).squeeze(1).reshape(B, T, D).to(x.device)


def build_future_chunk_targets(
    action_norm_np: np.ndarray,
    lengths_np: np.ndarray,
    chunk_H: int,
) -> tuple:
    B, T, D = action_norm_np.shape
    targets = np.zeros((B, T, chunk_H, D), dtype=np.float32)
    mask = np.zeros((B, T), dtype=np.float32)
    step_offsets = np.arange(chunk_H)
    for b in range(B):
        T_b = int(lengths_np[b])
        n = max(0, T_b - chunk_H + 1)
        if n > 0:
            row_idx = np.arange(n)[:, None]
            col_idx = row_idx + step_offsets
            targets[b, :n] = action_norm_np[b, col_idx]
            mask[b, :n] = 1.0
    return targets, mask


def build_past_chunk_targets(
    action_norm_np: np.ndarray,
    lengths_np: np.ndarray,
    past_chunk_H: int,
) -> tuple:
    """past_target[b, t, k] = action_norm_np[b, t-1-k]  if t-1-k >= 0 else 0.
    Slot k=0 corresponds to a_{t-1}, which is also the action input to the
    LSTM at step t (the LSTM sees {obs_{t-1}, a_{t-1}} and reconstructs
    a_{t-1}, a_{t-2}, …, a_{t-past_chunk_H})."""
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
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    normalizer: dict,
    device: str,
    epoch: int,
    writer: SummaryWriter = None,
    l1_on_hidden_state: bool = False,
    l1_weight: float = 0.0,
    future_loss_weight: float = 1.0,
    past_loss_weight: float = 1.0,
    past_temporal_weight: str = 'uniform',
    past_temporal_alpha: float = 2.0,
    past_dct_basis: Optional[torch.Tensor] = None,
    past_dct_freq_decay: float = 0.0,
    past_consistency_weight: float = 0.0,
    past_consistency_n_steps: list = None,
    past_abstraction: str = 'raw',
    action_noise_scale: float = 0.0,
    action_dropout_prob: float = 0.0,
    noise_warmup_ratio: float = 1.0,
    future_mode_cls_weight: float = 0.0,
    future_entropy_weight: float = 0.0,
    future_dct_basis: Optional[torch.Tensor] = None,
    future_dct_freq_decay: float = 0.0,
) -> dict:
    model.train()
    total_loss = 0.0
    total_future_mse = 0.0
    total_past_mse = 0.0
    total_l1 = 0.0
    total_consistency = 0.0
    total_vq = 0.0
    total_mode_cls = 0.0
    total_entropy = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        # Images are already float32 [0,1] from dataset
        image = batch['image'].to(device)           # (B, T, 3, H, W)
        agent_pos = batch['agent_pos'].to(device)   # (B, T, 2)
        actions = batch['action'].to(device)        # (B, T, 2)
        mask = batch['mask'].to(device)             # (B, T)
        lengths = batch['lengths']                  # CPU

        # Normalize agent_pos and action (image is already [0,1])
        agent_pos_norm = _normalize_batch(normalizer, agent_pos, 'agent_pos')
        action_norm = _normalize_batch(normalizer, actions, 'action')

        # Action noise augmentation
        if action_noise_scale > 0.0 or action_dropout_prob > 0.0:
            _scale = action_noise_scale * noise_warmup_ratio
            _drop = action_dropout_prob * noise_warmup_ratio
            action_norm_noisy = action_norm
            if _scale > 0.0:
                action_norm_noisy = action_norm + torch.randn_like(action_norm) * _scale
            action_norm_input = action_norm_noisy
            if _drop > 0.0:
                keep = (torch.rand(action_norm.shape[0], action_norm.shape[1], 1,
                                   device=device) >= _drop).float()
                action_norm_input = action_norm_noisy * keep
        else:
            action_norm_noisy = action_norm
            action_norm_input = action_norm

        # Build chunk targets (clean actions for future, noisy for past)
        action_clean_np = action_norm.cpu().numpy()
        action_noisy_np = action_norm_noisy.cpu().numpy()
        lengths_np = lengths.numpy()

        future_target = future_mask_t = None
        if model.use_future_head:
            future_np, future_mask_np = build_future_chunk_targets(
                action_clean_np, lengths_np, model.chunk_H)
            future_target = torch.from_numpy(future_np).to(device)
            future_mask_t = torch.from_numpy(future_mask_np).to(device)

        past_target = past_mask_t = None
        if model.use_past_head:
            past_np, past_mask_np = build_past_chunk_targets(
                action_noisy_np, lengths_np, model.past_chunk_H)
            past_target = torch.from_numpy(past_np).to(device)
            past_mask_t = torch.from_numpy(past_mask_np).to(device)

        # Forward: image encoder + LSTM
        (obs_pred, future_pred, past_pred, _, lstm_output,
         vq_loss, past_pred_coeffs, future_mode_logits, future_pred_coeffs) = model(
            image, agent_pos_norm, action_norm_input, lengths=lengths)

        loss_dict = compute_obs_action_chunk_loss(
            obs_pred=obs_pred, obs_target=None, obs_mask=mask,
            future_pred=future_pred, future_target=future_target,
            future_mask=future_mask_t,
            past_pred=past_pred, past_target=past_target,
            past_mask=past_mask_t,
            lstm_output=lstm_output,
            l1_on_hidden_state=l1_on_hidden_state,
            l1_weight=l1_weight,
            obs_loss_weight=0.0,
            future_loss_weight=future_loss_weight,
            past_loss_weight=past_loss_weight,
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

        total_loss += loss.item()
        total_future_mse += loss_dict['future_mse_loss'].item()
        total_past_mse += loss_dict['past_mse_loss'].item()
        total_l1 += loss_dict['l1_loss'].item()
        total_consistency += loss_dict['consistency_loss'].item()
        total_vq += loss_dict['vq_loss'].item()
        total_mode_cls += loss_dict['future_mode_cls_loss'].item()
        total_entropy += loss_dict['future_entropy_loss'].item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'fut': f"{loss_dict['future_mse_loss'].item():.4f}",
            'past': f"{loss_dict['past_mse_loss'].item():.4f}",
            'cons': f"{loss_dict['consistency_loss'].item():.4f}",
        })

        if writer is not None:
            gs = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/loss_step', loss.item(), gs)
            writer.add_scalar('train/future_mse_step', loss_dict['future_mse_loss'].item(), gs)
            writer.add_scalar('train/past_mse_step', loss_dict['past_mse_loss'].item(), gs)
            writer.add_scalar('train/consistency_step', loss_dict['consistency_loss'].item(), gs)

    return {
        'loss': total_loss / max(num_batches, 1),
        'future_mse': total_future_mse / max(num_batches, 1),
        'past_mse': total_past_mse / max(num_batches, 1),
        'l1_loss': total_l1 / max(num_batches, 1),
        'consistency_loss': total_consistency / max(num_batches, 1),
        'vq_loss': total_vq / max(num_batches, 1),
        'mode_cls_loss': total_mode_cls / max(num_batches, 1),
        'entropy_loss': total_entropy / max(num_batches, 1),
    }


# ---------------------------------------------------------------------------
# One validation epoch
# ---------------------------------------------------------------------------

def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    normalizer: dict,
    device: str,
    epoch: int,
    writer: SummaryWriter = None,
    l1_on_hidden_state: bool = False,
    l1_weight: float = 0.0,
    future_loss_weight: float = 1.0,
    past_loss_weight: float = 1.0,
    past_temporal_weight: str = 'uniform',
    past_temporal_alpha: float = 2.0,
    past_dct_basis: Optional[torch.Tensor] = None,
    past_dct_freq_decay: float = 0.0,
    past_consistency_weight: float = 0.0,
    past_consistency_n_steps: list = None,
    past_abstraction: str = 'raw',
    future_mode_cls_weight: float = 0.0,
    future_entropy_weight: float = 0.0,
    future_dct_basis: Optional[torch.Tensor] = None,
    future_dct_freq_decay: float = 0.0,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_future_mse = 0.0
    total_past_mse = 0.0
    total_l1 = 0.0
    total_consistency = 0.0
    total_vq = 0.0
    total_mode_cls = 0.0
    total_entropy = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            image = batch['image'].to(device)
            agent_pos = batch['agent_pos'].to(device)
            actions = batch['action'].to(device)
            mask = batch['mask'].to(device)
            lengths = batch['lengths']

            agent_pos_norm = _normalize_batch(normalizer, agent_pos, 'agent_pos')
            action_norm = _normalize_batch(normalizer, actions, 'action')

            action_norm_np = action_norm.cpu().numpy()
            lengths_np = lengths.numpy()

            future_target = future_mask_t = None
            if model.use_future_head:
                future_np, future_mask_np = build_future_chunk_targets(
                    action_norm_np, lengths_np, model.chunk_H)
                future_target = torch.from_numpy(future_np).to(device)
                future_mask_t = torch.from_numpy(future_mask_np).to(device)

            past_target = past_mask_t = None
            if model.use_past_head:
                past_np, past_mask_np = build_past_chunk_targets(
                    action_norm_np, lengths_np, model.past_chunk_H)
                past_target = torch.from_numpy(past_np).to(device)
                past_mask_t = torch.from_numpy(past_mask_np).to(device)

            (obs_pred, future_pred, past_pred, _, lstm_output,
             vq_loss, past_pred_coeffs, future_mode_logits, future_pred_coeffs) = model(
                image, agent_pos_norm, action_norm, lengths=lengths)

            loss_dict = compute_obs_action_chunk_loss(
                obs_pred=obs_pred, obs_target=None, obs_mask=mask,
                future_pred=future_pred, future_target=future_target,
                future_mask=future_mask_t,
                past_pred=past_pred, past_target=past_target,
                past_mask=past_mask_t,
                lstm_output=lstm_output,
                l1_on_hidden_state=l1_on_hidden_state,
                l1_weight=l1_weight,
                obs_loss_weight=0.0,
                future_loss_weight=future_loss_weight,
                past_loss_weight=past_loss_weight,
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

            total_loss += loss_dict['loss'].item()
            total_future_mse += loss_dict['future_mse_loss'].item()
            total_past_mse += loss_dict['past_mse_loss'].item()
            total_l1 += loss_dict['l1_loss'].item()
            total_consistency += loss_dict['consistency_loss'].item()
            total_vq += loss_dict['vq_loss'].item()
            total_mode_cls += loss_dict['future_mode_cls_loss'].item()
            total_entropy += loss_dict['future_entropy_loss'].item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{loss_dict['loss'].item():.4f}",
                'fut': f"{loss_dict['future_mse_loss'].item():.4f}",
            })

    return {
        'loss': total_loss / max(num_batches, 1),
        'future_mse': total_future_mse / max(num_batches, 1),
        'past_mse': total_past_mse / max(num_batches, 1),
        'l1_loss': total_l1 / max(num_batches, 1),
        'consistency_loss': total_consistency / max(num_batches, 1),
        'vq_loss': total_vq / max(num_batches, 1),
        'mode_cls_loss': total_mode_cls / max(num_batches, 1),
        'entropy_loss': total_entropy / max(num_batches, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train Vision LSTM with action chunk heads')

    # Config
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--dataset_class', type=str, default=None,
                        help='Dotted path to the LSTM image dataset class. '
                             'Defaults to PushTThreeGoalsLSTMImageDataset. '
                             'For the two-swap task use '
                             'memory_diffusion_policy.dataset.pusht_two_swap_lstm_image_dataset.'
                             'PushTTwoSwapLSTMImageDataset.')

    # Data
    parser.add_argument('--zarr_path', type=str,
                        default='data/pusht_three_goals_demo_vision.zarr')
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--action_step_subsample', type=int, default=1,
                        help='Subsample episodes at this rate to match DP n_action_steps')

    # Model
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--crop_shape', type=int, nargs=2, default=[76, 76])
    parser.add_argument('--num_kp', type=int, default=32,
                        help='Number of SpatialSoftmax keypoints (image feature dim = num_kp * 2)')
    parser.add_argument('--obs_encoder_group_norm', type=bool, default=True)
    parser.add_argument('--eval_fixed_crop', type=bool, default=True)
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze visual encoder (train only LSTM)')

    # Heads
    parser.add_argument('--use_future_head', type=bool, default=True)
    parser.add_argument('--chunk_H', type=int, default=16)
    parser.add_argument('--num_future_modes', type=int, default=1)
    parser.add_argument('--future_abstraction', type=str, default='raw',
                        choices=['raw', 'dct'])
    parser.add_argument('--future_n_bases', type=int, default=32)
    parser.add_argument('--future_dct_freq_decay', type=float, default=0.0)
    parser.add_argument('--future_mode_cls_weight', type=float, default=0.1)
    parser.add_argument('--future_entropy_weight', type=float, default=0.01)
    parser.add_argument('--use_past_head', type=bool, default=True)
    parser.add_argument('--past_chunk_H', type=int, default=None)
    parser.add_argument('--past_abstraction', type=str, default='raw',
                        choices=['raw', 'dct', 'segment_pool', 'dft', 'dwt'])
    parser.add_argument('--past_n_bases', type=int, default=32)
    parser.add_argument('--past_dct_freq_decay', type=float, default=0.0)
    parser.add_argument('--past_consistency_weight', type=float, default=0.0)
    parser.add_argument('--past_consistency_n_steps', type=int, nargs='+', default=[1],
                        help=('List of N values for multi-step consistency loss. '
                              'E.g. --past_consistency_n_steps 1 2 4 8.'))
    parser.add_argument('--past_temporal_weight', type=str, default='uniform')
    parser.add_argument('--past_temporal_alpha', type=float, default=2.0)

    # Loss
    parser.add_argument('--future_loss_weight', type=float, default=1.0)
    parser.add_argument('--past_loss_weight', type=float, default=1.0)
    parser.add_argument('--l1_on_hidden_state', type=bool, default=False)
    parser.add_argument('--l1_weight', type=float, default=0.0)

    # VQ
    parser.add_argument('--use_vq', type=bool, default=False)
    parser.add_argument('--vq_n_codes', type=int, default=512)
    parser.add_argument('--vq_commitment_weight', type=float, default=0.25)

    # Noise augmentation
    parser.add_argument('--action_noise_scale', type=float, default=0.0)
    parser.add_argument('--action_dropout_prob', type=float, default=0.0)
    parser.add_argument('--action_noise_warmup', type=int, default=0)

    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)

    # Output
    parser.add_argument('--output_dir', type=str,
                        default='./outputs/obs_action_chunk_lstm_image')

    # Logging — see _LogManager for the contract.
    parser.add_argument('--log_mode', type=str, default=None,
                        choices=['wandb', 'local', 'disabled'],
                        help="'wandb' (TB+wandb) | 'local' (TB only) | "
                             "'disabled' (none). When unset, falls back to "
                             "the legacy --use_wandb flag.")
    parser.add_argument('--log_per_step', type=_str2bool, default=True,
                        help="Write per-step (per-batch) TB scalars. "
                             "Set False for less disk usage on long runs.")
    # Legacy flag — still recognized for back-compat. Coerced via str2bool
    # so YAML strings like 'false' do the right thing.
    parser.add_argument('--use_wandb', type=_str2bool, default=False)
    parser.add_argument('--wandb_project', type=str, default='obs_action_chunk_lstm_image')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)

    args = parser.parse_args()

    # ---- Load YAML config -----------------------------------------------
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    run_lstm_pretrain_image(args)


def run_lstm_pretrain_image(args):
    """Run image-LSTM pretraining given a populated argparse.Namespace.

    Split from ``main`` so the hydra workspace
    ``memory_diffusion_policy.workspace.train_obs_action_chunk_lstm_image_workspace``
    can drive the same training loop without re-parsing CLI args.
    """
    # ---- Reproducibility ------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device if torch.cuda.is_available() else 'cpu'
    past_chunk_H = args.past_chunk_H if args.past_chunk_H is not None else args.chunk_H

    # Shape meta (must match DP task config exactly)
    shape_meta = {
        'obs': {
            'image': {'shape': [3, 96, 96], 'type': 'rgb'},
            'agent_pos': {'shape': [2], 'type': 'low_dim'},
        },
        'action': {'shape': [2]},
    }

    crop_shape = tuple(args.crop_shape) if isinstance(args.crop_shape, list) else args.crop_shape

    print(f"\n{'='*70}")
    print("Training Vision LSTM + Action Chunk Predictor")
    print(f"{'='*70}")
    print(f"Device               : {device}")
    print(f"Dataset              : {args.zarr_path}")
    print(f"action_step_subsample: {args.action_step_subsample}")
    print(f"Output               : {args.output_dir}")
    print(f"Hidden size          : {args.hidden_size}  |  Num layers: {args.num_layers}")
    print(f"chunk_H              : {args.chunk_H}  |  past_chunk_H: {past_chunk_H}")
    print(f"crop_shape           : {crop_shape}")
    print(f"num_kp               : {args.num_kp}")
    print(f"freeze_encoder       : {getattr(args, 'freeze_encoder', False)}")

    # ---- Output directory -----------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(vars(args), f)

    # ---- Datasets -------------------------------------------------------
    DatasetCls = _resolve_dataset_class(getattr(args, 'dataset_class', None))
    print(f"\nLoading dataset... ({DatasetCls.__module__}.{DatasetCls.__name__})")
    train_dataset = DatasetCls(
        zarr_path=args.zarr_path,
        val_ratio=args.val_ratio,
        seed=args.seed,
        action_step_subsample=args.action_step_subsample,
    )
    val_dataset = train_dataset.get_validation_dataset()
    print(f"Train episodes: {len(train_dataset)}  |  Val episodes: {len(val_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn_lstm_image,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_lstm_image,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    # ---- Normalizer (aligned with DP) -----------------------------------
    print("Computing normalizer...")
    normalizer = train_dataset.get_normalizer()
    torch.save(normalizer, output_dir / 'normalizer.pt')

    # ---- Model ----------------------------------------------------------
    print(f"\nCreating model...")
    model = ObsActionChunkLSTMImage(
        shape_meta=shape_meta,
        action_dim=2,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        chunk_H=args.chunk_H,
        use_obs_head=False,    # no obs head for vision
        use_future_head=bool(args.use_future_head),
        num_future_modes=int(getattr(args, 'num_future_modes', 1)),
        future_abstraction=getattr(args, 'future_abstraction', 'raw'),
        future_n_bases=int(getattr(args, 'future_n_bases', 32)),
        use_past_head=bool(args.use_past_head),
        past_chunk_H=past_chunk_H,
        past_abstraction=getattr(args, 'past_abstraction', 'raw'),
        past_n_bases=int(getattr(args, 'past_n_bases', 32)),
        use_vq=bool(getattr(args, 'use_vq', False)),
        vq_n_codes=int(getattr(args, 'vq_n_codes', 512)),
        vq_commitment_weight=float(getattr(args, 'vq_commitment_weight', 0.25)),
        crop_shape=crop_shape,
        obs_encoder_group_norm=bool(args.obs_encoder_group_norm),
        eval_fixed_crop=bool(args.eval_fixed_crop),
        freeze_encoder=bool(getattr(args, 'freeze_encoder', False)),
        num_kp=args.num_kp,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {n_params:,}")
    print(f"Trainable parameters : {n_trainable:,}")
    print(f"Visual encoder dim   : {model.obs_feature_dim}")
    print(f"LSTM input dim       : {model.lstm.input_dim}")

    # ---- Optimizer + scheduler ------------------------------------------
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    # ---- Logging --------------------------------------------------------
    log_mode = _resolve_log_mode(args)
    log_per_step = bool(getattr(args, 'log_per_step', True))
    log_manager = _LogManager(
        log_mode=log_mode,
        log_per_step=log_per_step,
        output_dir=output_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_config=vars(args),
    )
    print(f"Logging mode         : {log_manager.mode}  "
          f"(per_step={log_manager.per_step})")

    # ---- Basis tensors for DCT abstraction ------------------------------
    past_dct_basis_loss = None
    past_dct_freq_decay = float(getattr(args, 'past_dct_freq_decay', 0.0))
    if getattr(args, 'past_abstraction', 'raw') in ('dct', 'segment_pool', 'dft', 'dwt'):
        if hasattr(model.lstm, '_past_dct_basis'):
            past_dct_basis_loss = model.lstm._past_dct_basis

    future_dct_basis_loss = None
    future_dct_freq_decay = float(getattr(args, 'future_dct_freq_decay', 0.0))
    if getattr(args, 'future_abstraction', 'raw') == 'dct':
        if hasattr(model.lstm, '_future_dct_basis'):
            future_dct_basis_loss = model.lstm._future_dct_basis

    noise_scale = float(getattr(args, 'action_noise_scale', 0.0))
    noise_drop = float(getattr(args, 'action_dropout_prob', 0.0))
    noise_warmup = int(getattr(args, 'action_noise_warmup', 0))
    mode_cls_w = float(getattr(args, 'future_mode_cls_weight', 0.0))
    ent_w = float(getattr(args, 'future_entropy_weight', 0.0))
    past_abst = getattr(args, 'past_abstraction', 'raw')

    # ---- Training loop --------------------------------------------------
    print(f"\n{'='*70}")
    print("Training...")
    print(f"{'='*70}\n")

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, args.num_epochs + 1):
        warmup_ratio = min(1.0, epoch / noise_warmup) if noise_warmup > 0 else 1.0

        train_m = train_epoch(
            model, train_dataloader, optimizer, normalizer,
            device, epoch, log_manager.step_writer,
            l1_on_hidden_state=bool(args.l1_on_hidden_state),
            l1_weight=float(args.l1_weight),
            future_loss_weight=float(args.future_loss_weight),
            past_loss_weight=float(args.past_loss_weight),
            past_temporal_weight=getattr(args, 'past_temporal_weight', 'uniform'),
            past_temporal_alpha=float(getattr(args, 'past_temporal_alpha', 2.0)),
            past_dct_basis=past_dct_basis_loss,
            past_dct_freq_decay=past_dct_freq_decay,
            past_consistency_weight=float(getattr(args, 'past_consistency_weight', 0.0)),
            past_consistency_n_steps=list(getattr(args, 'past_consistency_n_steps', [1]) or [1]),
            past_abstraction=past_abst,
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
            device, epoch, log_manager.step_writer,
            l1_on_hidden_state=bool(args.l1_on_hidden_state),
            l1_weight=float(args.l1_weight),
            future_loss_weight=float(args.future_loss_weight),
            past_loss_weight=float(args.past_loss_weight),
            past_temporal_weight=getattr(args, 'past_temporal_weight', 'uniform'),
            past_temporal_alpha=float(getattr(args, 'past_temporal_alpha', 2.0)),
            past_dct_basis=past_dct_basis_loss,
            past_dct_freq_decay=past_dct_freq_decay,
            past_consistency_weight=float(getattr(args, 'past_consistency_weight', 0.0)),
            past_consistency_n_steps=list(getattr(args, 'past_consistency_n_steps', [1]) or [1]),
            past_abstraction=past_abst,
            future_mode_cls_weight=mode_cls_w,
            future_entropy_weight=ent_w,
            future_dct_basis=future_dct_basis_loss,
            future_dct_freq_decay=future_dct_freq_decay,
        )

        # -- TensorBoard (no-op when log_mode='disabled') --
        log_manager.scalar_epoch('train/loss_epoch', train_m['loss'], epoch)
        log_manager.scalar_epoch('train/future_mse', train_m['future_mse'], epoch)
        log_manager.scalar_epoch('train/past_mse', train_m['past_mse'], epoch)
        log_manager.scalar_epoch('train/consistency', train_m['consistency_loss'], epoch)
        log_manager.scalar_epoch('train/l1_loss', train_m['l1_loss'], epoch)
        log_manager.scalar_epoch('train/vq_loss', train_m['vq_loss'], epoch)
        log_manager.scalar_epoch('train/mode_cls_loss', train_m['mode_cls_loss'], epoch)
        log_manager.scalar_epoch('train/entropy_loss', train_m['entropy_loss'], epoch)
        log_manager.scalar_epoch('val/loss_epoch', val_m['loss'], epoch)
        log_manager.scalar_epoch('val/future_mse', val_m['future_mse'], epoch)
        log_manager.scalar_epoch('val/past_mse', val_m['past_mse'], epoch)
        log_manager.scalar_epoch('val/consistency', val_m['consistency_loss'], epoch)
        log_manager.scalar_epoch('val/l1_loss', val_m['l1_loss'], epoch)
        log_manager.scalar_epoch('val/vq_loss', val_m['vq_loss'], epoch)
        log_manager.scalar_epoch('val/mode_cls_loss', val_m['mode_cls_loss'], epoch)
        log_manager.scalar_epoch('val/entropy_loss', val_m['entropy_loss'], epoch)
        log_manager.scalar_epoch('lr', optimizer.param_groups[0]['lr'], epoch)

        # -- W&B (no-op unless log_mode='wandb') --
        log_manager.wandb_log({
            'epoch': epoch,
            'train/loss': train_m['loss'],
            'train/future_mse': train_m['future_mse'],
            'train/past_mse': train_m['past_mse'],
            'train/consistency': train_m['consistency_loss'],
            'train/l1_loss': train_m['l1_loss'],
            'train/vq_loss': train_m['vq_loss'],
            'train/mode_cls_loss': train_m['mode_cls_loss'],
            'train/entropy_loss': train_m['entropy_loss'],
            'val/loss': val_m['loss'],
            'val/future_mse': val_m['future_mse'],
            'val/past_mse': val_m['past_mse'],
            'val/consistency': val_m['consistency_loss'],
            'val/l1_loss': val_m['l1_loss'],
            'val/vq_loss': val_m['vq_loss'],
            'val/mode_cls_loss': val_m['mode_cls_loss'],
            'val/entropy_loss': val_m['entropy_loss'],
            'lr': optimizer.param_groups[0]['lr'],
        })

        print(
            f"\nEpoch {epoch}/{args.num_epochs}  "
            f"| Train  fut={train_m['future_mse']:.5f}  past={train_m['past_mse']:.5f}  "
            f"cons={train_m['consistency_loss']:.5f}  "
            f"| Val    fut={val_m['future_mse']:.5f}  past={val_m['past_mse']:.5f}  "
            f"cons={val_m['consistency_loss']:.5f}  "
            f"| LR={optimizer.param_groups[0]['lr']:.2e}"
        )

        scheduler.step(val_m['loss'])

        # -- Save best --
        if val_m['loss'] < best_val_loss:
            best_val_loss = val_m['loss']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_future_mse': val_m['future_mse'],
                'val_past_mse': val_m['past_mse'],
                'args': vars(args),
                'shape_meta': shape_meta,
            }, output_dir / 'best_model.pt')
            print(f"  [*] Best model  val_loss={best_val_loss:.5f}")

        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_m['loss'],
                'args': vars(args),
                'shape_meta': shape_meta,
            }, output_dir / f'checkpoint_epoch_{epoch:04d}.pt')

    print(f"\n{'='*70}")
    print(f"Training completed!  Best val_loss={best_val_loss:.5f} @ epoch {best_epoch}")
    print(f"{'='*70}")

    log_manager.close()


# ---------------------------------------------------------------------------
# Hydra workspace wrapper
# ---------------------------------------------------------------------------

from omegaconf import DictConfig, OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace


class TrainObsActionChunkLSTMImageWorkspace(BaseWorkspace):
    """BaseWorkspace adapter for the image ObsActionChunkLSTM pretraining loop.

    Mirror of TrainObsActionChunkLSTMWorkspace, but for the vision-conditioned
    LSTM pretraining (uses the same robomimic CNN encoder as the hybrid DP
    image policy so the LSTM feature space is aligned with DP at rollout).
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
        run_lstm_pretrain_image(args)


if __name__ == '__main__':
    main()
