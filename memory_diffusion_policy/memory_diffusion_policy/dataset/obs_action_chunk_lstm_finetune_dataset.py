"""
Dataset for **finetunable** LSTM + Diffusion Policy training.

Key difference from the frozen variant
(obs_action_chunk_lstm_chunked_latent_dataset.py):
  - LSTM hidden states are **not** pre-computed.  Instead each sample
    returns the *full episode* obs/action data together with the chunk
    position so the **policy** can run the LSTM live during training
    (with gradients flowing back through the LSTM).

Each training sample provides:
    obs:              (horizon, dp_obs_dim)       chunk observations (for DP)
    action:           (horizon, action_dim)       chunk actions
    episode_obs:      (T_episode, lstm_obs_dim)   full episode observations (for LSTM)
    episode_action:   (T_episode, action_dim)     full episode actions
    chunk_start_idx:  int                         chunk start in episode
    episode_len:      int                         actual episode length

When the LSTM was trained with goal keypoints but the DP does not use
them, ``lstm_include_goal_keypoints=True`` creates a second inner
episode dataset so that ``episode_obs`` has the correct (wider)
dimensionality for the LSTM while ``obs`` stays narrow for the DP.

A custom collate function (``finetune_collate_fn``) pads the variable-
length episode data to the maximum length within each mini-batch.
"""

from typing import Dict, List, Optional

import torch
import numpy as np
from pathlib import Path

from memory_diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from memory_diffusion_policy.dataset.pusht_three_goals_lstm_dataset import (
    PushTThreeGoalsLSTMDatasetWithIndicator,
)


# ======================================================================
# Custom collate function
# ======================================================================

def finetune_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function that pads variable-length episode data.

    Fixed-size tensors (obs, action) are simply stacked.
    Variable-length tensors (episode_obs, episode_action) are
    zero-padded to ``max(episode_len)`` within the mini-batch.
    """
    B = len(batch)

    # ---- Fixed-size chunk data: stack directly --------------------------
    obs = torch.stack([b["obs"] for b in batch])           # (B, H, obs_dim)
    action = torch.stack([b["action"] for b in batch])     # (B, H, act_dim)
    chunk_start_idx = torch.tensor(
        [b["chunk_start_idx"] for b in batch], dtype=torch.long)
    episode_len = torch.tensor(
        [b["episode_len"] for b in batch], dtype=torch.long)

    # ---- Variable-length episode data: pad to max_len -------------------
    max_len = int(episode_len.max().item())
    obs_dim = batch[0]["episode_obs"].shape[-1]
    action_dim = batch[0]["episode_action"].shape[-1]

    episode_obs = torch.zeros(B, max_len, obs_dim,
                              dtype=batch[0]["episode_obs"].dtype)
    episode_action = torch.zeros(B, max_len, action_dim,
                                 dtype=batch[0]["episode_action"].dtype)

    for i, b in enumerate(batch):
        L = b["episode_obs"].shape[0]
        episode_obs[i, :L] = b["episode_obs"]
        episode_action[i, :L] = b["episode_action"]

    return {
        "obs": obs,
        "action": action,
        "episode_obs": episode_obs,
        "episode_action": episode_action,
        "chunk_start_idx": chunk_start_idx,
        "episode_len": episode_len,
    }


# ======================================================================
# Dataset
# ======================================================================

class ObsActionChunkLSTMFinetuneDataset(BaseLowdimDataset):
    """
    Dataset for finetunable LSTM + Diffusion Policy training.

    Loads full episodes and chunks them identically to the frozen
    variant (``ObsActionChunkLSTMChunkedLatentDataset``), but instead
    of pre-computing LSTM hidden states it returns full episode data so
    the policy can run the LSTM live (with gradients).

    When the LSTM was trained with goal keypoints but the DP does not
    need them (``include_goal_keypoints=False``), set
    ``lstm_include_goal_keypoints=True`` to create a second inner
    episode dataset that provides full-dimensional episode obs for the
    LSTM.
    """

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        include_goal_keypoints: bool = False,
        lstm_include_goal_keypoints: Optional[bool] = None,
        # lstm_checkpoint_path kept for API compat
        # (not needed here — LSTM lives in the policy)
        lstm_checkpoint_path: Optional[str] = None,
        device: str = "cuda:0",
    ):
        # If not specified, LSTM uses same setting as DP
        if lstm_include_goal_keypoints is None:
            lstm_include_goal_keypoints = include_goal_keypoints

        self._include_goal_keypoints = include_goal_keypoints
        self._lstm_include_goal_keypoints = lstm_include_goal_keypoints

        # ---- Episode dataset for DP chunks ------------------------------
        self.episode_dataset = PushTThreeGoalsLSTMDatasetWithIndicator(
            zarr_path=zarr_path,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
            include_goal_keypoints=include_goal_keypoints,
        )

        # ---- Episode dataset for LSTM (may differ in obs_dim) -----------
        if lstm_include_goal_keypoints != include_goal_keypoints:
            self.lstm_episode_dataset = PushTThreeGoalsLSTMDatasetWithIndicator(
                zarr_path=zarr_path,
                seed=seed,
                val_ratio=val_ratio,
                max_train_episodes=max_train_episodes,
                include_goal_keypoints=lstm_include_goal_keypoints,
            )
            dp_obs_dim = self.episode_dataset[0]["obs"].shape[-1]
            lstm_obs_dim = self.lstm_episode_dataset[0]["obs"].shape[-1]
            print(
                f"LSTM episode dataset created with "
                f"include_goal_keypoints={lstm_include_goal_keypoints} "
                f"(lstm_obs_dim={lstm_obs_dim}, dp_obs_dim={dp_obs_dim})"
            )
        else:
            self.lstm_episode_dataset = self.episode_dataset

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.device = device
        self._zarr_path = zarr_path
        self._seed = seed
        self._val_ratio = val_ratio
        self._max_train_episodes = max_train_episodes

        # ---- Build chunk index (identical to frozen variant) ------------
        self._build_chunk_index()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_chunk_index(self):
        """Build flat index → (episode_idx, start_idx) mapping."""
        self.chunk_index = []

        for ep_idx in range(len(self.episode_dataset)):
            episode = self.episode_dataset[ep_idx]
            T = episode["obs"].shape[0]

            for start_idx in range(T):
                chunk_start = start_idx - self.pad_before
                chunk_end = chunk_start + self.horizon

                if chunk_end <= T + self.pad_after:
                    self.chunk_index.append((ep_idx, start_idx))

        print(
            f"Built chunk index: {len(self.chunk_index)} chunks "
            f"from {len(self.episode_dataset)} episodes"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def collate_fn(batch):
        """Return the custom collate function (used by the workspace)."""
        return finetune_collate_fn(batch)

    def get_validation_dataset(self):
        """Create validation dataset sharing the same replay buffer."""
        val_ds = ObsActionChunkLSTMFinetuneDataset.__new__(
            ObsActionChunkLSTMFinetuneDataset
        )
        val_ds.episode_dataset = self.episode_dataset.get_validation_dataset()
        val_ds._include_goal_keypoints = self._include_goal_keypoints
        val_ds._lstm_include_goal_keypoints = self._lstm_include_goal_keypoints
        if self.lstm_episode_dataset is self.episode_dataset:
            val_ds.lstm_episode_dataset = val_ds.episode_dataset
        else:
            val_ds.lstm_episode_dataset = self.lstm_episode_dataset.get_validation_dataset()
        val_ds.horizon = self.horizon
        val_ds.pad_before = self.pad_before
        val_ds.pad_after = self.pad_after
        val_ds.device = self.device
        val_ds._zarr_path = self._zarr_path
        val_ds._seed = self._seed
        val_ds._val_ratio = self._val_ratio
        val_ds._max_train_episodes = self._max_train_episodes
        val_ds._build_chunk_index()
        return val_ds

    def get_normalizer(self, mode="limits", **kwargs):
        return self.episode_dataset.get_normalizer(mode=mode, **kwargs)

    def get_all_actions(self):
        return self.episode_dataset.get_all_actions()

    def __len__(self):
        return len(self.chunk_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            obs:             (horizon, dp_obs_dim)   chunk observations (for DP)
            action:          (horizon, action_dim)   chunk actions
            episode_obs:     (T_ep, lstm_obs_dim)    full episode obs (for LSTM)
            episode_action:  (T_ep, action_dim)      full episode actions
            chunk_start_idx: int                     where chunk starts
            episode_len:     int                     actual episode length
        """
        ep_idx, start_idx = self.chunk_index[idx]

        # DP episode data (for chunk obs/action)
        episode = self.episode_dataset[ep_idx]
        obs_full = episode["obs"]             # (T, dp_obs_dim) Tensor
        action_full = episode["action"]       # (T, action_dim) Tensor

        # LSTM episode data (may have wider obs_dim)
        lstm_episode = self.lstm_episode_dataset[ep_idx]
        lstm_obs_full = lstm_episode["obs"]       # (T, lstm_obs_dim)
        lstm_action_full = lstm_episode["action"] # (T, action_dim)

        T = obs_full.shape[0]
        chunk_start = start_idx - self.pad_before
        chunk_end = chunk_start + self.horizon

        # ---- Extract chunk (with boundary padding) ----------------------
        if chunk_start < 0:
            n_pad = -chunk_start
            obs_before = obs_full[0:1].repeat(n_pad, 1)
            act_before = torch.zeros(n_pad, action_full.shape[-1],
                                     dtype=action_full.dtype)
            obs_chunk = torch.cat(
                [obs_before, obs_full[:chunk_end]], dim=0)
            act_chunk = torch.cat(
                [act_before, action_full[:chunk_end]], dim=0)
        else:
            end = min(chunk_end, T)
            obs_chunk = obs_full[chunk_start:end]
            act_chunk = action_full[chunk_start:end]

        if chunk_end > T:
            n_pad = chunk_end - T
            obs_after = obs_full[-1:].repeat(n_pad, 1)
            act_after = torch.zeros(n_pad, action_full.shape[-1],
                                    dtype=action_full.dtype)
            obs_chunk = torch.cat([obs_chunk, obs_after], dim=0)
            act_chunk = torch.cat([act_chunk, act_after], dim=0)

        # Compute the actual episode position corresponding to
        # the chunk start (clamped to valid range).  The policy
        # will use this to index into the LSTM hidden output.
        actual_chunk_start = max(0, chunk_start)

        return {
            "obs": obs_chunk,                                     # (H, dp_obs_dim)
            "action": act_chunk,                                  # (H, act_dim)
            "episode_obs": lstm_obs_full,                         # (T, lstm_obs_dim)
            "episode_action": lstm_action_full,                   # (T, act_dim)
            "chunk_start_idx": actual_chunk_start,                # int
            "episode_len": T,                                     # int
        }
