"""
Vision LSTM dataset for the Push-T three-goals swap task.

Mirrors PushTTwoSwapLSTMImageDataset / PushTThreeGoalsLSTMImageDataset:
full-episode batches consumed by ``collate_fn_lstm_image`` (re-exported
here for the trainer's import path), with ``action_step_subsample`` to
align timestep frequency with the eventual DP decision rate.

Compatible with ``train_obs_action_chunk_lstm_image.py`` — pass via:

    --dataset_class memory_diffusion_policy.dataset.pusht_three_goals_swap_lstm_image_dataset.PushTThreeGoalsSwapLSTMImageDataset

Per-sample dict layout:
    image:     (T, 3, 96, 96)  float32, [0, 1]   — raw pixels (normalised in model)
    agent_pos: (T, 2)          float32           — sliced from state[:, :2]
    action:    (T, 2)          float32

Zarr arrays used (matches the three-goals-swap demo schema):
    img:    (N, 96, 96, 3) uint8
    state:  (N, 8) float32       — agent (2) + blue pose (3) + red pose (3)
    action: (N, 2) float32
"""
from typing import Dict
import torch
import numpy as np
import copy

from diffusion_policy.common.replay_buffer import ReplayBuffer
from memory_diffusion_policy.common.sampler import get_val_mask, downsample_mask
from memory_diffusion_policy.model.common.normalizer import LinearNormalizer
from memory_diffusion_policy.dataset.base_dataset import BaseImageDataset
from memory_diffusion_policy.common.normalize_util import get_image_range_normalizer
from memory_diffusion_policy.dataset.pusht_three_goals_lstm_image_dataset import (
    collate_fn_lstm_image,  # re-exported for the trainer to import.
)

__all__ = ['PushTThreeGoalsSwapLSTMImageDataset', 'collate_fn_lstm_image']


class PushTThreeGoalsSwapLSTMImageDataset(BaseImageDataset):
    """
    Full-episode image dataset for LSTM encoder pretraining on the
    three-goals swap Push-T task.

    Identical __init__ kwargs (``zarr_path``, ``seed``, ``val_ratio``,
    ``max_train_episodes``, ``action_step_subsample``) as the existing
    LSTM image datasets so it slots into ``train_obs_action_chunk_lstm_image.py``
    without changes.
    """

    def __init__(self,
                 zarr_path,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 action_step_subsample=1):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.episode_indices = np.where(train_mask)[0]
        self.action_step_subsample = int(action_step_subsample)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.episode_indices = np.where(self.val_mask)[0]
        val_set.train_mask = self.val_mask
        val_set.val_mask = self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        # state is 8D; only the leading 2 dims (agent_pos) are surfaced
        # to the policy. Fit the normalizer on what the policy actually sees.
        state = np.asarray(self.replay_buffer['state'])
        agent_pos = state[:, :2] if state.ndim == 2 and state.shape[1] > 2 else state
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': agent_pos,
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.episode_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_idx = self.episode_indices[idx]
        ep = self.replay_buffer.get_episode(episode_idx, copy=False)

        image = np.moveaxis(ep['img'], -1, 1).astype(np.float32) / 255.0  # (T, 3, 96, 96)
        state = np.asarray(ep['state'], dtype=np.float32)
        # Slice 8D state -> 2D agent_pos. Tolerate 2D-state zarrs too.
        agent_pos = state[:, :2] if state.ndim == 2 and state.shape[1] > 2 else state
        action = ep['action'].astype(np.float32)                          # (T, 2)

        # Take the LAST step of every s-step window so each sample lines
        # up with a DP decision point (obs seen AFTER executing the chunk).
        s = self.action_step_subsample
        if s > 1:
            image = image[s - 1::s]
            agent_pos = agent_pos[s - 1::s]
            action = action[s - 1::s]

        return {
            'image': torch.from_numpy(image),
            'agent_pos': torch.from_numpy(agent_pos),
            'action': torch.from_numpy(action),
        }
