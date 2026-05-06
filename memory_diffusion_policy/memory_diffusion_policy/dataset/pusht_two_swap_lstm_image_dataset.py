"""
Vision LSTM dataset for the Push-T two-block swap task.

Mirrors PushTThreeGoalsLSTMImageDataset (full-episode batches with
collate_fn_lstm_image) but slices the 8D swap state down to the 2D
agent_pos expected by the image policy interface.
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
    collate_fn_lstm_image,  # re-exported for train script to import
)

__all__ = ['PushTTwoSwapLSTMImageDataset', 'collate_fn_lstm_image']


class PushTTwoSwapLSTMImageDataset(BaseImageDataset):
    """
    Full-episode image dataset for LSTM encoder pretraining on the
    two-block swap Push-T task.

    Zarr arrays used:
        img:    (N, 96, 96, 3) uint8
        state:  (N, 8) float32  (only the first 2 dims — agent_pos — are used)
        action: (N, 2) float32

    __len__ is the number of episodes; __getitem__ returns an (image,
    agent_pos, action) triple for a whole episode. Use the accompanying
    collate_fn_lstm_image to pad variable-length episodes into a batch.
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
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': np.asarray(self.replay_buffer['state'])[:, :2],
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

        image = np.moveaxis(ep['img'], -1, 1).astype(np.float32) / 255.0   # (T, 3, 96, 96)
        agent_pos = ep['state'][:, :2].astype(np.float32)                   # (T, 2)
        action = ep['action'].astype(np.float32)                            # (T, 2)

        if self.action_step_subsample > 1:
            s = self.action_step_subsample
            image = image[s-1::s]
            agent_pos = agent_pos[s-1::s]
            action = action[s-1::s]

        return {
            'image': torch.from_numpy(image),
            'agent_pos': torch.from_numpy(agent_pos),
            'action': torch.from_numpy(action),
        }
