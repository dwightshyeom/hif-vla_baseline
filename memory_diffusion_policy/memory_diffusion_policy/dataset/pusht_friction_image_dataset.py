"""
Vanilla-DP image dataset for the Push-T friction task.

Expected zarr layout (same convention as pusht_three_goals_demo_vision.zarr):

    data/
      img:    (N, 96, 96, 3) uint8    — rendered frames
      state:  (N, >=2) float32        — first 2 dims interpreted as agent_pos
      action: (N, 2) float32
    meta/
      episode_ends: (n_episodes,) int64

If ``state`` is wider than 2 (e.g. you additionally store block pose /
memory flags in the same array), only ``state[:, :2]`` is surfaced to the
policy as ``agent_pos`` — matching the image env interface.
"""
from typing import Dict
import copy
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from memory_diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from memory_diffusion_policy.model.common.normalizer import LinearNormalizer
from memory_diffusion_policy.dataset.base_dataset import BaseImageDataset
from memory_diffusion_policy.common.normalize_util import get_image_range_normalizer


class PushTFrictionImageDataset(BaseImageDataset):
    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None):
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

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        state = np.asarray(self.replay_buffer['state'])
        data = {
            'action': self.replay_buffer['action'],
            # Tolerate wider state arrays; first 2 dims are always agent_pos.
            'agent_pos': state[:, :2] if state.ndim == 2 and state.shape[1] > 2 else state,
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        state = sample['state']
        agent_pos = (state[..., :2] if state.ndim >= 2 and state.shape[-1] > 2
                     else state).astype(np.float32)
        image = np.moveaxis(sample['img'], -1, 1) / 255

        return {
            'obs': {
                'image': image,          # T, 3, 96, 96
                'agent_pos': agent_pos,  # T, 2
            },
            'action': sample['action'].astype(np.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)
