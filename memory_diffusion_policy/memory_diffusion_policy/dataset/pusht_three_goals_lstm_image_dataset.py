"""
Vision-based LSTM dataset for PushT three-goals.

Returns full episodes with (image, agent_pos, action) tuples.
Images are raw uint8 HWC — encoding happens in the model.
Agent_pos and action are float32 for normalization.

Supports action_step_subsample to align with DP decision frequency.
"""
from typing import Dict, List
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from memory_diffusion_policy.common.sampler import get_val_mask, downsample_mask
from memory_diffusion_policy.model.common.normalizer import LinearNormalizer
from memory_diffusion_policy.dataset.base_dataset import BaseImageDataset
from memory_diffusion_policy.common.normalize_util import get_image_range_normalizer


class PushTThreeGoalsLSTMImageDataset(BaseImageDataset):
    """
    Vision-based LSTM dataset for PushT three-goals.
    
    Returns full episodes. __len__ = number of episodes.
    Use collate_fn_lstm_image for DataLoader.
    
    Zarr arrays used:
        img:   (N, 96, 96, 3) uint8
        state: (N, 2) float32 — agent position
        action: (N, 2) float32
    """
    
    def __init__(self,
            zarr_path,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            action_step_subsample=1,
            ):
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
        """
        Normalizer aligned with DP:
          action  → limits normalizer
          agent_pos → limits normalizer
          image   → [0,1] range normalizer
        """
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'],  # already (N, 2)
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
        
        # image: (T, 96, 96, 3) uint8 -> (T, 3, 96, 96) float32 [0,1]
        image = np.moveaxis(ep['img'], -1, 1).astype(np.float32) / 255.0
        agent_pos = ep['state'].astype(np.float32)   # (T, 2)
        action = ep['action'].astype(np.float32)      # (T, 2)
        
        # Apply action step subsampling — take the LAST step of each window
        # to match DP decision‐point timing (obs seen after executing the chunk).
        if self.action_step_subsample > 1:
            s = self.action_step_subsample
            image = image[s-1::s]
            agent_pos = agent_pos[s-1::s]
            action = action[s-1::s]
        
        return {
            'image': torch.from_numpy(image),          # (T, 3, 96, 96)
            'agent_pos': torch.from_numpy(agent_pos),  # (T, 2)
            'action': torch.from_numpy(action),        # (T, 2)
        }


def collate_fn_lstm_image(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate variable-length vision episodes into padded batch.
    
    Returns:
        image:     (B, T_max, 3, 96, 96) float32
        agent_pos: (B, T_max, 2)
        action:    (B, T_max, 2)
        mask:      (B, T_max) bool
        lengths:   (B,) long
    """
    max_length = max(s['image'].shape[0] for s in batch)
    B = len(batch)
    
    img_shape = batch[0]['image'].shape[1:]  # (3, H, W)
    agent_dim = batch[0]['agent_pos'].shape[1]
    action_dim = batch[0]['action'].shape[1]
    
    image_pad = torch.zeros(B, max_length, *img_shape, dtype=torch.float32)
    agent_pad = torch.zeros(B, max_length, agent_dim, dtype=torch.float32)
    action_pad = torch.zeros(B, max_length, action_dim, dtype=torch.float32)
    mask = torch.zeros(B, max_length, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)
    
    for i, s in enumerate(batch):
        T = s['image'].shape[0]
        lengths[i] = T
        image_pad[i, :T] = s['image']
        agent_pad[i, :T] = s['agent_pos']
        action_pad[i, :T] = s['action']
        mask[i, :T] = True
    
    return {
        'image': image_pad,
        'agent_pos': agent_pad,
        'action': action_pad,
        'mask': mask,
        'lengths': lengths,
    }
