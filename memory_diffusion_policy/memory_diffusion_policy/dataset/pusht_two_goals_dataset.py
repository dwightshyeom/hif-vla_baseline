from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from memory_diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from memory_diffusion_policy.model.common.normalizer import LinearNormalizer
from memory_diffusion_policy.dataset.base_dataset import BaseLowdimDataset


class PushTTwoGoalsLowdimDataset(BaseLowdimDataset):
    """
    Dataset for PushT two-goals task.
    Includes both goal keypoints in the observation.
    """
    
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='keypoint',
            state_key='state',
            action_key='action',
            goal_1_key='goal_1_keypoint',
            goal_2_key='goal_2_keypoint',
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()
        
        # Load replay buffer with all necessary keys
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, state_key, action_key, goal_1_key, goal_2_key])

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
            episode_mask=train_mask
            )
        
        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key
        self.goal_1_key = goal_1_key
        self.goal_2_key = goal_2_key
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
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """
        Constructs observation by concatenating:
        - Current block keypoints (9, 2) -> (18,)
        - Agent position (2,)
        - Goal 1 keypoints (9, 2) -> (18,)
        - Goal 2 keypoints (9, 2) -> (18,)
        Total obs dimension: 18 + 2 + 18 + 18 = 56
        """
        keypoint = sample[self.obs_key]  # (T, 9, 2)
        state = sample[self.state_key]   # (T, 11) = agent_pos(2) + block_pose(3) + goal_1(3) + goal_2(3)
        goal_1_keypoint = sample[self.goal_1_key]  # (T, 9, 2)
        goal_2_keypoint = sample[self.goal_2_key]  # (T, 9, 2)
        
        # Extract agent position from state
        agent_pos = state[:, :2]
        
        # Flatten keypoints
        keypoint_flat = keypoint.reshape(keypoint.shape[0], -1)  # (T, 18)
        goal_1_flat = goal_1_keypoint.reshape(goal_1_keypoint.shape[0], -1)  # (T, 18)
        goal_2_flat = goal_2_keypoint.reshape(goal_2_keypoint.shape[0], -1)  # (T, 18)
        
        # Concatenate to form observation
        obs = np.concatenate([
            keypoint_flat,   # Current block keypoints (18)
            agent_pos,       # Agent position (2)
            goal_1_flat,     # Goal 1 keypoints (18)
            goal_2_flat      # Goal 2 keypoints (18)
        ], axis=-1)  # Total: 56 dimensions

        data = {
            'obs': obs,  # (T, 56)
            'action': sample[self.action_key],  # (T, 2)
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
