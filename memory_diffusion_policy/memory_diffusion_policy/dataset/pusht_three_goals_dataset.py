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


class PushTThreeGoalsLowdimDataset(BaseLowdimDataset):
    """
    Dataset for PushT three-goals task.
    Observation includes block keypoints and agent position.
    Optionally includes goal keypoints when goals vary across episodes.
    """
    
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='keypoint',
            state_key='state',
            action_key='action',
            goal_1_keypoint_key='goal_1_keypoint',
            goal_2_keypoint_key='goal_2_keypoint',
            goal_3_keypoint_key='goal_3_keypoint',
            include_goal_keypoints=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()
        
        self.include_goal_keypoints = bool(include_goal_keypoints)
        
        # Load replay buffer with necessary keys
        keys = [obs_key, state_key, action_key]
        if self.include_goal_keypoints:
            keys.extend([goal_1_keypoint_key, goal_2_keypoint_key, goal_3_keypoint_key])
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=keys)

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
        self.goal_1_keypoint_key = goal_1_keypoint_key
        self.goal_2_keypoint_key = goal_2_keypoint_key
        self.goal_3_keypoint_key = goal_3_keypoint_key
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
        - [Optional] Goal 1 keypoints (9, 2) -> (18,)
        - [Optional] Goal 2 keypoints (9, 2) -> (18,)
        - [Optional] Goal 3 keypoints (9, 2) -> (18,)
        
        Total obs dimension: 20 (without goals) or 74 (with goals)
        """
        keypoint = sample[self.obs_key]  # (T, 9, 2)
        state = sample[self.state_key]   # (T, 2) - only agent position
        
        # Extract agent position from state
        agent_pos = state  # Already just agent position
        
        # Flatten keypoints
        T = keypoint.shape[0]
        keypoint_flat = keypoint.reshape(T, -1)  # (T, 18)
        
        # Build obs components
        obs_parts = [
            keypoint_flat,   # Current block keypoints (18)
            agent_pos,       # Agent position (2)
        ]
        
        if self.include_goal_keypoints:
            goal_1_kp = sample[self.goal_1_keypoint_key]  # (T, 9, 2)
            goal_2_kp = sample[self.goal_2_keypoint_key]
            goal_3_kp = sample[self.goal_3_keypoint_key]
            obs_parts.extend([
                goal_1_kp.reshape(T, -1),  # (18,)
                goal_2_kp.reshape(T, -1),  # (18,)
                goal_3_kp.reshape(T, -1),  # (18,)
            ])
        
        # Concatenate to form observation
        obs = np.concatenate(obs_parts, axis=-1)  # 20 or 74 dimensions

        data = {
            'obs': obs,
            'action': sample[self.action_key],  # (T, 2)
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data



class PushTThreeGoalsLowdimDatasetWithIndicator(BaseLowdimDataset):
    """
    Dataset for PushT three-goals task.
    Since all goals are fixed, observation only includes block keypoints and agent position.
    Goal positions are not included in observation as they are constant.
    """
    
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='keypoint',
            state_key='state',
            action_key='action',
            indicator_key = 'goals_reached',
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()
        
        # Load replay buffer with necessary keys
        # Note: state only contains agent position (2D) for three goals env
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, state_key, action_key, indicator_key])

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
        self.indicator_key = indicator_key
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
        
        Total obs dimension: 18 + 2 = 20
        
        Note: Goals are fixed and not included in observation.
              The policy learns to visit all three goals based on the current block position.
        """
        keypoint = sample[self.obs_key]  # (T, 9, 2)
        state = sample[self.state_key]   # (T, 2) - only agent position
        
        # Extract agent position from state
        agent_pos = state  # Already just agent position
        
        # Flatten keypoints
        keypoint_flat = keypoint.reshape(keypoint.shape[0], -1)  # (T, 18)
        
        # Concatenate to form observation
        obs = np.concatenate([
            keypoint_flat,   # Current block keypoints (18)
            agent_pos        # Agent position (2)
        ], axis=-1)  # Total: 20 dimensions

        data = {
            'obs': obs,  # (T, 20)
            'action': sample[self.action_key],  # (T, 2)
            'indicator': sample[self.indicator_key]  # (T,)
        }
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # print("Obs data shape:", data['obs'].shape)
        # print("Action data shape:", data['action'].shape)
        # print("Indicator data shape:", data['indicator'].shape)
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # raise ValueError("ThreeIndicator Debugging: Check data shapes")
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


class PushTThreeGoalsLowdimDatasetWithGoals(BaseLowdimDataset):
    """
    Dataset for PushT three-goals task.
    Since all goals are fixed, observation only includes block keypoints and agent position.
    Goal positions are not included in observation as they are constant.
    """
    
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='keypoint',
            state_key='state',
            action_key='action',
            goal_1_key = 'goal_1_keypoint',
            goal_2_key = 'goal_2_keypoint',
            goal_3_key = 'goal_3_keypoint',
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()
        
        # Load replay buffer with necessary keys
        # Note: state only contains agent position (2D) for three goals env
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, state_key, action_key, goal_1_key, goal_2_key, goal_3_key])

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
        self.goal_3_key = goal_3_key
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
        
        Total obs dimension: 18 + 2 = 20
        
        Note: Goals are fixed and not included in observation.
              The policy learns to visit all three goals based on the current block position.
        """
        keypoint = sample[self.obs_key]  # (T, 9, 2)
        state = sample[self.state_key]   # (T, 2) - only agent position
        
        # Extract agent position from state
        agent_pos = state  # Already just agent position
        
        # Flatten keypoints
        keypoint_flat = keypoint.reshape(keypoint.shape[0], -1)  # (T, 18)

        goal_1 = sample[self.goal_1_key]
        goal_2 = sample[self.goal_2_key]
        goal_3 = sample[self.goal_3_key]
        goal_1_flat = goal_1.reshape(goal_1.shape[0], -1)
        goal_2_flat = goal_2.reshape(goal_2.shape[0], -1)
        goal_3_flat = goal_3.reshape(goal_3.shape[0], -1)
        
        # Concatenate to form observation
        obs = np.concatenate([
            keypoint_flat,   # Current block keypoints (18)
            agent_pos        # Agent position (2)
        ], axis=-1)  # Total: 20 dimensions

        data = {
            'obs': obs,  # (T, 20)
            'action': sample[self.action_key],  # (T, 2)
            'goal_1': goal_1_flat,  
            'goal_2': goal_2_flat,  
            'goal_3': goal_3_flat  
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


class PushTThreeGoalsLowdimDatasetWithIndicatorPercentage(BaseLowdimDataset):
    """
    Dataset for PushT three-goals task.
    Since all goals are fixed, observation only includes block keypoints and agent position.
    Goal positions are not included in observation as they are constant.
    
    This dataset converts the one-hot indicator (e.g., [1,1,0]) to a percentage (e.g., 2/3 = 0.6667).
    """
    
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='keypoint',
            state_key='state',
            action_key='action',
            indicator_key = 'goals_reached',
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()
        
        # Load replay buffer with necessary keys
        # Note: state only contains agent position (2D) for three goals env
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, state_key, action_key, indicator_key])

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
        self.indicator_key = indicator_key
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
        
        Total obs dimension: 18 + 2 = 20
        
        Note: Goals are fixed and not included in observation.
              The policy learns to visit all three goals based on the current block position.
        
        Indicator is converted from one-hot encoding (e.g., [1,1,0]) to percentage (e.g., 2/3 = 0.6667).
        """
        keypoint = sample[self.obs_key]  # (T, 9, 2)
        state = sample[self.state_key]   # (T, 2) - only agent position
        
        # Extract agent position from state
        agent_pos = state  # Already just agent position
        
        # Flatten keypoints
        keypoint_flat = keypoint.reshape(keypoint.shape[0], -1)  # (T, 18)
        
        # Concatenate to form observation
        obs = np.concatenate([
            keypoint_flat,   # Current block keypoints (18)
            agent_pos        # Agent position (2)
        ], axis=-1)  # Total: 20 dimensions

        # Convert indicator from one-hot to percentage
        # If indicator is (T, 3) with values like [1,1,0], sum and divide by 3
        # to get (T,) with values like 0.6667
        indicator = sample[self.indicator_key]  # (T, 3) or (T,)
        if indicator.ndim == 2:
            # Sum the goals achieved and divide by total number of goals (3)
            indicator_percentage = indicator.sum(axis=-1) / 3.0  # (T,)
        else:
            # Already in percentage format
            raise ValueError("Data shape problem")
            indicator_percentage = indicator

        data = {
            'obs': obs,  # (T, 20)
            'action': sample[self.action_key],  # (T, 2)
            'indicator_percentage': indicator_percentage  # (T,) - percentage of goals achieved
        }
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # print("Obs data shape:", data['obs'].shape)
        # print("Action data shape:", data['action'].shape)
        # print("Indicator data shape:", data['indicator'].shape)
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # raise ValueError("ThreeIndicator Debugging: Check data shapes")
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data