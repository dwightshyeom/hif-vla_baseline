from typing import Dict, List
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from memory_diffusion_policy.common.sampler import get_val_mask, downsample_mask
from memory_diffusion_policy.model.common.normalizer import LinearNormalizer
from memory_diffusion_policy.dataset.base_dataset import BaseLowdimDataset


def collate_fn_lstm(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for LSTM dataset that handles variable-length histories.
    
    Each sample has:
    - 'obs': (T_history_i, obs_dim) - variable length history
    - 'action': (horizon, action_dim) - fixed length (padded)
    - 'mask': (T_history_i,) - history validity (all ones)
    - 'action_mask': (horizon,) - action validity
    
    We need to pad histories to the same length within the batch.
    
    Args:
        batch: List of samples with variable-length histories
    
    Returns:
        Dictionary containing:
            - 'obs': (batch, T_max, obs_dim) padded histories
            - 'action': (batch, horizon, action_dim) actions
            - 'mask': (batch, T_max) history validity mask
            - 'action_mask': (batch, horizon) action validity mask
    """
    # Find max history length in batch
    max_history_len = max(sample['obs'].shape[0] for sample in batch)
    batch_size = len(batch)
    obs_dim = batch[0]['obs'].shape[1]
    horizon = batch[0]['action'].shape[0]
    action_dim = batch[0]['action'].shape[1]
    
    # Initialize padded tensors
    obs_padded = torch.zeros(batch_size, max_history_len, obs_dim, dtype=batch[0]['obs'].dtype)
    mask_padded = torch.zeros(batch_size, max_history_len, dtype=torch.bool)
    action_batch = torch.zeros(batch_size, horizon, action_dim, dtype=batch[0]['action'].dtype)
    action_mask_batch = torch.zeros(batch_size, horizon, dtype=torch.bool)
    
    # Fill in actual data
    for i, sample in enumerate(batch):
        history_len = sample['obs'].shape[0]
        obs_padded[i, :history_len] = sample['obs']
        mask_padded[i, :history_len] = sample['mask']
        action_batch[i] = sample['action']
        action_mask_batch[i] = sample['action_mask']
    
    return {
        'obs': obs_padded,
        'action': action_batch,
        'mask': mask_padded,
        'action_mask': action_mask_batch,
    }


class PushTThreeGoalsLSTMDataset(BaseLowdimDataset):
    """
    Dataset for training LSTM with variable memory on PushT three-goals task.
    
    Key differences from standard dataset:
    - __len__ returns total number of valid timesteps across all episodes
    - __getitem__(i) returns a single timestep sample with full history up to that point
    - Each timestep becomes a separate training sample
    - Much more data: ~200 samples per episode instead of 32
    
    Usage:
        dataset = PushTThreeGoalsLSTMDataset(zarr_path, max_memory_length=None)
        dataloader = DataLoader(
            dataset, 
            batch_size=64,  # Can use smaller batch size with more samples
            shuffle=True,
            collate_fn=collate_fn_lstm
        )
    """
    
    def __init__(self, 
            zarr_path, 
            obs_key='keypoint',
            state_key='state',
            action_key='action',
            horizon=16,  # Action prediction horizon
            max_memory_length=None,  # Max history length (None = infinite)
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        """
        Initialize LSTM dataset where each timestep is a separate sample.
        
        Args:
            zarr_path: Path to zarr dataset
            obs_key: Key for observation data (keypoints)
            state_key: Key for state data (agent position)
            action_key: Key for action data
            horizon: Number of future actions to predict
            max_memory_length: Maximum history length (None for infinite)
            seed: Random seed for train/val split
            val_ratio: Ratio of episodes to use for validation
            max_train_episodes: Maximum number of training episodes to use
        """
        super().__init__()
        
        # Load replay buffer with necessary keys
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, state_key, action_key])

        # Create train/val split at episode level
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key
        self.horizon = horizon
        self.max_memory_length = max_memory_length
        self.train_mask = train_mask
        self.val_mask = val_mask
        
        # Build index: map from flat index to (episode_idx, timestep)
        self.episode_indices = np.where(train_mask)[0]
        self._build_timestep_index()

    def _build_timestep_index(self):
        """
        Build index mapping from flat sample index to (episode_idx, timestep).
        Each timestep in each episode becomes a separate sample.
        """
        self.timestep_index = []  # List of (episode_idx, timestep) tuples
        
        for episode_idx in self.episode_indices:
            episode_length = self.replay_buffer.episode_lengths[episode_idx]
            # Each timestep t can predict actions [t:t+horizon]
            # We include all timesteps (even if action chunk is truncated)
            for t in range(episode_length):
                self.timestep_index.append((episode_idx, t))
        
        print(f"Built timestep index: {len(self.timestep_index)} samples from {len(self.episode_indices)} episodes")

    def get_validation_dataset(self):
        """Create validation dataset with same replay buffer but different episodes."""
        val_set = copy.copy(self)
        val_set.episode_indices = np.where(self.val_mask)[0]
        val_set.train_mask = self.val_mask
        val_set.val_mask = self.train_mask
        val_set._build_timestep_index()
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        """
        Compute normalizer from all data in the replay buffer.
        This works with variable-length episodes.
        """
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        """Return all actions in the replay buffer."""
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        """Return total number of timestep samples."""
        return len(self.timestep_index)

    def _sample_to_data(self, sample):
        """
        Convert sample to observation and action data.
        
        Constructs observation by concatenating:
        - Current block keypoints (9, 2) -> (18,)
        - Agent position (2,)
        Total obs dimension: 20
        
        Args:
            sample: Dictionary with keys from replay buffer
        
        Returns:
            Dictionary with 'obs' and 'action' keys
        """
        keypoint = sample[self.obs_key]  # (T, 9, 2)
        state = sample[self.state_key]   # (T, 2) - only agent position
        action = sample[self.action_key]  # (T, 2)
        
        # Extract agent position from state
        agent_pos = state  # Already just agent position
        
        # Flatten keypoints
        keypoint_flat = keypoint.reshape(keypoint.shape[0], -1)  # (T, 18)
        
        # Concatenate to form observation
        obs = np.concatenate([
            keypoint_flat,   # Current block keypoints (18)
            agent_pos,       # Agent position (2)
        ], axis=-1)  # Total: 20 dimensions

        data = {
            'obs': obs,  # (T, 20)
            'action': action,  # (T, 2)
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single timestep sample with its full history.
        
        Args:
            idx: Flat index into timestep_index
        
        Returns:
            Dictionary containing:
                - 'obs': (T_history, 20) tensor - history up to current timestep
                - 'action': (horizon, 2) tensor - future actions (padded if needed)
                - 'mask': (T_history,) bool tensor - all ones (valid history)
                - 'action_mask': (horizon,) bool tensor - which actions are valid
            where T_history depends on max_memory_length
        """
        episode_idx, timestep = self.timestep_index[idx]
        
        # Get full episode data from replay buffer
        episode_sample = self.replay_buffer.get_episode(episode_idx, copy=False)
        
        # Convert to observation and action
        data = self._sample_to_data(episode_sample)
        obs_full = data['obs']  # (T_episode, 20)
        action_full = data['action']  # (T_episode, 2)
        
        T_episode = obs_full.shape[0]
        
        # Get history up to current timestep
        if self.max_memory_length is not None and self.max_memory_length > 0:
            # Limited memory: take last max_memory_length observations up to t
            start_idx = max(0, timestep - self.max_memory_length + 1)
            obs_history = obs_full[start_idx:timestep+1]  # (min(max_memory_length, t+1), 20)
        else:
            # Infinite memory: take all observations up to t
            obs_history = obs_full[:timestep+1]  # (t+1, 20)
        
        # Get future actions [t:t+horizon]
        action_end = min(timestep + self.horizon, T_episode)
        action_chunk = action_full[timestep:action_end]  # (actual_horizon, 2)
        actual_horizon = action_chunk.shape[0]
        
        # Pad action chunk if needed
        if actual_horizon < self.horizon:
            padding = np.zeros((self.horizon - actual_horizon, 2), dtype=action_chunk.dtype)
            action_chunk = np.concatenate([action_chunk, padding], axis=0)
        
        # Create masks
        history_mask = np.ones(obs_history.shape[0], dtype=bool)  # All history is valid
        action_mask = np.zeros(self.horizon, dtype=bool)
        action_mask[:actual_horizon] = True  # Mark which actions are valid
        
        # Convert to torch tensors
        torch_data = {
            'obs': torch.from_numpy(obs_history),  # (T_history, 20)
            'action': torch.from_numpy(action_chunk),  # (horizon, 2)
            'mask': torch.from_numpy(history_mask),  # (T_history,)
            'action_mask': torch.from_numpy(action_mask),  # (horizon,)
        }
        
        return torch_data


class PushTThreeGoalsLSTMDatasetWithIndicator(BaseLowdimDataset):
    """
    Dataset for training LSTM with infinite memory on PushT three-goals task.
    Includes goal completion indicator.
    
    Key differences from standard dataset:
    - __len__ returns number of episodes, not timesteps
    - __getitem__(i) returns entire episode/trajectory
    - No SequenceSampler (no sliding window)
    - Variable-length trajectories per episode
    - Includes goal completion indicator
    - Use custom collate_fn_lstm_with_indicator for DataLoader
    
    Usage:
        dataset = PushTThreeGoalsLSTMDatasetWithIndicator(zarr_path)
        dataloader = DataLoader(
            dataset, 
            batch_size=16, 
            shuffle=True,
            collate_fn=collate_fn_lstm_with_indicator
        )
    """
    
    def __init__(self, 
            zarr_path, 
            obs_key='keypoint',
            state_key='state',
            action_key='action',
            indicator_key='goals_reached',
            goal_1_keypoint_key='goal_1_keypoint',
            goal_2_keypoint_key='goal_2_keypoint',
            goal_3_keypoint_key='goal_3_keypoint',
            include_goal_keypoints=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            action_step_subsample=1
            ):
        """
        Initialize LSTM dataset for full trajectory training with indicator.
        
        Args:
            zarr_path: Path to zarr dataset
            obs_key: Key for observation data (keypoints)
            state_key: Key for state data (agent position)
            action_key: Key for action data
            indicator_key: Key for goal completion indicator
            goal_1_keypoint_key: Key for goal 1 keypoints
            goal_2_keypoint_key: Key for goal 2 keypoints
            goal_3_keypoint_key: Key for goal 3 keypoints
            include_goal_keypoints: If True, append goal keypoints to obs (adds 54 dims)
            seed: Random seed for train/val split
            val_ratio: Ratio of episodes to use for validation
            action_step_subsample: Subsample rate to align with DP n_action_steps.
                When >1, only every N-th timestep is kept (e.g., 4 for n_action_steps=4).
                This matches the DP decision frequency during rollout.
            max_train_episodes: Maximum number of training episodes to use
        """
        super().__init__()
        
        self.include_goal_keypoints = bool(include_goal_keypoints)
        
        # Load replay buffer with necessary keys including indicator
        keys = [obs_key, state_key, action_key, indicator_key]
        if self.include_goal_keypoints:
            keys.extend([goal_1_keypoint_key, goal_2_keypoint_key, goal_3_keypoint_key])
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=keys)

        # Create train/val split at episode level
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key
        self.indicator_key = indicator_key
        self.goal_1_keypoint_key = goal_1_keypoint_key
        self.goal_2_keypoint_key = goal_2_keypoint_key
        self.goal_3_keypoint_key = goal_3_keypoint_key
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.action_step_subsample = int(action_step_subsample)
        
        # Get list of episode indices for training
        self.episode_indices = np.where(train_mask)[0]

    def get_validation_dataset(self):
        """Create validation dataset with same replay buffer but different episodes."""
        val_set = copy.copy(self)
        val_set.episode_indices = np.where(self.val_mask)[0]
        val_set.train_mask = self.val_mask
        val_set.val_mask = self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        """
        Compute normalizer from all data in the replay buffer.
        This works with variable-length episodes.
        """
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        """Return all actions in the replay buffer."""
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        """Return number of episodes, not timesteps."""
        return len(self.episode_indices)

    def _sample_to_data(self, sample):
        """
        Convert sample to observation, action, and indicator data.
        
        Constructs observation by concatenating:
        - Current block keypoints (9, 2) -> (18,)
        - Agent position (2,)
        - [Optional] Goal 1 keypoints (9, 2) -> (18,)
        - [Optional] Goal 2 keypoints (9, 2) -> (18,)
        - [Optional] Goal 3 keypoints (9, 2) -> (18,)
        Total obs dimension: 20 (without goals) or 74 (with goals)
        
        Args:
            sample: Dictionary with keys from replay buffer
        
        Returns:
            Dictionary with 'obs', 'action', and 'indicator' keys
        """
        keypoint = sample[self.obs_key]  # (T, 9, 2)
        state = sample[self.state_key]   # (T, 2) - only agent position
        action = sample[self.action_key]  # (T, 2)
        
        # Extract agent position from state
        agent_pos = state  # Already just agent position
        
        # Flatten keypoints
        keypoint_flat = keypoint.reshape(keypoint.shape[0], -1)  # (T, 18)
        
        # Create previous action by shifting action by 1 timestep
        # For the first timestep, use zero action
        T = keypoint.shape[0]
        prev_action = np.zeros_like(action)  # (T, 2)
        prev_action[1:] = action[:-1]  # Shift action by 1
        
        # Build obs components
        obs_parts = [
            keypoint_flat,   # Current block keypoints (18)
            agent_pos,       # Agent position (2)
        ]
        
        if self.include_goal_keypoints:
            goal_1_kp = sample[self.goal_1_keypoint_key]  # (T, 9, 2)
            goal_2_kp = sample[self.goal_2_keypoint_key]  # (T, 9, 2)
            goal_3_kp = sample[self.goal_3_keypoint_key]  # (T, 9, 2)
            obs_parts.extend([
                goal_1_kp.reshape(T, -1),  # Goal 1 keypoints (18)
                goal_2_kp.reshape(T, -1),  # Goal 2 keypoints (18)
                goal_3_kp.reshape(T, -1),  # Goal 3 keypoints (18)
            ])
        
        # Concatenate to form observation
        obs = np.concatenate(obs_parts, axis=-1)  # 20 or 74 dimensions

        data = {
            'obs': obs,  # (T, 20)
            'action': action,  # (T, 2)
            'indicator': sample[self.indicator_key],  # (T, 3)
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get entire trajectory for a single episode.
        
        Args:
            idx: Index into episode_indices (not raw episode index)
        
        Returns:
            Dictionary containing:
                - 'obs': (T_episode, 20) tensor
                - 'action': (T_episode, 2) tensor
                - 'indicator': (T_episode, 3) tensor
            where T_episode is the length of this specific episode
        """
        # Get actual episode index from train/val mask
        episode_idx = self.episode_indices[idx]
        
        # Get full episode data from replay buffer
        episode_sample = self.replay_buffer.get_episode(episode_idx, copy=False)
        
        # Convert to observation, action, and indicator
        data = self._sample_to_data(episode_sample)
        
        # Apply action step subsampling to align with DP decision frequency
        if self.action_step_subsample > 1:
            s = self.action_step_subsample
            data = {k: v[s-1::s] for k, v in data.items()}
        
        # Convert to torch tensors
        torch_data = dict_apply(data, torch.from_numpy)
        
        return torch_data


def collate_fn_lstm_with_indicator(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for LSTM dataset with indicator that handles variable-length trajectories.
    
    Args:
        batch: List of samples, each containing:
            - 'obs': (T_episode, obs_dim) tensor
            - 'action': (T_episode, action_dim) tensor
            - 'indicator': (T_episode, 3) tensor
    
    Returns:
        Dictionary containing:
            - 'obs': (batch, T_max, obs_dim) padded tensor
            - 'action': (batch, T_max, action_dim) padded tensor
            - 'indicator': (batch, T_max, 3) padded tensor
            - 'mask': (batch, T_max) binary mask (1=real, 0=padding)
            - 'lengths': (batch,) tensor of actual trajectory lengths
    """
    # Find max trajectory length in batch
    max_length = max(sample['obs'].shape[0] for sample in batch)
    batch_size = len(batch)
    obs_dim = batch[0]['obs'].shape[1]
    action_dim = batch[0]['action'].shape[1]
    indicator_dim = batch[0]['indicator'].shape[1]
    
    # Initialize padded tensors
    obs_padded = torch.zeros(batch_size, max_length, obs_dim, dtype=batch[0]['obs'].dtype)
    action_padded = torch.zeros(batch_size, max_length, action_dim, dtype=batch[0]['action'].dtype)
    indicator_padded = torch.zeros(batch_size, max_length, indicator_dim, dtype=batch[0]['indicator'].dtype)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill in actual data
    for i, sample in enumerate(batch):
        length = sample['obs'].shape[0]
        lengths[i] = length
        obs_padded[i, :length] = sample['obs']
        action_padded[i, :length] = sample['action']
        indicator_padded[i, :length] = sample['indicator']
        mask[i, :length] = 1
    
    return {
        'obs': obs_padded,
        'action': action_padded,
        'indicator': indicator_padded,
        'mask': mask,
        'lengths': lengths
    }
