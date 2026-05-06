"""
Low-dim (keypoint) dataset for the Push-T two-block swap task.

The swap zarr stores 8-D ``state`` (agent + blue pose + red pose) and the
per-episode ``blue_target_pose`` / ``red_target_pose`` but does NOT store
precomputed keypoints. At sample time we project the same 9-point local T
template used by PushTKeypointsEnv onto each pose, yielding the exact
observation layout that ``PushTKeypointsTwoSwapEnv._get_obs`` produces:

    [ blue_kps (18),
      red_kps  (18),
      agent_pos (2),
      (optional) blue_target_kps (18),
      (optional) red_target_kps  (18) ]    # Do = 38 or 74

followed by a ``Do``-length visibility mask. The env always emits full-1
masks for training data, so we do the same here (the env only applies
dropout through ``keypoint_visible_rate`` at eval time).
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
from memory_diffusion_policy.dataset.base_dataset import BaseLowdimDataset


def _project_keypoints(poses: np.ndarray, local_kps: np.ndarray) -> np.ndarray:
    """Affine-project a local (K, 2) template by per-step (..., 3) poses.

    poses:     (..., 3) — [x, y, theta]
    local_kps: (K, 2)
    returns:   (..., K, 2)

    Matches skimage.transform.AffineTransform used by PymunkKeypointManager:
        [xg, yg] = R(theta) @ [xl, yl] + [x, y]
    """
    x = poses[..., 0:1]           # (..., 1)
    y = poses[..., 1:2]
    theta = poses[..., 2:3]
    c = np.cos(theta)
    s = np.sin(theta)
    kx = local_kps[..., 0]         # (K,)
    ky = local_kps[..., 1]
    gx = c * kx - s * ky + x       # (..., K)
    gy = s * kx + c * ky + y
    return np.stack([gx, gy], axis=-1)


class PushTTwoSwapLowdimDataset(BaseLowdimDataset):
    """
    Vanilla-DP keypoint dataset for the swap task.

    Observations are built by:
      1. slicing ``state`` into agent_pos / blue_pose / red_pose,
      2. projecting blue & red pose into global keypoints via the shared
         T-block template, and
      3. (optional) projecting blue_target_pose / red_target_pose
         similarly — recommended when include_target_keypoints=True, since
         the blue↔red target assignment flips with blue_start_side.

    ``include_target_keypoints`` MUST match the env-side setting.
    """

    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 state_key='state',
                 action_key='action',
                 blue_target_key='blue_target_pose',
                 red_target_key='red_target_pose',
                 include_target_keypoints=True,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None):
        super().__init__()
        self.include_target_keypoints = bool(include_target_keypoints)

        keys = [state_key, action_key]
        if self.include_target_keypoints:
            keys.extend([blue_target_key, red_target_key])
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys)

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
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.state_key = state_key
        self.action_key = action_key
        self.blue_target_key = blue_target_key
        self.red_target_key = red_target_key

        # Local template — shared with PushTKeypointsEnv and the env wrapper
        # so training and eval keypoints are bit-identical.
        from memory_diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
        kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
        self.block_local_kps = np.asarray(
            kp_kwargs['local_keypoint_map']['block'], dtype=np.float32)

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
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        state = np.asarray(sample[self.state_key], dtype=np.float32)  # (..., 8)
        agent_pos = state[..., 0:2]                                   # (..., 2)
        blue_pose = state[..., 2:5]                                   # (..., 3)
        red_pose  = state[..., 5:8]                                   # (..., 3)

        blue_kps = _project_keypoints(blue_pose, self.block_local_kps)  # (..., 9, 2)
        red_kps  = _project_keypoints(red_pose,  self.block_local_kps)

        lead_shape = state.shape[:-1]  # (...,)
        parts = [
            blue_kps.reshape(*lead_shape, -1),   # (..., 18)
            red_kps.reshape(*lead_shape, -1),    # (..., 18)
            agent_pos,                           # (..., 2)
        ]

        if self.include_target_keypoints:
            blue_target_pose = np.asarray(
                sample[self.blue_target_key], dtype=np.float32)  # (..., 3)
            red_target_pose = np.asarray(
                sample[self.red_target_key], dtype=np.float32)
            blue_target_kps = _project_keypoints(blue_target_pose, self.block_local_kps)
            red_target_kps  = _project_keypoints(red_target_pose,  self.block_local_kps)
            parts.extend([
                blue_target_kps.reshape(*lead_shape, -1),  # (..., 18)
                red_target_kps.reshape(*lead_shape, -1),   # (..., 18)
            ])

        obs = np.concatenate(parts, axis=-1).astype(np.float32)  # (..., Do)
        return {
            'obs': obs,
            'action': np.asarray(sample[self.action_key], dtype=np.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)
