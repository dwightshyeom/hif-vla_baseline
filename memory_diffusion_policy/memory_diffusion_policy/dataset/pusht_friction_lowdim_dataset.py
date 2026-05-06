"""
Low-dim (keypoint) dataset for the Push-T friction task.

The friction env uses a smaller T (block_scale=16) so keypoints must come
from the scaled template returned by
``PushTKeypointsFrictionEnv.genenerate_keypoint_manager_params()``. Whether
the zarr stores precomputed keypoints (``keypoint`` key) or just the block
pose in ``state`` is supported below.

Expected zarr keys (match to your demo-collection script):

    state:    (N, >=5) float32
        [:, 0:2] = agent position (always)
        [:, 2:5] = block pose (x, y, theta) — used if ``keypoint`` is absent

    action:   (N, 2) float32

    keypoint: (N, 9, 2) float32   — optional; precomputed block keypoints
                                    using the friction env's scaled template.
                                    If missing, we project state[:, 2:5] via
                                    the same local template at sample time.

    memory_flags: (N, 2) float32  — optional; column 0 = current_trial /
                                    max_trials, column 1 = float(return_phase).
                                    Required when include_memory_flags=True.

Observation produced (matches PushTKeypointsFrictionEnv._get_obs layout):

    [ block_kps (18), agent_pos (2) ]                           # 20  dims
    [ block_kps (18), agent_pos (2), memory_flags (2) ]         # 22  dims   (include_memory_flags=True)

followed by an equal-length all-ones mask so the flat vector shape matches
what the keypoint env emits. ``include_memory_flags`` MUST match the
env_runner's setting.
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
    """Broadcast-friendly affine projection. See pusht_two_swap_lowdim_dataset."""
    x = poses[..., 0:1]
    y = poses[..., 1:2]
    theta = poses[..., 2:3]
    c = np.cos(theta)
    s = np.sin(theta)
    kx = local_kps[..., 0]
    ky = local_kps[..., 1]
    gx = c * kx - s * ky + x
    gy = s * kx + c * ky + y
    return np.stack([gx, gy], axis=-1)


class PushTFrictionLowdimDataset(BaseLowdimDataset):
    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 state_key='state',
                 action_key='action',
                 keypoint_key='keypoint',
                 memory_flags_key='memory_flags',
                 include_memory_flags=True,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None):
        super().__init__()
        self.include_memory_flags = bool(include_memory_flags)
        self.state_key = state_key
        self.action_key = action_key
        self.keypoint_key = keypoint_key
        self.memory_flags_key = memory_flags_key

        # Decide at construction which keys the zarr actually has so we
        # can skip loading arrays that don't exist.  Peek at the zarr's
        # data group directly — we previously probed via
        # ReplayBuffer.copy_from_path(keys=[state_key]) which only ever
        # exposed state_key, so _has_precomputed_kps was always False
        # and a 2-D state zarr couldn't be loaded.
        import os as _os
        import zarr as _zarr
        _src = _zarr.open(_os.path.expanduser(zarr_path), 'r')
        available = set(_src['data'].keys()) if 'data' in _src else set()
        self._has_precomputed_kps = keypoint_key in available
        self._has_memory_flags = memory_flags_key in available

        keys = [state_key, action_key]
        if self._has_precomputed_kps:
            keys.append(keypoint_key)
        if self.include_memory_flags:
            if not self._has_memory_flags:
                raise KeyError(
                    f"include_memory_flags=True but zarr at {zarr_path!r} "
                    f"has no '{memory_flags_key}' array. Either disable "
                    f"memory flags on both the dataset and env_runner, or "
                    f"re-collect demos with memory_flags written out.")
            keys.append(memory_flags_key)
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

        # Scaled keypoint template — shared with PushTKeypointsFrictionEnv.
        from memory_diffusion_policy.env.pusht.pusht_keypoints_friction_env import (
            PushTKeypointsFrictionEnv)
        kp_kwargs = PushTKeypointsFrictionEnv.genenerate_keypoint_manager_params()
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

    def _block_kps_from_sample(self, sample) -> np.ndarray:
        """(..., 9, 2) block keypoints, either precomputed or projected."""
        if self._has_precomputed_kps:
            return np.asarray(sample[self.keypoint_key], dtype=np.float32)
        state = np.asarray(sample[self.state_key], dtype=np.float32)
        if state.shape[-1] < 5:
            raise ValueError(
                "Lowdim friction dataset needs either a precomputed "
                "'keypoint' array OR at least 5-D state "
                "[agent_x, agent_y, block_x, block_y, block_theta]; "
                f"got state.shape={state.shape}.")
        return _project_keypoints(state[..., 2:5], self.block_local_kps)

    def _sample_to_data(self, sample):
        state = np.asarray(sample[self.state_key], dtype=np.float32)
        agent_pos = state[..., :2]
        block_kps = self._block_kps_from_sample(sample)  # (..., 9, 2)
        lead_shape = state.shape[:-1]

        parts = [
            block_kps.reshape(*lead_shape, -1),  # (..., 18)
            agent_pos,                           # (..., 2)
        ]
        if self.include_memory_flags:
            flags = np.asarray(sample[self.memory_flags_key], dtype=np.float32)
            parts.append(flags.reshape(*lead_shape, -1))  # (..., 2)

        obs = np.concatenate(parts, axis=-1).astype(np.float32)
        return {
            'obs': obs,
            'action': np.asarray(sample[self.action_key], dtype=np.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)
