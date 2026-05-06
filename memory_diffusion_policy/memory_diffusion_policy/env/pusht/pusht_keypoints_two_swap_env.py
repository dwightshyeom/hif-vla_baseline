"""
Keypoint-observation wrapper for the Push-T two-block swap task.

The swap zarr dataset does NOT store precomputed keypoints — the 9 block
keypoints are derived on the fly from each block's pymunk body (at eval) or
from the stored block pose (at training). Both sides use the exact same local
template produced by ``PushTKeypointsEnv.genenerate_keypoint_manager_params``
so the env observation and the dataset observation are bit-identical.

Observation layout (flat 1-D vector, same convention as PushTKeypointsEnv):

    [ blue_kps (18)                         # 9 × (x, y)
      red_kps  (18)
      agent_pos (2)                         # Base: 38 dims

      # Optional — always included by default because the blue↔red target
      # assignment flips with blue_start_side and the policy otherwise has
      # no way to know which block goes where.
      blue_target_kps (18)
      red_target_kps  (18) ]                # With targets: 74 dims

    [ obs_mask (same length as obs) ]       # binary visibility mask
"""
from typing import Dict, Optional
import numpy as np
from gym import spaces

from memory_diffusion_policy.env.pusht.pusht_two_swap_env import PushTTwoSwapEnv
from memory_diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager


class PushTKeypointsTwoSwapEnv(PushTTwoSwapEnv):
    """
    Low-dim keypoint observations for the swap task.

    Shares the local T-block keypoint template with PushTKeypointsEnv so that
    the same PyMunk-derived 9 points are used across single-goal, three-goals
    and swap tasks.
    """

    def __init__(self,
                 legacy=False,
                 block_cog=None,
                 damping=None,
                 render_action=True,
                 render_size=96,
                 reset_to_state=None,
                 target_a_pose=None,
                 target_b_pose=None,
                 agent_start_pos=None,
                 blue_start_side='left',
                 waypoint_center=None,
                 waypoint_radius=20,
                 waypoint_color='Black',
                 keypoint_visible_rate=1.0,
                 include_target_keypoints=True,
                 local_keypoint_map: Optional[Dict[str, np.ndarray]] = None,
                 color_map: Optional[Dict[str, np.ndarray]] = None):
        # Reuse the single-goal Push-T keypoint template (9 points on the T)
        # so the same local_keypoint_map['block'] drives every variant.
        if local_keypoint_map is None:
            kp_kwargs = self.genenerate_keypoint_manager_params()
            local_keypoint_map = kp_kwargs['local_keypoint_map']
            color_map = kp_kwargs['color_map']

        super().__init__(
            legacy=legacy,
            block_cog=block_cog,
            damping=damping,
            render_action=render_action,
            render_size=render_size,
            reset_to_state=reset_to_state,
            target_a_pose=target_a_pose,
            target_b_pose=target_b_pose,
            agent_start_pos=agent_start_pos,
            blue_start_side=blue_start_side,
            waypoint_center=waypoint_center,
            waypoint_radius=waypoint_radius,
            waypoint_color=waypoint_color,
        )

        self.kp_manager = PymunkKeypointManager(
            local_keypoint_map=local_keypoint_map,
            color_map=color_map,
        )
        self.keypoint_visible_rate = float(keypoint_visible_rate)
        self.include_target_keypoints = bool(include_target_keypoints)

        # Two blocks × 9 kps × 2 coords = 36, plus agent_pos = 2 -> 38.
        # Optional: blue_target (18) + red_target (18) = 36 -> 74.
        Do = 38 + (36 if self.include_target_keypoints else 0)
        Dobs = Do * 2  # concat(data, mask)
        ws = self.window_size
        low = np.zeros((Dobs,), dtype=np.float64)
        high = np.full_like(low, ws)
        high[Do:] = 1.0  # mask range is [0, 1]
        self.observation_space = spaces.Box(
            low=low, high=high, shape=low.shape, dtype=np.float64,
        )

    @classmethod
    def genenerate_keypoint_manager_params(cls):
        """Same 9-keypoint T-template used by PushTKeypointsEnv."""
        from memory_diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
        return PushTKeypointsEnv.genenerate_keypoint_manager_params()

    def _kps_from_body(self, body):
        return self.kp_manager.get_keypoints_global(
            pose_map={'block': body}, is_obj=True)['block']

    def _kps_from_pose(self, pose):
        return self.kp_manager.get_keypoints_global(
            pose_map={'block': pose}, is_obj=False)['block']

    def _get_obs(self):
        blue_kps = self._kps_from_body(self.blue_block)  # (9, 2)
        red_kps  = self._kps_from_body(self.red_block)   # (9, 2)
        agent_pos = np.array(self.agent.position, dtype=np.float64)  # (2,)

        # Keypoint visibility masking (agent_pos always visible).
        n_block_kps = blue_kps.shape[0] + red_kps.shape[0]  # 18
        visible = self.np_random.random(size=(n_block_kps,)) < self.keypoint_visible_rate
        kp_mask = np.repeat(visible[:, None], 2, axis=1).flatten()  # (36,)

        obs_data = np.concatenate([
            blue_kps.flatten(),
            red_kps.flatten(),
            agent_pos,
        ])  # (38,)
        obs_mask = np.concatenate([
            kp_mask,
            np.ones((2,), dtype=bool),
        ])  # (38,)

        if self.include_target_keypoints:
            blue_target_kps = self._kps_from_pose(self.blue_target_pose)  # (9, 2)
            red_target_kps  = self._kps_from_pose(self.red_target_pose)   # (9, 2)
            obs_data = np.concatenate([
                obs_data,
                blue_target_kps.flatten(),
                red_target_kps.flatten(),
            ])
            obs_mask = np.concatenate([
                obs_mask,
                np.ones((36,), dtype=bool),
            ])

        return np.concatenate([obs_data, obs_mask.astype(obs_data.dtype)])
