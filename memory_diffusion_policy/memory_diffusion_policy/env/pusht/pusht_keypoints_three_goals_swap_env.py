"""
Keypoint-observation wrapper for the Push-T three-goals swap task.

The 9 block keypoints are derived on the fly from each block's pymunk body.
Both colored blocks share the same local T template (the one produced by
``PushTKeypointsEnv.genenerate_keypoint_manager_params``), so the env
observation is bit-identical to the keypoints derived from the stored block
pose at training time.

Observation layout (flat 1-D vector):

    [ blue_kps         (18)   # 9 × (x, y)
      red_kps          (18)
      agent_pos        (2)
      blue_target_kps  (18)   # = keypoints at blue's swap target (= red's start)
      red_target_kps   (18)   # = keypoints at red's swap target  (= blue's start)
      empty_target_kps (18) ] # = keypoints at the intermediate (empty) target
                              # Total D = 92

    [ obs_mask (same length as obs) ]

The three pairs of "_target_kps" arrays let the policy disambiguate the
swap-role of each fixed green target on a per-episode basis. They change
identity (i.e. which fixed target index they reference) every reset() but
the underlying three positions never move.
"""

from typing import Dict, Optional
import numpy as np
from gym import spaces

from memory_diffusion_policy.env.pusht.pusht_three_goals_swap_env import PushTThreeGoalsSwapEnv
from memory_diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager


class PushTKeypointsThreeGoalsSwapEnv(PushTThreeGoalsSwapEnv):
    """
    Low-dim keypoint observations for the three-goals swap task.

    Shares the local T-block keypoint template with PushTKeypointsEnv so
    that all variants use the same 9 points.
    """

    def __init__(
        self,
        legacy=False,
        block_cog=None,
        damping=None,
        render_action=True,
        render_size=96,
        reset_to_state=None,
        goal_pose_1=None,
        goal_pose_2=None,
        goal_pose_3=None,
        agent_start_pos=None,
        success_threshold: float = 0.9,
        leave_threshold: float = 0.3,
        keypoint_visible_rate: float = 1.0,
        local_keypoint_map: Optional[Dict[str, np.ndarray]] = None,
        color_map: Optional[Dict[str, np.ndarray]] = None,
    ):
        # Reuse the shared 9-keypoint T template.
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
            goal_pose_1=goal_pose_1,
            goal_pose_2=goal_pose_2,
            goal_pose_3=goal_pose_3,
            agent_start_pos=agent_start_pos,
            success_threshold=success_threshold,
            leave_threshold=leave_threshold,
        )

        self.kp_manager = PymunkKeypointManager(
            local_keypoint_map=local_keypoint_map,
            color_map=color_map,
        )
        self.keypoint_visible_rate = float(keypoint_visible_rate)

        # 2 blocks × 9 kps × 2 = 36, agent_pos = 2, three target kps =
        # 3 × 18 = 54. Total D = 92, observation shape = (184,).
        Do = 2 * 18 + 2 + 3 * 18  # 92
        Dobs = Do * 2
        ws = self.window_size
        low = np.zeros((Dobs,), dtype=np.float64)
        high = np.full_like(low, ws)
        high[Do:] = 1.0  # mask range [0, 1]
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
        red_kps = self._kps_from_body(self.red_block)    # (9, 2)
        agent_pos = np.array(self.agent.position, dtype=np.float64)  # (2,)

        # Target / intermediate-target keypoints based on this episode's
        # role assignment. Identities change per episode but the three
        # underlying positions are fixed.
        blue_target_kps = self._kps_from_pose(self._goal_pose(self.blue_target_idx))
        red_target_kps = self._kps_from_pose(self._goal_pose(self.red_target_idx))
        empty_target_kps = self._kps_from_pose(self._goal_pose(self.empty_idx))

        # Keypoint visibility masking on the two block kp sets only.
        n_block_kps = blue_kps.shape[0] + red_kps.shape[0]  # 18
        visible = self.np_random.random(size=(n_block_kps,)) < self.keypoint_visible_rate
        kp_mask = np.repeat(visible[:, None], 2, axis=1).flatten()  # (36,)

        obs_data = np.concatenate([
            blue_kps.flatten(),
            red_kps.flatten(),
            agent_pos,
            blue_target_kps.flatten(),
            red_target_kps.flatten(),
            empty_target_kps.flatten(),
        ])
        obs_mask = np.concatenate([
            kp_mask,                                                  # block kps mask
            np.ones((2,), dtype=bool),                                # agent
            np.ones((54,), dtype=bool),                               # 3 target kps fully visible
        ])

        return np.concatenate([obs_data, obs_mask.astype(obs_data.dtype)])
