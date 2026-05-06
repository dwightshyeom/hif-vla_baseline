from typing import Dict, Optional
from gym import spaces
from memory_diffusion_policy.env.pusht.pusht_asym_env import PushTAsymEnv
from memory_diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from memory_diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager
import numpy as np


class PushTAsymKeypointsEnv(PushTAsymEnv):
    """
    PushT with asymmetric mass + keypoint observations.

    Keypoint generation reuses the same method as PushTKeypointsEnv since
    the T-block geometry is identical (only mass distribution differs).
    """

    def __init__(self,
            legacy=False,
            block_cog=None,
            damping=None,
            render_size=96,
            keypoint_visible_rate=1.0,
            agent_keypoints=False,
            draw_keypoints=False,
            reset_to_state=None,
            render_action=True,
            goal_pose=None,
            randomize_goal=False,
            heavy_mass=15.0,
            light_mass=0.1,
            local_keypoint_map: Dict[str, np.ndarray]=None,
            color_map: Optional[Dict[str, np.ndarray]]=None):
        super().__init__(
            legacy=legacy,
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            reset_to_state=reset_to_state,
            render_action=render_action,
            goal_pose=goal_pose,
            randomize_goal=randomize_goal,
            heavy_mass=heavy_mass,
            light_mass=light_mass)
        ws = self.window_size

        if local_keypoint_map is None:
            # Reuse keypoint params from standard PushTKeypointsEnv
            # (same T geometry, just different mass distribution)
            kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
            local_keypoint_map = kp_kwargs['local_keypoint_map']
            color_map = kp_kwargs['color_map']

        # Create observation spaces (same structure as PushTKeypointsEnv)
        Dblockkps = np.prod(local_keypoint_map['block'].shape)
        Dagentkps = np.prod(local_keypoint_map['agent'].shape)
        Dagentpos = 2

        Do = Dblockkps
        if agent_keypoints:
            Do += Dagentkps
        else:
            Do += Dagentpos
        Dobs = Do * 2

        low = np.zeros((Dobs,), dtype=np.float64)
        high = np.full_like(low, ws)
        high[Do:] = 1.

        self.observation_space = spaces.Box(
            low=low, high=high, shape=low.shape, dtype=np.float64)

        self.keypoint_visible_rate = keypoint_visible_rate
        self.agent_keypoints = agent_keypoints
        self.draw_keypoints = draw_keypoints
        self.kp_manager = PymunkKeypointManager(
            local_keypoint_map=local_keypoint_map,
            color_map=color_map)
        self.draw_kp_map = None
        self._cached_goal_keypoint = None

    def reset(self):
        obs = super().reset()
        # Cache goal keypoints once per episode for consistency
        goal_pose_map = {'block': self.goal_pose}
        goal_kp_map = self.kp_manager.get_keypoints_global(
            pose_map=goal_pose_map, is_obj=False)
        self._cached_goal_keypoint = goal_kp_map['block'].copy()
        return obs

    def _get_goal_keypoints(self):
        return self._cached_goal_keypoint

    def _get_info(self):
        info = super()._get_info()
        info['goal_keypoint'] = self._get_goal_keypoints()
        return info

    def _get_obs(self):
        # Get keypoints (same logic as PushTKeypointsEnv)
        obj_map = {'block': self.block}
        if self.agent_keypoints:
            obj_map['agent'] = self.agent

        kp_map = self.kp_manager.get_keypoints_global(
            pose_map=obj_map, is_obj=True)
        kps = np.concatenate(list(kp_map.values()), axis=0)

        # Select keypoints to drop
        n_kps = kps.shape[0]
        visible_kps = self.np_random.random(size=(n_kps,)) < self.keypoint_visible_rate
        kps_mask = np.repeat(visible_kps[:, None], 2, axis=1)

        # Save keypoints for rendering
        vis_kps = kps.copy()
        vis_kps[~visible_kps] = 0
        draw_kp_map = {
            'block': vis_kps[:len(kp_map['block'])]
        }
        if self.agent_keypoints:
            draw_kp_map['agent'] = vis_kps[len(kp_map['block']):]
        self.draw_kp_map = draw_kp_map

        # Construct obs
        obs = kps.flatten()
        obs_mask = kps_mask.flatten()
        if not self.agent_keypoints:
            agent_pos = np.array(self.agent.position)
            obs = np.concatenate([obs, agent_pos])
            obs_mask = np.concatenate([obs_mask, np.ones((2,), dtype=bool)])

        obs = np.concatenate([obs, obs_mask.astype(obs.dtype)], axis=0)
        return obs

    def _render_frame(self, mode):
        img = super()._render_frame(mode)
        if self.draw_keypoints:
            self.kp_manager.draw_keypoints(
                img, self.draw_kp_map, radius=int(img.shape[0] / 96))
        return img
