"""
Image-observation wrapper for the Push-T friction task.

Inherits from PushTKeypointsFrictionEnv (which owns the friction-gradient
physics, the two-trial state machine and the custom render) and just replaces
the keypoint observation with a {'image', 'agent_pos'} dict so the task slots
into every standard image policy / workspace.

The image policy is intentionally memory-LESS — the 2D memory flags exposed
by the parent (current_trial, return_phase) are NOT included in the obs,
since the whole point of this benchmark is whether a policy's backbone can
recover lane-attempt history from raw pixels. If you do want the flags in
the obs later, switch to the keypoint env or extend this class.
"""
from typing import Optional

import numpy as np
import cv2
from gym import spaces

from memory_diffusion_policy.env.pusht.pusht_keypoints_friction_env import PushTKeypointsFrictionEnv


class PushTImageFrictionEnv(PushTKeypointsFrictionEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
                 legacy: bool = False,
                 block_cog=None,
                 damping=None,
                 render_size: int = 96,
                 # --- friction-specific knobs (pass-through to parent) ---
                 track_y_lo: float = 190,
                 track_y_hi: float = 370,
                 divider_x: float = 256,
                 start_area_center=(256, 90),
                 start_area_half=(180, 70),
                 end_area_center=(256, 435),
                 end_area_half=(220, 65),
                 goal_t_pose: Optional[np.ndarray] = None,
                 coverage_threshold: float = 0.90,
                 block_scale: Optional[int] = None,
                 agent_radius: Optional[int] = None,
                 spawn_angle_range=(-np.pi / 6, np.pi / 6),
                 low_drag: float = 0.4,
                 high_drag: float = 500.0,
                 friction_full_at: float = 0.5,
                 friction_segments: int = 16,
                 max_trials: int = 2,
                 stuck_threshold: int = 40,
                 stuck_eps: float = 1.2,
                 reset_grace_steps: int = 30,
                 return_phase_timeout: int = 400,
                 lane_crossing_fail: bool = True,
                 debug_show_friction: bool = False):
        super().__init__(
            legacy=legacy,
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            # Keypoint bookkeeping is still active in the parent (the reset
            # caches goal keypoints for info dicts, etc.) but is not used
            # to build the observation here.
            keypoint_visible_rate=1.0,
            agent_keypoints=False,
            draw_keypoints=False,
            # We draw the action marker manually onto the cached frame only,
            # so the obs image shown to the policy never contains it.
            render_action=False,
            # Image obs = memory-less policy. See module docstring.
            include_memory_flags=False,
            track_y_lo=track_y_lo,
            track_y_hi=track_y_hi,
            divider_x=divider_x,
            start_area_center=start_area_center,
            start_area_half=start_area_half,
            end_area_center=end_area_center,
            end_area_half=end_area_half,
            goal_t_pose=goal_t_pose,
            coverage_threshold=coverage_threshold,
            block_scale=block_scale,
            agent_radius=agent_radius,
            spawn_angle_range=spawn_angle_range,
            low_drag=low_drag,
            high_drag=high_drag,
            friction_full_at=friction_full_at,
            friction_segments=friction_segments,
            max_trials=max_trials,
            stuck_threshold=stuck_threshold,
            stuck_eps=stuck_eps,
            reset_grace_steps=reset_grace_steps,
            return_phase_timeout=return_phase_timeout,
            lane_crossing_fail=lane_crossing_fail,
            debug_show_friction=debug_show_friction,
        )
        ws = self.window_size
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=1,
                shape=(3, render_size, render_size),
                dtype=np.float32,
            ),
            'agent_pos': spaces.Box(
                low=0, high=ws,
                shape=(2,), dtype=np.float32,
            ),
        })
        self.render_cache = None

    def _get_obs(self):
        img = self._render_frame(mode='rgb_array')
        agent_pos = np.array(self.agent.position, dtype=np.float32)
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {
            'image': img_obs,
            'agent_pos': agent_pos,
        }

        if self.latest_action is not None:
            action = np.array(self.latest_action)
            coord = (action / 512 * self.render_size).astype(np.int32)
            marker_size = int(8 / 96 * self.render_size)
            thickness = int(1 / 96 * self.render_size)
            cv2.drawMarker(img, coord,
                           color=(0, 0, 0), markerType=cv2.MARKER_CROSS,
                           markerSize=marker_size, thickness=thickness)
        self.render_cache = img

        return obs

    def render(self, mode):
        assert mode == 'rgb_array'
        if self.render_cache is None:
            self._get_obs()
        return self.render_cache
