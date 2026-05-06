"""
Image-observation wrapper for the THREE-TRACK friction task.

Mirrors PushTImageFrictionEnv but inherits from the 3-track keypoints env so
the policy gets {'image', 'agent_pos'} obs over the harder layout. Memory
flags are NOT included in the obs (memory-less image policy benchmark — the
backbone has to recover lane-attempt history from raw pixels).
"""
from typing import Optional

import numpy as np
import cv2
from gym import spaces

from memory_diffusion_policy.env.pusht.pusht_keypoints_friction_three_tracks_env import (
    PushTKeypointsFrictionThreeTracksEnv,
)


class PushTImageFrictionThreeTracksEnv(PushTKeypointsFrictionThreeTracksEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
                 legacy: bool = False,
                 block_cog=None,
                 damping=None,
                 render_size: int = 96,
                 # Geometry: equal-length start/end strips, equal-width lanes.
                 track_y_lo: float = 175.0,
                 track_y_hi: float = 336.0,
                 divider_1_x: float = 172.0,
                 divider_2_x: float = 339.0,
                 start_area_center=(256.0, 90.0),
                 start_area_half=(130.0, 65.0),
                 end_area_center=(256.0, 421.0),
                 end_area_half=(130.0, 65.0),
                 goal_t_pose: Optional[np.ndarray] = None,
                 coverage_threshold: float = 0.90,
                 block_scale: Optional[int] = None,
                 agent_radius: Optional[int] = None,
                 spawn_angle_range=(-np.pi / 6, np.pi / 6),
                 low_drag: float = 0.4,
                 high_drag: float = 500.0,
                 friction_full_at: float = 0.5,
                 friction_segments: int = 16,
                 max_trials: int = 3,
                 stuck_threshold: int = 40,
                 stuck_eps: float = 1.2,
                 reset_grace_steps: int = 30,
                 return_phase_timeout: int = 400,
                 lane_crossing_fail: bool = True,
                 debug_show_friction: bool = False,
                 # Return-to-original-location success criterion.
                 return_pos_tol: float = 25.0,
                 return_ang_tol: float = 0.35,
                 # Translucent red box rendered at the start area.
                 start_box_color=(220, 50, 50),
                 start_box_alpha: int = 70):
        super().__init__(
            legacy=legacy,
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            keypoint_visible_rate=1.0,
            agent_keypoints=False,
            draw_keypoints=False,
            render_action=False,
            include_memory_flags=False,
            track_y_lo=track_y_lo,
            track_y_hi=track_y_hi,
            divider_1_x=divider_1_x,
            divider_2_x=divider_2_x,
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
            return_pos_tol=return_pos_tol,
            return_ang_tol=return_ang_tol,
            start_box_color=start_box_color,
            start_box_alpha=start_box_alpha,
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
