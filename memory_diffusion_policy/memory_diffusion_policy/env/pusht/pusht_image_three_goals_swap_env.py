"""
Image-observation wrapper for the Push-T three-goals swap task.

Returns {'image': (3, H, W) float32 in [0,1], 'agent_pos': (2,)} instead of
the 8D low-dim state. The image already encodes everything the policy needs:
- the three fixed green targets,
- the two colored blocks (blue / red) at their current positions,
which is enough to infer the per-episode role assignment and the swap
intent.

The action marker is NOT rendered into the policy observation image, but
IS drawn onto ``render_cache`` so the video overlay shows it at eval time.
"""

from gym import spaces
from memory_diffusion_policy.env.pusht.pusht_three_goals_swap_env import PushTThreeGoalsSwapEnv
import numpy as np
import cv2


class PushTImageThreeGoalsSwapEnv(PushTThreeGoalsSwapEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(
        self,
        legacy=False,
        block_cog=None,
        damping=None,
        render_size=96,
        goal_pose_1=None,
        goal_pose_2=None,
        goal_pose_3=None,
        agent_start_pos=None,
        success_threshold: float = 0.9,
        leave_threshold: float = 0.3,
    ):
        super().__init__(
            legacy=legacy,
            block_cog=block_cog,
            damping=damping,
            render_action=False,
            render_size=render_size,
            reset_to_state=None,
            goal_pose_1=goal_pose_1,
            goal_pose_2=goal_pose_2,
            goal_pose_3=goal_pose_3,
            agent_start_pos=agent_start_pos,
            success_threshold=success_threshold,
            leave_threshold=leave_threshold,
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

        # Action marker is drawn onto the cached image used for video output
        # (so eval videos show it) but NOT onto the obs returned to the
        # policy.
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
