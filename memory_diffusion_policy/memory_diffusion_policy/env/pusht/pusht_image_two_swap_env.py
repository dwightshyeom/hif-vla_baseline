from gym import spaces
from memory_diffusion_policy.env.pusht.pusht_two_swap_env import PushTTwoSwapEnv
import numpy as np
import cv2


class PushTImageTwoSwapEnv(PushTTwoSwapEnv):
    """
    Image-observation wrapper for the Push-T swap task.

    Returns {'image': (3, H, W) float32 in [0,1], 'agent_pos': (2,)} instead of
    the 8D low-dim state. The action marker is NOT rendered into the
    observation image (we pass `render_action=False` through to the parent),
    but it IS drawn into `render_cache` so the video overlay still shows it.
    """
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
                 legacy=False,
                 block_cog=None,
                 damping=None,
                 render_size=96,
                 target_a_pose=None,
                 target_b_pose=None,
                 agent_start_pos=None,
                 blue_start_side='left',
                 waypoint_center=None,
                 waypoint_radius=20,
                 waypoint_color='Black'):
        super().__init__(
            legacy=legacy,
            block_cog=block_cog,
            damping=damping,
            render_action=False,
            render_size=render_size,
            reset_to_state=None,
            target_a_pose=target_a_pose,
            target_b_pose=target_b_pose,
            agent_start_pos=agent_start_pos,
            blue_start_side=blue_start_side,
            waypoint_center=waypoint_center,
            waypoint_radius=waypoint_radius,
            waypoint_color=waypoint_color,
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

        # Draw action marker only on the cached image used for video rendering.
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
