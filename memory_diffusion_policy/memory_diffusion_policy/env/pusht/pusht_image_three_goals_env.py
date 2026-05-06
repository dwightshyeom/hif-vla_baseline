from gym import spaces
from memory_diffusion_policy.env.pusht.pusht_keypoints_three_goals_env import PushTKeypointsThreeGoalsEnv
import numpy as np
import cv2


class PushTImageThreeGoalsEnv(PushTKeypointsThreeGoalsEnv):
    """
    Image-based observation wrapper for the three-goals PushT environment.
    Returns {'image': (3, H, W), 'agent_pos': (2,)} observations
    instead of keypoint vectors.
    """
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
            legacy=False,
            block_cog=None,
            damping=None,
            render_size=96,
            goal_pose_1=None,
            goal_pose_2=None,
            goal_pose_3=None):
        super().__init__(
            legacy=legacy,
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            keypoint_visible_rate=1.0,
            agent_keypoints=False,
            draw_keypoints=False,
            reset_to_state=None,
            render_action=False,
            goal_pose_1=goal_pose_1,
            goal_pose_2=goal_pose_2,
            goal_pose_3=goal_pose_3,
            include_goal_keypoints=False
        )
        ws = self.window_size
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3, render_size, render_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=0,
                high=ws,
                shape=(2,),
                dtype=np.float32
            )
        })
        self.render_cache = None

    def _get_obs(self):
        img = self._render_frame(mode='rgb_array')

        agent_pos = np.array(self.agent.position)
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {
            'image': img_obs,
            'agent_pos': agent_pos,
        }

        # draw action marker on cached render
        if self.latest_action is not None:
            action = np.array(self.latest_action)
            coord = (action / 512 * 96).astype(np.int32)
            marker_size = int(8 / 96 * self.render_size)
            thickness = int(1 / 96 * self.render_size)
            cv2.drawMarker(img, coord,
                color=(255, 0, 0), markerType=cv2.MARKER_CROSS,
                markerSize=marker_size, thickness=thickness)
        self.render_cache = img

        return obs

    def render(self, mode):
        assert mode == 'rgb_array'

        if self.render_cache is None:
            self._get_obs()

        return self.render_cache
