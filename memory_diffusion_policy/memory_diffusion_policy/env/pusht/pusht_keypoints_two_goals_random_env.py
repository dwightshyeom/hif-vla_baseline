from typing import Dict, Optional
import numpy as np
import pygame
import pymunk
from memory_diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv


class PushTKeypointsTwoGoalsRandomEnv(PushTKeypointsEnv):
    """
    PushT environment with two RANDOMIZED goals that can be completed in any order.
    The task requires visiting both goal positions.
    Each goal must be reached only once with the required overlap threshold.
    Both goals are active from the start and can be completed in any order.
    
    Unlike PushTKeypointsTwoGoalsEnv, goal positions are randomized on each reset.
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
            local_keypoint_map: Dict[str, np.ndarray]=None, 
            color_map: Optional[Dict[str, np.ndarray]]=None):
        """
        Args:
            All standard PushT environment arguments.
            Goal poses are randomized on each reset instead of being fixed.
        """
        # Initialize with placeholder goals (will be randomized in reset)
        self.goal_pose_1 = np.array([200, 256, np.pi/4])
        self.goal_pose_2 = np.array([312, 256, -np.pi/4])
        
        # Initialize parent with first goal
        super().__init__(
            legacy=legacy,
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            keypoint_visible_rate=keypoint_visible_rate,
            agent_keypoints=agent_keypoints,
            draw_keypoints=draw_keypoints,
            reset_to_state=reset_to_state,
            render_action=render_action,
            goal_pose=self.goal_pose_1,
            randomize_goal=False,  # We handle randomization ourselves
            local_keypoint_map=local_keypoint_map,
            color_map=color_map
        )
        
        # Track which goals have been visited
        self.goal_1_reached = False
        self.goal_2_reached = False
        self.current_goal_idx = 1  # Start with goal 1
        
        # Colors for rendering two goals (both same green as official code)
        # Defer color creation until pygame is initialized (in _render_frame)
        self.goal_color_1 = None
        self.goal_color_2 = None
        
        # Cache goal keypoints to ensure consistency across timesteps
        self._cached_goal_1_keypoints = None
        self._cached_goal_2_keypoints = None
    
    def _sample_two_random_goal_poses(self, block_pos):
        """
        Sample two random goal poses that are separated enough from each other
        and from the initial block position.
        Uses the same bounds as the base environment: [140, 372]
        
        Args:
            block_pos: Initial block position [x, y, angle]
        
        Returns:
            tuple: (goal_pose_1, goal_pose_2) where each is [x, y, theta]
        """
        rs = np.random.RandomState(seed=self._seed)
        
        # T-block is ~120x120 pixels
        # For non-overlapping T-blocks at different orientations, centers need ~150 pixels
        # For ensuring block needs to move, use ~120 pixels (one T-block width)
        min_goal_separation = 150  # Minimum distance between two goal centers (non-overlapping)
        min_block_separation = 120  # Minimum distance from initial block position (must move)
        max_attempts = 200  # Increased attempts for tighter constraints
        
        # Initialize with default values (fallback)
        x1, y1, theta1 = 200, 256, np.pi/4
        x2, y2, theta2 = 312, 256, -np.pi/4
        
        block_x, block_y = block_pos[0], block_pos[1]
        
        for attempt in range(max_attempts):
            # Sample first goal
            x1 = rs.randint(140, 372)
            y1 = rs.randint(140, 372)
            theta1 = rs.randn() * 2 * np.pi - np.pi
            
            # Sample second goal
            x2 = rs.randint(140, 372)
            y2 = rs.randint(140, 372)
            theta2 = rs.randn() * 2 * np.pi - np.pi
            
            # Check separation between two goals (ensure non-overlapping)
            dist_goals = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
            # Check separation from initial block position
            dist_block_goal1 = np.sqrt((x1 - block_x)**2 + (y1 - block_y)**2)
            dist_block_goal2 = np.sqrt((x2 - block_x)**2 + (y2 - block_y)**2)
            
            # All conditions must be satisfied
            if (dist_goals >= min_goal_separation and 
                dist_block_goal1 >= min_block_separation and 
                dist_block_goal2 >= min_block_separation):
                break
        
        goal_pose_1 = np.array([x1, y1, theta1])
        goal_pose_2 = np.array([x2, y2, theta2])
        
        return goal_pose_1, goal_pose_2
        
    def reset(self):
        """Reset the environment with randomized goal positions."""
        # First, we need to determine where the block will be initialized
        # Call parent reset first to initialize the block position
        obs = super().reset()
        
        # Get the current block position
        block_pos = np.array([self.block.position.x, self.block.position.y, self.block.angle])
        
        # Now sample goal positions that don't overlap with block or each other
        self.goal_pose_1, self.goal_pose_2 = self._sample_two_random_goal_poses(block_pos)
        self.goal_pose = self.goal_pose_1.copy()
        
        # Reset goal tracking
        self.goal_1_reached = False
        self.goal_2_reached = False
        self.current_goal_idx = 1
        
        # Cache goal keypoints once at reset to ensure consistency across timesteps
        goal_1_pose_map = {'block': self.goal_pose_1}
        goal_1_kp_map = self.kp_manager.get_keypoints_global(
            pose_map=goal_1_pose_map, is_obj=False)
        self._cached_goal_1_keypoints = goal_1_kp_map['block'].copy()
        
        goal_2_pose_map = {'block': self.goal_pose_2}
        goal_2_kp_map = self.kp_manager.get_keypoints_global(
            pose_map=goal_2_pose_map, is_obj=False)
        self._cached_goal_2_keypoints = goal_2_kp_map['block'].copy()
        
        return obs
    
    def _get_goal_keypoints(self):
        """
        Override to return keypoints for both goals.
        Returns keypoints for the current active goal for compatibility,
        but both are accessible via _get_info().
        """
        goal_pose_map = {'block': self.goal_pose}
        goal_kp_map = self.kp_manager.get_keypoints_global(
            pose_map=goal_pose_map, is_obj=False)
        return goal_kp_map['block']
    
    def _get_goal_1_keypoints(self):
        """Get keypoints for goal 1 from cache."""
        return self._cached_goal_1_keypoints
    
    def _get_goal_2_keypoints(self):
        """Get keypoints for goal 2 from cache."""
        return self._cached_goal_2_keypoints
    
    def _get_info(self):
        """
        Override parent's _get_info to add two-goal information.
        
        Returns:
            dict: Info dictionary with both goal keypoints and progress tracking
        """
        info = super()._get_info()
        
        # Add keypoints for both goals
        info['goal_1_keypoint'] = self._get_goal_1_keypoints()
        info['goal_2_keypoint'] = self._get_goal_2_keypoints()
        
        # Add goal poses
        info['goal_1_pose'] = self.goal_pose_1.copy()
        info['goal_2_pose'] = self.goal_pose_2.copy()
        
        # Add progress tracking
        info['goal_1_reached'] = self.goal_1_reached
        info['goal_2_reached'] = self.goal_2_reached
        info['current_goal_idx'] = self.current_goal_idx
        
        # Add coverage information
        from memory_diffusion_policy.env.pusht.pusht_env import pymunk_to_shapely
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)
        
        # Coverage for goal 1
        goal_1_body = self._get_goal_pose_body(self.goal_pose_1)
        goal_1_geom = pymunk_to_shapely(goal_1_body, self.block.shapes)
        intersection_1 = goal_1_geom.intersection(block_geom).area
        goal_1_area = goal_1_geom.area
        info['goal_1_coverage'] = intersection_1 / goal_1_area if goal_1_area > 0 else 0.0
        
        # Coverage for goal 2
        goal_2_body = self._get_goal_pose_body(self.goal_pose_2)
        goal_2_geom = pymunk_to_shapely(goal_2_body, self.block.shapes)
        intersection_2 = goal_2_geom.intersection(block_geom).area
        goal_2_area = goal_2_geom.area
        info['goal_2_coverage'] = intersection_2 / goal_2_area if goal_2_area > 0 else 0.0
        
        return info
    
    def step(self, action):
        """
        Step the environment with two-goal logic.
        
        Both goals can be completed in any order.
        - The block can overlap either goal first
        - Each goal needs to exceed success_threshold to be marked as reached
        - Each goal can only be reached once
        - The task is complete when both goals are reached
        """
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                acceleration = self.k_p * (action - self.agent.position) + \
                              self.k_v * (pymunk.vec2d.Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt
                self.space.step(dt)
        
        # Import here to avoid circular import
        from memory_diffusion_policy.env.pusht.pusht_env import pymunk_to_shapely
        
        # Get block geometry
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)
        
        # Check goal 1 coverage
        goal_1_body = self._get_goal_pose_body(self.goal_pose_1)
        goal_1_geom = pymunk_to_shapely(goal_1_body, self.block.shapes)
        intersection_1 = goal_1_geom.intersection(block_geom).area
        goal_1_area = goal_1_geom.area
        coverage_1 = intersection_1 / goal_1_area if goal_1_area > 0 else 0.0
        
        # Check goal 2 coverage
        goal_2_body = self._get_goal_pose_body(self.goal_pose_2)
        goal_2_geom = pymunk_to_shapely(goal_2_body, self.block.shapes)
        intersection_2 = goal_2_geom.intersection(block_geom).area
        goal_2_area = goal_2_geom.area
        coverage_2 = intersection_2 / goal_2_area if goal_2_area > 0 else 0.0
        
        # Update goal progress - both goals can be reached in any order
        if not self.goal_1_reached and coverage_1 > self.success_threshold:
            self.goal_1_reached = True
        
        if not self.goal_2_reached and coverage_2 > self.success_threshold:
            self.goal_2_reached = True
        
        # Compute reward based on progress
        num_goals_reached = int(self.goal_1_reached) + int(self.goal_2_reached)
        
        if num_goals_reached == 0:
            # No goals reached yet, reward based on best coverage
            best_coverage = max(coverage_1, coverage_2)
            reward = np.clip(best_coverage / self.success_threshold, 0, 1) * 0.5
            done = False
        elif num_goals_reached == 1:
            # One goal reached, reward based on the other goal's coverage
            if self.goal_1_reached:
                reward = 0.5 + np.clip(coverage_2 / self.success_threshold, 0, 1) * 0.5
            else:
                reward = 0.5 + np.clip(coverage_1 / self.success_threshold, 0, 1) * 0.5
            done = False
        else:
            # Both goals reached!
            reward = 1.0
            done = True
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, done, info
    
    def _render_frame(self, mode):
        """Override to render both goal poses."""
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()
        
        # Initialize colors lazily after pygame is initialized
        if self.goal_color_1 is None:
            self.goal_color_1 = pygame.Color('LightGreen')
            self.goal_color_2 = pygame.Color('LightGreen')

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        from diffusion_policy.env.pusht.pymunk_override import DrawOptions
        draw_options = DrawOptions(canvas)

        # Draw goal 1 (green if not reached, gray if reached)
        goal_1_body = self._get_goal_pose_body(self.goal_pose_1)
        color_1 = pygame.Color('Gray') if self.goal_1_reached else self.goal_color_1
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(
                goal_1_body.local_to_world(v), draw_options.surface) 
                for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, color_1, goal_points)

        # Draw goal 2 (green if not reached, gray if reached)
        goal_2_body = self._get_goal_pose_body(self.goal_pose_2)
        color_2 = pygame.Color('Gray') if self.goal_2_reached else self.goal_color_2
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(
                goal_2_body.local_to_world(v), draw_options.surface) 
                for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, color_2, goal_points)

        # Draw agent and block
        self.space.debug_draw(draw_options)
        
        # Draw keypoints if enabled
        if self.draw_keypoints and self.draw_kp_map is not None:
            self.kp_manager.draw_keypoints(
                canvas, self.draw_kp_map, radius=2)

        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

        # Convert to numpy array
        import cv2
        img = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        img = cv2.resize(img, (self.render_size, self.render_size))
        
        # Draw action marker if enabled
        if self.render_action and (self.latest_action is not None):
            action = np.array(self.latest_action)
            coord = (action / 512 * self.render_size).astype(np.int32)
            marker_size = int(8 / 96 * self.render_size)
            thickness = int(1 / 96 * self.render_size)
            cv2.drawMarker(img, coord,
                color=(255, 0, 0), markerType=cv2.MARKER_CROSS,
                markerSize=marker_size, thickness=thickness)
        
        return img
