from typing import Dict, Optional
import numpy as np
import pygame
import pymunk
import shapely.geometry as sg
from gym import spaces
from memory_diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from memory_diffusion_policy.env.pusht.pusht_env import pymunk_to_shapely


class PushTKeypointsThreeGoalsEnv(PushTKeypointsEnv):
    """
    PushT environment with three goals that can be completed in any order.
    The task requires visiting all three goal positions.
    Each goal must be reached only once with the required overlap threshold.
    All goals are active from the start and can be completed in any order.
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
            goal_pose_1=None,
            goal_pose_2=None,
            goal_pose_3=None,
            goal_pos_variation=0.0,
            goal_rot_variation=0.0,
            include_goal_keypoints=False,
            local_keypoint_map: Dict[str, np.ndarray]=None, 
            color_map: Optional[Dict[str, np.ndarray]]=None):
        """
        Args:
            goal_pose_1: First goal pose [x, y, theta]. Default: [160, 360, np.pi/4] (bottom-left)
            goal_pose_2: Second goal pose [x, y, theta]. Default: [256, 152, 0] (top-center)
            goal_pose_3: Third goal pose [x, y, theta]. Default: [352, 360, -np.pi/4] (bottom-right)
            goal_pos_variation: Max positional perturbation in pixels (applied to x, y independently). Default: 0.0
            goal_rot_variation: Max rotational perturbation in radians (applied to theta). Default: 0.0
            include_goal_keypoints: If True, append the 3 goal T-block keypoints (54 dims) to obs.
                This makes goal positions observable to the policy when goals vary.
            
        The three goals form a large equilateral triangle:
            - Goal 2 at the top
            - Goals 1 and 3 at the bottom (left and right)
            - Well-separated for challenging navigation
            
        When goal_pos_variation > 0 or goal_rot_variation > 0, goal poses are randomly
        perturbed on each reset() using the seeded RNG. Rejection sampling ensures:
            - All T-blocks stay within the workspace bounds
            - No two goal T-blocks overlap
        """
        # Set default three goals forming a large equilateral triangle
        # Triangle centered at (256, 256) with side length ~240 pixels
        if goal_pose_1 is None:
            goal_pose_1 = np.array([160, 360, np.pi/4])  # Bottom-left
        if goal_pose_2 is None:
            goal_pose_2 = np.array([256, 152, 0])  # Top-center
        if goal_pose_3 is None:
            goal_pose_3 = np.array([352, 360, -np.pi/4])  # Bottom-right
        
        # Store default (nominal) goal poses for variation sampling
        self._default_goal_pose_1 = np.array(goal_pose_1)
        self._default_goal_pose_2 = np.array(goal_pose_2)
        self._default_goal_pose_3 = np.array(goal_pose_3)
        
        self.goal_pos_variation = float(goal_pos_variation)
        self.goal_rot_variation = float(goal_rot_variation)
        self.include_goal_keypoints = bool(include_goal_keypoints)
        
        self.goal_pose_1 = np.array(goal_pose_1)
        self.goal_pose_2 = np.array(goal_pose_2)
        self.goal_pose_3 = np.array(goal_pose_3)
        
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
            goal_pose=goal_pose_1,
            randomize_goal=False,  # We have fixed goals
            local_keypoint_map=local_keypoint_map,
            color_map=color_map
        )
        
        # Track which goals have been visited
        self.goal_1_reached = False
        self.goal_2_reached = False
        self.goal_3_reached = False
        
        # Track which goals have been left after reaching
        self.goal_1_left = False
        self.goal_2_left = False
        self.goal_3_left = False
        
        # Track if task failed due to revisiting an already-achieved goal
        self.failed = False
        
        # Threshold for determining when a goal is truly "left" (much lower than success_threshold)
        # This allows nudging around the goal without triggering failure
        self.leave_threshold = 0.3  # Coverage must drop below 30% to be considered "left"
        
        # Colors for rendering three goals (all same green as official code)
        # Defer color creation until pygame is initialized (in _render_frame)
        self.goal_color = None
        
        # Cache goal keypoints to ensure consistency across timesteps
        # For fixed goals, compute once during init
        goal_1_pose_map = {'block': self.goal_pose_1}
        goal_1_kp_map = self.kp_manager.get_keypoints_global(
            pose_map=goal_1_pose_map, is_obj=False)
        self._cached_goal_1_keypoints = goal_1_kp_map['block'].copy()
        
        goal_2_pose_map = {'block': self.goal_pose_2}
        goal_2_kp_map = self.kp_manager.get_keypoints_global(
            pose_map=goal_2_pose_map, is_obj=False)
        self._cached_goal_2_keypoints = goal_2_kp_map['block'].copy()
        
        goal_3_pose_map = {'block': self.goal_pose_3}
        goal_3_kp_map = self.kp_manager.get_keypoints_global(
            pose_map=goal_3_pose_map, is_obj=False)
        self._cached_goal_3_keypoints = goal_3_kp_map['block'].copy()

        # Override observation space when goal keypoints are included
        if self.include_goal_keypoints:
            # 74 obs dims (20 base + 54 goal kps) + 74 mask dims = 148
            Do = 74
            Dobs = Do * 2
            ws = self.window_size
            low = np.zeros((Dobs,), dtype=np.float64)
            high = np.full_like(low, ws)
            high[Do:] = 1.  # mask range 0-1
            self.observation_space = spaces.Box(
                low=low, high=high, shape=low.shape, dtype=np.float64
            )
        
    def _update_goal_keypoints_cache(self):
        """Recompute cached goal keypoints from current goal poses."""
        for i, pose in enumerate([self.goal_pose_1, self.goal_pose_2, self.goal_pose_3], 1):
            pose_map = {'block': pose}
            kp_map = self.kp_manager.get_keypoints_global(
                pose_map=pose_map, is_obj=False)
            setattr(self, f'_cached_goal_{i}_keypoints', kp_map['block'].copy())

    def _get_obs(self):
        """
        Override parent _get_obs to optionally include goal keypoints.
        
        When include_goal_keypoints=True:
            obs = [block_kps(18), agent_pos(2), goal1_kps(18), goal2_kps(18), goal3_kps(18)]
            mask = [block_mask(18), agent_mask(2), ones(54)]
            Total: 74 + 74 = 148
        When include_goal_keypoints=False:
            Same as parent: [block_kps(18), agent_pos(2)] + mask = 40
        """
        obs = super()._get_obs()
        
        if not self.include_goal_keypoints:
            return obs
        
        # Split parent obs into data and mask
        Do = obs.shape[0] // 2
        obs_data = obs[:Do]
        obs_mask = obs[Do:]
        
        # Append goal keypoints (always fully visible)
        goal_kps = np.concatenate([
            self._cached_goal_1_keypoints.flatten(),
            self._cached_goal_2_keypoints.flatten(),
            self._cached_goal_3_keypoints.flatten(),
        ])  # (54,)
        goal_mask = np.ones(54, dtype=obs_data.dtype)
        
        obs_data = np.concatenate([obs_data, goal_kps])
        obs_mask = np.concatenate([obs_mask, goal_mask])
        
        return np.concatenate([obs_data, obs_mask])
    
    def _sample_varied_goals(self):
        """
        Sample three goal poses with random perturbation from default poses.
        Uses rejection sampling to guarantee:
        - All T-blocks are fully within workspace bounds
        - No two T-blocks overlap
        
        Returns:
            list of 3 np.ndarray: varied goal poses [x, y, theta]
        """
        rng = getattr(self, 'np_random', np.random.default_rng())
        bounds_box = sg.box(10, 10, 502, 502)
        max_attempts = 1000
        defaults = [self._default_goal_pose_1, self._default_goal_pose_2, self._default_goal_pose_3]
        
        for _ in range(max_attempts):
            poses = []
            for default_pose in defaults:
                dx = rng.uniform(-self.goal_pos_variation, self.goal_pos_variation)
                dy = rng.uniform(-self.goal_pos_variation, self.goal_pos_variation)
                dtheta = rng.uniform(-self.goal_rot_variation, self.goal_rot_variation)
                poses.append(np.array([
                    default_pose[0] + dx,
                    default_pose[1] + dy,
                    default_pose[2] + dtheta
                ]))
            
            # Check bounds: all T-blocks fully inside workspace
            geoms = []
            all_valid = True
            for pose in poses:
                body = self._get_goal_pose_body(pose)
                geom = pymunk_to_shapely(body, self.block.shapes)
                if not bounds_box.contains(geom):
                    all_valid = False
                    break
                geoms.append(geom)
            if not all_valid:
                continue
            
            # Check pairwise non-overlap
            overlap = False
            for i in range(len(geoms)):
                for j in range(i + 1, len(geoms)):
                    if geoms[i].intersection(geoms[j]).area > 1.0:
                        overlap = True
                        break
                if overlap:
                    break
            if not overlap:
                return poses
        
        print(f"Warning: Could not find valid varied goals after {max_attempts} attempts "
              f"(pos_var={self.goal_pos_variation}, rot_var={self.goal_rot_variation}). "
              f"Using default poses.")
        return [d.copy() for d in defaults]
    
    def reset(self):
        """Reset the environment and goal tracking."""
        super().reset()
        self.goal_1_reached = False
        self.goal_2_reached = False
        self.goal_3_reached = False
        self.goal_1_left = False
        self.goal_2_left = False
        self.goal_3_left = False
        self.failed = False
        
        # Apply goal variation if configured
        if self.goal_pos_variation > 0 or self.goal_rot_variation > 0:
            varied_poses = self._sample_varied_goals()
            self.goal_pose_1 = varied_poses[0]
            self.goal_pose_2 = varied_poses[1]
            self.goal_pose_3 = varied_poses[2]
        else:
            # Reset to defaults (in case they were previously varied)
            self.goal_pose_1 = self._default_goal_pose_1.copy()
            self.goal_pose_2 = self._default_goal_pose_2.copy()
            self.goal_pose_3 = self._default_goal_pose_3.copy()
        
        # Re-cache goal keypoints for the (possibly varied) poses
        self._update_goal_keypoints_cache()
        
        self.goal_pose = self.goal_pose_1.copy()
        
        # Recompute obs after goal variation to ensure goal keypoints are correct
        return self._get_obs()
    
    def _get_goal_keypoints(self):
        """
        Override to return keypoints for the current active goal.
        All three goal keypoints are accessible via _get_info().
        """
        goal_pose_map = {'block': self.goal_pose}
        goal_kp_map = self.kp_manager.get_keypoints_global(
            pose_map=goal_pose_map, is_obj=False)
        return goal_kp_map['block']
    
    def _get_goal_1_keypoints(self):
        """Get cached keypoints for goal 1."""
        return self._cached_goal_1_keypoints.copy()
    
    def _get_goal_2_keypoints(self):
        """Get cached keypoints for goal 2."""
        return self._cached_goal_2_keypoints.copy()
    
    def _get_goal_3_keypoints(self):
        """Get cached keypoints for goal 3."""
        return self._cached_goal_3_keypoints.copy()
    
    def _get_info(self):
        """
        Override parent's _get_info to add three-goal information.
        
        Returns:
            dict: Info dictionary with all three goal keypoints and progress tracking
        """
        info = super()._get_info()
        
        # Add keypoints for all three goals (cached for consistency)
        info['goal_1_keypoint'] = self._get_goal_1_keypoints()
        info['goal_2_keypoint'] = self._get_goal_2_keypoints()
        info['goal_3_keypoint'] = self._get_goal_3_keypoints()
        
        # Add goal poses
        info['goal_1_pose'] = self.goal_pose_1.copy()
        info['goal_2_pose'] = self.goal_pose_2.copy()
        info['goal_3_pose'] = self.goal_pose_3.copy()
        
        # Add progress tracking
        info['goal_1_reached'] = self.goal_1_reached
        info['goal_2_reached'] = self.goal_2_reached
        info['goal_3_reached'] = self.goal_3_reached
        
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
        
        # Coverage for goal 3
        goal_3_body = self._get_goal_pose_body(self.goal_pose_3)
        goal_3_geom = pymunk_to_shapely(goal_3_body, self.block.shapes)
        intersection_3 = goal_3_geom.intersection(block_geom).area
        goal_3_area = goal_3_geom.area
        info['goal_3_coverage'] = intersection_3 / goal_3_area if goal_3_area > 0 else 0.0
        
        return info
    
    def step(self, action):
        """
        Step the environment with strict three-goal logic.
        
        Success conditions:
        - All three goals are reached (in any order)
        - No already-achieved goal is revisited before completing all goals
        
        Failure condition:
        - Revisiting an already-achieved goal before all goals are completed
        
        Reward:
        - 1.0 if success (all three goals reached without revisiting)
        - 0.0 otherwise (including failure or incomplete)
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
        
        # Check goal 3 coverage
        goal_3_body = self._get_goal_pose_body(self.goal_pose_3)
        goal_3_geom = pymunk_to_shapely(goal_3_body, self.block.shapes)
        intersection_3 = goal_3_geom.intersection(block_geom).area
        goal_3_area = goal_3_geom.area
        coverage_3 = intersection_3 / goal_3_area if goal_3_area > 0 else 0.0
        
        # Update goal progress for newly reached goals (only if not failed)
        if not self.failed:
            if not self.goal_1_reached and coverage_1 > self.success_threshold:
                self.goal_1_reached = True
            
            if not self.goal_2_reached and coverage_2 > self.success_threshold:
                self.goal_2_reached = True
            
            if not self.goal_3_reached and coverage_3 > self.success_threshold:
                self.goal_3_reached = True
        
        # Track when goals are left (coverage drops significantly below leave_threshold after being reached)
        # Use a much lower threshold than success_threshold to allow nudging around the goal
        if self.goal_1_reached and not self.goal_1_left and coverage_1 < self.leave_threshold:
            self.goal_1_left = True
        
        if self.goal_2_reached and not self.goal_2_left and coverage_2 < self.leave_threshold:
            self.goal_2_left = True
        
        if self.goal_3_reached and not self.goal_3_left and coverage_3 < self.leave_threshold:
            self.goal_3_left = True
        
        # Check for failure: revisiting an already-achieved goal after leaving it
        # Only check if not all goals are reached yet
        num_goals_reached = int(self.goal_1_reached) + int(self.goal_2_reached) + int(self.goal_3_reached)
        
        if num_goals_reached < 3 and not self.failed:
            # Check if revisiting goal 1 after it was reached and left
            if self.goal_1_left and coverage_1 > self.success_threshold:
                self.failed = True
            # Check if revisiting goal 2 after it was reached and left
            elif self.goal_2_left and coverage_2 > self.success_threshold:
                self.failed = True
            # Check if revisiting goal 3 after it was reached and left
            elif self.goal_3_left and coverage_3 > self.success_threshold:
                self.failed = True
        
        # Recount goals after potential updates
        num_goals_reached = int(self.goal_1_reached) + int(self.goal_2_reached) + int(self.goal_3_reached)
        
        # Binary reward: 1 for success, 0 otherwise
        success = (num_goals_reached == 3) and not self.failed
        reward = 1.0 if success else 0.0
        
        # Episode ends on success or failure
        done = success or self.failed
        
        observation = self._get_obs()
        info = self._get_info()
        info['success'] = success
        info['failed'] = self.failed
        
        return observation, reward, done, info
    
    def _render_frame(self, mode):
        """Override to render all three goal poses."""
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        from diffusion_policy.env.pusht.pymunk_override import DrawOptions
        draw_options = DrawOptions(canvas)

        # Initialize colors if not done yet
        if self.goal_color is None:
            self.goal_color = pygame.Color('LightGreen')

        # Draw goal 1 (always green - no color change on completion)
        goal_1_body = self._get_goal_pose_body(self.goal_pose_1)
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(
                goal_1_body.local_to_world(v), draw_options.surface) 
                for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw goal 2 (always green - no color change on completion)
        goal_2_body = self._get_goal_pose_body(self.goal_pose_2)
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(
                goal_2_body.local_to_world(v), draw_options.surface) 
                for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw goal 3 (always green - no color change on completion)
        goal_3_body = self._get_goal_pose_body(self.goal_pose_3)
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(
                goal_3_body.local_to_world(v), draw_options.surface) 
                for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

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
