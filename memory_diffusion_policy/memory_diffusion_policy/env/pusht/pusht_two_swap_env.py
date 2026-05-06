from typing import Optional

import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import cv2

from gym import spaces

from memory_diffusion_policy.env.pusht.pusht_env import PushTEnv, pymunk_to_shapely
from diffusion_policy.env.pusht.pymunk_override import DrawOptions


class PushTTwoSwapEnv(PushTEnv):
    """
    Push-T "swap" environment.

    Two T-blocks (Blue and Red) are spawned inside two same-colored target
    zones (A on the left, B on the right). The objective is to SWAP the
    blocks: whichever zone Blue starts in, it must end up in the OTHER
    zone, and vice versa for Red.

    The spawn assignment is controlled by `blue_start_side`:
        'left'  (default): Blue starts in Target A, Red starts in Target B
                           -> Blue must reach Target B, Red must reach Target A
        'right'          : Blue starts in Target B, Red starts in Target A
                           -> Blue must reach Target A, Red must reach Target B

    Use `set_blue_start_side('l'|'r'|'left'|'right')` to change the
    assignment between episodes (e.g. during interactive data collection).

    Observation (8D):
        [agent_x, agent_y,
         blue_x, blue_y, blue_theta,
         red_x,  red_y,  red_theta]

    Reward / done:
        coverage_blue = overlap(Blue, blue_target_pose) / area(target)
        coverage_red  = overlap(Red,  red_target_pose ) / area(target)
        reward = clip(0.5 * (coverage_blue + coverage_red) / threshold, 0, 1)
        done   = (coverage_blue > threshold) AND (coverage_red > threshold)
    """

    VALID_SIDES = ('left', 'right')

    def __init__(self,
                 legacy: bool = False,
                 block_cog=None,
                 damping=None,
                 render_action: bool = True,
                 render_size: int = 96,
                 reset_to_state: Optional[np.ndarray] = None,
                 target_a_pose=None,
                 target_b_pose=None,
                 agent_start_pos=None,
                 blue_start_side: str = 'left',
                 waypoint_center=None,
                 waypoint_radius: int = 20,
                 waypoint_color: str = 'Black'):
        # Default target poses: A on the left, B on the right, both upright.
        if target_a_pose is None:
            target_a_pose = np.array([145.0, 256.0, np.pi/6])
        if target_b_pose is None:
            target_b_pose = np.array([367.0, 256.0, -np.pi/6])
        if agent_start_pos is None:
            # Neutral spot near the bottom of the workspace, between the two
            # target zones, so the operator can reach either block easily.
            agent_start_pos = np.array([256.0, 460.0])

        self.target_a_pose = np.asarray(target_a_pose, dtype=np.float64)
        self.target_b_pose = np.asarray(target_b_pose, dtype=np.float64)
        self.agent_start_pos = np.asarray(agent_start_pos, dtype=np.float64)

        # Visual-only waypoint marker at the center of the workspace.
        # This is a rendering overlay (no pymunk body, no collisions, not
        # written into observations) that gives the operator a stable
        # landmark to steer the agent toward during the neutral-passage
        # phase of the 5-step teleoperation protocol.
        if waypoint_center is None:
            # Center of the 512x512 workspace.
            waypoint_center = (256.0, 300.0)
        self.waypoint_center = np.asarray(waypoint_center, dtype=np.float64)
        self.waypoint_radius = int(waypoint_radius)
        self.waypoint_color_name = str(waypoint_color)
        # Deferred until pygame is initialised (in _setup or _render_frame).
        self.waypoint_color = None

        # Which zone Blue spawns in for the NEXT reset(). Red takes the
        # other zone; the swap target for each block is therefore the zone
        # it did NOT start in. Call `set_blue_start_side()` to change it
        # between episodes during data collection.
        self.blue_start_side = self._normalize_side(blue_start_side)

        # Populated by reset() based on blue_start_side.
        self.blue_start_pose = None
        self.red_start_pose = None
        self.blue_target_pose = None
        self.red_target_pose = None

        # Initialise the parent. We reuse most of the underlying machinery
        # (pymunk space, walls, agent kinematics, control loop, etc.) but
        # disable goal randomisation since we have two fixed target zones.
        super().__init__(
            legacy=legacy,
            block_cog=block_cog,
            damping=damping,
            render_action=render_action,
            render_size=render_size,
            reset_to_state=reset_to_state,
            goal_pose=None,
            randomize_goal=False,
        )

        ws = self.window_size
        # 8D observation: agent (x,y) + blue (x,y,theta) + red (x,y,theta)
        self.observation_space = spaces.Box(
            low=np.array([0, 0,
                          0, 0, 0,
                          0, 0, 0], dtype=np.float64),
            high=np.array([ws, ws,
                           ws, ws, np.pi * 2,
                           ws, ws, np.pi * 2], dtype=np.float64),
            shape=(8,),
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Spawn-side selection helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_side(side: str) -> str:
        s = str(side).strip().lower()
        if s in ('l', 'left'):
            return 'left'
        if s in ('r', 'right'):
            return 'right'
        raise ValueError(
            f"Invalid blue_start_side={side!r}. Expected one of "
            f"'l', 'r', 'left', 'right'."
        )

    def set_blue_start_side(self, side: str):
        """
        Set which zone Blue spawns in on the NEXT reset().
        Accepts 'l', 'r', 'left', or 'right'. Red always takes the other zone.
        """
        self.blue_start_side = self._normalize_side(side)

    def _resolve_start_and_target_poses(self):
        """Return (blue_start, red_start, blue_target, red_target) based on blue_start_side."""
        if self.blue_start_side == 'left':
            blue_start = self.target_a_pose.copy()
            red_start = self.target_b_pose.copy()
        else:  # 'right'
            blue_start = self.target_b_pose.copy()
            red_start = self.target_a_pose.copy()
        # Swap targets: each block must reach the zone it didn't start in.
        blue_target = red_start.copy()
        red_target = blue_start.copy()
        return blue_start, red_start, blue_target, red_target

    # ------------------------------------------------------------------
    # Setup / reset
    # ------------------------------------------------------------------
    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()

        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        # Resolve spawn + swap-target poses for this episode based on
        # the currently selected blue_start_side.
        blue_start, red_start, blue_target, red_target = \
            self._resolve_start_and_target_poses()
        self.blue_start_pose = blue_start
        self.red_start_pose = red_start
        self.blue_target_pose = blue_target
        self.red_target_pose = red_target

        # Agent (kinematic puck). Distinct from the Blue T-block color.
        self.agent = self.add_circle(tuple(self.agent_start_pos), 15)

        # Two T-blocks spawned aligned with their starting zones.
        self.blue_block = self.add_tee(
            position=tuple(self.blue_start_pose[:2]),
            angle=float(self.blue_start_pose[2]),
            color='DodgerBlue4',
        )
        self.red_block = self.add_tee(
            position=tuple(self.red_start_pose[:2]),
            angle=float(self.red_start_pose[2]),
            color='FireBrick4',
        )

        # Backwards-compat alias used by parent helpers like
        # _get_goal_pose_body() that ask for "self.block.shapes" to
        # mint goal-pose geometry. Both T-blocks share the same shape
        # template, so either body's shape list works as the template.
        self.block = self.blue_block

        # Both target zones share the SAME color (per spec).
        self.goal_color = pygame.Color('LightGreen')
        # Red waypoint marker color (visual only).
        self.waypoint_color = pygame.Color(self.waypoint_color_name)
        # The parent's single goal_pose is unused but we leave it set so
        # that any inherited helpers don't crash if invoked.
        self.goal_pose = self.target_a_pose.copy()

        # Collision tracking
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.90  # 95% coverage per block

    def reset(self):
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.blue_block.center_of_gravity = self.block_cog
            self.red_block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        # Default initial state: Blue aligned with its chosen start zone,
        # Red aligned with the other zone, agent at neutral spawn.
        if self.reset_to_state is not None:
            state = np.asarray(self.reset_to_state, dtype=np.float64)
        else:
            state = np.array([
                self.agent_start_pos[0], self.agent_start_pos[1],
                self.blue_start_pose[0], self.blue_start_pose[1], self.blue_start_pose[2],
                self.red_start_pose[0],  self.red_start_pose[1],  self.red_start_pose[2],
            ], dtype=np.float64)

        self._set_state(state)
        # (seed is preserved for downstream randomisation if a subclass adds it)
        _ = seed
        return self._get_obs()

    # ------------------------------------------------------------------
    # State <-> simulator
    # ------------------------------------------------------------------
    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        agent_pos = state[0:2]
        blue_pos = state[2:4]
        blue_angle = state[4]
        red_pos = state[5:7]
        red_angle = state[7]

        self.agent.position = agent_pos

        if self.legacy:
            self.blue_block.position = blue_pos
            self.blue_block.angle = blue_angle
            self.red_block.position = red_pos
            self.red_block.angle = red_angle
        else:
            self.blue_block.angle = blue_angle
            self.blue_block.position = blue_pos
            self.red_block.angle = red_angle
            self.red_block.position = red_pos

        # Settle the simulator one tick so the new pose takes effect.
        self.space.step(1.0 / self.sim_hz)

    def _get_obs(self):
        obs = np.array(
            tuple(self.agent.position)
            + tuple(self.blue_block.position) + (self.blue_block.angle % (2 * np.pi),)
            + tuple(self.red_block.position) + (self.red_block.angle % (2 * np.pi),),
            dtype=np.float64,
        )
        return obs

    def _compute_swap_coverages(self):
        """Coverage of each block inside its SWAP target (zone it didn't start in)."""
        block_geom_blue = pymunk_to_shapely(self.blue_block, self.blue_block.shapes)
        block_geom_red = pymunk_to_shapely(self.red_block, self.red_block.shapes)

        blue_target_body = self._get_goal_pose_body(self.blue_target_pose)
        red_target_body = self._get_goal_pose_body(self.red_target_pose)
        blue_target_geom = pymunk_to_shapely(blue_target_body, self.blue_block.shapes)
        red_target_geom = pymunk_to_shapely(red_target_body, self.blue_block.shapes)

        cov_blue = (
            block_geom_blue.intersection(blue_target_geom).area / blue_target_geom.area
            if blue_target_geom.area > 0 else 0.0
        )
        cov_red = (
            block_geom_red.intersection(red_target_geom).area / red_target_geom.area
            if red_target_geom.area > 0 else 0.0
        )
        return float(cov_blue), float(cov_red)

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))

        cov_blue, cov_red = self._compute_swap_coverages()

        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'blue_block_pose': np.array(list(self.blue_block.position) + [self.blue_block.angle]),
            'red_block_pose': np.array(list(self.red_block.position) + [self.red_block.angle]),
            'target_a_pose': self.target_a_pose.copy(),  # left zone (static)
            'target_b_pose': self.target_b_pose.copy(),  # right zone (static)
            'blue_start_pose': self.blue_start_pose.copy(),
            'red_start_pose': self.red_start_pose.copy(),
            'blue_target_pose': self.blue_target_pose.copy(),  # where Blue must go
            'red_target_pose': self.red_target_pose.copy(),    # where Red  must go
            'blue_start_side': self.blue_start_side,           # 'left' or 'right'
            'coverage_blue_in_target': cov_blue,
            'coverage_red_in_target': cov_red,
            'n_contacts': n_contact_points_per_step,
            'waypoint_center': self.waypoint_center.copy(),
            'waypoint_radius': float(self.waypoint_radius),
        }
        return info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for _ in range(n_steps):
                acceleration = (
                    self.k_p * (action - self.agent.position)
                    + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                )
                self.agent.velocity += acceleration * dt
                self.space.step(dt)

        # ----- Reward: joint coverage of swapped targets -----
        cov_blue, cov_red = self._compute_swap_coverages()

        joint_coverage = 0.5 * (cov_blue + cov_red)
        reward = float(np.clip(joint_coverage / self.success_threshold, 0.0, 1.0))
        done = bool(
            (cov_blue > self.success_threshold)
            and (cov_red > self.success_threshold)
        )

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, done, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render_frame(self, mode):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # ----- Draw both target zones (same color) using the T template -----
        for target_pose in (self.target_a_pose, self.target_b_pose):
            goal_body = self._get_goal_pose_body(target_pose)
            for shape in self.blue_block.shapes:
                goal_points = [
                    pymunk.pygame_util.to_pygame(
                        goal_body.local_to_world(v), draw_options.surface)
                    for v in shape.get_vertices()
                ]
                goal_points += [goal_points[0]]
                pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # ----- Draw static red waypoint marker at workspace center ----------
        # Drawn BEFORE debug_draw so the agent / T-blocks appear on top of
        # it when they cross over; the marker is purely visual (no pymunk
        # body, no collisions, no contribution to observations / rewards).
        if self.waypoint_color is None:
            self.waypoint_color = pygame.Color(self.waypoint_color_name)
        pygame.draw.circle(
            canvas,
            self.waypoint_color,
            (int(self.waypoint_center[0]), int(self.waypoint_center[1])),
            self.waypoint_radius,
        )

        # ----- Draw agent + both T-blocks (colors set on shapes) -----
        # space.debug_draw will pick up Blue/Red shape colors set in add_tee.
        self.space.debug_draw(draw_options)

        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

        img = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        img = cv2.resize(img, (self.render_size, self.render_size))

        if self.render_action and (self.latest_action is not None):
            action = np.array(self.latest_action)
            coord = (action / 512 * self.render_size).astype(np.int32)
            marker_size = int(8 / 96 * self.render_size)
            thickness = int(1 / 96 * self.render_size)
            cv2.drawMarker(
                img, coord,
                color=(0, 0, 0), markerType=cv2.MARKER_CROSS,
                markerSize=marker_size, thickness=thickness,
            )
        return img
