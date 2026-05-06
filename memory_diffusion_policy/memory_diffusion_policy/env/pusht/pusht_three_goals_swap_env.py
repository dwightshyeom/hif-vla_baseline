"""
Push-T "three-goals swap" environment.

Three GREEN target zones are at fixed positions arranged in a triangle (the
same layout as PushTKeypointsThreeGoalsEnv). Two T-blocks (BLUE and RED)
spawn perfectly aligned inside two of the three targets, chosen at random
per episode. The third target is initially empty.

The objective is to SWAP the two blocks so that BLUE ends up at RED's
starting target and RED ends up at BLUE's starting target. The third target
is the only place where a block can be temporarily parked while the other
moves through, so any valid solution must use this 3-step protocol:

    Phase 0 (initial)
        BLUE at b_start, RED at r_start, empty target at e_idx.
    Phase 1 (one block has been parked in the empty slot)
        Either BLUE or RED is the "first mover" (chosen freely by the
        policy) and is now overlapping the empty target.
    Phase 2 (the other block has reached its swap target)
        The "second mover" has reached the first mover's original start.
    Phase 3 (success — first mover has reached its swap target)
        The first mover has moved from the empty slot to the second
        mover's original start. Both blocks are swapped.

The phase machine is driven by per-block coverage of each of the three
targets. ``success_threshold`` (default 0.9) gates "reached"; a much lower
``leave_threshold`` (default 0.3) gates "left", to allow the operator /
policy to nudge around a target without spuriously triggering failure.

Reward / done:
    reward = 1.0 on success, 0.0 otherwise.
    done   = success OR failed.

Failure conditions:
    - After phase 0, the wrong block ever overlaps the empty/intermediate
      target. (Only the first mover is allowed to use it; the second mover
      must move directly start→target without parking.)
    - After phase 2, the second mover leaves its target (cov drops below
      leave_threshold). This catches the second mover bouncing off the
      target and going elsewhere.

Observation (8D, low-dim base):
    [agent_x, agent_y,
     blue_x, blue_y, blue_theta,
     red_x,  red_y,  red_theta]

Goal positions are fixed. The per-episode randomness is purely in
which-block-spawns-where; the policy infers the swap intent from the
visible block colors and target positions.
"""

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


class PushTThreeGoalsSwapEnv(PushTEnv):
    """
    Three-goals swap variant. See module docstring for full semantics.
    """

    def __init__(
        self,
        legacy: bool = False,
        block_cog=None,
        damping=None,
        render_action: bool = True,
        render_size: int = 96,
        reset_to_state: Optional[np.ndarray] = None,
        goal_pose_1=None,
        goal_pose_2=None,
        goal_pose_3=None,
        agent_start_pos=None,
        success_threshold: float = 0.9,
        leave_threshold: float = 0.3,
    ):
        # Default targets: large equilateral triangle, identical to
        # PushTKeypointsThreeGoalsEnv defaults so the visual layout matches.
        if goal_pose_1 is None:
            goal_pose_1 = np.array([160.0, 360.0, np.pi / 4])   # bottom-left
        if goal_pose_2 is None:
            goal_pose_2 = np.array([256.0, 152.0, 0.0])         # top
        if goal_pose_3 is None:
            goal_pose_3 = np.array([352.0, 360.0, -np.pi / 4])  # bottom-right
        if agent_start_pos is None:
            # Neutral spawn near the bottom edge so the operator can reach
            # any of the three targets quickly during teleop.
            agent_start_pos = np.array([256.0, 460.0])

        self.goal_pose_1 = np.asarray(goal_pose_1, dtype=np.float64)
        self.goal_pose_2 = np.asarray(goal_pose_2, dtype=np.float64)
        self.goal_pose_3 = np.asarray(goal_pose_3, dtype=np.float64)
        self.agent_start_pos = np.asarray(agent_start_pos, dtype=np.float64)

        # Per-episode assignment (filled in reset()).
        # All indices below are 0/1/2 indexing into the ordered triple
        # (goal_pose_1, goal_pose_2, goal_pose_3).
        self.empty_idx = None
        self.blue_start_idx = None
        self.red_start_idx = None
        self.blue_target_idx = None
        self.red_target_idx = None
        # Initial poses (set at reset)
        self.blue_start_pose = None
        self.red_start_pose = None

        # Phase machine + tracking flags.
        self.phase = 0
        self.first_mover = None  # 'blue' | 'red' | None
        self.failed = False
        # "reached" flips True when cov > success_threshold for the first time.
        # "left"    flips True after reached when cov < leave_threshold.
        # Indexed by (color, goal_idx) where goal_idx ∈ {0, 1, 2}.
        self._reached = {
            ('blue', 0): False, ('blue', 1): False, ('blue', 2): False,
            ('red', 0): False,  ('red', 1): False,  ('red', 2): False,
        }
        self._left = {k: False for k in self._reached}

        self._success_threshold = float(success_threshold)
        self._leave_threshold = float(leave_threshold)

        # Initialise the parent. We reuse the underlying machinery (pymunk
        # space, walls, agent kinematics, control loop) but disable goal
        # randomisation since we have three fixed targets.
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
        # 8D obs: agent (x,y) + blue (x,y,theta) + red (x,y,theta)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            high=np.array([ws, ws, ws, ws, np.pi * 2,
                           ws, ws, np.pi * 2], dtype=np.float64),
            shape=(8,),
            dtype=np.float64,
        )

        # Override parent's success_threshold (used by some shared code)
        # so any inherited helper that references it still does the right
        # thing, even though the swap logic uses the explicit value above.
        self.success_threshold = self._success_threshold

    # ------------------------------------------------------------------
    # Per-episode setup
    # ------------------------------------------------------------------
    def _sample_swap_assignment(self):
        """Pick which target is empty and which colored block starts where.

        Uses self.np_random for reproducibility (seed is set via env.seed()).
        Returns (empty_idx, blue_start_idx, red_start_idx).
        """
        rng = self.np_random
        # rng may be either numpy.random.RandomState (older gym) or
        # numpy.random.Generator (newer); handle both.
        if hasattr(rng, 'integers'):
            empty_idx = int(rng.integers(0, 3))
            blue_first = bool(rng.integers(0, 2))
        else:
            empty_idx = int(rng.randint(0, 3))
            blue_first = bool(rng.randint(0, 2))

        # The two non-empty target indices.
        non_empty = [i for i in range(3) if i != empty_idx]
        # blue takes the first / second of the two non-empty slots.
        if blue_first:
            blue_start_idx, red_start_idx = non_empty[0], non_empty[1]
        else:
            blue_start_idx, red_start_idx = non_empty[1], non_empty[0]
        return empty_idx, blue_start_idx, red_start_idx

    def _goal_pose(self, idx: int) -> np.ndarray:
        """Return the (3,) pose array for goal index ∈ {0, 1, 2}."""
        return [self.goal_pose_1, self.goal_pose_2, self.goal_pose_3][idx]

    # ------------------------------------------------------------------
    # Setup / reset
    # ------------------------------------------------------------------
    def _setup(self):
        # Re-seed np_random from self._seed at the start of every reset so
        # that the swap-role assignment (and any other np_random-driven
        # randomness in this env) is fully deterministic per seed: calling
        # reset() repeatedly with the same _seed always produces the same
        # initial state. Without this, np_random's Generator state advances
        # across calls and consecutive resets diverge.
        if self._seed is not None:
            self.np_random = np.random.default_rng(self._seed)

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

        # Sample empty/blue/red assignment for THIS episode.
        empty_idx, blue_start_idx, red_start_idx = self._sample_swap_assignment()
        self.empty_idx = empty_idx
        self.blue_start_idx = blue_start_idx
        self.red_start_idx = red_start_idx
        # Swap-target is "the other block's start". Both targets are also
        # one of the three fixed goal indices.
        self.blue_target_idx = red_start_idx
        self.red_target_idx = blue_start_idx

        # Cache the actual poses (also useful for downstream consumers).
        self.blue_start_pose = self._goal_pose(blue_start_idx).copy()
        self.red_start_pose = self._goal_pose(red_start_idx).copy()

        # Agent (kinematic puck).
        self.agent = self.add_circle(tuple(self.agent_start_pos), 15)

        # Two T-blocks aligned with their starting (green) targets.
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
        # _get_goal_pose_body() that need a shape template. Both blocks
        # share the T-template, so either body's shape list works.
        self.block = self.blue_block

        # All three targets are rendered the same green color (fixed).
        self.goal_color = pygame.Color('LightGreen')
        # The parent's single goal_pose is unused but kept for any inherited
        # helpers that reference it.
        self.goal_pose = self._goal_pose(blue_start_idx).copy()

        # Collision tracking
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        # Use the configured success threshold for both inherited usage and
        # phase-machine logic.
        self.success_threshold = self._success_threshold

    def reset(self):
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.blue_block.center_of_gravity = self.block_cog
            self.red_block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        # Reset phase-machine state.
        self.phase = 0
        self.first_mover = None
        self.failed = False
        for k in self._reached:
            self._reached[k] = False
            self._left[k] = False

        # Default initial state: blocks aligned with their start targets,
        # agent at neutral spawn.
        if self.reset_to_state is not None:
            state = np.asarray(self.reset_to_state, dtype=np.float64)
        else:
            state = np.array([
                self.agent_start_pos[0], self.agent_start_pos[1],
                self.blue_start_pose[0], self.blue_start_pose[1], self.blue_start_pose[2],
                self.red_start_pose[0],  self.red_start_pose[1],  self.red_start_pose[2],
            ], dtype=np.float64)

        self._set_state(state)
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

        self.space.step(1.0 / self.sim_hz)

    def _get_obs(self):
        obs = np.array(
            tuple(self.agent.position)
            + tuple(self.blue_block.position) + (self.blue_block.angle % (2 * np.pi),)
            + tuple(self.red_block.position) + (self.red_block.angle % (2 * np.pi),),
            dtype=np.float64,
        )
        return obs

    # ------------------------------------------------------------------
    # Coverage helpers
    # ------------------------------------------------------------------
    def _coverage(self, block_body, goal_idx):
        """Coverage of `block_body` inside the green target at `goal_idx`."""
        block_geom = pymunk_to_shapely(block_body, self.blue_block.shapes)
        goal_body = self._get_goal_pose_body(self._goal_pose(goal_idx))
        goal_geom = pymunk_to_shapely(goal_body, self.blue_block.shapes)
        if goal_geom.area <= 0:
            return 0.0
        return float(block_geom.intersection(goal_geom).area / goal_geom.area)

    def _all_coverages(self):
        """Returns a dict {(color, goal_idx): coverage_float}."""
        out = {}
        for goal_idx in range(3):
            out[('blue', goal_idx)] = self._coverage(self.blue_block, goal_idx)
            out[('red', goal_idx)] = self._coverage(self.red_block, goal_idx)
        return out

    def _swap_coverages(self):
        """Coverage of each block at its swap target. Used for the video overlay."""
        return (self._coverage(self.blue_block, self.blue_target_idx),
                self._coverage(self.red_block, self.red_target_idx))

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))

        cov = self._all_coverages()
        cov_blue, cov_red = self._swap_coverages()

        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'blue_block_pose': np.array(list(self.blue_block.position) + [self.blue_block.angle]),
            'red_block_pose': np.array(list(self.red_block.position) + [self.red_block.angle]),
            'goal_1_pose': self.goal_pose_1.copy(),
            'goal_2_pose': self.goal_pose_2.copy(),
            'goal_3_pose': self.goal_pose_3.copy(),
            'empty_idx': int(self.empty_idx) if self.empty_idx is not None else -1,
            'blue_start_idx': int(self.blue_start_idx) if self.blue_start_idx is not None else -1,
            'red_start_idx': int(self.red_start_idx) if self.red_start_idx is not None else -1,
            'blue_target_idx': int(self.blue_target_idx) if self.blue_target_idx is not None else -1,
            'red_target_idx': int(self.red_target_idx) if self.red_target_idx is not None else -1,
            'blue_start_pose': self.blue_start_pose.copy() if self.blue_start_pose is not None else np.zeros(3),
            'red_start_pose': self.red_start_pose.copy() if self.red_start_pose is not None else np.zeros(3),
            'cov_blue_in_target': cov_blue,
            'cov_red_in_target': cov_red,
            'cov_blue_at_g1': cov[('blue', 0)],
            'cov_blue_at_g2': cov[('blue', 1)],
            'cov_blue_at_g3': cov[('blue', 2)],
            'cov_red_at_g1': cov[('red', 0)],
            'cov_red_at_g2': cov[('red', 1)],
            'cov_red_at_g3': cov[('red', 2)],
            'phase': int(self.phase),
            'first_mover': self.first_mover if self.first_mover else 'none',
            'n_contacts': n_contact_points_per_step,
            'failed': bool(self.failed),
        }
        return info

    # ------------------------------------------------------------------
    # Phase machine + step
    # ------------------------------------------------------------------
    def _update_reach_left_flags(self, cov):
        """Toggle _reached / _left flags based on the current coverages."""
        st_thr = self._success_threshold
        lv_thr = self._leave_threshold
        for key, c in cov.items():
            if not self._reached[key] and c > st_thr:
                self._reached[key] = True
            if self._reached[key] and not self._left[key] and c < lv_thr:
                self._left[key] = True

    def _advance_phase_machine(self, cov):
        """Phase transitions + failure detection.

        Phase 0 → 1: one block reaches the empty target. The block that does
                      so first becomes the first_mover. (If both happen on
                      the same step, the one with higher coverage wins.)
        Phase 1 → 2: the *other* block reaches the first mover's start
                      target.
        Phase 2 → 3: the first mover reaches the *other* block's start
                      target. Phase 3 == success.

        Failure (≥ phase 1):
            second_mover ever covers the empty target (phase-1+ rule:
            second mover must move start→target directly; never park).
        Failure (≥ phase 2):
            second_mover leaves its target (cov drops below leave_threshold).
        """
        if self.failed or self.phase >= 3:
            return

        e = self.empty_idx
        bs = self.blue_start_idx
        rs = self.red_start_idx
        st = self._success_threshold
        lv = self._leave_threshold

        if self.phase == 0:
            blue_in_empty = cov[('blue', e)] > st
            red_in_empty = cov[('red', e)] > st
            if blue_in_empty and red_in_empty:
                # Tie-break: whichever has higher coverage at empty.
                if cov[('blue', e)] >= cov[('red', e)]:
                    self.first_mover = 'blue'
                else:
                    self.first_mover = 'red'
                self.phase = 1
            elif blue_in_empty:
                self.first_mover = 'blue'
                self.phase = 1
            elif red_in_empty:
                self.first_mover = 'red'
                self.phase = 1

        # All subsequent phases use first_mover/second_mover indexing.
        if self.first_mover == 'blue':
            sm = 'red'                 # second mover = red
            sm_target_idx = self.red_target_idx     # = bs
            fm_target_idx = self.blue_target_idx    # = rs
        elif self.first_mover == 'red':
            sm = 'blue'                # second mover = blue
            sm_target_idx = self.blue_target_idx    # = rs
            fm_target_idx = self.red_target_idx     # = bs
        else:
            sm = None
            sm_target_idx = None
            fm_target_idx = None

        # Failure check: second mover overlapping the empty target.
        if self.phase >= 1 and sm is not None:
            if cov[(sm, e)] > st:
                self.failed = True
                return

        # Phase 1 → 2
        if self.phase == 1 and sm is not None:
            if cov[(sm, sm_target_idx)] > st:
                self.phase = 2

        # Phase 2 failure: second mover left its target.
        if self.phase >= 2 and sm is not None:
            if cov[(sm, sm_target_idx)] < lv:
                self.failed = True
                return

        # Phase 2 → 3 (success)
        if self.phase == 2 and self.first_mover is not None:
            if cov[(self.first_mover, fm_target_idx)] > st:
                self.phase = 3
                return

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

        cov = self._all_coverages()
        self._update_reach_left_flags(cov)
        self._advance_phase_machine(cov)

        success = (self.phase >= 3) and (not self.failed)
        reward = 1.0 if success else 0.0
        done = bool(success or self.failed)

        observation = self._get_obs()
        info = self._get_info()
        info['success'] = success

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

        # Draw all three green targets (always same color, regardless of phase).
        for goal_pose in (self.goal_pose_1, self.goal_pose_2, self.goal_pose_3):
            goal_body = self._get_goal_pose_body(goal_pose)
            for shape in self.blue_block.shapes:
                goal_points = [
                    pymunk.pygame_util.to_pygame(
                        goal_body.local_to_world(v), draw_options.surface)
                    for v in shape.get_vertices()
                ]
                goal_points += [goal_points[0]]
                pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent + both T-blocks. space.debug_draw picks up per-shape colors.
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
