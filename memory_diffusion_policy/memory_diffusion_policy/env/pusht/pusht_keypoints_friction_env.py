"""
2D Push-T friction task — a partial-observability memory benchmark.

Arena (pygame coords, +y grows downward):

    +-----------------------------+   y=5        (top wall)
    |        [Start area]         |   spawn strip (NOT rendered —
    |           (block)           |   region is used for block spawn
    +-----------------------------+   y=track_y_lo  only, no floor tint)
    |               |             |
    |               |             |   one lane has a hidden friction
    |   Left lane   |  Right lane |   gradient that ramps to "wall"
    |               |             |   strength by mid-track and stalls
    |               |             |   the block — agent must back out
    |               |             |   and try the other lane.
    +-----------------------------+   y=track_y_hi
    |                             |
    |        ┌─────────┐          |   ONLY the T-shaped goal silhouette
    |        │    T    │          |   is rendered (no floor / region rect).
    |        │    │    │          |   Success = block's T-shape overlaps
    |        └────┘                |   the goal T with coverage > threshold.
    +-----------------------------+   y=506      (bottom wall)

Direction of progress is DOWNWARD. The agent spawns above the block, so the
first push drives the block southwards into the track zone. The high-friction
lane gets exponentially harder until the block is stuck near mid-track; the
correct response is to push the block back up into the start strip and
re-attempt down the other lane (the policy must remember which lane it
already tried — the task is partially observable).

Ported from the 3D PushTUnknownFriction task in
``memorydp/envs3D/tasks/push_t_unknown_friction.py``.
"""
from typing import Dict, Optional
import math

import numpy as np
import pygame
import pymunk
from pymunk.vec2d import Vec2d

from memory_diffusion_policy.env.pusht.pusht_env import PushTEnv, pymunk_to_shapely
from memory_diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from memory_diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager


class PushTKeypointsFrictionEnv(PushTKeypointsEnv):
    # Tuning knobs for the T-block and the mouse-controlled agent. Smaller
    # than KCW defaults (30 / 15) so the block / agent fit comfortably in the
    # arena and are easier to steer with a mouse — also shrunk further from
    # the previous 20 / 10 so the (now larger) end area gives ample space to
    # rotate / align the T against the goal silhouette.
    DEFAULT_BLOCK_SCALE = 16
    DEFAULT_AGENT_RADIUS = 7

    def __init__(self,
            # --- zone layout (pixel coords in 512x512 workspace) ---
            # Track lengthened (190..370 instead of 200..320) so the
            # high-friction lane has more distance to ramp up before the
            # block stalls — gives the policy a clearly committed bad
            # commit before it has to back out.
            track_y_lo=190,
            track_y_hi=370,
            divider_x=256,
            # Start strip: wider (180 vs 130) and shifted up so its bottom
            # edge sits ~30 px above the track line — the T-block has space
            # to spawn entirely behind the line at any spawn angle.
            start_area_center=(256, 90),
            start_area_half=(180, 70),
            # End "area" is no longer drawn as a floor — only the T-goal
            # silhouette is rendered. These values are still used to
            # auto-position the goal T centered at end_area_center and to
            # report the region in info{} for downstream consumers.
            end_area_center=(256, 435),
            end_area_half=(220, 65),
            # --- goal T-shape ---
            goal_t_pose=None,                 # auto: centered in end area
            coverage_threshold=0.90,
            # --- block / agent geometry ---
            block_scale=None,
            agent_radius=None,
            # T is only slightly tilted at spawn so it points mostly down
            # into the track zone; this keeps spawns robust inside the
            # start strip at any block size.
            spawn_angle_range=(-np.pi / 6, np.pi / 6),
            # --- friction dynamics ---
            # The drag is applied to both the block AND the (kinematic) agent.
            # Damping the block alone does not stop it: the kinematic agent
            # still pushes the block positionally through pymunk's contact
            # solver regardless of the block's damped velocity. Dragging the
            # AGENT's velocity when its own position is in the high-friction
            # zone is what actually creates a hard wall — the agent simply
            # cannot advance fast enough to shove the block further.
            #
            # high_drag = 500 sets the agent's terminal velocity in the wall
            # zone to ~3 px/s even with the PD controller at max accel,
            # well under the stuck_eps threshold.
            low_drag=0.4,
            high_drag=500.0,
            # Front-load the ramp so drag reaches `high_drag` by this
            # fraction of the track instead of only at the very exit.
            # 0.5 = block stalls around mid-track on the bad lane.
            friction_full_at=0.5,
            friction_segments=16,
            # --- multi-trial / memory logic ---
            max_trials=2,
            stuck_threshold=40,
            stuck_eps=1.2,
            reset_grace_steps=30,
            return_phase_timeout=400,
            lane_crossing_fail=True,
            # --- debug render ---
            # When True, tints the high-friction lane in the render so you
            # can SEE which side is the bad one. MUST be False for actual
            # data collection / policy training (the policy would cheat).
            debug_show_friction=False,
            # --- obs ---
            include_memory_flags=True,
            # --- inherited PushTKeypointsEnv kwargs ---
            legacy=False,
            block_cog=None,
            damping=None,
            render_size=256,
            keypoint_visible_rate=1.0,
            agent_keypoints=False,
            draw_keypoints=False,
            reset_to_state=None,
            render_action=True,
            local_keypoint_map: Dict[str, np.ndarray] = None,
            color_map: Optional[Dict[str, np.ndarray]] = None,
        ):
        self.block_scale = int(
            block_scale if block_scale is not None else self.DEFAULT_BLOCK_SCALE)
        self.agent_radius = int(
            agent_radius if agent_radius is not None else self.DEFAULT_AGENT_RADIUS)
        self.spawn_angle_range = (float(spawn_angle_range[0]),
                                  float(spawn_angle_range[1]))

        self.track_y_lo = float(track_y_lo)
        self.track_y_hi = float(track_y_hi)
        self.divider_x = float(divider_x)
        self.start_area_center = np.array(start_area_center, dtype=np.float64)
        self.start_area_half = np.array(start_area_half, dtype=np.float64)
        self.end_area_center = np.array(end_area_center, dtype=np.float64)
        self.end_area_half = np.array(end_area_half, dtype=np.float64)

        # Auto-center the goal T inside the end area when not supplied.
        # The T's visual span (at angle 0) is 4*scale tall starting at
        # body.position.y (top of bar) — centering the visual puts
        # body.position.y = end_area_center.y - 2*scale.
        if goal_t_pose is None:
            goal_t_pose = np.array([
                float(end_area_center[0]),
                float(end_area_center[1]) - 2.0 * self.block_scale,
                0.0,
            ])
        self.goal_t_pose = np.array(goal_t_pose, dtype=np.float64)
        self.coverage_threshold = float(coverage_threshold)

        self.low_drag = float(low_drag)
        self.high_drag = float(high_drag)
        self.friction_full_at = float(np.clip(friction_full_at, 1e-3, 1.0))
        self.friction_segments = int(friction_segments)
        self.max_trials = int(max_trials)
        self.stuck_threshold = int(stuck_threshold)
        self.stuck_eps = float(stuck_eps)
        self.reset_grace_steps = int(reset_grace_steps)
        self.return_phase_timeout = int(return_phase_timeout)
        self.lane_crossing_fail = bool(lane_crossing_fail)
        self.debug_show_friction = bool(debug_show_friction)
        self.include_memory_flags = bool(include_memory_flags)

        # Episode state — filled in reset().
        self.high_friction_side = "left"
        self.current_trial = 0
        self._stuck_counter = 0
        self._prev_block_pos = None
        self._return_phase = False
        self._return_phase_steps = 0
        self._grace_remaining = 0
        self._block_entered_track = False
        self._block_past_track = False
        self._lane_sign = 0
        self._force_done = False
        self._is_success = False
        self._latest_coverage = 0.0

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
            goal_pose=None,            # inherited goal_pose is unused here
            randomize_goal=False,
            local_keypoint_map=local_keypoint_map,
            color_map=color_map,
        )

        # Patch obs space to include memory flags if enabled.
        if self.include_memory_flags:
            from gym import spaces
            ws = self.window_size
            parent_Dobs = self.observation_space.shape[0]
            parent_Do = parent_Dobs // 2
            new_Do = parent_Do + 2
            new_Dobs = new_Do * 2
            low = np.zeros((new_Dobs,), dtype=np.float64)
            high = np.full_like(low, ws)
            high[parent_Do:new_Do] = 1.0     # memory flags are in [0, 1]
            high[new_Do:] = 1.0              # mask
            self.observation_space = spaces.Box(
                low=low, high=high, shape=low.shape, dtype=np.float64)

    # ==================== keypoint manager (scale-aware) ====================
    @classmethod
    def genenerate_keypoint_manager_params(cls,
            block_scale=None, agent_radius=None):
        """Override to regenerate keypoints for our (smaller) block/agent."""
        block_scale = (block_scale if block_scale is not None
                       else cls.DEFAULT_BLOCK_SCALE)
        agent_radius = (agent_radius if agent_radius is not None
                        else cls.DEFAULT_AGENT_RADIUS)

        class _ScaledPushTEnv(PushTEnv):
            def add_tee(self, position, angle, scale=30, **kw):
                return PushTEnv.add_tee(
                    self, position, angle, scale=block_scale, **kw)

            def add_circle(self, position, radius):
                return PushTEnv.add_circle(self, position, agent_radius)

        env = _ScaledPushTEnv()
        kp_manager = PymunkKeypointManager.create_from_pusht_env(env)
        return kp_manager.kwargs

    # ==================== pymunk setup ====================
    # Linear friction: F_max = drag * FRICTION_FORCE_PER_DRAG.
    # PD peak force = k_p * |mouse_offset|, maxed when user yanks the
    # mouse all the way to the opposite edge (~500 px) → ~205k. Sized so
    # drag=500 (wall zone) gives F_max=250k, strictly above any possible
    # PD force so a maxed mouse yank can't break through the wall. At
    # mid-drag (~80, about 1/3 into the bad lane) F_max=40k, which matches
    # a typical 100-px-offset push (40k) — the block starts visibly
    # stalling around mid-track under normal play.
    FRICTION_FORCE_PER_DRAG = 500.0
    # Rotational friction = BASE_ROT_FRICTION (always present, kills
    # accidental spinning everywhere) + linear_F_max * ROT_FRICTION_RATIO
    # (extra torque resistance inside the high-drag zone so a stuck
    # block can't just rotate out).
    BASE_ROT_FRICTION = 12000.0
    # Ratio of rotational-friction torque cap to linear-friction force cap.
    # Physically, for a plate of characteristic size L resting under uniform
    # pressure, integrating μ·N·r over the footprint gives
    #   τ_max / F_max  ≈  L / 3
    # so for our T (L ≈ 4 × block_scale = 64 px) the right ratio is ~20.
    # Using 25 for a slightly conservative lock in the wall zone so the
    # block doesn't rotate when it's supposed to be stuck.
    ROT_FRICTION_RATIO = 25.0
    AGENT_MASS = 1.0
    # Snappier PD tuning for the dynamic agent. At mass=1:
    #   time constant ~ 1 / (ζ·ω_n) = 1 / (0.87 · sqrt(400)) ≈ 57 ms.
    # (Parent's default 100/20 gave ~200 ms — felt sluggish.)
    PD_K_P = 400.0
    PD_K_V = 35.0

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

        # Agent is DYNAMIC (not kinematic): required so that the block's
        # floor friction can actually resist the push through Newton's 3rd
        # law. Moment = inf so the agent never rotates.
        a_body = pymunk.Body(
            mass=self.AGENT_MASS, moment=float('inf'),
            body_type=pymunk.Body.DYNAMIC)
        a_body.position = (256, 400)
        a_shape = pymunk.Circle(a_body, self.agent_radius)
        a_shape.friction = 0.0
        a_shape.color = pygame.Color('RoyalBlue')
        self.space.add(a_body, a_shape)
        self.agent = a_body

        self.block = self.add_tee((256, 300), 0, scale=self.block_scale)
        for s in self.block.shapes:
            s.friction = 0.0  # agent slides cleanly against block edges

        # Floor friction on the block: a PivotJoint tying the block's CoG
        # to the static world with max_bias=0 (no positional snap-back,
        # only a velocity-opposing force). Anchored at CoG (not body's
        # (0,0) which is the bar top) so this joint constrains ONLY
        # linear velocity — rotation about CoG moves the anchor zero
        # distance, so it doesn't feed into this constraint. Rotational
        # friction is handled cleanly by the GearJoint below.
        cog = tuple(self.block.center_of_gravity)
        world_anchor = tuple(
            self.block.local_to_world(self.block.center_of_gravity))
        self.floor_friction = pymunk.PivotJoint(
            self.space.static_body, self.block, world_anchor, cog)
        self.floor_friction.max_bias = 0
        self.floor_friction.max_force = 0
        self.space.add(self.floor_friction)

        # Rotational floor friction via gear joint (max_bias=0, velocity
        # only) so the block doesn't spin freely in the high-friction zone.
        self.floor_rot_friction = pymunk.GearJoint(
            self.space.static_body, self.block, 0.0, 1.0)
        self.floor_rot_friction.max_bias = 0
        self.floor_rot_friction.max_force = 0
        self.space.add(self.floor_rot_friction)

        self.goal_color = pygame.Color('LightGreen')
        self.goal_pose = self.goal_t_pose.copy()   # kept for base-class compat

        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = self.coverage_threshold

        # Override parent's PD gains for a snappier dynamic agent.
        self.k_p = self.PD_K_P
        self.k_v = self.PD_K_V

    # ==================== reset ====================
    def reset(self):
        """Custom block/agent placement and per-episode trial-state reset."""
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        rs = np.random.RandomState(seed=seed)
        self.high_friction_side = "left" if rs.rand() < 0.5 else "right"

        if self.reset_to_state is not None:
            state = np.array(self.reset_to_state)
            self._set_state(state)
        else:
            sx, sy = self.start_area_center
            hx, hy = self.start_area_half
            s = self.block_scale
            # pymunk's `body.local_to_world(v) = position + rotate(v, angle)`
            # does NOT subtract center_of_gravity, so vertex extent from
            # body.position is the distance to the furthest local vertex.
            # Stem tip (s/2, 4s): dist ≈ sqrt(16.25) * s ≈ 4.03 * s downward.
            # Upward extent (bar corner (-2s, s) at spawn_angle_range edge
            # ≈π/6) max |-y| = 2s*cos(π/6) + s*sin(π/6) = 2.23 * s.
            r_down = 4.03 * s
            lo_ang, hi_ang = self.spawn_angle_range
            ang_edge = max(abs(lo_ang), abs(hi_ang))
            r_up = (2.0 * s * math.cos(ang_edge)
                    + 1.0 * s * math.sin(ang_edge))
            r_side = 4.03 * s
            pad = 4.0

            bx_min = max(sx - hx + r_side, r_side + 6)
            bx_max = min(sx + hx - r_side, self.window_size - r_side - 6)
            # Require the stem tip stays above the start/track line and the
            # upward-pointing bar corner stays below the top wall (no agent
            # overlap check yet — that's applied to ay below).
            by_min = max(sy - hy + r_up, 5 + r_up + pad)
            by_max = min(sy + hy - r_down, self.track_y_lo - r_down - pad)

            def _sample():
                bx = (rs.uniform(bx_min, bx_max)
                      if bx_max > bx_min else float(sx))
                by = (rs.uniform(by_min, by_max)
                      if by_max > by_min
                      else float(min(sy, self.track_y_lo - r_down - pad)))
                block_angle = rs.uniform(lo_ang, hi_ang)
                ax = bx + rs.uniform(-25, 25)
                # Agent above the block, with enough clearance so no overlap
                # at the worst upward-tilted block corner.
                ay = max(by - (r_up + self.agent_radius + 4),
                         self.agent_radius + 8)
                return np.array([ax, ay, bx, by, block_angle])

            # Rejection sampling: place block, verify every block vertex
            # stays strictly above the start/track line and below the top
            # wall.
            for _ in range(30):
                state = _sample()
                self._set_state(state)
                # Zero any residual motion from _set_state's internal step.
                self.block.velocity = Vec2d(0, 0)
                self.block.angular_velocity = 0.0
                self.agent.velocity = Vec2d(0, 0)
                if self._block_bbox_within_start_strip():
                    break

        # Reset trial / return-phase state.
        self.current_trial = 0
        self._stuck_counter = 0
        self._prev_block_pos = None
        self._return_phase = False
        self._return_phase_steps = 0
        self._grace_remaining = self.reset_grace_steps
        self._block_entered_track = False
        self._block_past_track = False
        self._lane_sign = 0
        self._force_done = False
        self._is_success = False
        self._latest_coverage = 0.0

        # Populate the goal-keypoint cache (parent _get_info reads this).
        self._cached_goal_keypoint = self._compute_goal_t_keypoints()

        return self._get_obs()

    def _compute_goal_t_keypoints(self):
        goal_body = self._get_goal_pose_body(self.goal_t_pose)
        tf = self.kp_manager.get_tf_img_obj(goal_body)
        kp_local = self.kp_manager.local_keypoint_map['block']
        return np.asarray(tf(kp_local))

    def _block_world_bbox(self):
        xs, ys = [], []
        for shape in self.block.shapes:
            for v in shape.get_vertices():
                wp = self.block.local_to_world(v)
                xs.append(wp[0]); ys.append(wp[1])
        return min(xs), min(ys), max(xs), max(ys)

    def _block_bbox_within_start_strip(self, pad=2.0):
        mn_x, mn_y, mx_x, mx_y = self._block_world_bbox()
        return (mx_y < self.track_y_lo - pad
                and mn_y > 5 + pad
                and mn_x > 5 + pad
                and mx_x < self.window_size - 5 - pad)

    # ==================== step ====================
    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz

        if action is not None:
            self.latest_action = action
            action_vec = Vec2d(float(action[0]), float(action[1]))
            for _ in range(n_steps):
                # Force-based PD on the dynamic agent. Critically damped
                # at k_v = 2*sqrt(k_p*m) for AGENT_MASS=1 / k_p=100.
                err = action_vec - self.agent.position
                vel = self.agent.velocity
                fx = self.k_p * err.x - self.k_v * vel.x
                fy = self.k_p * err.y - self.k_v * vel.y
                self.agent.force = (fx, fy)
                # Update the block's floor-friction strength from the
                # drag gradient at its CURRENT position.
                self._update_block_floor_friction()
                self.space.step(dt)

        self._update_trial_state()

        observation = self._get_obs()
        done = self._is_success or self._force_done
        reward = 1.0 if self._is_success else 0.0
        info = self._get_info()
        return observation, reward, done, info

    # ==================== friction field ====================
    # Real Coulomb floor friction for a top-down view: pymunk has no
    # native floor concept (no gravity / normal force), so we model the
    # floor as a PivotJoint between the block and the static world, with
    # `max_bias=0` (no positional correction — so the joint won't pull
    # the block back toward its spawn) and `max_force` updated each
    # substep from the drag gradient.
    #
    # The joint applies a force opposing the block's velocity, capped at
    # `max_force`. Exactly Coulomb friction: any push above that force
    # slides the block; any push below it holds the block in place.
    # Because the agent is DYNAMIC, its contact-force reaction against a
    # heavily-fricted block naturally decelerates it — the "stuck" feel
    # is real physics, not a positional hack, and there's no jitter.
    def _update_block_floor_friction(self):
        drag = self._compute_drag_at(self.block.position)
        f_max = drag * self.FRICTION_FORCE_PER_DRAG
        self.floor_friction.max_force = f_max
        self.floor_rot_friction.max_force = (
            self.BASE_ROT_FRICTION + f_max * self.ROT_FRICTION_RATIO)

    def _compute_drag_at(self, pos):
        bx, by = float(pos[0]), float(pos[1])
        # Start / end strips: low drag everywhere.
        if by < self.track_y_lo or by > self.track_y_hi:
            return self.low_drag
        lane = "left" if bx < self.divider_x else "right"
        if lane != self.high_friction_side:
            return self.low_drag
        # High-friction lane: drag ramps geometrically from low_drag at the
        # entry to high_drag by `friction_full_at` of the way through, then
        # stays at high_drag for the rest of the lane. The compression
        # front-loads the curve so the block stalls around mid-track
        # instead of only at the very exit.
        t_raw = ((by - self.track_y_lo)
                 / max(self.track_y_hi - self.track_y_lo, 1e-6))
        t_compressed = min(t_raw / self.friction_full_at, 1.0)
        n = self.friction_segments
        seg = int(np.clip(t_compressed * n, 0, n - 1))
        t = seg / max(n - 1, 1)
        ratio = self.high_drag / max(self.low_drag, 1e-8)
        return self.low_drag * (ratio ** t)

    # ==================== coverage ====================
    def _compute_coverage(self):
        goal_body = self._get_goal_pose_body(self.goal_t_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)
        if goal_geom.area <= 0:
            return 0.0
        return goal_geom.intersection(block_geom).area / goal_geom.area

    # ==================== trial state machine ====================
    def _update_trial_state(self):
        if self._force_done or self._is_success:
            return

        block_pos = np.array(self.block.position, dtype=np.float64)
        self._latest_coverage = self._compute_coverage()

        if self._grace_remaining > 0:
            self._grace_remaining -= 1
            self._prev_block_pos = block_pos.copy()
            return

        # Success: block's T overlaps the goal T with enough coverage.
        if (self._latest_coverage > self.coverage_threshold
                and not self._return_phase):
            self._is_success = True
            return

        if self._return_phase:
            # Return complete when block is back above the start/track line.
            if block_pos[1] < self.track_y_lo:
                self._complete_return_phase()
                return
            self._return_phase_steps += 1
            if self._return_phase_steps >= self.return_phase_timeout:
                self._force_done = True
            return

        # ----- Normal push phase -----

        # Lane commit: first time the block center crosses into the track zone.
        if not self._block_entered_track and block_pos[1] > self.track_y_lo:
            self._block_entered_track = True
            self._lane_sign = -1 if block_pos[0] < self.divider_x else 1

        # Once the block exits the track at the end side, lane restrictions
        # and stuck detection are off — user may freely maneuver in end area.
        if (self._block_entered_track and not self._block_past_track
                and block_pos[1] > self.track_y_hi):
            self._block_past_track = True

        # Lane-crossing fail (only while inside the track zone).
        if (self.lane_crossing_fail and self._lane_sign != 0
                and self.track_y_lo <= block_pos[1] <= self.track_y_hi):
            cur_sign = -1 if block_pos[0] < self.divider_x else 1
            if cur_sign != self._lane_sign:
                self._force_done = True
                return

        # Backward push: block re-entered start strip before reaching end.
        if self._block_entered_track and block_pos[1] < self.track_y_lo:
            self._block_entered_track = False
            self._block_past_track = False
            self._lane_sign = 0
            if self.current_trial + 1 < self.max_trials:
                self.current_trial += 1
                self._stuck_counter = 0
                self._prev_block_pos = None
                self._grace_remaining = self.reset_grace_steps
            else:
                self._force_done = True
            return

        # Stuck detection — only while block is inside the track zone.
        # While in end area user may be slowly aligning; don't burn a trial.
        in_track = (self.track_y_lo <= block_pos[1] <= self.track_y_hi)
        if (self._block_entered_track and in_track
                and self._prev_block_pos is not None):
            disp = np.linalg.norm(block_pos[:2] - self._prev_block_pos[:2])
            if disp < self.stuck_eps:
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0
            if self._stuck_counter >= self.stuck_threshold:
                if self.current_trial + 1 < self.max_trials:
                    self._enter_return_phase()
                else:
                    self._force_done = True
        self._prev_block_pos = block_pos.copy()

    def _enter_return_phase(self):
        self._return_phase = True
        self._return_phase_steps = 0
        self._stuck_counter = 0
        self._prev_block_pos = None

    def _complete_return_phase(self):
        self._return_phase = False
        self.current_trial += 1
        self._stuck_counter = 0
        self._prev_block_pos = None
        self._block_entered_track = False
        self._block_past_track = False
        self._lane_sign = 0
        self._grace_remaining = self.reset_grace_steps

    # ==================== obs / info ====================
    def _get_obs(self):
        base = super()._get_obs()
        if not self.include_memory_flags:
            return base
        Do = base.shape[0] // 2
        data = base[:Do]
        mask = base[Do:]
        flags = np.array([
            self.current_trial / max(self.max_trials, 1),
            float(self._return_phase),
        ], dtype=data.dtype)
        flag_mask = np.ones(2, dtype=data.dtype)
        data = np.concatenate([data, flags])
        mask = np.concatenate([mask, flag_mask])
        return np.concatenate([data, mask])

    def _get_info(self):
        info = super()._get_info()
        info['current_trial'] = int(self.current_trial)
        info['return_phase'] = bool(self._return_phase)
        info['high_friction_side'] = self.high_friction_side
        info['is_success'] = bool(self._is_success)
        info['is_done'] = bool(self._is_success or self._force_done)
        info['end_area_center'] = self.end_area_center.copy()
        info['end_area_half'] = self.end_area_half.copy()
        info['start_area_center'] = self.start_area_center.copy()
        info['start_area_half'] = self.start_area_half.copy()
        info['goal_t_pose'] = self.goal_t_pose.copy()
        info['coverage'] = float(self._latest_coverage)
        info['coverage_threshold'] = float(self.coverage_threshold)
        info['lane_sign'] = int(self._lane_sign)
        info['block_past_track'] = bool(self._block_past_track)
        return info

    # ==================== render ====================
    def _render_frame(self, mode):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        from diffusion_policy.env.pusht.pymunk_override import DrawOptions
        draw_options = DrawOptions(canvas)

        ws = self.window_size

        # Neither the start-area floor nor the end-area floor is drawn —
        # the block still spawns inside the start strip and the goal T is
        # still auto-centered in the end strip, but both regions are
        # invisible in the render. start_area_* / end_area_* remain
        # exposed via info{} for downstream consumers.

        # Debug-only: tint the high-friction lane with a drag-gradient so
        # you can SEE both which side is bad AND how drag ramps up along
        # it. MUST be off for training.
        if self.debug_show_friction:
            lane_x0 = 5 if self.high_friction_side == "left" else int(self.divider_x)
            lane_x1 = int(self.divider_x) if self.high_friction_side == "left" else ws - 5
            # Draw the lane in horizontal slices, each shaded by its drag.
            n = 24
            for i in range(n):
                y0 = self.track_y_lo + i * (self.track_y_hi - self.track_y_lo) / n
                y1 = self.track_y_lo + (i + 1) * (self.track_y_hi - self.track_y_lo) / n
                probe_x = (lane_x0 + lane_x1) / 2
                d = self._compute_drag_at(np.array([probe_x, (y0 + y1) / 2]))
                # Drag ~ [low_drag, high_drag] → shade light to dark red.
                log_lo = math.log(max(self.low_drag, 1e-3))
                log_hi = math.log(max(self.high_drag, self.low_drag + 1e-3))
                t = (math.log(max(d, 1e-3)) - log_lo) / max(log_hi - log_lo, 1e-6)
                t = max(0.0, min(1.0, t))
                r = max(0, min(255, int(255 - 35 * t)))
                g = max(0, min(255, int(225 - 160 * t)))
                b = max(0, min(255, int(225 - 160 * t)))
                slice_rect = pygame.Rect(
                    lane_x0, int(y0), lane_x1 - lane_x0, int(y1 - y0) + 1)
                pygame.draw.rect(canvas, pygame.Color(r, g, b), slice_rect)
            # Block body-center dot — the x that actually picks the lane.
            bx = int(self.block.position[0])
            by = int(self.block.position[1])
            pygame.draw.circle(canvas, pygame.Color(0, 180, 220), (bx, by), 3)

        # Goal T-shape (the only end-zone marker).
        goal_body = self._get_goal_pose_body(self.goal_t_pose)
        for shape in self.block.shapes:
            pts = [pymunk.pygame_util.to_pygame(
                goal_body.local_to_world(v), draw_options.surface)
                for v in shape.get_vertices()]
            pts += [pts[0]]
            pygame.draw.polygon(canvas, pygame.Color(186, 238, 186), pts)
            pygame.draw.polygon(
                canvas, pygame.Color(60, 170, 60), pts, width=2)

        # Track divider + boundary lines — both lanes visually identical.
        pygame.draw.line(canvas, pygame.Color(140, 140, 140),
                         (int(self.divider_x), int(self.track_y_lo)),
                         (int(self.divider_x), int(self.track_y_hi)), 2)
        pygame.draw.line(canvas, pygame.Color(140, 140, 140),
                         (5, int(self.track_y_lo)),
                         (ws - 5, int(self.track_y_lo)), 2)
        pygame.draw.line(canvas, pygame.Color(140, 140, 140),
                         (5, int(self.track_y_hi)),
                         (ws - 5, int(self.track_y_hi)), 2)

        # Agent & block. Only draw shapes — skip constraints (otherwise the
        # floor-friction PivotJoint/GearJoint render as two purple anchor
        # dots: one tracking the block's CoG, one fixed at the block's
        # setup-time CoG on the divider center line).
        draw_options.flags = draw_options.DRAW_SHAPES
        self.space.debug_draw(draw_options)

        if self.draw_keypoints and self.draw_kp_map is not None:
            self.kp_manager.draw_keypoints(canvas, self.draw_kp_map, radius=2)

        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

        import cv2
        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)),
                           axes=(1, 0, 2))
        # INTER_AREA preserves thin features (track divider, boundary lines)
        # across the 512 -> render_size reduction; INTER_LINEAR (cv2 default)
        # blurs them below the visible threshold at render_size=96.
        img = cv2.resize(img, (self.render_size, self.render_size),
                         interpolation=cv2.INTER_AREA)

        if self.render_action and (self.latest_action is not None):
            action = np.array(self.latest_action)
            coord = (action / ws * self.render_size).astype(np.int32)
            marker_size = int(8 / 96 * self.render_size)
            thickness = int(1 / 96 * self.render_size)
            cv2.drawMarker(img, coord, color=(255, 0, 0),
                           markerType=cv2.MARKER_CROSS,
                           markerSize=marker_size, thickness=thickness)
        return img
