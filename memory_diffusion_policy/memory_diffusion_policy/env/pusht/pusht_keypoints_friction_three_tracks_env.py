"""
Push-T friction task — THREE-TRACK variant (memory benchmark, harder).

Layout (extending pusht_keypoints_friction_env.py):

    +---------------------------------+   y=5
    |  ░░ red rectangle (start) ░░    |   spawn strip — translucent red box
    +---------------------------------+   y=track_y_lo
    | lane 0  | lane 1  | lane 2      |   3 vertical lanes of EQUAL width.
    |         |         |             |   The two dividers are at fixed
    |         |         |             |   x-positions = 1/3 and 2/3 of the
    |         |         |             |   interior arena, so every lane has
    |         |         |             |   the same width. TWO lanes carry
    |         |         |             |   an IDENTICAL high-friction
    |         |         |             |   gradient (un-pushable past the
    |         |         |             |   ramp midpoint); ONE is normal.
    +---------------------------------+   y=track_y_hi
    |        Goal-T silhouette        |   end strip — same length as start
    +---------------------------------+   y=506

What's randomized per seed:
    * normal_lane_idx ∈ {0, 1, 2}  — which lane is the only viable path.
    * block + agent spawn pose      — within the start rectangle (rejection
      sampling against bbox extent).

What's FIXED across episodes (intentional, per spec):
    * The two divider x-coordinates (so all 3 lanes have the SAME width).
    * Track y-extents and start/end strip y-extents (start and end strips
      are the same length).

Trial machine — extends the 2-track parent with a strict mid-step
"return-to-original-location" success criterion:

    1) Block enters track (block_y > track_y_lo): commit to the lane.
    2) If block stalls in a high-friction lane OR the operator/policy
       backs the block out of the track, enter ``return_phase``.
    3) ``return_phase`` ONLY completes when the block is back inside the
       start strip *AND* its pose matches the original spawn pose
       (translation + rotation within configurable thresholds). Only
       then does ``current_trial`` increment.
    4) If ``return_phase`` exceeds ``return_phase_timeout`` steps without
       the block reaching the original location, the episode is forced
       done (failure).
    5) Episode succeeds when the block's coverage of the goal-T exceeds
       ``coverage_threshold`` while not in return_phase.

Therefore success = (find normal lane AND make goal coverage), with the
implicit pre-requisite of having completed every back-out via a clean
return-to-spawn — exactly the user-requested "must push the T back to the
original location at each midstep" criterion.

Determinism: same env._seed always yields the same initial layout
(normal_lane_idx + block / agent spawn). reset() re-seeds its
RandomState from self._seed every call.
"""
from typing import Dict, Optional
import math

import numpy as np
import pygame
import pymunk
from pymunk.vec2d import Vec2d

from memory_diffusion_policy.env.pusht.pusht_keypoints_friction_env import (
    PushTKeypointsFrictionEnv,
)


class PushTKeypointsFrictionThreeTracksEnv(PushTKeypointsFrictionEnv):
    """
    Three-track friction memory benchmark (equal-width lanes).
    """

    def __init__(
        self,
        # ---- Geometry: equal-length start / end strips, same-width lanes ----
        track_y_lo: float = 175.0,
        track_y_hi: float = 336.0,
        # Dividers default to 1/3 and 2/3 of the interior arena (5..506).
        # Set explicitly here so all three lanes have the SAME width.
        divider_1_x: float = 172.0,
        divider_2_x: float = 339.0,
        # Start / end strips: identical dimensions. The translucent red
        # rectangle drawn at start_area is also the actual spawn region,
        # so changing ``start_area_half`` here both shrinks the visual
        # box AND tightens where the T-block resets each episode.
        # Hard floor (with default block_scale=16): half_x > ~65,
        # half_y > ~51, otherwise the rejection sampler can't fit a
        # rotated T-block inside the box.
        start_area_center=(256.0, 90.0),
        start_area_half=(130.0, 65.0),
        end_area_center=(256.0, 421.0),
        end_area_half=(130.0, 65.0),
        # ---- Return-to-spawn success criterion ----
        # The block must come back to within `return_pos_tol` pixels and
        # `return_ang_tol` radians of its initial spawn pose for the
        # current trial to count and the next trial to begin.
        return_pos_tol: float = 25.0,
        return_ang_tol: float = 0.35,
        # ---- Visual ----
        # Translucent rectangle drawn over the start area showing where
        # the T-block resets at episode start.
        start_box_color=(220, 50, 50),
        start_box_alpha: int = 70,
        max_trials: int = 3,
        # ---- Success criterion ----
        # Goal-T coverage fraction the block must reach for success.
        # 0.90 = the block must overlap ≥ 90 % of the goal-T silhouette.
        coverage_threshold: float = 0.90,
        # All other kwargs are forwarded to the 2-track parent unchanged.
        **kwargs,
    ):
        # Geometry constants (frozen across episodes, not seed-driven).
        self.divider_1_x = float(divider_1_x)
        self.divider_2_x = float(divider_2_x)
        self.normal_lane_idx = 1  # placeholder; overwritten in reset()

        # Return-criterion thresholds.
        self.return_pos_tol = float(return_pos_tol)
        self.return_ang_tol = float(return_ang_tol)

        # Translucent-box rendering knobs.
        self.start_box_color = tuple(int(c) for c in start_box_color)
        self.start_box_alpha = int(start_box_alpha)

        # Replaces parent's 2-way `_lane_sign` with a 3-way index.
        self._committed_lane_idx = -1

        # Initial block pose snapshot (filled in reset()).
        self._initial_block_pos = np.zeros(2, dtype=np.float64)
        self._initial_block_angle = 0.0
        self._returns_completed = 0  # successful trial-end returns to spawn
        # Lanes the operator has committed to in PRIOR (already-ended) trials
        # of this episode. Re-committing to any of these in a later trial
        # immediately fails — the policy must explore a new lane each try.
        self._used_lanes = set()
        # Lane the operator was committed to when the current return_phase
        # started. During return_phase, dipping the block back into the
        # track is allowed ONLY in this same lane; any other lane → fail.
        # Reset to -1 by _complete_return_phase().
        self._return_phase_lane = -1

        kwargs.setdefault('max_trials', max_trials)
        kwargs.setdefault('coverage_threshold', coverage_threshold)
        kwargs.setdefault('track_y_lo', track_y_lo)
        kwargs.setdefault('track_y_hi', track_y_hi)
        kwargs.setdefault('start_area_center', start_area_center)
        kwargs.setdefault('start_area_half', start_area_half)
        kwargs.setdefault('end_area_center', end_area_center)
        kwargs.setdefault('end_area_half', end_area_half)
        super().__init__(**kwargs)

        # Sanity-check the lane geometry: every lane should be wide enough
        # for a full T-block to traverse without clipping the divider.
        s = self.block_scale
        min_w = 4.03 * s + 6.0  # block stem extent + small padding
        for lane_w in self._lane_widths():
            if lane_w < min_w:
                raise ValueError(
                    f"Lane width {lane_w:.1f} < required {min_w:.1f}. "
                    f"Adjust divider_1_x / divider_2_x or shrink block_scale."
                )

    # ------------------------------------------------------------------
    # Lane geometry helpers
    # ------------------------------------------------------------------
    def _lane_widths(self):
        ws = self.window_size
        return (
            self.divider_1_x - 5.0,
            self.divider_2_x - self.divider_1_x,
            (ws - 5.0) - self.divider_2_x,
        )

    def _lane_idx_at(self, bx: float) -> int:
        if bx < self.divider_1_x:
            return 0
        if bx < self.divider_2_x:
            return 1
        return 2

    def _lane_centers(self):
        ws = self.window_size
        return (
            (5.0 + self.divider_1_x) / 2.0,
            (self.divider_1_x + self.divider_2_x) / 2.0,
            (self.divider_2_x + (ws - 5.0)) / 2.0,
        )

    def _high_friction_lane_idxs(self):
        return tuple(i for i in range(3) if i != self.normal_lane_idx)

    # ------------------------------------------------------------------
    # Episode layout sampling. Only ``normal_lane_idx`` is seed-driven;
    # the dividers are fixed across episodes (equal-width lanes).
    # ------------------------------------------------------------------
    def _sample_normal_lane_idx(self, rs: np.random.RandomState) -> int:
        return int(rs.randint(0, 3))

    # ------------------------------------------------------------------
    # Reset — overrides parent. Same RandomState drives every per-episode
    # random draw so the initial state is fully reproducible per seed.
    # ------------------------------------------------------------------
    def reset(self):
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        rs = np.random.RandomState(seed=seed)

        # 1) Sample only the normal-lane index (dividers are fixed).
        self.normal_lane_idx = self._sample_normal_lane_idx(rs)

        # 2) Burn one draw to keep the parent's `high_friction_side` field
        #    populated (downstream consumers may read it). Not used for
        #    drag computation in this subclass.
        self.high_friction_side = "left" if rs.rand() < 0.5 else "right"

        # 3) Block / agent spawn — inlined from parent reset() so we share
        #    the same RandomState. Block spawns inside the (visible) red
        #    rectangle = start_area_*; rejection sampling guarantees the
        #    block bbox stays within the start strip.
        if self.reset_to_state is not None:
            state = np.array(self.reset_to_state)
            self._set_state(state)
        else:
            sx, sy = self.start_area_center
            hx, hy = self.start_area_half
            s = self.block_scale
            r_down = 4.03 * s
            lo_ang, hi_ang = self.spawn_angle_range
            ang_edge = max(abs(lo_ang), abs(hi_ang))
            r_up = (2.0 * s * math.cos(ang_edge)
                    + 1.0 * s * math.sin(ang_edge))
            r_side = 4.03 * s
            pad = 4.0

            bx_min = max(sx - hx + r_side, r_side + 6)
            bx_max = min(sx + hx - r_side, self.window_size - r_side - 6)
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
                ay = max(by - (r_up + self.agent_radius + 4),
                         self.agent_radius + 8)
                return np.array([ax, ay, bx, by, block_angle])

            for _ in range(30):
                state = _sample()
                self._set_state(state)
                self.block.velocity = Vec2d(0, 0)
                self.block.angular_velocity = 0.0
                self.agent.velocity = Vec2d(0, 0)
                if self._block_bbox_within_start_strip():
                    break

        # 4) Snapshot the initial block pose. This becomes the target for
        #    the "return-to-original-location" criterion at each midstep.
        self._initial_block_pos = np.array(
            self.block.position, dtype=np.float64).copy()
        self._initial_block_angle = float(self.block.angle)

        # 5) Trial / return-phase state.
        self.current_trial = 0
        self._stuck_counter = 0
        self._prev_block_pos = None
        self._return_phase = False
        self._return_phase_steps = 0
        self._grace_remaining = self.reset_grace_steps
        self._block_entered_track = False
        self._block_past_track = False
        self._lane_sign = 0                  # legacy (parent uses internally)
        self._committed_lane_idx = -1
        self._force_done = False
        self._is_success = False
        self._latest_coverage = 0.0
        self._returns_completed = 0
        self._used_lanes = set()
        self._return_phase_lane = -1

        self._cached_goal_keypoint = self._compute_goal_t_keypoints()
        return self._get_obs()

    # ------------------------------------------------------------------
    # Return-to-original-location check.
    # ------------------------------------------------------------------
    def _angle_diff(self, a: float, b: float) -> float:
        """Return signed angle diff in (-pi, pi]."""
        d = (a - b) % (2.0 * math.pi)
        if d > math.pi:
            d -= 2.0 * math.pi
        return d

    def _at_initial_location(self) -> bool:
        cur_pos = np.array(self.block.position, dtype=np.float64)
        pos_dist = float(np.linalg.norm(cur_pos - self._initial_block_pos))
        ang_diff = abs(self._angle_diff(
            float(self.block.angle), self._initial_block_angle))
        return (pos_dist < self.return_pos_tol
                and ang_diff < self.return_ang_tol)

    def _block_in_start_box(self) -> bool:
        """True iff the block CENTER is inside the red start rectangle.

        Looser criterion than _at_initial_location: ignores rotation and the
        original spawn position; only requires the block to be back in the
        start area at all. Used as the trial-end gate for return_phase.
        """
        bx, by = float(self.block.position[0]), float(self.block.position[1])
        sx, sy = float(self.start_area_center[0]), float(self.start_area_center[1])
        shx, shy = float(self.start_area_half[0]), float(self.start_area_half[1])
        return (sx - shx <= bx <= sx + shx) and (sy - shy <= by <= sy + shy)

    # ------------------------------------------------------------------
    # Drag field — 3-lane geometry. The two high-friction lanes share an
    # IDENTICAL drag profile (so they look indistinguishable from the
    # policy's pre-commit position).
    # ------------------------------------------------------------------
    def _compute_drag_at(self, pos):
        bx, by = float(pos[0]), float(pos[1])
        if by < self.track_y_lo or by > self.track_y_hi:
            return self.low_drag
        lane_idx = self._lane_idx_at(bx)
        if lane_idx == self.normal_lane_idx:
            return self.low_drag

        # High-friction lane: SAME geometric ramp as the parent. We inline
        # rather than super(), since the parent gates on its 2-lane field
        # ``high_friction_side`` which is unused here.
        t_raw = ((by - self.track_y_lo)
                 / max(self.track_y_hi - self.track_y_lo, 1e-6))
        t_compressed = min(t_raw / self.friction_full_at, 1.0)
        n = self.friction_segments
        seg = int(np.clip(t_compressed * n, 0, n - 1))
        t = seg / max(n - 1, 1)
        ratio = self.high_drag / max(self.low_drag, 1e-8)
        return self.low_drag * (ratio ** t)

    # ------------------------------------------------------------------
    # Trial / lane-crossing state machine — overrides 2-lane parent.
    # KEY DIFFERENCE: every trial transition (whether the block was
    # backed-out manually or stalled in a high-friction lane) goes
    # through ``return_phase``, which only completes when the block has
    # been pushed back to its original spawn pose.
    # ------------------------------------------------------------------
    def _update_trial_state(self):
        if self._force_done or self._is_success:
            return

        block_pos = np.array(self.block.position, dtype=np.float64)
        self._latest_coverage = self._compute_coverage()

        if self._grace_remaining > 0:
            self._grace_remaining -= 1
            self._prev_block_pos = block_pos.copy()
            return

        # Episode-level success: coverage of goal-T (only outside return phase).
        # Double-safety: only count as success if the block reached the goal
        # via the normal-friction lane. If coverage is somehow met while the
        # committed lane is a high-friction one (e.g. physics quirk, missed
        # lane-cross check), force fail instead.
        if (self._latest_coverage > self.coverage_threshold
                and not self._return_phase):
            if self._committed_lane_idx == self.normal_lane_idx:
                self._is_success = True
            else:
                self._force_done = True
            return

        if self._return_phase:
            # Wrong-lane-during-return fail: the operator must complete the
            # return (push T back into the start box) BEFORE attempting any
            # other lane. If the block dips back below track_y_lo into a
            # lane different from the one we just abandoned, fail
            # immediately — no half-committing to the next trial during
            # return.
            if (block_pos[1] > self.track_y_lo
                    and self._return_phase_lane >= 0
                    and self._lane_idx_at(block_pos[0]) != self._return_phase_lane):
                self._force_done = True
                return
            # Loose criterion: trial ends as soon as the block CENTER is
            # back inside the red start rectangle. Original spawn pose
            # tolerance is no longer required.
            if self._block_in_start_box():
                self._complete_return_phase()
                return
            self._return_phase_steps += 1
            if self._return_phase_steps >= self.return_phase_timeout:
                self._force_done = True
            return

        # ----- Normal push phase -----
        if not self._block_entered_track and block_pos[1] > self.track_y_lo:
            self._block_entered_track = True
            self._committed_lane_idx = self._lane_idx_at(block_pos[0])
            self._lane_sign = self._committed_lane_idx - 1
            # Repeat-lane fail: if this lane was already attempted in a
            # prior trial of this episode, the operator has wasted a try
            # on a known-bad lane. Fail immediately.
            if self._committed_lane_idx in self._used_lanes:
                self._force_done = True
                return

        if (self._block_entered_track and not self._block_past_track
                and block_pos[1] > self.track_y_hi):
            self._block_past_track = True

        # Lane-crossing fail (only while inside the track zone).
        if (self.lane_crossing_fail and self._committed_lane_idx >= 0
                and self.track_y_lo <= block_pos[1] <= self.track_y_hi):
            cur_lane = self._lane_idx_at(block_pos[0])
            if cur_lane != self._committed_lane_idx:
                self._force_done = True
                return

        # Backward push: block re-entered start strip BEFORE reaching end.
        # New behaviour vs. parent: instead of auto-incrementing the trial
        # counter, we enter ``return_phase`` and require the operator /
        # policy to push the block back to its original spawn pose.
        if self._block_entered_track and block_pos[1] < self.track_y_lo:
            # Abandon-correct-lane fail: if the operator committed to the
            # normal-friction lane (the only winning one) and then backs
            # the block out into the start strip, the episode fails
            # immediately regardless of remaining trials. Continuing to
            # explore other lanes after finding the right one wastes the
            # demonstration.
            if self._committed_lane_idx == self.normal_lane_idx:
                self._force_done = True
                return
            if self.current_trial + 1 < self.max_trials:
                # Snapshot _committed_lane_idx FIRST (inside
                # _enter_return_phase) before clearing it below, so the
                # mid-return wrong-lane check can compare against it.
                self._enter_return_phase()
            else:
                # Out of trials — but we still let the operator try to
                # finish: only force_done if the FINAL trial got backed out
                # too. Use the same trial increment path.
                self._force_done = True
            self._block_entered_track = False
            self._block_past_track = False
            self._lane_sign = 0
            # _committed_lane_idx intentionally NOT cleared here — it's
            # consumed by _complete_return_phase() (added to _used_lanes
            # then cleared) when the return finishes.
            return

        # Stuck detection — only while inside track zone and after grace.
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
        # Snapshot the lane the operator is being asked to abandon so the
        # return-phase branch can detect the operator switching to a
        # different lane mid-return (which is now a fail).
        self._return_phase_lane = int(self._committed_lane_idx)
        super()._enter_return_phase()

    def _complete_return_phase(self):
        """Trial successfully terminated — block is back at original spawn.

        Called only when ``return_phase`` AND ``_at_initial_location()``.
        """
        # Mark the lane just attempted as used so re-committing fails next time.
        if self._committed_lane_idx >= 0:
            self._used_lanes.add(int(self._committed_lane_idx))
        self._return_phase_lane = -1
        self._return_phase = False
        self.current_trial += 1
        self._returns_completed += 1
        self._stuck_counter = 0
        self._prev_block_pos = None
        self._block_entered_track = False
        self._block_past_track = False
        self._committed_lane_idx = -1
        self._lane_sign = 0
        self._grace_remaining = self.reset_grace_steps

    # ------------------------------------------------------------------
    # Info — extend with 3-track + return-criterion metadata.
    # ------------------------------------------------------------------
    def _get_info(self):
        info = super()._get_info()
        info['divider_1_x'] = float(self.divider_1_x)
        info['divider_2_x'] = float(self.divider_2_x)
        info['normal_lane_idx'] = int(self.normal_lane_idx)
        info['high_friction_lane_idxs'] = list(self._high_friction_lane_idxs())
        info['committed_lane_idx'] = int(self._committed_lane_idx)
        info['current_lane_idx'] = int(self._lane_idx_at(float(self.block.position[0])))
        info['initial_block_pos'] = self._initial_block_pos.copy()
        info['initial_block_angle'] = float(self._initial_block_angle)
        info['at_initial_location'] = bool(self._at_initial_location())
        info['in_start_box'] = bool(self._block_in_start_box())
        info['returns_completed'] = int(self._returns_completed)
        info['used_lanes'] = sorted(self._used_lanes)
        info['return_phase_lane'] = int(self._return_phase_lane)
        info['return_pos_tol'] = float(self.return_pos_tol)
        info['return_ang_tol'] = float(self.return_ang_tol)
        info['lane_widths'] = list(self._lane_widths())
        return info

    # ------------------------------------------------------------------
    # Render — draw a translucent red rectangle over the start area, two
    # equally-spaced dividers, and the goal-T silhouette.
    # ------------------------------------------------------------------
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

        # ── 1) Translucent red rectangle over the start area. Clamped to
        #    fit between the top wall and the track-top boundary so it
        #    never overlaps the divider lines.
        sx, sy = self.start_area_center
        shx, shy = self.start_area_half
        sx0 = int(max(5, sx - shx))
        sx1 = int(min(ws - 5, sx + shx))
        sy0 = int(max(5, sy - shy))
        sy1 = int(min(self.track_y_lo, sy + shy))
        if sx1 > sx0 and sy1 > sy0:
            box = pygame.Surface((sx1 - sx0, sy1 - sy0), pygame.SRCALPHA)
            box.fill((*self.start_box_color, self.start_box_alpha))
            canvas.blit(box, (sx0, sy0))
            # Solid 1-px outline for a clean edge.
            pygame.draw.rect(
                canvas, pygame.Color(*self.start_box_color),
                pygame.Rect(sx0, sy0, sx1 - sx0, sy1 - sy0), width=1)

        # ── 2) Optional debug shading of the high-friction lanes.
        if self.debug_show_friction:
            high_idxs = self._high_friction_lane_idxs()
            lane_xranges = [
                (5, int(self.divider_1_x)),
                (int(self.divider_1_x), int(self.divider_2_x)),
                (int(self.divider_2_x), ws - 5),
            ]
            for lane_idx in high_idxs:
                lane_x0, lane_x1 = lane_xranges[lane_idx]
                n = 24
                for i in range(n):
                    y0 = self.track_y_lo + i * (self.track_y_hi - self.track_y_lo) / n
                    y1 = self.track_y_lo + (i + 1) * (self.track_y_hi - self.track_y_lo) / n
                    probe_x = (lane_x0 + lane_x1) / 2.0
                    d = self._compute_drag_at(np.array([probe_x, (y0 + y1) / 2.0]))
                    log_lo = math.log(max(self.low_drag, 1e-3))
                    log_hi = math.log(max(self.high_drag, self.low_drag + 1e-3))
                    t = (math.log(max(d, 1e-3)) - log_lo) / max(log_hi - log_lo, 1e-6)
                    t = max(0.0, min(1.0, t))
                    r = max(0, min(255, int(255 - 35 * t)))
                    g = max(0, min(255, int(225 - 160 * t)))
                    b = max(0, min(255, int(225 - 160 * t)))
                    rect = pygame.Rect(
                        lane_x0, int(y0), lane_x1 - lane_x0, int(y1 - y0) + 1)
                    pygame.draw.rect(canvas, pygame.Color(r, g, b), rect)
            bx = int(self.block.position[0])
            by = int(self.block.position[1])
            pygame.draw.circle(canvas, pygame.Color(0, 180, 220), (bx, by), 3)

        # ── 3) Goal-T silhouette (the only marker in the end strip).
        goal_body = self._get_goal_pose_body(self.goal_t_pose)
        for shape in self.block.shapes:
            pts = [pymunk.pygame_util.to_pygame(
                goal_body.local_to_world(v), draw_options.surface)
                for v in shape.get_vertices()]
            pts += [pts[0]]
            pygame.draw.polygon(canvas, pygame.Color(186, 238, 186), pts)
            pygame.draw.polygon(
                canvas, pygame.Color(60, 170, 60), pts, width=2)

        # ── 4) Two dividers + track top/bottom boundary lines.
        for div_x in (self.divider_1_x, self.divider_2_x):
            pygame.draw.line(canvas, pygame.Color(140, 140, 140),
                             (int(div_x), int(self.track_y_lo)),
                             (int(div_x), int(self.track_y_hi)), 2)
        pygame.draw.line(canvas, pygame.Color(140, 140, 140),
                         (5, int(self.track_y_lo)),
                         (ws - 5, int(self.track_y_lo)), 2)
        pygame.draw.line(canvas, pygame.Color(140, 140, 140),
                         (5, int(self.track_y_hi)),
                         (ws - 5, int(self.track_y_hi)), 2)

        # ── 5) Agent + block shapes (skip constraint glyphs).
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
