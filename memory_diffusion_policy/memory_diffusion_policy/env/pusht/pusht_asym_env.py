"""
PushT environment with asymmetric mass distribution on the T-block.

The horizontal bar of the T is divided into 3 equal segments. Each episode,
one segment is randomly selected to be much heavier than the other two.
The vertical bar has the same mass as the light segments.

This creates an asymmetric center of gravity, so the agent must learn to push
the heavy segment (near the CoG) for effective translation, while pushing
light segments causes mostly rotation.
"""
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import cv2
from pymunk.vec2d import Vec2d
from memory_diffusion_policy.env.pusht.pusht_env import PushTEnv
from diffusion_policy.env.pusht.pymunk_override import DrawOptions


def _draw_dashed_line_pygame(surface, p1, p2, color,
                             thickness=2, dash_length=6, gap_length=4):
    """Draw a dashed line between p1 and p2 on a pygame surface."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    line_len = (dx * dx + dy * dy) ** 0.5
    if line_len < 1:
        return
    ux, uy = dx / line_len, dy / line_len
    pos = 0.0
    drawing = True
    while pos < line_len:
        seg = dash_length if drawing else gap_length
        end_pos = min(pos + seg, line_len)
        if drawing:
            x1 = int(round(p1[0] + ux * pos))
            y1 = int(round(p1[1] + uy * pos))
            x2 = int(round(p1[0] + ux * end_pos))
            y2 = int(round(p1[1] + uy * end_pos))
            pygame.draw.line(surface, color, (x1, y1), (x2, y2), thickness)
        pos = end_pos
        drawing = not drawing


class PushTAsymEnv(PushTEnv):
    """
    PushT with asymmetric mass distribution.

    The T-block's horizontal bar is split into 3 equal segments.
    One segment (randomly chosen per episode) is heavy, the other two and
    the vertical bar are light.

    Args:
        heavy_mass: Mass of the heavy segment (default: 15.0)
        light_mass: Mass of each light segment and the vertical bar (default: 0.1)
    """

    def __init__(self,
            legacy=False,
            block_cog=None,
            damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None,
            goal_pose=None,
            randomize_goal=False,
            heavy_mass=40.0,
            light_mass=0.1):
        self.heavy_mass = heavy_mass
        self.light_mass = light_mass
        self.heavy_segment = 0  # Will be randomized on each reset
        super().__init__(
            legacy=legacy,
            block_cog=block_cog,
            damping=damping,
            render_action=render_action,
            render_size=render_size,
            reset_to_state=reset_to_state,
            goal_pose=goal_pose,
            randomize_goal=randomize_goal)

    def reset(self):
        # Deterministically pick heavy segment based on seed
        seg_rng = np.random.RandomState(seed=(self._seed * 7 + 13) % (2**31))
        self.heavy_segment = seg_rng.randint(0, 3)
        return super().reset()

    def _setup(self):
        # Call parent _setup (creates default T-block, walls, agent, etc.)
        super()._setup()

        # Remove the default symmetric block
        for shape in list(self.block.shapes):
            self.space.remove(shape)
        self.space.remove(self.block)

        # Add asymmetric block at the default position
        self.block = self._add_tee_asymmetric((256, 300), 0)

    def _add_tee_asymmetric(self, position, angle, scale=30,
                            mask=pymunk.ShapeFilter.ALL_MASKS()):
        """
        Create a T-block with asymmetric mass distribution.

        The horizontal bar (120x30 px) is split into 3 equal segments of 40px.
        heavy_segment (0=left, 1=middle, 2=right) gets heavy_mass,
        all other parts (including vertical bar) get light_mass.

        The vertical bar uses light_mass to maximize the asymmetric effect:
        the CoG is strongly pulled into the heavy horizontal segment, so
        pushing the stem causes large rotation while pushing the heavy
        segment gives effective translation.
        """
        length = 4
        seg_width = length * scale / 3.0  # 40 pixels each
        x_start = -length * scale / 2.0   # -60

        # Horizontal bar: 3 equal segments (y: 0 to scale)
        vertices_left = [
            (x_start, scale),
            (x_start + seg_width, scale),
            (x_start + seg_width, 0),
            (x_start, 0)
        ]
        vertices_mid = [
            (x_start + seg_width, scale),
            (x_start + 2 * seg_width, scale),
            (x_start + 2 * seg_width, 0),
            (x_start + seg_width, 0)
        ]
        vertices_right = [
            (x_start + 2 * seg_width, scale),
            (x_start + 3 * seg_width, scale),
            (x_start + 3 * seg_width, 0),
            (x_start + 2 * seg_width, 0)
        ]
        # Vertical bar (y: scale to length*scale)
        vertices_vert = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale)
        ]

        all_vertices = [vertices_left, vertices_mid, vertices_right, vertices_vert]

        # Assign masses: heavy segment gets heavy_mass, others get light_mass
        masses = [self.light_mass, self.light_mass, self.light_mass, self.light_mass]
        masses[self.heavy_segment] = self.heavy_mass

        # Compute centroids of each segment (geometric center of rectangle)
        centroids = []
        for verts in all_vertices:
            cx = sum(v[0] for v in verts) / len(verts)
            cy = sum(v[1] for v in verts) / len(verts)
            centroids.append((cx, cy))

        # Weighted center of gravity
        total_mass = sum(masses)
        cog_x = sum(m * c[0] for m, c in zip(masses, centroids)) / total_mass
        cog_y = sum(m * c[1] for m, c in zip(masses, centroids)) / total_mass

        # Moment of inertia using parallel axis theorem
        # I_body = sum(I_segment_about_own_centroid + m * d^2)
        total_inertia = 0.0
        for m, verts, c in zip(masses, all_vertices, centroids):
            I_local = pymunk.moment_for_poly(m, vertices=verts)
            dx = c[0] - cog_x
            dy = c[1] - cog_y
            total_inertia += I_local + m * (dx * dx + dy * dy)

        # Create body
        body = pymunk.Body(total_mass, total_inertia)

        # All segments use the same color — the agent must discover
        # which segment is heavy through interaction, not visual cues.
        block_color = 'LightSlateGray'

        # Create shapes
        shapes = []
        for verts in all_vertices:
            shape = pymunk.Poly(body, verts)
            shape.color = pygame.Color(block_color)
            shape.filter = pymunk.ShapeFilter(mask=mask)
            shapes.append(shape)

        # Set body properties
        body.center_of_gravity = (cog_x, cog_y)
        body.position = position
        body.angle = angle
        body.friction = 1

        self.space.add(body, *shapes)
        return body

    def _draw_segment_dividers_on_canvas(self, canvas):
        """
        Draw dashed white lines at the two segment boundaries on the horizontal
        bar of the T-block onto the given pygame canvas.

        The dividers run from the bottom to the top of the horizontal bar
        (y=0 to y=scale in block-local coords) at x=-20 and x=20.
        """
        scale = 30
        length = 4
        seg_width = length * scale / 3.0   # 40 px
        x_start  = -length * scale / 2.0   # -60

        # Local x positions of the two divider lines
        boundary_xs = [x_start + seg_width, x_start + 2 * seg_width]  # -20, 20
        y_bottom = 0.0
        y_top    = float(scale)             # 30

        # White dashes are clearly visible on both LightSlateGray and DarkSlateGray
        divider_color = (255, 255, 255)

        for bx in boundary_xs:
            # Bottom and top of the divider in block-local coords
            p1_local = Vec2d(bx, y_bottom)
            p2_local = Vec2d(bx, y_top)

            # Transform to world coords, then to pygame canvas coords (y flipped)
            p1_canvas = pymunk.pygame_util.to_pygame(
                self.block.local_to_world(p1_local), canvas)
            p2_canvas = pymunk.pygame_util.to_pygame(
                self.block.local_to_world(p2_local), canvas)

            # 2 px thick, 6 px dash / 4 px gap  →  ~3 dashes per 30 px bar height
            _draw_dashed_line_pygame(canvas, p1_canvas, p2_canvas,
                                     divider_color, thickness=2,
                                     dash_length=6, gap_length=4)

    def _render_frame(self, mode):
        """Override to inject dashed segment dividers before the display update."""
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

        draw_options = DrawOptions(canvas)

        # Draw goal pose outline
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [
                pymunk.pygame_util.to_pygame(
                    goal_body.local_to_world(v), draw_options.surface)
                for v in shape.get_vertices()
            ]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent and block
        self.space.debug_draw(draw_options)

        # Draw dashed segment dividers on top of the block (before display update)
        self._draw_segment_dividers_on_canvas(canvas)

        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

        img = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        # INTER_AREA preserves thin features (dashed segment dividers) across
        # the 512 -> render_size reduction; INTER_LINEAR (cv2 default) blurs
        # them below the visible threshold at render_size=96.
        img = cv2.resize(img, (self.render_size, self.render_size),
                         interpolation=cv2.INTER_AREA)
        if self.render_action:
            if self.latest_action is not None:
                action = np.array(self.latest_action)
                coord = (action / 512 * self.render_size).astype(np.int32)
                marker_size = int(8 / 96 * self.render_size)
                thickness = int(1 / 96 * self.render_size)
                cv2.drawMarker(img, coord,
                    color=(255, 0, 0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size, thickness=thickness)
        return img

    def _get_info(self):
        info = super()._get_info()
        info['heavy_segment'] = self.heavy_segment
        return info
