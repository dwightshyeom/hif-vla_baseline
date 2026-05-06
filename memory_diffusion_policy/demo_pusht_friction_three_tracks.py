import os
# # Set SDL to use dummy video driver if no display is available
# # This allows pygame to run on headless systems
# if not os.environ.get('DISPLAY'):
#     os.environ['SDL_VIDEODRIVER'] = 'dummy'

import numpy as np
import click
from diffusion_policy.common.replay_buffer import ReplayBuffer
from memory_diffusion_policy.env.pusht.pusht_keypoints_friction_three_tracks_env import (
    PushTKeypointsFrictionThreeTracksEnv,
)
import pygame


def close_pygame_window(env) -> None:
    """Tear down pygame display between episodes so the next render(mode='human')
    creates a fresh, focused window — same trick as demo_swapt.py.
    """
    try:
        pygame.display.quit()
    except pygame.error:
        pass
    env.window = None
    env.screen = None
    env.clock = None


@click.command()
@click.option('-o', '--output', required=True)
@click.option('-rs', '--render_size', default=96, type=int)
@click.option('-hz', '--control_hz', default=10, type=int)
@click.option('-s', '--start_seed', default=None, type=int,
              help='Starting seed (default: continue from last episode)')
@click.option('--debug_show_friction', is_flag=True, default=False,
              help='Tint the high-friction lanes (FOR PILOT/DEBUG ONLY '
                   '— never use this for real data collection or the policy '
                   'will see which lanes are bad).')
def main(output, render_size, control_hz, start_seed, debug_show_friction):
    """
    Collect demonstrations for the Push-T friction THREE-TRACK task.

    Usage:
        python demo_pusht_friction_three_tracks.py -o data/friction_three_tracks_demo.zarr
        python demo_pusht_friction_three_tracks.py -o data/friction_three_tracks_demo.zarr -s 43

    Task:
        Three vertical lanes between the start strip and the goal-T region.
        Two lanes carry an IDENTICAL high-friction gradient that becomes
        un-pushable past mid-track; one lane is normal friction. The
        operator must:
            (1) commit to a lane,
            (2) when the block stalls (or sooner if you can tell), back the
                block out into the start strip,
            (3) try a different lane,
            (4) eventually find the normal-friction lane and push the
                block all the way through to overlap the goal-T silhouette
                (coverage > 0.9 → success).

        The two divider x-positions and the normal-lane index are sampled
        per episode from the seed, so the layout looks different every
        episode but is fully reproducible per seed.

    Controls:
        - Hover the mouse close to the blue agent puck to start recording.
        - Episode auto-terminates on success or after max_trials lane
          attempts have been exhausted.
        - Press "Q" to quit.
        - Press "R" to retry the current seed (discards buffered data).
        - Hold "Space" to pause.

    Recorded per-timestep data (saved to the Zarr replay buffer):
        - img                : rendered rgb_array frame (render_size^2, 3)
        - state (5D)         : [agent_x, agent_y, block_x, block_y, block_theta]
        - action (2D)        : teleop target (agent goal position)
        - divider_1_x        : (1,) float — episode-constant
        - divider_2_x        : (1,) float — episode-constant
        - normal_lane_idx    : (1,) int   — episode-constant (0/1/2)
        - committed_lane_idx : (1,) int   — current commit (-1 = none)
        - current_lane_idx   : (1,) int   — lane the block center is in now
        - current_trial      : (1,) int   — 0..max_trials-1
        - return_phase       : (1,) int   — 0 / 1
        - coverage           : (1,) float — block coverage of goal T
        - n_contacts         : (1,) float
    """

    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')

    # Schema check — fail fast if existing buffer was written by a different demo.
    expected_trailing_shapes = {
        'state': (5,),
        'action': (2,),
        'divider_1_x': (1,),
        'divider_2_x': (1,),
        'normal_lane_idx': (1,),
        'committed_lane_idx': (1,),
        'current_lane_idx': (1,),
        'current_trial': (1,),
        'return_phase': (1,),
        'coverage': (1,),
        'n_contacts': (1,),
    }
    legacy_keys = (
        'high_friction_side', 'goal_1_keypoint', 'blue_target_pose',
        'blue_start_side', 'empty_idx',
    )
    if replay_buffer.n_episodes > 0:
        schema_errors = []
        for key, expected in expected_trailing_shapes.items():
            if key in replay_buffer.data:
                actual = replay_buffer.data[key].shape[1:]
                if actual != expected:
                    schema_errors.append(
                        f'  - "{key}": on-disk trailing shape {actual} '
                        f'!= expected {expected}')
        for legacy_key in legacy_keys:
            if legacy_key in replay_buffer.data:
                schema_errors.append(
                    f'  - legacy key "{legacy_key}" exists (this buffer '
                    f'was written by a different demo)')
        if schema_errors:
            print('\nERROR: output buffer already exists with an incompatible schema:')
            for line in schema_errors:
                print(line)
            print(f'\nUse a fresh output path, or delete the existing buffer:')
            print(f'  rm -rf {output}')
            exit(1)

    env = PushTKeypointsFrictionThreeTracksEnv(
        render_size=render_size,
        render_action=False,
        debug_show_friction=debug_show_friction,
    )
    agent = env.teleop_agent()
    clock = pygame.time.Clock()

    if start_seed is not None:
        current_seed = start_seed
        print(f'Starting from seed {start_seed} '
              f'(skipping {replay_buffer.n_episodes} existing episodes)')
    else:
        current_seed = replay_buffer.n_episodes

    print("\n" + "=" * 60)
    print("Push-T Friction THREE-TRACK Data Collection")
    print("=" * 60)
    print("Task: find the normal-friction lane (1 of 3) and push T through.")
    print("Two of the three lanes ramp to a hard wall; back out and retry.")
    if debug_show_friction:
        print("\n!!! debug_show_friction=ON — DO NOT USE FOR REAL DATA. !!!\n")
    print("Controls: Mouse to move, 'R' retry, 'Q' quit, 'Space' pause")
    print("=" * 60 + "\n")

    while True:
        episode = list()
        seed = current_seed

        env.seed(seed)
        obs = env.reset()
        info = env._get_info()
        img = env.render(mode='human')

        d1 = info['divider_1_x']
        d2 = info['divider_2_x']
        normal_idx = info['normal_lane_idx']
        lane_label = {0: 'LEFT', 1: 'MIDDLE', 2: 'RIGHT'}

        print(f'\n[Episode {seed}] (Total saved: {replay_buffer.n_episodes})')
        print(f'  Layout: dividers at x={d1:.1f} / {d2:.1f}; '
              f'lane widths={[round(w,1) for w in info["lane_widths"]]}')
        print(f'  Normal lane (the one to find): {lane_label[normal_idx]}')
        print(f'  High-friction lanes: '
              f'{[lane_label[i] for i in info["high_friction_lane_idxs"]]}')
        print(f'  Initial block pose: pos={info["initial_block_pos"].round(1).tolist()}, '
              f'angle={info["initial_block_angle"]:.3f} rad')
        print(f'  Return tolerance for trial transitions: '
              f'pos<{info["return_pos_tol"]}, ang<{info["return_ang_tol"]:.2f}')
        print(f'  Move mouse close to the blue agent puck to start recording.')

        retry = False
        pause = False
        done = False
        plan_idx = 0

        printed_lane_commit = -1
        printed_return = False
        printed_success = False
        printed_failed = False

        def caption():
            phase = "R" if info.get('return_phase', False) else "P"
            cov = info.get('coverage', 0.0) * 100
            tr = info.get('current_trial', 0)
            mt = info.get('max_trials', 1) if 'max_trials' in info else env.max_trials
            return (f'Ep {seed} | {phase} T:{tr}/{mt} | '
                    f'Cov: {cov:5.1f}%  (debug={debug_show_friction})')

        pygame.display.set_caption(caption())

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        plan_idx += 1
                        pygame.display.set_caption(
                            f'Episode {seed} | PAUSED | plan_idx:{plan_idx}')
                        pause = True
                    elif event.key == pygame.K_r:
                        retry = True
                    elif event.key == pygame.K_q:
                        exit(0)
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        pause = False
                        pygame.display.set_caption(caption())

            if retry:
                break
            if pause:
                continue

            act = agent.act(obs)
            if act is not None:
                # 5D state: agent (2) + block pose (3)
                state = np.array([
                    info['pos_agent'][0], info['pos_agent'][1],
                    env.block.position[0], env.block.position[1],
                    env.block.angle,
                ], dtype=np.float64)

                data = {
                    'img': img,
                    'state': np.float32(state),
                    'action': np.float32(act),
                    'divider_1_x': np.float32([d1]),
                    'divider_2_x': np.float32([d2]),
                    'normal_lane_idx': np.int32([normal_idx]),
                    'committed_lane_idx': np.int32([info.get('committed_lane_idx', -1)]),
                    'current_lane_idx': np.int32([info.get('current_lane_idx', -1)]),
                    'current_trial': np.int32([info.get('current_trial', 0)]),
                    'return_phase': np.int32([1 if info.get('return_phase', False) else 0]),
                    'coverage': np.float32([info.get('coverage', 0.0)]),
                    'n_contacts': np.float32([info.get('n_contacts', 0)]),
                }
                episode.append(data)

            obs, reward, done, info = env.step(act)
            img = env.render(mode='human')

            committed = info.get('committed_lane_idx', -1)
            if committed != -1 and committed != printed_lane_commit:
                tag = lane_label.get(committed, '?')
                quality = 'NORMAL' if committed == normal_idx else 'HIGH-FRICTION'
                print(f'  [trial {info["current_trial"]}] committed to {tag} ({quality})')
                printed_lane_commit = committed

            if not printed_return and info.get('return_phase', False):
                print(f'  [return phase] block backing out — try another lane')
                printed_return = True
            if not printed_return and info.get('current_trial', 0) > 0 and not info.get('return_phase', False):
                # New trial entered without a return phase = manual back-out.
                pass

            is_success = info.get('is_success', False)
            is_done_force = info.get('is_done', False) and not is_success
            if is_success and not printed_success:
                print(f'  [SUCCESS] Coverage: {info.get("coverage", 0.0)*100:.1f}%')
                printed_success = True
            if is_done_force and not printed_failed:
                print(f'  [FAILED] Out of trials.')
                printed_failed = True

            pygame.display.set_caption(caption())
            clock.tick(control_hz)

        # Save / retry
        if not retry:
            if len(episode) > 0:
                data_dict = dict()
                for key in episode[0].keys():
                    data_dict[key] = np.stack([x[key] for x in episode])
                replay_buffer.add_episode(data_dict, compressors='disk')
                if printed_success:
                    status = "SUCCESS"
                elif printed_failed:
                    status = "FAILED"
                else:
                    status = "INCOMPLETE"
                print(f'\n[Episode {seed}] Saved with {len(episode)} '
                      f'timesteps | {status}')
                print(f'  Total episodes saved: {replay_buffer.n_episodes}')
            else:
                print(f'\n[Episode {seed}] Empty (no data collected), skipping save')
                print(f'  Total episodes saved: {replay_buffer.n_episodes}')
            current_seed += 1
            close_pygame_window(env)
        else:
            print(f'\n[Episode {seed}] Retrying '
                  f'(Total saved: {replay_buffer.n_episodes})')


if __name__ == "__main__":
    main()
