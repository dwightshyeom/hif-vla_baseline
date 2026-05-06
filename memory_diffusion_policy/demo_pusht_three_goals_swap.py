import os
# # Set SDL to use dummy video driver if no display is available
# # This allows pygame to run on headless systems
# if not os.environ.get('DISPLAY'):
#     os.environ['SDL_VIDEODRIVER'] = 'dummy'

import numpy as np
import click
from diffusion_policy.common.replay_buffer import ReplayBuffer
from memory_diffusion_policy.env.pusht.pusht_three_goals_swap_env import (
    PushTThreeGoalsSwapEnv,
)
import pygame


def close_pygame_window(env) -> None:
    """
    Tear down the pygame display window between episodes so the next
    render(mode='human') creates a fresh window — most X11 WMs map fresh
    windows on top with keyboard focus, which is what we want during
    teleop.
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
def main(output, render_size, control_hz, start_seed):
    """
    Collect demonstrations for the Push-T Three-Goals SWAP task.

    Usage:
        python demo_pusht_three_goals_swap.py -o data/three_goals_swap_demo.zarr
        python demo_pusht_three_goals_swap.py -o data/three_goals_swap_demo.zarr -s 43

    Task:
        Three GREEN target zones at fixed positions (triangle layout).
        BLUE and RED T-blocks each spawn perfectly aligned in two of the
        three zones — the third zone is empty. The empty zone and the
        BLUE/RED assignment to the two filled zones are randomly drawn
        from the episode seed.

        Goal: SWAP the two blocks. BLUE must end up where RED started, RED
        must end up where BLUE started, the originally-empty zone must
        again be empty.

    Recommended teleoperation protocol (free-choice — pick whichever block
    is geometrically convenient as "first mover"):
        (1) Push the FIRST mover from its start zone → the empty zone.
        (2) Push the SECOND mover from its start zone → the FIRST mover's
            original start zone.
        (3) Push the FIRST mover from the originally-empty zone → the
            SECOND mover's original start zone (its swap target).

    Failure conditions enforced by the env (will end the episode without
    saving credit):
        - The SECOND mover ever overlaps the originally-empty zone.
          (Only the first mover may "park" in the empty zone.)
        - After phase 2, the SECOND mover leaves its target zone.

    Controls:
        - Hover the mouse close to the blue agent puck to start recording.
        - Episode auto-terminates on success or failure.
        - Press "Q" to quit.
        - Press "R" to retry the current seed (discards buffered data).
        - Hold "Space" to pause.

    Recorded per-timestep data (saved to the Zarr replay buffer):
        - img              : rendered rgb_array frame (render_size^2, 3)
        - state (8D)       : [agent_x, agent_y,
                              blue_x, blue_y, blue_theta,
                              red_x,  red_y,  red_theta]
        - action (2D)      : teleop target (agent goal position)
        - empty_idx          : (1,) int — which goal is empty (0/1/2)
        - blue_start_idx     : (1,) int
        - red_start_idx      : (1,) int
        - blue_target_idx    : (1,) int — = red_start_idx
        - red_target_idx     : (1,) int — = blue_start_idx
        - cov_blue_in_target : (1,) coverage of BLUE in its swap target
        - cov_red_in_target  : (1,) coverage of RED  in its swap target
        - phase              : (1,) int — phase machine snapshot
        - n_contacts         : (1,)
    """

    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')

    # Schema check — fail fast if the existing buffer at `output` was
    # written by a different demo / older revision of this one.
    expected_trailing_shapes = {
        'state': (8,),
        'action': (2,),
        'empty_idx': (1,),
        'blue_start_idx': (1,),
        'red_start_idx': (1,),
        'blue_target_idx': (1,),
        'red_target_idx': (1,),
        'cov_blue_in_target': (1,),
        'cov_red_in_target': (1,),
        'phase': (1,),
        'n_contacts': (1,),
    }
    legacy_keys = (
        'goal_1_keypoint', 'goal_2_keypoint', 'goal_3_keypoint',
        'goal_1_reached', 'goal_2_reached',
        'blue_target_pose', 'red_target_pose',
        'blue_start_side',
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

    env = PushTThreeGoalsSwapEnv(
        render_size=render_size,
        render_action=False,
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
    print("Push-T Three-Goals SWAP Data Collection")
    print("=" * 60)
    print("Task: SWAP the two T-blocks (free-choice 3-step protocol).")
    print("Empty zone + color assignment vary per seed.")
    print("Controls: Mouse to move, 'R' retry, 'Q' quit, 'Space' pause")
    print("=" * 60 + "\n")

    while True:
        episode = list()
        seed = current_seed

        env.seed(seed)
        obs = env.reset()
        info = env._get_info()
        img = env.render(mode='human')

        empty_idx = info['empty_idx']
        blue_start_idx = info['blue_start_idx']
        red_start_idx = info['red_start_idx']
        blue_target_idx = info['blue_target_idx']
        red_target_idx = info['red_target_idx']

        idx_label = {0: '1 (bot-left)', 1: '2 (top)', 2: '3 (bot-right)'}

        print(f'\n[Episode {seed}] (Total saved: {replay_buffer.n_episodes})')
        print(f'  EMPTY  zone : Goal {idx_label[empty_idx]}')
        print(f'  BLUE   start: Goal {idx_label[blue_start_idx]}  '
              f'->  target Goal {idx_label[blue_target_idx]}')
        print(f'  RED    start: Goal {idx_label[red_start_idx]}   '
              f'->  target Goal {idx_label[red_target_idx]}')
        print(f'  Move mouse close to the blue agent puck to start recording.')

        retry = False
        pause = False
        done = False
        plan_idx = 0

        # State for one-shot console messages.
        printed_phase1 = False
        printed_phase2 = False
        printed_success = False
        printed_failed = False

        def caption():
            if info.get('failed', False):
                return f'Ep {seed} | FAILED (protocol violation)'
            phase = info.get('phase', 0)
            cb = info['cov_blue_in_target']
            cr = info['cov_red_in_target']
            return (f'Ep {seed} | Phase {phase} | '
                    f'BlueCov:{cb*100:5.1f}% | RedCov:{cr*100:5.1f}%')

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
                # state dim = 2 + 3 + 3 = 8 (agent + blue_pose + red_pose)
                state = np.concatenate([
                    info['pos_agent'],
                    info['blue_block_pose'],
                    info['red_block_pose'],
                ])

                data = {
                    'img': img,
                    'state': np.float32(state),
                    'action': np.float32(act),
                    'empty_idx': np.int32([empty_idx]),
                    'blue_start_idx': np.int32([blue_start_idx]),
                    'red_start_idx': np.int32([red_start_idx]),
                    'blue_target_idx': np.int32([blue_target_idx]),
                    'red_target_idx': np.int32([red_target_idx]),
                    'cov_blue_in_target': np.float32([info['cov_blue_in_target']]),
                    'cov_red_in_target': np.float32([info['cov_red_in_target']]),
                    'phase': np.int32([info['phase']]),
                    'n_contacts': np.float32([info['n_contacts']]),
                }
                episode.append(data)

            obs, reward, done, info = env.step(act)
            img = env.render(mode='human')

            phase = info.get('phase', 0)
            failed = info.get('failed', False)

            if not printed_phase1 and phase >= 1:
                fm = info.get('first_mover', 'none').upper()
                print(f'  [phase 1] {fm} parked in EMPTY zone.')
                printed_phase1 = True
            if not printed_phase2 and phase >= 2:
                print(f'  [phase 2] Second mover reached its target.')
                printed_phase2 = True
            if not printed_success and phase >= 3 and not failed:
                print(f'  [SUCCESS] Swap complete!')
                printed_success = True
            if not printed_failed and failed:
                print(f'  [FAILED] Protocol violation — episode ended.')
                printed_failed = True

            pygame.display.set_caption(caption())
            clock.tick(control_hz)

        # Save / retry handling.
        if not retry:
            if len(episode) > 0:
                data_dict = dict()
                for key in episode[0].keys():
                    data_dict[key] = np.stack([x[key] for x in episode])
                # Only save successful episodes by default? We follow the
                # two-swap convention of saving everything and letting the
                # user re-run (R) on failures.
                replay_buffer.add_episode(data_dict, compressors='disk')

                if printed_success:
                    status = "SUCCESS (swap complete)"
                elif printed_failed:
                    status = "FAILED (protocol violation)"
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
            # Don't increment seed on retry — same seed will be used again.


if __name__ == "__main__":
    main()
