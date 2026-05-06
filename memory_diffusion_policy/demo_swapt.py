import os
# # Set SDL to use dummy video driver if no display is available
# # This allows pygame to run on headless systems
# if not os.environ.get('DISPLAY'):
#     os.environ['SDL_VIDEODRIVER'] = 'dummy'

import numpy as np
import click
from diffusion_policy.common.replay_buffer import ReplayBuffer
from memory_diffusion_policy.env.pusht.pusht_two_swap_env import PushTTwoSwapEnv
import pygame


# Map between the string form of "blue_start_side" and the integer form
# we save to the replay buffer.
SIDE_STR_TO_INT = {'left': 0, 'right': 1}
SIDE_INT_TO_STR = {v: k for k, v in SIDE_STR_TO_INT.items()}


def close_pygame_window(env) -> None:
    """
    Tear down the pygame display window and clear the env's cached
    window/screen/clock handles, so that the NEXT call to
    `env.render(mode='human')` creates a brand-new window. Freshly-mapped
    windows are placed on top and given keyboard focus by virtually all
    X11 window managers, which is exactly the behavior we want between
    teleop episodes.

    Intentionally does NOT call `pygame.quit()` (that would also tear
    down the timing / event subsystems we reuse across episodes).
    """
    try:
        pygame.display.quit()
    except pygame.error:
        pass
    env.window = None
    env.screen = None
    env.clock = None


def prompt_spawn_side(seed: int) -> str:
    """
    Interactive prompt: ask the operator where BLUE should spawn for the
    upcoming episode. Returns 'left' or 'right'. Exits the program on 'q'.
    """
    prompt = (f'[Episode {seed}] Where should the BLUE block spawn? '
              f"[l=left / r=right / q=quit]: ")
    while True:
        raw = input(prompt).strip().lower()
        if raw in ('l', 'left'):
            return 'left'
        if raw in ('r', 'right'):
            return 'right'
        if raw in ('q', 'quit', 'exit'):
            print('Exiting data collection.')
            exit(0)
        print("  Invalid input. Enter 'l' (left), 'r' (right), or 'q' (quit).")


@click.command()
@click.option('-o', '--output', required=True)
@click.option('-rs', '--render_size', default=96, type=int)
@click.option('-hz', '--control_hz', default=10, type=int)
@click.option('-s', '--start_seed', default=None, type=int,
              help='Starting seed (default: continue from last episode)')
def main(output, render_size, control_hz, start_seed):
    """
    Collect demonstrations for the Push-T Two-Block SWAP task.

    Usage:
        python demo_swapt.py -o data/pusht_two_swap_demo.zarr
        python demo_swapt.py -o data/pusht_two_swap_demo.zarr -s 43

    Task:
        Two same-colored target zones (GREEN) are placed on the left (Target A)
        and right (Target B). One T-block is BLUE, the other is RED; each
        starts perfectly aligned inside one of the zones. Your goal is to
        SWAP them so that each block ends up inside the zone it did NOT
        start in.

        Before each episode you will be prompted to choose which side
        BLUE spawns on ('l' / 'r'). RED takes the other side. The swap
        targets are derived automatically:

            BLUE starts LEFT  -> BLUE must reach RIGHT, RED must reach LEFT
            BLUE starts RIGHT -> BLUE must reach LEFT,  RED must reach RIGHT

    Recommended 5-step teleoperation protocol:
        (1) Start on goals  - both T's already aligned in their start zones.
        (2) Clear BLUE      - push BLUE into a neutral area (e.g. upper half).
        (3) Clear RED       - push RED  into the opposite neutral area.
        (4) Align BLUE      - push BLUE into its swap target zone.
        (5) Align RED       - push RED  into its swap target zone.

    Controls:
        - At the terminal, enter 'l' or 'r' before each episode to pick
          BLUE's spawn side (or 'q' to quit entirely).
        - In the pygame window, hover the mouse close to the blue agent
          puck to start recording.
        - Episode auto-terminates when BOTH blocks exceed the success
          threshold (see env.success_threshold) in their swap targets.
        - Press "Q" to quit.
        - Press "R" to retry the current episode (discards buffered data).
        - Hold "Space" to pause.

    Recorded per-timestep data (saved to the Zarr replay buffer):
        - img              : rendered rgb_array frame (render_size^2, 3)
        - state (8D)       : [agent_x, agent_y,
                              blue_x, blue_y, blue_theta,
                              red_x,  red_y,  red_theta]
        - action (2D)      : teleop target (agent goal position)
        - blue_target_pose : (3,) pose BLUE must reach this episode
        - red_target_pose  : (3,) pose RED  must reach this episode
        - blue_start_side  : (1,) int  0 = left, 1 = right
        - cov_blue_in_target : current coverage of BLUE in its swap target
        - cov_red_in_target  : current coverage of RED  in its swap target
        - n_contacts       : contact count at this step
    """

    # create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')

    # Fail fast if the existing buffer at `output` was written by a
    # different demo/schema. We compare per-step (trailing) shapes of
    # the keys this demo writes against whatever is already on disk.
    # Mismatch -> abort immediately instead of crashing at add_episode
    # after the operator has already finished a long teleop episode.
    expected_trailing_shapes = {
        'state': (8,),
        'action': (2,),
        'blue_target_pose': (3,),
        'red_target_pose': (3,),
        'blue_start_side': (1,),
        'cov_blue_in_target': (1,),
        'cov_red_in_target': (1,),
        'n_contacts': (1,),
    }
    # Legacy/other-demo keys that should NOT appear in a swap-task buffer.
    legacy_keys = (
        'keypoint', 'goal_1_keypoint', 'goal_2_keypoint',
        'goal_1_reached', 'goal_2_reached',
        # Old names used by earlier revisions of this swap demo.
        'cov_blue_in_b', 'cov_red_in_a', 'target_a_pose', 'target_b_pose',
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
                    f'was written by a different demo or older revision)')
        if schema_errors:
            print('\nERROR: output buffer already exists with an incompatible '
                  'schema:')
            for line in schema_errors:
                print(line)
            print('\nThis demo writes state=(8,) = agent(2)+blue(3)+red(3) and '
                  'per-episode blue/red target poses.')
            print('Use a fresh output path, or delete the existing buffer:')
            print(f'  rm -rf {output}')
            exit(1)

    # create the two-block swap env
    env = PushTTwoSwapEnv(
        render_size=render_size,
        render_action=False,
    )
    agent = env.teleop_agent()
    clock = pygame.time.Clock()

    # Determine starting seed
    if start_seed is not None:
        current_seed = start_seed
        print(f'Starting from seed {start_seed} '
              f'(skipping {replay_buffer.n_episodes} existing episodes)')
    else:
        current_seed = replay_buffer.n_episodes

    print("\n" + "=" * 60)
    print("Push-T Two-Block SWAP Data Collection")
    print("=" * 60)
    print("Task: SWAP the two T-blocks to the opposite zone.")
    print("You will be prompted for BLUE's spawn side before each episode.")
    print("Controls: Mouse to move, 'R' retry, 'Q' quit, 'Space' pause")
    print("=" * 60 + "\n")

    # episode-level while loop
    while True:
        episode = list()
        seed = current_seed

        # Ask the operator which side BLUE should spawn on.
        side_str = prompt_spawn_side(seed)
        side_int = SIDE_STR_TO_INT[side_str]
        env.set_blue_start_side(side_str)

        print(f'\n[Episode {seed}] Starting episode | BLUE spawn: {side_str.upper()} '
              f'(Total saved: {replay_buffer.n_episodes})')

        # set seed for env (kept for reproducibility)
        env.seed(seed)

        # reset env and get initial observation / info / frame
        obs = env.reset()
        info = env._get_info()
        img = env.render(mode='human')

        # Cache per-episode goal poses (fixed for the whole episode).
        blue_target_cached = info['blue_target_pose'].copy()
        red_target_cached = info['red_target_pose'].copy()

        # Display spawn info to help the operator orient themselves.
        agent_pos = info['pos_agent']
        blue_pose = info['blue_block_pose']
        red_pose = info['red_block_pose']
        print(f'  Agent spawned at: ({agent_pos[0]:.1f}, {agent_pos[1]:.1f})')
        print(f'  BLUE T  start : ({blue_pose[0]:.1f}, {blue_pose[1]:.1f}, '
              f'theta={blue_pose[2]:.2f})  -> target '
              f'({blue_target_cached[0]:.0f}, {blue_target_cached[1]:.0f}, '
              f'theta={blue_target_cached[2]:.2f})')
        print(f'  RED  T  start : ({red_pose[0]:.1f}, {red_pose[1]:.1f}, '
              f'theta={red_pose[2]:.2f})  -> target '
              f'({red_target_cached[0]:.0f}, {red_target_cached[1]:.0f}, '
              f'theta={red_target_cached[2]:.2f})')
        print(f'  -> Move mouse close to the blue agent puck to start recording')

        # loop state
        retry = False
        pause = False
        done = False
        plan_idx = 0

        # Track coverage-crossing events so we only print once per crossing.
        blue_done_printed = False
        red_done_printed = False

        pygame.display.set_caption(
            f'Episode {seed} | Blue spawn: {side_str.upper()} | SWAP task')

        def live_caption():
            return (f'Ep {seed} | Blue:{side_str[0].upper()} | '
                    f'BlueCov:{info["coverage_blue_in_target"]*100:5.1f}% | '
                    f'RedCov:{info["coverage_red_in_target"]*100:5.1f}%')

        # step-level while loop
        while not done:
            # process keypress events
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
                        pygame.display.set_caption(live_caption())

            if retry:
                break
            if pause:
                continue

            # get action from mouse
            # None if the mouse is not close to the agent puck yet.
            act = agent.act(obs)
            if act is not None:
                # state dim = 2 + 3 + 3 = 8
                #   agent_pos + blue_block_pose + red_block_pose
                state = np.concatenate([
                    info['pos_agent'],         # (2,)
                    info['blue_block_pose'],   # (3,) [x, y, theta]
                    info['red_block_pose'],    # (3,) [x, y, theta]
                ])

                data = {
                    'img': img,
                    'state': np.float32(state),
                    'action': np.float32(act),
                    'blue_target_pose': np.float32(blue_target_cached),
                    'red_target_pose': np.float32(red_target_cached),
                    'blue_start_side': np.int32([side_int]),
                    'cov_blue_in_target': np.float32([info['coverage_blue_in_target']]),
                    'cov_red_in_target': np.float32([info['coverage_red_in_target']]),
                    'n_contacts': np.float32([info['n_contacts']]),
                }
                episode.append(data)

            # step env and render
            obs, reward, done, info = env.step(act)
            img = env.render(mode='human')

            # Fire one-shot console messages when each block first crosses threshold.
            if (not blue_done_printed
                    and info['coverage_blue_in_target'] > env.success_threshold):
                print(f'  [OK] BLUE aligned in its target '
                      f'(coverage {info["coverage_blue_in_target"]*100:.1f}%)')
                blue_done_printed = True
            if (not red_done_printed
                    and info['coverage_red_in_target'] > env.success_threshold):
                print(f'  [OK] RED  aligned in its target '
                      f'(coverage {info["coverage_red_in_target"]*100:.1f}%)')
                red_done_printed = True

            # Live caption with both coverages so the operator has instant feedback.
            pygame.display.set_caption(live_caption())

            clock.tick(control_hz)

        # Save or retry episode
        if not retry:
            if len(episode) > 0:
                data_dict = dict()
                for key in episode[0].keys():
                    data_dict[key] = np.stack([x[key] for x in episode])
                replay_buffer.add_episode(data_dict, compressors='disk')

                both_done = blue_done_printed and red_done_printed
                status = "SUCCESS (swap complete)" if both_done else "INCOMPLETE"
                print(f'\n[Episode {seed}] Saved with {len(episode)} timesteps '
                      f'| Blue spawn: {side_str.upper()} | {status}')
                print(f'  Total episodes saved: {replay_buffer.n_episodes}')
            else:
                print(f'\n[Episode {seed}] Empty (no data collected), skipping save')
                print(f'  Total episodes saved: {replay_buffer.n_episodes}')
            current_seed += 1

            # Close the pygame window now that the episode has ended.
            # The next episode's env.render(mode='human') will create a
            # fresh window, which X11 window managers map on top with
            # keyboard focus - no manual click needed.
            close_pygame_window(env)
        else:
            print(f'\n[Episode {seed}] Retrying '
                  f'(Total saved: {replay_buffer.n_episodes})')
            # Don't increment seed on retry, same seed will be used again.


if __name__ == "__main__":
    main()
