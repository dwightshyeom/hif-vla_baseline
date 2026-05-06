import numpy as np
import click
from diffusion_policy.common.replay_buffer import ReplayBuffer
from memory_diffusion_policy.env.pusht.pusht_asym_keypoints_env import PushTAsymKeypointsEnv
from memory_diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
import pygame


@click.command()
@click.option('-o', '--output', required=True)
@click.option('-rs', '--render_size', default=96, type=int)
@click.option('-hz', '--control_hz', default=10, type=int)
@click.option('-s', '--start_seed', default=None, type=int, help='Starting seed (default: continue from last episode)')
@click.option('--heavy_mass', default=15.0, type=float, help='Mass of the heavy segment')
@click.option('--light_mass', default=0.1, type=float, help='Mass of each light segment')
def main(output, render_size, control_hz, start_seed, heavy_mass, light_mass):
    """
    Collect demonstration for the Push-T Asymmetric Mass task.
    
    Usage: python demo_pusht_asym.py -o data/pusht_asym_demo.zarr
           python demo_pusht_asym.py -o data/pusht_asym_demo.zarr -s 10

    The T-block has asymmetric mass: one horizontal segment (shown darker)
    is much heavier than the others. Push the heavy segment for effective
    translation; pushing light segments causes mostly rotation.

    Which segment is heavy is randomized per episode (deterministic per seed).

    Controls:
    - Hover mouse close to the blue circle to start
    - Push the T block into the green area
    - Press "Q" to exit
    - Press "R" to retry
    - Hold "Space" to pause

    Recorded data includes:
    - state (8D): agent_pos (2D), block_pose (3D), goal_pose (3D)
    - keypoint (9, 2): T-block keypoints
    - goal_keypoint (9, 2): target keypoints at goal
    - action (2D): agent target position
    - n_contacts (1D): contact points
    - heavy_segment (1D): which horizontal segment is heavy (0=left, 1=mid, 2=right)
    """

    # Create replay buffer
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')

    # Create PushT asymmetric env with keypoints
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    env = PushTAsymKeypointsEnv(
        render_size=render_size,
        randomize_goal=False,
        render_action=False,
        heavy_mass=heavy_mass,
        light_mass=light_mass,
        **kp_kwargs)
    agent = env.teleop_agent()
    clock = pygame.time.Clock()

    # Determine starting seed
    if start_seed is not None:
        current_seed = start_seed
        print(f'Starting from seed {start_seed} (skipping {replay_buffer.n_episodes} existing episodes)')
    else:
        current_seed = replay_buffer.n_episodes

    segment_names = ['left', 'middle', 'right']

    # Episode-level while loop
    while True:
        episode = list()
        seed = current_seed
        
        # Set seed for env
        env.seed(seed)
        
        # Reset env and get observations
        obs = env.reset()
        info = env._get_info()
        img = env.render(mode='human')

        # Cache goal keypoint (fixed for entire episode)
        goal_keypoint_cached = info['goal_keypoint'].copy()
        heavy_segment = info['heavy_segment']

        agent_pos = info['pos_agent']
        print(f'Seed {seed} | Heavy segment: {segment_names[heavy_segment]} | '
              f'Agent at ({agent_pos[0]:.1f}, {agent_pos[1]:.1f}) | '
              f'Episodes saved: {replay_buffer.n_episodes}')

        # Loop state
        retry = False
        pause = False
        done = False
        plan_idx = 0
        pygame.display.set_caption(f'plan_idx:{plan_idx} heavy:{segment_names[heavy_segment]}')

        # Step-level while loop
        while not done:
            # Process keypress events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        plan_idx += 1
                        pygame.display.set_caption(f'plan_idx:{plan_idx} heavy:{segment_names[heavy_segment]}')
                        pause = True
                    elif event.key == pygame.K_r:
                        retry = True
                    elif event.key == pygame.K_q:
                        exit(0)
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        pause = False

            # Handle control flow
            if retry:
                break
            if pause:
                continue

            # Get action from mouse (None if mouse is not close to the agent)
            act = agent.act(obs)
            if act is not None:
                # state dim 2+3+3 (agent_pos + block_pose + goal_pose)
                state = np.concatenate([
                    info['pos_agent'],
                    info['block_pose'],
                    info['goal_pose']
                ])
                # Discard unused information such as visibility mask and agent pos
                keypoint = obs.reshape(2, -1)[0].reshape(-1, 2)[:9]

                data = {
                    'img': img,
                    'state': np.float32(state),
                    'keypoint': np.float32(keypoint),
                    'goal_keypoint': np.float32(goal_keypoint_cached),
                    'action': np.float32(act),
                    'n_contacts': np.float32([info['n_contacts']]),
                    'heavy_segment': np.int32([heavy_segment])
                }
                episode.append(data)

            # Step env and render
            obs, reward, done, info = env.step(act)
            img = env.render(mode='human')

            # Regulate control frequency
            clock.tick(control_hz)

        if not retry:
            # Save episode buffer to replay buffer
            if len(episode) > 0:
                data_dict = dict()
                for key in episode[0].keys():
                    data_dict[key] = np.stack([x[key] for x in episode])
                replay_buffer.add_episode(data_dict, compressors='disk')
                print(f'Saved seed {seed} ({len(episode)} steps). Total: {replay_buffer.n_episodes}')
            else:
                print(f'Skipping empty episode {seed}. Total: {replay_buffer.n_episodes}')
            current_seed += 1
        else:
            print(f'Retrying seed {seed}.')


if __name__ == "__main__":
    main()
