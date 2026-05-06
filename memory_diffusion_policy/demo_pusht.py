import numpy as np
import click
from diffusion_policy.common.replay_buffer import ReplayBuffer
from memory_diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
import pygame

@click.command()
@click.option('-o', '--output', required=True)
@click.option('-rs', '--render_size', default=96, type=int)
@click.option('-hz', '--control_hz', default=10, type=int)
@click.option('-s', '--start_seed', default=None, type=int, help='Starting seed (default: continue from last episode)')
def main(output, render_size, control_hz, start_seed):
    """
    Collect demonstration for the Push-T task.
    
    Usage: python demo_pusht.py -o data/pusht_one_goal_demo_1111.zarr
           python demo_pusht.py -o data/pusht_random_one_goal_demo.zarr -s 43  # Start from seed 43
    
    This script is compatible with both Linux and MacOS.
    Hover mouse close to the blue circle to start.
    Push the T block into the green area. 
    The episode will automatically terminate if the task is succeeded.
    Press "Q" to exit.
    Press "R" to retry.
    Hold "Space" to pause.
    
    The recorded data includes:
    - state (8D): agent position (2D), block pose (x, y, theta), goal pose (x, y, theta)
    - keypoint (9, 2): current T-block keypoints in global coordinates
    - goal_keypoint (9, 2): target keypoints where block keypoints should be at goal.
    """
    
    # create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')

    # create PushT env with keypoints
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    env = PushTKeypointsEnv(render_size=render_size, randomize_goal=False, render_action=False, **kp_kwargs)
    agent = env.teleop_agent()
    clock = pygame.time.Clock()
    
    # Determine starting seed
    if start_seed is not None:
        # Skip to the specified seed
        current_seed = start_seed
        print(f'Starting from seed {start_seed} (skipping {replay_buffer.n_episodes} existing episodes)')
    else:
        # Continue from where we left off
        current_seed = replay_buffer.n_episodes
    
    # episode-level while loop
    while True:
        episode = list()
        # Use current_seed instead of replay_buffer.n_episodes
        seed = current_seed
        print(f'Starting episode with seed {seed}, total episodes saved so far: {replay_buffer.n_episodes}')
        
        # set seed for env
        env.seed(seed)
        
        # reset env and get observations (including info and render for recording)
        obs = env.reset()
        info = env._get_info()
        img = env.render(mode='human')
        
        # Cache goal keypoint once per episode (it's fixed for the entire episode)
        goal_keypoint_cached = info['goal_keypoint'].copy()
        
        # Display agent position to help user find it
        agent_pos = info['pos_agent']
        print(f'  Agent spawned at: ({agent_pos[0]:.1f}, {agent_pos[1]:.1f}) - Move mouse close to start collecting')
        
        # loop state
        retry = False
        pause = False
        done = False
        plan_idx = 0
        pygame.display.set_caption(f'plan_idx:{plan_idx}')
        # step-level while loop
        while not done:
            # process keypress events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # hold Space to pause
                        plan_idx += 1
                        pygame.display.set_caption(f'plan_idx:{plan_idx}')
                        pause = True
                    elif event.key == pygame.K_r:
                        # press "R" to retry
                        retry=True
                    elif event.key == pygame.K_q:
                        # press "Q" to exit
                        exit(0)
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        pause = False

            # handle control flow
            if retry:
                break
            if pause:
                continue
            
            # get action from mouse
            # None if mouse is not close to the agent
            act = agent.act(obs)
            if not act is None:
                # teleop started
                # state dim 2+3+3 (agent_pos + block_pose + goal_pose)
                state = np.concatenate([
                    info['pos_agent'], 
                    info['block_pose'],
                    info['goal_pose']
                ])
                # discard unused information such as visibility mask and agent pos
                # for compatibility
                keypoint = obs.reshape(2,-1)[0].reshape(-1,2)[:9]
                
                # Use cached goal keypoint (fixed for entire episode)
                # goal_keypoint: where the block keypoints should be at goal (9, 2)
                
                data = {
                    'img': img,
                    'state': np.float32(state),
                    'keypoint': np.float32(keypoint),
                    'goal_keypoint': np.float32(goal_keypoint_cached),
                    'action': np.float32(act),
                    'n_contacts': np.float32([info['n_contacts']])
                }
                episode.append(data)
                
            # step env and render
            obs, reward, done, info = env.step(act)
            img = env.render(mode='human')
            
            # regulate control frequency
            clock.tick(control_hz)
        if not retry:
            # save episode buffer to replay buffer (on disk)
            if len(episode) > 0:
                data_dict = dict()
                for key in episode[0].keys():
                    data_dict[key] = np.stack(
                        [x[key] for x in episode])
                replay_buffer.add_episode(data_dict, compressors='disk')
                print(f'Saved seed {seed} with {len(episode)} timesteps. Total episodes: {replay_buffer.n_episodes}')
            else:
                print(f'Episode {seed} empty (no data collected), skipping save. Total episodes: {replay_buffer.n_episodes}')
            # Increment seed for next episode
            current_seed += 1
        else:
            print(f'Retrying seed {seed}. Total episodes: {replay_buffer.n_episodes}')
            # Don't increment seed on retry, will use same seed again


if __name__ == "__main__":
    main()