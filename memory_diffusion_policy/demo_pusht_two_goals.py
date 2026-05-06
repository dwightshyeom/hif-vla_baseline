import os
# # Set SDL to use dummy video driver if no display is available
# # This allows pygame to run on headless systems
# if not os.environ.get('DISPLAY'):
#     os.environ['SDL_VIDEODRIVER'] = 'dummy'

import numpy as np
import click
from diffusion_policy.common.replay_buffer import ReplayBuffer
from memory_diffusion_policy.env.pusht.pusht_keypoints_two_goals_env import PushTKeypointsTwoGoalsEnv
import pygame

@click.command()
@click.option('-o', '--output', required=True)
@click.option('-rs', '--render_size', default=96, type=int)
@click.option('-hz', '--control_hz', default=10, type=int)
@click.option('-s', '--start_seed', default=None, type=int, help='Starting seed (default: continue from last episode)')
def main(output, render_size, control_hz, start_seed):
    """
    Collect demonstration for the Push-T Two Goals task.
    
    Usage: python demo_pusht_two_goals.py -o data/pusht_fixed_two_goals_demo.zarr
           python demo_pusht_two_goals.py -o data/pusht_fixed_two_goals_demo.zarr -s 43  # Start from seed 43
    
    This script is compatible with both Linux and MacOS.
    
    Task: Push the T block to overlap with BOTH goal areas (can be completed in any order):
          - Both goal areas are GREEN initially
          - Complete either goal first, then complete the other
          - Completed goals turn GRAY
    
    Controls:
    - Hover mouse close to the blue circle to start.
    - The episode will automatically terminate when both goals are reached.
    - Press "Q" to exit.
    - Press "R" to retry.
    - Hold "Space" to pause.
    
    Visual feedback:
    - GREEN area = Active goal (not yet reached)
    - GRAY area = Completed goal
    
    The recorded data includes:
    - state (11D): agent position (2D), block pose (3D), goal_1 pose (3D), goal_2 pose (3D)
    - keypoint (9, 2): current T-block keypoints in global coordinates
    - goal_1_keypoint (9, 2): target keypoints for Goal 1
    - goal_2_keypoint (9, 2): target keypoints for Goal 2
    - goal_progress: which goals have been reached (goal_1_reached, goal_2_reached)
    """
    
    # create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')

    # create PushT two goals env with keypoints
    kp_kwargs = PushTKeypointsTwoGoalsEnv.genenerate_keypoint_manager_params()
    env = PushTKeypointsTwoGoalsEnv(
        render_size=render_size,
        render_action=False,
        **kp_kwargs
    )
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
    
    print("\n" + "="*60)
    print("Push-T Two Goals Data Collection")
    print("="*60)
    print("Task: Visit BOTH goal areas (can be in any order)")
    print("Controls: Mouse to move, 'R' to retry, 'Q' to quit, 'Space' to pause")
    print("="*60 + "\n")
    
    # episode-level while loop
    while True:
        episode = list()
        # Use current_seed instead of replay_buffer.n_episodes
        seed = current_seed
        print(f'\n[Episode {seed}] Starting episode (Total saved: {replay_buffer.n_episodes})')
        
        # set seed for env
        env.seed(seed)
        
        # reset env and get observations (including info and render for recording)
        obs = env.reset()
        info = env._get_info()
        img = env.render(mode='human')
        
        # Cache goal keypoints once per episode (they're fixed for the entire episode)
        goal_1_keypoint_cached = info['goal_1_keypoint'].copy()
        goal_2_keypoint_cached = info['goal_2_keypoint'].copy()
        
        # Display agent position to help user find it
        agent_pos = info['pos_agent']
        print(f'  Agent spawned at: ({agent_pos[0]:.1f}, {agent_pos[1]:.1f})')
        print(f'  Goal 1 (GREEN - left) at: ({env.goal_pose_1[0]:.1f}, {env.goal_pose_1[1]:.1f}, θ={env.goal_pose_1[2]:.2f})')
        print(f'  Goal 2 (GREEN - right) at: ({env.goal_pose_2[0]:.1f}, {env.goal_pose_2[1]:.1f}, θ={env.goal_pose_2[2]:.2f})')
        print(f'  → Both goals are active - complete them in any order')
        print(f'  → Move mouse close to agent to start collecting data')
        
        # loop state
        retry = False
        pause = False
        done = False
        plan_idx = 0
        
        # Track goal completion for printing messages
        goal_1_was_reached = False
        goal_2_was_reached = False
        
        # Determine which goal is on left/right based on x-coordinate
        goal_1_side = "left" if env.goal_pose_1[0] < env.goal_pose_2[0] else "right"
        goal_2_side = "right" if env.goal_pose_1[0] < env.goal_pose_2[0] else "left"
        
        # Determine caption based on which goals are reached
        num_goals_reached = int(env.goal_1_reached) + int(env.goal_2_reached)
        if num_goals_reached == 0:
            caption = f'Episode {seed} | Both goals active (complete in any order)'
        elif num_goals_reached == 1:
            if env.goal_1_reached:
                caption = f'Episode {seed} | Goal 1 ({goal_1_side}) ✓ | Goal 2 ({goal_2_side}) remaining'
            else:
                caption = f'Episode {seed} | Goal 2 ({goal_2_side}) ✓ | Goal 1 ({goal_1_side}) remaining'
        else:
            caption = f'Episode {seed} | Both goals complete!'
        
        pygame.display.set_caption(caption)
        
        # step-level while loop
        while not done:
            # process keypress events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # hold Space to pause
                        plan_idx += 1
                        pygame.display.set_caption(f'Episode {seed} | PAUSED | plan_idx:{plan_idx}')
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
                        # Update caption based on current state
                        num_goals_reached = int(env.goal_1_reached) + int(env.goal_2_reached)
                        if num_goals_reached == 0:
                            caption = f'Episode {seed} | Both goals active'
                        elif num_goals_reached == 1:
                            if env.goal_1_reached:
                                caption = f'Episode {seed} | Goal 1 ({goal_1_side}) ✓ | Goal 2 ({goal_2_side}) remaining'
                            else:
                                caption = f'Episode {seed} | Goal 2 ({goal_2_side}) ✓ | Goal 1 ({goal_1_side}) remaining'
                        else:
                            caption = f'Episode {seed} | Both goals complete!'
                        pygame.display.set_caption(caption)

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
                # state dim 2+3+3+3 (agent_pos + block_pose + goal_1_pose + goal_2_pose)
                state = np.concatenate([
                    info['pos_agent'], 
                    info['block_pose'],
                    info['goal_1_pose'],
                    info['goal_2_pose']
                ])
                
                # discard unused information such as visibility mask and agent pos
                # for compatibility
                keypoint = obs.reshape(2,-1)[0].reshape(-1,2)[:9]
                
                # Use cached goal keypoints (fixed for entire episode)
                # goal keypoints for both goals
                
                data = {
                    'img': img,
                    'state': np.float32(state),
                    'keypoint': np.float32(keypoint),
                    'goal_1_keypoint': np.float32(goal_1_keypoint_cached),
                    'goal_2_keypoint': np.float32(goal_2_keypoint_cached),
                    'action': np.float32(act),
                    'n_contacts': np.float32([info['n_contacts']]),
                    'goal_1_reached': np.bool_(info['goal_1_reached']),
                    'goal_2_reached': np.bool_(info['goal_2_reached'])
                }
                episode.append(data)
                
            # step env and render
            obs, reward, done, info = env.step(act)
            img = env.render(mode='human')
            
            # Print messages when goals are reached
            if info['goal_1_reached'] and not goal_1_was_reached:
                coverage = info['goal_1_coverage'] * 100
                print(f'  ✓ Goal on the {goal_1_side.upper()} completed! Coverage: {coverage:.1f}%')
                if not info['goal_2_reached']:
                    print(f'  → Now move the T-block to the {goal_2_side.upper()} goal area')
                goal_1_was_reached = True
                
                # Update caption
                if not info['goal_2_reached']:
                    pygame.display.set_caption(f'Episode {seed} | Goal 1 ({goal_1_side}) ✓ | Goal 2 ({goal_2_side}) remaining')
                else:
                    pygame.display.set_caption(f'Episode {seed} | Both goals complete!')
            
            if info['goal_2_reached'] and not goal_2_was_reached:
                coverage = info['goal_2_coverage'] * 100
                print(f'  ✓ Goal on the {goal_2_side.upper()} completed! Coverage: {coverage:.1f}%')
                if not info['goal_1_reached']:
                    print(f'  → Now move the T-block to the {goal_1_side.upper()} goal area')
                goal_2_was_reached = True
                
                # Update caption
                if not info['goal_1_reached']:
                    pygame.display.set_caption(f'Episode {seed} | Goal 2 ({goal_2_side}) ✓ | Goal 1 ({goal_1_side}) remaining')
                else:
                    pygame.display.set_caption(f'Episode {seed} | Both goals complete!')
            
            # Print final completion message when both are done
            if goal_1_was_reached and goal_2_was_reached and info['goal_1_reached'] and info['goal_2_reached']:
                if not hasattr(env, '_both_goals_message_printed'):
                    print(f'  ✓ TASK COMPLETE! Both goals visited!')
                    env._both_goals_message_printed = True
            
            # Optional: Print coverage during collection (helpful for teleoperation)
            # Uncomment if you want real-time feedback
            # if len(episode) > 0 and len(episode) % 20 == 0:  # Print every 2 seconds at 10Hz
            #     if not info['goal_1_reached']:
            #         print(f'  Goal 1 coverage: {info["goal_1_coverage"]*100:.1f}%')
            #     elif not info['goal_2_reached']:
            #         print(f'  Goal 2 coverage: {info["goal_2_coverage"]*100:.1f}%')
            
            # regulate control frequency
            clock.tick(control_hz)
            
        # Save or retry episode
        if not retry:
            # save episode buffer to replay buffer (on disk)
            if len(episode) > 0:
                data_dict = dict()
                for key in episode[0].keys():
                    data_dict[key] = np.stack(
                        [x[key] for x in episode])
                replay_buffer.add_episode(data_dict, compressors='disk')
                
                # Print summary
                both_reached = goal_1_was_reached and goal_2_was_reached
                status = "SUCCESS (both goals)" if both_reached else "INCOMPLETE"
                print(f'\n[Episode {seed}] Saved with {len(episode)} timesteps - {status}')
                print(f'  Total episodes saved: {replay_buffer.n_episodes}')
            else:
                print(f'\n[Episode {seed}] Empty (no data collected), skipping save')
                print(f'  Total episodes saved: {replay_buffer.n_episodes}')
            # Increment seed for next episode
            current_seed += 1
        else:
            print(f'\n[Episode {seed}] Retrying (Total saved: {replay_buffer.n_episodes})')
            # Don't increment seed on retry, will use same seed again


if __name__ == "__main__":
    main()
