import os
# # Set SDL to use dummy video driver if no display is available
# # This allows pygame to run on headless systems
# if not os.environ.get('DISPLAY'):
#     os.environ['SDL_VIDEODRIVER'] = 'dummy'

import numpy as np
import click
from diffusion_policy.common.replay_buffer import ReplayBuffer
from memory_diffusion_policy.env.pusht.pusht_keypoints_three_goals_env import PushTKeypointsThreeGoalsEnv
import pygame

@click.command()
@click.option('-o', '--output', required=True)
@click.option('-rs', '--render_size', default=96, type=int)
@click.option('-hz', '--control_hz', default=10, type=int)
@click.option('-s', '--start_seed', default=None, type=int, help='Starting seed (default: continue from last episode)')
@click.option('--goal_pos_variation', default=0.0, type=float,
              help='Max positional perturbation for goal poses in pixels (default: 0 = fixed goals)')
@click.option('--goal_rot_variation', default=0.0, type=float,
              help='Max rotational perturbation for goal poses in radians (default: 0 = fixed rotation)')
def main(output, render_size, control_hz, start_seed, goal_pos_variation, goal_rot_variation):
    """
    Collect demonstration for the Push-T Three Goals task.
    
    Usage: python demo_pusht_three_goals.py -o data/pusht_three_goals_demo_simple.zarr
           python demo_pusht_three_goals.py -o data/pusht_three_goals_demo.zarr -s 43  # Start from seed 43
           python demo_pusht_three_goals.py -o data/pusht_three_goals_demo_pos_30_rot_0.3.zarr --goal_pos_variation 30 --goal_rot_variation 0.3
    
    This script is compatible with both Linux and MacOS.
    
    Task: Push the T block to overlap with ALL THREE goal areas (can be completed in any order):
          - All three goal areas are GREEN initially
          - Complete the goals in any order you prefer
          - Completed goals turn GRAY
    
    Controls:
    - Hover mouse close to the blue circle to start.
    - The episode will automatically terminate when all three goals are reached.
    - Press "Q" to exit.
    - Press "R" to retry.
    - Hold "Space" to pause.
    
    Visual feedback:
    - GREEN area = Active goal (not yet reached)
    - GRAY area = Completed goal
    
    The recorded data includes:
    - state (2D): agent position only (block pose from keypoints, goals are fixed)
    - keypoint (9, 2): current T-block keypoints in global coordinates
    - action: agent target position
    - goals_reached (3,): one-hot indicator [goal_1, goal_2, goal_3] where
                          [0,0,0] = none reached
                          [1,0,0] = bottom-left reached
                          [1,1,0] = bottom-left and top reached
                          [1,1,1] = all three reached
    - goal_1_keypoint (9, 2): keypoints for goal 1 (varies per episode if goal variation is used)
    - goal_2_keypoint (9, 2): keypoints for goal 2 (varies per episode if goal variation is used)
    - goal_3_keypoint (9, 2): keypoints for goal 3 (varies per episode if goal variation is used)
    
    Note: When goal variation is 0 (default), goals are fixed and keypoints are constant.
          When goal variation > 0, goals change per episode (seeded by episode seed for
          reproducibility). Goal keypoints MUST be used as observation inputs for the
          diffusion policy to condition on the varying goal locations.
    """
    
    # create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')

    # create PushT three goals env with keypoints
    kp_kwargs = PushTKeypointsThreeGoalsEnv.genenerate_keypoint_manager_params()
    env = PushTKeypointsThreeGoalsEnv(
        render_size=render_size,
        render_action=False,
        goal_pos_variation=goal_pos_variation,
        goal_rot_variation=goal_rot_variation,
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
    print("Push-T Three Goals Data Collection")
    print("="*60)
    print("Task: Visit ALL THREE goal areas (can be in any order)")
    if goal_pos_variation > 0 or goal_rot_variation > 0:
        print(f"Goal Variation: pos={goal_pos_variation:.1f}px, rot={goal_rot_variation:.3f}rad")
        print("  Goals will be randomly perturbed each episode (seeded for reproducibility)")
    else:
        print("Goal Variation: OFF (fixed goal positions)")
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
        
        # Display agent position to help user find it
        agent_pos = info['pos_agent']
        print(f'  Agent spawned at: ({agent_pos[0]:.1f}, {agent_pos[1]:.1f})')
        print(f'  Goal 1 (GREEN - bottom-left) at: ({env.goal_pose_1[0]:.1f}, {env.goal_pose_1[1]:.1f}, θ={env.goal_pose_1[2]:.2f})')
        print(f'  Goal 2 (GREEN - top) at: ({env.goal_pose_2[0]:.1f}, {env.goal_pose_2[1]:.1f}, θ={env.goal_pose_2[2]:.2f})')
        print(f'  Goal 3 (GREEN - bottom-right) at: ({env.goal_pose_3[0]:.1f}, {env.goal_pose_3[1]:.1f}, θ={env.goal_pose_3[2]:.2f})')
        print(f'  → All three goals form an equilateral triangle')
        print(f'  → Complete them in any order')
        print(f'  → Move mouse close to agent to start collecting data')
        
        # loop state
        retry = False
        pause = False
        done = False
        plan_idx = 0
        
        # Track goal completion for printing messages
        goal_1_was_reached = False
        goal_2_was_reached = False
        goal_3_was_reached = False
        
        # Determine goal positions for labeling (bottom-left, top, bottom-right)
        # Goal 1: bottom-left, Goal 2: top, Goal 3: bottom-right
        goal_labels = {
            1: "bottom-left",
            2: "top", 
            3: "bottom-right"
        }
        
        # Determine caption based on which goals are reached
        def get_caption():
            num_goals_reached = int(env.goal_1_reached) + int(env.goal_2_reached) + int(env.goal_3_reached)
            if num_goals_reached == 0:
                return f'Episode {seed} | All three goals active (triangle formation)'
            elif num_goals_reached == 1:
                reached_str = ""
                remaining_str = ""
                if env.goal_1_reached:
                    reached_str = f"Goal 1 ({goal_labels[1]}) ✓"
                    remaining = [f"Goal 2 ({goal_labels[2]})", f"Goal 3 ({goal_labels[3]})"]
                elif env.goal_2_reached:
                    reached_str = f"Goal 2 ({goal_labels[2]}) ✓"
                    remaining = [f"Goal 1 ({goal_labels[1]})", f"Goal 3 ({goal_labels[3]})"]
                else:
                    reached_str = f"Goal 3 ({goal_labels[3]}) ✓"
                    remaining = [f"Goal 1 ({goal_labels[1]})", f"Goal 2 ({goal_labels[2]})"]
                remaining_str = ", ".join(remaining)
                return f'Episode {seed} | {reached_str} | Remaining: {remaining_str}'
            elif num_goals_reached == 2:
                reached = []
                remaining_str = ""
                if env.goal_1_reached:
                    reached.append(f"Goal 1 ({goal_labels[1]})")
                if env.goal_2_reached:
                    reached.append(f"Goal 2 ({goal_labels[2]})")
                if env.goal_3_reached:
                    reached.append(f"Goal 3 ({goal_labels[3]})")
                
                if not env.goal_1_reached:
                    remaining_str = f"Goal 1 ({goal_labels[1]})"
                elif not env.goal_2_reached:
                    remaining_str = f"Goal 2 ({goal_labels[2]})"
                else:
                    remaining_str = f"Goal 3 ({goal_labels[3]})"
                
                reached_str = ", ".join(reached)
                return f'Episode {seed} | {reached_str} ✓ | Remaining: {remaining_str}'
            else:
                return f'Episode {seed} | All three goals complete!'
        
        pygame.display.set_caption(get_caption())
        
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
                        pygame.display.set_caption(get_caption())

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
                # Only store agent position in state (block pose from keypoints, goals are fixed)
                state = np.concatenate([
                    info['pos_agent']
                ])
                
                # discard unused information such as visibility mask and agent pos
                # for compatibility
                keypoint = obs.reshape(2,-1)[0].reshape(-1,2)[:9]
                
                # Create one-hot indicator for goal completion
                # [goal_1_reached, goal_2_reached, goal_3_reached]
                # [0,0,0] = none reached
                # [1,0,0] = bottom-left reached
                # [1,1,0] = bottom-left and top reached
                # [1,1,1] = all three reached
                goals_reached = np.array([
                    float(info['goal_1_reached']),
                    float(info['goal_2_reached']),
                    float(info['goal_3_reached'])
                ], dtype=np.float32)
                
                data = {
                    'img': img,
                    'state': np.float32(state),  # Only agent position (2D)
                    'keypoint': np.float32(keypoint),  # Block keypoints (9, 2)
                    'action': np.float32(act),
                    'n_contacts': np.float32([info['n_contacts']]),
                    'goals_reached': goals_reached,  # One-hot indicator (3,)
                    'goal_1_keypoint': np.float32(info['goal_1_keypoint']),  # Goal 1 keypoints (9, 2)
                    'goal_2_keypoint': np.float32(info['goal_2_keypoint']),  # Goal 2 keypoints (9, 2)
                    'goal_3_keypoint': np.float32(info['goal_3_keypoint'])   # Goal 3 keypoints (9, 2)
                }
                episode.append(data)
                
            # step env and render
            obs, reward, done, info = env.step(act)
            img = env.render(mode='human')
            
            # Print messages when goals are reached
            if info['goal_1_reached'] and not goal_1_was_reached:
                coverage = info['goal_1_coverage'] * 100
                print(f'  ✓ Goal on the {goal_labels[1].upper()} completed! Coverage: {coverage:.1f}%')
                num_remaining = int(not info['goal_2_reached']) + int(not info['goal_3_reached'])
                if num_remaining > 0:
                    print(f'  → {num_remaining} goal{"s" if num_remaining > 1 else ""} remaining')
                goal_1_was_reached = True
                pygame.display.set_caption(get_caption())
            
            if info['goal_2_reached'] and not goal_2_was_reached:
                coverage = info['goal_2_coverage'] * 100
                print(f'  ✓ Goal on the {goal_labels[2].upper()} completed! Coverage: {coverage:.1f}%')
                num_remaining = int(not info['goal_1_reached']) + int(not info['goal_3_reached'])
                if num_remaining > 0:
                    print(f'  → {num_remaining} goal{"s" if num_remaining > 1 else ""} remaining')
                goal_2_was_reached = True
                pygame.display.set_caption(get_caption())
            
            if info['goal_3_reached'] and not goal_3_was_reached:
                coverage = info['goal_3_coverage'] * 100
                print(f'  ✓ Goal on the {goal_labels[3].upper()} completed! Coverage: {coverage:.1f}%')
                num_remaining = int(not info['goal_1_reached']) + int(not info['goal_2_reached'])
                if num_remaining > 0:
                    print(f'  → {num_remaining} goal{"s" if num_remaining > 1 else ""} remaining')
                goal_3_was_reached = True
                pygame.display.set_caption(get_caption())
            
            # Print final completion message when all three are done
            if goal_1_was_reached and goal_2_was_reached and goal_3_was_reached and \
               info['goal_1_reached'] and info['goal_2_reached'] and info['goal_3_reached']:
                if not hasattr(env, '_all_goals_message_printed'):
                    print(f'  ✓ TASK COMPLETE! All three goals visited!')
                    env._all_goals_message_printed = True
            
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
                all_reached = goal_1_was_reached and goal_2_was_reached and goal_3_was_reached
                status = "SUCCESS (all three goals)" if all_reached else "INCOMPLETE"
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
            # Reset the message flag
            if hasattr(env, '_all_goals_message_printed'):
                delattr(env, '_all_goals_message_printed')


if __name__ == "__main__":
    main()
