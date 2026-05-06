import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from memory_diffusion_policy.env.pusht.pusht_keypoints_two_goals_env import PushTKeypointsTwoGoalsEnv
from memory_diffusion_policy.env.pusht.pusht_keypoints_two_goals_random_env import PushTKeypointsTwoGoalsRandomEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from memory_diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from memory_diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from memory_diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from memory_diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


class PushTKeypointsTwoGoalsRunner(BaseLowdimRunner):
    """
    Environment runner for PushT two goals task.
    Copied from PushTKeypointsRunner and adapted for two-goal sequential task.
    """
    
    def __init__(self,
            output_dir,
            keypoint_visible_rate=1.0,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=400,  # Increased from 200 to allow visiting both goals
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            goal_pose_1=None,  # First goal position
            goal_pose_2=None   # Second goal position
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # Get keypoint manager params
        kp_kwargs = PushTKeypointsTwoGoalsEnv.genenerate_keypoint_manager_params()

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTKeypointsTwoGoalsEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        goal_pose_1=goal_pose_1,
                        goal_pose_2=goal_pose_2,
                        **kp_kwargs
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
    
    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval TwoGoalsRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                # create obs dict
                Do = obs.shape[-1] // 2
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[...,:self.n_obs_steps,:Do].astype(np.float32),
                    'obs_mask': obs[...,:self.n_obs_steps,Do:] > 0.5
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]

                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data


class PushTKeypointsTwoGoalsRunnerWithGoal(BaseLowdimRunner):
    """
    Environment runner for PushT two RANDOM goals task with goal keypoints included in observation.
    This uses PushTKeypointsTwoGoalsRandomEnv which randomizes both goal positions on each reset.
    
    The model receives both goal keypoints in the observation for goal-conditioned policy learning.
    
    Observation format: [goal_1_keypoint (18D), goal_2_keypoint (18D), block_keypoint (18D), agent_pos (2D)] = 56D
    """
    
    def __init__(self,
            output_dir,
            keypoint_visible_rate=1.0,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=400,  # Increased for two-goal task
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            randomize_goals=True,  # Kept for compatibility but always uses random env
            include_goal_keypoints=True  # If True, prepend both goal keypoints to obs
        ):
        """
        Args:
            randomize_goals: Kept for config compatibility. Always uses PushTKeypointsTwoGoalsRandomEnv.
            include_goal_keypoints: If True, prepends both goal keypoints to observation.
                This should match how the training data was prepared:
                obs = [goal_1_keypoint (18D), goal_2_keypoint (18D), block_keypoint (18D), agent_pos (2D)]
                Set to True when using PushTLowdimDatasetWithTwoGoal.
        """
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # Get keypoint manager params
        kp_kwargs = PushTKeypointsTwoGoalsRandomEnv.genenerate_keypoint_manager_params()

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTKeypointsTwoGoalsRandomEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        **kp_kwargs
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.include_goal_keypoints = include_goal_keypoints
    
    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()
            
            # Get goal keypoints immediately after reset (before first step)
            # Get cached goal keypoints from the environment to ensure consistency
            goal_1_keypoints = None
            goal_2_keypoints = None
            if self.include_goal_keypoints:
                # Use dill function to access the base environment through wrappers
                # MultiStepWrapper -> VideoRecordingWrapper -> PushTKeypointsTwoGoalsRandomEnv
                def get_goal_kps_fn(env):
                    # Navigate through wrappers to get base PushTKeypointsTwoGoalsRandomEnv
                    base_env = env.env.env  # unwrap twice
                    return base_env._cached_goal_1_keypoints, base_env._cached_goal_2_keypoints
                
                goal_kps_fn_dill = dill.dumps(get_goal_kps_fn)
                goal_kps_list = env.call('run_dill_function', goal_kps_fn_dill)
                
                # goal_kps_list is a list of (goal_1, goal_2) tuples, one per environment
                goal_1_keypoints = np.array([kps[0] for kps in goal_kps_list])  # (B, 9, 2)
                goal_2_keypoints = np.array([kps[1] for kps in goal_kps_list])  # (B, 9, 2)
                
                # Debug: Verify goal keypoints are retrieved and will remain consistent
                # print(f"\n{'='*80}")
                # print(f"ROLLOUT EPISODE {chunk_idx+1} - Goal Keypoints Retrieved")
                # print(f"{'='*80}")
                # print(f"goal_1_keypoints shape: {goal_1_keypoints.shape}")  # Should be (B, 9, 2)
                # print(f"goal_2_keypoints shape: {goal_2_keypoints.shape}")  # Should be (B, 9, 2)
                # print(f"goal_1_keypoints[0, 0:2]: {goal_1_keypoints[0, 0:2]}")  # First 2 keypoints
                # print(f"goal_2_keypoints[0, 0:2]: {goal_2_keypoints[0, 0:2]}")  # First 2 keypoints
                # print(f"These will remain CONSTANT throughout this episode.")
                # print(f"{'='*80}\n")

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval TwoGoalsRunnerWithGoal {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            step_count = 0
            while not done:
                Do = obs.shape[-1] // 2
                
                # Extract block keypoints and agent position from observation
                # obs format: [block_keypoints (9×2 flattened), agent_pos (2), masks...]
                block_keypoints = obs[...,:self.n_obs_steps,:Do].astype(np.float32)  # (B, T, Do)
                obs_mask = obs[...,:self.n_obs_steps,Do:] > 0.5  # (B, T, Do)
                
                if self.include_goal_keypoints:
                    # Reshape and repeat goal keypoints for all timesteps in observation window
                    goal_1_keypoints_flat = goal_1_keypoints.reshape(goal_1_keypoints.shape[0], -1)  # (B, 18)
                    goal_1_keypoints_seq = np.repeat(
                        goal_1_keypoints_flat[:, None, :], 
                        self.n_obs_steps, 
                        axis=1
                    )  # (B, T, 18)
                    
                    goal_2_keypoints_flat = goal_2_keypoints.reshape(goal_2_keypoints.shape[0], -1)  # (B, 18)
                    goal_2_keypoints_seq = np.repeat(
                        goal_2_keypoints_flat[:, None, :], 
                        self.n_obs_steps, 
                        axis=1
                    )  # (B, T, 18)
                    
                    # Concatenate: [goal_1_keypoint, goal_2_keypoint, block_keypoint, agent_pos]
                    # block_keypoints already contains [block_kps + agent_pos] from env
                    obs_with_goals = np.concatenate([
                        goal_1_keypoints_seq,  # (B, T, 18)
                        goal_2_keypoints_seq,  # (B, T, 18)
                        block_keypoints        # (B, T, 20) = [block_kps(18) + agent_pos(2)]
                    ], axis=-1)  # (B, T, 56)
                    
                    # # Debug: Print at first step to verify order
                    # if step_count == 0:
                    #     print(f"\nSTEP 0 - Observation Construction:")
                    #     print(f"  obs_with_goals shape: {obs_with_goals.shape}")  # Should be (B, T, 56)
                    #     print(f"  Order: [goal_1(18D), goal_2(18D), block(18D), agent(2D)]")
                    #     print(f"  First env, first timestep:")
                    #     print(f"    goal_1_kp[0:4]: {obs_with_goals[0, 0, 0:4]}")  # dims 0-3
                    #     print(f"    goal_2_kp[0:4]: {obs_with_goals[0, 0, 18:22]}")  # dims 18-21
                    #     print(f"    block_kp[0:4]: {obs_with_goals[0, 0, 36:40]}")  # dims 36-39
                    #     print(f"    agent_pos: {obs_with_goals[0, 0, 54:56]}")  # dims 54-55
                    #     print(f"  ✓ This matches training data order!\n")
                    
                    # Create mask for goal keypoints (always visible)
                    goal_1_mask = np.ones((obs_mask.shape[0], obs_mask.shape[1], 18), dtype=bool)
                    goal_2_mask = np.ones((obs_mask.shape[0], obs_mask.shape[1], 18), dtype=bool)
                    obs_mask_with_goals = np.concatenate([goal_1_mask, goal_2_mask, obs_mask], axis=-1)
                    
                    # create obs dict
                    np_obs_dict = {
                        'obs': obs_with_goals,
                        'obs_mask': obs_mask_with_goals
                    }
                else:
                    # Original behavior without goal keypoints
                    np_obs_dict = {
                        'obs': block_keypoints,
                        'obs_mask': obs_mask
                    }
                
                if self.past_action and (past_action is not None):
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]

                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
