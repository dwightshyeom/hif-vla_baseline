import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from memory_diffusion_policy.env.pusht.pusht_asym_keypoints_env import PushTAsymKeypointsEnv
from memory_diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from memory_diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from memory_diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from memory_diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from memory_diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


class PushTAsymKeypointsRunner(BaseLowdimRunner):
    """
    Runner for PushT with asymmetric mass, using keypoint observations.

    Optionally prepends heavy_segment one-hot (3D) and/or goal keypoints (18D)
    to the observation so the policy knows which segment is heavy and where
    the goal is.
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
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            randomize_goal=False,
            heavy_mass=15.0,
            light_mass=0.1,
            include_goal_keypoint=False,
            include_heavy_segment=False
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTAsymKeypointsEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        randomize_goal=randomize_goal,
                        heavy_mass=heavy_mass,
                        light_mass=light_mass,
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
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

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
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

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
        self.include_goal_keypoint = include_goal_keypoint
        self.include_heavy_segment = include_heavy_segment

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
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function',
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            # Get episode-constant info from env after reset
            goal_keypoints = None
            heavy_segments = None

            if self.include_goal_keypoint or self.include_heavy_segment:
                def get_episode_info_fn(env):
                    base_env = env.env.env  # unwrap MultiStepWrapper -> VideoRecordingWrapper
                    result = {}
                    if hasattr(base_env, '_get_goal_keypoints'):
                        result['goal_keypoint'] = base_env._get_goal_keypoints()
                    result['heavy_segment'] = base_env.heavy_segment
                    return result

                info_fn_dill = dill.dumps(get_episode_info_fn)
                info_list = env.call('run_dill_function', info_fn_dill)

                if self.include_goal_keypoint:
                    goal_keypoints = np.array([info['goal_keypoint'] for info in info_list])
                if self.include_heavy_segment:
                    heavy_segments = np.array([info['heavy_segment'] for info in info_list])

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushTAsymKeypointsRunner {chunk_idx+1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                Do = obs.shape[-1] // 2

                # Extract block keypoints and agent position from observation
                block_keypoints = obs[..., :self.n_obs_steps, :Do].astype(np.float32)
                obs_mask = obs[..., :self.n_obs_steps, Do:] > 0.5

                # Build observation with optional prepended features
                obs_parts = []
                mask_parts = []

                if self.include_heavy_segment and heavy_segments is not None:
                    # One-hot encode heavy segment (3D)
                    B = block_keypoints.shape[0]
                    T = self.n_obs_steps
                    heavy_onehot = np.zeros((B, 3), dtype=np.float32)
                    heavy_onehot[np.arange(B), heavy_segments[:B]] = 1.0
                    heavy_onehot_seq = np.repeat(heavy_onehot[:, None, :], T, axis=1)
                    obs_parts.append(heavy_onehot_seq)
                    mask_parts.append(np.ones((B, T, 3), dtype=bool))

                if self.include_goal_keypoint and goal_keypoints is not None:
                    B = block_keypoints.shape[0]
                    T = self.n_obs_steps
                    goal_kp_flat = goal_keypoints[:B].reshape(B, -1)
                    goal_kp_seq = np.repeat(goal_kp_flat[:, None, :], T, axis=1)
                    obs_parts.append(goal_kp_seq)
                    mask_parts.append(np.ones((B, T, goal_kp_flat.shape[-1]), dtype=bool))

                obs_parts.append(block_keypoints)
                mask_parts.append(obs_mask)

                final_obs = np.concatenate(obs_parts, axis=-1)
                final_mask = np.concatenate(mask_parts, axis=-1)

                np_obs_dict = {
                    'obs': final_obs,
                    'obs_mask': final_mask
                }

                if self.past_action and (past_action is not None):
                    np_obs_dict['past_action'] = past_action[
                        :, -(self.n_obs_steps - 1):].astype(np.float32)

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

                # handle latency_steps
                action = np_action_dict['action'][:, self.n_latency_steps:]

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
            log_data[prefix + f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix + 'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
