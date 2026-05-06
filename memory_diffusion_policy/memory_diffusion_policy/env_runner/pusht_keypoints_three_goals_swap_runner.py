"""
Env runner for the Push-T three-goals swap task (keypoint observations).

Mirrors PushTKeypointsTwoSwapRunner: the flat env obs has shape
(2*Do,) where the leading Do dims are the data and the trailing Do dims are
the visibility mask. The runner splits and forwards (data, mask) to a
lowdim policy.

Per-episode swap-role randomisation (which target is empty, which color
goes where) is driven entirely by env.seed(); the runner only assigns
distinct seeds per env.
"""

import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv

from memory_diffusion_policy.env.pusht.pusht_keypoints_three_goals_swap_env import (
    PushTKeypointsThreeGoalsSwapEnv,
)
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from memory_diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from memory_diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecordingWrapper, VideoRecorder,
)

from memory_diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from memory_diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


class PushTKeypointsThreeGoalsSwapRunner(BaseLowdimRunner):
    def __init__(
        self,
        output_dir,
        keypoint_visible_rate=1.0,
        n_train=10,
        n_train_vis=3,
        train_start_seed=0,
        n_test=22,
        n_test_vis=6,
        legacy_test=False,
        test_start_seed=10000,
        max_steps=600,
        n_obs_steps=8,
        n_action_steps=8,
        n_latency_steps=0,
        fps=10,
        crf=22,
        past_action=False,
        tqdm_interval_sec=5.0,
        n_envs=None,
    ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTKeypointsThreeGoalsSwapEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1,
                    ),
                    file_path=None,
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps,
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

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

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

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

            env.call_each('run_dill_function',
                args_list=[(x,) for x in this_init_fns])

            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps,
                desc=f"Eval PushTKeypointsThreeGoalsSwap {chunk_idx+1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                Do = obs.shape[-1] // 2
                np_obs_dict = {
                    'obs': obs[..., :self.n_obs_steps, :Do].astype(np.float32),
                    'obs_mask': obs[..., :self.n_obs_steps, Do:] > 0.5,
                }
                if self.past_action and (past_action is not None):
                    np_obs_dict['past_action'] = past_action[
                        :, -(self.n_obs_steps - 1):].astype(np.float32)

                obs_dict = dict_apply(np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device))

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'][:, self.n_latency_steps:]

                # Phase + per-block coverage overlay on the video.
                def _set_progress_overlay(env_wrapper):
                    inner = env_wrapper.env.env  # → PushTKeypointsThreeGoalsSwapEnv
                    cov_b, cov_r = inner._swap_coverages()
                    env_wrapper.env.step_based_overlay = (
                        f"P{inner.phase} B:{cov_b:.2f} R:{cov_r:.2f}")
                    return None

                env.call_each('run_dill_function',
                    args_list=[(dill.dumps(_set_progress_overlay),)] * n_envs)

                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            success = 1.0 if max_reward >= 1.0 else 0.0
            max_rewards[prefix].append(success)
            log_data[prefix + f'sim_max_reward_{seed}'] = max_reward
            log_data[prefix + f'sim_success_{seed}'] = success

            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f'sim_video_{seed}'] = sim_video

        for prefix, value in max_rewards.items():
            name = prefix + 'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
