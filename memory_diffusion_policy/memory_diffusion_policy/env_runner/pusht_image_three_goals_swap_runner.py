"""
Env runner for the Push-T three-goals swap task (image observations).

Mirrors PushTImageTwoSwapRunner / PushTImageThreeGoalsRunner. Supports:
  - vanilla DP policies (only predict_action() needed)
  - LSTM+DP policies via predict_past_actions() (past-action PDFs) and
    update_lstm_with_intermediates() (intermediate-obs feeding)
  - "wandb" / "local" rollout_mode (videos to wandb vs. on-disk per epoch)

Per-episode swap-role randomisation (which target is empty, which color
goes where) is driven entirely by env.seed(); the runner only assigns
distinct seeds per env.
"""

import wandb
import numpy as np
import torch
import collections
import gc
import pathlib
import shutil
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle

from memory_diffusion_policy.env.pusht.pusht_image_three_goals_swap_env import (
    PushTImageThreeGoalsSwapEnv,
)
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from memory_diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from memory_diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecordingWrapper, VideoRecorder,
)

from memory_diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from memory_diffusion_policy.env_runner.base_image_runner import BaseImageRunner


# ---------------------------------------------------------------------------
# Past-action visualisation helpers (image-based runner)
# Identical to the two-swap / three-goals image runners so PDFs look
# uniform across tasks.
# ---------------------------------------------------------------------------
_CANVAS_SIZE = 512
_AGENT_RADIUS = 15.0

_C_GT_AGENT = "#4169e1"
_C_PAST_GT = "#fb8500"
_C_PAST_PRED = "#bc6c25"


def _draw_trajectory(ax, points, base_color, alpha_range=(0.35, 0.95),
                     zorder=8, label=None):
    H = len(points)
    if H < 1:
        return
    alphas = np.linspace(alpha_range[0], alpha_range[1], H)
    base_rgba = np.array(matplotlib.colors.to_rgba(base_color))
    for h in range(H):
        colour = (*base_rgba[:3], float(alphas[h]))
        ax.plot(points[h, 0], points[h, 1], 'o', color=colour,
                markersize=5, zorder=zorder, label=(label if h == 0 else None))
        if h > 0:
            ax.plot([points[h - 1, 0], points[h, 0]],
                    [points[h - 1, 1], points[h, 1]],
                    '-', color=colour, linewidth=1.5, zorder=zorder)
    start_col = (*base_rgba[:3], min(1.0, float(alphas[0]) + 0.3))
    end_col = (*base_rgba[:3], min(1.0, float(alphas[-1]) + 0.2))
    ax.plot(points[0, 0], points[0, 1], '*', color=start_col, markersize=14,
            zorder=zorder + 2, markeredgecolor='white', markeredgewidth=0.8)
    ax.plot(points[-1, 0], points[-1, 1], 'D', color=end_col, markersize=7,
            zorder=zorder + 2, markeredgecolor='white', markeredgewidth=0.8)


def _render_past_action_pdf(
    agent_pos, action_history, past_pred,
    seed, step, epoch, output_dir,
    env_image=None,
):
    past_H = len(past_pred)
    n_hist = len(action_history)
    gt_points = []
    pred_points = []
    for i in range(past_H):
        idx = n_hist - 1 - i
        if idx < 0:
            break
        gt_points.append(action_history[idx])
        pred_points.append(past_pred[i])
    if len(gt_points) < 2:
        return

    gt_points = np.array(gt_points)
    pred_points = np.array(pred_points)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, _CANVAS_SIZE)
    ax.set_ylim(_CANVAS_SIZE, 0)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_visible(False)

    if env_image is not None:
        img_hw3 = np.moveaxis(env_image, 0, -1)
        ax.imshow(img_hw3, extent=[0, _CANVAS_SIZE, _CANVAS_SIZE, 0],
                  interpolation='bilinear', zorder=0)

    ax.add_patch(Circle(agent_pos, _AGENT_RADIUS, facecolor=_C_GT_AGENT,
                        edgecolor="#3a4a5c", linewidth=1.0, alpha=0.9, zorder=5))

    _draw_trajectory(ax, gt_points, _C_PAST_GT, alpha_range=(0.35, 0.85),
                     zorder=8, label="Past GT")
    _draw_trajectory(ax, pred_points, _C_PAST_PRED, alpha_range=(0.35, 0.85),
                     zorder=9, label="Past pred")

    ax.legend(loc="upper right", fontsize=9,
              handles=[
                  mpatches.Patch(facecolor=_C_PAST_GT, label="Past GT (orange)"),
                  mpatches.Patch(facecolor=_C_PAST_PRED, label="Past pred (sienna)"),
              ])
    ax.set_title(f"epoch {epoch}  seed {seed}  step {step}", fontsize=10)

    pdf_path = output_dir / f"{epoch}_{seed}_{step}.pdf"
    plt.savefig(pdf_path, dpi=110, format="pdf", bbox_inches="tight")
    plt.close(fig)


class PushTImageThreeGoalsSwapRunner(BaseImageRunner):
    def __init__(
        self,
        output_dir,
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
        fps=10,
        crf=22,
        render_size=96,
        past_action=False,
        tqdm_interval_sec=5.0,
        n_envs=None,
        rollout_mode="wandb",
    ):
        super().__init__(output_dir)
        if rollout_mode not in ("wandb", "local"):
            raise ValueError(
                f"Unknown rollout_mode={rollout_mode!r}. "
                "Choose from: 'wandb', 'local'."
            )
        if n_envs is None:
            n_envs = n_train + n_test

        steps_per_render = max(10 // fps, 1)

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTImageThreeGoalsSwapEnv(
                        legacy=legacy_test,
                        render_size=render_size,
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
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
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
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.rollout_mode = rollout_mode

    def run(self, policy: BaseImagePolicy, epoch: int = 0):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        # Past-action viz hook (only fires if policy implements predict_past_actions)
        _has_past_head = hasattr(policy, 'predict_past_actions')
        _past_viz_dir = None
        _viz_rng = None
        if _has_past_head:
            _past_viz_dir = pathlib.Path(self.output_dir) / "past_action_viz"
            _past_viz_dir.mkdir(parents=True, exist_ok=True)
            _viz_rng = np.random.RandomState(seed=epoch * 31 + 7)

        local_video_dir = None
        if self.rollout_mode == "local":
            local_video_dir = pathlib.Path(self.output_dir) / f"epoch_{epoch}_rollout"
            local_video_dir.mkdir(parents=True, exist_ok=True)

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

            action_history_per_env = [[] for _ in range(this_n_active_envs)]
            _viz_targets = set()
            if _has_past_head and _viz_rng is not None:
                n_pick_envs = min(4, this_n_active_envs)
                picked_envs = _viz_rng.choice(
                    this_n_active_envs, size=n_pick_envs, replace=False)
                total_calls = max(1, self.max_steps // self.n_action_steps)
                n_pick_steps = min(5, max(1, total_calls - 10))
                for ei in picked_envs:
                    earliest = min(10, total_calls - 1)
                    if earliest < total_calls - 1:
                        picked_steps = _viz_rng.choice(
                            np.arange(earliest, total_calls),
                            size=n_pick_steps, replace=False)
                    else:
                        picked_steps = [earliest]
                    for cs in picked_steps:
                        _viz_targets.add((int(ei), int(cs)))
            call_step = 0

            pbar = tqdm.tqdm(total=self.max_steps,
                desc=f"Eval PushTImageThreeGoalsSwap {chunk_idx + 1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    np_obs_dict['past_action'] = past_action[
                        :, -(self.n_obs_steps - 1):].astype(np.float32)

                obs_dict = dict_apply(np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device))

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']

                # Past-action viz
                if _has_past_head and _past_viz_dir is not None:
                    envs_to_viz = [ei for ei in range(this_n_active_envs)
                                   if (ei, call_step) in _viz_targets]
                    if envs_to_viz:
                        past_pred_t = policy.predict_past_actions()
                        if past_pred_t is not None:
                            for env_idx in envs_to_viz:
                                pred_np = past_pred_t[env_idx].cpu().numpy()
                                ap = np_obs_dict['agent_pos'][env_idx, -1]
                                env_img = np_obs_dict['image'][env_idx, -1]
                                seed_val = self.env_seeds[start + env_idx]
                                _render_past_action_pdf(
                                    agent_pos=ap,
                                    action_history=action_history_per_env[env_idx],
                                    past_pred=pred_np,
                                    seed=seed_val,
                                    step=call_step,
                                    epoch=epoch,
                                    output_dir=_past_viz_dir,
                                    env_image=env_img,
                                )

                # Phase + per-block coverage overlay on the video.
                def _set_progress_overlay(env_wrapper):
                    inner = env_wrapper.env.env
                    cov_b, cov_r = inner._swap_coverages()
                    env_wrapper.env.step_based_overlay = (
                        f"P{inner.phase} B:{cov_b:.2f} R:{cov_r:.2f}")
                    return None

                env.call_each('run_dill_function',
                    args_list=[(dill.dumps(_set_progress_overlay),)] * n_envs)

                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # Feed intermediate observations to LSTM if policy supports it.
                if not done and getattr(policy, '_feed_intermediate_obs', False):
                    intermediate_obs_list = env.call('get_intermediate_obs_dict')
                    first_valid = next(
                        (r for r in intermediate_obs_list if r is not None), None)
                    if first_valid is not None:
                        stacked = {}
                        for key in first_valid.keys():
                            stacked[key] = np.stack([
                                r[key] if r is not None
                                else np.zeros_like(first_valid[key])
                                for r in intermediate_obs_list])
                        intermediate_obs_torch = dict_apply(
                            stacked,
                            lambda x: torch.from_numpy(x).to(device=device))
                        actions_torch = torch.from_numpy(action).to(
                            device=device, dtype=torch.float32)
                        with torch.no_grad():
                            policy.update_lstm_with_intermediates(
                                intermediate_obs_torch, actions_torch)

                if _has_past_head:
                    for env_idx in range(this_n_active_envs):
                        for k in range(action.shape[1]):
                            action_history_per_env[env_idx].append(
                                action[env_idx, k].copy())
                call_step += 1

                pbar.update(action.shape[1])
            pbar.close()

            chunk_paths = env.render()[this_local_slice]
            all_video_paths[this_global_slice] = chunk_paths
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

            if self.rollout_mode == "local":
                for j, vp in enumerate(chunk_paths):
                    if vp is None:
                        continue
                    global_i = start + j
                    seed_v = self.env_seeds[global_i]
                    prefix_v = self.env_prefixs[global_i]
                    src = pathlib.Path(vp)
                    if src.exists():
                        tag = prefix_v.rstrip('/').replace('/', '_') or 'env'
                        dest = local_video_dir / f"{tag}_seed_{seed_v}{src.suffix}"
                        try:
                            shutil.move(str(src), str(dest))
                        except Exception as e:
                            print(f"[rollout_mode=local] failed to move "
                                  f"{src} → {dest}: {e}")
                    all_video_paths[global_i] = None

                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

        _ = env.reset()

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            # Reward is 1.0 only on success (all phases completed without
            # protocol violation). Use >= 1.0 as the binary success signal.
            success = 1.0 if max_reward >= 1.0 else 0.0
            max_rewards[prefix].append(success)
            log_data[prefix + f'sim_max_reward_{seed}'] = max_reward
            log_data[prefix + f'sim_success_{seed}'] = success

            video_path = all_video_paths[i]
            if video_path is not None:
                if self.rollout_mode == "local":
                    src = pathlib.Path(video_path)
                    if src.exists():
                        tag = prefix.rstrip('/').replace('/', '_') or 'env'
                        dest = local_video_dir / f"{tag}_seed_{seed}{src.suffix}"
                        try:
                            shutil.move(str(src), str(dest))
                        except Exception as e:
                            print(f"[rollout_mode=local] failed to move "
                                  f"{src} → {dest}: {e}")
                else:
                    sim_video = wandb.Video(video_path)
                    log_data[prefix + f'sim_video_{seed}'] = sim_video

        for prefix, value in max_rewards.items():
            name = prefix + 'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
