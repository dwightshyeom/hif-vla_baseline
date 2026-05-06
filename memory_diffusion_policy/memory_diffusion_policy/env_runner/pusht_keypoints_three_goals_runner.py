import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import os
import cv2
import wandb.sdk.data_types.video as wv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Polygon

from memory_diffusion_policy.env.pusht.pusht_keypoints_three_goals_env import PushTKeypointsThreeGoalsEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from memory_diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from memory_diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from memory_diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from memory_diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


# ---------------------------------------------------------------------------
# PushT geometry constants (for past-action visualisation)
# ---------------------------------------------------------------------------
_CANVAS_SIZE = 512
_T_SCALE     = 30
_T_LENGTH    = 4

_T_VERTS_CROSSBAR = np.array([
    [-_T_LENGTH * _T_SCALE / 2,  0],
    [ _T_LENGTH * _T_SCALE / 2,  0],
    [ _T_LENGTH * _T_SCALE / 2,  _T_SCALE],
    [-_T_LENGTH * _T_SCALE / 2,  _T_SCALE],
], dtype=float)

_T_VERTS_STEM = np.array([
    [-_T_SCALE / 2,  _T_SCALE],
    [-_T_SCALE / 2,  _T_LENGTH * _T_SCALE],
    [ _T_SCALE / 2,  _T_LENGTH * _T_SCALE],
    [ _T_SCALE / 2,  _T_SCALE],
], dtype=float)

_AGENT_RADIUS = 15.0

_GOAL_POSES = [
    np.array([160, 360, np.pi / 4]),
    np.array([256, 152, 0.0]),
    np.array([352, 360, -np.pi / 4]),
]

# Colours
_C_GT_AGENT   = "#4169e1"
_C_PAST_GT    = "#fb8500"   # orange
_C_PAST_PRED  = "#bc6c25"   # sienna
_C_GOAL       = "#90ee90"
_C_GOAL_EDGE  = "#228b22"


def _tblock_world_verts(x, y, angle):
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return (_T_VERTS_CROSSBAR @ R.T + [x, y],
            _T_VERTS_STEM     @ R.T + [x, y])


def _pose_from_keypoints(local_kps, global_kps):
    mu_l = local_kps.mean(axis=0)
    mu_g = global_kps.mean(axis=0)
    H_mat = (local_kps - mu_l).T @ (global_kps - mu_g)
    U, _, Vt = np.linalg.svd(H_mat)
    d  = np.linalg.det(Vt.T @ U.T)
    R  = Vt.T @ np.diag([1.0, d]) @ U.T
    angle = float(math.atan2(R[1, 0], R[0, 0]))
    trans = mu_g - R @ mu_l
    return float(trans[0]), float(trans[1]), angle


def _draw_goal_regions(ax, goal_poses=None):
    poses = goal_poses if goal_poses is not None else _GOAL_POSES
    for gp in poses:
        x, y, a = float(gp[0]), float(gp[1]), float(gp[2])
        for v in _tblock_world_verts(x, y, a):
            ax.add_patch(Polygon(v, closed=True, facecolor=_C_GOAL,
                                 edgecolor=_C_GOAL_EDGE, linewidth=0.8,
                                 alpha=0.35, zorder=1))


def _draw_trajectory(ax, points, base_color, alpha_range=(0.35, 0.95),
                     zorder=8, label=None):
    H = len(points)
    if H < 1:
        return
    alphas    = np.linspace(alpha_range[0], alpha_range[1], H)
    base_rgba = np.array(matplotlib.colors.to_rgba(base_color))
    for h in range(H):
        colour = (*base_rgba[:3], float(alphas[h]))
        ax.plot(points[h, 0], points[h, 1], 'o', color=colour,
                markersize=5, zorder=zorder, label=(label if h == 0 else None))
        if h > 0:
            ax.plot([points[h-1, 0], points[h, 0]],
                    [points[h-1, 1], points[h, 1]],
                    '-', color=colour, linewidth=1.5, zorder=zorder)
    start_col = (*base_rgba[:3], min(1.0, float(alphas[0])  + 0.3))
    end_col   = (*base_rgba[:3], min(1.0, float(alphas[-1]) + 0.2))
    ax.plot(points[0, 0],  points[0, 1],  '*', color=start_col, markersize=14,
            zorder=zorder+2, markeredgecolor='white', markeredgewidth=0.8)
    ax.plot(points[-1, 0], points[-1, 1], 'D', color=end_col,   markersize=7,
            zorder=zorder+2, markeredgecolor='white', markeredgewidth=0.8)


def _render_past_action_pdf(
    obs_now, agent_pos, action_history, past_pred,
    local_kps, seed, step, epoch, output_dir,
):
    """
    Render GT vs predicted past-action trajectories and save as PDF.

    Args:
        obs_now:        (20,) current observation (keypoints + agent).
        agent_pos:      (2,) current agent position.
        action_history: list of (2,) past *executed* actions (oldest→newest).
        past_pred:      (past_chunk_H, 2) predicted past actions (recent→older).
        local_kps:      (9, 2) local block keypoints for pose estimation.
        seed / step / epoch: for filename.
        output_dir:     pathlib.Path for saving.
    """
    past_H = len(past_pred)

    # Extract goal poses from observation when goal keypoints are included
    if obs_now.shape[-1] >= 74:
        goal_poses = []
        for g in range(3):
            start = 20 + g * 18
            goal_kps = obs_now[start:start + 18].reshape(9, 2)
            gx, gy, ga = _pose_from_keypoints(local_kps, goal_kps)
            goal_poses.append(np.array([gx, gy, ga]))
    else:
        goal_poses = None

    # GT past actions: indices [t-1, t-2, …, t-pastH] (recent→older)
    # slot 0 is a_{t-1}, the most recent action (just executed).
    n_hist = len(action_history)
    gt_points = []
    pred_points = []
    for i in range(past_H):
        idx = n_hist - 1 - i          # t-1, t-2, …
        if idx < 0:
            break
        gt_points.append(action_history[idx])
        pred_points.append(past_pred[i])
    if len(gt_points) < 2:
        return  # not enough history to draw

    gt_points   = np.array(gt_points)
    pred_points = np.array(pred_points)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, _CANVAS_SIZE)
    ax.set_ylim(_CANVAS_SIZE, 0)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_visible(False)

    _draw_goal_regions(ax, goal_poses=goal_poses)

    # Draw current T-block + agent
    kp_curr = obs_now[:18].reshape(9, 2)
    xc, yc, ac = _pose_from_keypoints(local_kps, kp_curr)
    for v in _tblock_world_verts(xc, yc, ac):
        ax.add_patch(Polygon(v, closed=True, facecolor="#778da9",
                             edgecolor="#3a4a5c", linewidth=1.2,
                             alpha=0.9, zorder=4))
    ax.add_patch(Circle(agent_pos, _AGENT_RADIUS, facecolor=_C_GT_AGENT,
                        edgecolor="#3a4a5c", linewidth=1.0, alpha=0.9, zorder=5))

    # Draw trajectories
    _draw_trajectory(ax, gt_points, _C_PAST_GT, alpha_range=(0.35, 0.85),
                     zorder=8, label="Past GT")
    _draw_trajectory(ax, pred_points, _C_PAST_PRED, alpha_range=(0.35, 0.85),
                     zorder=9, label="Past pred")

    ax.legend(loc="upper right", fontsize=9,
              handles=[
                  mpatches.Patch(facecolor=_C_PAST_GT,   label="Past GT (orange)"),
                  mpatches.Patch(facecolor=_C_PAST_PRED, label="Past pred (sienna)"),
              ])
    ax.set_title(f"epoch {epoch}  seed {seed}  step {step}", fontsize=10)

    pdf_path = output_dir / f"{epoch}_{seed}_{step}.pdf"
    plt.savefig(pdf_path, dpi=110, format="pdf", bbox_inches="tight")
    plt.close(fig)


class PushTKeypointsThreeGoalsRunner(BaseLowdimRunner):
    """
    Environment runner for PushT three goals task.
    Copied from PushTKeypointsTwoGoalsRunner and adapted for three-goal task.
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
            max_steps=500,  # Increased from 400 to allow visiting all three goals
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
            goal_pose_2=None,  # Second goal position
            goal_pose_3=None,  # Third goal position
            goal_pos_variation=0.0,  # Max positional perturbation in pixels
            goal_rot_variation=0.0,  # Max rotational perturbation in radians
            include_goal_keypoints=False  # Include goal keypoints in observation
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
        kp_kwargs = PushTKeypointsThreeGoalsEnv.genenerate_keypoint_manager_params()

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTKeypointsThreeGoalsEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        goal_pose_1=goal_pose_1,
                        goal_pose_2=goal_pose_2,
                        goal_pose_3=goal_pose_3,
                        goal_pos_variation=goal_pos_variation,
                        goal_rot_variation=goal_rot_variation,
                        include_goal_keypoints=include_goal_keypoints,
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
    
    def run(self, policy: BaseLowdimPolicy, epoch: int = 0):
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

        # ── Past-action visualisation setup ──────────────────────────────
        _has_past_head = hasattr(policy, 'predict_past_actions')
        _past_viz_dir = None
        _local_kps = None
        _viz_rng = None
        if _has_past_head:
            _past_viz_dir = pathlib.Path(self.output_dir) / "past_action_viz"
            _past_viz_dir.mkdir(parents=True, exist_ok=True)
            # Get local block keypoints for pose estimation
            from memory_diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
            from memory_diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager
            kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
            kp_manager = PymunkKeypointManager(**kp_kwargs)
            _local_kps = kp_manager.local_keypoint_map["block"]
            # Per-epoch RNG for random sampling of seeds/steps
            _viz_rng = np.random.RandomState(seed=epoch * 31 + 7)

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
            
            # Initialize step counter for each environment
            current_step_per_env = np.zeros(this_n_active_envs, dtype=np.int32)
            
            # ── Past-action viz: per-env action history & sampling ───────
            # action_history[env_idx] = list of (action_dim,) arrays (oldest→newest)
            action_history_per_env = [[] for _ in range(this_n_active_envs)]
            # Randomly choose which envs and at which call-step to visualise
            _viz_targets = set()   # set of (env_idx, call_step)
            if _has_past_head and _viz_rng is not None:
                # Randomly pick up to 4 envs from this chunk
                n_pick_envs = min(4, this_n_active_envs)
                picked_envs = _viz_rng.choice(
                    this_n_active_envs, size=n_pick_envs, replace=False)
                # Compute total expected call-steps in an episode
                total_calls = max(1, self.max_steps // self.n_action_steps)
                # For each picked env, pick 5 call-step indices (not too early)
                n_pick_steps = min(5, max(1, total_calls - 10))
                for ei in picked_envs:
                    earliest = min(10, total_calls - 1)  # skip first ~10 steps
                    if earliest < total_calls - 1:
                        picked_steps = _viz_rng.choice(
                            np.arange(earliest, total_calls), size=n_pick_steps,
                            replace=False)
                    else:
                        picked_steps = [earliest]
                    for cs in picked_steps:
                        _viz_targets.add((int(ei), int(cs)))
            # call_step counter (incremented once per policy predict_action)
            call_step = 0
            
            # Initialize goals_reached buffer for tracking indicator across timesteps
            # goals_reached is a (3,) array: [goal_1_reached, goal_2_reached, goal_3_reached]
            goals_reached_buffer = collections.deque(maxlen=self.n_obs_steps)
            # Initialize with zeros (no goals reached at start)
            for _ in range(self.n_obs_steps):
                goals_reached_buffer.append(np.zeros((this_n_active_envs, 3), dtype=np.float32))
            
            # Track LSTM indicator predictions for visualization (if using LSTM indicator)
            # Store per-environment predictions: list of (3,) arrays
            lstm_predictions_per_env = [[] for _ in range(this_n_active_envs)]
            
            # Track LSTM progression predictions for visualization (if using LSTM progression)
            # Store per-environment predictions: list of floats
            lstm_progression_per_env = [[] for _ in range(this_n_active_envs)]

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval ThreeGoalsRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                # create obs dict
                Do = obs.shape[-1] // 2
                
                # Use full sliding window of observations for proper temporal context
                # obs from MultiStepWrapper: (n_envs, n_obs_steps, obs_dim)
                # Extract keypoints + agent_pos: (n_envs, n_obs_steps, 20)
                raw_obs = obs[:, :, :Do].astype(np.float32)
                obs_mask = obs[:, :, Do:] > 0.5
                
                # Create obs dict with full n_obs_steps timesteps
                np_obs_dict = {
                    'obs': raw_obs,  # (n_envs, n_obs_steps, 20)
                    'obs_mask': obs_mask  # (n_envs, n_obs_steps, 20)
                }
                
                # Add goals_reached indicator with full n_obs_steps history
                # Stack the buffer: (n_obs_steps, n_envs, 3) -> (n_envs, n_obs_steps, 3)
                goals_reached_history = np.stack(list(goals_reached_buffer), axis=1)  # (n_envs, n_obs_steps, 3)
                np_obs_dict['goals_reached'] = goals_reached_history.astype(np.float32)
                
                # Calculate step-based progression for non-LSTM mode
                # If policy uses non-LSTM progression, it needs step/max_steps for each env
                # progression = current_step / max_steps, range [0, 1]
                if (hasattr(policy, 'use_lstm_progression') and not policy.use_lstm_progression):
                    step_progression_values = current_step_per_env / self.max_steps  # (n_envs,)
                    step_progression_values = np.minimum(step_progression_values, 1.0)  # Clamp to [0, 1]
                    # Expand to match obs shape: (n_envs, n_obs_steps, 1)
                    step_progression = np.tile(step_progression_values[:, np.newaxis, np.newaxis], 
                                               (1, self.n_obs_steps, 1))  # (n_envs, n_obs_steps, 1)
                    np_obs_dict['progression'] = step_progression.astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                
                # If using LSTM indicator, capture predictions for visualization
                if hasattr(policy, 'use_lstm_indicator') and policy.use_lstm_indicator:
                    # Get LSTM predictions from the most recent timestep
                    # obs_dict['obs'] is (n_envs, n_obs_steps, 20)
                    # We want predictions for the newest observation
                    with torch.no_grad():
                        lstm_indicator = policy.predict_indicator_with_lstm(obs_dict['obs'])
                        # lstm_indicator: (n_envs, n_obs_steps, 3)
                        # Get the last timestep prediction for each env
                        current_lstm_pred = lstm_indicator[:, -1, :].cpu().numpy()  # (n_envs, 3)
                        
                        # Store for each environment
                        for env_idx in range(this_n_active_envs):
                            lstm_predictions_per_env[env_idx].append(current_lstm_pred[env_idx])  # (3,)
                
                # If using LSTM progression, capture predictions for visualization
                if hasattr(policy, 'use_lstm_progression') and policy.use_lstm_progression:
                    # Get LSTM progression predictions from the most recent timestep
                    # obs_dict['obs'] is (n_envs, n_obs_steps, 20)
                    with torch.no_grad():
                        lstm_progression = policy.predict_progression_with_lstm(obs_dict['obs'])
                        # lstm_progression: (n_envs, n_obs_steps, 1)
                        # Get the last timestep prediction for each env
                        current_prog_pred = lstm_progression[:, -1, 0].cpu().numpy()  # (n_envs,)
                        
                        # Store for each environment
                        for env_idx in range(this_n_active_envs):
                            lstm_progression_per_env[env_idx].append(float(current_prog_pred[env_idx]))

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]

                # ── Past-action visualisation ────────────────────────────
                if _has_past_head and _past_viz_dir is not None:
                    # Check if any env needs viz at this call_step
                    envs_to_viz = [ei for ei in range(this_n_active_envs)
                                   if (ei, call_step) in _viz_targets]
                    if envs_to_viz:
                        past_pred_t = policy.predict_past_actions()
                        if past_pred_t is not None:
                            for env_idx in envs_to_viz:
                                pred_np = past_pred_t[env_idx].cpu().numpy()
                                obs_now = raw_obs[env_idx, -1]
                                agent_pos = obs_now[18:20]
                                seed_val = self.env_seeds[start + env_idx]
                                _render_past_action_pdf(
                                    obs_now=obs_now,
                                    agent_pos=agent_pos,
                                    action_history=action_history_per_env[env_idx],
                                    past_pred=pred_np,
                                    local_kps=_local_kps,
                                    seed=seed_val,
                                    step=call_step,
                                    epoch=epoch,
                                    output_dir=_past_viz_dir,
                                )

                # Calculate and display step-based progression on video overlay BEFORE taking step
                # This ensures the overlay matches the current state when the frame is captured
                step_based_text_overlays = []
                for env_idx in range(this_n_active_envs):
                    step_progression = current_step_per_env[env_idx] / self.max_steps
                    step_progression = min(step_progression, 1.0)  # Clamp to [0, 1]
                    indicator_percentage = goals_reached_history[env_idx].sum(axis=-1)[-1] / 3.0  # Percentage of goals achieved at the most recent timestep, range [0, 1]
                    # Format: S:0.45 (step-based progression)
                    # text = f"S:{step_progression:.2f}"
                    text = f"{indicator_percentage:.2f}"  # Add indicator percentage to overlay
                    step_based_text_overlays.append(text)
                
                # Set step-based text overlay on each env's VideoRecordingWrapper
                def _set_step_overlay_fn(env_wrapper, text):
                    """Helper function to set step-based progression overlay"""
                    # env_wrapper is MultiStepWrapper, env_wrapper.env is VideoRecordingWrapper
                    if hasattr(env_wrapper.env, 'step_based_overlay'):
                        env_wrapper.env.step_based_overlay = text
                    return None
                
                env.call_each('run_dill_function', 
                    args_list=[[dill.dumps(lambda env, t=text: _set_step_overlay_fn(env, t))] 
                               for text in step_based_text_overlays])

                # Set text overlay for video recording (LSTM predictions)
                if hasattr(policy, 'use_lstm_indicator') and policy.use_lstm_indicator:
                    # Update overlay for each environment
                    # Build list of text overlays for each env
                    text_overlays = []
                    for env_idx in range(this_n_active_envs):
                        if len(lstm_predictions_per_env[env_idx]) > 0:
                            pred = lstm_predictions_per_env[env_idx][-1]  # (3,) latest prediction
                            # Format: G1:1 G2:0 G3:1 (binary indicators)
                            text = f"{int(pred[0])} {int(pred[1])} {int(pred[2])}"
                            text_overlays.append(text)
                        else:
                            text_overlays.append(None)
                    
                    # Set text_overlay on each env's VideoRecordingWrapper
                    # For AsyncVectorEnv, we need to use call_each to execute on each worker
                    def _set_overlay_fn(env_wrapper, text):
                        """Helper function to set text overlay on VideoRecordingWrapper"""
                        # env_wrapper is MultiStepWrapper, env_wrapper.env is VideoRecordingWrapper
                        if hasattr(env_wrapper.env, 'text_overlay'):
                            env_wrapper.env.text_overlay = text
                        return None
                    
                    # Create args list: each element is [text] for that environment
                    args_list = [[text] for text in text_overlays]
                    env.call_each('run_dill_function', 
                        args_list=[[dill.dumps(lambda env, t=text: _set_overlay_fn(env, t))] 
                                   for text in text_overlays])
                
                # Set text overlay for video recording (LSTM progression predictions)
                if hasattr(policy, 'use_lstm_progression') and policy.use_lstm_progression:
                    # Update overlay for each environment
                    # Build list of text overlays for each env
                    text_overlays = []
                    for env_idx in range(this_n_active_envs):
                        if len(lstm_progression_per_env[env_idx]) > 0:
                            prog = lstm_progression_per_env[env_idx][-1]  # float, latest prediction
                            # Format: P:0.65 (progression value with 2 decimal places)
                            text = f"{prog:.2f}"
                            text_overlays.append(text)
                        else:
                            text_overlays.append(None)
                    
                    # Set text_overlay on each env's VideoRecordingWrapper
                    # For AsyncVectorEnv, we need to use call_each to execute on each worker
                    def _set_prog_overlay_fn(env_wrapper, text):
                        """Helper function to set text overlay on VideoRecordingWrapper"""
                        # env_wrapper is MultiStepWrapper, env_wrapper.env is VideoRecordingWrapper
                        if hasattr(env_wrapper.env, 'text_overlay'):
                            env_wrapper.env.text_overlay = text
                        return None
                    
                    # Create args list: each element is [text] for that environment
                    env.call_each('run_dill_function', 
                        args_list=[[dill.dumps(lambda env, t=text: _set_prog_overlay_fn(env, t))] 
                                   for text in text_overlays])
                
                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                
                # Pass intermediate observations to the policy for LSTM
                # multi-step update (if the policy supports it).
                if hasattr(policy, 'update_lstm_intermediate_obs'):
                    # Retrieve per-substep obs from each env
                    # Returns list of arrays, each (n_substeps, obs_dim*2)
                    # n_substeps may be < n_action_steps if env terminated early
                    intermediate_obs_list = env.call('get_intermediate_obs')
                    n_act = self.n_action_steps
                    padded = []
                    for arr in intermediate_obs_list[:this_n_active_envs]:
                        if arr is None or len(arr) == 0:
                            # No obs collected (shouldn't happen, but be safe)
                            padded.append(np.tile(obs[0:1, -1, :], (n_act, 1)))
                        elif len(arr) < n_act:
                            # Env terminated early — repeat last obs to fill
                            pad = np.tile(arr[-1:], (n_act - len(arr), 1))
                            padded.append(np.concatenate([arr, pad], axis=0))
                        else:
                            padded.append(arr[:n_act])
                    intermediate_obs_all = np.stack(padded)  # (n_envs, n_act, obs_dim*2)
                    # Extract keypoints (first half): (n_envs, n_act, 20)
                    Do = intermediate_obs_all.shape[-1] // 2
                    intermediate_raw = intermediate_obs_all[:, :, :Do].astype(
                        np.float32)
                    policy.update_lstm_intermediate_obs(
                        torch.from_numpy(intermediate_raw).to(device=device))
                
                # Increment step counter for each environment after taking action
                current_step_per_env += self.n_action_steps
                
                past_action = action

                # ── Update per-env action history for past-action viz ────
                if _has_past_head:
                    for env_idx in range(this_n_active_envs):
                        # action shape: (n_envs, n_action_steps, action_dim)
                        for k in range(action.shape[1]):
                            action_history_per_env[env_idx].append(
                                action[env_idx, k].copy())

                # Increment call_step
                call_step += 1
                
                # Extract goals_reached from info and update buffer
                # info is a list of dicts, one per environment
                # Each dict contains arrays of shape (n_obs_steps,) from MultiStepWrapper
                # We need to extract the most recent goals_reached for each env
                current_goals_reached_list = []
                for env_idx in range(this_n_active_envs):
                    env_info = info[env_idx]
                    # Extract the most recent (last) timestep from the n_obs_steps
                    goal_1 = float(env_info['goal_1_reached'][-1])
                    goal_2 = float(env_info['goal_2_reached'][-1])
                    goal_3 = float(env_info['goal_3_reached'][-1])
                    current_goals_reached_list.append([goal_1, goal_2, goal_3])
                
                current_goals_reached = np.array(current_goals_reached_list, dtype=np.float32)  # (n_envs, 3)
                goals_reached_buffer.append(current_goals_reached)

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
                # Check if file exists and has content before logging
                import os
                if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                    sim_video = wandb.Video(video_path)
                    log_data[prefix+f'sim_video_{seed}'] = sim_video
                    print(f"Logging video for {prefix}seed {seed}: {video_path} ({os.path.getsize(video_path)} bytes)")
                else:
                    print(f"Warning: Video file {video_path} does not exist or is empty for {prefix}seed {seed}")
            else:
                print(f"Warning: No video path for {prefix}seed {seed} (video_path is None)")

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
