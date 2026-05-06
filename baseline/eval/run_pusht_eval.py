"""
run_pusht_eval.py

Evaluates a fine-tuned HiF-VLA checkpoint on the Swap-T (Push-T) environment.

Usage:
    python baseline/eval/run_pusht_eval.py \\
        --pretrained_checkpoint /path/to/checkpoint \\
        --zarr_path memory_diffusion_policy/swap_3t_dataset_320 \\
        --num_episodes 50 \\
        --max_steps 500 \\
        --unnorm_key swap_3t

The script:
  1. Loads the VLA, action head, proprio projector, motion manager, and motion
     encoder from the checkpoint directory.
  2. Instantiates the PushTKeypointsThreeGoalsSwapEnv.
  3. For each episode, maintains a rolling window of optical-flow frames and
     feeds them as ``his_motion_seq`` to ``predict_action``.
  4. Reports per-episode success, average coverage, and overall success rate.
"""

import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Path setup — "pusht" in argv[0] triggers PUSHT_CONSTANTS in constants.py
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = Path(__file__).resolve().parent       # baseline/eval/
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent             # project_root/
_HIF_VLA_ROOT = _PROJECT_ROOT / "HiF-VLA"
_MEM_DIFF_ROOT = _PROJECT_ROOT / "memory_diffusion_policy"

for _p in [str(_PROJECT_ROOT), str(_HIF_VLA_ROOT), str(_MEM_DIFF_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# diffusion_policy is not installed as a pip package; locate it from known paths.
# We require the package to have env/pusht/pymunk_override.py (not just an empty dir).
def _add_diffusion_policy_to_path():
    _sentinel = Path("env") / "pusht" / "pymunk_override.py"

    def _has_real_pkg(parent: Path) -> bool:
        return (parent / "diffusion_policy" / _sentinel).is_file()

    # Already importable and has the env subpackage?
    try:
        import diffusion_policy
        import importlib
        dp_path = Path(diffusion_policy.__file__).parent
        if (dp_path / _sentinel).is_file():
            return
        # Found but incomplete — remove from sys.path and keep searching
        sys.path = [p for p in sys.path if Path(p) != dp_path.parent]
    except ImportError:
        pass

    candidates = [
        # common developer workspace layouts — check these before third_party
        Path.home() / "workspace" / "diffusion_policy",
        Path.home() / "Desktop" / "diffusion_policy",
        Path.home() / "workspace" / "diffusion_policy_original",
        Path.home() / "workspace" / "score_diffusion_policy",
        # vendored inside memory_diffusion_policy (may be empty on some setups)
        _MEM_DIFF_ROOT / "third_party",
    ]
    for candidate in candidates:
        if _has_real_pkg(candidate):
            _s = str(candidate)
            if _s not in sys.path:
                sys.path.insert(0, _s)
            return
    raise ImportError(
        "Could not locate a complete 'diffusion_policy' package (needs env/pusht/pymunk_override.py). "
        "Add its parent directory to PYTHONPATH or install it with pip."
    )

_add_diffusion_policy_to_path()

import draccus
import numpy as np
import torch
import cv2
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    get_action_head,
    get_hismotion_encoder,
    get_motion_manager,
    get_proprio_projector,
    get_vla,
    model_is_on_hf_hub,
    normalize_proprio,
    update_auto_map,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)

from baseline.dataset.optical_flow import (
    FLOW_H,
    FLOW_W,
    compute_frame_flow,
    load_flow_stats,
    normalize_flow,
)
from baseline.dataset.pusht_hifvla_dataset import DATASET_NAME, PUSHT_LANGUAGE_INSTRUCTION

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    # fmt: off
    pretrained_checkpoint: Union[str, Path] = ""   # Path to fine-tuned checkpoint dir
    zarr_path: str = ""                            # Zarr store (only used to locate flow_stats)
    flow_stats_path: Optional[str] = None          # Path to flow_stats.npz (or auto-detected)

    model_family: str = "openvla"
    use_l1_regression: bool = True
    use_film: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = True
    center_crop: bool = True
    lora_rank: int = 32
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    unnorm_key: str = DATASET_NAME                 # "swap_3t"
    history_length: int = 8

    num_episodes: int = 50
    max_steps: int = 500
    num_open_loop_steps: int = 8                   # Full action chunk by default
    seed: int = 42

    # Logging
    log_dir: str = "experiments/logs/pusht"
    run_id_note: Optional[str] = None

    # Video recording
    save_video: bool = True            # Save MP4 rollout videos
    n_video_episodes: int = 10         # How many episodes to record (first N)
    video_fps: int = 10                # Playback FPS
    # fmt: on


# ---------------------------------------------------------------------------
# Optical flow helpers for online rollout
# ---------------------------------------------------------------------------

class FlowBuffer:
    """
    Maintains a rolling deque of raw grayscale frames and computes the last
    ``history_length`` normalised optical-flow tensors on demand.
    """

    def __init__(
        self,
        history_length: int,
        flow_mean: np.ndarray,
        flow_std: np.ndarray,
    ) -> None:
        self.history_length = history_length
        self.flow_mean = flow_mean
        self.flow_std  = flow_std
        # Keep history_length + 1 frames so we can compute history_length flows.
        self._frames: deque = deque(maxlen=history_length + 1)

    def push(self, rgb_frame: np.ndarray) -> None:
        """Add a new uint8 RGB frame (H, W, 3)."""
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        self._frames.append(gray)

    def get_flow_tensor(self) -> torch.Tensor:
        """
        Return a ``(1, history_length, 2, FLOW_H, FLOW_W)`` bfloat16 tensor
        ready to pass as ``his_motion_seq`` to ``vla.predict_action``.

        Frames without a predecessor are zero-padded.
        """
        frames = list(self._frames)
        result = np.zeros((self.history_length, 2, FLOW_H, FLOW_W), dtype=np.float32)
        # Compute flows for consecutive pairs in the buffer.
        n_flows = min(len(frames) - 1, self.history_length)
        start_buf = len(frames) - 1 - n_flows   # index of first "prev" frame
        start_out = self.history_length - n_flows
        for i in range(n_flows):
            prev = frames[start_buf + i]
            curr = frames[start_buf + i + 1]
            result[start_out + i] = compute_frame_flow(prev, curr)
        # Normalise
        result = normalize_flow(result, self.flow_mean, self.flow_std)
        return torch.from_numpy(result[None]).to(DEVICE, dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_components(cfg: EvalConfig):
    """Load all HiF-VLA checkpoint components."""
    import dataclasses
    from peft import PeftModel

    ckpt_dir  = Path(cfg.pretrained_checkpoint)
    lora_dir  = ckpt_dir / "lora_adapter"
    is_lora_ckpt = lora_dir.exists() and (lora_dir / "adapter_config.json").exists()

    if is_lora_ckpt:
        # Training saved only the LoRA adapter + auxiliary heads.
        # We must load the frozen base VLA first, then apply the adapter.
        with open(lora_dir / "adapter_config.json") as f:
            adapter_cfg_json = json.load(f)
        base_model_path = str(Path(adapter_cfg_json["base_model_name_or_path"]).resolve())
        print(f"LoRA checkpoint detected.  Base model : {base_model_path}")
        print(f"                           LoRA adapter: {lora_dir}")

        # Build a temporary cfg that points get_vla() at the base model.
        base_cfg = dataclasses.replace(cfg, pretrained_checkpoint=base_model_path)

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        update_auto_map(base_model_path)
        check_model_logic_mismatch(base_model_path)

        # Load base VLA (weights only, no dataset stats yet).
        vla = get_vla(base_cfg)

        # Apply LoRA adapter on top of the base model.
        vla = PeftModel.from_pretrained(vla, str(lora_dir))
        vla.eval()

        # Load dataset stats from the fine-tuned checkpoint dir (not base).
        from experiments.robot.openvla_utils import _load_dataset_stats
        # Set norm_stats on the base model (predict_action runs as self=base model).
        _load_dataset_stats(vla.get_base_model(), str(ckpt_dir))

        # Processor lives in the checkpoint dir (saved by processor.save_pretrained).
        processor = AutoProcessor.from_pretrained(str(ckpt_dir), trust_remote_code=True)

        # All auxiliary heads are loaded from the checkpoint dir via cfg.
        llm_dim = vla.get_base_model().llm_dim
    else:
        # Standard path: checkpoint dir contains the full model (merged or HF Hub).
        print(f"Loading checkpoint: {cfg.pretrained_checkpoint}")
        if not model_is_on_hf_hub(str(cfg.pretrained_checkpoint)):
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
            update_auto_map(str(cfg.pretrained_checkpoint))
            check_model_logic_mismatch(str(cfg.pretrained_checkpoint))
        vla       = get_vla(cfg)
        processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
        llm_dim   = vla.llm_dim

    action_head       = get_action_head(cfg, llm_dim) if cfg.use_l1_regression else None
    proprio_projector = get_proprio_projector(cfg, llm_dim, PROPRIO_DIM) if cfg.use_proprio else None
    motion_manager    = get_motion_manager(cfg, llm_dim)
    motion_encoder    = get_hismotion_encoder(
        cfg,
        in_channels=2,
        hidden_dim=llm_dim // 4,
        out_dim=llm_dim // 4,
        num_frames=cfg.history_length // 2,
        num_patches=64,
    )

    motion_token = motion_manager.get_motion_token(batch_size=1)

    print(f"Loaded: VLA llm_dim={llm_dim}, ACTION_DIM={ACTION_DIM}, PROPRIO_DIM={PROPRIO_DIM}")
    return vla, processor, action_head, proprio_projector, motion_token, motion_encoder


# ---------------------------------------------------------------------------
# Observation preparation
# ---------------------------------------------------------------------------

def get_pusht_observation(
    info: Dict[str, Any],
    rgb_frame: np.ndarray,
) -> Dict[str, Any]:
    """
    Convert env info dict to the observation dict expected by the policy.

    Parameters
    ----------
    info : Dict[str, Any]
        Info dict returned by env._get_info() or env.step().
        Must contain ``pos_agent`` (2,), ``blue_block_pose`` (3,),
        ``red_block_pose`` (3,) — exactly the fields recorded in the zarr
        ``data/state`` array during demonstration collection.
    rgb_frame : np.ndarray
        Current environment render (96×96×3 uint8 RGB).
    """
    # Reconstruct the 8-D state that was saved during data collection:
    #   [agent_x, agent_y, blue_x, blue_y, blue_theta, red_x, red_y, red_theta]
    proprio = np.concatenate([
        np.array(info["pos_agent"],       dtype=np.float32),   # (2,)
        np.array(info["blue_block_pose"], dtype=np.float32),   # (3,)
        np.array(info["red_block_pose"],  dtype=np.float32),   # (3,)
    ])  # (8,)

    return {
        "full_image": rgb_frame,
        "state": proprio,
    }


# ---------------------------------------------------------------------------
# Single episode rollout
# ---------------------------------------------------------------------------

def _save_video(frames: List[np.ndarray], path: str, fps: int) -> None:
    """Write a list of uint8 RGB frames to an MP4 file using cv2."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def run_episode(
    cfg: EvalConfig,
    env,
    vla,
    processor,
    action_head,
    proprio_projector,
    motion_token,
    motion_encoder,
    flow_buf: FlowBuffer,
    seed: int,
    log_file=None,
    video_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one episode and return result metrics."""
    # Old gym API: seed separately, then reset without kwargs.
    # New gymnasium API: reset(seed=seed). Try both.
    if hasattr(env, 'seed'):
        env.seed(seed)
        obs = env.reset()
    else:
        obs = env.reset(seed=seed)
    if isinstance(obs, tuple):
        obs, _ = obs

    # Get initial info dict so we have state at step 0 before any env.step().
    info: Dict[str, Any] = env._get_info()

    flow_buf._frames.clear()
    action_queue: deque = deque()

    success = False
    total_reward = 0.0
    cov_blue_max = 0.0
    cov_red_max  = 0.0
    video_frames: List[np.ndarray] = []

    for step_t in range(cfg.max_steps):
        # Get current frame from env render
        rgb_frame = env.render(mode="rgb_array") if hasattr(env, "render") else env._render()
        if rgb_frame is None:
            rgb_frame = np.zeros((96, 96, 3), dtype=np.uint8)

        if video_path is not None:
            video_frames.append(rgb_frame.astype(np.uint8))

        # Update optical-flow buffer
        flow_buf.push(rgb_frame.astype(np.uint8))

        # Build observation from info dict (gives exact same 8-D state as training).
        observation = get_pusht_observation(info, rgb_frame)

        if len(action_queue) == 0:
            # Build normalised his_motion_seq
            his_motion_seq = flow_buf.get_flow_tensor()   # (1, H, 2, 16, 16) bfloat16

            # Normalise proprio for VLA input
            if cfg.use_proprio:
                proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
                observation["state"] = normalize_proprio(
                    observation["state"], proprio_norm_stats
                )

            # Prepare image
            img_pil = Image.fromarray(observation["full_image"]).convert("RGB")
            prompt  = (
                f"In: What action should the robot take to "
                f"{PUSHT_LANGUAGE_INSTRUCTION.lower()}?\nOut:"
            )
            inputs = processor(prompt, img_pil).to(DEVICE, dtype=torch.bfloat16)

            # Build proprio tensor
            proprio_tensor = (
                torch.tensor(observation["state"], dtype=torch.bfloat16)
                .unsqueeze(0)
                .to(DEVICE)
                if cfg.use_proprio else None
            )

            with torch.inference_mode():
                actions, _ = vla.predict_action(
                    **inputs,
                    unnorm_key=cfg.unnorm_key,
                    do_sample=False,
                    proprio=proprio_tensor,
                    proprio_projector=proprio_projector,
                    action_head=action_head,
                    motion_token=motion_token,
                    motion_encoder=motion_encoder,
                    his_motion_seq=his_motion_seq,
                    use_film=cfg.use_film,
                )

            # actions: (NUM_ACTIONS_CHUNK, ACTION_DIM=2) numpy
            for i in range(min(cfg.num_open_loop_steps, len(actions))):
                action_queue.append(actions[i])

        action = action_queue.popleft()

        # Step environment
        try:
            step_result = env.step(action)
        except Exception as e:
            print(f"  env.step error: {e}")
            break

        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, done, truncated, info = step_result

        total_reward += float(reward)

        # Track coverage metrics if available
        if isinstance(info, dict):
            cov_blue_max = max(cov_blue_max, float(info.get("cov_blue_in_target", 0.0)))
            cov_red_max  = max(cov_red_max,  float(info.get("cov_red_in_target",  0.0)))
            if info.get("success", False) or done:
                success = True
                break

        if done:
            break

    if video_path is not None and video_frames:
        _save_video(video_frames, video_path, cfg.video_fps)

    return {
        "success":      success,
        "total_reward": total_reward,
        "steps":        step_t + 1,
        "cov_blue_max": cov_blue_max,
        "cov_red_max":  cov_red_max,
        "video_path":   video_path,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

@draccus.wrap()
def evaluate(cfg: EvalConfig) -> None:
    assert cfg.pretrained_checkpoint, "Provide --pretrained_checkpoint"

    # Resolve to absolute path so AutoModelForVision2Seq doesn't mistake a
    # relative path for a HuggingFace Hub repo ID.  We resolve relative to
    # _PROJECT_ROOT (not cwd) because the launch script cd's into HiF-VLA/.
    if not model_is_on_hf_hub(str(cfg.pretrained_checkpoint)):
        p = Path(cfg.pretrained_checkpoint)
        if not p.is_absolute():
            p = (_PROJECT_ROOT / p).resolve()
        cfg.pretrained_checkpoint = str(p)

    # Locate flow statistics (needed to normalise optical flow online)
    flow_stats_path = cfg.flow_stats_path
    if flow_stats_path is None and cfg.zarr_path:
        flow_stats_path = str(Path(cfg.zarr_path).parent / "flow_stats.npz")
    if flow_stats_path is None or not os.path.exists(flow_stats_path):
        raise FileNotFoundError(
            f"Flow statistics file not found at '{flow_stats_path}'. "
            "Run training first to generate it (or pass --flow_stats_path explicitly)."
        )
    flow_mean, flow_std = load_flow_stats(flow_stats_path)
    print(f"Loaded flow stats: mean={flow_mean}, std={flow_std}")

    # Set random seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Load model
    vla, processor, action_head, proprio_projector, motion_token, motion_encoder = \
        load_model_components(cfg)

    # Create environment
    try:
        from memory_diffusion_policy.env.pusht.pusht_keypoints_three_goals_swap_env import (
            PushTKeypointsThreeGoalsSwapEnv,
        )
        env = PushTKeypointsThreeGoalsSwapEnv(render_size=96)
        print("Using PushTKeypointsThreeGoalsSwapEnv")
    except ImportError as e:
        raise ImportError(
            f"Could not import Swap-T environment: {e}\n"
            f"Make sure memory_diffusion_policy is importable (added to sys.path above)."
        )

    # Logging
    os.makedirs(cfg.log_dir, exist_ok=True)
    run_id = f"EVAL-pusht-{time.strftime('%Y%m%d-%H%M%S')}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    log_path = os.path.join(cfg.log_dir, run_id + ".json")

    # Video output directory (sibling of the JSON log)
    video_dir = None
    if cfg.save_video and cfg.n_video_episodes > 0:
        video_dir = os.path.join(cfg.log_dir, run_id + "_videos")
        os.makedirs(video_dir, exist_ok=True)
        print(f"Recording first {cfg.n_video_episodes} episodes → {video_dir}/")

    flow_buf = FlowBuffer(cfg.history_length, flow_mean, flow_std)
    results: List[Dict] = []

    print(f"\nRunning {cfg.num_episodes} episodes …\n")
    for ep_i in range(cfg.num_episodes):
        ep_seed = cfg.seed + ep_i

        video_path = None
        if video_dir is not None and ep_i < cfg.n_video_episodes:
            status_tag = "ep{:03d}_seed{}".format(ep_i, ep_seed)
            video_path = os.path.join(video_dir, f"{status_tag}.mp4")

        result = run_episode(
            cfg=cfg,
            env=env,
            vla=vla,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            motion_token=motion_token,
            motion_encoder=motion_encoder,
            flow_buf=flow_buf,
            seed=ep_seed,
            video_path=video_path,
        )
        results.append(result)
        status = "SUCCESS" if result["success"] else "FAIL   "

        # Rename video to include outcome tag
        if result.get("video_path") and os.path.exists(result["video_path"]):
            old_path = result["video_path"]
            outcome = "SUCCESS" if result["success"] else "FAIL"
            new_path = old_path.replace(".mp4", f"_{outcome}.mp4")
            os.rename(old_path, new_path)
            result["video_path"] = new_path

        video_note = f" | video saved" if result.get("video_path") else ""
        print(
            f"  ep {ep_i:3d}/{cfg.num_episodes} | {status} | "
            f"steps={result['steps']:4d} | "
            f"cov_blue={result['cov_blue_max']:.3f} | "
            f"cov_red={result['cov_red_max']:.3f}"
            f"{video_note}"
        )

    # Aggregate
    n_success  = sum(r["success"]      for r in results)
    avg_reward = np.mean([r["total_reward"] for r in results])
    avg_cov_b  = np.mean([r["cov_blue_max"] for r in results])
    avg_cov_r  = np.mean([r["cov_red_max"]  for r in results])

    summary = {
        "checkpoint":   str(cfg.pretrained_checkpoint),
        "num_episodes": cfg.num_episodes,
        "success_rate": n_success / cfg.num_episodes,
        "avg_reward":   float(avg_reward),
        "avg_cov_blue": float(avg_cov_b),
        "avg_cov_red":  float(avg_cov_r),
        "episodes":     results,
    }

    print(
        f"\n{'='*60}\n"
        f"Success rate : {n_success}/{cfg.num_episodes} "
        f"({100 * n_success / cfg.num_episodes:.1f}%)\n"
        f"Avg reward   : {avg_reward:.3f}\n"
        f"Avg cov blue : {avg_cov_b:.3f}\n"
        f"Avg cov red  : {avg_cov_r:.3f}\n"
        f"{'='*60}"
    )

    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved → {log_path}")


if __name__ == "__main__":
    evaluate()
