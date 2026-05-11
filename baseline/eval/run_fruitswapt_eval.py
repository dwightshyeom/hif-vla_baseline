"""
run_fruitswapt_eval.py

Evaluates a fine-tuned HiF-VLA checkpoint on the FruitSwapVision robosuite
environment.

Usage:
    python baseline/eval/run_fruitswapt_eval.py \\
        --pretrained_checkpoint runs/fruitswapt_hifvla/<run_id>--<step>_chkpt \\
        --flow_stats_path runs/fruitswapt_hifvla/<run_id>/flow_stats.npz \\
        --unnorm_key fruit_swap_vision \\
        --num_episodes 50 \\
        --max_steps 600

The script:
  1. Loads the VLA, action head, proprio projector, motion manager, and motion
     encoder from the checkpoint directory.
  2. Instantiates FruitSwapVision (robosuite) at 96×96 with agentview camera.
  3. For each episode, maintains a rolling window of optical-flow frames and
     feeds them as his_motion_seq to vla.predict_action.
  4. Executes action chunks open-loop (default: full 8-step chunk).
  5. Reports per-episode success, task phase reached, and overall success rate.
"""

import json
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Path setup — "fruitswapt" in argv triggers LIBERO constants (ACTION_DIM=7)
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = Path(__file__).resolve().parent       # baseline/eval/
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent             # project root
_HIF_VLA_ROOT = _PROJECT_ROOT / "HiF-VLA"
_ROBOSUITE_ROOT = _PROJECT_ROOT / "robosuite_pomdp"

for _p in [str(_PROJECT_ROOT), str(_HIF_VLA_ROOT), str(_ROBOSUITE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import draccus
import cv2
import numpy as np
import torch
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

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Language instruction must match what was used during training.
FRUITSWAPT_LANGUAGE_INSTRUCTION: str = (
    "use the empty patch as a buffer to swap the magenta and cyan cubes "
    "to each other's starting positions"
)
FRUITSWAPT_DATASET_NAME: str = "fruit_swap_vision"


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_fruitswapt_env(
    image_size: int = 96,
    camera_name: str = "agentview",
    horizon: int = 1000,
    seed: Optional[int] = None,
):
    """
    Instantiate a FruitSwapVision robosuite environment.

    Returns the environment object (raw robosuite, not gym-wrapped).
    Camera observations are enabled at ``image_size × image_size``.
    """
    # robosuite_pomdp re-exports robosuite.make; the custom environments are
    # registered when the package is imported.
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config

    controller_config = load_composite_controller_config(
        controller="BASIC", robot="Panda"
    )

    env = suite.make(
        env_name="FruitSwapVision",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=[camera_name],
        camera_heights=image_size,
        camera_widths=image_size,
        camera_depths=False,
        control_freq=20,
        horizon=horizon,
        ignore_done=False,
        hard_reset=True,
    )

    if seed is not None:
        np.random.seed(seed)

    return env


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def get_fruitswapt_observation(
    obs: Dict[str, Any],
    camera_key: str = "agentview_image",
) -> Dict[str, Any]:
    """
    Convert a robosuite obs dict to the format expected by the policy.

    Returns
    -------
    dict with:
        "full_image" : (H, W, 3) uint8 RGB
        "state"      : (8,) float32 – [eef_pos(3), eef_quat(4), gripper(1)]
    """
    rgb = np.asarray(obs[camera_key], dtype=np.uint8)

    eef_pos   = np.asarray(obs["robot0_eef_pos"],      dtype=np.float32)   # (3,)
    eef_quat  = np.asarray(obs["robot0_eef_quat"],     dtype=np.float32)   # (4,)
    grip_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)   # (2,)

    state = np.concatenate([eef_pos, eef_quat, grip_qpos[:1]])  # (8,)

    return {"full_image": rgb, "state": state}


# ---------------------------------------------------------------------------
# Optical-flow buffer (same as run_pusht_eval.py)
# ---------------------------------------------------------------------------

class FlowBuffer:
    """Rolling deque of grayscale frames for online optical-flow computation."""

    def __init__(
        self,
        history_length: int,
        flow_mean: np.ndarray,
        flow_std: np.ndarray,
    ) -> None:
        self.history_length = history_length
        self.flow_mean = flow_mean
        self.flow_std  = flow_std
        self._frames: deque = deque(maxlen=history_length + 1)

    def push(self, rgb_frame: np.ndarray) -> None:
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        self._frames.append(gray)

    def get_flow_tensor(self) -> torch.Tensor:
        """Return (1, history_length, 2, FLOW_H, FLOW_W) bfloat16 tensor."""
        frames = list(self._frames)
        result = np.zeros((self.history_length, 2, FLOW_H, FLOW_W), dtype=np.float32)
        n_flows  = min(len(frames) - 1, self.history_length)
        start_buf = len(frames) - 1 - n_flows
        start_out = self.history_length - n_flows
        for i in range(n_flows):
            result[start_out + i] = compute_frame_flow(
                frames[start_buf + i], frames[start_buf + i + 1]
            )
        result = normalize_flow(result, self.flow_mean, self.flow_std)
        return torch.from_numpy(result[None]).to(DEVICE, dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# LoRA / checkpoint loading (mirrors run_pusht_eval.py)
# ---------------------------------------------------------------------------

def _resolve_lora_base_model_path(saved: str) -> str:
    saved = (saved or "").strip()
    if not saved:
        return "openvla/openvla-7b"
    expanded = os.path.expanduser(saved)
    if os.path.isdir(expanded):
        return str(Path(expanded).resolve())
    if model_is_on_hf_hub(saved):
        return saved

    norm = saved.replace("\\", "/")
    rev_m = re.search(r"/snapshots/([a-f0-9]{40})(?:/|$)", norm)

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub_root = hf_home / "hub"
    for model_dir_name in (
        "models--openvla--openvla-7b",
        "models--OpenVLA--openvla-7b",
    ):
        snap_root = hub_root / model_dir_name / "snapshots"
        if not snap_root.is_dir():
            continue
        children = [c for c in snap_root.iterdir() if c.is_dir()]
        if not children:
            continue
        if rev_m is not None:
            want = rev_m.group(1)
            for c in children:
                if c.name == want:
                    return str(c.resolve())
        children.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(children[0].resolve())

    return "openvla/openvla-7b"


def load_model_components(cfg):
    """Load all HiF-VLA checkpoint components."""
    import dataclasses
    from peft import PeftModel

    ckpt_dir = Path(cfg.pretrained_checkpoint)
    lora_dir = ckpt_dir / "lora_adapter"
    is_lora  = lora_dir.exists() and (lora_dir / "adapter_config.json").exists()

    if is_lora:
        with open(lora_dir / "adapter_config.json") as f:
            adapter_cfg_json = json.load(f)
        base_model_path = _resolve_lora_base_model_path(
            adapter_cfg_json["base_model_name_or_path"]
        )
        print(f"LoRA checkpoint detected.  Base model : {base_model_path}")
        print(f"                           LoRA adapter: {lora_dir}")

        base_cfg = dataclasses.replace(cfg, pretrained_checkpoint=base_model_path)

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        update_auto_map(base_model_path)
        check_model_logic_mismatch(base_model_path)

        vla = get_vla(base_cfg)
        vla = PeftModel.from_pretrained(vla, str(lora_dir))
        vla.eval()

        from experiments.robot.openvla_utils import _load_dataset_stats
        _load_dataset_stats(vla.get_base_model(), str(ckpt_dir))

        processor = AutoProcessor.from_pretrained(str(ckpt_dir), trust_remote_code=True)
        llm_dim = vla.get_base_model().llm_dim
    else:
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

    _pdim = int(proprio_projector.proprio_dim) if proprio_projector is not None else PROPRIO_DIM
    print(f"Loaded: VLA llm_dim={llm_dim}, ACTION_DIM={ACTION_DIM}, proprio_dim={_pdim}")
    return vla, processor, action_head, proprio_projector, motion_token, motion_encoder


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    # fmt: off
    pretrained_checkpoint: Union[str, Path] = ""   # Path to fine-tuned checkpoint dir
    flow_stats_path: Optional[str] = None          # Path to flow_stats_*.npz

    model_family: str = "openvla"
    use_l1_regression: bool = True
    use_film: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = True
    center_crop: bool = True
    lora_rank: int = 32
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    unnorm_key: str = FRUITSWAPT_DATASET_NAME
    history_length: int = 8

    # Environment
    image_size: int = 96            # camera resolution (H and W)
    camera_key: str = "agentview_image"   # which obs key holds the RGB image
    horizon: int = 1000             # env max steps per episode (robosuite horizon)

    num_episodes: int = 50
    max_steps: int = 600
    num_open_loop_steps: int = 8    # action chunk steps executed per inference call
    seed: int = 42

    # Logging
    log_dir: str = "experiments/logs/fruitswapt"
    run_id_note: Optional[str] = None

    # Video recording
    save_video: bool = True
    n_video_episodes: int = 10
    video_fps: int = 10
    # fmt: on


# ---------------------------------------------------------------------------
# Video saving
# ---------------------------------------------------------------------------

def _save_video(frames: List[np.ndarray], path: str, fps: int) -> None:
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


# ---------------------------------------------------------------------------
# Single episode rollout
# ---------------------------------------------------------------------------

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
    video_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one episode and return result metrics."""
    np.random.seed(seed)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs, _ = obs

    flow_buf._frames.clear()
    action_queue: deque = deque()

    success      = False
    total_reward = 0.0
    max_phase    = 0
    video_frames: List[np.ndarray] = []

    for step_t in range(cfg.max_steps):
        observation = get_fruitswapt_observation(obs, cfg.camera_key)
        rgb_frame   = observation["full_image"]   # (H, W, 3) uint8

        if video_path is not None:
            video_frames.append(rgb_frame)

        flow_buf.push(rgb_frame)

        if len(action_queue) == 0:
            his_motion_seq = flow_buf.get_flow_tensor()   # (1, H, 2, 16, 16) bfloat16

            if cfg.use_proprio:
                proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
                raw_state = np.asarray(observation["state"], dtype=np.float32)
                if proprio_projector is not None:
                    pd  = int(proprio_projector.proprio_dim)
                    q01 = np.asarray(proprio_norm_stats["q01"], dtype=np.float32)
                    q99 = np.asarray(proprio_norm_stats["q99"], dtype=np.float32)
                    raw_state = raw_state[..., :pd]
                    stats_slice = {
                        **proprio_norm_stats,
                        "q01": q01[:pd].tolist(),
                        "q99": q99[:pd].tolist(),
                    }
                    norm_state = normalize_proprio(raw_state, stats_slice)
                else:
                    norm_state = normalize_proprio(raw_state, proprio_norm_stats)
            else:
                norm_state = None

            img_pil = Image.fromarray(rgb_frame).convert("RGB")
            prompt  = (
                f"In: What action should the robot take to "
                f"{FRUITSWAPT_LANGUAGE_INSTRUCTION}?\nOut:"
            )
            inputs = processor(prompt, img_pil).to(DEVICE, dtype=torch.bfloat16)

            proprio_tensor = (
                torch.tensor(norm_state, dtype=torch.bfloat16)
                .unsqueeze(0)
                .to(DEVICE)
                if cfg.use_proprio and norm_state is not None else None
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

            # actions: (NUM_ACTIONS_CHUNK, ACTION_DIM=7) numpy float32
            for i in range(min(cfg.num_open_loop_steps, len(actions))):
                action_queue.append(actions[i])

        action = action_queue.popleft()

        try:
            step_result = env.step(action)
        except Exception as e:
            print(f"  env.step error at step {step_t}: {e}")
            break

        # robosuite always returns (obs, reward, done, info)
        obs, reward, done, info = step_result

        total_reward += float(reward)

        if isinstance(info, dict):
            max_phase = max(max_phase, int(info.get("task_phase", 0)))
            if info.get("task_phase", 0) == 3 or info.get("failed", False):
                success = info.get("task_phase", 0) == 3
                if video_path is not None:
                    video_frames.append(
                        get_fruitswapt_observation(obs, cfg.camera_key)["full_image"]
                    )
                break

        if done:
            success = env._check_success() if hasattr(env, "_check_success") else False
            break

    if video_path is not None and video_frames:
        _save_video(video_frames, video_path, cfg.video_fps)

    return {
        "success":      success,
        "total_reward": total_reward,
        "steps":        step_t + 1,
        "max_phase":    max_phase,
        "video_path":   video_path,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

@draccus.wrap()
def evaluate(cfg: EvalConfig) -> None:
    assert cfg.pretrained_checkpoint, "Provide --pretrained_checkpoint"

    if not model_is_on_hf_hub(str(cfg.pretrained_checkpoint)):
        p = Path(cfg.pretrained_checkpoint)
        if not p.is_absolute():
            p = (_PROJECT_ROOT / p).resolve()
        cfg.pretrained_checkpoint = str(p)

    # Locate flow statistics
    if cfg.flow_stats_path is None:
        raise ValueError(
            "Provide --flow_stats_path (generated by training; look in the run dir "
            "for a file named flow_stats*.npz)."
        )
    flow_stats_path = str(Path(cfg.flow_stats_path).resolve())
    if not os.path.exists(flow_stats_path):
        raise FileNotFoundError(
            f"Flow statistics file not found: '{flow_stats_path}'. "
            "Run training first to generate it."
        )
    flow_mean, flow_std = load_flow_stats(flow_stats_path)
    print(f"Loaded flow stats: mean={flow_mean}, std={flow_std}")

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Load model
    vla, processor, action_head, proprio_projector, motion_token, motion_encoder = \
        load_model_components(cfg)

    # Create environment
    print(f"Creating FruitSwapVision env (image_size={cfg.image_size}) …")
    env = make_fruitswapt_env(
        image_size=cfg.image_size,
        camera_name=cfg.camera_key.replace("_image", ""),  # "agentview_image" → "agentview"
        horizon=cfg.horizon,
    )
    print("Environment ready.")

    # Logging
    os.makedirs(cfg.log_dir, exist_ok=True)
    run_id   = f"EVAL-fruitswapt-{time.strftime('%Y%m%d-%H%M%S')}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    log_path = os.path.join(cfg.log_dir, run_id + ".json")

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
            video_path = os.path.join(
                video_dir, f"ep{ep_i:03d}_seed{ep_seed}.mp4"
            )

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

        # Rename video with outcome tag
        if result.get("video_path") and os.path.exists(result["video_path"]):
            old_p  = result["video_path"]
            outcome = "SUCCESS" if result["success"] else "FAIL"
            new_p  = old_p.replace(".mp4", f"_{outcome}.mp4")
            os.rename(old_p, new_p)
            result["video_path"] = new_p

        status = "SUCCESS" if result["success"] else "FAIL   "
        print(
            f"  ep {ep_i:3d}/{cfg.num_episodes} | {status} | "
            f"steps={result['steps']:4d} | "
            f"max_phase={result['max_phase']} | "
            f"reward={result['total_reward']:.3f}"
        )

    # Summary
    n_eps = len(results)
    n_success = sum(r["success"] for r in results)
    success_rate = n_success / n_eps
    avg_phase    = sum(r["max_phase"] for r in results) / n_eps
    avg_steps    = sum(r["steps"]     for r in results) / n_eps

    print(
        f"\n{'='*60}\n"
        f"  Eval summary over {n_eps} episodes\n"
        f"  Success rate : {success_rate:.3f}  ({n_success}/{n_eps})\n"
        f"  Avg max phase: {avg_phase:.2f}\n"
        f"  Avg steps    : {avg_steps:.1f}\n"
        f"{'='*60}"
    )

    log_data = {
        "run_id":           run_id,
        "checkpoint":       str(cfg.pretrained_checkpoint),
        "unnorm_key":       cfg.unnorm_key,
        "num_episodes":     n_eps,
        "success_rate":     success_rate,
        "avg_max_phase":    avg_phase,
        "avg_steps":        avg_steps,
        "episodes":         results,
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    print(f"Results saved → {log_path}")

    env.close()


if __name__ == "__main__":
    evaluate()
