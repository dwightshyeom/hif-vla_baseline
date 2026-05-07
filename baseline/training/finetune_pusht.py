"""
finetune_pusht.py

Fine-tunes HiF-VLA on the Swap-T (Push-T) task.

This script is a direct adaptation of HiF-VLA/vla-scripts/finetune.py with
the following differences:
  - Reads from a zarr store via PushTHiFVLADataset instead of RLDS.
  - ACTION_DIM=2, PROPRIO_DIM=8 (auto-detected because "pusht" is in argv[0]).
  - Optical flow replaces MPEG-4 motion vectors.
  - A fixed dummy language instruction is used.

Run with:
    accelerate launch --num_processes <N_GPU> \\
        baseline/training/finetune_pusht.py \\
        --vla_path /path/to/openvla-weights \\
        --zarr_path memory_diffusion_policy/swap_3t_dataset_320 \\
        --run_root_dir /path/to/runs \\
        --use_l1_regression True \\
        --use_proprio True \\
        --batch_size 4 \\
        --max_steps 50005

Or for single-GPU debugging:
    python baseline/training/finetune_pusht.py --vla_path ... --zarr_path ...
"""

import itertools
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

# ---------------------------------------------------------------------------
# Path setup — must happen before any HiF-VLA imports so that the "pusht"
# keyword appears in sys.argv and prismatic/vla/constants.py selects
# PUSHT_CONSTANTS (ACTION_DIM=2, PROPRIO_DIM=8).
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # baseline/training/
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent              # project_root/
_HIF_VLA_ROOT = _PROJECT_ROOT / "HiF-VLA"
for _p in [str(_PROJECT_ROOT), str(_HIF_VLA_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import draccus
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
import wandb
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)
from motion_layers.motion_tokenizer import HisMotionEncoder
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import JointExpert, MotionTokenManager
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_motion_mse_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from baseline.dataset.pusht_hifvla_dataset import MultiTaskPushTHiFVLADataset, PushTHiFVLADataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"

    # Dataset
    zarr_path: str = ""   # Path to swap_3t_dataset_320 zarr store (ignored if task_manifest is set)
    task_manifest: Optional[str] = None  # JSON list of tasks: zarr_path, dataset_name, language
    run_root_dir: Path = Path("runs/pusht_hifvla")
    flow_cache_path: Optional[str] = None  # .npy cache for pre-computed flow
    flow_stats_path: Optional[str] = None  # .npz cache for flow mean/std

    # Architecture
    use_l1_regression: bool = True
    use_film: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = True

    # Training
    batch_size: int = 1
    learning_rate: float = 5e-4
    lr_warmup_steps: int = 1000
    num_steps_before_decay: int = 50_000
    grad_accumulation_steps: int = 1
    max_steps: int = 50_005
    save_freq: int = 1_000
    save_latest_checkpoint_only: bool = False
    resume: bool = False
    resume_step: Optional[int] = None
    image_aug: bool = True
    val_fraction: float = 0.1
    use_val_set: bool = False
    val_freq: int = 5_000
    val_time_limit: int = 120

    # LoRA
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    merge_lora_during_training: bool = False

    # Gradient clipping (0 = disabled)
    max_grad_norm: float = 1.0

    # HiF-VLA motion encoder
    history_length: int = 8

    # HuggingFace Hub dataset (cloud training)
    # If use_hf_dataset=True, the zarr is downloaded from HF Hub.
    # zarr_path is ignored when use_hf_dataset=True.
    use_hf_dataset: bool = False
    hf_dataset_repo: str = "MemoryManip/Memory-T-Bench"
    hf_dataset_filename: str = "swapt-shuffle/data.zarr.zip"

    # Rollout during training (0 = disabled)
    rollout_freq: int = 0          # Run rollout every N gradient steps (0 = off)
    n_rollout_episodes: int = 3    # Episodes per rollout evaluation
    rollout_max_steps: int = 500   # Max steps per rollout episode
    rollout_fps: int = 10          # Video FPS for W&B logging
    rollout_unnorm_key: str = "swap_3t"  # Dataset stats key for unnormalising actions
    rollout_num_open_loop: int = 8  # Action chunk steps executed per inference

    # Logging
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "hifvla-pusht"
    run_id_note: Optional[str] = None
    run_id_override: Optional[str] = None
    wandb_log_freq: int = 10
    # fmt: on


def _resolve_path_for_training(
    raw: str,
    project_root: Path,
    *,
    must_be_file: bool = False,
) -> Path:
    """Resolve paths from the task manifest or CLI.

    Manifest zarr paths are almost always relative to the **repo root**
    (e.g. ``./swapt-shuffle/data.zarr.zip``), while the training script often
    runs with ``cwd`` = ``HiF-VLA``. We therefore try **project_root** before
    ``cwd`` for normal relative paths.

    Paths that start with ``..`` are resolved against **cwd** first (e.g.
    ``../configs`` from ``HiF-VLA``).
    """
    p = Path(raw)
    if p.is_absolute():
        out = p.resolve()
        _ok = out.is_file() if must_be_file else out.exists()
        if not _ok:
            raise FileNotFoundError(f"Path does not exist: {out}")
        return out

    parts = p.parts
    prefer_cwd_first = bool(parts and parts[0] == "..")

    cand_root = (project_root / p).resolve()
    cand_cwd = (Path.cwd() / p).resolve()

    if prefer_cwd_first:
        order = [cand_cwd, cand_root]
    else:
        order = [cand_root, cand_cwd]

    seen: set = set()
    candidates: List[Path] = []
    for c in order:
        k = str(c)
        if k not in seen:
            seen.add(k)
            candidates.append(c)

    for c in candidates:
        ok = c.is_file() if must_be_file else c.exists()
        if ok:
            return c

    raise FileNotFoundError(
        f"Could not find {raw!r}. Looked under repo root and cwd:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + f"\nPlace zarr files next to the manifest (under {project_root}), "
        "or use absolute paths / hf:// URLs."
    )


def load_task_manifest(manifest_path: Path, project_root: Path) -> List[Dict[str, Any]]:
    """Load a JSON manifest of multi-task training specs.

    Each entry must include:
      - ``zarr_path``: path to zarr dir or ``.zip`` (relative to ``project_root``
        if not absolute); ``hf://org/repo/path`` is allowed.
      - ``dataset_name``: key for ``dataset_statistics.json`` / ``norm_stats``.
      - ``language`` or ``language_instruction``: task phrase for the prompt.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("task manifest must be a JSON array of task objects")
    tasks: List[Dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"task_manifest[{i}] must be a JSON object")
        zp = item.get("zarr_path")
        if not zp:
            raise ValueError(f"task_manifest[{i}] missing 'zarr_path'")
        dname = item.get("dataset_name")
        if not dname:
            raise ValueError(f"task_manifest[{i}] missing 'dataset_name'")
        lang = item.get("language_instruction") or item.get("language")
        if not lang:
            raise ValueError(
                f"task_manifest[{i}] needs 'language' or 'language_instruction'"
            )
        if isinstance(zp, str) and zp.startswith("hf://"):
            zpath_resolved = zp
        else:
            zpath_resolved = str(_resolve_path_for_training(zp, project_root, must_be_file=False))
        tasks.append({
            "zarr_path": zpath_resolved,
            "dataset_name": dname,
            "language_instruction": lang,
        })
    return tasks


def remove_ddp_prefix(state_dict: dict) -> dict:
    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def get_run_id(cfg: FinetuneConfig) -> str:
    if cfg.run_id_override:
        return cfg.run_id_override
    if cfg.resume:
        run_id = cfg.vla_path.split("/")[-1]
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
        return run_id
    run_id = (
        f"{cfg.vla_path.split('/')[-1]}+{'multitask' if cfg.task_manifest and str(cfg.task_manifest).strip() else 'swap_3t'}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
        f"+his-{cfg.history_length}"
    )
    if cfg.use_lora:
        run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.image_aug:
        run_id += "--image_aug"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    return run_id


def load_checkpoint(module_name: str, path: str, step: int) -> dict:
    ckpt_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    return remove_ddp_prefix(
        torch.load(ckpt_path, weights_only=True, map_location="cpu")
    )


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    return DDP(
        module,
        device_ids=[device_id],
        find_unused_parameters=find_unused,
        gradient_as_bucket_view=True,
    )


def count_parameters(module: nn.Module, name: str) -> None:
    n = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {n:,}")


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    module = module_class(**module_args)
    count_parameters(module, module_name)
    if cfg.resume:
        sd = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(sd)
    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)
    return wrap_ddp(module, device_id, find_unused_params)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def run_forward_pass(
    vla,
    motion_encoder,
    action_head,
    proprio_projector,
    motion_token,
    batch: Dict,
    action_tokenizer: ActionTokenizer,
    device_id: int,
    use_l1_regression: bool,
    use_proprio: bool,
    use_film: bool,
    num_patches: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    metrics: Dict[str, float] = {}

    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)
    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask   = get_next_actions_mask(ground_truth_token_ids)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"].to(device_id),
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            motion_token=motion_token,
            use_film=use_film,
        )

    last_hidden = output.hidden_states[-1]                           # (B, seq, D)
    motion_hidden_states = (
        last_hidden[:, num_patches + 1 : num_patches + 1 + NUM_ACTIONS_CHUNK]
        .reshape(batch["input_ids"].shape[0], NUM_ACTIONS_CHUNK, -1)
    )
    text_hidden_states = last_hidden[:, num_patches + NUM_ACTIONS_CHUNK : -1]

    batch_size = batch["input_ids"].shape[0]
    actions_hidden_states = (
        text_hidden_states[current_action_mask | next_actions_mask]
        .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )

    with torch.autocast("cuda", dtype=torch.bfloat16):
        his_motion_feature = motion_encoder(
            batch["mv_history"].to(device_id)
        )  # (B, out_dim)

    pre_action, res_motion = action_head(
        actions_hidden_states,
        motion_hidden_states.to(torch.bfloat16),
        time_cond=his_motion_feature,
    )

    gt_motion        = batch["mv_future"].to(device_id).to(torch.bfloat16)
    motion_res_loss  = compute_motion_mse_loss(res_motion, gt_motion)
    pred_action_loss = torch.nn.L1Loss()(ground_truth_actions, pre_action)

    if use_l1_regression:
        loss = 0.01 * motion_res_loss + pred_action_loss
        metrics["loss_value"]        = loss.item()
        metrics["action_l1_loss"]    = pred_action_loss.item()
        metrics["motion_mse_loss"]   = motion_res_loss.item()
    else:
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)
        curr_acc = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        curr_l1  = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids,
            mask=current_action_mask,
        )
        metrics.update({
            "loss_value":          loss.item(),
            "curr_action_accuracy": curr_acc.item(),
            "curr_action_l1_loss":  curr_l1.item(),
        })

    return loss, metrics


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_training_checkpoint(
    cfg: FinetuneConfig,
    run_dir: Path,
    log_step: int,
    vla,
    processor,
    motion_encoder,
    proprio_projector,
    motion_manager,
    action_head,
    train_dataset: PushTHiFVLADataset,
    distributed_state: PartialState,
) -> None:
    if cfg.save_latest_checkpoint_only:
        ckpt_dir  = run_dir
        ckpt_suffix = "latest_checkpoint.pt"
    else:
        ckpt_dir    = Path(str(run_dir) + f"--{log_step}_chkpt")
        ckpt_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = ckpt_dir / "lora_adapter"

    if distributed_state.is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, ckpt_dir)
        print(f"Saving checkpoint at step {log_step} → {ckpt_dir}")

    dist.barrier()

    if distributed_state.is_main_process:
        processor.save_pretrained(ckpt_dir)
        vla.module.save_pretrained(adapter_dir)
        torch.save(
            motion_encoder.state_dict(),
            ckpt_dir / f"motion_encoder--{ckpt_suffix}",
        )
        torch.save(
            motion_manager.state_dict(),
            ckpt_dir / f"motion_manager--{ckpt_suffix}",
        )
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(
                proprio_projector.state_dict(),
                ckpt_dir / f"proprio_projector--{ckpt_suffix}",
            )
        if cfg.use_l1_regression and action_head is not None:
            torch.save(
                action_head.state_dict(),
                ckpt_dir / f"action_head--{ckpt_suffix}",
            )
        if cfg.use_film:
            torch.save(
                vla.module.vision_backbone.state_dict(),
                ckpt_dir / f"vision_backbone--{ckpt_suffix}",
            )

    dist.barrier()

    if cfg.use_lora and cfg.merge_lora_during_training:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True,
        )
        merged = PeftModel.from_pretrained(base_vla, adapter_dir).merge_and_unload()
        if distributed_state.is_main_process:
            merged.save_pretrained(ckpt_dir)
        dist.barrier()


# ---------------------------------------------------------------------------
# Rollout utilities (used during training to evaluate the model live)
# ---------------------------------------------------------------------------

def _setup_rollout_env():
    """
    Locate the diffusion_policy package (not pip-installed) and import the
    Swap-T environment.  Returns the env class or raises ImportError.
    """
    import importlib

    sentinel = Path("env") / "pusht" / "pymunk_override.py"

    def _has_real_pkg(parent: Path) -> bool:
        return (parent / "diffusion_policy" / sentinel).is_file()

    # Already importable with the real subpackage?
    try:
        import diffusion_policy
        dp_path = Path(diffusion_policy.__file__).parent
        if not (dp_path / sentinel).is_file():
            sys.path = [p for p in sys.path if Path(p) != dp_path.parent]
            raise ImportError("incomplete diffusion_policy")
    except ImportError:
        candidates = [
            Path.home() / "workspace" / "diffusion_policy",
            Path.home() / "Desktop" / "diffusion_policy",
            _PROJECT_ROOT / "memory_diffusion_policy" / "third_party",
        ]
        for candidate in candidates:
            if _has_real_pkg(candidate):
                _s = str(candidate)
                if _s not in sys.path:
                    sys.path.insert(0, _s)
                break
        else:
            raise ImportError(
                "diffusion_policy not found. Install it or add its parent dir to PYTHONPATH."
            )

    from memory_diffusion_policy.env.pusht.pusht_keypoints_three_goals_swap_env import (
        PushTKeypointsThreeGoalsSwapEnv,
    )
    return PushTKeypointsThreeGoalsSwapEnv


def do_training_rollout(
    cfg: "FinetuneConfig",
    vla,
    processor,
    action_head,
    proprio_projector,
    motion_manager,
    motion_encoder,
    device_id: int,
    gradient_step_idx: int,
    flow_stats_path: str,
    actual_proprio_dim: int = 8,
) -> None:
    """
    Run ``cfg.n_rollout_episodes`` evaluation episodes with the *current* model
    weights, record frames, and log results + videos to W&B.

    Switches vla to eval mode and restores train mode afterwards.
    Silently skips if environment dependencies are not available.
    """
    import cv2
    from collections import deque
    from PIL import Image
    from baseline.dataset.optical_flow import compute_frame_flow, load_flow_stats, normalize_flow, FLOW_H, FLOW_W
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK
    from experiments.robot.openvla_utils import normalize_proprio
    PUSHT_LANGUAGE_INSTRUCTION = "swap the colored blocks to their target positions"

    try:
        EnvClass = _setup_rollout_env()
    except ImportError as e:
        print(f"[rollout] Skipping (environment not available): {e}")
        return

    if not os.path.exists(flow_stats_path):
        print(f"[rollout] Skipping (flow stats not found at {flow_stats_path})")
        return

    flow_mean, flow_std = load_flow_stats(flow_stats_path)

    # Build a single motion_token for batch=1
    motion_token_1 = motion_manager.module.get_motion_token(1)

    # Switch to eval
    was_training = vla.training
    vla.eval()
    if action_head is not None:
        action_head.eval()
    if proprio_projector is not None:
        proprio_projector.eval()
    motion_encoder.eval()

    env = EnvClass(render_size=96)
    results = []
    video_clips: list = []   # list of (T, H, W, 3) uint8 numpy arrays

    for ep_i in range(cfg.n_rollout_episodes):
        seed = 1000 + gradient_step_idx * cfg.n_rollout_episodes + ep_i

        if hasattr(env, "seed"):
            env.seed(seed)
            obs = env.reset()
        else:
            obs = env.reset(seed=seed)
        if isinstance(obs, tuple):
            obs, _ = obs

        info: dict = env._get_info()

        # Rolling flow buffer
        flow_frames: deque = deque(maxlen=cfg.history_length + 1)
        action_queue: deque = deque()

        success     = False
        total_rew   = 0.0
        cov_blue    = 0.0
        cov_red     = 0.0
        ep_frames   = []

        for step_t in range(cfg.rollout_max_steps):
            try:
                rgb = env.render(mode="rgb_array") if hasattr(env, "render") else env._render()
            except Exception:
                rgb = None
            if rgb is None:
                rgb = np.zeros((96, 96, 3), dtype=np.uint8)

            ep_frames.append(rgb.astype(np.uint8))
            flow_frames.append(rgb.astype(np.uint8))

            if len(action_queue) == 0:
                # Build his_motion_seq: (1, history_length, 2, FLOW_H, FLOW_W)
                frames_list = list(flow_frames)
                flow_stack = []
                for i in range(1, len(frames_list)):
                    f = compute_frame_flow(frames_list[i - 1], frames_list[i])
                    f = normalize_flow(f, flow_mean, flow_std)
                    flow_stack.append(f)
                # Pad to history_length if we don't have enough yet
                while len(flow_stack) < cfg.history_length:
                    flow_stack.insert(0, np.zeros((2, FLOW_H, FLOW_W), dtype=np.float32))
                flow_stack = flow_stack[-cfg.history_length:]
                his_motion_seq = (
                    torch.tensor(np.stack(flow_stack), dtype=torch.bfloat16)
                    .unsqueeze(0)
                    .to(device_id)
                )  # (1, history_length, 2, FLOW_H, FLOW_W)

                # Build proprio — match the dimensionality that the dataset uses.
                # 2D: just agent xy (pos_agent only)
                # 8D: full state (agent xy + blue xytheta + red xytheta)
                if actual_proprio_dim == 2:
                    proprio = np.array(info["pos_agent"], dtype=np.float32)  # (2,)
                else:
                    proprio = np.concatenate([
                        np.array(info["pos_agent"],       dtype=np.float32),  # (2,)
                        np.array(info["blue_block_pose"], dtype=np.float32),  # (3,)
                        np.array(info["red_block_pose"],  dtype=np.float32),  # (3,)
                    ])  # (8,)
                if cfg.use_proprio:
                    proprio_norm = vla.norm_stats[cfg.rollout_unnorm_key]["proprio"]
                    proprio = normalize_proprio(proprio, proprio_norm)
                proprio_tensor = (
                    torch.tensor(proprio, dtype=torch.bfloat16).unsqueeze(0).to(device_id)
                    if cfg.use_proprio else None
                )

                img_pil = Image.fromarray(rgb).convert("RGB")
                prompt  = (
                    f"In: What action should the robot take to "
                    f"{PUSHT_LANGUAGE_INSTRUCTION}?\nOut:"
                )
                inputs = processor(prompt, img_pil).to(device_id, dtype=torch.bfloat16)

                with torch.inference_mode():
                    actions, _ = vla.predict_action(
                        **inputs,
                        unnorm_key=cfg.rollout_unnorm_key,
                        do_sample=False,
                        proprio=proprio_tensor,
                        proprio_projector=proprio_projector,
                        action_head=action_head,
                        motion_token=motion_token_1,
                        motion_encoder=motion_encoder,
                        his_motion_seq=his_motion_seq,
                        use_film=cfg.use_film,
                    )

                for i in range(min(cfg.rollout_num_open_loop, len(actions))):
                    action_queue.append(actions[i])

            action = action_queue.popleft()
            try:
                step_result = env.step(action)
            except Exception as e:
                print(f"  [rollout] env.step error: {e}")
                break

            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, done, _trunc, info = step_result

            total_rew += float(reward)
            if isinstance(info, dict):
                cov_blue = max(cov_blue, float(info.get("cov_blue_in_target", 0.0)))
                cov_red  = max(cov_red,  float(info.get("cov_red_in_target",  0.0)))
                if info.get("success", False) or done:
                    success = True
                    break
            if done:
                break

        results.append({
            "success": success, "steps": step_t + 1,
            "cov_blue": cov_blue, "cov_red": cov_red,
        })
        video_clips.append(np.stack(ep_frames))  # (T, H, W, 3)

    env.close()

    # Restore training mode
    if was_training:
        vla.train()
        if action_head is not None:
            action_head.train()
        if proprio_projector is not None:
            proprio_projector.train()
        motion_encoder.train()

    # Build W&B log dict
    n_eps = len(results)
    success_rate = sum(r["success"] for r in results) / n_eps
    log_dict: dict = {
        "Rollout/success_rate":  success_rate,
        "Rollout/avg_steps":     sum(r["steps"]     for r in results) / n_eps,
        "Rollout/avg_cov_blue":  sum(r["cov_blue"]  for r in results) / n_eps,
        "Rollout/avg_cov_red":   sum(r["cov_red"]   for r in results) / n_eps,
    }
    for i, (res, clip) in enumerate(zip(results, video_clips)):
        tag = "SUCCESS" if res["success"] else "FAIL"
        # W&B Video expects (T, C, H, W) uint8
        clip_chw = clip.transpose(0, 3, 1, 2)
        log_dict[f"Rollout/ep{i:02d}_{tag}"] = wandb.Video(
            clip_chw, fps=cfg.rollout_fps, format="mp4"
        )

    wandb.log(log_dict, step=gradient_step_idx)
    print(
        f"[rollout @ step {gradient_step_idx}] "
        f"success_rate={success_rate:.2f} "
        f"({sum(r['success'] for r in results)}/{n_eps})"
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def _init_single_gpu_distributed() -> None:
    os.environ.update({
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29500",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
    })
    dist.init_process_group(backend="nccl", rank=0, world_size=1)


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    assert cfg.use_lora,          "Only LoRA fine-tuning is supported."
    assert cfg.use_l1_regression, "Must use L1 regression (use_l1_regression=True)."

    use_manifest = bool(cfg.task_manifest and str(cfg.task_manifest).strip())
    assert use_manifest or cfg.use_hf_dataset or cfg.zarr_path, (
        "Provide --task_manifest, --zarr_path, OR set --use_hf_dataset True."
    )
    if use_manifest and cfg.use_hf_dataset:
        raise ValueError(
            "Do not combine --use_hf_dataset with --task_manifest. "
            "Put hf://... URLs in each task's zarr_path in the JSON instead."
        )

    cfg.vla_path = cfg.vla_path.rstrip("/")
    # Resolve to absolute path so AutoProcessor/AutoModel can find local checkpoints
    # regardless of which directory accelerate launches the script from.
    if not model_is_on_hf_hub(cfg.vla_path):
        cfg.vla_path = str(Path(cfg.vla_path).resolve())

    tasks_list: Optional[List[Dict[str, Any]]] = None

    if use_manifest:
        manifest_path = _resolve_path_for_training(
            cfg.task_manifest, _PROJECT_ROOT, must_be_file=True
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"task manifest not found: {cfg.task_manifest!r} "
                f"(resolved to {manifest_path}). "
                "Use an absolute path or a path relative to the repo root / cwd."
            )
        tasks_list = load_task_manifest(manifest_path, _PROJECT_ROOT)
        print(f"[task_manifest] Loaded {len(tasks_list)} tasks from {manifest_path}")
    elif cfg.use_hf_dataset:
        cfg.zarr_path = f"hf://{cfg.hf_dataset_repo}/{cfg.hf_dataset_filename}"
        print(
            f"[HF Dataset] Will download zarr from {cfg.hf_dataset_repo} / "
            f"{cfg.hf_dataset_filename}"
        )
    else:
        cfg.zarr_path = str(Path(cfg.zarr_path).resolve())

    cfg.run_root_dir = Path(cfg.run_root_dir).resolve()

    # ------------------------------------------------------------------
    # Proprio dim: max over all tasks / single zarr (for one projector).
    # ------------------------------------------------------------------
    if tasks_list is not None:
        dims: List[int] = []
        for t in tasks_list:
            _st = PushTHiFVLADataset.open_zarr(t["zarr_path"])
            dims.append(int(_st["data/state"].shape[1]))
            del _st
        actual_proprio_dim = max(dims)
        for t, d in zip(tasks_list, dims):
            print(f"  task {t['dataset_name']!r}: state_dim={d}")
        print(f"[Dataset] proprio_output_dim={actual_proprio_dim} (max across tasks)")
        if actual_proprio_dim != PROPRIO_DIM:
            print(
                f"[NOTE] PROPRIO_DIM constant={PROPRIO_DIM} (Push-T default); "
                f"training uses max zarr width {actual_proprio_dim}."
            )
    else:
        _store = PushTHiFVLADataset.open_zarr(cfg.zarr_path)
        actual_proprio_dim = int(_store["data/state"].shape[1])
        del _store
        if actual_proprio_dim != PROPRIO_DIM:
            print(
                f"[WARNING] zarr data/state has {actual_proprio_dim} columns but "
                f"PROPRIO_DIM constant={PROPRIO_DIM}. "
                f"Using actual dataset value ({actual_proprio_dim}) for the projector."
            )
        else:
            print(f"[Dataset] proprio_dim={actual_proprio_dim} matches PROPRIO_DIM constant.")

    print(f"Fine-tuning OpenVLA `{cfg.vla_path}` on Push-T (ACTION_DIM={ACTION_DIM})")

    run_id  = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Distributed setup.
    # accelerate's simple_launcher does not set distributed env vars when
    # --num_processes 1 is used, so PartialState falls back to non-distributed
    # mode and never calls dist.init_process_group().  DDP requires the process
    # group to exist, so we set the required env vars first when they are absent.
    if "RANK" not in os.environ:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    if distributed_state.is_main_process:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=f"ft+{run_id}",
            mode="online",
        )

    print(
        f"Constants: NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK}, "
        f"ACTION_DIM={ACTION_DIM}, PROPRIO_DIM={PROPRIO_DIM}"
    )

    # ------------------------------------------------------------------
    # Load VLA backbone
    # ------------------------------------------------------------------
    if model_is_on_hf_hub(cfg.vla_path):
        cfg.vla_path = snapshot_download(repo_id=cfg.vla_path)
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)

    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # LoRA
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)
    vla.print_trainable_parameters()

    # Gradient checkpointing: trades compute for activation memory.
    # Enables training a 7B model on a 24 GB GPU.
    vla.enable_input_require_grads()
    vla.gradient_checkpointing_enable()

    # FiLM (optional)
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        if cfg.resume:
            sd = load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(sd)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # DDP is incompatible with gradient checkpointing when some parameters
    # are conditionally unused (causes "ready twice" or "static_graph" errors).
    # For single-GPU runs we skip DDP entirely and add a .module alias so the
    # rest of the code (vla.module.llm_dim, vla.module.save_pretrained, etc.)
    # continues to work unchanged.
    if distributed_state.num_processes > 1:
        vla = wrap_ddp(vla, device_id, find_unused=True)
    else:
        # Bypass nn.Module.__setattr__ so 'module' is stored in plain __dict__
        # instead of being registered as a child submodule (which would cause
        # infinite recursion in vla.train() / vla.parameters()).
        object.__setattr__(vla, 'module', vla)

    # Proprio projector — use the dim actually present in the data, not the constant.
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg, device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": actual_proprio_dim},
        )

    # Action head (JointExpert)
    action_head = init_module(
        JointExpert,
        "action_head",
        cfg, device_id,
        {
            "depth":      6,
            "input_dim":  vla.module.llm_dim,
            "hidden_dim": vla.module.llm_dim // 4,
            "action_dim": ACTION_DIM,
        },
        to_bf16=True,
    )

    # Historical-motion encoder
    motion_encoder = init_module(
        HisMotionEncoder,
        "motion_encoder",
        cfg, device_id,
        {
            "in_channels": 2,
            "hidden_dim":  vla.module.llm_dim // 4,
            "out_dim":     vla.module.llm_dim // 4,
            "num_frames":  cfg.history_length // 2,   # temporal size after stride-2 conv
            "num_patches": 64,                         # spatial size after stride-2 conv (8×8)
        },
    )

    # Foresight motion-query tokens
    motion_manager = init_module(
        MotionTokenManager,
        "motion_manager",
        cfg, device_id,
        {"llm_dim": vla.module.llm_dim},
    )
    # motion_token is generated per-batch inside the training loop to handle
    # variable batch sizes (e.g. the last batch may be smaller than cfg.batch_size).

    # Number of vision patches (+ 1 for proprio if used)
    NUM_PATCHES = (
        vla.module.vision_backbone.get_num_patches()
        * vla.module.vision_backbone.get_num_images_in_input()
    )
    if cfg.use_proprio:
        NUM_PATCHES += 1

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    trainable_params += [p for p in motion_encoder.parameters() if p.requires_grad]
    trainable_params += [p for p in action_head.parameters()    if p.requires_grad]
    trainable_params += [p for p in motion_manager.parameters() if p.requires_grad]
    if cfg.use_proprio:
        trainable_params += [p for p in proprio_projector.parameters() if p.requires_grad]

    print(f"Total trainable params: {sum(p.numel() for p in trainable_params):,}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],
        gamma=0.1,
    )
    original_lr = optimizer.param_groups[0]["lr"]

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    if dist.is_available() and dist.is_initialized():
        ddp_rank = dist.get_rank()
        ddp_world_size = dist.get_world_size()
    else:
        ddp_rank = 0
        ddp_world_size = 1

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    if tasks_list is not None:
        train_dataset = MultiTaskPushTHiFVLADataset(
            tasks=tasks_list,
            action_tokenizer=action_tokenizer,
            base_tokenizer=processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
            run_dir=run_dir,
            history_length=cfg.history_length,
            train=True,
            val_fraction=cfg.val_fraction,
            proprio_output_dim=actual_proprio_dim,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )
        flow_stats = train_dataset.rollout_flow_stats_path
    else:
        # Default cache paths: store next to zarr for local datasets, or in
        # run_dir for HF Hub downloads (where the zarr lives inside the HF cache).
        if cfg.zarr_path.startswith("hf://"):
            flow_cache = cfg.flow_cache_path or str(run_dir / "flow_cache.npy")
            flow_stats = cfg.flow_stats_path or str(run_dir / "flow_stats.npz")
        else:
            flow_cache = cfg.flow_cache_path or str(
                Path(cfg.zarr_path).parent / "flow_cache.npy"
            )
            flow_stats = cfg.flow_stats_path or str(
                Path(cfg.zarr_path).parent / "flow_stats.npz"
            )

        train_dataset = PushTHiFVLADataset(
            zarr_path=cfg.zarr_path,
            action_tokenizer=action_tokenizer,
            base_tokenizer=processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
            history_length=cfg.history_length,
            flow_cache_path=flow_cache,
            flow_stats_path=flow_stats,
            stats_save_path=str(run_dir / "dataset_statistics.json"),
            train=True,
            val_fraction=cfg.val_fraction,
            proprio_output_dim=actual_proprio_dim,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )

    if distributed_state.is_main_process:
        eff_bs = cfg.batch_size * ddp_world_size * cfg.grad_accumulation_steps
        print(
            f"[DDP] world_size={ddp_world_size}  per-GPU batch={cfg.batch_size}  "
            f"grad_accum={cfg.grad_accumulation_steps}  "
            f"global batch (approx)={eff_bs}"
        )
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    recent_metrics: Dict = {
        "loss_value":       deque(maxlen=cfg.grad_accumulation_steps),
        "action_l1_loss":   deque(maxlen=cfg.grad_accumulation_steps),
        "motion_mse_loss":  deque(maxlen=cfg.grad_accumulation_steps),
    }

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        # IterableDataset yields each sample once per DataLoader pass.  Without
        # cycling, training would stop after a single epoch (~few k steps) even
        # when max_steps is much larger.  itertools.cycle restarts the loader;
        # each new pass re-calls PushTHiFVLADataset.__iter__ (fresh shuffle).
        for batch_idx, batch in enumerate(itertools.cycle(dataloader)):

            # Regenerate motion token for actual batch size (handles last batch).
            actual_bs = batch["input_ids"].shape[0]
            motion_token = motion_manager.module.get_motion_token(actual_bs)

            loss, metrics = run_forward_pass(
                vla=vla,
                motion_encoder=motion_encoder,
                action_head=action_head,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                motion_token=motion_token,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
            )

            (loss / cfg.grad_accumulation_steps).backward()

            for k, v in metrics.items():
                if k in recent_metrics:
                    recent_metrics[k].append(v)

            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            log_step = (
                gradient_step_idx if not cfg.resume
                else cfg.resume_step + gradient_step_idx
            )

            # W&B logging
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                smoothed = {
                    k: sum(dq) / len(dq)
                    for k, dq in recent_metrics.items() if dq
                }
                wandb.log(
                    {f"Train/{k}": v for k, v in smoothed.items()},
                    step=log_step,
                )

            # LR warmup
            if cfg.lr_warmup_steps > 0:
                lr_prog = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                for pg in optimizer.param_groups:
                    pg["lr"] = original_lr * (0.1 + 0.9 * lr_prog)

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                if cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # Checkpoint
            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    motion_encoder=motion_encoder,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    motion_manager=motion_manager,
                    action_head=action_head,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                )

            # Rollout evaluation (only on main process)
            if (
                cfg.rollout_freq > 0
                and gradient_step_idx > 0
                and log_step % cfg.rollout_freq == 0
                and distributed_state.is_main_process
            ):
                do_training_rollout(
                    cfg=cfg,
                    vla=vla,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    motion_manager=motion_manager,
                    motion_encoder=motion_encoder,
                    device_id=device_id,
                    gradient_step_idx=log_step,
                    flow_stats_path=flow_stats,
                    actual_proprio_dim=actual_proprio_dim,
                )

            if log_step >= cfg.max_steps:
                print(f"Reached max_steps={cfg.max_steps}. Stopping.")
                break


if __name__ == "__main__":
    finetune()
