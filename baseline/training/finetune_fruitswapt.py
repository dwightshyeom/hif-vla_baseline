"""
finetune_fruitswapt.py

Fine-tunes HiF-VLA on the FruitSwapVision robosuite task.

This script is a direct adaptation of baseline/training/finetune_pusht.py with
the following differences:
  - Reads from a Zarr store converted from robosuite HDF5 via
    baseline/dataset/convert_robosuite_to_zarr.py.
  - ACTION_DIM=7, PROPRIO_DIM=8  (auto-detected via LIBERO defaults in
    prismatic/vla/constants.py because "pusht" is NOT in argv).
  - Fixed language instruction describes the fruit-swap task.
  - In-training rollout is disabled by default (rollout_freq=0); use the
    separate baseline/eval/run_fruitswapt_eval.py script instead.

Run with:
    accelerate launch --num_processes <N_GPU> \\
        baseline/training/finetune_fruitswapt.py \\
        --vla_path openvla/openvla-7b \\
        --zarr_path /path/to/fruit_swap_zarr \\
        --run_root_dir runs/fruitswapt_hifvla \\
        --use_l1_regression True \\
        --use_proprio True \\
        --batch_size 4 \\
        --max_steps 50005

Or via config file:
    accelerate launch --num_processes <N_GPU> \\
        baseline/training/finetune_fruitswapt.py \\
        --config configs/train_hifvla_fruitswapt.yaml
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
# Path setup — must happen BEFORE any HiF-VLA imports so that the "libero"
# detection path in prismatic/vla/constants.py is used (ACTION_DIM=7,
# PROPRIO_DIM=8).  Do NOT include "pusht" here.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # baseline/training/
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent              # project root
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
# Task-specific constants
# ---------------------------------------------------------------------------

FRUITSWAPT_LANGUAGE_INSTRUCTION: str = (
    "use the empty patch as a buffer to swap the magenta and cyan cubes to each other's starting positions"
)
FRUITSWAPT_DATASET_NAME: str = "fruit_swap_vision"


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"

    # Dataset
    zarr_path: str = ""   # Path to fruit_swap_zarr (ignored if task_manifest set)
    task_manifest: Optional[str] = None  # JSON list of tasks: zarr_path, dataset_name, language
    run_root_dir: Path = Path("runs/fruitswapt_hifvla")
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

    # Rollout during training (disabled by default; use run_fruitswapt_eval.py instead)
    rollout_freq: int = 0
    n_rollout_episodes: int = 3
    rollout_max_steps: int = 600
    rollout_fps: int = 10
    rollout_unnorm_key: str = FRUITSWAPT_DATASET_NAME
    rollout_num_open_loop: int = 8

    # Logging
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "hifvla-fruitswapt"
    run_id_note: Optional[str] = None
    run_id_override: Optional[str] = None
    wandb_log_freq: int = 10
    # fmt: on


# ---------------------------------------------------------------------------
# Utilities (identical to finetune_pusht.py)
# ---------------------------------------------------------------------------

def _resolve_path_for_training(
    raw: str,
    project_root: Path,
    *,
    must_be_file: bool = False,
) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    cand_cwd = (Path.cwd() / p).resolve()
    exists_cwd = cand_cwd.is_file() if must_be_file else cand_cwd.exists()
    if exists_cwd:
        return cand_cwd
    cand_root = (project_root / p).resolve()
    exists_root = cand_root.is_file() if must_be_file else cand_root.exists()
    if exists_root:
        return cand_root
    return cand_cwd


def load_task_manifest(manifest_path: Path, project_root: Path) -> List[Dict[str, Any]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("task manifest must be a JSON array of task objects")
    tasks: List[Dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"task_manifest[{i}] must be a JSON object")
        zp   = item.get("zarr_path")
        dname = item.get("dataset_name")
        lang  = item.get("language_instruction") or item.get("language")
        if not zp:
            raise ValueError(f"task_manifest[{i}] missing 'zarr_path'")
        if not dname:
            raise ValueError(f"task_manifest[{i}] missing 'dataset_name'")
        if not lang:
            raise ValueError(f"task_manifest[{i}] needs 'language' or 'language_instruction'")
        if isinstance(zp, str) and zp.startswith("hf://"):
            zpath_resolved = zp
        else:
            zpath_resolved = str(_resolve_path_for_training(zp, project_root))
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
    use_manifest = bool(cfg.task_manifest and str(cfg.task_manifest).strip())
    task_tag = "multitask" if use_manifest else FRUITSWAPT_DATASET_NAME
    run_id = (
        f"{cfg.vla_path.split('/')[-1]}+{task_tag}"
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

    gt_motion       = batch["mv_future"].to(device_id).to(torch.bfloat16)
    motion_res_loss = compute_motion_mse_loss(res_motion, gt_motion)
    pred_action_loss = torch.nn.L1Loss()(ground_truth_actions, pre_action)

    if use_l1_regression:
        loss = 0.01 * motion_res_loss + pred_action_loss
        metrics["loss_value"]      = loss.item()
        metrics["action_l1_loss"]  = pred_action_loss.item()
        metrics["motion_mse_loss"] = motion_res_loss.item()
    else:
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)
        curr_acc = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        curr_l1 = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids,
            mask=current_action_mask,
        )
        metrics.update({
            "loss_value":           loss.item(),
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
    train_dataset,
    distributed_state: PartialState,
) -> None:
    if cfg.save_latest_checkpoint_only:
        ckpt_dir    = run_dir
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
# Main training function
# ---------------------------------------------------------------------------

@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    assert cfg.use_lora,          "Only LoRA fine-tuning is supported."
    assert cfg.use_l1_regression, "Must use L1 regression (use_l1_regression=True)."

    use_manifest = bool(cfg.task_manifest and str(cfg.task_manifest).strip())
    assert use_manifest or cfg.zarr_path, (
        "Provide --task_manifest OR --zarr_path."
    )

    cfg.vla_path = cfg.vla_path.rstrip("/")
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
                f"(resolved to {manifest_path})."
            )
        tasks_list = load_task_manifest(manifest_path, _PROJECT_ROOT)
        print(f"[task_manifest] Loaded {len(tasks_list)} tasks from {manifest_path}")
    else:
        cfg.zarr_path = str(Path(cfg.zarr_path).resolve())

    cfg.run_root_dir = Path(cfg.run_root_dir).resolve()

    # ------------------------------------------------------------------
    # Proprio dim: read actual dim from zarr data/state
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
    else:
        _store = PushTHiFVLADataset.open_zarr(cfg.zarr_path)
        actual_proprio_dim = int(_store["data/state"].shape[1])
        del _store
        print(f"[Dataset] proprio_dim={actual_proprio_dim}")

    print(
        f"Fine-tuning OpenVLA on FruitSwapVision "
        f"(ACTION_DIM={ACTION_DIM}, PROPRIO_DIM={actual_proprio_dim})"
    )

    run_id  = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Distributed setup
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

    vla.enable_input_require_grads()
    vla.gradient_checkpointing_enable()

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

    if distributed_state.num_processes > 1:
        vla = wrap_ddp(vla, device_id, find_unused=True)
    else:
        object.__setattr__(vla, 'module', vla)

    # Proprio projector
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
            "action_dim": ACTION_DIM,   # 7 for FruitSwap
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
            "num_frames":  cfg.history_length // 2,
            "num_patches": 64,
        },
    )

    # Foresight motion-query tokens
    motion_manager = init_module(
        MotionTokenManager,
        "motion_manager",
        cfg, device_id,
        {"llm_dim": vla.module.llm_dim},
    )

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
        )
        flow_stats = train_dataset.rollout_flow_stats_path
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
            dataset_name=FRUITSWAPT_DATASET_NAME,
            language_instruction=FRUITSWAPT_LANGUAGE_INSTRUCTION,
            proprio_output_dim=actual_proprio_dim,
        )

    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # ------------------------------------------------------------------
    # Rank-based dataset sharding for true DDP data-parallel speedup.
    # Each process owns a non-overlapping 1/world_size slice of the
    # valid_indices list, so every gradient step collectively covers
    # batch_size * world_size distinct samples instead of each GPU
    # independently sampling from the full pool.
    # ------------------------------------------------------------------
    if distributed_state.num_processes > 1:
        rank       = distributed_state.local_process_index
        world_size = distributed_state.num_processes
        if hasattr(train_dataset, "valid_indices"):
            # Single-task dataset: shard directly.
            train_dataset.valid_indices = train_dataset.valid_indices[rank::world_size]
            print(
                f"[Rank {rank}/{world_size}] Sharded to "
                f"{len(train_dataset.valid_indices)} valid steps"
            )
        elif hasattr(train_dataset, "_datasets"):
            # MultiTask dataset: shard each child.
            for ds in train_dataset._datasets:
                ds.valid_indices = ds.valid_indices[rank::world_size]
            total = sum(len(ds.valid_indices) for ds in train_dataset._datasets)
            print(
                f"[Rank {rank}/{world_size}] MultiTask sharded to "
                f"{total} valid steps across {len(train_dataset._datasets)} tasks"
            )

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
        num_workers=2,          # overlap data prep with GPU compute
        persistent_workers=True,
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
        for batch_idx, batch in enumerate(itertools.cycle(dataloader)):

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

            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                smoothed = {
                    k: sum(dq) / len(dq)
                    for k, dq in recent_metrics.items() if dq
                }
                wandb.log(
                    {f"Train/{k}": v for k, v in smoothed.items()},
                    step=log_step,
                )

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

            if log_step >= cfg.max_steps:
                print(f"Reached max_steps={cfg.max_steps}. Stopping.")
                break


if __name__ == "__main__":
    finetune()
