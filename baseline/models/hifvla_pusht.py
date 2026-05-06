"""
hifvla_pusht.py

Push-T / Swap-T specific configuration and model factory for HiF-VLA.

This file provides:
  - ``PushTHiFVLAConfig`` — a dataclass that centralises every
    hyper-parameter needed to build or evaluate the HiF-VLA model on the
    Swap-T task.  Training code (finetune_pusht.py) uses FinetuneConfig
    which mirrors these fields; this class exists for clean stand-alone
    usage and documentation.
  - ``build_hifvla_pusht()`` — factory that instantiates all HiF-VLA
    modules (VLA backbone, action head, proprio projector, motion encoder,
    motion token manager) with Push-T dimensions and returns them as a dict
    ready for use in a training or eval loop.

Swap-T task adaptations vs. the LIBERO / CALVIN originals:
  ┌────────────────────────┬───────────────────────────────────────────────┐
  │ Original (LIBERO)      │ Push-T / Swap-T                               │
  ├────────────────────────┼───────────────────────────────────────────────┤
  │ ACTION_DIM = 7         │ ACTION_DIM = 2  (xy delta)                    │
  │ PROPRIO_DIM = 8        │ PROPRIO_DIM = 8 (agent_xy + blue + red poses) │
  │ NUM_ACTIONS_CHUNK = 8  │ NUM_ACTIONS_CHUNK = 8  (unchanged)            │
  │ MPEG-4 motion vectors  │ Farneback optical flow (16×16 spatial grid)   │
  │ Per-episode language   │ Fixed string "swap the colored blocks …"      │
  └────────────────────────┴───────────────────────────────────────────────┘
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# HiF-VLA root must be on sys.path before importing prismatic / motion_layers.
_HERE = Path(__file__).resolve().parent               # baseline/models/
_PROJECT_ROOT = _HERE.parent.parent                   # project root
_HIF_VLA_ROOT = _PROJECT_ROOT / "HiF-VLA"
for _p in [str(_PROJECT_ROOT), str(_HIF_VLA_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class PushTHiFVLAConfig:
    """All hyper-parameters for HiF-VLA on the Swap-T task."""

    # ── Paths ────────────────────────────────────────────────────────────
    pretrained_checkpoint: str = ""
    """Path to the openvla-7b base weights (local dir or HF Hub id)."""

    # ── Task dimensions (must match PUSHT_CONSTANTS in constants.py) ──────
    action_dim: int = 2
    """Push-T uses 2-D continuous (x, y) delta actions."""

    num_actions_chunk: int = 8
    """Action prediction horizon (number of future steps per forward pass)."""

    proprio_dim: int = 8
    """State = agent_xy (2) + blue_block_xytheta (3) + red_block_xytheta (3)."""

    # ── Motion encoder ────────────────────────────────────────────────────
    history_length: int = 8
    """Number of historical optical-flow frames fed to the motion encoder."""

    flow_h: int = 16
    """Spatial height of the downsampled flow grid (pixels → macroblocks)."""

    flow_w: int = 16
    """Spatial width of the downsampled flow grid."""

    # ── HiF-VLA architecture ─────────────────────────────────────────────
    n_joint_expert_layers: int = 6
    """Depth of the JointExpert action head (paper default = 6)."""

    mv_loss_weight: float = 0.01
    """λ: weight of the motion-prediction MSE loss relative to action L1."""

    # ── LoRA ─────────────────────────────────────────────────────────────
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0

    # ── Misc ─────────────────────────────────────────────────────────────
    use_film: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = True
    use_l1_regression: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    unnorm_key: str = "swap_3t"
    """Key into dataset_statistics.json used for action / proprio unnorm."""

    language_instruction: str = "swap the colored blocks to their target positions"
    """Fixed language prompt used for all Swap-T episodes."""

    # Derived (set automatically from the loaded VLA, not by the user)
    _llm_dim: Optional[int] = field(default=None, repr=False)

    @property
    def llm_dim(self) -> int:
        if self._llm_dim is None:
            raise ValueError("llm_dim is not set — call build_hifvla_pusht() first.")
        return self._llm_dim

    @property
    def motion_hidden_dim(self) -> int:
        return self.llm_dim // 4

    @property
    def num_frames_after_conv(self) -> int:
        """Temporal size of flow feature map after stride-2 3-D conv."""
        return self.history_length // 2

    @property
    def num_patches_after_conv(self) -> int:
        """Spatial size of flow feature map after stride-2 3-D conv (H/2 × W/2)."""
        return (self.flow_h // 2) * (self.flow_w // 2)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_hifvla_pusht(
    cfg: PushTHiFVLAConfig,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Instantiate all HiF-VLA components configured for the Swap-T task.

    Returns a dict with keys:
        ``vla``               — OpenVLAForActionPrediction (LoRA-wrapped)
        ``processor``         — PrismaticProcessor
        ``action_head``       — JointExpert  (6-layer Transformer)
        ``proprio_projector`` — ProprioProjector  (or None if use_proprio=False)
        ``motion_encoder``    — HisMotionEncoder  (3-D conv + ViT)
        ``motion_manager``    — MotionTokenManager (foresight query tokens)
        ``config``            — the updated PushTHiFVLAConfig (llm_dim filled in)

    Parameters
    ----------
    cfg : PushTHiFVLAConfig
        Configuration (``pretrained_checkpoint`` must be set).
    device : torch.device
        Target device.  For multi-GPU training use the per-process device_id
        from Accelerate instead of calling this factory directly.
    """
    assert cfg.pretrained_checkpoint, "Set cfg.pretrained_checkpoint before calling build_hifvla_pusht()."

    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.models.action_heads import JointExpert, MotionTokenManager
    from prismatic.models.projectors import ProprioProjector
    from motion_layers.motion_tokenizer import HisMotionEncoder
    from experiments.robot.openvla_utils import model_is_on_hf_hub, update_auto_map, check_model_logic_mismatch

    if not model_is_on_hf_hub(cfg.pretrained_checkpoint):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        update_auto_map(cfg.pretrained_checkpoint)
        check_model_logic_mismatch(cfg.pretrained_checkpoint)

    # ── VLA backbone ──────────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # LoRA
    if cfg.use_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)

    vla = vla.to(device)
    llm_dim: int = vla.llm_dim
    cfg._llm_dim = llm_dim

    # ── Action head (JointExpert) ─────────────────────────────────────────
    action_head = JointExpert(
        depth=cfg.n_joint_expert_layers,
        input_dim=llm_dim,
        hidden_dim=llm_dim // 4,
        action_dim=cfg.action_dim,          # 2 for Push-T
    ).to(torch.bfloat16).to(device)

    # ── Proprio projector ──────────────────────────────────────────────────
    proprio_projector: Optional[nn.Module] = None
    if cfg.use_proprio:
        proprio_projector = ProprioProjector(
            llm_dim=llm_dim,
            proprio_dim=cfg.proprio_dim,    # 8 for Swap-T
        ).to(torch.bfloat16).to(device)

    # ── Historical motion encoder ──────────────────────────────────────────
    motion_encoder = HisMotionEncoder(
        in_channels=2,                          # optical flow: u + v
        hidden_dim=llm_dim // 4,
        out_dim=llm_dim // 4,
        num_frames=cfg.num_frames_after_conv,   # history_length // 2 = 4
        num_patches=cfg.num_patches_after_conv, # (16//2)^2 = 64
    ).to(torch.bfloat16).to(device)

    # ── Foresight motion-query tokens ──────────────────────────────────────
    motion_manager = MotionTokenManager(llm_dim=llm_dim).to(torch.bfloat16).to(device)

    print(
        f"[build_hifvla_pusht] Built model:\n"
        f"  llm_dim          = {llm_dim}\n"
        f"  action_dim       = {cfg.action_dim}\n"
        f"  proprio_dim      = {cfg.proprio_dim}\n"
        f"  num_actions_chunk= {cfg.num_actions_chunk}\n"
        f"  history_length   = {cfg.history_length}\n"
        f"  flow_grid        = {cfg.flow_h}×{cfg.flow_w}\n"
    )

    return {
        "vla":               vla,
        "processor":         processor,
        "action_head":       action_head,
        "proprio_projector": proprio_projector,
        "motion_encoder":    motion_encoder,
        "motion_manager":    motion_manager,
        "config":            cfg,
    }
