"""
Diffusion Policy (vision-based) conditioned on observation + hidden states from
a **finetunable** pretrained ObsActionChunkLSTMImage.

Key differences from the frozen variant
(diffusion_unet_hybrid_image_policy_with_obs_action_chunk_lstm.py):

Training
--------
  - The LSTM *core* (recurrent weights + heads) is **trainable**.  Gradients
    from the diffusion action loss flow back through the LSTM recurrent weights
    so they can adapt to better serve the downstream DP.
  - The LSTM's internal visual encoder (CNN) is kept **frozen** by default to
    avoid prohibitively large memory usage during full-episode forward passes.
  - During training, the LSTM core is run live on per-episode obs-features
    (pre-encoded by the frozen encoder in the dataset) + actions, giving a
    fresh hidden state with gradients at the chunk position.

Rollout / predict_action
------------------------
  Identical to the frozen variant: the LSTM is stepped once per DP decision
  using the current observation and the last executed action.

Finetuning schedule
-------------------
  The workspace controls which parameters are trainable via three helpers:
    freeze_lstm()    / unfreeze_lstm()   — LSTM core parameters
    freeze_dp()      / unfreeze_dp()     — DP (UNet + obs_encoder + projections)
  Schedules: 'joint', 'warmup_then_joint', 'alternating'
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from pathlib import Path

from memory_diffusion_policy.model.common.normalizer import LinearNormalizer
from memory_diffusion_policy.policy.base_image_policy import BaseImagePolicy
from memory_diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from memory_diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from memory_diffusion_policy.model.obs_action_chunk_lstm_image import ObsActionChunkLSTMImage
from memory_diffusion_policy.model.obs_action_chunk_lstm import build_dct_basis, dct_forward
from robomimic.algo import algo_factory
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import memory_diffusion_policy.model.vision.crop_randomizer as dmvc


def _load_torch(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ======================================================================
# Vector Quantizer (shared with frozen variant)
# ======================================================================

class VectorQuantizer(nn.Module):
    """Straight-through VQ for LSTM hidden-state discretisation."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight,
                         -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        squeeze = False
        if z.dim() == 2:
            z = z.unsqueeze(1)
            squeeze = True

        B, T, D = z.shape
        flat = z.reshape(-1, D)

        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2.0 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        idx = dist.argmin(dim=1)
        z_q_flat = self.embedding(idx)

        codebook_loss = F.mse_loss(z_q_flat, flat.detach())
        commitment_loss = F.mse_loss(flat, z_q_flat.detach())
        vq_loss = codebook_loss + self.commitment_weight * commitment_loss

        z_q = z + (z_q_flat.reshape(B, T, D) - z).detach()

        if squeeze:
            z_q = z_q.squeeze(1)
        return z_q, vq_loss


# ======================================================================
# Attention Gate (shared with frozen variant)
# ======================================================================

class HiddenStateAttentionGate(nn.Module):
    """Channel-wise attention gate for LSTM hidden states."""

    def __init__(
        self,
        hidden_dim: int,
        obs_dim: int = 0,
        reduction: int = 4,
        mode: str = "self",
    ) -> None:
        super().__init__()
        assert mode in ("self", "obs_conditioned")
        if mode == "obs_conditioned" and obs_dim <= 0:
            raise ValueError(
                "obs_dim must be > 0 for obs_conditioned attention gate")
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim

        in_dim = hidden_dim + obs_dim if mode == "obs_conditioned" else hidden_dim
        mid_dim = max(hidden_dim // reduction, 8)

        self.gate_net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h: torch.Tensor,
        obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.mode == "obs_conditioned":
            if obs is None:
                raise ValueError(
                    "obs must be provided for obs_conditioned attention gate")
            gate_input = torch.cat([h, obs], dim=-1)
        else:
            gate_input = h
        gate = self.gate_net(gate_input)
        return h * gate


# ======================================================================
# Main policy class
# ======================================================================

class DiffusionUnetHybridImagePolicyWithObsActionChunkLSTMFinetune(BaseImagePolicy):
    """
    Vision-based Diffusion Policy conditioned on a finetunable LSTM's
    hidden state.

    The LSTM core is trained jointly with the DP (after an optional
    warmup phase where only the DP is trained).  The LSTM's visual
    encoder is always frozen to keep memory requirements manageable.
    """

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        crop_shape=(76, 76),
        num_kp=32,
        diffusion_step_embed_dim=128,
        down_dims=(512, 1024, 2048),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        obs_encoder_group_norm=True,
        eval_fixed_crop=True,
        # ---- LSTM parameters ----
        lstm_checkpoint_path: Optional[str] = None,
        lstm_hidden_size: int = 64,
        # Hidden projection
        use_hidden_projection: bool = True,
        projection_output_dim: Optional[int] = None,
        projection_num_layers: int = 2,
        # VQ
        use_vq: bool = False,
        vq_num_embeddings: int = 128,
        vq_commitment_weight: float = 0.25,
        vq_loss_weight: float = 1.0,
        # Attention gate
        use_attention_gate: bool = False,
        attention_gate_reduction: int = 4,
        attention_gate_mode: str = "self",
        # Obs feature projection
        use_obs_projection: bool = False,
        obs_projection_output_dim: Optional[int] = None,
        obs_projection_num_layers: int = 2,
        # Finetuning options
        lstm_lr_scale: float = 0.1,
        freeze_lstm_epochs: int = 0,
        # Intermediate obs feeding (rollout only)
        feed_intermediate_obs_to_lstm: bool = False,
        # LSTM latent type
        lstm_latent_type: str = "hidden",
        # ---- DCT action-chunk loss ----
        # Adds an MSE loss on truncated DCT coefficients of the predicted
        # action chunk (recovered from epsilon prediction) vs the GT action
        # chunk (both in DP-normalised space).
        #   use_dct_loss:    enable the DCT loss
        #   dct_loss_weight: alpha — scale applied to the DCT loss
        #   dct_loss_on_dp:  controls post-warmup routing only (the warm-up
        #                    phase always suppresses DCT — see workspace).
        #       True  → all components get raw_loss + alpha*dct_loss
        #               (single backward).
        #       False → DP-only params (UNet, obs_encoder, obs_projection)
        #                  receive raw_loss only.
        #               LSTM-pathway params (LSTM core + hidden_vq +
        #                  hidden_attention_gate + hidden_projection)
        #                  receive raw_loss + alpha*dct_loss.
        #               (workspace does a dual backward: raw_loss globally,
        #                then alpha*dct_loss restricted to the LSTM-pathway
        #                via torch.autograd.grad and added on top.)
        #   dct_n_bases:     number of DCT basis functions to keep
        #                    (≤ horizon). When == horizon the basis is
        #                    full and ||DCT(diff)||² == ||diff||² (Parseval),
        #                    so the DCT loss equals an MSE on x_0_pred vs GT.
        #                    < horizon → low-pass focus on smooth components.
        use_dct_loss: bool = False,
        dct_loss_weight: float = 1.0,
        dct_loss_on_dp: bool = False,
        dct_n_bases: Optional[int] = None,
        # ---- scheduler.step kwargs ----
        **kwargs,
    ):
        super().__init__()
        assert obs_as_global_cond, \
            "Only obs_as_global_cond=True is supported."

        # ================================================================
        # 1.  DP obs_encoder (same architecture as vanilla hybrid policy)
        # ================================================================
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]

        obs_config = {"low_dim": [], "rgb": [], "depth": [], "scan": []}
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)
            type_ = attr.get("type", "low_dim")
            if type_ == "rgb":
                obs_config["rgb"].append(key)
            elif type_ == "low_dim":
                obs_config["low_dim"].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type_}")

        config = get_robomimic_config(
            algo_name="bc_rnn", hdf5_type="image",
            task_name="square", dataset_type="ph")
        with config.unlocked():
            config.observation.modalities.obs = obs_config
            config.observation.encoder.rgb.core_kwargs.pool_kwargs.num_kp = num_kp
            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality["obs_randomizer_class"] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        ObsUtils.initialize_obs_utils_with_config(config)
        policy_algo = algo_factory(
            algo_name=config.algo_name, config=config,
            obs_key_shapes=obs_key_shapes, ac_dim=action_dim, device="cpu")
        obs_encoder = policy_algo.nets["policy"].nets["encoder"].nets["obs"]

        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features))
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc))

        obs_feature_dim = obs_encoder.output_shape()[0]

        # Obs feature projection
        self.obs_projection: Optional[nn.Sequential] = None
        if use_obs_projection and obs_projection_output_dim is not None:
            layers = []
            in_dim = obs_feature_dim
            for _ in range(obs_projection_num_layers - 1):
                mid_dim = max(obs_projection_output_dim,
                              (in_dim + obs_projection_output_dim) // 2)
                layers.extend([nn.Linear(in_dim, mid_dim), nn.ReLU()])
                in_dim = mid_dim
            layers.append(nn.Linear(in_dim, obs_projection_output_dim))
            self.obs_projection = nn.Sequential(*layers)
            projected_obs_feature_dim = obs_projection_output_dim
            print(f"Obs projection: {obs_feature_dim} → {obs_projection_output_dim} "
                  f"({obs_projection_num_layers} layers)")
        else:
            projected_obs_feature_dim = obs_feature_dim

        # ================================================================
        # 2.  LSTM conditioning
        # ================================================================
        self._lstm_latent_type = lstm_latent_type
        self.lstm_hidden_size = lstm_hidden_size
        self.use_hidden_projection = use_hidden_projection
        self.projection_output_dim = projection_output_dim
        self.use_vq = use_vq
        self.vq_loss_weight = vq_loss_weight
        self.lstm_lr_scale = lstm_lr_scale
        self.freeze_lstm_epochs = freeze_lstm_epochs

        # Pretrained LSTM (TRAINABLE core, frozen encoder)
        self.lstm_model: Optional[ObsActionChunkLSTMImage] = None
        self._lstm_normalizer: Optional[LinearNormalizer] = None
        if lstm_checkpoint_path is not None:
            self._load_lstm_checkpoint(lstm_checkpoint_path, shape_meta)

        # Hidden projection
        self.hidden_projection: Optional[nn.Sequential] = None
        if use_hidden_projection and projection_output_dim is not None:
            layers = []
            in_dim = self.lstm_hidden_size
            for _ in range(projection_num_layers - 1):
                mid_dim = max(projection_output_dim,
                              (in_dim + projection_output_dim) // 2)
                layers.extend([nn.Linear(in_dim, mid_dim), nn.ReLU()])
                in_dim = mid_dim
            layers.append(nn.Linear(in_dim, projection_output_dim))
            self.hidden_projection = nn.Sequential(*layers)
            print(f"Hidden projection: {self.lstm_hidden_size} → {projection_output_dim} "
                  f"({projection_num_layers} layers)")

        # VQ
        self.hidden_vq: Optional[VectorQuantizer] = None
        if use_vq:
            self.hidden_vq = VectorQuantizer(
                num_embeddings=vq_num_embeddings,
                embedding_dim=self.lstm_hidden_size,
                commitment_weight=vq_commitment_weight)
            print(f"VQ: {vq_num_embeddings} codes, dim={self.lstm_hidden_size}")

        # Attention gate
        self.hidden_attention_gate: Optional[HiddenStateAttentionGate] = None
        if use_attention_gate:
            gate_obs_dim = (
                projected_obs_feature_dim
                if attention_gate_mode == "obs_conditioned"
                else 0)
            self.hidden_attention_gate = HiddenStateAttentionGate(
                hidden_dim=self.lstm_hidden_size,
                obs_dim=gate_obs_dim,
                reduction=attention_gate_reduction,
                mode=attention_gate_mode)
            print(f"Attention gate: mode={attention_gate_mode!r}, "
                  f"hidden_dim={self.lstm_hidden_size}, reduction={attention_gate_reduction}")

        # Compute hidden feature dim after optional VQ + projection
        if use_hidden_projection and projection_output_dim is not None:
            hidden_feature_dim = projection_output_dim
        else:
            hidden_feature_dim = self.lstm_hidden_size

        # ================================================================
        # 3.  Diffusion U-Net
        # ================================================================
        global_cond_dim = (
            (projected_obs_feature_dim + hidden_feature_dim) * n_obs_steps)

        model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale)

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False)
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.projected_obs_feature_dim = projected_obs_feature_dim
        self.hidden_feature_dim = hidden_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self._feed_intermediate_obs = feed_intermediate_obs_to_lstm
        self._intermediates_updated = False
        self._last_decision_obs = None
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # Stateful rollout buffers (identical to frozen variant)
        self._lstm_hidden = None
        self._prev_action = None

        # ---- DCT loss configuration ----
        self.use_dct_loss = bool(use_dct_loss)
        self.dct_loss_weight = float(dct_loss_weight)
        self.dct_loss_on_dp = bool(dct_loss_on_dp)
        if self.use_dct_loss:
            n_bases = int(dct_n_bases) if dct_n_bases is not None else int(horizon)
            assert 1 <= n_bases <= horizon, (
                f"dct_n_bases={n_bases} must be in [1, horizon={horizon}]")
            self.dct_n_bases = n_bases
            basis = build_dct_basis(N=int(horizon), K=n_bases)  # (horizon, K)
            self.register_buffer("_dct_basis", basis)
            print(
                f"DCT loss enabled: weight={self.dct_loss_weight}, "
                f"on_dp={self.dct_loss_on_dp}, "
                f"n_bases={n_bases}/{horizon}"
            )
        else:
            self.dct_n_bases = None

        print(f"Diffusion params:  {sum(p.numel() for p in self.model.parameters()):,.0f}")
        print(f"Encoder params:    {sum(p.numel() for p in self.obs_encoder.parameters()):,.0f}")
        if self.lstm_model is not None:
            print(f"LSTM core params:  "
                  f"{sum(p.numel() for p in self.lstm_model.lstm.parameters()):,.0f} (trainable)")
            print(f"LSTM encoder params: "
                  f"{sum(p.numel() for p in self.lstm_model.obs_encoder.parameters()):,.0f} (frozen)")

    # ------------------------------------------------------------------
    # LSTM checkpoint loading (LSTM core trainable, encoder frozen)
    # ------------------------------------------------------------------

    def _load_lstm_checkpoint(
        self,
        checkpoint_path: str,
        shape_meta: dict,
    ):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"LSTM checkpoint not found: {checkpoint_path}")

        ckpt = _load_torch(checkpoint_path)
        ckpt_args = ckpt.get("args", {})

        self.lstm_model = ObsActionChunkLSTMImage(
            shape_meta=shape_meta,
            action_dim=int(ckpt_args.get("action_dim", 2)),
            hidden_size=int(ckpt_args.get("hidden_size", 256)),
            num_layers=int(ckpt_args.get("num_layers", 2)),
            dropout=float(ckpt_args.get("dropout", 0.1)),
            chunk_H=int(ckpt_args.get("chunk_H", 16)),
            use_obs_head=False,
            use_future_head=bool(ckpt_args.get("use_future_head", True)),
            num_future_modes=int(ckpt_args.get("num_future_modes", 1)),
            future_abstraction=ckpt_args.get("future_abstraction", "raw"),
            future_n_bases=int(ckpt_args.get("future_n_bases", 32)),
            use_past_head=bool(ckpt_args.get("use_past_head", True)),
            past_chunk_H=int(ckpt_args.get(
                "past_chunk_H", ckpt_args.get("chunk_H", 16))),
            past_abstraction=ckpt_args.get("past_abstraction", "raw"),
            past_n_bases=int(ckpt_args.get("past_n_bases", 32)),
            use_vq=bool(ckpt_args.get("use_vq", False)),
            vq_n_codes=int(ckpt_args.get("vq_n_codes", 512)),
            vq_commitment_weight=float(
                ckpt_args.get("vq_commitment_weight", 0.25)),
            crop_shape=tuple(ckpt_args.get("crop_shape", (76, 76))),
            obs_encoder_group_norm=bool(
                ckpt_args.get("obs_encoder_group_norm", True)),
            eval_fixed_crop=True,
            freeze_encoder=True,   # encoder always frozen
        )
        self.lstm_model.load_state_dict(ckpt["model_state"])

        # Freeze visual encoder, leave LSTM core trainable
        for p in self.lstm_model.obs_encoder.parameters():
            p.requires_grad_(False)
        for p in self.lstm_model.lstm.parameters():
            p.requires_grad_(True)

        inner = self.lstm_model.lstm
        lstm_latent_dim = inner.hidden_size  # only 'hidden' supported
        if lstm_latent_dim != self.lstm_hidden_size:
            import warnings
            warnings.warn(
                f"Config lstm_hidden_size={self.lstm_hidden_size} != "
                f"actual={lstm_latent_dim}. Overwriting.")
        self.lstm_hidden_size = lstm_latent_dim

        # Save the LSTM normalizer path for later
        # (set externally via set_lstm_normalizer)
        self._lstm_normalizer = None

        print(f"Loaded ObsActionChunkLSTMImage from: {checkpoint_path}")
        print(f"  hidden_size={inner.hidden_size}, "
              f"obs_feature_dim={self.lstm_model.obs_feature_dim}")
        print("  LSTM core: TRAINABLE | LSTM encoder: FROZEN")

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def _dp_parameters(self):
        """Yield all DP parameters (obs_encoder + model + projections)."""
        yield from self.obs_encoder.parameters()
        yield from self.model.parameters()
        if self.obs_projection is not None:
            yield from self.obs_projection.parameters()
        if self.hidden_projection is not None:
            yield from self.hidden_projection.parameters()
        if self.hidden_vq is not None:
            yield from self.hidden_vq.parameters()
        if self.hidden_attention_gate is not None:
            yield from self.hidden_attention_gate.parameters()

    def _lstm_core_parameters(self):
        """Yield LSTM core parameters (NOT the encoder)."""
        if self.lstm_model is not None:
            yield from self.lstm_model.lstm.parameters()

    def _lstm_pathway_parameters(self):
        """Yield ALL parameters along the LSTM-conditioning pathway.

        Used by the workspace's dual-backward path (post warm-up) when
        ``dct_loss_on_dp=False`` — these params receive ``raw_loss + α·dct_loss``
        (the raw loss reaches them via a global backward pass, then the
        DCT contribution is added on top via ``torch.autograd.grad``). The
        DP-only params (yielded by :meth:`_dp_only_parameters`) receive
        only the raw loss in that branch.

        Includes:
          - LSTM core (recurrent weights, heads)          — trainable when unfrozen
          - DP-side hidden_vq (VQ on LSTM hidden state)   — if enabled
          - DP-side hidden_attention_gate                  — if enabled
          - DP-side hidden_projection (MLP projector)      — if enabled

        Excludes:
          - LSTM visual encoder (always frozen)
          - DP UNet, DP obs_encoder, DP obs_projection
        """
        if self.lstm_model is not None:
            yield from self.lstm_model.lstm.parameters()
        if self.hidden_vq is not None:
            yield from self.hidden_vq.parameters()
        if self.hidden_attention_gate is not None:
            yield from self.hidden_attention_gate.parameters()
        if self.hidden_projection is not None:
            yield from self.hidden_projection.parameters()

    def _dp_only_parameters(self):
        """Yield DP-only parameters that DO NOT touch the LSTM hidden state.

        Counterpart to :meth:`_lstm_pathway_parameters`. The two together
        cover everything in :meth:`_dp_parameters` ∪ :meth:`_lstm_core_parameters`
        with no overlap.

        Used by the workspace's dual-backward path (post warm-up) when
        ``dct_loss_on_dp=False`` — these params receive ONLY the raw loss
        (no DCT contribution). Note that these params are no longer
        ``inputs=`` restricted: the post-warmup raw backward is global, so
        DP gets raw via the normal graph traversal.

        Includes:
          - DP obs_encoder
          - DP UNet (self.model)
          - DP obs_projection (if enabled)
        """
        yield from self.obs_encoder.parameters()
        yield from self.model.parameters()
        if self.obs_projection is not None:
            yield from self.obs_projection.parameters()

    def freeze_lstm(self):
        """Freeze the LSTM core (encoder is always frozen)."""
        for p in self._lstm_core_parameters():
            p.requires_grad_(False)

    def unfreeze_lstm(self):
        """Unfreeze the LSTM core."""
        for p in self._lstm_core_parameters():
            p.requires_grad_(True)

    def freeze_dp(self):
        """Freeze the DP (obs_encoder + UNet + projections)."""
        for p in self._dp_parameters():
            p.requires_grad_(False)

    def unfreeze_dp(self):
        """Unfreeze the DP."""
        for p in self._dp_parameters():
            p.requires_grad_(True)

    def get_parameter_groups(self, base_lr: float):
        """
        Return AdamW parameter groups with separate LRs for DP and LSTM core.
        """
        lstm_lr = base_lr * self.lstm_lr_scale
        dp_params = [p for p in self._dp_parameters() if p.requires_grad]
        lstm_params = [p for p in self._lstm_core_parameters()
                       if p.requires_grad]
        groups = []
        if dp_params:
            groups.append({"params": dp_params, "lr": base_lr})
        if lstm_params:
            groups.append({"params": lstm_params, "lr": lstm_lr})
        return groups

    # ------------------------------------------------------------------
    # Normalizer helpers
    # ------------------------------------------------------------------

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        # load_state_dict clones CPU tensors even when the policy is on GPU,
        # so move the normalizer back to the policy's device.
        try:
            device = next(self.parameters()).device
            self.normalizer.to(device)
        except StopIteration:
            pass

    def set_lstm_normalizer(self, normalizer: LinearNormalizer):
        """Set the normalizer used by the LSTM during rollout."""
        self._lstm_normalizer = normalizer
        # LinearNormalizer params live on CPU by default; move them to the
        # policy device so rollout normalize() calls don't pull GPU tensors
        # back to CPU via x.to(device=scale.device).
        try:
            device = next(self.parameters()).device
            self._lstm_normalizer.to(device)
        except StopIteration:
            pass

    # ------------------------------------------------------------------
    # Reset (called at episode start)
    # ------------------------------------------------------------------

    def reset(self):
        self._lstm_hidden = None
        self._prev_action = None
        self._intermediates_updated = False
        self._last_decision_obs = None

    # ------------------------------------------------------------------
    # Diffusion sampling (identical to frozen variant)
    # ------------------------------------------------------------------

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(
                trajectory, t, local_cond=local_cond, global_cond=global_cond)
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator, **kwargs).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    # ------------------------------------------------------------------
    # LSTM hidden-state extraction — TRAINING (with gradients)
    # ------------------------------------------------------------------

    def _extract_lstm_hidden_train(
        self,
        episode_obs_features: torch.Tensor,   # (B, T_max, obs_feature_dim)
        episode_action_norm: torch.Tensor,     # (B, T_max, action_dim) LSTM-normalised
        episode_len: torch.Tensor,             # (B,)
        chunk_start_idx: torch.Tensor,         # (B,)
    ) -> torch.Tensor:
        """
        Run the LSTM core live on pre-encoded episode features (with gradients).

        The LSTM's visual encoder is NOT called here (features are pre-computed
        by the dataset using the frozen encoder).  Gradients flow through the
        LSTM recurrent weights.

        Args:
            episode_obs_features: pre-encoded observations (already LSTM-normalised,
                                  since agent_pos was baked in by the encoder).
            episode_action_norm:  LSTM-normalised actions.
            episode_len:          actual episode lengths (for pack_padded_sequence).
            chunk_start_idx:      episode-relative start of the DP chunk.

        Returns:
            hidden: (B, lstm_hidden_size) — hidden state at chunk_start_idx.
        """
        assert self.lstm_model is not None
        B = episode_obs_features.shape[0]
        device = episode_obs_features.device

        # Run LSTM core directly (skip the frozen encoder)
        lstm_output, _ = self.lstm_model.lstm.extract_hidden_states(
            obs=episode_obs_features,
            actions=episode_action_norm,
            lengths=episode_len,
            hidden_state=None,
        )  # (B, T_max, hidden_size)

        # Gather hidden state at chunk_start_idx for each batch element
        max_idx = (episode_len - 1).unsqueeze(1)                   # (B, 1)
        idx = chunk_start_idx.unsqueeze(1).clamp(max=max_idx)      # (B, 1)
        idx_exp = idx.unsqueeze(2).expand(B, 1, lstm_output.shape[2])
        hidden = torch.gather(lstm_output, 1, idx_exp).squeeze(1)  # (B, hidden_size)

        return hidden

    # ------------------------------------------------------------------
    # LSTM hidden-state extraction — ROLLOUT (identical to frozen)
    # ------------------------------------------------------------------

    def _extract_lstm_hidden_rollout(
        self,
        raw_obs_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Advance the LSTM and return the updated hidden state (rollout).

        ``raw_obs_dict`` must contain 'image' (B, 3, H, W) float [0,1]
        and 'agent_pos' (B, 2) raw (un-normalised).

        The LSTM is stepped once per DP decision.  On the first call,
        a zero input is fed to initialise the hidden state.
        """
        assert self.lstm_model is not None
        assert self._lstm_normalizer is not None, \
            "Call set_lstm_normalizer() before rollout."

        inner_lstm = self.lstm_model.lstm
        device = raw_obs_dict["image"].device
        B = raw_obs_dict["image"].shape[0]

        # Move LSTM to device if needed
        lstm_dev = next(self.lstm_model.parameters()).device
        if lstm_dev != device:
            self.lstm_model = self.lstm_model.to(device)
        self.lstm_model.eval()

        if self._lstm_hidden is None:
            # First call: feed zero input (matches pretraining convention)
            input_dim = inner_lstm.input_dim
            lstm_input = torch.zeros(B, 1, input_dim, device=device)
            with torch.no_grad():
                lstm_out, (h_n, c_n) = inner_lstm.lstm(
                    lstm_input, None)
                self._lstm_hidden = (h_n, c_n)
                hidden = lstm_out.squeeze(1)

        elif self._feed_intermediate_obs and self._intermediates_updated:
            # Intermediates were already applied via update_lstm_with_intermediates()
            hidden = self._lstm_hidden[0][-1]  # (B, hidden_size)
            self._intermediates_updated = False

        else:
            # Standard: step LSTM once with current obs + last action
            cur_image = raw_obs_dict["image"].unsqueeze(1)       # (B,1,3,H,W)
            cur_pos = raw_obs_dict["agent_pos"].unsqueeze(1)     # (B,1,2)

            # Normalise agent_pos for the LSTM encoder
            pos_norm = self._lstm_normalizer["agent_pos"].normalize(cur_pos)

            with torch.no_grad():
                obs_feat = self.lstm_model.encode_obs(
                    cur_image, pos_norm).squeeze(1)              # (B, obs_feat_dim)

            prev_act_norm = self._lstm_normalizer["action"].normalize(
                self._prev_action.unsqueeze(1)).squeeze(1)       # (B, action_dim)

            lstm_input = torch.cat(
                [obs_feat, prev_act_norm], dim=-1).unsqueeze(1)  # (B, 1, input_dim)

            with torch.no_grad():
                lstm_out, (h_n, c_n) = inner_lstm.lstm(
                    lstm_input, self._lstm_hidden)
                self._lstm_hidden = (h_n, c_n)
                hidden = lstm_out.squeeze(1)

        return hidden   # (B, hidden_size)

    # ------------------------------------------------------------------
    # Intermediate obs update (rollout, called by runner)
    # ------------------------------------------------------------------

    def update_lstm_with_intermediates(
        self,
        intermediate_obs_dict: Dict[str, torch.Tensor],
        executed_actions: torch.Tensor,
    ) -> None:
        """
        Step the LSTM through all intermediate observations produced
        during execution of the previous action chunk.

        Args:
            intermediate_obs_dict: 'image' (B, n_action_steps, 3, H, W) [0,1],
                                   'agent_pos' (B, n_action_steps, 2) raw
            executed_actions:      (B, n_action_steps, action_dim) raw
        """
        assert self.lstm_model is not None
        assert self._lstm_normalizer is not None
        assert self._last_decision_obs is not None, \
            "Call predict_action() before update_lstm_with_intermediates()"

        inner_lstm = self.lstm_model.lstm
        device = intermediate_obs_dict["image"].device
        n_steps = executed_actions.shape[1]

        if next(self.lstm_model.parameters()).device != device:
            self.lstm_model = self.lstm_model.to(device)
        self.lstm_model.eval()

        with torch.no_grad():
            for i in range(n_steps):
                if i == 0:
                    obs_img = self._last_decision_obs["image"].unsqueeze(1)
                    obs_ap = self._last_decision_obs["agent_pos"].unsqueeze(1)
                else:
                    obs_img = intermediate_obs_dict["image"][:, i - 1: i]
                    obs_ap = intermediate_obs_dict["agent_pos"][:, i - 1: i]

                pos_norm = self._lstm_normalizer["agent_pos"].normalize(obs_ap)
                obs_feat = self.lstm_model.encode_obs(
                    obs_img, pos_norm).squeeze(1)

                act_norm = self._lstm_normalizer["action"].normalize(
                    executed_actions[:, i: i + 1]).squeeze(1)

                lstm_input = torch.cat(
                    [obs_feat, act_norm], dim=-1).unsqueeze(1)

                _, (h_n, c_n) = inner_lstm.lstm(lstm_input, self._lstm_hidden)
                self._lstm_hidden = (h_n, c_n)

        self._intermediates_updated = True

    # ------------------------------------------------------------------
    # Predict action (inference / rollout)
    # ------------------------------------------------------------------

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: raw observations from the env runner.
          'image':     (B, To, 3, H, W) float32 [0, 1]
          'agent_pos': (B, To, 2) float32 (raw coordinates)
        """
        assert "past_action" not in obs_dict

        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps
        device = self.device
        dtype = self.dtype

        # Reset LSTM if batch size changed
        if self._lstm_hidden is not None:
            if self._lstm_hidden[0].size(1) != B:
                self.reset()

        # DP obs encoding
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)           # (B*To, obs_feat_dim)
        nobs_features = nobs_features.reshape(B, To, -1)       # (B, To, obs_feat_dim)
        if self.obs_projection is not None:
            nobs_features = self.obs_projection(nobs_features)

        # LSTM hidden state using raw obs (before DP normalisation)
        raw_obs_latest = dict_apply(obs_dict, lambda x: x[:, -1, ...])
        lstm_hidden = self._extract_lstm_hidden_rollout(raw_obs_latest)

        # Save decision obs for update_lstm_with_intermediates()
        if self._feed_intermediate_obs:
            self._last_decision_obs = dict_apply(
                raw_obs_latest, lambda x: x.detach().clone())

        # VQ → attention gate → projection
        if self.hidden_vq is not None:
            lstm_hidden, _ = self.hidden_vq(lstm_hidden)
        if self.hidden_attention_gate is not None:
            if self.hidden_attention_gate.mode == "obs_conditioned":
                obs_for_gate = nobs_features[:, -1, :]
                lstm_hidden = self.hidden_attention_gate(lstm_hidden, obs_for_gate)
            else:
                lstm_hidden = self.hidden_attention_gate(lstm_hidden)
        if self.hidden_projection is not None:
            lstm_hidden = self.hidden_projection(lstm_hidden)

        # Expand hidden to obs window: (B, To, hidden_feature_dim)
        lstm_features = lstm_hidden.unsqueeze(1).expand(B, To, -1)

        # Concat obs + hidden → global_cond
        global_cond = torch.cat(
            [nobs_features, lstm_features], dim=-1)  # (B, To, obs+hidden)
        global_cond = global_cond.reshape(B, -1)

        cond_data = torch.zeros((B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            cond_data, cond_mask, global_cond=global_cond, **self.kwargs)

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        self._prev_action = action[:, -1, :].detach()

        return {
            "action": action,
            "action_pred": action_pred,
        }

    # ------------------------------------------------------------------
    # Training — compute loss
    # ------------------------------------------------------------------

    def compute_loss(self, batch):
        """
        Expected batch keys (from ObsActionChunkLSTMImageFinetuneDataset):
            obs:                  dict {image: (B,H,3,96,96), agent_pos: (B,H,2)}
            action:               (B, H, action_dim)
            episode_obs_features: (B, T_max, obs_feature_dim)  pre-encoded
            episode_action_norm:  (B, T_max, action_dim)  LSTM-normalised
            chunk_start_idx:      (B,)
            episode_len:          (B,)

        Returns:
            * Scalar tensor when ``use_dct_loss=False`` — the workspace calls
              ``.backward()`` directly.
            * Dict when ``use_dct_loss=True`` (regardless of ``dct_loss_on_dp``):
                  {
                    'loss_main':       scalar — raw action loss (graph live).
                    'loss_dct':        scalar — α·dct_loss        (graph live).
                    'loss_raw_value':  detached scalar (raw loss, for log).
                    'loss_dct_value':  detached scalar (unscaled DCT, for log).
                    'vq_loss_value':   detached scalar (VQ loss, for log).
                  }
              The workspace decides how to combine/route the two losses based
              on the schedule (warm-up vs. post warm-up) and on
              ``dct_loss_on_dp``. See
              ``TrainDiffusionUnetHybridLSTMFinetuneWorkspace._compute_loss_and_backward``.
        """
        assert "valid_mask" not in batch

        device = next(self.parameters()).device

        # ---- Run LSTM core live (WITH gradients) -----------------------
        lstm_hidden = self._extract_lstm_hidden_train(
            episode_obs_features=batch["episode_obs_features"],
            episode_action_norm=batch["episode_action_norm"],
            episode_len=batch["episode_len"],
            chunk_start_idx=batch["chunk_start_idx"],
        )  # (B, hidden_size)

        # ---- DP forward ------------------------------------------------
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]

        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)            # (B*To, obs_feat)
        nobs_features = nobs_features.reshape(
            batch_size, self.n_obs_steps, -1)

        if self.obs_projection is not None:
            nobs_features = self.obs_projection(nobs_features)

        # Expand hidden: (B, To, hidden_size)
        lstm_feat = lstm_hidden.unsqueeze(1).expand(
            batch_size, self.n_obs_steps, -1)

        # VQ → attention gate → projection
        vq_loss = torch.tensor(0.0, device=device)
        if self.hidden_vq is not None:
            lstm_feat, vq_loss = self.hidden_vq(lstm_feat)
        if self.hidden_attention_gate is not None:
            if self.hidden_attention_gate.mode == "obs_conditioned":
                lstm_feat = self.hidden_attention_gate(lstm_feat, nobs_features)
            else:
                lstm_feat = self.hidden_attention_gate(lstm_feat)
        if self.hidden_projection is not None:
            lstm_feat = self.hidden_projection(lstm_feat)

        # Concat and flatten → global_cond
        global_cond = torch.cat(
            [nobs_features, lstm_feat], dim=-1)
        global_cond = global_cond.reshape(batch_size, -1)

        trajectory = nactions
        cond_data = trajectory
        condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()

        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred = self.model(
            noisy_trajectory, timesteps,
            local_cond=None, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type: {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss_raw = loss.mean()

        if self.hidden_vq is not None:
            loss_raw = loss_raw + self.vq_loss_weight * vq_loss

        # ================================================================
        # Optional DCT loss on the predicted action chunk
        # ================================================================
        if not self.use_dct_loss:
            return loss_raw

        loss_dct = self._compute_dct_loss(
            pred=pred,
            noisy_trajectory=noisy_trajectory,
            timesteps=timesteps,
            trajectory=trajectory,
            loss_mask=loss_mask,
            pred_type=pred_type,
        )

        # The workspace decides routing based on warm-up state and
        # dct_loss_on_dp; we always expose raw and α·dct separately.
        return {
            "loss_main": loss_raw,
            "loss_dct": self.dct_loss_weight * loss_dct,
            "loss_raw_value": loss_raw.detach(),
            "loss_dct_value": loss_dct.detach(),
            "vq_loss_value": vq_loss.detach()
                if torch.is_tensor(vq_loss) else torch.tensor(0.0, device=device),
        }

    # ------------------------------------------------------------------
    # DCT loss helper
    # ------------------------------------------------------------------

    def _compute_dct_loss(
        self,
        pred: torch.Tensor,                # UNet output (B, H, A) — eps or sample
        noisy_trajectory: torch.Tensor,    # (B, H, A) at timestep t
        timesteps: torch.Tensor,           # (B,) long
        trajectory: torch.Tensor,          # (B, H, A) ground-truth normalised
        loss_mask: torch.Tensor,           # (B, H, A) bool — True where loss applies
        pred_type: str,
    ) -> torch.Tensor:
        """
        MSE between truncated DCT coefficients of the predicted action chunk
        (recovered from epsilon prediction if necessary) and those of the
        ground-truth action chunk.

        Both signals live in DP-normalised action space. The diff is masked
        in the time domain (zeroing conditioned positions) before the DCT,
        which is consistent with the time-domain raw loss.

        Returns: scalar tensor.
        """
        # Recover predicted clean action chunk x_0_pred in normalised space
        if pred_type == "epsilon":
            alphas = self.noise_scheduler.alphas_cumprod.to(pred.device)
            alpha_prod_t = alphas[timesteps].view(-1, 1, 1)            # (B,1,1)
            beta_prod_t = 1.0 - alpha_prod_t
            x_0_pred = (
                noisy_trajectory - beta_prod_t.sqrt() * pred
            ) / alpha_prod_t.sqrt()
        elif pred_type == "sample":
            x_0_pred = pred
        else:
            raise ValueError(
                f"DCT loss does not support prediction_type={pred_type!r}")

        # Mask conditioned positions (their loss should not contribute).
        diff = (x_0_pred - trajectory) * loss_mask.type(pred.dtype)

        # Project the residual onto the DCT basis and MSE on coefficients.
        # diff: (B, H, A); basis: (H, K)  →  coeffs: (B, K, A)
        coeffs = dct_forward(diff, self._dct_basis)
        loss_dct = (coeffs ** 2).mean()
        return loss_dct
