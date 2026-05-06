"""
Diffusion Policy (vision-based) conditioned on observation + hidden states
from a pretrained ObsActionChunkLSTMImage.

During training:
  - LSTM hidden states come pre-computed from the dataset ('lstm_hidden' key).
  - The policy concatenates (image obs features, projected LSTM hidden)
    as the global conditioning signal.

During rollout (predict_action):
  - The frozen vision LSTM is stateful: hidden state accumulates across calls.
  - At the first step, raw zeros are fed (matching the LSTM training convention
    where forward() prepends zeros BEFORE normalisation).
  - On subsequent calls the LSTM is advanced ONCE per DP decision using
    the observation (image + agent_pos) at the decision point and the
    LAST action from the previous chunk.  This matches the LSTM pretraining
    resolution (action_step_subsample=4 → one step per DP decision).
  - The env runner does NOT need to call update_lstm_intermediate_obs().

Key difference from state-based DiffusionUnetLowdimPolicyWithObsActionChunkLSTM:
  - obs_encoder is the robomimic CNN (same as vanilla DiffusionUnetHybridImagePolicy).
  - The LSTM is ObsActionChunkLSTMImage (wraps its own obs_encoder + LSTM).
  - LSTM is stepped once per DP decision (not n_action_steps times).
  - global_cond = [obs_features || hidden_features] (image features + projected LSTM hidden).
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
from memory_diffusion_policy.model.obs_action_chunk_lstm import dct_inverse
from robomimic.algo import algo_factory
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import memory_diffusion_policy.model.vision.crop_randomizer as dmvc


def _load_torch(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class VectorQuantizer(nn.Module):
    """Straight-through VQ for LSTM hidden-state discretisation."""

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 commitment_weight: float = 0.25) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight,
                         -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


class HiddenStateAttentionGate(nn.Module):
    """Learnable channel-wise attention gate for LSTM hidden states.

    Learns *which dimensions* of the hidden state are useful, acting as a
    soft, per-feature selector (values in [0, 1]).

    Two modes
    ---------
    ``'self'`` (default)
        Gate depends only on the hidden state itself (squeeze-excitation style):

            gate = sigmoid( MLP(h) )
            out  = h * gate

    ``'obs_conditioned'``
        Gate is additionally conditioned on the current observation features,
        so the policy can select different memory dimensions depending on what
        it currently sees:

            gate = sigmoid( MLP( concat(h, obs) ) )
            out  = h * gate

    In both modes the output shape matches the input shape exactly, so the
    gate is a transparent drop-in between VQ and the projection MLP.

    Args:
        hidden_dim:  dimensionality of the LSTM hidden state (or VQ output).
        obs_dim:     dimensionality of the (projected) obs features – only
                     used when ``mode='obs_conditioned'``.
        reduction:   bottleneck factor for the internal MLP (hidden_dim // reduction).
        mode:        ``'self'`` or ``'obs_conditioned'``.
    """

    def __init__(
        self,
        hidden_dim: int,
        obs_dim: int = 0,
        reduction: int = 4,
        mode: str = 'self',
    ) -> None:
        super().__init__()
        assert mode in ('self', 'obs_conditioned'), \
            f"attention_gate_mode must be 'self' or 'obs_conditioned', got {mode!r}"
        if mode == 'obs_conditioned' and obs_dim <= 0:
            raise ValueError(
                "obs_dim must be > 0 when attention_gate_mode='obs_conditioned'")
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim

        in_dim = hidden_dim + obs_dim if mode == 'obs_conditioned' else hidden_dim
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
        """Apply attention gating.

        Args:
            h:   ``(B, hidden_dim)`` or ``(B, T, hidden_dim)``
            obs: ``(B, obs_dim)``   or ``(B, T, obs_dim)``
                 Required when ``mode='obs_conditioned'``.

        Returns:
            Gated hidden state, same shape as ``h``.
        """
        if self.mode == 'obs_conditioned':
            if obs is None:
                raise ValueError(
                    "obs must be provided for obs_conditioned attention gate")
            gate_input = torch.cat([h, obs], dim=-1)
        else:
            gate_input = h

        gate = self.gate_net(gate_input)   # (..., hidden_dim), values in (0, 1)
        return h * gate


class DiffusionUnetHybridImagePolicyWithObsActionChunkLSTM(BaseImagePolicy):
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
        lstm_hidden_size: int = 256,
        # Hidden projection
        use_hidden_projection: bool = True,
        projection_output_dim: Optional[int] = None,
        projection_num_layers: int = 2,
        # VQ
        use_vq: bool = False,
        vq_num_embeddings: int = 128,
        vq_commitment_weight: float = 0.25,
        vq_loss_weight: float = 1.0,
        # LSTM latent type  ('hidden' | 'past_dct_coeff' | 'future_action')
        lstm_latent_type: str = 'hidden',
        # Intermediate obs mode: step LSTM once per raw action instead of once per DP decision
        feed_intermediate_obs_to_lstm: bool = False,
        # ---- Attention gate on LSTM hidden state ----
        use_attention_gate: bool = False,
        attention_gate_reduction: int = 4,
        attention_gate_mode: str = 'self',  # 'self' | 'obs_conditioned'
        # ---- Obs feature projection ----
        use_obs_projection: bool = False,
        obs_projection_output_dim: Optional[int] = None,
        obs_projection_num_layers: int = 2,
        # ---- scheduler.step kwargs ----
        **kwargs,
    ):
        super().__init__()
        assert obs_as_global_cond, \
            "Only obs_as_global_cond=True is supported for image policies with LSTM."

        # ================================================================
        # 1.  Build the DP's own obs_encoder (same as vanilla hybrid policy)
        # ================================================================
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']

        obs_config = {'low_dim': [], 'rgb': [], 'depth': [], 'scan': []}
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            type_ = attr.get('type', 'low_dim')
            if type_ == 'rgb':
                obs_config['rgb'].append(key)
            elif type_ == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type_}")

        config = get_robomimic_config(
            algo_name='bc_rnn', hdf5_type='image',
            task_name='square', dataset_type='ph')

        with config.unlocked():
            config.observation.modalities.obs = obs_config
            config.observation.encoder.rgb.core_kwargs.pool_kwargs.num_kp = num_kp
            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        ObsUtils.initialize_obs_utils_with_config(config)

        policy = algo_factory(
            algo_name=config.algo_name, config=config,
            obs_key_shapes=obs_key_shapes, ac_dim=action_dim, device='cpu')
        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']

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

        obs_feature_dim = obs_encoder.output_shape()[0]  # e.g. 66

        # Obs feature projection (projects obs_encoder output to a desired dim)
        self.obs_projection: Optional[nn.Sequential] = None
        if use_obs_projection and obs_projection_output_dim is not None:
            layers = []
            in_dim = obs_feature_dim
            for i in range(obs_projection_num_layers - 1):
                mid_dim = max(obs_projection_output_dim,
                              (in_dim + obs_projection_output_dim) // 2)
                layers.extend([nn.Linear(in_dim, mid_dim), nn.ReLU()])
                in_dim = mid_dim
            layers.append(nn.Linear(in_dim, obs_projection_output_dim))
            self.obs_projection = nn.Sequential(*layers)
            projected_obs_feature_dim = obs_projection_output_dim
            print(f"Obs projection: {obs_feature_dim} → {obs_projection_output_dim} "
                  f"({obs_projection_num_layers} layer(s))")
        else:
            projected_obs_feature_dim = obs_feature_dim

        # ================================================================
        # 2.  LSTM conditioning dimensions
        # ================================================================
        self._lstm_latent_type = lstm_latent_type
        self.lstm_hidden_size = lstm_hidden_size
        self.use_hidden_projection = use_hidden_projection
        self.projection_output_dim = projection_output_dim
        self.use_vq = use_vq
        self.vq_loss_weight = vq_loss_weight

        # Load frozen LSTM (sets self.lstm_hidden_size correctly)
        self.lstm_model: Optional[ObsActionChunkLSTMImage] = None
        if lstm_checkpoint_path is not None:
            self._load_lstm_checkpoint(lstm_checkpoint_path, shape_meta)

        # Hidden projection
        self.hidden_projection: Optional[nn.Sequential] = None
        if use_hidden_projection and projection_output_dim is not None:
            layers = []
            in_dim = self.lstm_hidden_size
            for i in range(projection_num_layers - 1):
                mid_dim = max(projection_output_dim,
                              (in_dim + projection_output_dim) // 2)
                layers.extend([nn.Linear(in_dim, mid_dim), nn.ReLU()])
                in_dim = mid_dim
            layers.append(nn.Linear(in_dim, projection_output_dim))
            self.hidden_projection = nn.Sequential(*layers)
            print(f"Hidden projection: {self.lstm_hidden_size} → {projection_output_dim} "
                  f"({projection_num_layers} layer(s))")

        # VQ
        self.hidden_vq: Optional[VectorQuantizer] = None
        if use_vq:
            vq_dim = self.lstm_hidden_size
            self.hidden_vq = VectorQuantizer(
                num_embeddings=vq_num_embeddings,
                embedding_dim=vq_dim,
                commitment_weight=vq_commitment_weight)
            print(f"VQ: {vq_num_embeddings} codes, dim={vq_dim}, "
                  f"commitment={vq_commitment_weight}")

        # Attention gate (applied after VQ, before projection MLP)
        # For obs_conditioned mode the gate needs the projected obs feature dim,
        # which is known here as projected_obs_feature_dim.
        self.hidden_attention_gate: Optional[HiddenStateAttentionGate] = None
        if use_attention_gate:
            gate_obs_dim = (
                projected_obs_feature_dim
                if attention_gate_mode == 'obs_conditioned'
                else 0
            )
            self.hidden_attention_gate = HiddenStateAttentionGate(
                hidden_dim=self.lstm_hidden_size,
                obs_dim=gate_obs_dim,
                reduction=attention_gate_reduction,
                mode=attention_gate_mode,
            )
            print(f"Attention gate: mode={attention_gate_mode!r}, "
                  f"hidden_dim={self.lstm_hidden_size}, "
                  f"obs_dim={gate_obs_dim}, "
                  f"reduction={attention_gate_reduction}")

        # Compute the hidden feature dim after optional VQ + projection
        if use_hidden_projection and projection_output_dim is not None:
            hidden_feature_dim = projection_output_dim
        else:
            hidden_feature_dim = self.lstm_hidden_size

        # ================================================================
        # 3.  Diffusion U-Net
        # ================================================================
        # global_cond_dim = projected_obs_feature_dim * n_obs_steps  +  hidden_feature_dim * n_obs_steps
        global_cond_dim = (projected_obs_feature_dim + hidden_feature_dim) * n_obs_steps

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
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # Stateful rollout buffers
        self._lstm_hidden = None         # (h_n, c_n)
        self._prev_action = None         # (B, action_dim)  last executed action
        self._lstm_normalizer: Optional[LinearNormalizer] = None
        self._feed_intermediate_obs = feed_intermediate_obs_to_lstm
        self._intermediates_updated = False  # True after update_lstm_with_intermediates()
        self._last_decision_obs = None       # saved raw obs from last predict_action()

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision encoder params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
        if self.obs_projection is not None:
            print("Obs projection params: %e" % sum(
                p.numel() for p in self.obs_projection.parameters()))
        if self.hidden_projection is not None:
            print("Hidden projection params: %e" % sum(
                p.numel() for p in self.hidden_projection.parameters()))
        if self.hidden_attention_gate is not None:
            print("Attention gate params: %e" % sum(
                p.numel() for p in self.hidden_attention_gate.parameters()))

    # ------------------------------------------------------------------
    # LSTM checkpoint loading
    # ------------------------------------------------------------------

    def _load_lstm_checkpoint(self, checkpoint_path: str, shape_meta: dict):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"LSTM checkpoint not found: {checkpoint_path}")

        ckpt = _load_torch(checkpoint_path)
        ckpt_args = ckpt.get("args", {})

        # Re-create the ObsActionChunkLSTMImage with the saved config
        self.lstm_model = ObsActionChunkLSTMImage(
            shape_meta=shape_meta,
            action_dim=int(ckpt_args.get("action_dim", 2)),
            hidden_size=int(ckpt_args.get("hidden_size", 256)),
            num_layers=int(ckpt_args.get("num_layers", 2)),
            dropout=float(ckpt_args.get("dropout", 0.1)),
            chunk_H=int(ckpt_args.get("chunk_H", 16)),
            use_obs_head=bool(ckpt_args.get("use_obs_head", False)),
            use_future_head=bool(ckpt_args.get("use_future_head", True)),
            num_future_modes=int(ckpt_args.get("num_future_modes", 1)),
            future_abstraction=str(ckpt_args.get("future_abstraction", "raw")),
            future_n_bases=int(ckpt_args.get("future_n_bases", 32)),
            use_past_head=bool(ckpt_args.get("use_past_head", True)),
            past_chunk_H=int(ckpt_args.get("past_chunk_H", 75)),
            past_abstraction=str(ckpt_args.get("past_abstraction", "dct")),
            past_n_bases=int(ckpt_args.get("past_n_bases", 32)),
            use_vq=bool(ckpt_args.get("use_vq", False)),
            vq_n_codes=int(ckpt_args.get("vq_n_codes", 512)),
            vq_commitment_weight=float(ckpt_args.get("vq_commitment_weight", 0.25)),
            crop_shape=tuple(ckpt_args.get("crop_shape", (76, 76))),
            obs_encoder_group_norm=bool(ckpt_args.get("obs_encoder_group_norm", True)),
            eval_fixed_crop=True,  # always use fixed crop for frozen encoder at eval
            freeze_encoder=True,   # always frozen during DP training
            num_kp=int(ckpt_args.get("num_kp", 32)),
        )
        self.lstm_model.load_state_dict(ckpt["model_state"])
        self.lstm_model.eval()
        for p in self.lstm_model.parameters():
            p.requires_grad_(False)

        # Determine latent dim based on latent type
        import warnings
        inner_lstm = self.lstm_model.lstm
        if self._lstm_latent_type == 'past_dct_coeff':
            assert inner_lstm.past_chunk_head is not None
            lstm_latent_dim = inner_lstm.past_n_bases * inner_lstm.action_dim
            print(f"  Using past DCT coefficients as latent: dim={lstm_latent_dim}")
        elif self._lstm_latent_type == 'future_action':
            assert inner_lstm.future_chunk_head is not None
            if inner_lstm.future_abstraction == 'dct':
                lstm_latent_dim = inner_lstm.num_future_modes * inner_lstm.future_n_bases * inner_lstm.action_dim
            else:
                lstm_latent_dim = inner_lstm.num_future_modes * inner_lstm.chunk_H * inner_lstm.action_dim
            print(f"  Using future action prediction as latent: dim={lstm_latent_dim}")
        else:
            lstm_latent_dim = inner_lstm.hidden_size

        if lstm_latent_dim != self.lstm_hidden_size:
            warnings.warn(
                f"Config lstm_hidden_size={self.lstm_hidden_size} does not match "
                f"actual LSTM latent dim={lstm_latent_dim}. Overwriting.")
        self.lstm_hidden_size = lstm_latent_dim

        print(f"Loaded pretrained ObsActionChunkLSTMImage from: {checkpoint_path}")
        print(f"  hidden_size={inner_lstm.hidden_size}, "
              f"lstm_latent_type={self._lstm_latent_type}, "
              f"lstm_latent_dim={self.lstm_hidden_size}")

        # Save the LSTM normalizer for rollout (will be set via set_lstm_normalizer)
        self._lstm_normalizer: Optional[LinearNormalizer] = None

    # ------------------------------------------------------------------
    # Reset (called at episode start by the env runner)
    # ------------------------------------------------------------------

    def reset(self):
        self._lstm_hidden = None
        self._prev_action = None
        self._intermediates_updated = False
        self._last_decision_obs = None

    # ------------------------------------------------------------------
    # Diffusion sampling
    # ------------------------------------------------------------------

    def conditional_sample(self, condition_data, condition_mask,
                           local_cond=None, global_cond=None,
                           generator=None, **kwargs):
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
            model_output = model(trajectory, t,
                                 local_cond=local_cond, global_cond=global_cond)
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator, **kwargs).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    # ------------------------------------------------------------------
    # LSTM hidden-state extraction (rollout)
    # ------------------------------------------------------------------

    def _extract_lstm_hidden_rollout(
        self,
        raw_obs_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Return the LSTM latent for the current DP decision.

        Two modes controlled by ``self._feed_intermediate_obs``:

        **Standard mode** (feed_intermediate_obs=False)
            LSTM is stepped ONCE per DP decision using the decision-point
            observation and the last executed action (matches
            action_step_subsample=n_action_steps pretraining).

        **Intermediate mode** (feed_intermediate_obs=True)
            LSTM was already stepped n_action_steps times in
            ``update_lstm_with_intermediates()`` right after the
            previous env.step().  Here we simply read the accumulated
            hidden state (matches action_step_subsample=1 pretraining).

        At the very first DP decision zeros are fed (both modes).

        Args:
            raw_obs_dict: dict with 'image' (B, 3, H, W) float [0,1] and
                          'agent_pos' (B, 2) float RAW (not DP-normalised).

        Returns:
            hidden: (B, lstm_hidden_size) latent feature.
        """
        assert self.lstm_model is not None
        inner_lstm = self.lstm_model.lstm

        device = raw_obs_dict['image'].device
        B = raw_obs_dict['image'].shape[0]

        # Move LSTM to device if needed
        if next(self.lstm_model.parameters()).device != device:
            self.lstm_model = self.lstm_model.to(device)
        self.lstm_model.eval()

        if self._lstm_hidden is None:
            # ---- First call: feed raw zeros (both modes) ----
            input_dim = inner_lstm.input_dim  # obs_feature_dim + action_dim
            lstm_input = torch.zeros(B, 1, input_dim, device=device)
            with torch.no_grad():
                lstm_out, (h_n, c_n) = inner_lstm.lstm(
                    lstm_input, self._lstm_hidden)
                self._lstm_hidden = (h_n, c_n)
                hidden = lstm_out.squeeze(1)  # (B, hidden_size)

        elif self._feed_intermediate_obs:
            # ---- Intermediate mode: hidden already updated ----
            if not self._intermediates_updated:
                import warnings
                warnings.warn(
                    "feed_intermediate_obs=True but "
                    "update_lstm_with_intermediates() was not called "
                    "since the last predict_action.  Using stale hidden.")
            hidden = self._lstm_hidden[0][-1]  # last layer h_n: (B, hidden_size)
            self._intermediates_updated = False

        else:
            # ---- Standard mode: step LSTM once with (obs, prev_action) ----
            assert self._lstm_normalizer is not None, \
                "LSTM normalizer not set. Call set_lstm_normalizer() first."

            cur_image = raw_obs_dict['image'].unsqueeze(1)
            cur_agent_pos = raw_obs_dict['agent_pos'].unsqueeze(1)
            cur_agent_pos_norm = self._lstm_normalizer['agent_pos'].normalize(
                cur_agent_pos)

            with torch.no_grad():
                obs_features = self.lstm_model.encode_obs(
                    cur_image, cur_agent_pos_norm)
                obs_feat = obs_features.squeeze(1)

            prev_action_norm = self._lstm_normalizer['action'].normalize(
                self._prev_action.unsqueeze(1)).squeeze(1)

            lstm_input = torch.cat(
                [obs_feat, prev_action_norm], dim=-1).unsqueeze(1)

            with torch.no_grad():
                lstm_out, (h_n, c_n) = inner_lstm.lstm(
                    lstm_input, self._lstm_hidden)
                self._lstm_hidden = (h_n, c_n)
                hidden = lstm_out.squeeze(1)

        # Apply frozen head to get the latent
        if self._lstm_latent_type == 'past_dct_coeff':
            with torch.no_grad():
                hidden = inner_lstm.past_chunk_head(hidden)
        elif self._lstm_latent_type == 'future_action':
            with torch.no_grad():
                hidden = inner_lstm.future_chunk_head(hidden)

        return hidden

    # ------------------------------------------------------------------
    # Intermediate-obs LSTM update (called by the runner after env.step)
    # ------------------------------------------------------------------

    def update_lstm_with_intermediates(
        self,
        intermediate_obs_dict: Dict[str, torch.Tensor],
        executed_actions: torch.Tensor,
    ) -> None:
        """
        Step the frozen LSTM through ALL intermediate observations that
        the environment produced during execution of the action chunk.

        This gives the LSTM full-resolution information (one step per raw
        action) and matches ``action_step_subsample=1`` pretraining.

        The LSTM shift convention is  input_t = [obs_{t-1}, a_{t-1}].
        For n_action_steps=4 executed actions a₀…a₃ producing intermediate
        observations obs₁…obs₄ (obs after each action):

            LSTM step 0: input = [obs₀, a₀]  → h₁
            LSTM step 1: input = [obs₁, a₁]  → h₂
            LSTM step 2: input = [obs₂, a₂]  → h₃
            LSTM step 3: input = [obs₃, a₃]  → h₄

        where obs₀ = the decision-point observation saved from the
        preceding ``predict_action()`` call (``self._last_decision_obs``).

        Args:
            intermediate_obs_dict: dict with
                'image':     (B, n_action_steps, 3, H, W) float [0,1]
                'agent_pos': (B, n_action_steps, 2) float raw
            executed_actions: (B, n_action_steps, action_dim) float raw
        """
        assert self.lstm_model is not None
        assert self._lstm_normalizer is not None
        assert self._last_decision_obs is not None, \
            "predict_action() must be called before update_lstm_with_intermediates()"

        inner_lstm = self.lstm_model.lstm
        device = intermediate_obs_dict['image'].device
        n_steps = executed_actions.shape[1]

        # Ensure LSTM is on correct device
        if next(self.lstm_model.parameters()).device != device:
            self.lstm_model = self.lstm_model.to(device)
        self.lstm_model.eval()

        with torch.no_grad():
            for i in range(n_steps):
                # --- obs input: shifted by 1 (obs at t-1) ---
                if i == 0:
                    obs_img = self._last_decision_obs['image'].unsqueeze(1)
                    obs_ap = self._last_decision_obs['agent_pos'].unsqueeze(1)
                else:
                    obs_img = intermediate_obs_dict['image'][:, i - 1 : i]
                    obs_ap = intermediate_obs_dict['agent_pos'][:, i - 1 : i]

                obs_ap_norm = self._lstm_normalizer['agent_pos'].normalize(obs_ap)
                obs_feat = self.lstm_model.encode_obs(
                    obs_img, obs_ap_norm).squeeze(1)          # (B, obs_feat_dim)

                # --- action input ---
                act_norm = self._lstm_normalizer['action'].normalize(
                    executed_actions[:, i : i + 1]).squeeze(1)  # (B, action_dim)

                lstm_input = torch.cat(
                    [obs_feat, act_norm], dim=-1).unsqueeze(1)  # (B, 1, input_dim)

                _, (h_n, c_n) = inner_lstm.lstm(
                    lstm_input, self._lstm_hidden)
                self._lstm_hidden = (h_n, c_n)

        self._intermediates_updated = True

    # ------------------------------------------------------------------
    # Predict action (inference / rollout)
    # ------------------------------------------------------------------

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: raw observations from the env runner.
          'image':     (B, To, 3, H, W) float32 [0, 1]
          'agent_pos': (B, To, 2) float32 (raw coordinates)
        """
        assert 'past_action' not in obs_dict

        # Normalise input for DP
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

        # Encode image obs through the DP's encoder
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)  # (B*To, obs_feature_dim)
        nobs_features = nobs_features.reshape(B, To, -1)  # (B, To, obs_feature_dim)

        # Project obs features if enabled
        if self.obs_projection is not None:
            nobs_features = self.obs_projection(nobs_features)  # (B, To, projected_obs_feature_dim)

        # Extract LSTM hidden state using RAW observation (before DP normalisation)
        # The LSTM's encode_obs expects images in [0,1] and agent_pos un-normalised.
        raw_obs_latest = dict_apply(obs_dict, lambda x: x[:, -1, ...])  # (B, ...)
        lstm_hidden = self._extract_lstm_hidden_rollout(raw_obs_latest)  # (B, lstm_hidden_size)

        # Save decision obs for update_lstm_with_intermediates()
        if self._feed_intermediate_obs:
            self._last_decision_obs = dict_apply(
                raw_obs_latest, lambda x: x.detach().clone())

        # VQ → attention gate → projection
        if self.hidden_vq is not None:
            lstm_hidden, _ = self.hidden_vq(lstm_hidden)
        if self.hidden_attention_gate is not None:
            if self.hidden_attention_gate.mode == 'obs_conditioned':
                # Use the latest obs step as context for the gate
                obs_for_gate = nobs_features[:, -1, :]  # (B, proj_obs_feat_dim)
                lstm_hidden = self.hidden_attention_gate(lstm_hidden, obs_for_gate)
            else:
                lstm_hidden = self.hidden_attention_gate(lstm_hidden)
        if self.hidden_projection is not None:
            lstm_hidden = self.hidden_projection(lstm_hidden)

        # Expand hidden to obs window: (B, To, hidden_feature_dim)
        lstm_features = lstm_hidden.unsqueeze(1).expand(B, To, -1)

        # Concat obs + hidden → global_cond
        global_cond = torch.cat([nobs_features, lstm_features], dim=-1)  # (B, To, obs_feat + hidden_feat)
        global_cond = global_cond.reshape(B, -1)  # (B, To * (obs_feat + hidden_feat))

        # Build conditioning
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # Run diffusion sampling
        nsample = self.conditional_sample(
            cond_data, cond_mask,
            global_cond=global_cond,
            **self.kwargs)

        # Unnormalise action
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        # Buffer for next LSTM step:
        # Only need the last executed action; the observation will be taken
        # from the NEXT call's obs_dict (current decision point).
        self._prev_action = action[:, -1, :].detach()  # (B, Da)

        result = {
            'action': action,
            'action_pred': action_pred,
        }
        return result

    # ------------------------------------------------------------------
    # Past-action prediction (from frozen past_chunk_head)
    # ------------------------------------------------------------------

    def predict_past_actions(self) -> Optional[torch.Tensor]:
        """
        Use the frozen LSTM past_chunk_head to predict past actions from
        the current LSTM hidden state.

        Returns:
            past_pred: (B, past_chunk_H, action_dim) in raw (pixel) space,
                       ordered recent→older: [a_{t-1}, a_{t-2}, …, a_{t-pastH}].
                       Slot 0 is a_{t-1} — the action just seen as LSTM input.
                       None if past_chunk_head is unavailable or LSTM not yet run.
        """
        if (self.lstm_model is None
                or self._lstm_hidden is None):
            return None

        inner_lstm = self.lstm_model.lstm
        if inner_lstm.past_chunk_head is None:
            return None

        device = self._lstm_hidden[0].device
        h_last = self._lstm_hidden[0][-1]  # (B, hidden_size)

        with torch.no_grad():
            past_flat = inner_lstm.past_chunk_head(h_last)
            B = h_last.shape[0]
            if inner_lstm.past_abstraction in ('dct', 'segment_pool'):
                K = inner_lstm.past_n_bases
                D = inner_lstm.action_dim
                coeffs = past_flat.view(B, K, D)
                past_norm = dct_inverse(coeffs, inner_lstm._past_dct_basis)
            else:
                past_norm = past_flat.view(
                    B, inner_lstm.past_chunk_H, inner_lstm.action_dim)

            # Denormalise using the LSTM normalizer (past actions are in
            # LSTM-normalised space, not DP-normalised space)
            pH, D = inner_lstm.past_chunk_H, inner_lstm.action_dim
            flat = past_norm.reshape(B * pH, D).unsqueeze(1)
            flat_raw = self._lstm_normalizer['action'].unnormalize(flat)
            past_raw = flat_raw.squeeze(1).reshape(B, pH, D)

        return past_raw

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def set_lstm_normalizer(self, normalizer: LinearNormalizer):
        """Set the normalizer used by the frozen LSTM during rollout."""
        self._lstm_normalizer = normalizer

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch

        # Pre-computed LSTM hidden states from dataset
        features = batch['lstm_hidden']  # (B, T, lstm_hidden_size)

        # Normalise obs and action
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # Encode images through DP's encoder
        this_nobs = dict_apply(
            nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)  # (B*To, obs_feature_dim)
        nobs_features = nobs_features.reshape(batch_size, self.n_obs_steps, -1)

        # Project obs features if enabled
        if self.obs_projection is not None:
            nobs_features = self.obs_projection(nobs_features)  # (B, To, projected_obs_feature_dim)

        # Process LSTM features: only use n_obs_steps
        lstm_feat = features[:, :self.n_obs_steps, :]  # (B, To, lstm_hidden_size)

        # VQ → attention gate → projection hidden
        vq_loss = torch.tensor(0.0, device=features.device)
        if self.hidden_vq is not None:
            lstm_feat, vq_loss = self.hidden_vq(lstm_feat)
        if self.hidden_attention_gate is not None:
            if self.hidden_attention_gate.mode == 'obs_conditioned':
                # nobs_features: (B, To, proj_obs_feat_dim)
                lstm_feat = self.hidden_attention_gate(lstm_feat, nobs_features)
            else:
                lstm_feat = self.hidden_attention_gate(lstm_feat)
        if self.hidden_projection is not None:
            lstm_feat = self.hidden_projection(lstm_feat)

        # Concat and flatten → global_cond
        global_cond = torch.cat([nobs_features, lstm_feat], dim=-1)
        global_cond = global_cond.reshape(batch_size, -1)

        trajectory = nactions
        cond_data = trajectory

        # Generate mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Forward diffusion
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred = self.model(noisy_trajectory, timesteps,
                          local_cond=None, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        # Add VQ loss
        if self.hidden_vq is not None:
            loss = loss + self.vq_loss_weight * vq_loss

        return loss
