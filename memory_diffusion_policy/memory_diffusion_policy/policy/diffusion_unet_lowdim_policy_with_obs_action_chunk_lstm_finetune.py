"""
Diffusion Policy conditioned on observation + hidden states from a
**finetunable** pretrained ObsActionChunkLSTM.

Key difference from the frozen variant
(diffusion_unet_lowdim_policy_with_obs_action_chunk_lstm.py):
  - During training the LSTM is **not** frozen: gradients from the
    diffusion action loss flow back through the LSTM encoder so that
    it can adapt its hidden-state representations to better serve
    the downstream policy.
  - During training, the LSTM is run live on full episode data
    (provided by the dataset) to produce hidden states with gradients.
  - During rollout the LSTM is stateful (identical to the frozen
    variant): hidden state accumulates across calls.

Everything else (VQ, hidden/obs projections, diffusion sampling,
intermediate-obs multi-step rollout) behaves identically to the
frozen variant so the two can be compared fairly.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from pathlib import Path

from memory_diffusion_policy.model.common.normalizer import LinearNormalizer
from memory_diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from memory_diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from memory_diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from memory_diffusion_policy.model.obs_action_chunk_lstm import ObsActionChunkLSTM, dct_inverse


def _load_torch(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ======================================================================
# VectorQuantizer (identical to frozen variant)
# ======================================================================

class VectorQuantizer(nn.Module):
    """
    Straight-through Vector Quantizer for LSTM hidden-state discretization.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 commitment_weight: float = 0.25) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight,
                         -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(
        self, z: torch.Tensor,
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
# Main policy class
# ======================================================================

class DiffusionUnetLowdimPolicyWithObsActionChunkLSTMFinetune(BaseLowdimPolicy):
    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_scheduler: DDPMScheduler,
        horizon,
        obs_dim,
        action_dim,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_local_cond=False,
        obs_as_global_cond=False,
        pred_action_steps_only=False,
        oa_step_convention=False,
        # ObsActionChunkLSTM parameters
        lstm_checkpoint_path: Optional[str] = None,
        lstm_hidden_size: int = 512,
        # Hidden-state projection
        use_hidden_projection: bool = True,
        projection_output_dim: Optional[int] = None,
        projection_num_layers: int = 2,
        projection_concat: bool = False,
        raw_obs_dim: int = 20,
        # Observation projection
        use_obs_projection: bool = False,
        obs_projection_output_dim: int = 20,
        obs_projection_num_layers: int = 2,
        # Vector quantization
        use_vq: bool = False,
        vq_num_embeddings: int = 128,
        vq_commitment_weight: float = 0.25,
        vq_loss_weight: float = 1.0,
        # Finetuning options
        lstm_lr_scale: float = 1.0,
        freeze_lstm_epochs: int = 0,
        # Path to LSTM normalizer (defaults to <checkpoint_dir>/normalizer.pt)
        lstm_normalizer_path: Optional[str] = None,
        # parameters passed to scheduler.step
        **kwargs,
    ):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.lstm_hidden_size = lstm_hidden_size
        self.use_hidden_projection = use_hidden_projection
        self.projection_output_dim = projection_output_dim
        self.projection_concat = projection_concat
        self.raw_obs_dim = raw_obs_dim
        self.use_obs_projection = use_obs_projection
        self.obs_projection_output_dim = obs_projection_output_dim
        self.use_vq = use_vq
        self.vq_loss_weight = vq_loss_weight
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.lstm_lr_scale = lstm_lr_scale
        self.freeze_lstm_epochs = freeze_lstm_epochs
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # ---- Pretrained LSTM (TRAINABLE) --------------------------------
        self.lstm_model: Optional[ObsActionChunkLSTM] = None
        self._lstm_obs_dim: int = raw_obs_dim
        self._lstm_keypoint_only: bool = False
        self._lstm_keypoint_dim: int = 18
        # Separate normalizer for LSTM episode data (may differ from DP normalizer)
        self._lstm_normalizer: Optional[LinearNormalizer] = None

        # Stateful rollout buffers (identical to frozen variant)
        self._lstm_hidden = None
        self._prev_action_chunk = None
        self._prev_intermediate_obs = None
        self._prev_newest_obs = None

        if lstm_checkpoint_path is not None:
            self._load_lstm_checkpoint(lstm_checkpoint_path, lstm_normalizer_path)

        # ---- Learnable hidden-state projection --------------------------
        self.hidden_projection: Optional[nn.Sequential] = None
        if use_hidden_projection and projection_output_dim is not None:
            if projection_concat:
                proj_in_dim = raw_obs_dim + self.lstm_hidden_size
            else:
                proj_in_dim = self.lstm_hidden_size
            layers = []
            in_dim = proj_in_dim
            for i in range(projection_num_layers - 1):
                mid_dim = max(projection_output_dim,
                              (in_dim + projection_output_dim) // 2)
                layers.extend([nn.Linear(in_dim, mid_dim), nn.ReLU()])
                in_dim = mid_dim
            layers.append(nn.Linear(in_dim, projection_output_dim))
            self.hidden_projection = nn.Sequential(*layers)
            mode = "concat-then-project" if projection_concat else "project-hidden"
            print(
                f"Hidden projection ({mode}): "
                f"{proj_in_dim} → {projection_output_dim} "
                f"({projection_num_layers} layer(s))"
            )
        elif not use_hidden_projection:
            print(f"Hidden projection disabled — using raw hidden "
                  f"({self.lstm_hidden_size}-dim)")

        # ---- Learnable observation projection ---------------------------
        self.obs_projection: Optional[nn.Sequential] = None
        if use_obs_projection:
            layers = []
            in_dim = raw_obs_dim
            for i in range(obs_projection_num_layers - 1):
                mid_dim = max(obs_projection_output_dim,
                              (in_dim + obs_projection_output_dim) // 2)
                layers.extend([nn.Linear(in_dim, mid_dim), nn.ReLU()])
                in_dim = mid_dim
            layers.append(nn.Linear(in_dim, obs_projection_output_dim))
            self.obs_projection = nn.Sequential(*layers)
            print(
                f"Obs projection: {raw_obs_dim} → {obs_projection_output_dim} "
                f"({obs_projection_num_layers} layer(s))"
            )

        # ---- Vector Quantizer (optional) --------------------------------
        self.hidden_vq: Optional[VectorQuantizer] = None
        if use_vq:
            vq_dim = self.lstm_hidden_size
            self.hidden_vq = VectorQuantizer(
                num_embeddings=vq_num_embeddings,
                embedding_dim=vq_dim,
                commitment_weight=vq_commitment_weight,
            )
            print(
                f"VQ: {vq_num_embeddings} codes, dim={vq_dim}, "
                f"commitment={vq_commitment_weight}, "
                f"loss_weight={vq_loss_weight}"
            )

    # ------------------------------------------------------------------
    # Parameter groups (for separate LSTM learning rate)
    # ------------------------------------------------------------------

    def get_parameter_groups(self, base_lr: float):
        """Return parameter groups with optional LSTM lr scaling."""
        lstm_params = []
        other_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("lstm_model."):
                lstm_params.append(param)
            else:
                other_params.append(param)
        groups = [{"params": other_params, "lr": base_lr}]
        if lstm_params:
            groups.append({
                "params": lstm_params,
                "lr": base_lr * self.lstm_lr_scale,
            })
        return groups

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers (used by workspace training schedule)
    # ------------------------------------------------------------------

    def freeze_lstm(self):
        """Freeze all LSTM parameters (no gradient updates)."""
        if self.lstm_model is not None:
            for p in self.lstm_model.parameters():
                p.requires_grad_(False)

    def unfreeze_lstm(self):
        """Unfreeze all LSTM parameters (allow gradient updates)."""
        if self.lstm_model is not None:
            for p in self.lstm_model.parameters():
                p.requires_grad_(True)

    def freeze_dp(self):
        """Freeze all non-LSTM parameters (UNet, projections, VQ)."""
        for name, p in self.named_parameters():
            if not name.startswith("lstm_model."):
                p.requires_grad_(False)

    def unfreeze_dp(self):
        """Unfreeze all non-LSTM parameters."""
        for name, p in self.named_parameters():
            if not name.startswith("lstm_model."):
                p.requires_grad_(True)

    # ------------------------------------------------------------------
    # LSTM checkpoint loading  (NOT frozen)
    # ------------------------------------------------------------------

    def _load_lstm_checkpoint(self, checkpoint_path: str,
                               lstm_normalizer_path: Optional[str] = None):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"LSTM checkpoint not found: {checkpoint_path}"
            )

        ckpt = _load_torch(checkpoint_path)
        ckpt_args = ckpt.get("args", {})

        keypoint_only = bool(ckpt_args.get("keypoint_only", False))
        keypoint_dim = int(ckpt_args.get("keypoint_dim", 18))
        ckpt_include_goal_kps = bool(
            ckpt_args.get("include_goal_keypoints", False))
        base_obs_dim = 74 if ckpt_include_goal_kps else 20
        lstm_obs_dim = keypoint_dim if keypoint_only else base_obs_dim

        self.lstm_model = ObsActionChunkLSTM(
            obs_dim=lstm_obs_dim,
            action_dim=2,
            hidden_size=int(ckpt_args.get("hidden_size", 256)),
            num_layers=int(ckpt_args.get("num_layers", 2)),
            dropout=float(ckpt_args.get("dropout", 0.1)),
            chunk_H=int(ckpt_args.get("chunk_H", 16)),
            use_obs_head=bool(ckpt_args.get("use_obs_head", True)),
            use_future_head=bool(ckpt_args.get("use_future_head", True)),
            use_past_head=bool(ckpt_args.get("use_past_head", False)),
            past_chunk_H=int(ckpt_args.get("past_chunk_H", 16)),
            past_abstraction=str(ckpt_args.get("past_abstraction", "raw")),
            past_n_bases=int(ckpt_args.get("past_n_bases", 32)),
            action_only=bool(ckpt_args.get("action_only", False)),
            use_vq=bool(ckpt_args.get("use_vq", False)),
            vq_n_codes=int(ckpt_args.get("vq_n_codes", 512)),
            vq_commitment_weight=float(
                ckpt_args.get("vq_commitment_weight", 0.25)),
        )
        self.lstm_model.load_state_dict(ckpt["model_state"])

        # *** KEY DIFFERENCE: LSTM is kept trainable ***
        # (in the frozen variant, all params are set to requires_grad_(False))
        self.lstm_model.train()

        self._lstm_obs_dim = lstm_obs_dim
        self._lstm_keypoint_only = keypoint_only
        self._lstm_keypoint_dim = keypoint_dim

        actual_hidden = self.lstm_model.hidden_size
        if actual_hidden != self.lstm_hidden_size:
            import warnings
            warnings.warn(
                f"Config lstm_hidden_size={self.lstm_hidden_size} does not "
                f"match actual LSTM checkpoint hidden_size={actual_hidden}. "
                f"Overwriting to {actual_hidden}. Please update your config!"
            )
        self.lstm_hidden_size = actual_hidden

        print(f"Loaded pretrained ObsActionChunkLSTM (TRAINABLE) from: "
              f"{checkpoint_path}")
        print(
            f"  hidden_size={self.lstm_model.hidden_size}, "
            f"lstm_obs_dim={lstm_obs_dim}, keypoint_only={keypoint_only}, "
            f"action_only={self.lstm_model.action_only}"
        )

        # ---- Load LSTM normalizer (same as frozen variant) --------------
        norm_path = (
            Path(lstm_normalizer_path) if lstm_normalizer_path
            else checkpoint_path.parent / "normalizer.pt"
        )
        if norm_path.exists():
            self._lstm_normalizer = _load_torch(norm_path)
            print(f"  Loaded LSTM normalizer from: {norm_path}")
        else:
            print(
                f"  WARNING: LSTM normalizer not found at {norm_path}. "
                f"Will fall back to DP normalizer for episode data."
            )

    # ------------------------------------------------------------------
    # Reset (called at episode start by the env runner)
    # ------------------------------------------------------------------

    def reset(self):
        """Reset LSTM hidden state and obs/action buffers."""
        self._lstm_hidden = None
        self._prev_action_chunk = None
        self._prev_intermediate_obs = None
        self._prev_newest_obs = None

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
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(
                trajectory, t,
                local_cond=local_cond, global_cond=global_cond,
            )
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator, **kwargs,
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    # ------------------------------------------------------------------
    # LSTM normalizer helper
    # ------------------------------------------------------------------

    def _normalize_for_lstm(
        self, x: torch.Tensor, key: str,
    ) -> torch.Tensor:
        """Normalise (B, T, D) tensor using the LSTM's own normalizer."""
        assert self._lstm_normalizer is not None
        B, T, D = x.shape
        xf = x.reshape(B * T, D).unsqueeze(1)   # (B*T, 1, D)
        norm = self._lstm_normalizer[key]
        norm.to(x.device)
        out = norm.normalize(xf)
        return out.squeeze(1).reshape(B, T, D)

    # ------------------------------------------------------------------
    # LSTM hidden-state extraction (TRAINING — live, with gradients)
    # ------------------------------------------------------------------

    def _extract_lstm_hidden_state_train(
        self,
        episode_obs: torch.Tensor,
        episode_action: torch.Tensor,
        episode_len: torch.Tensor,
        chunk_start_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the LSTM on full episode data and extract hidden states for
        the chunk positions.  Gradients flow through the LSTM.

        Args:
            episode_obs:      (B, T_max, obs_dim)  zero-padded episode obs
            episode_action:   (B, T_max, action_dim) zero-padded episode actions
            episode_len:      (B,) actual episode lengths
            chunk_start_idx:  (B,) chunk start positions in the episode

        Returns:
            chunk_hidden: (B, horizon, hidden_size) hidden states for chunk
        """
        assert self.lstm_model is not None
        B = episode_obs.shape[0]
        device = episode_obs.device
        H = self.horizon

        # Normalise episode obs/action for the LSTM using the LSTM's
        # own normalizer (which may differ from the DP normalizer when
        # the LSTM was trained with a different obs_dim, e.g. with goal
        # keypoints while the DP does not use them).
        if self._lstm_normalizer is not None:
            nobs = self._normalize_for_lstm(episode_obs, "obs")
            naction = self._normalize_for_lstm(episode_action, "action")
        else:
            nobs = self.normalizer["obs"].normalize(episode_obs)
            naction = self.normalizer["action"].normalize(episode_action)

        if self._lstm_keypoint_only:
            nobs = nobs[:, :, :self._lstm_keypoint_dim]

        # Run LSTM on full episodes (with pack_padded_sequence for
        # efficiency with variable-length episodes)
        lstm_output, _ = self.lstm_model.extract_hidden_states(
            nobs, naction,
            lengths=episode_len,
            hidden_state=None,
        )  # (B, T_max, hidden_size)

        # Extract chunk-aligned hidden states using gather
        # chunk_start_idx: (B,)
        # We need hidden[:, chunk_start:chunk_start+H, :]
        offsets = torch.arange(H, device=device).unsqueeze(0)  # (1, H)
        indices = chunk_start_idx.unsqueeze(1) + offsets       # (B, H)

        # Clamp to valid range (episode_len - 1)
        max_idx = (episode_len - 1).unsqueeze(1)               # (B, 1)
        indices = indices.clamp(max=max_idx)

        # Gather: expand indices for hidden_size dimension
        indices_expanded = indices.unsqueeze(2).expand(
            B, H, lstm_output.shape[2])                        # (B, H, hidden_size)
        chunk_hidden = torch.gather(
            lstm_output, 1, indices_expanded)                   # (B, H, hidden_size)

        return chunk_hidden

    # ------------------------------------------------------------------
    # LSTM hidden-state extraction (ROLLOUT — identical to frozen)
    # ------------------------------------------------------------------

    def _extract_lstm_hidden_state_rollout(
        self, current_obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Advance the LSTM and return the updated hidden state (rollout).
        Uses the LSTM's own normalizer for obs/action normalization.
        """
        assert self.lstm_model is not None
        B = current_obs.shape[0]
        device = current_obs.device
        input_dim = self.lstm_model.input_dim

        if next(self.lstm_model.parameters()).device != device:
            self.lstm_model = self.lstm_model.to(device)
        self.lstm_model.eval()

        # Choose normalizer for LSTM inputs
        if self._lstm_normalizer is not None:
            _norm_obs = lambda x: self._normalize_for_lstm(x, "obs")
            _norm_act = lambda x: self._normalize_for_lstm(x, "action")
        else:
            _norm_obs = lambda x: self.normalizer["obs"].normalize(x)
            _norm_act = lambda x: self.normalizer["action"].normalize(x)

        if self._prev_action_chunk is None:
            lstm_input = torch.zeros(B, 1, input_dim, device=device)
            with torch.no_grad():
                lstm_out, (h_n, c_n) = self.lstm_model.lstm(
                    lstm_input, self._lstm_hidden
                )
                self._lstm_hidden = (h_n, c_n)
                hidden = lstm_out.squeeze(1)
        elif self._prev_intermediate_obs is None:
            act_norm = _norm_act(
                self._prev_action_chunk[:, 0, :].unsqueeze(1)
            ).squeeze(1)
            if self.lstm_model.action_only:
                lstm_input = act_norm.unsqueeze(1)
            else:
                obs_norm = _norm_obs(
                    self._prev_newest_obs.unsqueeze(1)
                ).squeeze(1)
                if self._lstm_keypoint_only:
                    obs_norm = obs_norm[:, :self._lstm_keypoint_dim]
                lstm_input = torch.cat(
                    [obs_norm, act_norm], dim=-1
                ).unsqueeze(1)
            with torch.no_grad():
                lstm_out, (h_n, c_n) = self.lstm_model.lstm(
                    lstm_input, self._lstm_hidden
                )
                self._lstm_hidden = (h_n, c_n)
                hidden = lstm_out.squeeze(1)
        else:
            n_steps = self._prev_action_chunk.shape[1]
            obs_for_lstm = torch.cat([
                self._prev_newest_obs.unsqueeze(1),
                self._prev_intermediate_obs[:, :-1, :],
            ], dim=1)

            hidden = None
            for k in range(n_steps):
                step_obs = obs_for_lstm[:, k, :]
                step_action = self._prev_action_chunk[:, k, :]

                act_norm = _norm_act(
                    step_action.unsqueeze(1)
                ).squeeze(1)

                if self.lstm_model.action_only:
                    lstm_input = act_norm.unsqueeze(1)
                else:
                    obs_norm = _norm_obs(
                        step_obs.unsqueeze(1)
                    ).squeeze(1)
                    if self._lstm_keypoint_only:
                        obs_norm = obs_norm[:, :self._lstm_keypoint_dim]
                    lstm_input = torch.cat(
                        [obs_norm, act_norm], dim=-1
                    ).unsqueeze(1)

                with torch.no_grad():
                    lstm_out, (h_n, c_n) = self.lstm_model.lstm(
                        lstm_input, self._lstm_hidden
                    )
                    self._lstm_hidden = (h_n, c_n)
                    hidden = lstm_out.squeeze(1)

        return hidden

    # ------------------------------------------------------------------
    # Past-action prediction (identical to frozen variant)
    # ------------------------------------------------------------------

    def predict_past_actions(self) -> Optional[torch.Tensor]:
        if (self.lstm_model is None
                or self.lstm_model.past_chunk_head is None
                or self._lstm_hidden is None):
            return None

        device = self._lstm_hidden[0].device
        h_last = self._lstm_hidden[0][-1]

        with torch.no_grad():
            past_flat = self.lstm_model.past_chunk_head(h_last)
            B = h_last.shape[0]
            if self.lstm_model.past_abstraction in ('dct', 'segment_pool'):
                K = self.lstm_model.past_n_bases
                D = self.lstm_model.action_dim
                coeffs = past_flat.view(B, K, D)
                past_norm = dct_inverse(
                    coeffs, self.lstm_model._past_dct_basis)
            else:
                past_norm = past_flat.view(
                    B, self.lstm_model.past_chunk_H,
                    self.lstm_model.action_dim)

            pH, D = self.lstm_model.past_chunk_H, self.lstm_model.action_dim
            flat = past_norm.reshape(B * pH, D).unsqueeze(1)
            flat_raw = self.normalizer["action"].unnormalize(flat)
            past_raw = flat_raw.squeeze(1).reshape(B, pH, D)

        return past_raw

    # ------------------------------------------------------------------
    # Intermediate obs update (identical to frozen variant)
    # ------------------------------------------------------------------

    def update_lstm_intermediate_obs(self, intermediate_obs: torch.Tensor):
        self._prev_intermediate_obs = intermediate_obs.detach()

    # ------------------------------------------------------------------
    # Predict action (inference / rollout)
    # ------------------------------------------------------------------

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        assert "obs" in obs_dict
        raw_obs = obs_dict["obs"]            # (B, T, env_obs_dim)
        B = raw_obs.shape[0]
        device = raw_obs.device

        if self._lstm_hidden is not None:
            if self._lstm_hidden[0].size(1) != B:
                self.reset()

        # The env may provide wider obs (e.g. 74-dim with goal keypoints)
        # for the LSTM, while the DP only uses the first raw_obs_dim dims.
        # Save full obs for LSTM, slice for DP.
        newest_obs = raw_obs[:, -1, :]       # full obs for LSTM rollout
        hidden = self._extract_lstm_hidden_state_rollout(newest_obs)

        features = hidden.unsqueeze(1).expand(
            B, raw_obs.shape[1], self.lstm_hidden_size
        )

        # Slice obs to raw_obs_dim for DP (drops goal keypoints if present)
        dp_obs = raw_obs[:, :, :self.raw_obs_dim]
        nobs = self.normalizer["obs"].normalize(dp_obs)
        if self.obs_projection is not None:
            nobs = self.obs_projection(nobs)

        if not self.projection_concat:
            if self.hidden_vq is not None:
                features, _ = self.hidden_vq(features)
            if self.hidden_projection is not None:
                features = self.hidden_projection(features)
            full_obs = torch.cat([nobs, features], dim=-1)
        else:
            if self.hidden_vq is not None:
                features, _ = self.hidden_vq(features)
            full_obs = torch.cat([nobs, features], dim=-1)
            if self.hidden_projection is not None:
                full_obs = self.hidden_projection(full_obs)

        To = self.n_obs_steps
        assert full_obs.shape[-1] == self.obs_dim
        T = self.horizon
        Da = self.action_dim
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            local_cond = torch.zeros(B, T, self.obs_dim,
                                     device=device, dtype=dtype)
            local_cond[:, :To] = full_obs[:, :To]
            shape = (B, T, Da)
            cond_data = torch.zeros(
                size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            global_cond = full_obs[:, :To].reshape(B, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(
                size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            shape = (B, T, Da + self.obs_dim)
            cond_data = torch.zeros(
                size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = full_obs[:, :To]
            cond_mask[:, :To, Da:] = True

        nsample = self.conditional_sample(
            cond_data, cond_mask,
            local_cond=local_cond, global_cond=global_cond,
            **self.kwargs,
        )

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        self._prev_newest_obs = newest_obs.detach()
        self._prev_action_chunk = action.detach()

        result = {
            "action": action,
            "action_pred": action_pred,
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[..., Da:]
            obs_pred = self.normalizer["obs"].unnormalize(nobs_pred)
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            result["action_obs_pred"] = obs_pred[:, start:end]
            result["obs_pred"] = obs_pred
        return result

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """
        Compute diffusion loss with LSTM run LIVE (with gradients).

        Expected batch keys:
            obs:              (B, H, obs_dim)       chunk observations
            action:           (B, H, action_dim)    chunk actions
            episode_obs:      (B, T_max, obs_dim)   full episode obs (padded)
            episode_action:   (B, T_max, action_dim) full episode actions (padded)
            chunk_start_idx:  (B,)                  chunk start in episode
            episode_len:      (B,)                  actual episode lengths
        """
        assert "valid_mask" not in batch

        # ---- Run LSTM live on full episode data (WITH gradients) --------
        features = self._extract_lstm_hidden_state_train(
            episode_obs=batch["episode_obs"],
            episode_action=batch["episode_action"],
            episode_len=batch["episode_len"],
            chunk_start_idx=batch["chunk_start_idx"],
        )  # (B, H, hidden_size)

        # ---- Normalise chunk obs and action for DP ----------------------
        nobs = self.normalizer["obs"].normalize(batch["obs"])
        naction = self.normalizer["action"].normalize(batch["action"])
        if self.obs_projection is not None:
            nobs = self.obs_projection(nobs)
        obs = nobs
        action = naction

        # ---- Build conditioning (identical to frozen variant) -----------
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            local_cond = obs
            local_cond[:, self.n_obs_steps:, :] = 0
        elif self.obs_as_global_cond:
            vq_loss = torch.tensor(0.0, device=features.device)
            if not self.projection_concat:
                if self.hidden_vq is not None:
                    features, vq_loss = self.hidden_vq(features)
                if self.hidden_projection is not None:
                    features = self.hidden_projection(features)
                x = torch.cat([obs, features], dim=-1)
            else:
                if self.hidden_vq is not None:
                    features, vq_loss = self.hidden_vq(features)
                x = torch.cat([obs, features], dim=-1)
                if self.hidden_projection is not None:
                    x = self.hidden_projection(x)
            obs = x
            global_cond = obs[:, :self.n_obs_steps, :].reshape(
                obs.shape[0], -1
            )
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:, start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # ---- Forward diffusion + loss (identical to frozen variant) -----
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(
                trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(
            trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        pred = self.model(
            noisy_trajectory, timesteps,
            local_cond=local_cond, global_cond=global_cond,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()

        # Add VQ loss if active
        if self.hidden_vq is not None and self.obs_as_global_cond:
            loss = loss + self.vq_loss_weight * vq_loss

        return loss
