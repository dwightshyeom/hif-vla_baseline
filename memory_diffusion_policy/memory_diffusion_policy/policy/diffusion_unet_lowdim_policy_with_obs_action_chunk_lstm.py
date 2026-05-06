"""
Diffusion Policy conditioned on observation + hidden states from a pretrained
ObsActionChunkLSTM.

During training:
  - LSTM hidden states come pre-computed from the dataset ('lstm_hidden' key).
  - The policy concatenates (normalised obs, raw LSTM hidden states)
    as the conditioning signal.

During rollout (predict_action):
  - The frozen LSTM is stateful: hidden state accumulates across calls.
  - At the first step, raw zeros are fed (matching the training convention
    where forward() prepends zeros BEFORE normalisation).
  - On subsequent calls the LSTM is advanced n_action_steps times using
    the real intermediate observations collected from the environment
    (provided via update_lstm_intermediate_obs()) and the actions from
    the previous chunk.
  - The env runner must call update_lstm_intermediate_obs() after each
    env.step() to supply the intermediate observations.
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


class VectorQuantizer(nn.Module):
    """
    Straight-through Vector Quantizer for LSTM hidden-state discretization.

    Loss terms:
        codebook_loss   = ||stop_grad(h) - e||^2   (moves codebook -> h)
        commitment_loss = ||h - stop_grad(e)||^2   (moves h -> codebook)
        total_vq_loss   = codebook_loss + commitment_weight * commitment_loss
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
        """
        Args:
            z: (B, T, D) or (B, D) continuous features.
        Returns:
            z_q : same shape as z, quantized (straight-through).
            loss: scalar VQ loss.
        """
        squeeze = False
        if z.dim() == 2:
            z = z.unsqueeze(1)
            squeeze = True

        B, T, D = z.shape
        flat = z.reshape(-1, D)  # (B*T, D)

        # Nearest-neighbour: ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z . e^T
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2.0 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )  # (B*T, K)
        idx = dist.argmin(dim=1)  # (B*T,)
        z_q_flat = self.embedding(idx)  # (B*T, D)

        codebook_loss = F.mse_loss(z_q_flat, flat.detach())
        commitment_loss = F.mse_loss(flat, z_q_flat.detach())
        vq_loss = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through estimator
        z_q = z + (z_q_flat.reshape(B, T, D) - z).detach()

        if squeeze:
            z_q = z_q.squeeze(1)
        return z_q, vq_loss


class DiffusionUnetLowdimPolicyWithObsActionChunkLSTM(BaseLowdimPolicy):
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
        # Hidden-state projection: reduce high-dim hidden state
        use_hidden_projection: bool = True,
        projection_output_dim: Optional[int] = None,
        projection_num_layers: int = 2,
        # False = project hidden then concat with obs
        # True  = concat obs+hidden first, then project together
        projection_concat: bool = False,
        raw_obs_dim: int = 20,
        # Observation projection: reduce high-dim obs to lower dim
        use_obs_projection: bool = False,
        obs_projection_output_dim: int = 20,
        obs_projection_num_layers: int = 2,
        # Vector quantization of LSTM hidden state
        use_vq: bool = False,
        vq_num_embeddings: int = 128,
        vq_commitment_weight: float = 0.25,
        vq_loss_weight: float = 1.0,
        # LSTM conditioning mode
        # 'hidden'              – raw LSTM hidden state (default)
        # 'past_dct_coeff'      – DCT coefficients from the pretrained past_chunk_head
        #                         Requires use_past_head=True in the LSTM checkpoint.
        # 'future_action'       – predicted future actions from future_chunk_head
        #                         Requires use_future_head=True in the LSTM checkpoint.
        # 'hidden+future_action'– both hidden state AND future action prediction,
        #                         each with independent VQ/MLP, concatenated.
        lstm_latent_type: str = 'hidden',
        # Future-action stream projection (only for 'hidden+future_action')
        use_future_action_projection: bool = True,
        future_action_projection_output_dim: Optional[int] = None,
        future_action_projection_num_layers: int = 2,
        # Future-action stream VQ (only for 'hidden+future_action')
        use_future_action_vq: bool = False,
        future_action_vq_num_embeddings: int = 128,
        future_action_vq_commitment_weight: float = 0.25,
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
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # ---- Pretrained LSTM (frozen) -----------------------------------
        self.lstm_model: Optional[ObsActionChunkLSTM] = None
        self._lstm_obs_dim: int = raw_obs_dim
        self._lstm_keypoint_only: bool = False
        self._lstm_keypoint_dim: int = 18
        # Must be set before _load_lstm_checkpoint so the method can use it
        # to compute the correct latent feature dimension.
        self._lstm_latent_type: str = lstm_latent_type
        self._future_action_dim: int = 0  # set by _load_lstm_checkpoint

        # Stateful rollout buffers
        self._lstm_hidden = None          # (h_n, c_n)
        self._prev_action_chunk = None    # (B, n_action_steps, action_dim)
        self._prev_intermediate_obs = None  # (B, n_action_steps, raw_obs_dim)
        self._prev_newest_obs = None      # (B, raw_obs_dim) obs at previous call

        if lstm_checkpoint_path is not None:
            self._load_lstm_checkpoint(lstm_checkpoint_path)

        # ---- Learnable hidden-state projection ----------------------------
        # NOTE: self.lstm_hidden_size may have been updated by
        # _load_lstm_checkpoint above, so use self.lstm_hidden_size here.
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
            print(f"Hidden projection disabled — using raw hidden ({self.lstm_hidden_size}-dim)")

        # ---- Learnable observation projection ----------------------------
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

        # ---- Vector Quantizer (optional) ---------------------------------
        # VQ operates on whatever the LSTM latent is (hidden state or DCT coeffs).
        # lstm_hidden_size is auto-set to the correct latent dim after checkpoint load.
        # Pipeline: latent → VQ → project → concat with obs.
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
                f"commitment={vq_commitment_weight}, loss_weight={vq_loss_weight}"
            )

        # ---- Future-action stream (for 'hidden+future_action' mode) ------
        self.future_action_projection: Optional[nn.Sequential] = None
        self.future_action_vq: Optional[VectorQuantizer] = None
        self._use_future_action_projection = use_future_action_projection
        self._future_action_projection_output_dim = future_action_projection_output_dim
        if lstm_latent_type == 'hidden+future_action':
            # Build future-action VQ
            if use_future_action_vq and self._future_action_dim > 0:
                self.future_action_vq = VectorQuantizer(
                    num_embeddings=future_action_vq_num_embeddings,
                    embedding_dim=self._future_action_dim,
                    commitment_weight=future_action_vq_commitment_weight,
                )
                print(
                    f"Future-action VQ: {future_action_vq_num_embeddings} codes, "
                    f"dim={self._future_action_dim}, "
                    f"commitment={future_action_vq_commitment_weight}"
                )
            # Build future-action projection
            if use_future_action_projection and future_action_projection_output_dim is not None:
                fa_in = self._future_action_dim
                layers = []
                in_dim = fa_in
                for i in range(future_action_projection_num_layers - 1):
                    mid_dim = max(future_action_projection_output_dim,
                                  (in_dim + future_action_projection_output_dim) // 2)
                    layers.extend([nn.Linear(in_dim, mid_dim), nn.ReLU()])
                    in_dim = mid_dim
                layers.append(nn.Linear(in_dim, future_action_projection_output_dim))
                self.future_action_projection = nn.Sequential(*layers)
                print(
                    f"Future-action projection: {fa_in} → "
                    f"{future_action_projection_output_dim} "
                    f"({future_action_projection_num_layers} layer(s))"
                )

    # ------------------------------------------------------------------
    # LSTM checkpoint loading
    # ------------------------------------------------------------------

    def _load_lstm_checkpoint(self, checkpoint_path: str):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"LSTM checkpoint not found: {checkpoint_path}"
            )

        ckpt = _load_torch(checkpoint_path)
        ckpt_args = ckpt.get("args", {})

        keypoint_only = bool(ckpt_args.get("keypoint_only", False))
        keypoint_dim = int(ckpt_args.get("keypoint_dim", 18))
        ckpt_include_goal_kps = bool(ckpt_args.get("include_goal_keypoints", False))
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
            future_abstraction=str(ckpt_args.get("future_abstraction", "raw")),
            future_n_bases=int(ckpt_args.get("future_n_bases", 32)),
            num_future_modes=int(ckpt_args.get("num_future_modes", 1)),
            action_only=bool(ckpt_args.get("action_only", False)),
            use_vq=bool(ckpt_args.get("use_vq", False)),
            vq_n_codes=int(ckpt_args.get("vq_n_codes", 512)),
            vq_commitment_weight=float(ckpt_args.get("vq_commitment_weight", 0.25)),
        )
        self.lstm_model.load_state_dict(ckpt["model_state"])
        self.lstm_model.eval()
        # Freeze — the LSTM is pretrained and should not be updated.
        for p in self.lstm_model.parameters():
            p.requires_grad_(False)

        self._lstm_obs_dim = lstm_obs_dim
        self._lstm_keypoint_only = keypoint_only
        self._lstm_keypoint_dim = keypoint_dim

        # Determine the latent feature dimension based on latent type
        import warnings

        # Compute future-action output dimension (for modes that use it)
        m = self.lstm_model
        if m.future_chunk_head is not None:
            if m.future_abstraction == 'dct':
                fa_dim = m.num_future_modes * m.future_n_bases * m.action_dim
            else:
                fa_dim = m.num_future_modes * m.chunk_H * m.action_dim
        else:
            fa_dim = 0
        self._future_action_dim = fa_dim

        if self._lstm_latent_type == 'past_dct_coeff':
            assert self.lstm_model.past_chunk_head is not None, (
                "lstm_latent_type='past_dct_coeff' requires use_past_head=True "
                "in the LSTM checkpoint"
            )
            lstm_latent_dim = self.lstm_model.past_n_bases * self.lstm_model.action_dim
            print(
                f"  Using past DCT coefficients as latent: "
                f"K={self.lstm_model.past_n_bases}, D={self.lstm_model.action_dim}, "
                f"latent_dim={lstm_latent_dim}"
            )
        elif self._lstm_latent_type == 'future_action':
            assert self.lstm_model.future_chunk_head is not None, (
                "lstm_latent_type='future_action' requires use_future_head=True "
                "in the LSTM checkpoint"
            )
            lstm_latent_dim = fa_dim
            print(
                f"  Using future action prediction as latent: "
                f"dim={lstm_latent_dim}"
            )
        elif self._lstm_latent_type == 'hidden+future_action':
            assert self.lstm_model.future_chunk_head is not None, (
                "lstm_latent_type='hidden+future_action' requires "
                "use_future_head=True in the LSTM checkpoint"
            )
            # For combined mode, lstm_hidden_size tracks the hidden stream only;
            # the future-action stream dimension is _future_action_dim.
            lstm_latent_dim = self.lstm_model.hidden_size
            print(
                f"  Using hidden + future action as latent: "
                f"hidden_dim={lstm_latent_dim}, future_action_dim={fa_dim}"
            )
        else:
            lstm_latent_dim = self.lstm_model.hidden_size

        if lstm_latent_dim != self.lstm_hidden_size:
            warnings.warn(
                f"Config lstm_hidden_size={self.lstm_hidden_size} does not match "
                f"actual LSTM latent dim={lstm_latent_dim} "
                f"(type='{self._lstm_latent_type}'). "
                f"Overwriting to {lstm_latent_dim}. Please update your config!"
            )
        self.lstm_hidden_size = lstm_latent_dim

        print(f"Loaded pretrained ObsActionChunkLSTM from: {checkpoint_path}")
        print(
            f"  hidden_size={self.lstm_model.hidden_size}, "
            f"lstm_obs_dim={lstm_obs_dim}, keypoint_only={keypoint_only}, "
            f"action_only={self.lstm_model.action_only}, "
            f"lstm_latent_type={self._lstm_latent_type}"
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
    # Diffusion sampling
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
    # LSTM hidden-state extraction (rollout)
    # ------------------------------------------------------------------

    def _extract_lstm_hidden_state_rollout(
        self, current_obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Advance the LSTM and return the updated hidden state.

        At the very first call (no buffer), a single raw-zero input is used,
        matching the training convention where forward() prepends zeros.

        For subsequent calls, the LSTM is advanced n_action_steps times using
        the actual (obs, action) pairs from the previous chunk, provided via
        update_lstm_intermediate_obs().

        Args:
            current_obs: (B, raw_obs_dim) current raw observation (e.g. 20-d).

        Returns:
            hidden: (B, hidden_size) LSTM hidden state after all steps.
        """
        assert self.lstm_model is not None
        B = current_obs.shape[0]
        device = current_obs.device
        input_dim = self.lstm_model.input_dim  # obs_dim + action_dim

        # Move LSTM to same device if needed
        if next(self.lstm_model.parameters()).device != device:
            self.lstm_model = self.lstm_model.to(device)
        self.lstm_model.eval()

        if self._prev_action_chunk is None:
            # ---- First call: feed raw zeros -----------------------------
            # In training, forward() prepends zeros BEFORE normalisation.
            # So the first LSTM input is raw zeros, not normalised zeros.
            lstm_input = torch.zeros(B, 1, input_dim, device=device)
            with torch.no_grad():
                lstm_out, (h_n, c_n) = self.lstm_model.lstm(
                    lstm_input, self._lstm_hidden
                )
                self._lstm_hidden = (h_n, c_n)
                hidden = lstm_out.squeeze(1)  # (B, hidden_size)
        elif self._prev_intermediate_obs is None:
            # ---- No intermediate obs (e.g. training sampling) -----------
            # Fall back to single-step: feed (prev_newest_obs, first action)
            act_norm = self.normalizer["action"].normalize(
                self._prev_action_chunk[:, 0, :].unsqueeze(1)
            ).squeeze(1)
            if self.lstm_model.action_only:
                lstm_input = act_norm.unsqueeze(1)
            else:
                obs_norm = self.normalizer["obs"].normalize(
                    self._prev_newest_obs.unsqueeze(1)
                ).squeeze(1)
                if self._lstm_keypoint_only:
                    obs_norm = obs_norm[:, : self._lstm_keypoint_dim]
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
            # ---- Full multi-step: advance for all n_action_steps --------
            # LSTM shift convention: input at step t = [obs_{t-1}, act_{t-1}]
            #
            # _prev_action_chunk:     (B, n, action_dim)  — actions [a_t, ..., a_{t+n-1}]
            # _prev_newest_obs:       (B, raw_obs_dim)    — obs_t (obs at start of chunk)
            # _prev_intermediate_obs: (B, n, raw_obs_dim) — [obs_{t+1}, ..., obs_{t+n}]
            #                                               (obs AFTER each action)
            #
            # For LSTM step t+k+1, input = [obs_{t+k}, act_{t+k}]
            # obs_{t+0} = _prev_newest_obs
            # obs_{t+1} = intermediate[0], ..., obs_{t+n-1} = intermediate[n-2]
            n_steps = self._prev_action_chunk.shape[1]

            # Build shifted obs: [prev_newest_obs, intermediate[0], ..., intermediate[n-2]]
            obs_for_lstm = torch.cat([
                self._prev_newest_obs.unsqueeze(1),        # (B, 1, raw_obs_dim)
                self._prev_intermediate_obs[:, :-1, :],    # (B, n-1, raw_obs_dim)
            ], dim=1)  # (B, n, raw_obs_dim)

            hidden = None
            for k in range(n_steps):
                step_obs = obs_for_lstm[:, k, :]               # (B, raw_obs_dim)
                step_action = self._prev_action_chunk[:, k, :]  # (B, action_dim)

                # Normalise
                act_norm = self.normalizer["action"].normalize(
                    step_action.unsqueeze(1)
                ).squeeze(1)

                if self.lstm_model.action_only:
                    lstm_input = act_norm.unsqueeze(1)  # (B, 1, action_dim)
                else:
                    obs_norm = self.normalizer["obs"].normalize(
                        step_obs.unsqueeze(1)
                    ).squeeze(1)
                    if self._lstm_keypoint_only:
                        obs_norm = obs_norm[:, : self._lstm_keypoint_dim]
                    lstm_input = torch.cat(
                        [obs_norm, act_norm], dim=-1
                    ).unsqueeze(1)  # (B, 1, input_dim)

                with torch.no_grad():
                    lstm_out, (h_n, c_n) = self.lstm_model.lstm(
                        lstm_input, self._lstm_hidden
                    )
                    self._lstm_hidden = (h_n, c_n)
                    hidden = lstm_out.squeeze(1)

        # Apply frozen head(s) to transform the raw hidden state
        if self._lstm_latent_type == 'past_dct_coeff':
            with torch.no_grad():
                hidden = self.lstm_model.past_chunk_head(hidden)  # (B, K*D)
            return hidden
        elif self._lstm_latent_type == 'future_action':
            with torch.no_grad():
                future_action = self.lstm_model.future_chunk_head(hidden)  # (B, fa_dim)
            return future_action
        elif self._lstm_latent_type == 'hidden+future_action':
            with torch.no_grad():
                future_action = self.lstm_model.future_chunk_head(hidden)  # (B, fa_dim)
            return hidden, future_action

        return hidden

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
                or self.lstm_model.past_chunk_head is None
                or self._lstm_hidden is None):
            return None

        device = self._lstm_hidden[0].device
        # h_n: (num_layers, B, hidden_size) — take the last layer
        h_last = self._lstm_hidden[0][-1]  # (B, hidden_size)

        with torch.no_grad():
            past_flat = self.lstm_model.past_chunk_head(h_last)  # (B, pastH*D) or (B, K*D)
            B = h_last.shape[0]
            if self.lstm_model.past_abstraction in ('dct', 'segment_pool'):
                # head outputs K basis coefficients per action dim → inverse to pastH steps
                K = self.lstm_model.past_n_bases
                D = self.lstm_model.action_dim
                coeffs = past_flat.view(B, K, D)           # (B, K, D)
                past_norm = dct_inverse(coeffs, self.lstm_model._past_dct_basis)  # (B, pastH, D)
            else:
                past_norm = past_flat.view(
                    B, self.lstm_model.past_chunk_H, self.lstm_model.action_dim
                )  # (B, pastH, D) in LP-normalised space

            # Denormalise to raw (pixel) space
            pH, D = self.lstm_model.past_chunk_H, self.lstm_model.action_dim
            flat = past_norm.reshape(B * pH, D).unsqueeze(1)  # (B*pH, 1, D)
            flat_raw = self.normalizer["action"].unnormalize(flat)
            past_raw = flat_raw.squeeze(1).reshape(B, pH, D)  # (B, pastH, D)

        return past_raw

    # ------------------------------------------------------------------
    # Intermediate obs update (called by env runner after env.step)
    # ------------------------------------------------------------------

    def update_lstm_intermediate_obs(self, intermediate_obs: torch.Tensor):
        """
        Store the intermediate observations collected from the environment
        during the previous n_action_steps, so the LSTM can process them
        at the next predict_action call.

        Args:
            intermediate_obs: (B, n_action_steps, raw_obs_dim) real obs from
                each sub-step of the previous action chunk execution.
        """
        self._prev_intermediate_obs = intermediate_obs.detach()

    # ------------------------------------------------------------------
    # Predict action (inference / rollout)
    # ------------------------------------------------------------------

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict must include "obs" key with shape (B, To, raw_obs_dim).
        Returns dict with "action" key.
        """
        assert "obs" in obs_dict
        raw_obs = obs_dict["obs"]  # (B, To, 20)
        B = raw_obs.shape[0]
        device = raw_obs.device

        # Reset LSTM if batch size changed (e.g. eval vs train)
        if self._lstm_hidden is not None:
            if self._lstm_hidden[0].size(1) != B:
                self.reset()

        # ---- LSTM hidden state from buffered (prev_obs, prev_action) ----
        newest_obs = raw_obs[:, -1, :]  # (B, 20)
        lstm_out = self._extract_lstm_hidden_state_rollout(newest_obs)

        # Unpack based on latent type
        if self._lstm_latent_type == 'hidden+future_action':
            hidden, future_action = lstm_out  # (B, H), (B, fa_dim)
        else:
            hidden = lstm_out  # (B, latent_dim)

        # Expand hidden to observation window shape
        features = hidden.unsqueeze(1).expand(
            B, raw_obs.shape[1], self.lstm_hidden_size
        )  # (B, To, hidden_size)

        # ---- Normalise obs, optionally project obs, build full_obs ------
        nobs = self.normalizer["obs"].normalize(raw_obs)
        if self.obs_projection is not None:
            nobs = self.obs_projection(nobs)

        if not self.projection_concat:
            # VQ → project hidden → concat with obs
            if self.hidden_vq is not None:
                features, _ = self.hidden_vq(features)
            if self.hidden_projection is not None:
                features = self.hidden_projection(features)
            parts = [nobs, features]

            # Future-action stream (only for 'hidden+future_action')
            if self._lstm_latent_type == 'hidden+future_action':
                fa_features = future_action.unsqueeze(1).expand(
                    B, raw_obs.shape[1], self._future_action_dim
                )
                if self.future_action_vq is not None:
                    fa_features, _ = self.future_action_vq(fa_features)
                if self.future_action_projection is not None:
                    fa_features = self.future_action_projection(fa_features)
                parts.append(fa_features)

            full_obs = torch.cat(parts, dim=-1)
        else:
            # VQ on raw hidden, then concat obs+hidden, then project
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

        # ---- Build conditioning -----------------------------------------
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            local_cond = torch.zeros(B, T, self.obs_dim,
                                     device=device, dtype=dtype)
            local_cond[:, :To] = full_obs[:, :To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            global_cond = full_obs[:, :To].reshape(B, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            shape = (B, T, Da + self.obs_dim)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = full_obs[:, :To]
            cond_mask[:, :To, Da:] = True

        # ---- Run diffusion sampling ------------------------------------
        nsample = self.conditional_sample(
            cond_data, cond_mask,
            local_cond=local_cond, global_cond=global_cond,
            **self.kwargs,
        )

        # ---- Denormalise action -----------------------------------------
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

        # ---- Buffer for next LSTM multi-step update ----------------------
        # Store newest_obs + action chunk. The env runner will call
        # update_lstm_intermediate_obs() after env.step() to provide
        # the real intermediate observations.
        self._prev_newest_obs = newest_obs.detach()    # (B, raw_obs_dim)
        self._prev_action_chunk = action.detach()      # (B, n_action_steps, Da)

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
        assert "valid_mask" not in batch

        # Pre-computed LSTM latent features from dataset.
        # Shape: (B, T, lstm_hidden_size), where lstm_hidden_size depends
        # on lstm_latent_type ('hidden', 'past_dct_coeff', 'future_action',
        # or hidden stream of 'hidden+future_action').
        features = batch["lstm_hidden"]  # (B, T, lstm_hidden_size)

        # Optional second stream for 'hidden+future_action'
        fa_features = batch.get("lstm_future_action", None)  # (B, T, fa_dim) or None

        # Normalise obs and action
        nobs = self.normalizer["obs"].normalize(batch["obs"])
        naction = self.normalizer["action"].normalize(batch["action"])
        if self.obs_projection is not None:
            nobs = self.obs_projection(nobs)
        obs = nobs
        action = naction

        # ---- Build conditioning -----------------------------------------
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            local_cond = obs
            local_cond[:, self.n_obs_steps :, :] = 0
        elif self.obs_as_global_cond:
            vq_loss = torch.tensor(0.0, device=features.device)
            fa_vq_loss = torch.tensor(0.0, device=features.device)
            if not self.projection_concat:
                # VQ → project hidden → concat with obs
                if self.hidden_vq is not None:
                    features, vq_loss = self.hidden_vq(features)
                if self.hidden_projection is not None:
                    features = self.hidden_projection(features)
                parts = [obs, features]

                # Future-action stream (only for 'hidden+future_action')
                if fa_features is not None:
                    if self.future_action_vq is not None:
                        fa_features, fa_vq_loss = self.future_action_vq(fa_features)
                    if self.future_action_projection is not None:
                        fa_features = self.future_action_projection(fa_features)
                    parts.append(fa_features)

                x = torch.cat(parts, dim=-1)
            else:
                # VQ → concat obs+hidden → project
                if self.hidden_vq is not None:
                    features, vq_loss = self.hidden_vq(features)
                x = torch.cat([obs, features], dim=-1)
                if self.hidden_projection is not None:
                    x = self.hidden_projection(x)
            obs = x
            global_cond = obs[:, : self.n_obs_steps, :].reshape(
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

        # ---- Forward diffusion + loss -----------------------------------
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
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
        if self.obs_as_global_cond:
            if self.hidden_vq is not None:
                loss = loss + self.vq_loss_weight * vq_loss
            if self.future_action_vq is not None:
                loss = loss + self.vq_loss_weight * fa_vq_loss

        return loss
