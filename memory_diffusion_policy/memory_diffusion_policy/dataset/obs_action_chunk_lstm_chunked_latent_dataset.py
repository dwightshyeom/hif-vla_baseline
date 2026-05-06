"""
Dataset wrapper that loads full episodes, extracts LSTM hidden states from a
pretrained ObsActionChunkLSTM, then chunks for Diffusion Policy training.

Workflow:
1. Load full episodes from PushTThreeGoalsLSTMDatasetWithIndicator
2. Load pretrained ObsActionChunkLSTM (frozen)
3. Feed full (obs_norm, action_norm) sequences → extract per-step hidden states
4. Cache hidden states on CPU
5. Chunk episodes into DP-compatible format (horizon, dim)
6. Return per chunk: {obs, lstm_hidden, action}

The LSTM processes concat(obs_{t-1}, action_{t-1}) at each step t
(shift is handled internally by ObsActionChunkLSTM.forward).
The hidden state h_t encodes memory of all (obs, action) pairs up to t-1.

Each training sample is a chunk of:
    obs:         (horizon, obs_dim)        — raw observations
    lstm_hidden: (horizon, hidden_size)    — LSTM hidden states (raw, not normalised)
    action:      (horizon, action_dim)     — actions to predict
"""

from typing import Dict, Optional

import torch
import numpy as np
from pathlib import Path

from memory_diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from memory_diffusion_policy.dataset.pusht_three_goals_lstm_dataset import (
    PushTThreeGoalsLSTMDatasetWithIndicator,
)
from memory_diffusion_policy.model.obs_action_chunk_lstm import ObsActionChunkLSTM


def _load_torch(path, map_location="cpu"):
    """Compat loader that works with both old and new PyTorch."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class ObsActionChunkLSTMChunkedLatentDataset(BaseLowdimDataset):
    """
    Wrapper that pre-computes LSTM hidden states from a pretrained
    ObsActionChunkLSTM and chunks episodes for Diffusion Policy training.

    The LSTM processes concat(obs_{t-1}, action_{t-1}) at each step t (the
    shift is applied internally).  The hidden state h_t encodes memory of
    all (obs, action) pairs up to time t-1.

    Dataset items are aligned so that at position t in the chunk:
        obs[t]         = observation at time t
        lstm_hidden[t] = h_t (memory up to t-1)
        action[t]      = action at time t (to be predicted by DP)
    """

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        lstm_checkpoint_path: Optional[str] = None,
        lstm_normalizer_path: Optional[str] = None,
        include_goal_keypoints: bool = False,
        lstm_latent_type: str = 'hidden',  # 'hidden' | 'past_dct_coeff' | 'future_action' | 'hidden+future_action'
        device: str = "cuda:0",
    ):
        """
        Args:
            zarr_path: Path to zarr dataset.
            horizon: Chunk length for DP.
            pad_before: Padding before chunk.
            pad_after: Padding after chunk.
            seed: Random seed.
            val_ratio: Validation ratio.
            max_train_episodes: Max training episodes.
            lstm_checkpoint_path: Path to pretrained ObsActionChunkLSTM checkpoint.
            lstm_normalizer_path: Path to LSTM normalizer (defaults to
                                  <checkpoint_dir>/normalizer.pt).
            lstm_latent_type: 'hidden' = use LSTM hidden state (default);
                              'past_dct_coeff' = use DCT coefficients from the
                               pretrained past_chunk_head;
                              'future_action' = use predicted future actions from
                               the pretrained future_chunk_head;
                              'hidden+future_action' = use both hidden state and
                               predicted future actions (returned as two keys).
            device: Device for LSTM inference.
        """
        # ---- Episode dataset --------------------------------------------
        self.episode_dataset = PushTThreeGoalsLSTMDatasetWithIndicator(
            zarr_path=zarr_path,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
            include_goal_keypoints=include_goal_keypoints,
        )

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.device = device

        self._lstm_latent_type = lstm_latent_type

        # Cache: ep_idx → (T, latent_dim) tensor on CPU
        # latent_dim = hidden_size (for 'hidden') or past_n_bases*action_dim (for 'past_dct_coeff')
        # or future_action_dim (for 'future_action') or hidden_size (hidden stream of 'hidden+future_action')
        self.hidden_states_cache: Dict[int, torch.Tensor] = {}
        # Second cache for the future-action stream (only 'hidden+future_action')
        self.future_action_cache: Dict[int, torch.Tensor] = {}

        if lstm_checkpoint_path is None:
            raise ValueError("lstm_checkpoint_path is required")

        # ---- Load pretrained LSTM model ---------------------------------
        ckpt_path = Path(lstm_checkpoint_path)
        assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

        print(f"Loading pretrained ObsActionChunkLSTM from: {ckpt_path}")
        ckpt = _load_torch(ckpt_path)
        ckpt_args = ckpt.get("args", {})

        keypoint_only = bool(ckpt_args.get("keypoint_only", False))
        keypoint_dim = int(ckpt_args.get("keypoint_dim", 18))
        ckpt_include_goal_kps = bool(ckpt_args.get("include_goal_keypoints", False))
        base_obs_dim = 74 if ckpt_include_goal_kps else 20
        lstm_obs_dim = keypoint_dim if keypoint_only else base_obs_dim
        action_dim = 2

        lstm_model = ObsActionChunkLSTM(
            obs_dim=lstm_obs_dim,
            action_dim=action_dim,
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
        lstm_model.load_state_dict(ckpt["model_state"])
        lstm_model.eval()

        # Determine the feature (latent) dimension based on latent type
        # Compute future-action dim (for modes that need it)
        m = lstm_model
        if m.future_chunk_head is not None:
            if m.future_abstraction == 'dct':
                self._future_action_dim = m.num_future_modes * m.future_n_bases * m.action_dim
            else:
                self._future_action_dim = m.num_future_modes * m.chunk_H * m.action_dim
        else:
            self._future_action_dim = 0

        if lstm_latent_type == 'past_dct_coeff':
            assert lstm_model.past_chunk_head is not None, (
                "lstm_latent_type='past_dct_coeff' requires use_past_head=True "
                "in the LSTM checkpoint"
            )
            self.lstm_latent_dim = lstm_model.past_n_bases * lstm_model.action_dim
            print(
                f"  Using past DCT coefficients as latent: "
                f"K={lstm_model.past_n_bases}, D={lstm_model.action_dim}, "
                f"latent_dim={self.lstm_latent_dim}"
            )
        elif lstm_latent_type == 'future_action':
            assert lstm_model.future_chunk_head is not None, (
                "lstm_latent_type='future_action' requires use_future_head=True "
                "in the LSTM checkpoint"
            )
            self.lstm_latent_dim = self._future_action_dim
            print(
                f"  Using future action prediction as latent: "
                f"dim={self.lstm_latent_dim}"
            )
        elif lstm_latent_type == 'hidden+future_action':
            assert lstm_model.future_chunk_head is not None, (
                "lstm_latent_type='hidden+future_action' requires "
                "use_future_head=True in the LSTM checkpoint"
            )
            self.lstm_latent_dim = lstm_model.hidden_size
            print(
                f"  Using hidden + future action as latent: "
                f"hidden_dim={self.lstm_latent_dim}, "
                f"future_action_dim={self._future_action_dim}"
            )
        else:
            self.lstm_latent_dim = lstm_model.hidden_size
        self.lstm_hidden_size = self.lstm_latent_dim  # backward-compat alias

        self._lstm_obs_dim = lstm_obs_dim
        self._keypoint_only = keypoint_only
        self._keypoint_dim = keypoint_dim

        print(
            f"  hidden_size={lstm_model.hidden_size}, lstm_obs_dim={lstm_obs_dim}, "
            f"keypoint_only={keypoint_only}, lstm_latent_type={lstm_latent_type}"
        )

        # ---- Normalizer (same one used during LSTM training) ------------
        norm_path = Path(lstm_normalizer_path) if lstm_normalizer_path else (
            ckpt_path.parent / "normalizer.pt"
        )
        if norm_path.exists():
            lstm_normalizer = _load_torch(norm_path)
            print(f"  Loaded LSTM normalizer from: {norm_path}")
        else:
            print(
                f"  LSTM normalizer not found at {norm_path}, "
                "using dataset normalizer"
            )
            lstm_normalizer = self.episode_dataset.get_normalizer()

        # ---- Move to device & pre-compute hidden states -----------------
        if torch.cuda.is_available() and "cuda" in device:
            lstm_model = lstm_model.to(device)
            print(f"  Moved LSTM to {device}")

        # Pre-compute for ALL episodes (train + val) so that
        # get_validation_dataset() can reuse the same cache keyed by
        # actual replay-buffer episode index.
        n_total = self.episode_dataset.replay_buffer.n_episodes
        print(
            f"Pre-computing LSTM hidden states for "
            f"{n_total} episodes (all train + val)..."
        )
        self._precompute_hidden_states(lstm_model, lstm_normalizer, device,
                                       n_total=n_total)

        # Free the model
        del lstm_model, lstm_normalizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Released LSTM model from memory")

        # ---- Build chunk index ------------------------------------------
        self._build_chunk_index()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_seq(normalizer, x: torch.Tensor, key: str, device: str):
        """Normalise (B, T, D) using LP normalizer on *device*."""
        B, T, D = x.shape
        xf = x.reshape(B * T, D).unsqueeze(1)          # (B*T, 1, D)
        norm = normalizer[key]
        norm.to(device)
        out = norm.normalize(xf)
        return out.squeeze(1).reshape(B, T, D)

    def _precompute_hidden_states(self, lstm_model, normalizer, device,
                                   n_total=None):
        """Run LSTM on every episode and cache per-step hidden states.

        Hidden states are keyed by the *actual* replay-buffer episode index
        so that both train and validation datasets can look them up correctly.
        """
        lstm_model.eval()

        if n_total is None:
            n_total = self.episode_dataset.replay_buffer.n_episodes

        replay_buffer = self.episode_dataset.replay_buffer

        for actual_ep_idx in range(n_total):
            ep_sample = replay_buffer.get_episode(actual_ep_idx, copy=False)
            data = self.episode_dataset._sample_to_data(ep_sample)
            obs_full = torch.from_numpy(data["obs"]).float()    # (T, obs_dim)
            action_full = torch.from_numpy(data["action"]).float()  # (T, 2)
            T = obs_full.shape[0]

            obs_t = obs_full.unsqueeze(0)        # (1, T, obs_dim)
            action_t = action_full.unsqueeze(0)   # (1, T, 2)

            obs_norm = self._normalize_seq(normalizer, obs_t, "obs", device)
            action_norm = self._normalize_seq(normalizer, action_t, "action", device)

            if self._keypoint_only:
                obs_norm = obs_norm[:, :, : self._keypoint_dim]

            lengths = torch.tensor([T])

            with torch.no_grad():
                lstm_output, _ = lstm_model.extract_hidden_states(
                    obs_norm.to(device),
                    action_norm.to(device),
                    lengths=lengths,
                    hidden_state=None,
                )
                # lstm_output: (1, T, hidden_size)
                if self._lstm_latent_type == 'past_dct_coeff':
                    B_ep, T_ep, H = lstm_output.shape
                    features = lstm_model.past_chunk_head(
                        lstm_output.reshape(B_ep * T_ep, H)
                    ).reshape(B_ep, T_ep, -1)  # (1, T, K*D)
                elif self._lstm_latent_type == 'future_action':
                    B_ep, T_ep, H = lstm_output.shape
                    features = lstm_model.future_chunk_head(
                        lstm_output.reshape(B_ep * T_ep, H)
                    ).reshape(B_ep, T_ep, -1)  # (1, T, fa_dim)
                elif self._lstm_latent_type == 'hidden+future_action':
                    B_ep, T_ep, H = lstm_output.shape
                    fa_features = lstm_model.future_chunk_head(
                        lstm_output.reshape(B_ep * T_ep, H)
                    ).reshape(B_ep, T_ep, -1)  # (1, T, fa_dim)
                    features = lstm_output  # hidden stream: (1, T, hidden_size)
                else:
                    features = lstm_output  # (1, T, hidden_size)

            self.hidden_states_cache[actual_ep_idx] = features.squeeze(0).cpu()
            if self._lstm_latent_type == 'hidden+future_action':
                self.future_action_cache[actual_ep_idx] = fa_features.squeeze(0).cpu()

            if (actual_ep_idx + 1) % 50 == 0:
                print(
                    f"  Processed {actual_ep_idx + 1}/{n_total} episodes"
                )

        print(
            f"Finished pre-computing hidden states for "
            f"{n_total} episodes"
        )

    def _build_chunk_index(self):
        """Build flat index → (episode_idx, start_idx) mapping."""
        self.chunk_index = []

        for ep_idx in range(len(self.episode_dataset)):
            episode = self.episode_dataset[ep_idx]
            T = episode["obs"].shape[0]

            for start_idx in range(T):
                chunk_start = start_idx - self.pad_before
                chunk_end = chunk_start + self.horizon

                if chunk_end <= T + self.pad_after:
                    self.chunk_index.append((ep_idx, start_idx))

        print(
            f"Built chunk index: {len(self.chunk_index)} chunks "
            f"from {len(self.episode_dataset)} episodes"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_validation_dataset(self):
        """Create validation dataset (shares pre-computed hidden states)."""
        val_ds = ObsActionChunkLSTMChunkedLatentDataset.__new__(
            ObsActionChunkLSTMChunkedLatentDataset
        )
        val_ds.episode_dataset = self.episode_dataset.get_validation_dataset()
        val_ds.horizon = self.horizon
        val_ds.pad_before = self.pad_before
        val_ds.pad_after = self.pad_after
        val_ds.device = self.device
        val_ds._lstm_latent_type = self._lstm_latent_type
        val_ds.lstm_latent_dim = self.lstm_latent_dim
        val_ds.lstm_hidden_size = self.lstm_hidden_size  # backward-compat alias
        val_ds._lstm_obs_dim = self._lstm_obs_dim
        val_ds._keypoint_only = self._keypoint_only
        val_ds._keypoint_dim = self._keypoint_dim
        val_ds.hidden_states_cache = self.hidden_states_cache   # shared
        val_ds.future_action_cache = self.future_action_cache   # shared
        val_ds._future_action_dim = self._future_action_dim
        val_ds._build_chunk_index()
        return val_ds

    def get_normalizer(self, mode="limits", **kwargs):
        """Get normalizer from episode dataset."""
        return self.episode_dataset.get_normalizer(mode=mode, **kwargs)

    def get_all_actions(self):
        """Get all actions from episode dataset."""
        return self.episode_dataset.get_all_actions()

    def __len__(self):
        return len(self.chunk_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            obs:                 (horizon, obs_dim)      raw observations
            action:              (horizon, 2)            actions
            lstm_hidden:         (horizon, hidden_size)  LSTM hidden/latent features
            lstm_future_action:  (horizon, fa_dim)       future action features
                                 (only for 'hidden+future_action' mode)
        """
        ep_idx, start_idx = self.chunk_index[idx]

        episode = self.episode_dataset[ep_idx]
        obs_full = episode["obs"]             # (T, obs_dim)
        action_full = episode["action"]       # (T, 2)

        # Look up hidden states by actual replay-buffer episode index
        actual_ep_idx = self.episode_dataset.episode_indices[ep_idx]
        features_full = self.hidden_states_cache[actual_ep_idx]  # (T, hidden_size)

        # Optional future-action stream
        has_fa = self._lstm_latent_type == 'hidden+future_action'
        if has_fa:
            fa_full = self.future_action_cache[actual_ep_idx]  # (T, fa_dim)

        T = obs_full.shape[0]
        chunk_start = start_idx - self.pad_before
        chunk_end = chunk_start + self.horizon

        # ---- Extract chunk (with boundary padding) ----------------------
        if chunk_start < 0:
            n_pad = -chunk_start
            obs_before = obs_full[0:1].repeat(n_pad, 1)
            act_before = torch.zeros(n_pad, action_full.shape[-1],
                                     dtype=action_full.dtype)
            feat_before = features_full[0:1].repeat(n_pad, 1)

            obs_chunk = torch.cat([obs_before, obs_full[:chunk_end]], dim=0)
            act_chunk = torch.cat([act_before, action_full[:chunk_end]], dim=0)
            feat_chunk = torch.cat([feat_before, features_full[:chunk_end]], dim=0)

            if has_fa:
                fa_before = fa_full[0:1].repeat(n_pad, 1)
                fa_chunk = torch.cat([fa_before, fa_full[:chunk_end]], dim=0)
        else:
            end = min(chunk_end, T)
            obs_chunk = obs_full[chunk_start:end]
            act_chunk = action_full[chunk_start:end]
            feat_chunk = features_full[chunk_start:end]

            if has_fa:
                fa_chunk = fa_full[chunk_start:end]

        if chunk_end > T:
            n_pad = chunk_end - T
            obs_after = obs_full[-1:].repeat(n_pad, 1)
            act_after = torch.zeros(n_pad, action_full.shape[-1],
                                    dtype=action_full.dtype)
            feat_after = features_full[-1:].repeat(n_pad, 1)

            obs_chunk = torch.cat([obs_chunk, obs_after], dim=0)
            act_chunk = torch.cat([act_chunk, act_after], dim=0)
            feat_chunk = torch.cat([feat_chunk, feat_after], dim=0)

            if has_fa:
                fa_after = fa_full[-1:].repeat(n_pad, 1)
                fa_chunk = torch.cat([fa_chunk, fa_after], dim=0)

        result = {
            "obs": obs_chunk,            # (horizon, obs_dim)
            "action": act_chunk,          # (horizon, 2)
            "lstm_hidden": feat_chunk,    # (horizon, hidden_size)
        }
        if has_fa:
            result["lstm_future_action"] = fa_chunk  # (horizon, fa_dim)
        return result
