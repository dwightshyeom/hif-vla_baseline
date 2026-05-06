"""
Dataset for **finetunable** vision-LSTM + Diffusion Policy training.

Key difference from the frozen variant
(obs_action_chunk_lstm_image_chunked_latent_dataset.py):
  - LSTM hidden states are NOT pre-computed.  Instead, each sample
    returns the *full episode* obs-features (pre-encoded with the frozen
    visual encoder) and actions so that the **policy** can run the LSTM
    core live during training (with gradients flowing through the LSTM
    recurrent weights but NOT through the visual encoder).

Pre-encoding strategy
---------------------
The LSTM's internal visual encoder (CNN) is expensive to run per
training step on full episodes.  We pre-encode all episode images once
at dataset-creation time using the frozen LSTM obs_encoder, cache the
resulting feature vectors on CPU, and store them alongside the raw
actions.  The policy can then run *only the LSTM core* live during
training, which is cheap.

Each training sample provides:
    obs:                   dict {image: (H, 3, 96, 96),
                                 agent_pos: (H, 2)}    — DP chunk (raw images + raw agent_pos)
    action:                (H, 2)                       — DP chunk (raw actions)
    episode_obs_features:  (T_ep, obs_feature_dim)      — pre-encoded obs features (LSTM-normalised)
    episode_action_norm:   (T_ep, 2)                    — LSTM-normalised actions
    chunk_start_idx:       int                          — episode-relative start of chunk
    episode_len:           int                          — actual episode length

A custom collate function pads variable-length episode data.
"""

from typing import Dict, List, Optional
from pathlib import Path

import torch
import numpy as np

from memory_diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.replay_buffer import ReplayBuffer
from memory_diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from memory_diffusion_policy.model.common.normalizer import LinearNormalizer
from memory_diffusion_policy.common.normalize_util import get_image_range_normalizer


def _load_torch(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ======================================================================
# Custom collate function
# ======================================================================

def finetune_image_collate_fn(
    batch: List[Dict],
) -> Dict[str, torch.Tensor]:
    """
    Collate function that pads variable-length episode data.

    Fixed-size tensors (obs, action) are simply stacked.
    Variable-length tensors (episode_obs_features, episode_action_norm)
    are zero-padded to max(episode_len) within the mini-batch.
    """
    B = len(batch)

    # Fixed-size chunk data
    obs_images = torch.stack([b["obs"]["image"] for b in batch])
    obs_agent_pos = torch.stack([b["obs"]["agent_pos"] for b in batch])
    actions = torch.stack([b["action"] for b in batch])
    chunk_start_idx = torch.tensor(
        [b["chunk_start_idx"] for b in batch], dtype=torch.long)
    episode_len = torch.tensor(
        [b["episode_len"] for b in batch], dtype=torch.long)

    # Variable-length episode data: pad to max episode length
    max_len = int(episode_len.max().item())
    feat_dim = batch[0]["episode_obs_features"].shape[-1]
    action_dim = batch[0]["episode_action_norm"].shape[-1]

    episode_obs_features = torch.zeros(
        B, max_len, feat_dim,
        dtype=batch[0]["episode_obs_features"].dtype)
    episode_action_norm = torch.zeros(
        B, max_len, action_dim,
        dtype=batch[0]["episode_action_norm"].dtype)

    for i, b in enumerate(batch):
        L = b["episode_obs_features"].shape[0]
        episode_obs_features[i, :L] = b["episode_obs_features"]
        episode_action_norm[i, :L] = b["episode_action_norm"]

    return {
        "obs": {
            "image": obs_images,          # (B, H, 3, 96, 96)
            "agent_pos": obs_agent_pos,   # (B, H, 2)
        },
        "action": actions,                          # (B, H, 2)
        "episode_obs_features": episode_obs_features,  # (B, T_max, feat_dim)
        "episode_action_norm": episode_action_norm,    # (B, T_max, 2)
        "chunk_start_idx": chunk_start_idx,             # (B,)
        "episode_len": episode_len,                     # (B,)
    }


# ======================================================================
# Dataset
# ======================================================================

class ObsActionChunkLSTMImageFinetuneDataset(BaseImageDataset):
    """
    Dataset for finetunable vision-LSTM + Diffusion Policy training.

    Pre-encodes all episode images with the frozen LSTM visual encoder,
    caches the resulting features, then returns per-chunk:
        - obs chunk   (images + agent_pos)  for the DP's own encoder
        - action chunk                       for the DP
        - full episode features + actions    for live LSTM-core execution

    Parameters
    ----------
    zarr_path:
        Path to the zarr dataset with keys 'img', 'state', 'action'.
    lstm_checkpoint_path:
        Path to the pretrained ObsActionChunkLSTMImage checkpoint.
    lstm_normalizer_path:
        Path to the LSTM normalizer (defaults to
        <checkpoint_dir>/normalizer.pt).
    horizon, pad_before, pad_after, seed, val_ratio, max_train_episodes:
        Same as in the frozen dataset.
    """

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 8,
        pad_before: int = 0,
        pad_after: int = 3,
        seed: int = 42,
        val_ratio: float = 0.02,
        max_train_episodes: Optional[int] = None,
        lstm_checkpoint_path: Optional[str] = None,
        lstm_normalizer_path: Optional[str] = None,
        device: str = "cuda:0",
    ):
        super().__init__()

        if lstm_checkpoint_path is None:
            raise ValueError("lstm_checkpoint_path is required")

        # ---- Load replay buffer ----------------------------------------
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=["img", "state", "action"])

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        # ---- Load pretrained vision LSTM (encoder will be frozen) ------
        from memory_diffusion_policy.model.obs_action_chunk_lstm_image import (
            ObsActionChunkLSTMImage,
        )

        ckpt_path = Path(lstm_checkpoint_path)
        assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
        print(f"Loading pretrained ObsActionChunkLSTMImage from: {ckpt_path}")

        ckpt = _load_torch(ckpt_path)
        ckpt_args = ckpt.get("args", {})

        shape_meta = ckpt.get("shape_meta", {
            "obs": {
                "image": {"shape": [3, 96, 96], "type": "rgb"},
                "agent_pos": {"shape": [2], "type": "low_dim"},
            },
            "action": {"shape": [2]},
        })

        crop_shape = tuple(ckpt_args.get("crop_shape", [76, 76]))
        past_chunk_H = ckpt_args.get(
            "past_chunk_H", ckpt_args.get("chunk_H", 16))

        lstm_model = ObsActionChunkLSTMImage(
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
            past_chunk_H=past_chunk_H,
            past_abstraction=ckpt_args.get("past_abstraction", "raw"),
            past_n_bases=int(ckpt_args.get("past_n_bases", 32)),
            use_vq=bool(ckpt_args.get("use_vq", False)),
            vq_n_codes=int(ckpt_args.get("vq_n_codes", 512)),
            vq_commitment_weight=float(
                ckpt_args.get("vq_commitment_weight", 0.25)),
            crop_shape=crop_shape,
            obs_encoder_group_norm=bool(
                ckpt_args.get("obs_encoder_group_norm", True)),
            eval_fixed_crop=True,
            freeze_encoder=True,  # always freeze encoder for pre-encoding
        )
        lstm_model.load_state_dict(ckpt["model_state"])
        lstm_model.eval()
        for p in lstm_model.parameters():
            p.requires_grad_(False)

        self.lstm_hidden_size = int(ckpt_args.get("hidden_size", 256))
        self.obs_feature_dim = lstm_model.obs_feature_dim
        # Subsample rate the LSTM was pretrained with.  Stored in the checkpoint
        # under ckpt["args"]["action_step_subsample"] (same as hidden_size etc.).
        # Must be applied when pre-encoding so the LSTM core sees its expected
        # temporal resolution during finetuning.
        self.action_step_subsample = int(ckpt_args.get("action_step_subsample", 1))
        print(
            f"  obs_feature_dim={self.obs_feature_dim}, "
            f"hidden_size={self.lstm_hidden_size}, "
            f"action_step_subsample={self.action_step_subsample}"
        )

        # ---- LSTM normalizer -------------------------------------------
        norm_path = (
            Path(lstm_normalizer_path)
            if lstm_normalizer_path
            else ckpt_path.parent / "normalizer.pt"
        )
        if not norm_path.exists():
            raise FileNotFoundError(
                f"LSTM normalizer not found at {norm_path}. "
                f"Provide lstm_normalizer_path explicitly.")
        lstm_normalizer = _load_torch(norm_path)
        print(f"  Loaded LSTM normalizer from: {norm_path}")
        self._lstm_normalizer = lstm_normalizer

        # ---- Pre-encode episodes (obs_encoder frozen) ------------------
        dev = (
            torch.device(device)
            if torch.cuda.is_available() and "cuda" in device
            else torch.device("cpu")
        )
        lstm_model = lstm_model.to(dev)
        print(f"  Moved LSTM encoder to {dev}")

        n_total = self.replay_buffer.n_episodes
        print(
            f"Pre-encoding episode images for {n_total} episodes "
            f"(frozen obs_encoder)...")

        # Cache: ep_idx → (T_raw, obs_feature_dim) tensor on CPU
        self._obs_feature_cache: Dict[int, torch.Tensor] = {}
        # Cache: ep_idx → (T_raw, action_dim) normalized action on CPU
        self._action_norm_cache: Dict[int, torch.Tensor] = {}

        self._precompute_episode_features(lstm_model, lstm_normalizer, dev)

        del lstm_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Released LSTM encoder from memory")

        # ---- Chunk sampler ---------------------------------------------
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )

        # Pre-compute global → episode mapping
        self._episode_idxs = self.replay_buffer.get_episode_idxs()
        ends = self.replay_buffer.episode_ends[:]
        self._episode_starts = np.zeros(len(ends), dtype=np.int64)
        self._episode_starts[1:] = ends[:-1]

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    def _precompute_episode_features(self, lstm_model, normalizer, dev):
        """Run frozen obs_encoder on every episode; cache features + actions."""
        lstm_model.eval()
        n_total = self.replay_buffer.n_episodes

        for ep_idx in range(n_total):
            ep = self.replay_buffer.get_episode(ep_idx, copy=False)

            # Image: (T, H, W, 3) uint8 → (T, 3, H, W) float32 [0, 1]
            image_raw = np.moveaxis(ep["img"], -1, 1).astype(np.float32) / 255.0
            # Slice state -> agent_pos: tolerate wider state arrays (asym/swap
            # store extra task fields beyond agent_pos in dims 2:).
            state_raw = np.asarray(ep["state"], dtype=np.float32)
            agent_pos_raw = state_raw[:, :2] if state_raw.ndim == 2 and state_raw.shape[1] > 2 else state_raw
            action_raw = ep["action"].astype(np.float32)

            # Apply the same temporal subsampling used during LSTM pretraining.
            # This ensures the LSTM core sees its expected temporal resolution.
            s = self.action_step_subsample
            if s > 1:
                image_raw    = image_raw[::s]
                agent_pos_raw = agent_pos_raw[::s]
                action_raw   = action_raw[::s]

            T_raw = len(image_raw)

            # To tensors: (1, T, ...)
            image_t = torch.from_numpy(image_raw).unsqueeze(0).to(dev)
            agent_t = torch.from_numpy(agent_pos_raw).unsqueeze(0).to(dev)
            action_t = torch.from_numpy(action_raw).unsqueeze(0).to(dev)

            # Normalise agent_pos for the LSTM encoder
            agent_norm_t = self._normalize_seq(
                normalizer, agent_t, "agent_pos", dev)

            # Normalise action for the LSTM core
            action_norm_t = self._normalize_seq(
                normalizer, action_t, "action", dev)

            with torch.no_grad():
                # encode_obs: (1, T, obs_feature_dim)
                obs_features = lstm_model.encode_obs(
                    image_t, agent_norm_t)

            self._obs_feature_cache[ep_idx] = obs_features.squeeze(0).cpu()
            self._action_norm_cache[ep_idx] = action_norm_t.squeeze(0).cpu()

            if (ep_idx + 1) % 20 == 0:
                print(f"  Encoded {ep_idx + 1}/{n_total} episodes")

        print(f"Finished pre-encoding {n_total} episodes")

    @staticmethod
    def _normalize_seq(normalizer, x: torch.Tensor, key: str, dev):
        """Normalize (B, T, D) tensor using normalizer[key]."""
        B, T, D = x.shape
        flat = x.reshape(B * T, D).unsqueeze(1)
        norm_obj = normalizer[key]
        norm_obj = norm_obj.to(dev)
        out = norm_obj.normalize(flat)
        return out.squeeze(1).reshape(B, T, D)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_validation_dataset(self):
        val_set = ObsActionChunkLSTMImageFinetuneDataset.__new__(
            ObsActionChunkLSTMImageFinetuneDataset
        )
        val_set.replay_buffer = self.replay_buffer
        val_set.train_mask = ~self.train_mask
        val_set.horizon = self.horizon
        val_set.pad_before = self.pad_before
        val_set.pad_after = self.pad_after
        val_set.lstm_hidden_size = self.lstm_hidden_size
        val_set.obs_feature_dim = self.obs_feature_dim
        val_set._lstm_normalizer = self._lstm_normalizer
        val_set.action_step_subsample = self.action_step_subsample
        val_set._obs_feature_cache = self._obs_feature_cache   # shared
        val_set._action_norm_cache = self._action_norm_cache   # shared
        val_set._episode_idxs = self._episode_idxs             # shared
        val_set._episode_starts = self._episode_starts         # shared
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        """Return a DP normalizer (for images + agent_pos + action)."""
        state = np.asarray(self.replay_buffer["state"])
        agent_pos = state[:, :2] if state.ndim == 2 and state.shape[1] > 2 else state
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": agent_pos,
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["image"] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    @staticmethod
    def collate_fn(batch):
        return finetune_image_collate_fn(batch)

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.sampler.sample_sequence(idx)
        buffer_start, buffer_end, sample_start, sample_end = (
            self.sampler.indices[idx])

        # ---- DP chunk data -------------------------------------------
        # Image: (H, H_img, W_img, 3) uint8 → (H, 3, H_img, W_img) float32
        image = (
            np.moveaxis(sample["img"], -1, 1).astype(np.float32) / 255.0)
        state = np.asarray(sample["state"], dtype=np.float32)
        agent_pos = state[:, :2] if state.ndim == 2 and state.shape[1] > 2 else state
        action = sample["action"].astype(np.float32)

        # ---- Episode lookup ------------------------------------------
        ep_idx = int(self._episode_idxs[buffer_start])
        ep_start = int(self._episode_starts[ep_idx])

        # chunk_start_idx: episode-relative index of first real data point,
        # converted to the LSTM's subsampled timestep space.
        chunk_start_idx = int(buffer_start - ep_start) // self.action_step_subsample

        # Full episode cached data
        ep_obs_feat = self._obs_feature_cache[ep_idx]   # (T_ep, feat_dim)
        ep_act_norm = self._action_norm_cache[ep_idx]   # (T_ep, 2)
        episode_len = int(ep_obs_feat.shape[0])

        return {
            "obs": {
                "image": torch.from_numpy(image),        # (H, 3, 96, 96)
                "agent_pos": torch.from_numpy(agent_pos),# (H, 2)
            },
            "action": torch.from_numpy(action),          # (H, 2)
            "episode_obs_features": ep_obs_feat,         # (T_ep, feat_dim)
            "episode_action_norm": ep_act_norm,          # (T_ep, 2)
            "chunk_start_idx": chunk_start_idx,          # int
            "episode_len": episode_len,                  # int
        }
