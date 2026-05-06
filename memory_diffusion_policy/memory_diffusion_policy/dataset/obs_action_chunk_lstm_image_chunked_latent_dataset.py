"""
Vision-based dataset wrapper that loads full episodes, extracts LSTM hidden
states from a pretrained ObsActionChunkLSTMImage (frozen), then chunks for
Diffusion Policy (image-based) training.

Workflow:
1. Load full episodes from PushTThreeGoalsImageDataset (images + agent_pos + action).
2. Load pretrained ObsActionChunkLSTMImage (frozen).
3. Feed full (image, agent_pos_norm, action_norm) sequences through the frozen model
   → extract per-step hidden states.
4. Cache hidden states on CPU.
5. Chunk episodes into DP-compatible format (horizon, dim).
6. Return per chunk: {obs: {image, agent_pos}, lstm_hidden, action}.

The vision LSTM is pretrained with action_step_subsample=4 (the LAST step of
each 4-step window).  During pre-computation here we replicate the same
subsampling so that each hidden state h_t corresponds to one DP decision point.
Then we un-subsample (repeat) the hidden states back to raw timestep resolution
so that the DP sampler (SequenceSampler) can chunk at raw resolution.

Resolution alignment:
  - LSTM pretraining: sees subsampled steps (every 4th, last of window)
  - Dataset pre-computation: replicates the same subsampling
  - DP training: DP sees raw-resolution chunks, but each h_t within a
    4-step window is identical (the state after the last LSTM step).
    This is correct because during rollout the LSTM is also stepped once
    per DP decision, and the hidden state persists for the 4 action steps.
"""

from typing import Dict, Optional

import torch
import numpy as np
from pathlib import Path

from memory_diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.replay_buffer import ReplayBuffer
from memory_diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from memory_diffusion_policy.model.common.normalizer import LinearNormalizer
from memory_diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply


def _load_torch(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class ObsActionChunkLSTMImageChunkedLatentDataset(BaseImageDataset):
    """
    Pre-computes LSTM hidden states from a pretrained ObsActionChunkLSTMImage
    and chunks episodes for vision-based Diffusion Policy training.

    Each training sample is a chunk of:
        obs:         {image: (H, 3, 96, 96), agent_pos: (H, 2)}
        lstm_hidden: (H, hidden_size)  — LSTM hidden states
        action:      (H, 2)            — actions to predict
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
        action_step_subsample: int = 4,
        lstm_latent_type: str = 'hidden',
        device: str = "cuda:0",
    ):
        super().__init__()

        if lstm_checkpoint_path is None:
            raise ValueError("lstm_checkpoint_path is required")

        # ---- Load replay buffer -----------------------------------------
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])

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
        self.action_step_subsample = int(action_step_subsample)
        self._lstm_latent_type = lstm_latent_type

        # Cache: actual_ep_idx → (T_raw, hidden_size) tensor on CPU
        self.hidden_states_cache: Dict[int, torch.Tensor] = {}

        # ---- Load pretrained vision LSTM --------------------------------
        from memory_diffusion_policy.model.obs_action_chunk_lstm_image import (
            ObsActionChunkLSTMImage,
        )

        ckpt_path = Path(lstm_checkpoint_path)
        assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

        print(f"Loading pretrained ObsActionChunkLSTMImage from: {ckpt_path}")
        ckpt = _load_torch(ckpt_path)
        ckpt_args = ckpt.get("args", {})

        shape_meta = ckpt.get("shape_meta", {
            'obs': {
                'image': {'shape': [3, 96, 96], 'type': 'rgb'},
                'agent_pos': {'shape': [2], 'type': 'low_dim'},
            },
            'action': {'shape': [2]},
        })

        crop_shape = tuple(ckpt_args.get('crop_shape', [76, 76]))
        past_chunk_H = ckpt_args.get('past_chunk_H', ckpt_args.get('chunk_H', 16))

        lstm_model = ObsActionChunkLSTMImage(
            shape_meta=shape_meta,
            action_dim=2,
            hidden_size=int(ckpt_args.get('hidden_size', 256)),
            num_layers=int(ckpt_args.get('num_layers', 2)),
            dropout=float(ckpt_args.get('dropout', 0.1)),
            chunk_H=int(ckpt_args.get('chunk_H', 16)),
            use_obs_head=False,
            use_future_head=bool(ckpt_args.get('use_future_head', True)),
            num_future_modes=int(ckpt_args.get('num_future_modes', 1)),
            future_abstraction=ckpt_args.get('future_abstraction', 'raw'),
            future_n_bases=int(ckpt_args.get('future_n_bases', 32)),
            use_past_head=bool(ckpt_args.get('use_past_head', True)),
            past_chunk_H=past_chunk_H,
            past_abstraction=ckpt_args.get('past_abstraction', 'raw'),
            past_n_bases=int(ckpt_args.get('past_n_bases', 32)),
            use_vq=bool(ckpt_args.get('use_vq', False)),
            vq_n_codes=int(ckpt_args.get('vq_n_codes', 512)),
            vq_commitment_weight=float(ckpt_args.get('vq_commitment_weight', 0.25)),
            crop_shape=crop_shape,
            obs_encoder_group_norm=bool(ckpt_args.get('obs_encoder_group_norm', True)),
            eval_fixed_crop=True,
            freeze_encoder=False,
        )
        lstm_model.load_state_dict(ckpt["model_state"])
        lstm_model.eval()
        for p in lstm_model.parameters():
            p.requires_grad_(False)

        # Determine latent dim
        inner = lstm_model.lstm
        if lstm_latent_type == 'hidden':
            self.lstm_latent_dim = inner.hidden_size
        else:
            raise ValueError(
                f"Unsupported lstm_latent_type='{lstm_latent_type}' "
                "for vision LSTM. Only 'hidden' is supported."
            )
        self.lstm_hidden_size = self.lstm_latent_dim

        print(
            f"  hidden_size={inner.hidden_size}, "
            f"obs_feature_dim={lstm_model.obs_feature_dim}, "
            f"action_step_subsample={self.action_step_subsample}, "
            f"lstm_latent_type={lstm_latent_type}"
        )

        # ---- LSTM normalizer (for action/agent_pos normalisation) -------
        norm_path = (
            Path(lstm_normalizer_path) if lstm_normalizer_path
            else ckpt_path.parent / "normalizer.pt"
        )
        if norm_path.exists():
            lstm_normalizer = _load_torch(norm_path)
            print(f"  Loaded LSTM normalizer from: {norm_path}")
        else:
            raise FileNotFoundError(
                f"LSTM normalizer not found at {norm_path}"
            )
        self._lstm_normalizer = lstm_normalizer

        # ---- Move model to device & pre-compute -------------------------
        if torch.cuda.is_available() and "cuda" in device:
            lstm_model = lstm_model.to(device)
            print(f"  Moved vision LSTM to {device}")

        n_total = self.replay_buffer.n_episodes
        print(
            f"Pre-computing vision LSTM hidden states for "
            f"{n_total} episodes..."
        )
        self._precompute_hidden_states(lstm_model, lstm_normalizer, device)

        del lstm_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Released vision LSTM model from memory")

        # ---- Build sampler (raw resolution) -----------------------------
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )

        # Pre-compute global→episode mapping for fast __getitem__
        self._episode_idxs = self.replay_buffer.get_episode_idxs()  # (N_total,) int
        ends = self.replay_buffer.episode_ends[:]
        self._episode_starts = np.zeros(len(ends), dtype=np.int64)
        self._episode_starts[1:] = ends[:-1]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_seq(normalizer, x: torch.Tensor, key: str, device: str):
        B, T, D = x.shape
        xf = x.reshape(B * T, D).unsqueeze(1)
        norm = normalizer[key]
        norm.to(device)
        out = norm.normalize(xf)
        return out.squeeze(1).reshape(B, T, D)

    def _precompute_hidden_states(self, lstm_model, normalizer, device):
        """Run frozen vision LSTM on every episode, cache hidden states."""
        lstm_model.eval()
        n_total = self.replay_buffer.n_episodes
        s = self.action_step_subsample

        for ep_idx in range(n_total):
            ep = self.replay_buffer.get_episode(ep_idx, copy=False)

            # Raw episode data
            image_raw = np.moveaxis(ep['img'], -1, 1).astype(np.float32) / 255.0
            # Slice state -> agent_pos: tolerate wider state arrays (asym/swap
            # store extra task fields beyond agent_pos in dims 2:).
            state_raw = np.asarray(ep['state'], dtype=np.float32)
            agent_pos_raw = state_raw[:, :2] if state_raw.ndim == 2 and state_raw.shape[1] > 2 else state_raw
            action_raw = ep['action'].astype(np.float32)
            T_raw = len(image_raw)

            # Subsample: take last step of each window (matching LSTM pretraining)
            if s > 1:
                image_sub = image_raw[s - 1::s]
                agent_pos_sub = agent_pos_raw[s - 1::s]
                action_sub = action_raw[s - 1::s]
            else:
                image_sub = image_raw
                agent_pos_sub = agent_pos_raw
                action_sub = action_raw
            T_sub = len(image_sub)

            # To tensors: (1, T_sub, ...)
            image_t = torch.from_numpy(image_sub).unsqueeze(0).to(device)
            agent_t = torch.from_numpy(agent_pos_sub).unsqueeze(0).to(device)
            action_t = torch.from_numpy(action_sub).unsqueeze(0).to(device)

            # Normalise agent_pos and action (image normalisation is done inside encode_obs)
            agent_norm = self._normalize_seq(normalizer, agent_t, 'agent_pos', device)
            action_norm = self._normalize_seq(normalizer, action_t, 'action', device)

            lengths = torch.tensor([T_sub])

            with torch.no_grad():
                # extract_hidden_states returns (1, T_sub, hidden_size)
                lstm_output, _ = lstm_model.extract_hidden_states(
                    image_t, agent_norm, action_norm, lengths=lengths,
                )
                # lstm_output: (1, T_sub, hidden_size)
                h_sub = lstm_output.squeeze(0).cpu()  # (T_sub, hidden_size)

            # Un-subsample: repeat each hidden state s times to fill raw resolution
            if s > 1:
                h_raw = torch.zeros(T_raw, h_sub.shape[-1], dtype=h_sub.dtype)
                for i in range(T_sub):
                    raw_start = i * s
                    raw_end = min(raw_start + s, T_raw)
                    h_raw[raw_start:raw_end] = h_sub[i]
                # For initial steps before first LSTM step (indices 0..s-2),
                # use the first hidden state
                # Actually h_sub[0] already covers raw_start=0..s-1, so this is fine.
            else:
                h_raw = h_sub

            self.hidden_states_cache[ep_idx] = h_raw

            if (ep_idx + 1) % 20 == 0:
                print(f"  Processed {ep_idx + 1}/{n_total} episodes")

        print(f"Finished pre-computing hidden states for {n_total} episodes")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_validation_dataset(self):
        val_set = ObsActionChunkLSTMImageChunkedLatentDataset.__new__(
            ObsActionChunkLSTMImageChunkedLatentDataset
        )
        val_set.replay_buffer = self.replay_buffer
        val_set.train_mask = ~self.train_mask
        val_set.horizon = self.horizon
        val_set.pad_before = self.pad_before
        val_set.pad_after = self.pad_after
        val_set.action_step_subsample = self.action_step_subsample
        val_set._lstm_latent_type = self._lstm_latent_type
        val_set.lstm_latent_dim = self.lstm_latent_dim
        val_set.lstm_hidden_size = self.lstm_hidden_size
        val_set.hidden_states_cache = self.hidden_states_cache  # shared
        val_set._lstm_normalizer = self._lstm_normalizer
        val_set._episode_idxs = self._episode_idxs              # shared
        val_set._episode_starts = self._episode_starts           # shared
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        state = np.asarray(self.replay_buffer['state'])
        agent_pos = state[:, :2] if state.ndim == 2 and state.shape[1] > 2 else state
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': agent_pos,
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # sample_sequence handles padding automatically
        sample = self.sampler.sample_sequence(idx)

        # Image: (T, 96, 96, 3) uint8 -> (T, 3, 96, 96) float32 [0,1]
        image = np.moveaxis(sample['img'], -1, 1).astype(np.float32) / 255.0
        state = np.asarray(sample['state'], dtype=np.float32)
        agent_pos = state[:, :2] if state.ndim == 2 and state.shape[1] > 2 else state
        action = sample['action'].astype(np.float32)

        # ---- Look up pre-computed hidden states ----
        # self.sampler.indices[idx] = [buffer_start, buffer_end, sample_start, sample_end]
        buffer_start, buffer_end, sample_start, sample_end = self.sampler.indices[idx]

        # Determine episode index (all buffer indices in one sample belong to the same episode)
        ep_idx = int(self._episode_idxs[buffer_start])
        ep_start = int(self._episode_starts[ep_idx])

        # Local offsets within the episode for the buffer range
        local_start = buffer_start - ep_start
        local_end = buffer_end - ep_start  # exclusive

        # Cached hidden states for this episode: (T_ep, hidden_size)
        h_ep = self.hidden_states_cache[ep_idx]

        # Slice the hidden states for the buffer range
        h_buf = h_ep[local_start:local_end]  # (buffer_end - buffer_start, hidden_size)

        # Apply the same padding that sample_sequence applies:
        # sample_start = number of padding steps at the beginning (repeat first element)
        # The total output length is self.horizon (= self.sampler.sequence_length)
        if sample_start > 0:
            # pad beginning by repeating first element
            pad_front = h_buf[:1].expand(sample_start, -1)
            h_seq = torch.cat([pad_front, h_buf], dim=0)
        else:
            h_seq = h_buf

        total_len = self.horizon
        if h_seq.shape[0] < total_len:
            # pad end by repeating last element
            n_pad = total_len - h_seq.shape[0]
            pad_back = h_seq[-1:].expand(n_pad, -1)
            h_seq = torch.cat([h_seq, pad_back], dim=0)

        lstm_hidden = h_seq  # (total_len, hidden_size)

        data = {
            'obs': {
                'image': torch.from_numpy(image),           # (T, 3, 96, 96)
                'agent_pos': torch.from_numpy(agent_pos),   # (T, 2)
            },
            'action': torch.from_numpy(action),             # (T, 2)
            'lstm_hidden': lstm_hidden,                     # (T, hidden_size)
        }
        return data
