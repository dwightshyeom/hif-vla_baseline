"""
pusht_hifvla_dataset.py

Zarr-backed PyTorch IterableDataset for the Swap-T task that produces batches
whose dict keys match those expected by the HiF-VLA finetune.py training loop
(i.e., the same interface as RLDSBatchTransform).

Key adaptations vs. the RLDS pipeline:
  ┌─────────────────────────────┬──────────────────────────────────────────────────┐
  │ Original (LIBERO)           │ This file (Push-T)                               │
  ├─────────────────────────────┼──────────────────────────────────────────────────┤
  │ RLDS / TF Datasets          │ Zarr store (Diffusion Policy format)             │
  │ ACTION_DIM = 7              │ ACTION_DIM = 2  (xy delta)                       │
  │ PROPRIO_DIM = 8             │ PROPRIO_DIM = 8 (agent_xy + blue xytheta + red)  │
  │ MPEG-4 motion vectors       │ Dense Farneback optical flow (16×16)             │
  │ Per-episode language inst.  │ Fixed dummy string                               │
  │ 224×224 images (JPEG)       │ Resize 96×96 RGB → 224×224                      │
  └─────────────────────────────┴──────────────────────────────────────────────────┘

Batch dict produced per sample:
  pixel_values   – image tensor after processor.image_processor.apply_transform
  input_ids      – tokenized [prompt + action_chunk] (int64 tensor)
  labels         – same with non-action positions = IGNORE_INDEX
  actions        – (NUM_ACTIONS_CHUNK, ACTION_DIM=2) float32, normalised to [-1,1]
  dataset_name   – "swap_3t" (str)
  proprio        – (PROPRIO_DIM=8,) float32, normalised to [-1,1]
  mv_history     – (history_length, 2, 16, 16) float32, normalised optical flow
  mv_future      – (NUM_ACTIONS_CHUNK, 2, 16, 16) float32, normalised optical flow
  episode_idx    – int (episode index within the zarr store)
  motion_path    – "" (unused; retained for interface compatibility)
  time_step      – int (step index within the episode)
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import zarr
from PIL import Image
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

# Ensure HiF-VLA source is importable when this dataset is used standalone.
# The training script adds the HiF-VLA root to sys.path before importing this.

from baseline.dataset.optical_flow import (
    FLOW_H,
    FLOW_W,
    compute_flow_stats,
    load_flow_stats,
    normalize_flow,
    precompute_flow,
    save_flow_stats,
)

# These imports require HiF-VLA to be on sys.path (done by the training script).
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)

# Fixed language instruction used for all Swap-T episodes.
PUSHT_LANGUAGE_INSTRUCTION: str = "swap the colored blocks to their target positions"

# Dataset identifier used as the key in dataset_statistics.json.
DATASET_NAME: str = "swap_3t"


# ---------------------------------------------------------------------------
# Action / proprio normalisation helpers
# ---------------------------------------------------------------------------

def _compute_bounds_q99(
    data: np.ndarray,
) -> Dict[str, List[float]]:
    """Return q01 / q99 per column for BOUNDS_Q99 normalisation."""
    q01 = np.percentile(data, 1, axis=0).astype(np.float32)
    q99 = np.percentile(data, 99, axis=0).astype(np.float32)
    return {"q01": q01.tolist(), "q99": q99.tolist()}


def _normalize_bounds_q99(
    x: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
) -> np.ndarray:
    """Clip-normalise to [-1, 1] using per-dimension q01 / q99."""
    return np.clip(
        2.0 * (x - q01) / (q99 - q01 + 1e-8) - 1.0,
        -1.0,
        1.0,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class PushTHiFVLADataset(IterableDataset):
    """
    Zarr-backed IterableDataset for the Swap-T task.

    Parameters
    ----------
    zarr_path : str
        Path to the zarr store produced by demo_pusht_three_goals_swap.py.
    action_tokenizer : ActionTokenizer
        HiF-VLA action tokenizer (bins continuous actions → token strings).
    base_tokenizer : PreTrainedTokenizerBase
        LLM tokenizer (converts strings to integer token IDs).
    image_transform : callable
        ``processor.image_processor.apply_transform`` – converts a PIL Image
        to the normalised float tensor expected by the VLM.
    prompt_builder_fn : Type[PromptBuilder]
        Prompt-builder class (e.g. ``PurePromptBuilder``).
    history_length : int
        Number of optical-flow frames to include in ``mv_history``.
        Must equal the ``history_length`` passed to finetune_pusht.py.
    flow_cache_path : str, optional
        Path to a ``.npy`` file for the pre-computed flow array.  Computed
        and saved on first run; loaded directly on subsequent runs.
    flow_stats_path : str, optional
        Path to a ``.npz`` file for flow mean/std statistics.  Computed from
        training episodes and saved on first run.
    stats_save_path : str, optional
        Path where ``dataset_statistics.json`` will be written.  Pass
        ``run_dir / "dataset_statistics.json"`` from the training script.
    train : bool
        If True use training episodes; if False use held-out validation eps.
    val_fraction : float
        Fraction of episodes (from the end) reserved for validation.
    """

    @staticmethod
    def open_zarr(zarr_path: str):
        """Open a zarr store from a local directory, a .zip file, or an HF Hub path.

        Accepts:
          - Local directory zarr  : ``/path/to/store``
          - Local zip zarr        : ``/path/to/store.zarr.zip`` or ``*.zip``
          - HF Hub zip zarr       : auto-downloaded if ``zarr_path`` starts with
                                    ``hf://``  (format: ``hf://<repo_id>/<filename>``,
                                    repo_type=dataset)
        """
        if zarr_path.startswith("hf://"):
            from huggingface_hub import hf_hub_download
            # Format: hf://<repo_id>/<filename>
            _, rest = zarr_path.split("hf://", 1)
            repo_id, filename = rest.split("/", 1)
            print(f"[PushTHiFVLADataset] Downloading zarr from HF Hub: {repo_id}/{filename}")
            zarr_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")

        if zarr_path.endswith(".zip"):
            return zarr.open(zarr.ZipStore(zarr_path, mode="r"), mode="r")
        return zarr.open(zarr_path, "r")

    def __init__(
        self,
        zarr_path: str,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform,
        prompt_builder_fn: Type[PromptBuilder],
        history_length: int = 8,
        flow_cache_path: Optional[str] = None,
        flow_stats_path: Optional[str] = None,
        stats_save_path: Optional[str] = None,
        train: bool = True,
        val_fraction: float = 0.1,
    ) -> None:
        self.zarr_path = zarr_path
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.history_length = history_length

        # ------------------------------------------------------------------
        # Load zarr arrays
        # ------------------------------------------------------------------
        store = self.open_zarr(zarr_path)
        # Load into NumPy for fast random access during training.
        print("[PushTHiFVLADataset] Loading zarr arrays …")
        self.imgs    = store["data/img"][:]        # (N, 96, 96, 3) uint8
        self.actions = store["data/action"][:]     # (N, 2)          float32
        self.states  = store["data/state"][:]      # (N, 8)          float32
        self.episode_ends   = store["meta/episode_ends"][:]  # (E,)  int64
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]])
        n_eps = len(self.episode_ends)

        # ------------------------------------------------------------------
        # Train / val episode split
        # ------------------------------------------------------------------
        val_start = int(n_eps * (1.0 - val_fraction))
        self.ep_indices: List[int] = (
            list(range(0, val_start)) if train
            else list(range(val_start, n_eps))
        )

        # Steps belonging to training episodes (used for computing statistics).
        train_steps = np.concatenate([
            np.arange(int(self.episode_starts[ep]), int(self.episode_ends[ep]))
            for ep in range(0, val_start)
        ]).astype(int)

        # ------------------------------------------------------------------
        # Optical flow: pre-compute or load from cache
        # ------------------------------------------------------------------
        self.flows = precompute_flow(
            self.imgs, self.episode_ends, cache_path=flow_cache_path
        )

        # Flow normalisation statistics (training episodes only).
        if flow_stats_path is not None and os.path.exists(flow_stats_path):
            self.flow_mean, self.flow_std = load_flow_stats(flow_stats_path)
        else:
            train_mask = np.zeros(len(self.flows), dtype=bool)
            train_mask[train_steps] = True
            self.flow_mean, self.flow_std = compute_flow_stats(self.flows, valid_mask=train_mask)
            if flow_stats_path is not None:
                save_flow_stats(self.flow_mean, self.flow_std, flow_stats_path)

        # ------------------------------------------------------------------
        # Action / proprio normalisation statistics (training episodes only)
        # ------------------------------------------------------------------
        train_actions = self.actions[train_steps]
        train_states  = self.states[train_steps]

        action_stats = _compute_bounds_q99(train_actions)
        proprio_stats = _compute_bounds_q99(train_states)

        self.action_q01 = np.array(action_stats["q01"], dtype=np.float32)
        self.action_q99 = np.array(action_stats["q99"], dtype=np.float32)
        self.proprio_q01 = np.array(proprio_stats["q01"], dtype=np.float32)
        self.proprio_q99 = np.array(proprio_stats["q99"], dtype=np.float32)

        # dataset_statistics.json  ─ compatible with OpenVLA's save/load logic
        self.dataset_statistics: Dict[str, Any] = {
            DATASET_NAME: {
                "action": action_stats,
                "proprio": proprio_stats,
            }
        }
        if stats_save_path is not None:
            save_path = Path(stats_save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(self.dataset_statistics, f, indent=2)
            print(f"[PushTHiFVLADataset] Saved dataset stats → {save_path}")

        # ------------------------------------------------------------------
        # Build list of valid (episode_idx, global_step) pairs.
        # A step at global index t is valid when t + NUM_ACTIONS_CHUNK <= ep_end,
        # i.e., there are enough future steps for the full action chunk.
        # ------------------------------------------------------------------
        self.valid_indices: List[Tuple[int, int]] = []
        for ep in self.ep_indices:
            ep_s = int(self.episode_starts[ep])
            ep_e = int(self.episode_ends[ep])
            # Need NUM_ACTIONS_CHUNK future steps; also need at least 1 step
            # of future flow for mv_future[0].
            for t in range(ep_s, ep_e - NUM_ACTIONS_CHUNK):
                self.valid_indices.append((ep, t))

        print(
            f"[PushTHiFVLADataset] {'train' if train else 'val'}: "
            f"{len(self.ep_indices)} episodes, {len(self.valid_indices)} valid steps."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_flow_history(self, global_t: int, ep_start: int) -> np.ndarray:
        """
        Return ``history_length`` optical-flow frames ending at ``global_t``
        (inclusive).  Steps before the episode start are zero-padded.

        Returns
        -------
        np.ndarray  shape (history_length, 2, FLOW_H, FLOW_W) float32, normalised.
        """
        result = np.zeros((self.history_length, 2, FLOW_H, FLOW_W), dtype=np.float32)
        for i, offset in enumerate(range(-(self.history_length - 1), 1)):
            idx = global_t + offset
            if idx >= ep_start:
                result[i] = self.flows[idx]
        return normalize_flow(result, self.flow_mean, self.flow_std)

    def _get_flow_future(self, global_t: int) -> np.ndarray:
        """
        Return ``NUM_ACTIONS_CHUNK`` optical-flow frames starting one step
        after ``global_t``.  Caller must ensure the steps exist (guaranteed
        by valid_indices construction).

        Returns
        -------
        np.ndarray  shape (NUM_ACTIONS_CHUNK, 2, FLOW_H, FLOW_W) float32, normalised.
        """
        result = self.flows[global_t + 1 : global_t + 1 + NUM_ACTIONS_CHUNK].copy()
        return normalize_flow(result, self.flow_mean, self.flow_std)

    def _build_sample(self, ep_idx: int, global_t: int) -> Dict[str, Any]:
        ep_start = int(self.episode_starts[ep_idx])

        # ── Image ──────────────────────────────────────────────────────────
        img_pil = Image.fromarray(self.imgs[global_t]).convert("RGB")

        # ── Action chunk (current + future) ────────────────────────────────
        raw_actions = self.actions[global_t : global_t + NUM_ACTIONS_CHUNK]   # (C, 2)
        norm_actions = _normalize_bounds_q99(raw_actions, self.action_q01, self.action_q99)

        # ── Proprio ────────────────────────────────────────────────────────
        raw_state = self.states[global_t]      # (8,)
        norm_proprio = _normalize_bounds_q99(
            raw_state[None], self.proprio_q01, self.proprio_q99
        )[0]                                    # (8,)

        # ── Tokenise (mirrors RLDSBatchTransform exactly) ──────────────────
        current_action  = norm_actions[0]       # (2,)
        future_actions  = norm_actions[1:]      # (C-1, 2)

        future_actions_string  = "".join(self.action_tokenizer(future_actions))
        current_action_string  = self.action_tokenizer(current_action)
        action_chunk_string    = current_action_string + future_actions_string
        action_chunk_len       = len(action_chunk_string)

        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {
                "from": "human",
                "value": f"What action should the robot take to {PUSHT_LANGUAGE_INSTRUCTION}?",
            },
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids_list = self.base_tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True
        ).input_ids
        labels_list = list(input_ids_list)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        labels    = torch.tensor(labels_list,    dtype=torch.long)
        pixel_values = self.image_transform(img_pil)

        # Mask out everything except the action tokens in the label.
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX

        # ── Optical flow ───────────────────────────────────────────────────
        mv_history = self._get_flow_history(global_t, ep_start)
        mv_future  = self._get_flow_future(global_t)

        return {
            "pixel_values": pixel_values,
            "input_ids":    input_ids,
            "labels":       labels,
            "actions":      torch.tensor(norm_actions,  dtype=torch.float32),
            "dataset_name": DATASET_NAME,
            "proprio":      torch.tensor(norm_proprio,  dtype=torch.float32),
            "mv_history":   torch.tensor(mv_history,    dtype=torch.float32),
            "mv_future":    torch.tensor(mv_future,     dtype=torch.float32),
            "episode_idx":  ep_idx,
            "motion_path":  "",         # retained for interface compatibility
            "time_step":    global_t - ep_start,
        }

    # ------------------------------------------------------------------
    # IterableDataset interface
    # ------------------------------------------------------------------

    def __iter__(self):
        rng = np.random.default_rng()
        indices = list(self.valid_indices)
        rng.shuffle(indices)
        for ep_idx, global_t in indices:
            yield self._build_sample(ep_idx, global_t)

    def __len__(self) -> int:
        return len(self.valid_indices)
