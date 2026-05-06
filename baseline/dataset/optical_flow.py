"""
optical_flow.py

Dense optical flow extraction from Push-T RGB frames as a proxy for the
MPEG-4 motion vectors used in the original HiF-VLA pipeline.

Design decisions:
  - Farneback dense flow: captures full-field motion, no invalid sentinels (unlike
    MPEG-4 P-frame vectors which need the -10000 masking scheme).
  - Flow is normalized by image resolution → relative displacement in [roughly -1, 1].
  - Spatial downsampling from 96×96 to 16×16 via bilinear resize (consistent with
    the 16×16 spatial grid used inside MotionVectorProcessor for MPEG-4 vectors).
  - Per-channel (u, v) mean/std normalization computed from training data; stored
    alongside the cache so that eval uses the same statistics.
  - Episode-boundary frames get zero flow (no preceding frame to diff against).
"""

import os
from typing import Optional, Tuple

import cv2
import numpy as np

# Target spatial resolution matching HiF-VLA's motion vector grid
FLOW_H: int = 16
FLOW_W: int = 16


# ---------------------------------------------------------------------------
# Single-frame flow computation
# ---------------------------------------------------------------------------

def compute_frame_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """
    Dense Farneback optical flow between two uint8 grayscale frames.

    Returns
    -------
    np.ndarray
        Shape (2, FLOW_H, FLOW_W) float32.  Channel 0 = horizontal (u),
        channel 1 = vertical (v), both in relative units (divided by frame
        dimensions before resize so that values are roughly in [-1, 1] for
        full-image motions).
    """
    flow_full = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )  # (H, W, 2) float32, pixel-unit displacement

    h, w = flow_full.shape[:2]
    flow_full[:, :, 0] /= w   # normalise to relative units
    flow_full[:, :, 1] /= h

    # Bilinear downsample to 16×16
    flow_small = cv2.resize(flow_full, (FLOW_W, FLOW_H), interpolation=cv2.INTER_LINEAR)
    return flow_small.transpose(2, 0, 1).astype(np.float32)  # (2, H, W)


# ---------------------------------------------------------------------------
# Whole-dataset pre-computation
# ---------------------------------------------------------------------------

def precompute_flow(
    imgs: np.ndarray,
    episode_ends: np.ndarray,
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """
    Pre-compute per-step optical flow for every step in the zarr dataset.

    ``flow[t]`` = flow from ``imgs[t-1]`` → ``imgs[t]`` (within the same episode).
    ``flow[ep_start]`` = zeros for every episode start.

    Parameters
    ----------
    imgs : np.ndarray
        Shape (N, H, W, 3) uint8 RGB.
    episode_ends : np.ndarray
        Cumulative end indices, shape (E,), as stored in ``meta/episode_ends``.
    cache_path : str, optional
        Path to ``.npy`` file.  If it exists the cached array is returned
        immediately; otherwise the result is saved there.

    Returns
    -------
    np.ndarray
        Shape (N, 2, FLOW_H, FLOW_W) float32.
    """
    if cache_path is not None and os.path.exists(cache_path):
        print(f"[optical_flow] Loading cached flow from {cache_path}")
        return np.load(cache_path)

    n_steps = len(imgs)
    flows = np.zeros((n_steps, 2, FLOW_H, FLOW_W), dtype=np.float32)

    ep_starts = np.concatenate([[0], episode_ends[:-1]])
    n_ep = len(episode_ends)

    print(f"[optical_flow] Pre-computing flow for {n_steps} steps across {n_ep} episodes …")
    for ep_i, (ep_s, ep_e) in enumerate(zip(ep_starts, episode_ends)):
        if ep_i % 50 == 0:
            print(f"  episode {ep_i}/{n_ep} (global step {ep_s})")
        for t in range(int(ep_s) + 1, int(ep_e)):
            prev_gray = cv2.cvtColor(imgs[t - 1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(imgs[t], cv2.COLOR_RGB2GRAY)
            flows[t] = compute_frame_flow(prev_gray, curr_gray)

    if cache_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.save(cache_path, flows)
        print(f"[optical_flow] Saved flow cache → {cache_path}")

    return flows


# ---------------------------------------------------------------------------
# Statistics  (computed once from training steps, reused at eval)
# ---------------------------------------------------------------------------

def compute_flow_stats(
    flows: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-channel mean and std of the optical flow.

    Parameters
    ----------
    flows : np.ndarray
        Shape (N, 2, H, W).
    valid_mask : np.ndarray, optional
        Boolean mask of shape (N,) selecting training steps.  If None all
        steps are used.

    Returns
    -------
    mean : np.ndarray  shape (2,)
    std  : np.ndarray  shape (2,)
    """
    if valid_mask is not None:
        data = flows[valid_mask]
    else:
        data = flows
    # data: (N', 2, H, W)
    flat = data.reshape(len(data), 2, -1)          # (N', 2, H*W)
    mean = flat.mean(axis=(0, 2)).astype(np.float32)   # (2,)
    std  = flat.std(axis=(0, 2)).astype(np.float32)    # (2,)
    std  = np.maximum(std, 1e-6)                       # avoid division by zero
    return mean, std


def save_flow_stats(mean: np.ndarray, std: np.ndarray, path: str) -> None:
    """Save flow normalisation statistics as a ``.npz`` file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, mean=mean, std=std)
    print(f"[optical_flow] Saved flow stats → {path}")


def load_flow_stats(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load flow normalisation statistics from a ``.npz`` file."""
    data = np.load(path)
    return data["mean"].astype(np.float32), data["std"].astype(np.float32)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalize_flow(
    flow: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Subtract per-channel mean and divide by std.

    Parameters
    ----------
    flow : np.ndarray
        Shape (..., 2, H, W).
    mean, std : np.ndarray
        Shape (2,).

    Returns
    -------
    np.ndarray  same shape as ``flow``, float32.
    """
    m = mean.reshape((2, 1, 1))
    s = std.reshape((2, 1, 1))
    return ((flow - m) / s).astype(np.float32)


def unnormalize_flow(
    flow: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Inverse of :func:`normalize_flow`."""
    m = mean.reshape((2, 1, 1))
    s = std.reshape((2, 1, 1))
    return (flow * s + m).astype(np.float32)
