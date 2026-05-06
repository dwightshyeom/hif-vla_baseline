"""
test_dataset.py

Sanity-check script for PushTHiFVLADataset (Step 2.5 of the plan).

Validates:
  - All expected keys are present in a sample dict
  - Tensor shapes match expectations
  - No NaN or Inf values in any tensor
  - Optical flow plausibility via multiple visualizations (see --checks flag)

Optical-flow checks produced
──────────────────────────────
  flow_overlay.png            – magnitude heatmap + quiver on the current frame
  raw_farneback.png           – raw Farneback flow DIRECTLY from two zarr frames
                                (completely bypasses the dataset pipeline)
  flow_history_grid.png       – all 8 history frames as magnitude heatmaps
  flow_source_frames.png      – the two source frames that produced a chosen flow
  flow_magnitude_stats.png    – per-step magnitude statistics across a full episode

Usage (without VLA weights — uses dummy tokenizer/transform, fast):
    cd /home/arclab/HIF-VLA_baseline
    python baseline/dataset/test_dataset.py \\
        --zarr_path memory_diffusion_policy/swap_3t_dataset_320

Run *all* flow diagnostics (slower but thorough):
    python baseline/dataset/test_dataset.py \\
        --zarr_path memory_diffusion_policy/swap_3t_dataset_320 \\
        --checks all \\
        --output_dir /tmp/flow_debug

Usage (with real VLA weights — validates tokenized sequences too):
    cd /home/arclab/HIF-VLA_baseline/HiF-VLA
    python ../baseline/dataset/test_dataset.py \\
        --zarr_path ../memory_diffusion_policy/swap_3t_dataset_320 \\
        --vla_path ../openvla_oft/openvla_clean_weights
"""

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# IMPORTANT: inject "pusht" into sys.argv BEFORE any prismatic import.
# prismatic/vla/constants.py detects the robot platform by scanning sys.argv
# at import time. Without this, it defaults to LIBERO (ACTION_DIM=7 instead
# of 2), which causes shape mismatches throughout the dataset.
# ---------------------------------------------------------------------------
if "pusht" not in " ".join(sys.argv).lower():
    sys.argv.append("--_pusht_marker")   # dummy arg; never parsed

import matplotlib
matplotlib.use("Agg")   # headless-safe
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cv2
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = Path(__file__).resolve().parent           # baseline/dataset/
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent                 # project root
_HIF_VLA_ROOT = _PROJECT_ROOT / "HiF-VLA"

for _p in [str(_PROJECT_ROOT), str(_HIF_VLA_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Dummy stubs used when no VLA path is provided
# ---------------------------------------------------------------------------

class _DummyTokenizer:
    model_max_length = 2048
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        # Return a tiny fake encoding so the dataset doesn't crash.
        class _Enc:
            input_ids = [1, 2, 3]
        return _Enc()


class _DummyActionTokenizer:
    def __call__(self, action):
        if action.ndim == 1:
            return "A" * len(action)
        return "".join("A" * action.shape[-1] for _ in range(len(action)))


class _DummyPromptBuilder:
    def __init__(self, *a, **kw):
        self._prompt = ""

    def add_turn(self, role, text):
        self._prompt += text

    def get_prompt(self):
        return self._prompt


def _dummy_image_transform(pil_img):
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))   # (3, H, W)


# ---------------------------------------------------------------------------
# Expected shapes (PUSHT_CONSTANTS: NUM_ACTIONS_CHUNK=8, ACTION_DIM=2, PROPRIO_DIM=8)
# ---------------------------------------------------------------------------
NUM_ACTIONS_CHUNK = 8
ACTION_DIM        = 2
PROPRIO_DIM       = 8
HISTORY_LENGTH    = 8
FLOW_H, FLOW_W    = 16, 16

EXPECTED_SHAPES = {
    "actions":    (NUM_ACTIONS_CHUNK, ACTION_DIM),
    "proprio":    (PROPRIO_DIM,),
    "mv_history": (HISTORY_LENGTH, 2, FLOW_H, FLOW_W),
    "mv_future":  (NUM_ACTIONS_CHUNK, 2, FLOW_H, FLOW_W),
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def check_shapes(sample: dict) -> None:
    print("\n--- Shape check ---")
    for key, expected in EXPECTED_SHAPES.items():
        tensor = sample[key]
        actual = tuple(tensor.shape)
        status = "OK" if actual == expected else f"MISMATCH (expected {expected})"
        print(f"  {key:20s}: {str(actual):30s} {status}")

    # pixel_values can vary by VLA processor; just print its shape.
    pv = sample.get("pixel_values")
    if pv is not None:
        print(f"  {'pixel_values':20s}: {str(tuple(pv.shape))}")

    # input_ids / labels: just check they're 1-D int64.
    for key in ("input_ids", "labels"):
        t = sample.get(key)
        if t is not None:
            ok = t.dtype == torch.long and t.ndim == 1
            print(f"  {key:20s}: shape={tuple(t.shape)} dtype={t.dtype}  {'OK' if ok else 'BAD dtype/ndim'}")


def check_nan_inf(sample: dict) -> None:
    print("\n--- NaN / Inf check ---")
    all_ok = True
    for key, val in sample.items():
        if not isinstance(val, torch.Tensor):
            continue
        if val.is_floating_point():
            has_nan = torch.isnan(val).any().item()
            has_inf = torch.isinf(val).any().item()
            if has_nan or has_inf:
                print(f"  {key}: NaN={has_nan}  Inf={has_inf}  *** BAD ***")
                all_ok = False
    if all_ok:
        print("  All tensors clean (no NaN / Inf).")


# ---------------------------------------------------------------------------
# Helper: tensor → uint8 HWC image
# ---------------------------------------------------------------------------

def _tensor_to_img(pv) -> np.ndarray:
    """Convert pixel_values tensor (CHW float or HWC float/uint8) to uint8 HWC."""
    if isinstance(pv, torch.Tensor):
        pv = pv.numpy()
    if pv.ndim == 3 and pv.shape[0] in (1, 3):   # CHW → HWC
        pv = pv.transpose(1, 2, 0)
    if pv.dtype != np.uint8:
        pv = np.clip(pv, 0, 1)
        pv = (pv * 255).astype(np.uint8)
    return pv


# ---------------------------------------------------------------------------
# Check 1: magnitude heatmap + quiver overlay (fixed scale)
# ---------------------------------------------------------------------------

def plot_flow_overlay(sample: dict, out_path: str) -> None:
    """
    Side-by-side: current RGB | magnitude heatmap | quiver on image.

    The quiver uses **relative-unit** flow values (before mean/std
    normalisation) if the dataset exposes them, otherwise falls back to
    the normalised mv_history values but scales arrows proportionally to
    their actual magnitude so that large arrows really mean large motion.
    """
    img = _tensor_to_img(sample["pixel_values"])
    img_h, img_w = img.shape[:2]

    # Last historical flow frame: mv_history[-1], shape (2, 16, 16)
    last_flow = sample["mv_history"][-1].numpy()   # (2, H, W)
    u = last_flow[0]   # horizontal
    v = last_flow[1]   # vertical

    mag = np.sqrt(u ** 2 + v ** 2)

    H, W = u.shape
    xs = np.linspace(0, img_w - 1, W)
    ys = np.linspace(0, img_h - 1, H)
    X, Y = np.meshgrid(xs, ys)

    # Adaptive scale: arrows whose length = mean_mag occupy ~5 % of image width.
    mean_mag = float(mag.mean()) + 1e-6
    arrow_scale = mean_mag / (0.05 * img_w) + 1e-9   # data-units per pixel

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 0: raw image
    axes[0].imshow(img)
    axes[0].set_title("Current frame")
    axes[0].axis("off")

    # Panel 1: magnitude heatmap
    axes[1].imshow(img)
    hm = axes[1].imshow(
        mag,
        extent=[0, img_w, img_h, 0],   # match image coords
        cmap="hot", alpha=0.6,
        vmin=0, vmax=max(mag.max(), 1e-3),
    )
    fig.colorbar(hm, ax=axes[1], fraction=0.046, label="||flow||")
    axes[1].set_title("Flow magnitude (normalised units)")
    axes[1].axis("off")

    # Panel 2: quiver on image (adaptive scale)
    axes[2].imshow(img)
    axes[2].quiver(
        X, Y, u, -v,         # -v because matplotlib y-axis is flipped
        color="lime",
        scale=arrow_scale,
        scale_units="xy",    # 'xy' → one data unit = one pixel
        width=0.004,
        headwidth=4,
    )
    axes[2].set_title(f"Quiver (normalised; mean_mag={mean_mag:.3f})")
    axes[2].axis("off")

    fig.suptitle(
        f"ep={sample.get('episode_idx','?')}  t={sample.get('time_step','?')}  "
        f"(mv_history[-1], normalised flow)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Check 2: raw Farneback directly from zarr (no dataset pipeline)
# ---------------------------------------------------------------------------

def plot_raw_farneback(zarr_path: str, episode_idx: int, start_step_in_ep: int,
                        n_pairs: int, out_path: str) -> None:
    """
    Loads consecutive raw frames directly from the zarr store, runs
    Farneback (same parameters as optical_flow.py), and visualises:

      col 0: frame t
      col 1: frame t+1
      col 2: magnitude heatmap (pixel-unit, NOT relative, NOT normalised)
      col 3: quiver overlaid on frame t+1 (arrows show pixel displacement)

    This is the cleanest sanity-check because it completely bypasses the
    dataset, normalisation, and caching layers.

    A plausible result: arrows should be small (<5 px) for static areas
    and point in the direction of moving objects/agent.
    """
    import zarr

    store = zarr.open(str(zarr_path), "r")
    imgs         = store["data"]["img"]                  # (N, H, W, 3)
    episode_ends = store["meta"]["episode_ends"][:]      # (E,)
    ep_starts    = np.concatenate([[0], episode_ends[:-1]])

    ep_start = int(ep_starts[episode_idx])
    ep_end   = int(episode_ends[episode_idx])

    t0 = ep_start + start_step_in_ep
    t0 = min(t0, ep_end - n_pairs - 1)   # clamp to valid range
    indices = range(t0, t0 + n_pairs + 1)

    frames = [np.array(imgs[i]) for i in indices]   # list of (H, W, 3) uint8

    fig, axes = plt.subplots(n_pairs, 4, figsize=(16, 3.5 * n_pairs))
    if n_pairs == 1:
        axes = axes[np.newaxis, :]   # always 2-D

    for row, (t_prev, t_curr) in enumerate(zip(indices, list(indices)[1:])):
        f_prev = frames[row]
        f_curr = frames[row + 1]

        gray_prev = cv2.cvtColor(f_prev, cv2.COLOR_RGB2GRAY)
        gray_curr = cv2.cvtColor(f_curr, cv2.COLOR_RGB2GRAY)

        flow_raw = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )  # (H, W, 2) — raw pixel-unit displacement

        H, W = flow_raw.shape[:2]
        mag = np.sqrt(flow_raw[:, :, 0] ** 2 + flow_raw[:, :, 1] ** 2)

        # Sparse quiver on a 16×16 grid
        step_y = max(H // 16, 1)
        step_x = max(W // 16, 1)
        ys = np.arange(step_y // 2, H, step_y)
        xs = np.arange(step_x // 2, W, step_x)
        Xq, Yq = np.meshgrid(xs, ys)
        Uq = flow_raw[Yq, Xq, 0]
        Vq = flow_raw[Yq, Xq, 1]

        mean_mag = float(mag.mean()) + 1e-6
        arrow_scale = mean_mag / (0.05 * W) + 1e-9

        ax = axes[row]

        ax[0].imshow(f_prev)
        ax[0].set_title(f"Frame {t_prev} (prev)")
        ax[0].axis("off")

        ax[1].imshow(f_curr)
        ax[1].set_title(f"Frame {t_curr} (curr)")
        ax[1].axis("off")

        ax[2].imshow(f_curr)
        hm = ax[2].imshow(mag, cmap="hot", alpha=0.6,
                          vmin=0, vmax=max(mag.max(), 0.5),
                          extent=[0, W, H, 0])
        fig.colorbar(hm, ax=ax[2], fraction=0.046, label="px/frame")
        ax[2].set_title(f"Magnitude (px)  mean={mean_mag:.2f}")
        ax[2].axis("off")

        ax[3].imshow(f_curr)
        ax[3].quiver(
            Xq, Yq, Uq, -Vq,
            color="cyan", scale=arrow_scale, scale_units="xy",
            width=0.003, headwidth=4,
        )
        ax[3].set_title("Quiver (raw px displacement)")
        ax[3].axis("off")

    fig.suptitle(
        f"RAW Farneback — ep {episode_idx}, steps {list(indices)[0]}–{list(indices)[-1]}\n"
        f"(no normalisation, no downsampling, directly from zarr)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Check 3: all 8 history flow frames as magnitude heatmaps
# ---------------------------------------------------------------------------

def plot_flow_history_grid(sample: dict, out_path: str) -> None:
    """
    Show the full mv_history as a 2×8 grid:
      Top row   – frame thumbnails (if pixel_values is the right size)
      Bottom row – per-frame magnitude heatmaps (normalised flow units)

    Frame 0 should be near-zero (episode start or far in the past with
    zero-padding); later frames should show increasing motion near moving
    objects.
    """
    mv = sample["mv_history"].numpy()   # (T, 2, H, W)
    T  = mv.shape[0]

    img = _tensor_to_img(sample["pixel_values"])

    fig, axes = plt.subplots(2, T, figsize=(2.5 * T, 5))

    mags = [np.sqrt(mv[t, 0] ** 2 + mv[t, 1] ** 2) for t in range(T)]
    global_max = max(float(m.max()) for m in mags) + 1e-6

    for t in range(T):
        # Top row: current frame (same image for all — we don't store history frames)
        axes[0, t].imshow(img)
        axes[0, t].set_title(f"t−{T - 1 - t}", fontsize=8)
        axes[0, t].axis("off")

        # Bottom row: magnitude heatmap
        hm = axes[1, t].imshow(mags[t], cmap="hot",
                               vmin=0, vmax=global_max, aspect="auto")
        axes[1, t].set_title(f"||f|| mean={mags[t].mean():.3f}", fontsize=7)
        axes[1, t].axis("off")

    fig.colorbar(hm, ax=axes[1, :], fraction=0.02, label="||flow|| (norm. units)")
    fig.suptitle(
        f"mv_history — ep={sample.get('episode_idx','?')} t={sample.get('time_step','?')}\n"
        f"Left=oldest (t−{T-1}), Right=most recent (t−0).",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Check 4: episode-boundary zero-flow sanity check
# ---------------------------------------------------------------------------

def check_episode_boundary_flow(dataset, n_episodes: int = 5) -> None:
    """
    Verify that the first valid step of the first N episodes has
    mv_history frames that are zero (or very small) at the episode start.

    Episode starts are zero-padded, so mv_history[0] for the first sample
    in an episode should be all zeros.  If it isn't, either the episode
    boundary detection or the zero-padding logic is broken.
    """
    print(f"\n--- Episode-boundary zero-flow check (first {n_episodes} episodes) ---")

    ep_indices = dataset.valid_indices   # list of (ep_idx, global_t) tuples

    # Group by episode
    from collections import defaultdict
    by_ep = defaultdict(list)
    for ep_idx, global_t in ep_indices:
        by_ep[ep_idx].append(global_t)

    checked = 0
    for ep_idx in sorted(by_ep.keys())[:n_episodes]:
        steps = sorted(by_ep[ep_idx])
        first_t = steps[0]

        sample = dataset._build_sample(ep_idx, first_t)
        mv = sample["mv_history"].numpy()   # (T, 2, H, W)

        # Frame 0 of history is the furthest in the past — should be zero-padded
        zero_frame_mag = np.abs(mv[0]).mean()
        # Count how many history frames are non-zero
        nonzero_frames = sum(np.abs(mv[t]).mean() > 1e-4 for t in range(mv.shape[0]))

        status = "OK" if zero_frame_mag < 1e-3 else "WARNING – non-zero!"
        print(
            f"  ep {ep_idx:4d}  first_step={first_t:5d}  "
            f"mv_history[0] mean_abs={zero_frame_mag:.5f}  {status}  "
            f"non-zero frames: {nonzero_frames}/{mv.shape[0]}"
        )
        checked += 1

    print(f"  Checked {checked} episodes.")


# ---------------------------------------------------------------------------
# Check 5: per-step flow magnitude statistics over a full episode
# ---------------------------------------------------------------------------

def plot_episode_flow_stats(zarr_path: str, episode_idx: int, out_path: str) -> None:
    """
    Compute and plot raw Farneback flow magnitude for every step in one
    episode.  This answers: "Is there any motion at all and does it correlate
    with when the agent/block should be moving?"

    Three sub-plots:
      - Mean flow magnitude (px/frame) per step
      - Max  flow magnitude (px/frame) per step
      - Histogram of all per-pixel magnitudes in the episode
    """
    import zarr

    store = zarr.open(str(zarr_path), "r")
    imgs         = store["data"]["img"][:]
    episode_ends = store["meta"]["episode_ends"][:]
    ep_starts    = np.concatenate([[0], episode_ends[:-1]])

    ep_start = int(ep_starts[episode_idx])
    ep_end   = int(episode_ends[episode_idx])
    n_steps  = ep_end - ep_start

    print(f"  Computing raw flow for episode {episode_idx} "
          f"(steps {ep_start}–{ep_end}, {n_steps} frames) …")

    mean_mags = np.zeros(n_steps, dtype=np.float32)
    max_mags  = np.zeros(n_steps, dtype=np.float32)
    all_mags  = []

    for i in range(1, n_steps):
        gray_prev = cv2.cvtColor(imgs[ep_start + i - 1], cv2.COLOR_RGB2GRAY)
        gray_curr = cv2.cvtColor(imgs[ep_start + i],     cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        mean_mags[i] = float(mag.mean())
        max_mags[i]  = float(mag.max())
        all_mags.append(mag.ravel())

    all_mags_flat = np.concatenate(all_mags) if all_mags else np.array([0.0])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    steps = np.arange(n_steps)
    axes[0].plot(steps, mean_mags, label="mean", color="steelblue")
    axes[0].set_xlabel("Step within episode")
    axes[0].set_ylabel("Flow magnitude (px/frame)")
    axes[0].set_title("Mean flow magnitude per step")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, max_mags, label="max", color="tomato")
    axes[1].set_xlabel("Step within episode")
    axes[1].set_ylabel("Flow magnitude (px/frame)")
    axes[1].set_title("Max flow magnitude per step")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(all_mags_flat, bins=80, color="steelblue", edgecolor="none")
    axes[2].set_xlabel("Per-pixel magnitude (px/frame)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Distribution of all flow magnitudes")
    axes[2].grid(True, alpha=0.3)

    overall_mean = float(mean_mags[1:].mean())
    overall_max  = float(max_mags.max())
    print(
        f"  Episode {episode_idx}: mean_mag={overall_mean:.3f} px/frame  "
        f"max_mag={overall_max:.3f} px/frame  "
        f"(values near 0 → static; >2 → clear motion)"
    )

    fig.suptitle(
        f"Raw Farneback statistics — episode {episode_idx}  "
        f"(mean={overall_mean:.2f}, max={overall_max:.2f} px/frame)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Check 6: show the source frame pair that produced a dataset sample's flow
# ---------------------------------------------------------------------------

def plot_flow_source_frames(zarr_path: str, sample: dict, out_path: str) -> None:
    """
    For the current time_step in the sample, load the actual raw frames
    that produced mv_history[-1] (the most recent history flow) and
    display them side by side next to the normalised flow magnitude.

    This makes it easy to confirm visually that the arrows correspond to
    moving objects.
    """
    import zarr

    store = zarr.open(str(zarr_path), "r")
    imgs         = store["data"]["img"]
    episode_ends = store["meta"]["episode_ends"][:]
    ep_starts    = np.concatenate([[0], episode_ends[:-1]])

    ep_idx = sample.get("episode_idx", 0)
    t_cur  = sample.get("time_step", 1)

    ep_start = int(ep_starts[ep_idx])
    global_t = ep_start + t_cur

    # mv_history[-1] is the flow from global_t-1 → global_t
    t_prev = max(global_t - 1, ep_start)

    f_prev = np.array(imgs[t_prev])
    f_curr = np.array(imgs[global_t])

    gray_prev = cv2.cvtColor(f_prev, cv2.COLOR_RGB2GRAY)
    gray_curr = cv2.cvtColor(f_curr, cv2.COLOR_RGB2GRAY)
    flow_raw = cv2.calcOpticalFlowFarneback(
        gray_prev, gray_curr, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag_raw = np.sqrt(flow_raw[:, :, 0] ** 2 + flow_raw[:, :, 1] ** 2)

    # Normalised flow from the sample for comparison
    last_flow_norm = sample["mv_history"][-1].numpy()   # (2, 16, 16)
    mag_norm = np.sqrt(last_flow_norm[0] ** 2 + last_flow_norm[1] ** 2)

    H, W = f_curr.shape[:2]
    step_y = max(H // 16, 1)
    step_x = max(W // 16, 1)
    ys = np.arange(step_y // 2, H, step_y)
    xs = np.arange(step_x // 2, W, step_x)
    Xq, Yq = np.meshgrid(xs, ys)
    Uq = flow_raw[Yq, Xq, 0]
    Vq = flow_raw[Yq, Xq, 1]
    mean_mag = float(mag_raw.mean()) + 1e-6
    arrow_scale = mean_mag / (0.05 * W) + 1e-9

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].imshow(f_prev)
    axes[0].set_title(f"Frame {t_prev} (prev)")
    axes[0].axis("off")

    axes[1].imshow(f_curr)
    axes[1].set_title(f"Frame {global_t} (curr)")
    axes[1].axis("off")

    axes[2].imshow(f_curr)
    axes[2].imshow(mag_raw, cmap="hot", alpha=0.6,
                   vmin=0, vmax=max(mag_raw.max(), 0.5),
                   extent=[0, W, H, 0])
    axes[2].quiver(Xq, Yq, Uq, -Vq, color="cyan",
                   scale=arrow_scale, scale_units="xy",
                   width=0.003, headwidth=4)
    axes[2].set_title(f"Raw flow (px)  mean={mean_mag:.2f}")
    axes[2].axis("off")

    axes[3].imshow(f_curr)
    axes[3].imshow(
        mag_norm,
        cmap="hot", alpha=0.6, vmin=0,
        vmax=max(float(mag_norm.max()), 1e-3),
        extent=[0, W, H, 0],
    )
    axes[3].set_title(f"Dataset flow (normalised)\nmean={float(mag_norm.mean()):.3f}")
    axes[3].axis("off")

    fig.suptitle(
        f"Source frames for mv_history[-1]  "
        f"ep={ep_idx}  t_prev={t_prev}  t_curr={global_t}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PushTHiFVLADataset unit test")
    parser.add_argument(
        "--zarr_path",
        required=True,
        help="Path to swap_3t_dataset_320 zarr store",
    )
    parser.add_argument(
        "--vla_path",
        default=None,
        help="(Optional) Path to openvla weights for real tokenizer/processor",
    )
    parser.add_argument(
        "--history_length", type=int, default=HISTORY_LENGTH,
        help="History length passed to PushTHiFVLADataset (default: 8)",
    )
    parser.add_argument(
        "--flow_cache_path", default=None,
        help="Path to pre-computed flow cache .npy (optional)",
    )
    parser.add_argument(
        "--output_dir", default=".",
        help="Directory to save diagnostic PNGs (default: current dir)",
    )
    parser.add_argument(
        "--episode_idx", type=int, default=0,
        help="Episode index for raw Farneback / stats plots (default: 0)",
    )
    parser.add_argument(
        "--start_step", type=int, default=5,
        help="First step within episode for raw Farneback visualization (default: 5)",
    )
    parser.add_argument(
        "--n_pairs", type=int, default=3,
        help="Number of consecutive frame pairs to show in raw Farneback plot (default: 3)",
    )
    parser.add_argument(
        "--checks",
        default="all",
        help=(
            "Comma-separated list of checks to run, or 'all'.  "
            "Available: shapes,nan,overlay,raw_farneback,history_grid,"
            "source_frames,boundary,ep_stats"
        ),
    )
    # Absorb the dummy pusht-marker arg injected at the top of this script.
    parser.add_argument("--_pusht_marker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    active = set()
    if args.checks.strip().lower() == "all":
        active = {
            "shapes", "nan", "overlay", "raw_farneback",
            "history_grid", "source_frames", "boundary", "ep_stats",
        }
    else:
        active = {c.strip() for c in args.checks.split(",")}

    print(f"Active checks: {sorted(active)}")

    # ------------------------------------------------------------------
    # Build tokenizer / processor
    # ------------------------------------------------------------------
    if args.vla_path is not None:
        print(f"Loading processor from {args.vla_path} …")
        from transformers import AutoProcessor
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.processing_prismatic import (
            PrismaticImageProcessor,
            PrismaticProcessor,
        )
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        processor = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)

        from prismatic.vla.action_tokenizer import ActionTokenizer
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        action_tokenizer   = ActionTokenizer(processor.tokenizer)
        base_tokenizer     = processor.tokenizer
        image_transform    = processor.image_processor.apply_transform
        prompt_builder_fn  = PurePromptBuilder
    else:
        print("No --vla_path given; using dummy tokenizer/transform stubs.")
        action_tokenizer   = _DummyActionTokenizer()
        base_tokenizer     = _DummyTokenizer()
        image_transform    = _dummy_image_transform
        prompt_builder_fn  = _DummyPromptBuilder

    # ------------------------------------------------------------------
    # Instantiate dataset (needed for most checks)
    # ------------------------------------------------------------------
    need_dataset = active & {"shapes", "nan", "overlay", "history_grid",
                              "source_frames", "boundary"}
    sample = None
    dataset = None

    if need_dataset:
        from baseline.dataset.pusht_hifvla_dataset import PushTHiFVLADataset

        dataset = PushTHiFVLADataset(
            zarr_path=args.zarr_path,
            action_tokenizer=action_tokenizer,
            base_tokenizer=base_tokenizer,
            image_transform=image_transform,
            prompt_builder_fn=prompt_builder_fn,
            history_length=args.history_length,
            flow_cache_path=args.flow_cache_path,
            train=True,
        )

        print(f"\nDataset length: {len(dataset)} valid (episode, step) pairs")
        assert len(dataset) > 0, "Dataset is empty — check zarr_path!"

        print("\nFetching first sample …")
        sample = next(iter(dataset))
        print(f"\n  episode_idx = {sample['episode_idx']}")
        print(f"  time_step   = {sample['time_step']}")
        print(f"  dataset_name= {sample['dataset_name']}")

    # ------------------------------------------------------------------
    # Value range summary (always printed when dataset loaded)
    # ------------------------------------------------------------------
    if sample is not None:
        print("\n--- Value range summary ---")
        for key in ("actions", "proprio", "mv_history", "mv_future"):
            t = sample[key]
            print(
                f"  {key:20s}: min={t.min():.3f}  max={t.max():.3f}  "
                f"mean={t.float().mean():.3f}  std={t.float().std():.3f}"
            )

    # ------------------------------------------------------------------
    # Shape checks
    # ------------------------------------------------------------------
    if "shapes" in active and sample is not None:
        check_shapes(sample)

    # ------------------------------------------------------------------
    # NaN / Inf checks
    # ------------------------------------------------------------------
    if "nan" in active and sample is not None:
        check_nan_inf(sample)

    # ------------------------------------------------------------------
    # Check 1: magnitude heatmap + quiver overlay
    # ------------------------------------------------------------------
    if "overlay" in active and sample is not None:
        print("\n[overlay] magnitude heatmap + quiver …")
        plot_flow_overlay(sample, os.path.join(args.output_dir, "flow_overlay.png"))

    # ------------------------------------------------------------------
    # Check 2: raw Farneback directly from zarr
    # ------------------------------------------------------------------
    if "raw_farneback" in active:
        print("\n[raw_farneback] direct Farneback from zarr frames …")
        plot_raw_farneback(
            zarr_path=args.zarr_path,
            episode_idx=args.episode_idx,
            start_step_in_ep=args.start_step,
            n_pairs=args.n_pairs,
            out_path=os.path.join(args.output_dir, "raw_farneback.png"),
        )

    # ------------------------------------------------------------------
    # Check 3: all 8 history flow frames as magnitude heatmaps
    # ------------------------------------------------------------------
    if "history_grid" in active and sample is not None:
        print("\n[history_grid] plotting all history flow magnitudes …")
        plot_flow_history_grid(sample, os.path.join(args.output_dir, "flow_history_grid.png"))

    # ------------------------------------------------------------------
    # Check 4: episode-boundary zero-flow check
    # ------------------------------------------------------------------
    if "boundary" in active and dataset is not None:
        check_episode_boundary_flow(dataset, n_episodes=5)

    # ------------------------------------------------------------------
    # Check 5: per-step flow magnitude statistics over a full episode
    # ------------------------------------------------------------------
    if "ep_stats" in active:
        print(f"\n[ep_stats] computing per-step stats for episode {args.episode_idx} …")
        plot_episode_flow_stats(
            zarr_path=args.zarr_path,
            episode_idx=args.episode_idx,
            out_path=os.path.join(args.output_dir, "flow_magnitude_stats.png"),
        )

    # ------------------------------------------------------------------
    # Check 6: source frames + raw vs normalised flow
    # ------------------------------------------------------------------
    if "source_frames" in active and sample is not None:
        print("\n[source_frames] loading source frames from zarr …")
        plot_flow_source_frames(
            zarr_path=args.zarr_path,
            sample=sample,
            out_path=os.path.join(args.output_dir, "flow_source_frames.png"),
        )

    print("\nAll checks complete.\n")
    print(f"Outputs written to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
