"""
convert_robosuite_to_zarr.py

Converts a robosuite HDF5 demonstration dataset (e.g. from
fruit_swap_vision_tele.py) into the Zarr store format expected by
PushTHiFVLADataset / HiF-VLA training.

HDF5 source layout (one demo):
    data/demo_N/
        actions                    (T, 7)  float64  – OSC_POSE delta [Δpos(3), Δrot_aa(3), grip(1)]
        obs/
            agentview_image        (T, H, W, 3) uint8
            robot0_eef_pos         (T, 3)  float64
            robot0_eef_quat        (T, 4)  float64
            robot0_gripper_qpos    (T, 2)  float64
    mask/
        train   (N_train,)  bytes  – e.g. b"demo_0", b"demo_5", …
        valid   (N_val,)    bytes

Zarr output layout:
    data/
        img              (N_total, img_size, img_size, 3)  uint8
        action           (N_total, 7)                      float32
        state            (N_total, 8)                      float32
    meta/
        episode_ends     (E,)                              int64

The proprio state (8-D) is:
    robot0_eef_pos  (3)  +  robot0_eef_quat  (4)  +  robot0_gripper_qpos[:,0]  (1)

Usage:
    python baseline/dataset/convert_robosuite_to_zarr.py \\
        --input  /path/to/demo.hdf5 \\
        --output /path/to/fruit_swap_zarr \\
        [--image-size 96] \\
        [--image-key agentview_image] \\
        [--no-mask]   # ignore mask/train|valid, process all demos alphabetically
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import zarr
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_demo_keys(mask_dataset) -> list:
    """Decode a bytes array of demo keys like b'demo_0' → ['demo_0', …]."""
    return [k.decode("utf-8") if isinstance(k, bytes) else k for k in mask_dataset[:]]


def _resize_image(img_hwc: np.ndarray, size: int) -> np.ndarray:
    """Resize a uint8 (H, W, 3) image to (size, size, 3) using PIL LANCZOS."""
    pil = Image.fromarray(img_hwc)
    pil = pil.resize((size, size), Image.LANCZOS)
    return np.asarray(pil, dtype=np.uint8)


def _extract_demo(
    demo_grp: h5py.Group,
    image_key: str,
    img_size: int,
) -> dict:
    """
    Extract one demo from the HDF5 group and return arrays:
        img     (T, img_size, img_size, 3) uint8
        action  (T, 7)  float32
        state   (T, 8)  float32
    """
    T = demo_grp["actions"].shape[0]

    # ── Actions ──────────────────────────────────────────────────────────────
    actions = np.asarray(demo_grp["actions"], dtype=np.float32)   # (T, 7)

    # ── Proprio state ─────────────────────────────────────────────────────────
    eef_pos   = np.asarray(demo_grp["obs/robot0_eef_pos"],         dtype=np.float32)   # (T, 3)
    eef_quat  = np.asarray(demo_grp["obs/robot0_eef_quat"],        dtype=np.float32)   # (T, 4)
    grip_qpos = np.asarray(demo_grp["obs/robot0_gripper_qpos"],    dtype=np.float32)   # (T, 2)
    state = np.concatenate([eef_pos, eef_quat, grip_qpos[:, :1]], axis=1)  # (T, 8)

    # ── Images ────────────────────────────────────────────────────────────────
    raw_imgs = demo_grp[f"obs/{image_key}"]   # HDF5 dataset (T, H, W, 3)
    if raw_imgs.shape[1] == img_size and raw_imgs.shape[2] == img_size:
        imgs = np.asarray(raw_imgs, dtype=np.uint8)
    else:
        imgs = np.stack(
            [_resize_image(raw_imgs[t], img_size) for t in range(T)],
            axis=0,
        )  # (T, img_size, img_size, 3)

    return {"img": imgs, "action": actions, "state": state}


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def convert(
    hdf5_path: str,
    zarr_path: str,
    image_size: int = 96,
    image_key: str = "agentview_image",
    use_mask: bool = True,
) -> None:
    """
    Convert robosuite HDF5 → Zarr.

    Parameters
    ----------
    hdf5_path  : path to the source HDF5 file
    zarr_path  : destination directory for the Zarr store
    image_size : target image resolution (both H and W)
    image_key  : which camera obs to use (default: agentview_image)
    use_mask   : if True, order demos as train-split first then valid-split;
                 if False, sort all demos alphabetically
    """
    hdf5_path = Path(hdf5_path)
    zarr_path = Path(zarr_path)

    print(f"Opening HDF5 : {hdf5_path}")
    with h5py.File(hdf5_path, "r") as f:
        data_grp = f["data"]
        all_demo_keys = sorted(
            data_grp.keys(), key=lambda k: int(k.split("_")[-1])
        )
        print(f"Found {len(all_demo_keys)} demos: {all_demo_keys[:5]}{'…' if len(all_demo_keys) > 5 else ''}")

        # ── Determine ordering: train first, valid after ─────────────────────
        if use_mask and "mask" in f:
            train_keys = _decode_demo_keys(f["mask/train"])
            valid_keys = _decode_demo_keys(f["mask/valid"])
            # keep only keys that actually exist in data/
            train_keys = [k for k in train_keys if k in data_grp]
            valid_keys = [k for k in valid_keys if k in data_grp]
            ordered_keys = train_keys + valid_keys
            leftover = [k for k in all_demo_keys if k not in set(ordered_keys)]
            if leftover:
                print(
                    f"[WARN] {len(leftover)} demos not in mask/train or mask/valid; "
                    "appending at the end."
                )
                ordered_keys += leftover
            print(
                f"Using mask split: {len(train_keys)} train + "
                f"{len(valid_keys)} valid demos"
            )
        else:
            ordered_keys = list(all_demo_keys)
            print(f"No mask used; processing {len(ordered_keys)} demos alphabetically.")

        # ── First pass: check image key exists ───────────────────────────────
        first_demo = data_grp[ordered_keys[0]]
        if f"obs/{image_key}" not in first_demo:
            available = list(first_demo["obs"].keys())
            raise KeyError(
                f"Image key 'obs/{image_key}' not found in demo '{ordered_keys[0]}'. "
                f"Available obs keys: {available}"
            )
        orig_shape = first_demo[f"obs/{image_key}"].shape
        print(f"Source images: {orig_shape[1]}×{orig_shape[2]} → target {image_size}×{image_size}")

        # ── Accumulate all demos ──────────────────────────────────────────────
        all_imgs    = []
        all_actions = []
        all_states  = []
        episode_ends = []
        cumulative  = 0

        for i, key in enumerate(ordered_keys):
            demo_data = _extract_demo(data_grp[key], image_key, image_size)
            T = demo_data["img"].shape[0]
            all_imgs.append(demo_data["img"])
            all_actions.append(demo_data["action"])
            all_states.append(demo_data["state"])
            cumulative += T
            episode_ends.append(cumulative)
            if (i + 1) % 10 == 0 or (i + 1) == len(ordered_keys):
                print(f"  Processed {i + 1}/{len(ordered_keys)} demos  ({cumulative} steps total)")

    imgs    = np.concatenate(all_imgs,    axis=0)  # (N, img_size, img_size, 3)
    actions = np.concatenate(all_actions, axis=0)  # (N, 7)
    states  = np.concatenate(all_states,  axis=0)  # (N, 8)
    ep_ends = np.array(episode_ends, dtype=np.int64)

    print(f"\nTotal steps : {imgs.shape[0]}")
    print(f"Episodes    : {len(ep_ends)}")
    print(f"imgs shape  : {imgs.shape}")
    print(f"action shape: {actions.shape}")
    print(f"state shape : {states.shape}")

    # ── Write Zarr ────────────────────────────────────────────────────────────
    print(f"\nWriting Zarr to {zarr_path} …")
    zarr_path.mkdir(parents=True, exist_ok=True)

    store = zarr.open(str(zarr_path), mode="w")

    data = store.require_group("data")
    meta = store.require_group("meta")

    data.array(
        "img",
        imgs,
        dtype="uint8",
        chunks=(min(100, len(imgs)), image_size, image_size, 3),
        compressor=zarr.Blosc(cname="lz4", clevel=5),
    )
    data.array(
        "action",
        actions,
        dtype="float32",
        chunks=(min(1000, len(actions)), actions.shape[1]),
    )
    data.array(
        "state",
        states,
        dtype="float32",
        chunks=(min(1000, len(states)), states.shape[1]),
    )
    meta.array(
        "episode_ends",
        ep_ends,
        dtype="int64",
        chunks=(len(ep_ends),),
    )

    print("Done.")
    print(f"\nZarr store ready at: {zarr_path}")
    print("\nArray summary:")
    print(f"  data/img           {store['data/img'].shape}  {store['data/img'].dtype}")
    print(f"  data/action        {store['data/action'].shape}  {store['data/action'].dtype}")
    print(f"  data/state         {store['data/state'].shape}  {store['data/state'].dtype}")
    print(f"  meta/episode_ends  {store['meta/episode_ends'].shape}  {store['meta/episode_ends'].dtype}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert robosuite FruitSwap HDF5 demos → HiF-VLA Zarr store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the source HDF5 file (e.g. demo.hdf5).",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Destination directory for the Zarr store.",
    )
    parser.add_argument(
        "--image-size", type=int, default=96,
        help="Target image resolution (both H and W). HiF-VLA uses 96.",
    )
    parser.add_argument(
        "--image-key", default="agentview_image",
        choices=["agentview_image", "robot0_eye_in_hand_image"],
        help="Which camera observation to store as the main image.",
    )
    parser.add_argument(
        "--no-mask", action="store_true",
        help=(
            "Ignore mask/train and mask/valid splits. "
            "Process all demos in alphabetical key order."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convert(
        hdf5_path=args.input,
        zarr_path=args.output,
        image_size=args.image_size,
        image_key=args.image_key,
        use_mask=not args.no_mask,
    )
