#!/usr/bin/env python3
"""
Produce a slim copy of pusht_asym_demos_320.zarr with images downsampled to
96x96x3 (matching pusht_three_goals_demo_vision_320.zarr).

Only the img field is changed. Every other field (action, state,
goal_keypoint, heavy_segment, keypoint, n_contacts) and the
meta/episode_ends array are copied verbatim.

The source zarr is never modified. If the destination already exists the
script aborts rather than overwrite.

Run from the repo root:
    docker compose run --rm dev python resize_asym_zarr.py
"""
import os
import numpy as np
import zarr
import cv2
from tqdm import tqdm


SRC = 'data/pusht_asym_demos_320.zarr'
DST = 'data/pusht_asym_demos_320_96.zarr'
OUT_H, OUT_W = 96, 96


def main():
    if not os.path.isdir(SRC):
        raise SystemExit(f'Source not found: {SRC}')
    if os.path.exists(DST):
        raise SystemExit(f'Refusing to overwrite existing {DST}')

    src = zarr.open(SRC, mode='r')
    dst = zarr.open(DST, mode='w')

    # ---- meta/episode_ends (verbatim) -----------------------------------
    meta_src = src['meta/episode_ends']
    dst.create_group('meta')
    dst['meta'].create_dataset(
        'episode_ends',
        data=meta_src[:],
        chunks=meta_src.chunks,
        dtype=meta_src.dtype,
        compressor=meta_src.compressor,
    )
    print(f'meta/episode_ends: {meta_src.shape} {meta_src.dtype}')

    dst.create_group('data')

    # ---- data/* verbatim (everything except img) -------------------------
    for key in src['data'].keys():
        if key == 'img':
            continue
        arr = src['data'][key]
        print(f'copy  data/{key:<22} {arr.shape} {arr.dtype}')
        dst['data'].create_dataset(
            key,
            data=arr[:],
            chunks=arr.chunks,
            dtype=arr.dtype,
            compressor=arr.compressor,
        )

    # ---- data/img: resize 256 -> 96 in chunks ---------------------------
    img_src = src['data/img']
    N, _, _, C = img_src.shape
    chunk_N = img_src.chunks[0]
    print(f'resize data/img           {img_src.shape} -> (N, {OUT_H}, {OUT_W}, {C})')
    img_dst = dst['data'].zeros(
        'img',
        shape=(N, OUT_H, OUT_W, C),
        chunks=(chunk_N, OUT_H, OUT_W, C),
        dtype=img_src.dtype,
        compressor=img_src.compressor,
    )

    for start in tqdm(range(0, N, chunk_N), desc='resize'):
        end = min(start + chunk_N, N)
        batch = img_src[start:end]
        out = np.empty((end - start, OUT_H, OUT_W, C), dtype=img_src.dtype)
        for i in range(end - start):
            out[i] = cv2.resize(batch[i], (OUT_W, OUT_H),
                                interpolation=cv2.INTER_AREA)
        img_dst[start:end] = out

    print('\nDone.')
    print(f'  src size: {_du(SRC)}')
    print(f'  dst size: {_du(DST)}')


def _du(path: str) -> str:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total < 1024:
            return f'{total:.2f} {unit}'
        total /= 1024
    return f'{total:.2f} TB'


if __name__ == '__main__':
    main()
