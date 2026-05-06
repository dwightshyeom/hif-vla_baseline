#!/usr/bin/env python3
"""
Save the final frame of every episode in
data/pusht_2d_friction_demos_320_96.zarr as a PNG.

Output directory: data/pusht_2d_friction_demos_320_96_last_frames/
File names:       ep_0000.png, ep_0001.png, ...

Run from the repo root:
    docker compose run --rm dev python plot_friction_episode_last_frames.py
"""
import os
import numpy as np
import zarr
import cv2
from tqdm import tqdm


SRC = 'data/pusht_2d_friction_demos_320_96.zarr'
OUT = 'data/pusht_2d_friction_demos_320_96_last_frames'


def main():
    if not os.path.isdir(SRC):
        raise SystemExit(f'Source not found: {SRC}')
    os.makedirs(OUT, exist_ok=True)

    src = zarr.open(SRC, mode='r')
    ends = np.asarray(src['meta/episode_ends'][:])
    img = src['data/img']

    last_indices = ends - 1
    print(f'Episodes: {len(ends)}  | image array: {img.shape} {img.dtype}')

    for ep, t in enumerate(tqdm(last_indices, desc='save')):
        frame = img[int(t)]
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(OUT, f'ep_{ep:04d}.png'), bgr)

    print(f'Done. Wrote {len(ends)} images to {OUT}')


if __name__ == '__main__':
    main()
