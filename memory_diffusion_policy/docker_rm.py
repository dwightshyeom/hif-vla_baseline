#!/usr/bin/env python3
"""
Delete files/folders inside the memory_diffusion_policy repo by shelling out to a
throwaway alpine container that has root inside its mount. Useful when an
artifact (training outputs, generated zarrs, etc.) was created by `docker
compose run` as root and your host user can't `rm` it without sudo.

How to use:
    1. Edit the PATHS list below to list whatever you want to delete.
       Paths can be absolute or relative to the repo root.
    2. Run: python docker_rm.py
       Add --dry-run to print the docker command without executing it,
       or --yes to skip the confirmation prompt.
"""
# ---------------------------------------------------------------------------
# Edit this list. Each entry is a path (file or directory) you want deleted.
# Paths may be absolute, or relative to the repo root.
# ---------------------------------------------------------------------------
PATHS = ['data/outputs/2026.04.25']
# file_name = 'obs_action_chunk_lstm_image_three_goals_vision_step_1_hidden_64'
# PATHS = [
#     f'outputs/{file_name}/checkpoint_epoch_0050.pt',
#     f'outputs/{file_name}/checkpoint_epoch_0100.pt',
#     f'outputs/{file_name}/checkpoint_epoch_0150.pt',
#     f'outputs/{file_name}/checkpoint_epoch_0200.pt',
#     f'outputs/{file_name}/checkpoint_epoch_0250.pt',
#     f'outputs/{file_name}/checkpoint_epoch_0300.pt',
#     f'outputs/{file_name}/checkpoint_epoch_0350.pt',
#     f'outputs/{file_name}/checkpoint_epoch_0400.pt',
#     f'outputs/{file_name}/checkpoint_epoch_0450.pt',
#     f'outputs/{file_name}/checkpoint_epoch_0500.pt',
#     f'outputs/{file_name}/checkpoint_epoch_0550.pt',
#     f'outputs/{file_name}/checkpoint_epoch_0600.pt',
#     f'outputs/{file_name}/checkpoint_epoch_0650.pt',
#     f'outputs/{file_name}/checkpoint_epoch_0700.pt',
#     f'outputs/{file_name}/checkpoint_epoch_0750.pt',
#     f'outputs/{file_name}/checkpoint_epoch_0800.pt',
#     f'outputs/{file_name}/checkpoint_epoch_0850.pt',
#     f'outputs/{file_name}/checkpoint_epoch_0900.pt',
#     f'outputs/{file_name}/checkpoint_epoch_0950.pt',
#     f'outputs/{file_name}/checkpoint_epoch_1000.pt',
# ]
# ---------------------------------------------------------------------------

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

# Hard-blocked targets even when inside the repo.
BLOCKED = {
    REPO_ROOT,
    REPO_ROOT / '.git',
}


def resolve_inside_repo(p: str) -> Path:
    candidate = Path(p).resolve() if os.path.isabs(p) else (REPO_ROOT / p).resolve()
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError:
        raise SystemExit(f'Refusing: {candidate} is outside {REPO_ROOT}')
    if candidate in BLOCKED:
        raise SystemExit(f'Refusing to delete protected path: {candidate}')
    return candidate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-y', '--yes', action='store_true',
                    help='skip the confirmation prompt')
    ap.add_argument('--dry-run', action='store_true',
                    help="print the docker command but don't run it")
    args = ap.parse_args()

    if not PATHS:
        print(f'PATHS is empty in {Path(__file__).name} — nothing to delete.')
        print('Edit the PATHS list near the top of the file and rerun.')
        return 0

    targets, missing = [], []
    for p in PATHS:
        abs_path = resolve_inside_repo(p)
        (targets if abs_path.exists() else missing).append(abs_path)

    if missing:
        print('The following paths do not exist (skipping):')
        for m in missing:
            print(f'  - {m}')

    if not targets:
        print('Nothing to delete.')
        return 0

    # If we're already root (i.e. running inside the dev container), delete
    # natively via shutil. Otherwise spawn a one-shot alpine container.
    inside_container = (os.geteuid() == 0)
    mode = 'native rm (running as root)' if inside_container else 'docker run alpine rm'

    print(f'About to delete via {mode}:')
    for t in targets:
        kind = 'dir ' if t.is_dir() else 'file'
        print(f'  {kind}  {t}')

    if not args.yes and not args.dry_run:
        ans = input('\nProceed? [y/N] ').strip().lower()
        if ans != 'y':
            print('Aborted.')
            return 1

    if inside_container:
        if args.dry_run:
            for t in targets:
                print(f'would remove: {t}')
            return 0
        for t in targets:
            if t.is_dir() and not t.is_symlink():
                shutil.rmtree(t)
            else:
                t.unlink()
        print('Done.')
        return 0

    # Host path: spawn alpine to do the rm with the repo mounted at /repo.
    container_paths = [str(Path('/repo') / t.relative_to(REPO_ROOT)) for t in targets]
    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{REPO_ROOT}:/repo',
        'alpine',
        'rm', '-rf', '--',
        *container_paths,
    ]

    print('\n$ ' + ' '.join(cmd))
    if args.dry_run:
        return 0

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f'docker rm failed: exit {e.returncode}', file=sys.stderr)
        return e.returncode
    except FileNotFoundError:
        print('docker not found on PATH', file=sys.stderr)
        return 127

    print('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
