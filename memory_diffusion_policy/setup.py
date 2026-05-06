"""Memory_DP install: also exposes the bundled upstream diffusion_policy submodule.

`memory_diffusion_policy` (this repo) and `diffusion_policy` (the submodule under
third_party/diffusion_policy) are installed together by a single
`pip install -e .` from the repo root, so callers can `import diffusion_policy`
to reach upstream code without a separate install step.
"""
from pathlib import Path

from setuptools import find_namespace_packages, setup

THIRD_PARTY_DP = Path('third_party/diffusion_policy')

own_packages = find_namespace_packages(
    include=['memory_diffusion_policy', 'memory_diffusion_policy.*'],
)
upstream_packages = []
if THIRD_PARTY_DP.exists():
    upstream_packages = find_namespace_packages(
        where=str(THIRD_PARTY_DP),
        include=['diffusion_policy', 'diffusion_policy.*'],
    )

setup(
    name='memory_diffusion_policy',
    packages=own_packages + upstream_packages,
    package_dir={
        'diffusion_policy': str(THIRD_PARTY_DP / 'diffusion_policy'),
    },
)
