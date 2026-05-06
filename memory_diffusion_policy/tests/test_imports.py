"""Smoke tests: every public module under memory_diffusion_policy/ imports.

Catches simple regressions like a missed rename, a deleted upstream symbol,
or a broken cross-module import.  Runs in a few seconds and should be the
first sanity check after any structural change.
"""
import importlib
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG_DIR = ROOT / "memory_diffusion_policy"


# Subpackages that depend on real-robot-only libraries (pyrealsense2,
# rtde_control, spacemouse) which aren't installed in the training/test
# environment.  Importing them is verified at deploy time, not here.
SKIP_SUBPKGS = ("memory_diffusion_policy.real_world", "memory_diffusion_policy.shared_memory")


def _walk_modules():
    for py in PKG_DIR.rglob("*.py"):
        if py.name == "__init__.py":
            continue
        rel = py.relative_to(ROOT).with_suffix("")
        # Skip private templates / scripts we don't expect to import standalone.
        parts = rel.parts
        if any(p.startswith("_") and not p.startswith("__") for p in parts):
            continue
        modname = ".".join(parts)
        if any(modname.startswith(prefix) for prefix in SKIP_SUBPKGS):
            continue
        yield modname


MODULES = sorted(_walk_modules())


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)


def test_upstream_diffusion_policy_importable():
    """The bundled upstream submodule must be importable as `diffusion_policy`."""
    importlib.import_module("diffusion_policy.workspace.base_workspace")
    importlib.import_module("diffusion_policy.common.pytorch_util")
    importlib.import_module("diffusion_policy.common.replay_buffer")
