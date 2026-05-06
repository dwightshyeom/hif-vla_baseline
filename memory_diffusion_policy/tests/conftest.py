"""Test-suite-wide configuration.

Suppresses known noisy warnings (pkg_resources deprecation, gym env
re-registration on second import) so the pytest output stays focused
on actual failures.
"""
import warnings

# pkg_resources deprecation lives deep in wandb; we can't fix it here.
warnings.filterwarnings(
    "ignore",
    message=".*pkg_resources is deprecated.*",
    category=UserWarning,
)
# gym registers our envs at module import time; re-importing test modules
# triggers a noisy WARN that we don't care about.
warnings.filterwarnings(
    "ignore",
    message=r".*Overriding environment.*",
    category=UserWarning,
)
