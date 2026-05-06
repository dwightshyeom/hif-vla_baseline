"""
Backward-compatibility shim.

The Push-T friction task environment was originally defined in this file.
To match the naming convention used by every other PushT variant
(``pusht_keypoints_<task>_env.py`` / ``pusht_image_<task>_env.py``) the
class has moved to :mod:`pusht_keypoints_friction_env`.

Existing imports of the form::

    from memory_diffusion_policy.env.pusht.pusht_friction_env import PushTKeypointsFrictionEnv

continue to work via this re-export. New code should import from
``memory_diffusion_policy.env.pusht.pusht_keypoints_friction_env`` directly.
"""
from memory_diffusion_policy.env.pusht.pusht_keypoints_friction_env import (
    PushTKeypointsFrictionEnv,
)

__all__ = ["PushTKeypointsFrictionEnv"]
