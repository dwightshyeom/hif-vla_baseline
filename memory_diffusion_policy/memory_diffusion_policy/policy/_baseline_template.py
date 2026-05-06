"""Skeleton for plugging a new baseline (e.g. a VLA model) into Memory_DP.

Copy this file under a descriptive name (`my_vla_policy.py`, etc.), implement
the four override points, and register the resulting class via a Hydra
``_target_`` in your task / workspace config.

Two abstract bases are exported by upstream and re-used here:

  * ``memory_diffusion_policy.policy.base_image_policy.BaseImagePolicy``
  * ``memory_diffusion_policy.policy.base_lowdim_policy.BaseLowdimPolicy``

Both inherit from ``ModuleAttrMixin`` (so ``self.device`` / ``self.dtype`` work
automatically) and require exactly two methods to be overridden:

    predict_action(obs_dict)        # rollout-time inference
    set_normalizer(normalizer)      # called by the workspace once the dataset
                                    # has computed its LinearNormalizer

Optional but commonly overridden:

    reset()                         # for stateful policies (RNN/LSTM hidden
                                    # state, attention KV cache, etc.)
    compute_loss(batch)             # if your training loss isn't the standard
                                    # diffusion-DDPM loss, add it here and
                                    # call it from your workspace's run loop

See the existing six policies for fully worked examples:
  policy/diffusion_unet_lowdim_policy.py                                (vanilla DP, lowdim)
  policy/diffusion_unet_hybrid_image_policy.py                          (vanilla DP, image)
  policy/diffusion_unet_lowdim_policy_with_obs_action_chunk_lstm.py     (frozen LSTM + DP, lowdim)
  policy/diffusion_unet_hybrid_image_policy_with_obs_action_chunk_lstm.py (frozen LSTM + DP, image)
  policy/diffusion_unet_lowdim_policy_with_obs_action_chunk_lstm_finetune.py    (finetune LSTM + DP, lowdim)
  policy/diffusion_unet_hybrid_image_policy_with_obs_action_chunk_lstm_finetune.py (finetune LSTM + DP, image)
"""
from typing import Dict

import torch
import torch.nn as nn

from memory_diffusion_policy.policy.base_image_policy import BaseImagePolicy
from memory_diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from memory_diffusion_policy.model.common.normalizer import LinearNormalizer


class BaselineImagePolicyTemplate(BaseImagePolicy):
    """Image (vision) baseline template — subclass and rename me.

    Constructor signature should at minimum accept ``shape_meta`` (the dict
    declared in each ``config/task/*_image*.yaml``) so the workspace can
    instantiate via ``hydra.utils.instantiate(cfg.policy)``.
    """

    def __init__(self, shape_meta: Dict, **kwargs):
        super().__init__()
        # 1. Read shape_meta to size your network:
        #    obs_keys     = list(shape_meta['obs'].keys())
        #    action_dim   = shape_meta['action']['shape'][0]
        # 2. Build your modules (CNN/ViT trunk, transformer blocks, etc.).
        # 3. self.normalizer = LinearNormalizer()  # filled in via set_normalizer
        raise NotImplementedError("Replace this skeleton with your model.")

    def set_normalizer(self, normalizer: LinearNormalizer):
        """Called once by the workspace before training starts."""
        self.normalizer.load_state_dict(normalizer.state_dict())

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Rollout-time inference. Must return a dict with at least
        ``'action'`` of shape ``(B, n_action_steps, action_dim)`` in the
        un-normalised action space.

        Add ``'action_pred'`` (full predicted horizon) if you want it
        logged by the env_runner alongside ``'action'``.
        """
        raise NotImplementedError()

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Optional — only needed if your workspace calls ``policy.compute_loss``.

        ``batch`` is whatever your task's dataset returns from ``__getitem__``.
        """
        raise NotImplementedError()


class BaselineLowdimPolicyTemplate(BaseLowdimPolicy):
    """Lowdim (state-only) baseline template — subclass and rename me."""

    def __init__(self, obs_dim: int, action_dim: int, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # Build your network here.
        raise NotImplementedError("Replace this skeleton with your model.")

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()
