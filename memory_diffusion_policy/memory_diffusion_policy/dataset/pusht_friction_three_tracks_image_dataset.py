"""
Vanilla-DP image dataset for the Push-T friction THREE-TRACK task.

Functionally identical to PushTFrictionImageDataset — both expect a zarr
with arrays {img, state, action} and surface state[:, :2] as agent_pos.
This subclass exists so the three-tracks task config can target a
clearly-named dataset and so any future task-specific tweaks (e.g.,
filtering by ``normal_lane_idx``, weighting per-trial samples) can land
here without affecting the 2-track dataset.

The two LSTM-based image datasets
(obs_action_chunk_lstm_image_chunked_latent_dataset.py and
obs_action_chunk_lstm_image_finetune_dataset.py) already accept any
state-array width via ``state[:, :2]``-style slicing, so the same zarr
works with all three workspaces (vanilla DP, frozen LSTM, finetune).
"""
from memory_diffusion_policy.dataset.pusht_friction_image_dataset import (
    PushTFrictionImageDataset,
)


class PushTFrictionThreeTracksImageDataset(PushTFrictionImageDataset):
    """Same loader/sampler/normalizer as the 2-track friction dataset.

    Currently no behaviour overrides — this class exists to give the
    three-tracks task a stable, task-specific dataset symbol that can be
    extended later without disturbing the 2-track pipeline.
    """
    pass
