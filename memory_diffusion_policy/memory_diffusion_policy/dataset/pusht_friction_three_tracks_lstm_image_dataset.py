"""
LSTM-pretraining image dataset for the Push-T friction THREE-TRACK task.

Functionally identical to PushTFrictionLSTMImageDataset — both expect a
zarr with arrays {img, state, action} and surface state[:, :2] as
agent_pos. This subclass exists so the three-tracks LSTM pretraining
config can target a clearly-named dataset and so any future task-specific
tweaks (e.g. weighting return-phase steps differently, filtering by
``normal_lane_idx``) can land here without affecting the 2-track
pipeline.

Drop-in compatible with ``train_obs_action_chunk_lstm_image.py``'s
``--dataset_class`` hook. Re-exports ``collate_fn_lstm_image`` for the
trainer's import path.

Use:
    python train_obs_action_chunk_lstm_image.py \\
        --zarr_path data/pusht_friction_three_tracks_demo.zarr \\
        --dataset_class memory_diffusion_policy.dataset.pusht_friction_three_tracks_lstm_image_dataset.PushTFrictionThreeTracksLSTMImageDataset \\
        --action_step_subsample 1 \\
        ... [other LSTM args] ...
"""
from memory_diffusion_policy.dataset.pusht_friction_lstm_image_dataset import (
    PushTFrictionLSTMImageDataset,
    collate_fn_lstm_image,
)

__all__ = ['PushTFrictionThreeTracksLSTMImageDataset', 'collate_fn_lstm_image']


class PushTFrictionThreeTracksLSTMImageDataset(PushTFrictionLSTMImageDataset):
    """Same loader / sampler / normalizer as the 2-track friction LSTM dataset.

    Currently no behaviour overrides — exists to give the three-tracks
    LSTM pretraining a stable, task-specific dataset symbol that can be
    extended later without disturbing the 2-track pipeline.
    """
    pass
