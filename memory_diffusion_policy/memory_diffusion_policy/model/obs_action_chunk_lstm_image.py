"""
Vision-based ObsActionChunkLSTM.

Wraps a visual encoder (same architecture as DiffusionUnetHybridImagePolicy)
and the existing ObsActionChunkLSTM.

Pipeline per timestep:
    image_{t} -> obs_encoder -> visual_feature  (obs_feature_dim,)
    agent_pos_{t}                                (2,)
    action_{t}                                   (2,)
    
    LSTM input: [visual_feature, agent_pos_norm, action_norm]
    (The shift by one timestep is handled inside ObsActionChunkLSTM.forward)

The visual encoder is created using the same robomimic factory as the vanilla DP
to ensure identical feature representations.
"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from memory_diffusion_policy.model.obs_action_chunk_lstm import ObsActionChunkLSTM
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from diffusion_policy.common.pytorch_util import replace_submodules
from robomimic.algo import algo_factory
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import memory_diffusion_policy.model.vision.crop_randomizer as dmvc


def build_obs_encoder(
    shape_meta: dict,
    crop_shape=(76, 76),
    obs_encoder_group_norm=True,
    eval_fixed_crop=True,
    num_kp=32,
):
    """
    Build the same robomimic obs_encoder used in DiffusionUnetHybridImagePolicy.
    
    Returns:
        obs_encoder: nn.Module
        obs_feature_dim: int
    """
    action_shape = shape_meta['action']['shape']
    action_dim = action_shape[0]
    obs_shape_meta = shape_meta['obs']
    
    obs_config = {'low_dim': [], 'rgb': [], 'depth': [], 'scan': []}
    obs_key_shapes = dict()
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        obs_key_shapes[key] = list(shape)
        type_ = attr.get('type', 'low_dim')
        if type_ == 'rgb':
            obs_config['rgb'].append(key)
        elif type_ == 'low_dim':
            obs_config['low_dim'].append(key)
    
    config = get_robomimic_config(
        algo_name='bc_rnn', hdf5_type='image',
        task_name='square', dataset_type='ph')
    
    with config.unlocked():
        config.observation.modalities.obs = obs_config
        config.observation.encoder.rgb.core_kwargs.pool_kwargs.num_kp = num_kp
        if crop_shape is None:
            for key, modality in config.observation.encoder.items():
                if modality.obs_randomizer_class == 'CropRandomizer':
                    modality['obs_randomizer_class'] = None
        else:
            ch, cw = crop_shape
            for key, modality in config.observation.encoder.items():
                if modality.obs_randomizer_class == 'CropRandomizer':
                    modality.obs_randomizer_kwargs.crop_height = ch
                    modality.obs_randomizer_kwargs.crop_width = cw
    
    ObsUtils.initialize_obs_utils_with_config(config)
    
    policy = algo_factory(
        algo_name=config.algo_name, config=config,
        obs_key_shapes=obs_key_shapes, ac_dim=action_dim, device='cpu')
    
    obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
    
    if obs_encoder_group_norm:
        replace_submodules(
            root_module=obs_encoder,
            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
            func=lambda x: nn.GroupNorm(
                num_groups=x.num_features // 16,
                num_channels=x.num_features))
    
    if eval_fixed_crop:
        replace_submodules(
            root_module=obs_encoder,
            predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
            func=lambda x: dmvc.CropRandomizer(
                input_shape=x.input_shape,
                crop_height=x.crop_height,
                crop_width=x.crop_width,
                num_crops=x.num_crops,
                pos_enc=x.pos_enc))
    
    obs_feature_dim = obs_encoder.output_shape()[0]
    return obs_encoder, obs_feature_dim


class ObsActionChunkLSTMImage(nn.Module):
    """
    Vision-based LSTM model for PushT.
    
    Wraps a visual encoder + ObsActionChunkLSTM.
    obs_dim for the inner LSTM = obs_feature_dim (visual encoder output).
    agent_pos is fed through the encoder as a low_dim input alongside the image.
    
    The obs_encoder processes {'image': (B*T, 3, H, W), 'agent_pos': (B*T, 2)}
    and outputs a single feature vector per timestep, which becomes the obs
    input to the LSTM.
    """
    
    def __init__(
        self,
        shape_meta: dict,
        action_dim: int = 2,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        chunk_H: int = 16,
        use_obs_head: bool = False,
        use_future_head: bool = True,
        num_future_modes: int = 1,
        future_abstraction: str = 'raw',
        future_n_bases: int = 32,
        use_past_head: bool = True,
        past_chunk_H: Optional[int] = None,
        past_abstraction: str = 'raw',
        past_n_bases: int = 32,
        use_vq: bool = False,
        vq_n_codes: int = 512,
        vq_commitment_weight: float = 0.25,
        crop_shape: Tuple[int, int] = (76, 76),
        obs_encoder_group_norm: bool = True,
        eval_fixed_crop: bool = True,
        freeze_encoder: bool = False,
        num_kp: int = 32,
    ):
        super().__init__()
        
        # Build visual encoder (same as vanilla DP)
        self.obs_encoder, self.obs_feature_dim = build_obs_encoder(
            shape_meta=shape_meta,
            crop_shape=crop_shape,
            obs_encoder_group_norm=obs_encoder_group_norm,
            eval_fixed_crop=eval_fixed_crop,
            num_kp=num_kp,
        )
        
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.obs_encoder.parameters():
                p.requires_grad = False
        
        # The LSTM's obs_dim = visual encoder output dim
        self.lstm = ObsActionChunkLSTM(
            obs_dim=self.obs_feature_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            chunk_H=chunk_H,
            use_obs_head=use_obs_head,
            use_future_head=use_future_head,
            num_future_modes=num_future_modes,
            future_abstraction=future_abstraction,
            future_n_bases=future_n_bases,
            use_past_head=use_past_head,
            past_chunk_H=past_chunk_H,
            past_abstraction=past_abstraction,
            past_n_bases=past_n_bases,
            action_only=False,
            use_vq=use_vq,
            vq_n_codes=vq_n_codes,
            vq_commitment_weight=vq_commitment_weight,
        )
        
        # Expose key properties
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.chunk_H = chunk_H
        self.use_obs_head = use_obs_head
        self.use_future_head = use_future_head
        self.use_past_head = use_past_head
        self.past_chunk_H = self.lstm.past_chunk_H
    
    def encode_obs(self, image: torch.Tensor, agent_pos: torch.Tensor) -> torch.Tensor:
        """
        Encode image + agent_pos through the robomimic obs_encoder.
        
        Args:
            image:     (B, T, 3, H, W) float32 [0, 1] raw
            agent_pos: (B, T, 2) float32 (LP-normalised)
        
        Returns:
            features: (B, T, obs_feature_dim)
        """
        B, T = image.shape[:2]
        
        # Normalise images to [-1, 1] to match DP's encoder input convention
        image_norm = image * 2.0 - 1.0
        
        # Flatten batch and time for encoder
        obs_dict = {
            'image': image_norm.reshape(B * T, *image_norm.shape[2:]),
            'agent_pos': agent_pos.reshape(B * T, *agent_pos.shape[2:]),
        }
        
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.obs_encoder(obs_dict)
        else:
            features = self.obs_encoder(obs_dict)
        
        # Reshape back to (B, T, feature_dim)
        return features.reshape(B, T, -1)
    
    def forward(
        self,
        image: torch.Tensor,
        agent_pos: torch.Tensor,
        actions: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        hidden_state=None,
    ):
        """
        Full forward pass: encode images, then run LSTM.
        
        Args:
            image:     (B, T, 3, H, W) normalized images
            agent_pos: (B, T, 2) normalized agent positions
            actions:   (B, T, 2) normalized actions
            lengths:   (B,) episode lengths
            hidden_state: optional LSTM hidden state
        
        Returns: same as ObsActionChunkLSTM.forward(), but obs_pred
                 is in visual feature space (obs_feature_dim), not image space.
        """
        obs_features = self.encode_obs(image, agent_pos)
        return self.lstm.forward(
            obs=obs_features,
            actions=actions,
            lengths=lengths,
            hidden_state=hidden_state,
        )
    
    def predict(self, image, agent_pos, actions, lengths=None, hidden_state=None):
        """No-gradient forward pass."""
        with torch.no_grad():
            obs_features = self.encode_obs(image, agent_pos)
            return self.lstm.predict(
                obs=obs_features,
                actions=actions,
                lengths=lengths,
                hidden_state=hidden_state,
            )
    
    def extract_hidden_states(self, image, agent_pos, actions, lengths=None, hidden_state=None):
        """Extract per-step LSTM hidden states."""
        obs_features = self.encode_obs(image, agent_pos)
        return self.lstm.extract_hidden_states(
            obs=obs_features,
            actions=actions,
            lengths=lengths,
            hidden_state=hidden_state,
        )
    
    def reset(self):
        self.lstm.reset()
