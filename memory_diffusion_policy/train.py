"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace

python train.py --config-dir=. --config-name=train_diffusion_unet_lowdim_workspace.yaml task=pusht_three_goals_lowdim training.seed=42 training.device=cuda:0 training.resume=False hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python train.py --config-dir=. --config-name=train_diffusion_unet_lowdim_workspace.yaml task=pusht_one_random_goal_lowdim training.seed=42 training.device=cuda:0 training.resume=False hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python train.py --config-dir=. --config-name=train_diffusion_unet_lowdim_sequential_workspace.yaml task=pusht_three_goals_lowdim_indicator_sequential training.seed=42 training.device=cuda:0 training.resume=False hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python train.py --config-dir=. --config-name=train_diffusion_unet_lowdim_latent_memory_slide_window_workspace.yaml task=pusht_three_goals_lowdim_latent_memory_slide_window training.seed=42 training.device=cuda:0 training.resume=False hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python train.py --config-dir=. --config-name=train_diffusion_unet_lowdim_mlp_workspace.yaml task=pusht_three_goals_lowdim_mlp_encoder training.seed=42 training.device=cuda:0 training.resume=False hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python train.py --config-dir=. --config-name=train_diffusion_unet_lowdim_lstm_auxiliary_workspace.yaml task=pusht_three_goals_lowdim_lstm_auxiliary training.seed=42 training.device=cuda:0 training.resume=False hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python train.py --config-dir=. --config-name=train_diffusion_unet_lowdim_indicator_latent_sequential_workspace.yaml task=pusht_three_goals_lowdim_indicator_latent_sequential training.seed=42 training.device=cuda:0 training.resume=False hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
"""
'''
python train.py --config-dir=. --config-name=train_diffusion_unet_lowdim_with_goals_workspace.yaml task=pusht_three_goals_lowdim_with_goals training.seed=42 training.device=cuda:0 training.resume=False hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python train.py --config-dir=. --config-name=train_diffusion_unet_lowdim_progression_sequential_workspace.yaml task=pusht_three_goals_lowdim_progression_sequential training.seed=42 training.device=cuda:0 training.resume=False hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python train.py --config-dir=. --config-name=train_diffusion_unet_lowdim_indicator_percentage_workspace.yaml task=pusht_three_goals_lowdim_indicator_percentage training.seed=42 training.device=cuda:0 training.resume=False hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python train.py --config-dir=. --config-name=train_diffusion_unet_lowdim_indicator_vq_latent_sequential_workspace.yaml task=pusht_three_goals_lowdim_indicator_vq_latent_sequential training.seed=42 training.device=cuda:0 training.resume=False hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python train.py --config-dir=. --config-name=train_diffusion_unet_lowdim_indicator_and_indicator_latent_sequential_workspace.yaml task=pusht_three_goals_lowdim_indicator_and_indicator_latent_sequential training.seed=42 training.device=cuda:0 training.resume=False hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
-----

python train.py --config-dir=. --config-name=train_diffusion_unet_lowdim_joint_lstm_workspace.yaml task=pusht_three_goals_lowdim_joint_lstm training.seed=42 training.device=cuda:0 training.resume=False hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
#####

python train.py --config-dir=. --config-name=train_diffusion_unet_lowdim_obs_action_chunk_lstm_workspace.yaml task=pusht_three_goals_lowdim_obs_action_chunk_lstm training.seed=42 training.device=cuda:0 training.resume=False hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python train.py --config-dir=. --config-name=train_diffusion_unet_lowdim_varying_goals_workspace.yaml task=pusht_three_goals_varying_lowdim training.seed=42 training.device=cuda:0 training.resume=False hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python train.py   --config-dir=.   --config-name=train_diffusion_unet_lowdim_obs_action_chunk_lstm_finetune_workspace.yaml   task=pusht_three_goals_lowdim_obs_action_chunk_lstm_finetune   training.seed=42 training.device=cuda:0 training.resume=False   hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

======================================================
python train.py --config-name=train_diffusion_unet_hybrid_three_goals_swap_workspace.yaml task=pusht_image_three_goals_swap training.device=cuda:0 training.resume=False hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

NVIDIA_VISIBLE_DEVICES=1 docker compose run --rm dev python train.py --config-name=train_diffusion_unet_hybrid_lstm_workspace_scale_5.yaml task=pusht_image_three_goals_obs_action_chunk_lstm training.device=cuda:0 training.resume=False hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

NVIDIA_VISIBLE_DEVICES=0 docker compose run --rm dev python train.py --config-name=train_diffusion_unet_hybrid_lstm_finetune_workspace_scale_1.yaml task=pusht_image_three_goals_obs_action_chunk_lstm_finetune training.device=cuda:0 training.resume=False hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
##########
NVIDIA_VISIBLE_DEVICES=0 docker compose run --rm dev python train.py --config-name=train_diffusion_unet_hybrid_lstm_workspace_scale_1.yaml task=pusht_image_friction_obs_action_chunk_lstm training.device=cuda:0 training.resume=False hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

NVIDIA_VISIBLE_DEVICES=0 docker compose run --rm dev python train.py --config-name=train_diffusion_unet_hybrid_lstm_workspace_scale_1.yaml task=pusht_image_three_goals_obs_action_chunk_lstm training.device=cuda:0 training.resume=False hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
'''

import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'memory_diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
