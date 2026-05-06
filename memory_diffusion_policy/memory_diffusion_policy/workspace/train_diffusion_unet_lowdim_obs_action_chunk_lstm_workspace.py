"""
Workspace entry-point for training Diffusion Policy conditioned on
pretrained ObsActionChunkLSTM hidden states.

Reuses the generic TrainDiffusionUnetLowdimWorkspace and selects the
matching Hydra config via ``config_name = <this_file_stem>``.
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import hydra
import pathlib
from omegaconf import OmegaConf

from memory_diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import (
    TrainDiffusionUnetLowdimWorkspace,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
