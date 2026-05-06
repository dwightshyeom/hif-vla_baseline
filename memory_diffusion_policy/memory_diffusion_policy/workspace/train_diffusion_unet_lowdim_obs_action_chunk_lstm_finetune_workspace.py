if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm
import shutil

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from memory_diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from memory_diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from memory_diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)


# %%
class TrainDiffusionUnetLowdimLSTMFinetuneWorkspace(BaseWorkspace):
    """
    Training workspace for Diffusion Policy with a **finetunable**
    pretrained ObsActionChunkLSTM.

    Key differences from TrainDiffusionUnetLowdimWorkspace:
      - Uses a custom collate function from the dataset to handle
        variable-length full episode data.
      - Supports separate LSTM learning rate via ``lstm_lr_scale``.
      - LSTM parameters are updated during training (not frozen).
    """
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # Set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Configure model
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # Configure optimizer with optional separate LSTM learning rate
        base_lr = cfg.optimizer.lr
        lstm_lr_scale = getattr(cfg, "lstm_lr_scale", 1.0)

        if hasattr(self.model, "get_parameter_groups") and lstm_lr_scale != 1.0:
            param_groups = self.model.get_parameter_groups(base_lr)
            # Remove 'lr' from optimizer config (it's in param groups)
            opt_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
            opt_cfg.pop("lr", None)
            opt_cfg.pop("_target_", None)
            self.optimizer = torch.optim.AdamW(param_groups, **opt_cfg)
            print(f"Using separate LSTM lr scale: {lstm_lr_scale}")
        else:
            self.optimizer = hydra.utils.instantiate(
                cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # Resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # Configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)

        # Use custom collate function if dataset provides one
        dl_kwargs = OmegaConf.to_container(cfg.dataloader, resolve=True)
        val_dl_kwargs = OmegaConf.to_container(
            cfg.val_dataloader, resolve=True)

        collate_fn = getattr(dataset, "collate_fn", None)
        if collate_fn is not None:
            dl_kwargs["collate_fn"] = collate_fn
            val_dl_kwargs["collate_fn"] = collate_fn
            print("Using custom collate function from dataset")

        train_dataloader = DataLoader(dataset, **dl_kwargs)
        normalizer = dataset.get_normalizer()

        # Validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **val_dl_kwargs)

        # Set normalizer
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # LR scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs)
                // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        # EMA
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema, model=self.ema_model)

        # Env runner
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=self.output_dir)
        print("##### env_runner:", env_runner)
        assert isinstance(env_runner, BaseLowdimRunner)

        # Logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})

        # Checkpointing
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        # Device
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # Training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()

                # ========= Apply finetune schedule ==========
                schedule = getattr(cfg, "finetune_schedule", "joint")
                if schedule == "joint":
                    # Mode 1: train everything together (default)
                    self.model.unfreeze_lstm()
                    self.model.unfreeze_dp()
                elif schedule == "warmup_then_joint":
                    # Mode 2: freeze LSTM for freeze_lstm_epochs, then both
                    freeze_epochs = getattr(cfg, "freeze_lstm_epochs", 0)
                    if self.epoch < freeze_epochs:
                        self.model.freeze_lstm()
                        self.model.unfreeze_dp()
                        if self.epoch == 0:
                            print(f"[schedule] warmup_then_joint: "
                                  f"freezing LSTM for first {freeze_epochs} epochs")
                    else:
                        self.model.unfreeze_lstm()
                        self.model.unfreeze_dp()
                        if self.epoch == freeze_epochs:
                            print(f"[schedule] warmup_then_joint: "
                                  f"unfreezing LSTM at epoch {self.epoch}")
                elif schedule == "alternating":
                    # Mode 3: alternate DP-only / LSTM-only phases
                    period = getattr(cfg, "alternating_epoch_period", 10)
                    phase = (self.epoch // period) % 2
                    if phase == 0:
                        # Even phase: train DP, freeze LSTM
                        self.model.freeze_lstm()
                        self.model.unfreeze_dp()
                        if self.epoch % period == 0:
                            print(f"[schedule] alternating: epoch {self.epoch} "
                                  f"— training DP only (period={period})")
                    else:
                        # Odd phase: train LSTM, freeze DP
                        self.model.unfreeze_lstm()
                        self.model.freeze_dp()
                        if self.epoch % period == 0:
                            print(f"[schedule] alternating: epoch {self.epoch} "
                                  f"— training LSTM only (period={period})")
                else:
                    raise ValueError(
                        f"Unknown finetune_schedule: {schedule}. "
                        f"Choose from: joint, warmup_then_joint, alternating")

                # ========= Train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(
                            batch,
                            lambda x: x.to(device, non_blocking=True),
                        )
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # Compute loss (LSTM is run live inside)
                        raw_loss = self.model.compute_loss(batch)
                        loss = (raw_loss
                                / cfg.training.gradient_accumulate_every)
                        loss.backward()

                        # Step optimizer
                        if (self.global_step
                                % cfg.training.gradient_accumulate_every
                                == 0):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # Update EMA
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(
                            loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = (
                            batch_idx == len(train_dataloader) - 1)
                        if not is_last_batch:
                            wandb_run.log(
                                step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None
                                and batch_idx
                                >= cfg.training.max_train_steps - 1):
                            break

                # End of epoch
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= Eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # Rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(
                        policy, epoch=self.epoch)
                    step_log.update(runner_log)

                # Validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(
                                    batch,
                                    lambda x: x.to(
                                        device, non_blocking=True),
                                )
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps
                                        is not None
                                        and batch_idx
                                        >= cfg.training.max_val_steps
                                        - 1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(
                                torch.tensor(val_losses)).item()
                            step_log["val_loss"] = val_loss

                # Sample
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = train_sampling_batch
                        obs_dict = {"obs": batch["obs"]}
                        result = policy.predict_action(obs_dict)

                        gt_action = batch["action"]
                        if cfg.pred_action_steps_only:
                            pred_action = result["action"]
                            start = cfg.n_obs_steps - 1
                            end = start + cfg.n_action_steps
                            gt_action = gt_action[:, start:end]
                        else:
                            pred_action = result["action_pred"]
                        mse = F.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()
                        del batch, obs_dict, gt_action
                        del result, pred_action, mse

                # Checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    metric_dict = {
                        k.replace("/", "_"): v
                        for k, v in step_log.items()
                    }
                    topk_ckpt_path = topk_manager.get_ckpt_path(
                        metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                # ========= End of epoch ==========
                policy.train()

                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


# Required import for sampling mse
import torch.nn.functional as F


@hydra.main(
    version_base=None,
    config_path=str(
        pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainDiffusionUnetLowdimLSTMFinetuneWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
