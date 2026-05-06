if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import copy
import random

import hydra
import torch
import numpy as np
import wandb
import tqdm
import shutil
import pathlib
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from memory_diffusion_policy.policy.diffusion_unet_hybrid_image_policy_with_obs_action_chunk_lstm_finetune import (
    DiffusionUnetHybridImagePolicyWithObsActionChunkLSTMFinetune,
)
from memory_diffusion_policy.dataset.base_dataset import BaseImageDataset
from memory_diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDiffusionUnetHybridLSTMFinetuneWorkspace(BaseWorkspace):
    """
    Workspace for vision-based Diffusion Policy with a **finetunable**
    pretrained ObsActionChunkLSTMImage.

    Supports three finetuning schedules:
        'joint'             — train LSTM core + DP together from epoch 0.
        'warmup_then_joint' — freeze LSTM for freeze_lstm_epochs, then train both.
        'alternating'       — alternate DP-only / LSTM-only phases.

    Key differences from TrainDiffusionUnetHybridLSTMWorkspace (frozen):
        - LSTM core parameters receive separate (lower) learning rate.
        - Finetune schedule controls freeze/unfreeze per epoch.
        - Dataset provides full episode data for live LSTM forward pass.
    """

    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: DiffusionUnetHybridImagePolicyWithObsActionChunkLSTMFinetune
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetHybridImagePolicyWithObsActionChunkLSTMFinetune = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # ---- Optimizer with separate LSTM learning rate ----------------
        base_lr = cfg.optimizer.lr
        lstm_lr_scale = float(getattr(cfg, "lstm_lr_scale", 1.0))

        if (hasattr(self.model, "get_parameter_groups")
                and lstm_lr_scale != 1.0):
            param_groups = self.model.get_parameter_groups(base_lr)
            opt_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
            opt_cfg.pop("lr", None)
            opt_cfg.pop("_target_", None)
            self.optimizer = torch.optim.AdamW(param_groups, **opt_cfg)
            print(f"Using separate LSTM lr scale: {lstm_lr_scale} "
                  f"(LSTM lr = {base_lr * lstm_lr_scale:.2e})")
        else:
            self.optimizer = hydra.utils.instantiate(
                cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    # ------------------------------------------------------------------
    # Loss + backward.
    #
    # Routing semantics:
    #   Warm-up (LSTM core fully frozen — detected by checking
    #   _lstm_core_parameters().requires_grad):
    #     → DCT is suppressed entirely. All trainable params (DP +
    #       hidden_vq + hidden_attention_gate + hidden_projection) get
    #       only raw_loss. This keeps the warm-up identical to
    #       use_dct_loss=False, which is the only way to avoid the DCT
    #       gradient leaking into the projection modules (the LSTM core
    #       it was meant to shape is frozen).
    #
    #   Post warm-up (LSTM core trainable):
    #     use_dct_loss=False              standard single backward
    #       → all trainable params get raw_loss
    #
    #     use_dct_loss=True, dct_loss_on_dp=True       single backward
    #       → all trainable params get raw_loss + α·dct_loss
    #
    #     use_dct_loss=True, dct_loss_on_dp=False      DUAL backward
    #       → DP-only params (UNet, obs_encoder, obs_projection)   ← raw_loss
    #       → LSTM-pathway params (LSTM core + hidden_vq +
    #         hidden_attention_gate + hidden_projection)            ← raw + α·dct
    # ------------------------------------------------------------------
    def _compute_loss_and_backward(
        self,
        batch,
        accumulate_every: int,
    ) -> dict:
        """Compute the policy loss and route gradients per the routing table
        above. Returns logging dict with float values:
          'total_loss': what was actually back-propagated (raw, or raw + α·dct)
          'loss_raw' / 'loss_dct': only when DCT loss is enabled (logged even
              when DCT is suppressed during warm-up, for monitoring).
        """
        result = self.model.compute_loss(batch)

        # use_dct_loss=False: model returns a scalar.
        if not isinstance(result, dict):
            loss = result / accumulate_every
            loss.backward()
            return {"total_loss": float(result.detach().item())}

        # use_dct_loss=True: dict with raw + α·dct exposed separately.
        loss_main = result["loss_main"]
        loss_dct  = result["loss_dct"]
        raw_v = float(result["loss_raw_value"].item())
        dct_v = float(result["loss_dct_value"].item())

        # Warm-up detection: LSTM core fully frozen.
        lstm_core_trainable = any(
            p.requires_grad for p in self.model._lstm_core_parameters()
        )

        # ----- Warm-up path: suppress DCT, single backward of raw -----
        # All trainable params (DP + projections) get raw_loss only,
        # exactly matching the use_dct_loss=False behavior.
        if not lstm_core_trainable:
            loss = loss_main / accumulate_every
            loss.backward()
            return {
                "total_loss": raw_v,
                "loss_raw":   raw_v,
                "loss_dct":   dct_v,  # logged for monitoring; not back-propagated
            }

        # ----- Post warm-up, dct_loss_on_dp=True: single backward -----
        # Both DP and LSTM-pathway get raw + α·dct.
        if self.model.dct_loss_on_dp:
            loss = (loss_main + loss_dct) / accumulate_every
            loss.backward()
            total = raw_v + self.model.dct_loss_weight * dct_v
            return {
                "total_loss": total,
                "loss_raw":   raw_v,
                "loss_dct":   dct_v,
            }

        # ----- Post warm-up, dct_loss_on_dp=False: dual backward -----
        # Pass 1: raw_loss to ALL trainable params (no inputs= restriction)
        #         — DP gets raw via the global graph traversal.
        # Pass 2: α·dct_loss restricted to LSTM-pathway via autograd.grad
        #         and added on top of the existing .grad — the LSTM
        #         pathway ends up with raw + α·dct, DP keeps just raw.
        loss_main_scaled = loss_main / accumulate_every
        loss_dct_scaled  = loss_dct  / accumulate_every

        lstm_pathway_params = [
            p for p in self.model._lstm_pathway_parameters() if p.requires_grad
        ]

        loss_main_scaled.backward(retain_graph=True)

        if lstm_pathway_params:
            grads = torch.autograd.grad(
                loss_dct_scaled,
                lstm_pathway_params,
                allow_unused=True,
                retain_graph=False,
            )
            for p, g in zip(lstm_pathway_params, grads):
                if g is None:
                    continue
                if p.grad is None:
                    p.grad = g.detach()
                else:
                    p.grad = p.grad + g.detach()

        total = raw_v + self.model.dct_loss_weight * dct_v
        return {
            "total_loss": total,
            "loss_raw":   raw_v,
            "loss_dct":   dct_v,
        }

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # ---- Resume ----------------------------------------------------
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # ---- Dataset ---------------------------------------------------
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        # Use custom collate_fn if the dataset provides one
        dl_kwargs = OmegaConf.to_container(cfg.dataloader, resolve=True)
        val_dl_kwargs = OmegaConf.to_container(cfg.val_dataloader, resolve=True)

        collate_fn = getattr(dataset, "collate_fn", None)
        if collate_fn is not None:
            dl_kwargs["collate_fn"] = collate_fn
            val_dl_kwargs["collate_fn"] = collate_fn
            print("Using custom collate_fn from dataset")

        train_dataloader = DataLoader(dataset, **dl_kwargs)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **val_dl_kwargs)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # Set LSTM normalizer (from dataset, from pretraining)
        if (hasattr(dataset, "_lstm_normalizer")
                and dataset._lstm_normalizer is not None):
            lstm_norm = dataset._lstm_normalizer
        else:
            lstm_norm = normalizer
        self.model.set_lstm_normalizer(lstm_norm)
        if cfg.training.use_ema:
            self.ema_model.set_lstm_normalizer(lstm_norm)

        # ---- LR scheduler ----------------------------------------------
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs
            ) // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        # ---- EMA -------------------------------------------------------
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # ---- Env runner ------------------------------------------------
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # ---- Logging ---------------------------------------------------
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})

        # ---- Checkpointing ---------------------------------------------
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        # ---- Device ----------------------------------------------------
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

        # ---- Training loop ---------------------------------------------
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()

                # ======== Apply finetune schedule =======================
                schedule = str(getattr(cfg, "finetune_schedule", "joint"))

                if schedule == "joint":
                    # Always train both
                    self.model.unfreeze_lstm()
                    self.model.unfreeze_dp()

                elif schedule == "warmup_then_joint":
                    freeze_epochs = int(getattr(cfg, "freeze_lstm_epochs", 0))
                    if self.epoch < freeze_epochs:
                        self.model.freeze_lstm()
                        self.model.unfreeze_dp()
                        if self.epoch == 0:
                            print(
                                f"[schedule] warmup_then_joint: "
                                f"freezing LSTM for first {freeze_epochs} epochs"
                            )
                    else:
                        self.model.unfreeze_lstm()
                        self.model.unfreeze_dp()
                        if self.epoch == freeze_epochs:
                            print(
                                f"[schedule] warmup_then_joint: "
                                f"unfreezing LSTM at epoch {self.epoch}"
                            )

                elif schedule == "alternating":
                    period = int(getattr(cfg, "alternating_epoch_period", 10))
                    phase = (self.epoch // period) % 2
                    if phase == 0:
                        # Even phase: DP only
                        self.model.freeze_lstm()
                        self.model.unfreeze_dp()
                        if self.epoch % period == 0:
                            print(
                                f"[schedule] alternating: epoch {self.epoch} "
                                f"— DP only (period={period})"
                            )
                    else:
                        # Odd phase: LSTM only
                        self.model.unfreeze_lstm()
                        self.model.freeze_dp()
                        if self.epoch % period == 0:
                            print(
                                f"[schedule] alternating: epoch {self.epoch} "
                                f"— LSTM only (period={period})"
                            )

                else:
                    raise ValueError(
                        f"Unknown finetune_schedule: {schedule!r}. "
                        "Choose from: joint, warmup_then_joint, alternating"
                    )

                # ======== Train for this epoch ==========================
                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(
                            batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        loss_info = self._compute_loss_and_backward(
                            batch, cfg.training.gradient_accumulate_every)
                        raw_loss_cpu = loss_info["total_loss"]

                        if (self.global_step
                                % cfg.training.gradient_accumulate_every == 0):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        if cfg.training.use_ema:
                            ema.step(self.model)

                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }
                        # Per-component logs (only present when DCT loss is enabled)
                        if "loss_raw" in loss_info:
                            step_log["train_loss_raw"] = loss_info["loss_raw"]
                        if "loss_dct" in loss_info:
                            step_log["train_loss_dct"] = loss_info["loss_dct"]

                        is_last_batch = (
                            batch_idx == len(train_dataloader) - 1)
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None
                                and batch_idx
                                >= cfg.training.max_train_steps - 1):
                            break

                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ======== Eval ==========================================
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # Rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy, epoch=self.epoch)
                    step_log.update(runner_log)

                # Validation loss
                if (self.epoch % cfg.training.val_every) == 0:
                    # Mirror the training-time DCT suppression: while the
                    # LSTM core is frozen, train_loss does not include DCT,
                    # so val_loss should not either.
                    val_lstm_core_trainable = any(
                        p.requires_grad
                        for p in self.model._lstm_core_parameters()
                    )
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
                                    lambda x: x.to(device, non_blocking=True))
                                result = self.model.compute_loss(batch)
                                if isinstance(result, dict):
                                    if val_lstm_core_trainable:
                                        # raw + alpha*dct (matches training)
                                        val_total = (
                                            result["loss_raw_value"]
                                            + self.model.dct_loss_weight
                                            * result["loss_dct_value"]
                                        )
                                    else:
                                        # warm-up: DCT is suppressed in training
                                        val_total = result["loss_raw_value"]
                                    val_losses.append(val_total)
                                else:
                                    val_losses.append(result)
                                if (cfg.training.max_val_steps is not None
                                        and batch_idx
                                        >= cfg.training.max_val_steps - 1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(
                                torch.tensor(val_losses)).item()
                            step_log["val_loss"] = val_loss

                # Sample action MSE
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = train_sampling_batch
                        obs_dict = batch["obs"]
                        gt_action = batch["action"]

                        result = policy.predict_action(obs_dict)
                        pred_action = result["action_pred"]
                        mse = F.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()
                        del obs_dict, gt_action, result, pred_action, mse

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
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                # ======== End of epoch ==================================
                policy.train()

                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


@hydra.main(
    version_base=None,
    config_path=str(
        pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainDiffusionUnetHybridLSTMFinetuneWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
