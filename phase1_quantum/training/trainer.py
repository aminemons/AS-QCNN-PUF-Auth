"""
Core training loop for the Hybrid QCNN PUF model.

Supports two-phase training:
  Phase 1 (warm-up):  VQC frozen, encoder + head trained with higher LR
  Phase 2 (joint):    full end-to-end backprop through quantum circuit

Logging: TensorBoard + console
"""

import time
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .losses import PUFLoss
from .callbacks import EarlyStopping, BestModelCheckpoint

logger = logging.getLogger(__name__)


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = (logits > 0).long()
    return (preds == targets.long()).float().mean().item()


class Trainer:
    """
    Two-phase trainer for HybridQCNN.

    Args:
        model:        HybridQCNN instance
        device:       torch.device (cuda recommended)
        cfg:          config dict from yaml
        run_tag:      string identifier for this run (e.g. "3xor")
        output_dir:   base output directory
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        cfg: dict,
        run_tag: str = "run",
        output_dir: str = "results",
    ):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg
        self.run_tag = run_tag
        self.output_dir = Path(output_dir)

        self.ckpt_dir = self.output_dir / "checkpoints" / run_tag
        self.tb_dir = self.output_dir / "tensorboard" / run_tag
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        self.criterion = PUFLoss(lambda_ent=cfg.get("lambda_ent", 0.01))
        self.scaler = GradScaler(enabled=False)  # quantum sim stays float32

        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [],   "val_acc": [],
        }

    # ------------------------------------------------------------------
    def _make_optimizer(self, phase: int) -> torch.optim.Optimizer:
        if phase == 1:
            # Only encoder + head parameters
            params = [
                {"params": self.model.encoder.parameters(), "lr": self.cfg["lr_warmup"]},
                {"params": self.model.head.parameters(),    "lr": self.cfg["lr_warmup"]},
            ]
        else:
            params = [
                {"params": self.model.encoder.parameters(), "lr": self.cfg["lr_joint"]},
                {"params": self.model.vqc.parameters(),     "lr": self.cfg["lr_vqc"]},
                {"params": self.model.head.parameters(),    "lr": self.cfg["lr_joint"]},
            ]
        return AdamW(params, weight_decay=self.cfg.get("weight_decay", 1e-4))

    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, optimizer, train: bool):
        self.model.train(train)
        total_loss = total_acc = 0.0
        n_batches = len(loader)

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            bar = tqdm(loader, desc="train" if train else "val", leave=False, ncols=90)
            for challenges, responses in bar:
                challenges = challenges.to(self.device, non_blocking=True)
                responses = responses.to(self.device, non_blocking=True)

                logits = self.model(challenges)
                loss_tuple = self.criterion(logits, responses)
                loss = loss_tuple[0]

                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                acc = _accuracy(logits.detach(), responses)
                total_loss += loss.item()
                total_acc += acc
                bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

        return total_loss / n_batches, total_acc / n_batches

    # ------------------------------------------------------------------
    def train_phase(
        self,
        phase: int,
        n_epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint: Optional[BestModelCheckpoint] = None,
        early_stop: Optional[EarlyStopping] = None,
        global_step_offset: int = 0,
    ) -> int:
        """Run one training phase. Returns number of epochs completed."""
        if phase == 1:
            logger.info(f"=== Phase 1: Warm-up ({n_epochs} epochs, VQC frozen) ===")
            self.model.freeze_vqc()
        else:
            logger.info(f"=== Phase 2: Joint training ({n_epochs} epochs, all unfrozen) ===")
            self.model.unfreeze_vqc()

        optimizer = self._make_optimizer(phase)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

        global_step = global_step_offset
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._run_epoch(train_loader, optimizer, train=True)
            val_loss,   val_acc   = self._run_epoch(val_loader, None, train=False)
            scheduler.step()
            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            self.writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, global_step)
            self.writer.add_scalars("acc",  {"train": train_acc,  "val": val_acc},  global_step)
            self.writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

            logger.info(
                f"[Phase {phase}] Ep {epoch:3d}/{n_epochs} | "
                f"loss {train_loss:.4f}/{val_loss:.4f} | "
                f"acc {train_acc:.4f}/{val_acc:.4f} | {elapsed:.1f}s"
            )

            if checkpoint:
                checkpoint.step(self.model, val_acc, epoch, tag=f"ph{phase}_{self.run_tag}")
            if early_stop and early_stop.step(val_loss):
                logger.info("Early stopping triggered.")
                break

        return global_step

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        """Full two-phase training."""
        cfg = self.cfg
        ckpt = BestModelCheckpoint(str(self.ckpt_dir), metric_name="val_acc", mode="max")
        es   = EarlyStopping(patience=cfg.get("early_stop_patience", 12), mode="min")

        # Phase 1
        gs = self.train_phase(
            phase=1,
            n_epochs=cfg["epochs_warmup"],
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint=ckpt,
        )

        # Phase 2
        self.train_phase(
            phase=2,
            n_epochs=cfg["epochs_joint"],
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint=ckpt,
            early_stop=es,
            global_step_offset=gs,
        )

        self.writer.close()
        logger.info(f"Training complete. Best checkpoint: {ckpt.best_path}")
        return ckpt.best_path, self.history
