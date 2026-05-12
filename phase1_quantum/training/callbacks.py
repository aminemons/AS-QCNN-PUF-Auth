"""Early stopping and LR scheduling callbacks."""

import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stops training when val loss does not improve for `patience` epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.triggered = False

    def step(self, metric: float) -> bool:
        improved = (
            metric < self.best - self.min_delta
            if self.mode == "min"
            else metric > self.best + self.min_delta
        )
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement.")
        return self.triggered


class BestModelCheckpoint:
    """Saves checkpoint when validation metric improves."""

    def __init__(self, save_dir: str, metric_name: str = "val_acc", mode: str = "max"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.best_path = None

    def step(self, model: torch.nn.Module, metric: float, epoch: int, tag: str = "") -> bool:
        improved = (
            metric > self.best if self.mode == "max" else metric < self.best
        )
        if improved:
            self.best = metric
            fname = f"best_{tag}_ep{epoch:03d}_{self.metric_name}_{metric:.4f}.pt"
            path = self.save_dir / fname
            torch.save(model.state_dict(), path)
            self.best_path = path
            logger.info(f"Checkpoint saved: {path}")
            return True
        return False
