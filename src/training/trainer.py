"""
Generic trainer that wraps the training loop with logging and history.

Each task-specific module (classification, translation, etc.) builds its own
loss_fn and metric_fn, then passes them to this trainer.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.loops import train_one_epoch, evaluate


class Trainer:
    """General-purpose training manager.

    Args:
        model: The model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer instance.
        loss_fn: Callable (model, batch, device) -> scalar loss.
        device: Target device.
        metric_fn: Optional callable (model, batch, device) -> dict of metrics.
        scheduler: Optional learning rate scheduler (step called per epoch).
        clip_grad_norm: Gradient clipping value. None to disable.
        use_amp: Use Automatic Mixed Precision (CUDA only).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: torch.device,
        metric_fn: Optional[Callable] = None,
        scheduler=None,
        clip_grad_norm: float = 1.0,
        use_amp: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.metric_fn = metric_fn
        self.scheduler = scheduler
        self.clip_grad_norm = clip_grad_norm

        # AMP is only used with CUDA
        self.scaler = None
        if use_amp and device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()

        # History for plotting
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_metrics: List[Dict] = []

    def train(self, epochs: int, verbose: bool = True) -> Dict:
        """Run the full training loop.

        Args:
            epochs: Number of epochs to train.
            verbose: Print progress each epoch.

        Returns:
            Dictionary with final metrics and training history.
        """
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            # Training
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.loss_fn,
                self.device,
                clip_grad_norm=self.clip_grad_norm,
                scaler=self.scaler,
            )

            # Validation
            val_loss, metrics = evaluate(
                self.model,
                self.val_loader,
                self.loss_fn,
                self.device,
                metric_fn=self.metric_fn,
            )

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            if metrics:
                self.val_metrics.append(metrics)

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if verbose:
                metrics_str = ""
                if metrics:
                    metrics_str = "  " + "  ".join(
                        f"{k}={v:.4f}" for k, v in metrics.items()
                    )
                print(
                    f"Epoch {epoch:3d}/{epochs}  "
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
                    f"{metrics_str}"
                )

        final_metrics = {"train_loss": self.train_losses[-1], "val_loss": self.val_losses[-1]}
        if self.val_metrics:
            final_metrics.update(self.val_metrics[-1])

        return final_metrics

    def get_history(self) -> Dict:
        """Return training history for plotting.

        Returns:
            dict with 'train_loss', 'val_loss', and any metric lists.
        """
        history = {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
        }
        if self.val_metrics:
            # Collect metrics by key
            for key in self.val_metrics[0]:
                history[key] = [m[key] for m in self.val_metrics]
        return history
