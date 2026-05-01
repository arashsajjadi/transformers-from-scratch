"""
Generic training and evaluation loops.

These functions handle one epoch of training or evaluation.
They are called by task-specific trainers.
"""

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: torch.device,
    clip_grad_norm: Optional[float] = 1.0,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    """Run one full training epoch.

    Args:
        model: The model to train.
        loader: DataLoader for training data.
        optimizer: Optimizer.
        loss_fn: Callable that takes (model, batch) and returns a scalar loss.
        device: Target device.
        clip_grad_norm: Max gradient norm for clipping. None to disable.
        scaler: Optional AMP GradScaler for mixed precision. CUDA only.

    Returns:
        Average loss over all batches.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="  train", leave=False):
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = loss_fn(model, batch, device)
            scaler.scale(loss).backward()
            if clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss_fn(model, batch, device)
            loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    device: torch.device,
    metric_fn: Optional[Callable] = None,
) -> Tuple[float, Optional[Dict]]:
    """Run one evaluation epoch.

    Args:
        model: The model to evaluate.
        loader: DataLoader for validation/test data.
        loss_fn: Same callable as used in training.
        device: Target device.
        metric_fn: Optional callable that takes (model, batch, device) and
                   returns a dict of metric values.

    Returns:
        avg_loss: Average loss over all batches.
        metrics: dict from metric_fn, or None.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_metrics = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval ", leave=False):
            loss = loss_fn(model, batch, device)
            total_loss += loss.item()
            n_batches += 1

            if metric_fn is not None:
                batch_metrics = metric_fn(model, batch, device)
                for k, v in batch_metrics.items():
                    all_metrics.setdefault(k, []).append(v)

    avg_loss = total_loss / max(n_batches, 1)

    if all_metrics:
        averaged = {k: sum(v) / len(v) for k, v in all_metrics.items()}
        return avg_loss, averaged

    return avg_loss, None
