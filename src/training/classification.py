"""
Loss and metric functions for classification tasks (text and image).
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.device import move_batch_to_device


def classification_loss_fn(model: nn.Module, batch, device: torch.device) -> torch.Tensor:
    """Cross-entropy loss for text or image classification.

    Batch format: (inputs, labels) where inputs can be token IDs or image tensors.

    Args:
        model: Classification model.
        batch: Tuple of (inputs, labels).
        device: Target device.

    Returns:
        scalar loss tensor.
    """
    inputs, labels = batch
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Handle models that return (logits, extra) or just logits
    output = model(inputs)
    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output

    return F.cross_entropy(logits, labels)


def classification_metric_fn(model: nn.Module, batch, device: torch.device) -> Dict[str, float]:
    """Compute accuracy for a batch.

    Args:
        model: Classification model.
        batch: Tuple of (inputs, labels).
        device: Target device.

    Returns:
        dict with 'accuracy'.
    """
    inputs, labels = batch
    inputs = inputs.to(device)
    labels = labels.to(device)

    output = model(inputs)
    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output

    preds = logits.argmax(dim=-1)
    acc = (preds == labels).float().mean().item()
    return {"accuracy": acc}


def collect_predictions(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect all predictions and labels from a DataLoader.

    Args:
        model: Classification model.
        loader: DataLoader.
        device: Target device.

    Returns:
        (all_preds, all_labels) as 1D tensors.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_preds), torch.cat(all_labels)
