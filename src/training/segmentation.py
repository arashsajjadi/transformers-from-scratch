"""
Loss and metric functions for semantic segmentation.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics.segmentation import pixel_accuracy, mean_iou, dice_score


def segmentation_loss_fn(model: nn.Module, batch, device: torch.device) -> torch.Tensor:
    """Cross-entropy loss for semantic segmentation.

    Batch format: (images, masks).

    Args:
        model: Segmentation model (UNet or ViTSegmenter).
        batch: (images [B, C, H, W], masks [B, H, W]).
        device: Target device.

    Returns:
        scalar cross-entropy loss.
    """
    images, masks = batch
    images = images.to(device)
    masks = masks.to(device)

    logits = model(images)  # [B, num_classes, H, W]
    return F.cross_entropy(logits, masks)


def segmentation_metric_fn(model: nn.Module, batch, device: torch.device) -> Dict[str, float]:
    """Compute pixel accuracy, mean IoU, and Dice for a batch.

    Args:
        model: Segmentation model.
        batch: (images, masks).
        device: Target device.

    Returns:
        dict with 'pixel_accuracy', 'mean_iou', 'dice'.
    """
    images, masks = batch
    images = images.to(device)
    masks = masks.to(device)

    logits = model(images)  # [B, num_classes, H, W]
    preds = logits.argmax(dim=1)  # [B, H, W]

    pa = pixel_accuracy(preds, masks)
    miou = mean_iou(preds, masks, num_classes=logits.size(1))
    dice = dice_score(preds, masks, num_classes=logits.size(1))

    return {"pixel_accuracy": pa, "mean_iou": miou, "dice": dice}
