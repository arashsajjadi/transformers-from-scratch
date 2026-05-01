"""
Segmentation metrics.

Implements:
  - pixel_accuracy
  - intersection_over_union (per class)
  - mean_iou
  - dice_score
"""

import torch


def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of correctly classified pixels.

    Args:
        preds: [B, H, W] predicted class indices.
        targets: [B, H, W] true class indices.

    Returns:
        Pixel accuracy as float in [0, 1].
    """
    correct = (preds == targets).float().sum()
    total = targets.numel()
    return (correct / total).item()


def intersection_over_union(
    preds: torch.Tensor,
    targets: torch.Tensor,
    class_id: int,
) -> float:
    """IoU for a single class.

    IoU = (pred == c AND target == c) / (pred == c OR target == c)

    Args:
        preds: [B, H, W] predicted class indices.
        targets: [B, H, W] true class indices.
        class_id: Which class to compute IoU for.

    Returns:
        IoU as float in [0, 1]. Returns 0 if the class is absent in both.
    """
    pred_mask = (preds == class_id)
    target_mask = (targets == class_id)

    intersection = (pred_mask & target_mask).float().sum().item()
    union = (pred_mask | target_mask).float().sum().item()

    if union == 0:
        return float("nan")  # class absent in both pred and target
    return intersection / union


def mean_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_nan: bool = True,
) -> float:
    """Mean IoU across all classes (excluding absent classes by default).

    Args:
        preds: [B, H, W] predicted class indices.
        targets: [B, H, W] true class indices.
        num_classes: Total number of classes.
        ignore_nan: If True, skip classes that are absent in both pred and target.

    Returns:
        Mean IoU as float in [0, 1].
    """
    ious = []
    for c in range(num_classes):
        iou = intersection_over_union(preds, targets, c)
        if not (ignore_nan and iou != iou):  # skip NaN
            ious.append(iou)

    if not ious:
        return 0.0
    return sum(ious) / len(ious)


def dice_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    smooth: float = 1.0,
) -> float:
    """Mean Dice score across all classes.

    Dice = 2 * |P ∩ G| / (|P| + |G|)

    Args:
        preds: [B, H, W] predicted class indices.
        targets: [B, H, W] true class indices.
        num_classes: Total number of classes.
        smooth: Smoothing constant to avoid division by zero.

    Returns:
        Mean Dice score as float in [0, 1].
    """
    dice_scores = []
    for c in range(num_classes):
        pred_mask = (preds == c).float()
        target_mask = (targets == c).float()

        intersection = (pred_mask * target_mask).sum().item()
        pred_sum = pred_mask.sum().item()
        target_sum = target_mask.sum().item()

        if pred_sum + target_sum == 0:
            continue  # class absent in both

        d = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
        dice_scores.append(d)

    if not dice_scores:
        return 0.0
    return sum(dice_scores) / len(dice_scores)
