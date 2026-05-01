"""
Segmentation loss functions.

Implements:
  - DiceLoss
  - combined_cross_entropy_dice_loss

DiceLoss:
    Directly optimizes the overlap between predicted and ground-truth regions.
    Better than cross-entropy when classes are heavily imbalanced (e.g., small objects).

    Dice coefficient = 2 * |P ∩ G| / (|P| + |G|)
    Dice loss = 1 - Dice coefficient

The default training in this course uses CrossEntropyLoss for simplicity.
DiceLoss is explained here for educational purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss for semantic segmentation.

    Works by computing the overlap between the predicted soft probabilities
    and the one-hot encoded ground truth.

    Args:
        num_classes: Number of segmentation classes.
        smooth: Smoothing constant to avoid division by zero. Default 1.0.
        ignore_index: Class index to exclude from loss. Default -1 (no exclusion).
    """

    def __init__(
        self, num_classes: int, smooth: float = 1.0, ignore_index: int = -1
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            logits: [B, num_classes, H, W] — raw model output.
            targets: [B, H, W] — integer class labels.

        Returns:
            Scalar Dice loss.
        """
        # Convert logits to soft probabilities
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        # One-hot encode targets: [B, H, W] -> [B, C, H, W]
        B, C, H, W = probs.shape
        targets_onehot = F.one_hot(targets.clamp(min=0), num_classes=C)  # [B, H, W, C]
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()     # [B, C, H, W]

        # Build mask for pixels to include
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).unsqueeze(1).float()  # [B, 1, H, W]
            probs = probs * mask
            targets_onehot = targets_onehot * mask

        # Sum over spatial dimensions H, W
        intersection = (probs * targets_onehot).sum(dim=(2, 3))  # [B, C]
        union = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))  # [B, C]

        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)  # [B, C]

        # Average over classes and batch
        return 1.0 - dice_per_class.mean()


def combined_cross_entropy_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    alpha: float = 0.5,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Combine Cross-Entropy and Dice losses.

    total_loss = alpha * CE_loss + (1 - alpha) * Dice_loss

    This combination can help when some classes are small or imbalanced.

    Args:
        logits: [B, num_classes, H, W]
        targets: [B, H, W]
        num_classes: Number of classes.
        alpha: Weight for cross-entropy (1-alpha for Dice). Default 0.5.
        smooth: Smoothing for Dice. Default 1.0.

    Returns:
        Scalar combined loss.
    """
    ce_loss = F.cross_entropy(logits, targets)
    dice_fn = DiceLoss(num_classes, smooth)
    dice = dice_fn(logits, targets)
    return alpha * ce_loss + (1 - alpha) * dice
