"""
Simple CNN baseline for image classification.

This model is intentionally simple so we can compare it fairly against ViT.
Architecture:
    image
    -> Conv2d (in_channels -> 32 filters, 3x3)
    -> BatchNorm + ReLU
    -> MaxPool (2x2)
    -> Conv2d (32 -> 64 filters, 3x3)
    -> BatchNorm + ReLU
    -> MaxPool (2x2)
    -> Flatten
    -> Linear(64 * (H/4) * (W/4), 256)
    -> ReLU + Dropout
    -> Linear(256, num_classes)
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Two-layer CNN for image classification.

    Args:
        in_channels: Number of input channels. 1 for grayscale, 3 for RGB.
        num_classes: Number of output classes.
        image_size: Input image height/width (assumed square).
        dropout: Dropout probability before the final linear layer.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        image_size: int = 28,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # ─── Convolutional blocks ─────────────────────────────────────────────
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # [B, 32, H, W]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [B, 32, H/2, W/2]
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, H/2, W/2]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [B, 64, H/4, W/4]
        )

        # ─── Classifier ───────────────────────────────────────────────────────
        # After two MaxPool(2,2) layers, spatial size = image_size // 4
        spatial_size = image_size // 4
        flat_dim = 64 * spatial_size * spatial_size

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the CNN classifier.

        Args:
            x: [batch, in_channels, H, W]

        Returns:
            logits: [batch, num_classes]
        """
        x = self.conv_block1(x)   # [B, 32, H/2, W/2]
        x = self.conv_block2(x)   # [B, 64, H/4, W/4]
        logits = self.classifier(x)  # [B, num_classes]
        return logits

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
