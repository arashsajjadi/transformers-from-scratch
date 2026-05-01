"""
U-Net for semantic segmentation.

U-Net has an encoder (downsampling path) and a decoder (upsampling path).
The key feature is skip connections that pass encoder feature maps
directly to the corresponding decoder level.

This preserves spatial information that is lost during downsampling.

Architecture:
    image [B, C, H, W]
    -> DownBlock1 → features1, [B, 32, H/2, W/2]
    -> DownBlock2 → features2, [B, 64, H/4, W/4]
    -> DownBlock3 → features3, [B, 128, H/8, W/8]
    -> Bottleneck  → [B, 256, H/8, W/8]
    -> UpBlock3 (+ skip from features3) → [B, 128, H/4, W/4]
    -> UpBlock2 (+ skip from features2) → [B, 64, H/2, W/2]
    -> UpBlock1 (+ skip from features1) → [B, 32, H, W]
    -> Conv1x1 → [B, num_classes, H, W]

Reference: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
           (Ronneberger et al., 2015)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv → BatchNorm → ReLU blocks.

    This is the basic building block used in U-Net.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Encoder block: DoubleConv followed by MaxPool.

    Returns both the feature map (before pooling) and the pooled output.
    The feature map before pooling is used as a skip connection.

    Args:
        in_channels: Input channels.
        out_channels: Output channels after DoubleConv.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply double conv, then pool.

        Args:
            x: [B, in_channels, H, W]

        Returns:
            skip: [B, out_channels, H, W]  (before pooling — used as skip connection)
            pooled: [B, out_channels, H/2, W/2]  (after pooling)
        """
        skip = self.conv(x)     # [B, out_channels, H, W]
        pooled = self.pool(skip)  # [B, out_channels, H/2, W/2]
        return skip, pooled


class UpBlock(nn.Module):
    """Decoder block: upsample, concatenate skip connection, then DoubleConv.

    The skip connection from the encoder brings back spatial detail
    that was lost during downsampling.

    Args:
        in_channels: Channels coming in from the previous decoder level.
        skip_channels: Channels from the skip connection.
        out_channels: Output channels after DoubleConv.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        # Bilinear upsampling doubles the spatial size
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # After concat: in_channels + skip_channels
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample, concatenate skip, and apply DoubleConv.

        Args:
            x: [B, in_channels, H, W]  (from previous decoder level)
            skip: [B, skip_channels, 2H, 2W]  (encoder skip connection)

        Returns:
            [B, out_channels, 2H, 2W]
        """
        x = self.upsample(x)  # [B, in_channels, 2H, 2W]

        # Pad if needed (can happen when input size is odd)
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3], 0, skip.shape[2] - x.shape[2]])

        x = torch.cat([skip, x], dim=1)  # [B, skip+in, 2H, 2W]
        x = self.conv(x)                 # [B, out_channels, 2H, 2W]
        return x


class UNet(nn.Module):
    """U-Net for semantic segmentation.

    Args:
        in_channels: Number of input image channels.
        num_classes: Number of segmentation classes.
        base_channels: Number of channels in the first encoder block.
                       Doubles at each downsampling level.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        c = base_channels  # shorthand

        # Encoder (downsampling path)
        self.down1 = DownBlock(in_channels, c)       # -> [B, c, H/2]
        self.down2 = DownBlock(c, c * 2)             # -> [B, 2c, H/4]
        self.down3 = DownBlock(c * 2, c * 4)         # -> [B, 4c, H/8]

        # Bottleneck
        self.bottleneck = DoubleConv(c * 4, c * 8)   # [B, 8c, H/8]

        # Decoder (upsampling path)
        self.up3 = UpBlock(c * 8, c * 4, c * 4)     # + skip from down3
        self.up2 = UpBlock(c * 4, c * 2, c * 2)     # + skip from down2
        self.up1 = UpBlock(c * 2, c, c)              # + skip from down1

        # Final 1x1 convolution to produce class logits
        self.output_conv = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run U-Net segmentation.

        Args:
            x: [B, in_channels, H, W]

        Returns:
            logits: [B, num_classes, H, W]  (same spatial size as input)
        """
        # ─── Encoder ──────────────────────────────────────────────────────────
        skip1, x = self.down1(x)  # skip1: [B, c, H], x: [B, c, H/2]
        skip2, x = self.down2(x)  # skip2: [B, 2c, H/2], x: [B, 2c, H/4]
        skip3, x = self.down3(x)  # skip3: [B, 4c, H/4], x: [B, 4c, H/8]

        # ─── Bottleneck ───────────────────────────────────────────────────────
        x = self.bottleneck(x)  # [B, 8c, H/8]

        # ─── Decoder ──────────────────────────────────────────────────────────
        x = self.up3(x, skip3)  # [B, 4c, H/4]
        x = self.up2(x, skip2)  # [B, 2c, H/2]
        x = self.up1(x, skip1)  # [B, c, H]

        # ─── Output ───────────────────────────────────────────────────────────
        logits = self.output_conv(x)  # [B, num_classes, H, W]

        return logits

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
