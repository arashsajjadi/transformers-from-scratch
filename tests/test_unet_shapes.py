"""
Tests for U-Net output shapes.
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.unet import UNet, DoubleConv, DownBlock, UpBlock


class TestDoubleConv:
    """Tests for the DoubleConv building block."""

    def test_output_shape(self):
        """Output should have same spatial size as input."""
        dc = DoubleConv(3, 32)
        x = torch.randn(2, 3, 64, 64)
        out = dc(x)
        assert out.shape == (2, 32, 64, 64)

    def test_channel_change(self):
        """Output channels should match out_channels argument."""
        dc = DoubleConv(16, 64)
        x = torch.randn(1, 16, 32, 32)
        out = dc(x)
        assert out.shape[1] == 64


class TestDownBlock:
    """Tests for DownBlock (encoder step)."""

    def test_returns_skip_and_pooled(self):
        """Should return skip (same size) and pooled (half size)."""
        down = DownBlock(3, 32)
        x = torch.randn(1, 3, 64, 64)
        skip, pooled = down(x)

        assert skip.shape == (1, 32, 64, 64)
        assert pooled.shape == (1, 32, 32, 32)


class TestUpBlock:
    """Tests for UpBlock (decoder step)."""

    def test_output_shape(self):
        """After upsampling + skip concatenation, spatial size should double."""
        up = UpBlock(in_channels=64, skip_channels=32, out_channels=32)
        x = torch.randn(1, 64, 16, 16)     # decoder input
        skip = torch.randn(1, 32, 32, 32)  # skip connection from encoder
        out = up(x, skip)
        assert out.shape == (1, 32, 32, 32)


class TestUNet:
    """Tests for the full U-Net model."""

    @pytest.mark.parametrize("image_size", [64, 128])
    def test_output_shape_equals_input(self, image_size):
        """Output segmentation map should have same spatial size as input image."""
        batch = 2
        in_channels = 3
        num_classes = 4

        unet = UNet(in_channels=in_channels, num_classes=num_classes, base_channels=16)
        x = torch.randn(batch, in_channels, image_size, image_size)
        logits = unet(x)

        assert logits.shape == (batch, num_classes, image_size, image_size)

    def test_grayscale_input(self):
        """Works with 1-channel input."""
        unet = UNet(in_channels=1, num_classes=4, base_channels=16)
        x = torch.randn(1, 1, 64, 64)
        logits = unet(x)
        assert logits.shape == (1, 4, 64, 64)

    def test_gradient_flows(self):
        """Gradients should flow from the output back to the input."""
        unet = UNet(in_channels=3, num_classes=4, base_channels=8)
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        logits = unet(x)
        logits.sum().backward()
        assert x.grad is not None

    def test_parameter_count_positive(self):
        """Model should have trainable parameters."""
        unet = UNet(in_channels=3, num_classes=4, base_channels=16)
        assert unet.count_parameters() > 0
