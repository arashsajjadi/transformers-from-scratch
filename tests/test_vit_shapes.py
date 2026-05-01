"""
Tests for Vision Transformer output shapes.
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.vit import PatchEmbedding, VisionTransformer


class TestPatchEmbedding:
    """Tests for PatchEmbedding."""

    def test_output_shape_grayscale(self):
        """Grayscale image (1 channel) should produce correct patch tokens."""
        batch = 2
        image_size = 28
        patch_size = 7
        in_channels = 1
        d_model = 64

        pe = PatchEmbedding(image_size, patch_size, in_channels, d_model)
        x = torch.randn(batch, in_channels, image_size, image_size)
        out = pe(x)

        num_patches = (image_size // patch_size) ** 2
        assert out.shape == (batch, num_patches, d_model)

    def test_output_shape_rgb(self):
        """RGB image (3 channels) should produce correct patch tokens."""
        batch = 2
        image_size = 32
        patch_size = 8
        in_channels = 3
        d_model = 128

        pe = PatchEmbedding(image_size, patch_size, in_channels, d_model)
        x = torch.randn(batch, in_channels, image_size, image_size)
        out = pe(x)

        num_patches = (image_size // patch_size) ** 2
        assert out.shape == (batch, num_patches, d_model)

    def test_num_patches(self):
        """num_patches should equal (image_size / patch_size)^2."""
        pe = PatchEmbedding(image_size=28, patch_size=7, in_channels=1, d_model=64)
        assert pe.num_patches == 16  # 4 * 4

    def test_patch_size_not_divisible_raises(self):
        """patch_size that doesn't divide image_size should raise AssertionError."""
        with pytest.raises(AssertionError):
            PatchEmbedding(image_size=28, patch_size=6, in_channels=1, d_model=64)

    def test_gradient_flows(self):
        """Gradients should flow through the patch projection."""
        pe = PatchEmbedding(28, 7, 1, 64)
        x = torch.randn(1, 1, 28, 28, requires_grad=True)
        out = pe(x)
        out.sum().backward()
        assert x.grad is not None


class TestVisionTransformer:
    """Tests for VisionTransformer."""

    def test_logits_shape(self):
        """Output logits shape should be [batch, num_classes]."""
        batch = 2
        image_size = 28
        patch_size = 7
        in_channels = 1
        num_classes = 10

        vit = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            d_model=64,
            num_heads=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
        )
        x = torch.randn(batch, in_channels, image_size, image_size)
        logits = vit(x)

        assert logits.shape == (batch, num_classes)

    def test_logits_shape_rgb(self):
        """Works with 3-channel input."""
        batch = 2
        vit = VisionTransformer(
            image_size=32, patch_size=8, in_channels=3, num_classes=5,
            d_model=32, num_heads=4, num_layers=1, dim_feedforward=64, dropout=0.0,
        )
        x = torch.randn(batch, 3, 32, 32)
        logits = vit(x)
        assert logits.shape == (batch, 5)

    def test_return_patch_tokens(self):
        """When return_patch_tokens=True, also returns patch token tensor."""
        vit = VisionTransformer(
            image_size=28, patch_size=7, in_channels=1, num_classes=10,
            d_model=64, num_heads=4, num_layers=2, dim_feedforward=128, dropout=0.0,
        )
        x = torch.randn(1, 1, 28, 28)
        logits, patch_tokens = vit(x, return_patch_tokens=True)

        num_patches = (28 // 7) ** 2
        assert logits.shape == (1, 10)
        assert patch_tokens.shape == (1, num_patches, 64)

    def test_gradient_flows(self):
        """Gradients should flow from the loss back to the inputs."""
        vit = VisionTransformer(
            image_size=28, patch_size=7, in_channels=1, num_classes=10,
            d_model=32, num_heads=4, num_layers=1, dim_feedforward=64, dropout=0.0,
        )
        x = torch.randn(2, 1, 28, 28, requires_grad=True)
        logits = vit(x)
        logits.sum().backward()
        assert x.grad is not None

    def test_parameter_count(self):
        """Parameter count should be positive and reasonable for a tiny model."""
        vit = VisionTransformer(
            image_size=28, patch_size=7, in_channels=1, num_classes=10,
            d_model=64, num_heads=4, num_layers=2, dim_feedforward=128,
        )
        n_params = vit.count_parameters()
        assert n_params > 0
        assert n_params < 10_000_000  # should be a small model
