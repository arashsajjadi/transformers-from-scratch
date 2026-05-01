"""
Tests for positional encoding modules.
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    Simple2DPositionalEncoding,
)


class TestSinusoidalPositionalEncoding:
    """Tests for SinusoidalPositionalEncoding."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        d_model = 64
        max_len = 128
        batch = 2
        seq_len = 10

        pe = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)
        x = torch.zeros(batch, seq_len, d_model)
        out = pe(x)

        assert out.shape == (batch, seq_len, d_model)

    def test_encoding_is_added(self):
        """Output should differ from input due to positional encoding."""
        d_model = 32
        pe = SinusoidalPositionalEncoding(d_model, max_len=64, dropout=0.0)
        x = torch.zeros(1, 8, d_model)
        out = pe(x)
        # At least some values should be non-zero
        assert not torch.allclose(out, x)

    def test_different_positions_have_different_encodings(self):
        """Adjacent positions should have different encoding vectors."""
        d_model = 64
        pe = SinusoidalPositionalEncoding(d_model, max_len=128, dropout=0.0)
        x = torch.zeros(1, 10, d_model)
        out = pe(x)

        # Positions 0 and 1 should differ
        assert not torch.allclose(out[0, 0], out[0, 1])

    def test_no_learned_parameters(self):
        """Sinusoidal encoding has no trainable parameters."""
        pe = SinusoidalPositionalEncoding(d_model=64, max_len=128)
        params = list(pe.parameters())
        assert len(params) == 0

    def test_short_sequence(self):
        """Works with sequence length of 1."""
        d_model = 16
        pe = SinusoidalPositionalEncoding(d_model, max_len=64, dropout=0.0)
        x = torch.zeros(1, 1, d_model)
        out = pe(x)
        assert out.shape == (1, 1, d_model)


class TestLearnedPositionalEncoding:
    """Tests for LearnedPositionalEncoding."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        d_model = 64
        max_len = 128
        batch = 3
        seq_len = 12

        pe = LearnedPositionalEncoding(d_model, max_len, dropout=0.0)
        x = torch.zeros(batch, seq_len, d_model)
        out = pe(x)

        assert out.shape == (batch, seq_len, d_model)

    def test_has_learned_parameters(self):
        """Learned encoding should have trainable parameters."""
        pe = LearnedPositionalEncoding(d_model=64, max_len=128)
        params = list(pe.parameters())
        assert len(params) > 0

    def test_gradient_flows(self):
        """Gradients should flow through the learned embeddings."""
        d_model = 32
        pe = LearnedPositionalEncoding(d_model, max_len=32, dropout=0.0)
        x = torch.randn(1, 8, d_model)
        out = pe(x)
        out.sum().backward()
        # Check that position embedding has grad
        assert pe.position_embedding.weight.grad is not None


class TestSimple2DPositionalEncoding:
    """Tests for Simple2DPositionalEncoding."""

    def test_output_shape(self):
        """Output shape should match input."""
        d_model = 64
        h, w = 4, 4
        batch = 2
        num_patches = h * w

        pe = Simple2DPositionalEncoding(d_model, h, w, dropout=0.0)
        x = torch.zeros(batch, num_patches, d_model)
        out = pe(x)

        assert out.shape == (batch, num_patches, d_model)

    def test_rectangular_grid(self):
        """Works with non-square patch grids."""
        d_model = 32
        h, w = 4, 8
        num_patches = h * w

        pe = Simple2DPositionalEncoding(d_model, h, w, dropout=0.0)
        x = torch.zeros(1, num_patches, d_model)
        out = pe(x)

        assert out.shape == (1, num_patches, d_model)
