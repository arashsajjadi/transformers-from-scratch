"""
Tests for attention module output shapes.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add repo root to path so src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.attention import ScaledDotProductAttention, MultiHeadAttention


class TestScaledDotProductAttention:
    """Tests for ScaledDotProductAttention."""

    def setup_method(self):
        self.batch = 2
        self.heads = 4
        self.seq_q = 5
        self.seq_k = 6
        self.d_k = 8
        self.d_v = 8
        self.attn = ScaledDotProductAttention(dropout=0.0)

    def test_output_shape(self):
        """Output should be [batch, heads, seq_q, d_v]."""
        Q = torch.randn(self.batch, self.heads, self.seq_q, self.d_k)
        K = torch.randn(self.batch, self.heads, self.seq_k, self.d_k)
        V = torch.randn(self.batch, self.heads, self.seq_k, self.d_v)

        output, weights = self.attn(Q, K, V)

        assert output.shape == (self.batch, self.heads, self.seq_q, self.d_v)

    def test_weights_shape(self):
        """Attention weights should be [batch, heads, seq_q, seq_k]."""
        Q = torch.randn(self.batch, self.heads, self.seq_q, self.d_k)
        K = torch.randn(self.batch, self.heads, self.seq_k, self.d_k)
        V = torch.randn(self.batch, self.heads, self.seq_k, self.d_v)

        _, weights = self.attn(Q, K, V)

        assert weights.shape == (self.batch, self.heads, self.seq_q, self.seq_k)

    def test_weights_sum_to_one(self):
        """Attention weights should sum to approximately 1 over the key dimension."""
        Q = torch.randn(self.batch, self.heads, self.seq_q, self.d_k)
        K = torch.randn(self.batch, self.heads, self.seq_k, self.d_k)
        V = torch.randn(self.batch, self.heads, self.seq_k, self.d_v)

        _, weights = self.attn(Q, K, V)
        sums = weights.sum(dim=-1)  # [batch, heads, seq_q]

        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_causal_mask_blocks_future(self):
        """With a causal mask, upper-triangle weights should be zero."""
        seq = 4
        Q = torch.randn(1, 1, seq, self.d_k)
        K = torch.randn(1, 1, seq, self.d_k)
        V = torch.randn(1, 1, seq, self.d_k)

        # Upper-triangle = True means mask those positions
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]

        _, weights = self.attn(Q, K, V, mask=mask)

        # Upper-triangle weights should be ~0
        upper = weights[0, 0][mask[0, 0]]
        assert torch.all(upper < 1e-6)

    def test_self_attention_square_sequence(self):
        """Q, K, V from same sequence gives square attention matrix."""
        seq = 7
        d = 16
        x = torch.randn(1, 1, seq, d)
        _, weights = self.attn(x, x, x)
        assert weights.shape == (1, 1, seq, seq)


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""

    def setup_method(self):
        self.batch = 2
        self.seq = 8
        self.d_model = 64
        self.num_heads = 4
        self.mha = MultiHeadAttention(self.d_model, self.num_heads, dropout=0.0)

    def test_self_attention_output_shape(self):
        """Self-attention output should be [batch, seq, d_model]."""
        x = torch.randn(self.batch, self.seq, self.d_model)
        output, _ = self.mha(x, x, x)
        assert output.shape == (self.batch, self.seq, self.d_model)

    def test_cross_attention_output_shape(self):
        """Cross-attention output should match query shape."""
        seq_q = 5
        seq_k = 10
        query = torch.randn(self.batch, seq_q, self.d_model)
        key = torch.randn(self.batch, seq_k, self.d_model)
        value = torch.randn(self.batch, seq_k, self.d_model)

        output, _ = self.mha(query, key, value)
        assert output.shape == (self.batch, seq_q, self.d_model)

    def test_return_weights(self):
        """When return_weights=True, weights tensor should have correct shape."""
        x = torch.randn(self.batch, self.seq, self.d_model)
        _, weights = self.mha(x, x, x, return_weights=True)

        assert weights is not None
        assert weights.shape == (self.batch, self.num_heads, self.seq, self.seq)

    def test_no_weights_by_default(self):
        """By default, weights should be None."""
        x = torch.randn(self.batch, self.seq, self.d_model)
        _, weights = self.mha(x, x, x)
        assert weights is None

    def test_wrong_d_model_raises(self):
        """d_model not divisible by num_heads should raise AssertionError."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=65, num_heads=4)

    def test_gradient_flows(self):
        """Gradients should flow back through the attention layer."""
        x = torch.randn(2, 4, self.d_model, requires_grad=True)
        output, _ = self.mha(x, x, x)
        output.sum().backward()
        assert x.grad is not None
