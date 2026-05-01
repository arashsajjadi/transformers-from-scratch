"""
Attention mechanisms built from scratch.

Implements:
  - ScaledDotProductAttention
  - MultiHeadAttention

These are the core building blocks of every Transformer model in this course.

Key paper: "Attention Is All You Need" (Vaswani et al., 2017)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Compute scaled dot-product attention.

    Formula:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    The scaling factor 1/sqrt(d_k) prevents the dot products from growing too
    large as d_k increases, which would push the softmax into regions with
    very small gradients.

    Args:
        dropout: Dropout probability applied to attention weights. Default 0.0.
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run scaled dot-product attention.

        Args:
            query:  [batch, heads, seq_q, d_k]
            key:    [batch, heads, seq_k, d_k]
            value:  [batch, heads, seq_k, d_v]
            mask:   Optional boolean mask of shape [batch, heads, seq_q, seq_k]
                    or broadcastable to that shape.
                    True (or 1) means "mask this position out" (set to -inf).

        Returns:
            output:           [batch, heads, seq_q, d_v]
            attention_weights: [batch, heads, seq_q, seq_k]
        """
        d_k = query.size(-1)  # dimension of each key/query vector

        # Step 1: Compute raw scores — [batch, heads, seq_q, seq_k]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Step 2: Apply mask (set masked positions to -infinity before softmax)
        if mask is not None:
            scores = scores.masked_fill(mask.bool(), float("-inf"))

        # Step 3: Softmax across the key dimension → attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [batch, heads, seq_q, seq_k]

        # Avoid NaN in rows where all positions are masked
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

        # Step 4: Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)

        # Step 5: Weighted sum of values — [batch, heads, seq_q, d_v]
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer.

    Instead of computing a single attention function with d_model-dimensional
    keys, queries, and values, we project them into num_heads independent
    d_k-dimensional subspaces and compute attention in parallel.

    This allows the model to jointly attend to information from different
    representation subspaces at different positions.

    d_k = d_model // num_heads

    Architecture:
        input
        -> linear Q, K, V projections
        -> split into num_heads
        -> scaled dot-product attention (per head)
        -> concatenate heads
        -> linear output projection

    Args:
        d_model: Total embedding dimension. Must be divisible by num_heads.
        num_heads: Number of parallel attention heads.
        dropout: Dropout probability. Default 0.0.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        # Linear projections for Q, K, V and output
        # We use separate weights for each head (stacked into one big matrix for efficiency)
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # projects Q
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # projects K
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # projects V
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # output projection

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]."""
        batch_size, seq_len, _ = x.size()
        # Step 1: [batch, seq_len, num_heads, d_k]
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # Step 2: [batch, num_heads, seq_len, d_k]
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape [batch, num_heads, seq_len, d_k] -> [batch, seq_len, d_model]."""
        batch_size, _, seq_len, _ = x.size()
        # Step 1: [batch, seq_len, num_heads, d_k]
        x = x.transpose(1, 2).contiguous()
        # Step 2: [batch, seq_len, d_model]
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run multi-head attention.

        Args:
            query:  [batch, seq_q, d_model]
            key:    [batch, seq_k, d_model]
            value:  [batch, seq_k, d_model]
            mask:   Optional mask [batch, 1, seq_q, seq_k] or broadcastable.
            return_weights: If True, return attention weights tensor.

        Returns:
            output:  [batch, seq_q, d_model]
            weights: [batch, num_heads, seq_q, seq_k] if return_weights else None
        """
        # Step 1: Linear projections
        Q = self.w_q(query)  # [batch, seq_q, d_model]
        K = self.w_k(key)    # [batch, seq_k, d_model]
        V = self.w_v(value)  # [batch, seq_k, d_model]

        # Step 2: Split into heads → [batch, num_heads, seq, d_k]
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # Step 3: Scaled dot-product attention per head
        attended, weights = self.attention(Q, K, V, mask=mask)
        # attended: [batch, num_heads, seq_q, d_k]
        # weights:  [batch, num_heads, seq_q, seq_k]

        # Step 4: Merge heads back → [batch, seq_q, d_model]
        attended = self._merge_heads(attended)

        # Step 5: Final linear projection
        output = self.w_o(attended)  # [batch, seq_q, d_model]

        if return_weights:
            return output, weights
        return output, None
