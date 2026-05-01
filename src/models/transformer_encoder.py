"""
Transformer encoder: feedforward block, encoder block, and stacked encoder.

Uses pre-norm style (also called Pre-LN):
    x = x + attention(LayerNorm(x))
    x = x + feedforward(LayerNorm(x))

Pre-norm is more stable than post-norm (the original paper's style) and is
used in most modern Transformers.

Architecture of one encoder block:
    input
    -> LayerNorm
    -> Multi-Head Self-Attention
    -> Residual connection
    -> LayerNorm
    -> FeedForward (linear -> GELU -> linear)
    -> Residual connection
    -> output
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network (FFN).

    Applied independently to each position in the sequence.
    Uses two linear layers with a GELU activation in between.

    Architecture:
        x -> Linear(d_model, dim_feedforward) -> GELU -> Dropout -> Linear(dim_feedforward, d_model) -> Dropout

    The intermediate dimension (dim_feedforward) is typically 4x d_model.

    Args:
        d_model: Input and output dimension.
        dim_feedforward: Hidden layer dimension.
        dropout: Dropout probability. Default 0.1.
    """

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network.

        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model]
        """
        x = self.linear1(x)       # [batch, seq_len, dim_feedforward]
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)       # [batch, seq_len, d_model]
        x = self.dropout2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """One Transformer encoder block (pre-norm style).

    Components:
    1. Multi-head self-attention with residual connection
    2. Position-wise feed-forward with residual connection

    Both sub-layers use LayerNorm before the operation (pre-norm).

    Args:
        d_model: Model embedding dimension.
        num_heads: Number of attention heads.
        dim_feedforward: Hidden size in the FFN.
        dropout: Dropout probability applied in attention and FFN.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Self-attention components
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward components
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run one encoder block.

        Args:
            x: [batch, seq_len, d_model]
            src_key_padding_mask: Optional [batch, seq_len] boolean mask.
                True means "this position is padding, ignore it."
            return_weights: If True, return attention weights.

        Returns:
            x: [batch, seq_len, d_model]
            weights: [batch, num_heads, seq_len, seq_len] or None
        """
        # Pre-norm self-attention
        residual = x
        x_norm = self.norm1(x)

        # Expand padding mask to 4D for attention: [batch, 1, 1, seq_len]
        attn_mask = None
        if src_key_padding_mask is not None:
            attn_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)

        attended, weights = self.self_attention(
            x_norm, x_norm, x_norm, mask=attn_mask, return_weights=return_weights
        )
        x = residual + self.dropout1(attended)  # residual connection

        # Pre-norm feed-forward
        residual = x
        x = residual + self.dropout2(self.feed_forward(self.norm2(x)))  # residual connection

        return x, weights


class TransformerEncoder(nn.Module):
    """Stack of N Transformer encoder blocks.

    Args:
        d_model: Model embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of encoder blocks to stack.
        dim_feedforward: FFN hidden size.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        # Final layer norm after all blocks (common in pre-norm style)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        return_all_weights: bool = False,
    ) -> Tuple[torch.Tensor, list]:
        """Run all encoder blocks in sequence.

        Args:
            x: [batch, seq_len, d_model]
            src_key_padding_mask: Optional [batch, seq_len] boolean padding mask.
            return_all_weights: If True, return attention weights from all layers.

        Returns:
            x: [batch, seq_len, d_model]
            all_weights: list of attention weight tensors (one per layer), or list of None
        """
        all_weights = []

        for layer in self.layers:
            x, weights = layer(
                x,
                src_key_padding_mask=src_key_padding_mask,
                return_weights=return_all_weights,
            )
            all_weights.append(weights)

        x = self.norm(x)
        return x, all_weights
