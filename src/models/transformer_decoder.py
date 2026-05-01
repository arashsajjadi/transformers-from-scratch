"""
Transformer decoder: decoder block, stacked decoder, and causal mask generation.

The decoder has two attention sub-layers:
  1. Masked self-attention — each target position can only attend to previous
     target positions (causal / autoregressive).
  2. Cross-attention — each target position attends to all encoder outputs.

Architecture of one decoder block:
    target input
    -> LayerNorm
    -> Masked Multi-Head Self-Attention
    -> Residual connection
    -> LayerNorm
    -> Multi-Head Cross-Attention (query from decoder, key/value from encoder)
    -> Residual connection
    -> LayerNorm
    -> FeedForward
    -> Residual connection
    -> output
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.attention import MultiHeadAttention
from src.models.transformer_encoder import FeedForward


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Generate a causal (autoregressive) mask.

    In the causal mask, position i can only attend to positions 0..i.
    Positions j > i are masked out.

    The mask is a boolean tensor where True means "block this position".

    Example for seq_len=4:
        [[F, T, T, T],
         [F, F, T, T],
         [F, F, F, T],
         [F, F, F, F]]

    Args:
        seq_len: Sequence length.
        device: Target device.

    Returns:
        [seq_len, seq_len] boolean tensor (upper triangle = True = blocked).
    """
    # torch.triu gives the upper triangle. We block positions above the diagonal.
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    return mask


class TransformerDecoderBlock(nn.Module):
    """One Transformer decoder block (pre-norm style).

    Contains three sub-layers:
    1. Masked self-attention (causal)
    2. Cross-attention (attends to encoder output)
    3. Feed-forward network

    Args:
        d_model: Model embedding dimension.
        num_heads: Number of attention heads.
        dim_feedforward: FFN hidden size.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Masked self-attention
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-forward
        self.norm3 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        target: torch.Tensor,
        encoder_output: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        return_cross_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run one decoder block.

        Args:
            target: [batch, tgt_len, d_model] — decoder input embeddings.
            encoder_output: [batch, src_len, d_model] — output from the encoder.
            causal_mask: [tgt_len, tgt_len] causal mask (blocks future positions).
            tgt_key_padding_mask: [batch, tgt_len] padding mask for target.
            src_key_padding_mask: [batch, src_len] padding mask for encoder output.
            return_cross_weights: If True, return cross-attention weights.

        Returns:
            output: [batch, tgt_len, d_model]
            cross_weights: [batch, num_heads, tgt_len, src_len] or None
        """
        tgt_len = target.size(1)

        # ─── Sub-layer 1: Masked self-attention ───────────────────────────────
        residual = target
        normed = self.norm1(target)

        # Combine causal mask and target padding mask
        self_attn_mask = None
        if causal_mask is not None:
            # causal_mask: [tgt_len, tgt_len] → broadcast over batch and heads
            self_attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, tgt_len]
        if tgt_key_padding_mask is not None:
            pad_mask = tgt_key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, tgt_len]
            if self_attn_mask is not None:
                self_attn_mask = self_attn_mask | pad_mask
            else:
                self_attn_mask = pad_mask

        attended, _ = self.self_attention(normed, normed, normed, mask=self_attn_mask)
        target = residual + self.dropout1(attended)

        # ─── Sub-layer 2: Cross-attention ─────────────────────────────────────
        residual = target
        normed = self.norm2(target)

        # Padding mask for encoder keys: [B, 1, 1, src_len]
        cross_attn_mask = None
        if src_key_padding_mask is not None:
            cross_attn_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)

        cross_attended, cross_weights = self.cross_attention(
            normed, encoder_output, encoder_output,
            mask=cross_attn_mask,
            return_weights=return_cross_weights,
        )
        target = residual + self.dropout2(cross_attended)

        # ─── Sub-layer 3: Feed-forward ────────────────────────────────────────
        residual = target
        target = residual + self.dropout3(self.feed_forward(self.norm3(target)))

        return target, cross_weights


class TransformerDecoder(nn.Module):
    """Stack of N Transformer decoder blocks.

    Args:
        d_model: Embedding dimension.
        num_heads: Attention heads.
        num_layers: Number of decoder blocks.
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
            TransformerDecoderBlock(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        target: torch.Tensor,
        encoder_output: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        return_last_cross_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run all decoder blocks.

        Args:
            target: [batch, tgt_len, d_model]
            encoder_output: [batch, src_len, d_model]
            causal_mask: [tgt_len, tgt_len]
            tgt_key_padding_mask: [batch, tgt_len]
            src_key_padding_mask: [batch, src_len]
            return_last_cross_weights: Return cross-attention from the last layer.

        Returns:
            output: [batch, tgt_len, d_model]
            cross_weights: from last layer, or None
        """
        cross_weights = None
        for i, layer in enumerate(self.layers):
            is_last = (i == len(self.layers) - 1)
            target, cross_weights = layer(
                target,
                encoder_output,
                causal_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                src_key_padding_mask=src_key_padding_mask,
                return_cross_weights=(return_last_cross_weights and is_last),
            )

        target = self.norm(target)
        return target, cross_weights
