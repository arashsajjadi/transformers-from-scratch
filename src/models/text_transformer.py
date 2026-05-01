"""
Transformer encoder for text classification.

Architecture:
    text
    -> tokenizer (word IDs from src/data/tiny_text.py)
    -> token embedding
    -> sinusoidal positional encoding
    -> Transformer encoder (N blocks)
    -> mean pooling over token dimension
    -> linear classifier
    -> class logits

Mean pooling:
    Averages all token embeddings (excluding padding) to produce a single
    fixed-size vector that summarizes the whole sentence.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.models.transformer_encoder import TransformerEncoder
from src.models.positional_encoding import SinusoidalPositionalEncoding


class TextTransformerClassifier(nn.Module):
    """Transformer encoder for text classification.

    Uses mean pooling by default. Optionally supports a CLS token.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        num_classes: Number of output classes.
        d_model: Embedding and model dimension.
        num_heads: Number of attention heads.
        num_layers: Number of encoder blocks.
        dim_feedforward: FFN hidden size.
        max_len: Maximum sequence length for positional encoding.
        dropout: Dropout probability.
        pad_idx: Index of the padding token (default 0).
        use_cls_token: If True, prepend a learnable CLS token and use it for classification.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        max_len: int = 128,
        dropout: float = 0.1,
        pad_idx: int = 0,
        use_cls_token: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.use_cls_token = use_cls_token

        # Token embedding: integer IDs -> d_model vectors
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # Positional encoding (fixed sinusoidal)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        # Stacked encoder blocks
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, dim_feedforward, dropout)

        # Classification head: d_model -> num_classes
        self.classifier = nn.Linear(d_model, num_classes)

        # Optional CLS token parameter
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embedding and linear weights."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """Run the text classification model.

        Args:
            input_ids: [batch, seq_len] integer token IDs.
            return_weights: If True, return attention weights from all encoder layers.

        Returns:
            logits: [batch, num_classes]
            all_weights: list of attention weight tensors or list of None values
        """
        batch_size = input_ids.size(0)

        # Build padding mask: True where input_ids == pad_idx
        # Shape: [batch, seq_len]
        padding_mask = (input_ids == self.pad_idx)

        # Step 1: Token embeddings → [batch, seq_len, d_model]
        x = self.token_embedding(input_ids) * (self.d_model ** 0.5)  # scale by sqrt(d_model)

        # Step 2: Add positional encoding
        x = self.positional_encoding(x)

        # Step 3: Optionally prepend CLS token
        if self.use_cls_token:
            cls = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, d_model]
            x = torch.cat([cls, x], dim=1)  # [batch, 1+seq_len, d_model]
            # Extend padding mask with False for CLS (never pad CLS)
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # Step 4: Encoder
        x, all_weights = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
            return_all_weights=return_weights,
        )

        # Step 5: Pool to a single vector
        if self.use_cls_token:
            # Use the CLS token output (position 0)
            pooled = x[:, 0, :]  # [batch, d_model]
        else:
            # Mean pool over non-padding positions
            # Create a mask of shape [batch, seq_len, 1] with 1.0 for real tokens
            non_pad_mask = (~padding_mask).float().unsqueeze(-1)  # [batch, seq_len, 1]
            pooled = (x * non_pad_mask).sum(dim=1) / non_pad_mask.sum(dim=1).clamp(min=1)
            # Shape: [batch, d_model]

        # Step 6: Classification
        logits = self.classifier(pooled)  # [batch, num_classes]

        return logits, all_weights

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
