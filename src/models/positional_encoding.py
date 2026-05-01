"""
Positional encoding implementations.

Transformers process all tokens in parallel, so they have no built-in sense of
order. Positional encodings inject position information into the embeddings.

Implements:
  - SinusoidalPositionalEncoding  (fixed, no learned parameters)
  - LearnedPositionalEncoding     (learned, slightly better for short sequences)
  - Simple2DPositionalEncoding    (for vision tasks: row + column encoding)

Key insight:
  "dog bites man" and "man bites dog" have the same token set but different
  meanings. Without positional encoding, a Transformer cannot tell them apart.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding from the original Transformer paper.

    For each position pos and each dimension i:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Properties:
    - No learned parameters.
    - Can generalize to sequences longer than seen at training time.
    - Similar positions have similar encodings.
    - The difference between positions is predictable.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length. Default 512.
        dropout: Dropout applied after adding encoding. Default 0.0.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the encoding matrix: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions

        # Register as buffer: moves with the model to CUDA/MPS but is not a parameter
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to token embeddings.

        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            [batch_size, seq_len, d_model] with positional information added.
        """
        seq_len = x.size(1)
        # self.pe has shape [1, max_len, d_model]; slice to actual seq_len
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings.

    Instead of fixed sine/cosine values, each position has a learnable vector.
    This is slightly simpler and often works just as well for sequences up to
    max_len tokens long.

    Limitation: cannot generalize beyond max_len.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length.
        dropout: Dropout probability. Default 0.0.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Learnable embedding table: position index -> d_model-dimensional vector
        self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional embeddings to token embeddings.

        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        # Position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        pos_embeddings = self.position_embedding(positions)  # [1, seq_len, d_model]
        x = x + pos_embeddings
        return self.dropout(x)


class Simple2DPositionalEncoding(nn.Module):
    """Simple 2D positional encoding for Vision Transformer (ViT).

    For images, tokens are patches arranged on a 2D grid.
    This encoding gives each patch a position in (row, column) space.

    We use separate sinusoidal encodings for rows and columns, then add them:
        PE(patch) = PE_row(row) + PE_col(col)

    Args:
        d_model: Embedding dimension. Must be even.
        num_patches_h: Number of patch rows.
        num_patches_w: Number of patch columns.
        dropout: Dropout probability. Default 0.0.
    """

    def __init__(
        self,
        d_model: int,
        num_patches_h: int,
        num_patches_w: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for 2D positional encoding."
        self.dropout = nn.Dropout(p=dropout)

        half_d = d_model // 2

        def sinusoidal_1d(n_pos: int, dim: int) -> torch.Tensor:
            """Build [n_pos, dim] sinusoidal encoding."""
            pe = torch.zeros(n_pos, dim)
            pos = torch.arange(n_pos, dtype=torch.float).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
            )
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            return pe

        pe_row = sinusoidal_1d(num_patches_h, half_d)  # [H, d/2]
        pe_col = sinusoidal_1d(num_patches_w, half_d)  # [W, d/2]

        # For each patch at (r, c): encode = [pe_row[r], pe_col[c]]
        pe = torch.zeros(num_patches_h, num_patches_w, d_model)
        for r in range(num_patches_h):
            for c in range(num_patches_w):
                pe[r, c, :half_d] = pe_row[r]
                pe[r, c, half_d:] = pe_col[c]

        # Flatten spatial dims: [1, H*W, d_model]
        pe = pe.view(1, num_patches_h * num_patches_w, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add 2D positional encoding.

        Args:
            x: [batch, num_patches, d_model]

        Returns:
            [batch, num_patches, d_model]
        """
        x = x + self.pe
        return self.dropout(x)
