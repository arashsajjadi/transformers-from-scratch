"""
Vision Transformer (ViT) from scratch.

Key idea:
    - Split an image into fixed-size patches.
    - Treat each patch like a word token.
    - Feed the sequence of patch tokens into a Transformer encoder.
    - Use a special CLS token for classification.

Architecture:
    image [B, C, H, W]
    -> split into patches [B, num_patches, patch_pixels]
    -> linear patch embedding [B, num_patches, d_model]
    -> prepend CLS token [B, 1 + num_patches, d_model]
    -> add learned positional embedding
    -> Transformer encoder
    -> take CLS token output [B, d_model]
    -> linear classifier [B, num_classes]

Reference: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
"""

import torch
import torch.nn as nn

from src.models.transformer_encoder import TransformerEncoder
from src.models.positional_encoding import LearnedPositionalEncoding


class PatchEmbedding(nn.Module):
    """Split an image into patches and project each patch to d_model.

    For a 28x28 image with patch_size=7:
        - Number of patches = (28/7) * (28/7) = 4 * 4 = 16
        - Each patch has 7 * 7 * in_channels pixels (flattened)
        - A linear layer projects these pixels to d_model

    Args:
        image_size: Height/width of the square input image.
        patch_size: Height/width of each square patch. Must divide image_size.
        in_channels: Number of image channels (1=grayscale, 3=RGB).
        d_model: Output embedding dimension.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        d_model: int,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size  # pixels per patch

        # Linear projection: flatten patch pixels → d_model
        self.projection = nn.Linear(self.patch_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split image into patches and project.

        Args:
            x: [batch, in_channels, image_size, image_size]

        Returns:
            [batch, num_patches, d_model]
        """
        batch_size, C, H, W = x.shape
        P = self.patch_size

        # Step 1: Reshape into patches
        # [B, C, H, W] -> [B, C, H/P, P, W/P, P]
        x = x.reshape(batch_size, C, H // P, P, W // P, P)
        # -> [B, H/P, W/P, P, P, C]  (spatial grid of patches)
        x = x.permute(0, 2, 4, 3, 5, 1)
        # -> [B, num_patches, patch_pixels]
        x = x.reshape(batch_size, self.num_patches, self.patch_dim)

        # Step 2: Linear projection to d_model
        x = self.projection(x)  # [B, num_patches, d_model]

        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) for image classification.

    Args:
        image_size: Input image height/width (square).
        patch_size: Patch height/width. Must divide image_size.
        in_channels: Input channels (1 or 3).
        num_classes: Number of output classes.
        d_model: Embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of encoder blocks.
        dim_feedforward: FFN hidden size.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        num_classes: int = 10,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size

        # Step 1: Patch embedding
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, d_model)
        num_patches = self.patch_embedding.num_patches

        # Step 2: CLS token — a learnable vector prepended to the patch sequence
        # After encoding, its output represents the whole image
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Step 3: Learned positional embedding for CLS + all patches
        self.positional_encoding = LearnedPositionalEncoding(
            d_model, max_len=num_patches + 1, dropout=dropout
        )

        # Step 4: Transformer encoder
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, dim_feedforward, dropout)

        # Step 5: Final layer norm + classifier head
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        # Initialize weights
        nn.init.zeros_(self.classifier.bias)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)

    def forward(self, x: torch.Tensor, return_patch_tokens: bool = False):
        """Run the Vision Transformer.

        Args:
            x: [batch, in_channels, image_size, image_size]
            return_patch_tokens: If True, also return the encoded patch tokens.

        Returns:
            logits: [batch, num_classes]
            patch_tokens (optional): [batch, num_patches, d_model]
        """
        batch_size = x.size(0)

        # Step 1: Patch embeddings — [batch, num_patches, d_model]
        patch_emb = self.patch_embedding(x)

        # Step 2: Prepend CLS token — [batch, 1+num_patches, d_model]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, patch_emb], dim=1)

        # Step 3: Add positional encoding
        x = self.positional_encoding(x)

        # Step 4: Transformer encoder — [batch, 1+num_patches, d_model]
        x, _ = self.encoder(x)

        # Step 5: Extract CLS token and classify
        cls_output = self.norm(x[:, 0, :])   # [batch, d_model]
        logits = self.classifier(cls_output)  # [batch, num_classes]

        if return_patch_tokens:
            return logits, x[:, 1:, :]  # patch tokens without CLS
        return logits

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
