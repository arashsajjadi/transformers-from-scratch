"""
Vision Transformer (ViT) based semantic segmenter.

After the ViT encoder, the patch tokens (excluding CLS) are reshaped back
into a 2D feature map, then decoded with a small convolutional decoder
to produce a full-resolution segmentation mask.

Architecture:
    image [B, C, H, W]
    -> PatchEmbedding -> [B, num_patches, d_model]
    -> Sinusoidal positional encoding
    -> Transformer encoder -> [B, num_patches, d_model]
    -> Remove CLS (or don't add one), reshape to [B, d_model, h_patches, w_patches]
    -> Conv decoder (upsample back to H, W)
    -> [B, num_classes, H, W]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.transformer_encoder import TransformerEncoder
from src.models.positional_encoding import SinusoidalPositionalEncoding
from src.models.vit import PatchEmbedding


class ViTSegmenter(nn.Module):
    """ViT-based semantic segmentation model.

    Args:
        image_size: Input image height/width.
        patch_size: Patch size. Must divide image_size.
        in_channels: Input channels.
        num_classes: Number of segmentation classes.
        d_model: Transformer embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of encoder blocks.
        dim_feedforward: FFN hidden size.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 4,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_per_side = image_size // patch_size  # e.g. 128/16 = 8
        self.num_patches = self.num_patches_per_side ** 2

        # Patch embedding
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, d_model)

        # Positional encoding (no CLS token in this segmenter)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, self.num_patches + 10, dropout)

        # Transformer encoder
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, dim_feedforward, dropout)

        # Convolutional decoder: upsample from (num_patches_per_side)^2 tokens back to (image_size)^2
        # We need to upsample by a factor of patch_size in each dimension
        n_upsample = int(math.log2(patch_size))  # number of 2x upsample steps
        assert 2 ** n_upsample == patch_size, (
            f"patch_size ({patch_size}) must be a power of 2 for the decoder"
        )

        decoder_layers = []
        in_ch = d_model
        out_ch = max(num_classes * 4, 16)
        for i in range(n_upsample):
            is_last = (i == n_upsample - 1)
            out_ch_i = num_classes if is_last else out_ch
            decoder_layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch_i, kernel_size=2, stride=2),
            ])
            if not is_last:
                decoder_layers.extend([
                    nn.BatchNorm2d(out_ch_i),
                    nn.ReLU(inplace=True),
                ])
            in_ch = out_ch_i

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ViT segmenter.

        Args:
            x: [B, in_channels, image_size, image_size]

        Returns:
            logits: [B, num_classes, image_size, image_size]
        """
        B = x.size(0)
        H_p = self.num_patches_per_side

        # Step 1: Patch embeddings → [B, num_patches, d_model]
        tokens = self.patch_embedding(x)

        # Step 2: Add positional encoding
        tokens = self.positional_encoding(tokens)

        # Step 3: Transformer encoder → [B, num_patches, d_model]
        tokens, _ = self.encoder(tokens)

        # Step 4: Reshape to 2D feature map → [B, d_model, H_p, W_p]
        feature_map = tokens.transpose(1, 2).reshape(B, -1, H_p, H_p)

        # Step 5: Convolutional decoder → upsample to [B, num_classes, image_size, image_size]
        logits = self.decoder(feature_map)

        return logits

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
