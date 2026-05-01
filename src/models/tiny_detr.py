"""
Tiny DETR: DETR-style object detection with a Transformer decoder.

Key idea (from "End-to-End Object Detection with Transformers", Carion et al., 2020):
  - A CNN backbone extracts image features.
  - These features are flattened into a sequence of tokens.
  - A Transformer encoder processes these visual tokens.
  - N learnable "object queries" attend to the encoded features via cross-attention.
  - Each query produces one prediction: class + bounding box.
  - No NMS needed because the set of predictions is fixed and trained to be non-overlapping.

This implementation uses a simple greedy matcher instead of Hungarian matching
to keep the code educational and easy to understand.

Architecture:
    image
    -> TinyCNNBackbone -> feature map
    -> flatten + 2D positional encoding -> visual tokens
    -> Transformer encoder
    -> N learnable object queries
    -> Transformer decoder
    -> class head (including no-object class)
    -> box head
"""

import math
from typing import Tuple

import torch
import torch.nn as nn

from src.models.transformer_encoder import TransformerEncoder
from src.models.transformer_decoder import TransformerDecoder
from src.models.positional_encoding import SinusoidalPositionalEncoding


class TinyCNNBackbone(nn.Module):
    """Lightweight CNN backbone for DETR.

    Extracts a feature map from an image.
    Output stride is 8 (3 downsampling steps with stride 2 each via pooling).

    Args:
        in_channels: Input image channels.
        out_channels: Feature map channels (projected to d_model afterwards).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # /2

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # /4

            nn.Conv2d(64, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # /8
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature map.

        Args:
            x: [B, in_channels, H, W]

        Returns:
            [B, out_channels, H/8, W/8]
        """
        return self.backbone(x)


class TinyDETR(nn.Module):
    """DETR-style detector with a small CNN backbone and Transformer.

    Args:
        in_channels: Input image channels.
        num_classes: Number of foreground object classes.
        num_queries: Number of object queries (maximum detectable objects).
        image_size: Input image height/width.
        d_model: Transformer embedding dimension.
        num_heads: Attention heads.
        num_encoder_layers: Encoder blocks.
        num_decoder_layers: Decoder blocks.
        dim_feedforward: FFN hidden size.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        num_queries: int = 5,
        image_size: int = 128,
        d_model: int = 64,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_classes = num_classes
        # +1 for the "no object" class
        self.num_classes_with_bg = num_classes + 1
        self.no_object_class = num_classes  # last index is no-object

        # CNN backbone
        backbone_channels = 128
        self.backbone = TinyCNNBackbone(in_channels, backbone_channels)

        # The backbone reduces spatial size by 8x
        feat_h = image_size // 8
        feat_w = image_size // 8
        num_tokens = feat_h * feat_w

        # Project backbone features to d_model
        self.input_proj = nn.Conv2d(backbone_channels, d_model, kernel_size=1)

        # Positional encoding for the flattened feature tokens
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, num_tokens + 10, dropout)

        # Transformer encoder processes visual tokens
        self.encoder = TransformerEncoder(
            d_model, num_heads, num_encoder_layers, dim_feedforward, dropout
        )

        # Learnable object queries (one per potential object)
        self.query_embeddings = nn.Embedding(num_queries, d_model)

        # Transformer decoder: queries attend to encoded visual tokens
        self.decoder = TransformerDecoder(
            d_model, num_heads, num_decoder_layers, dim_feedforward, dropout
        )

        # Prediction heads
        self.class_head = nn.Linear(d_model, self.num_classes_with_bg)
        self.box_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, 4),
            nn.Sigmoid(),  # normalized (cx, cy, w, h) in [0, 1]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.class_head.weight)
        nn.init.zeros_(self.class_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run TinyDETR.

        Args:
            x: [B, in_channels, H, W]

        Returns:
            class_logits: [B, num_queries, num_classes+1]
            boxes: [B, num_queries, 4] in normalized (cx, cy, w, h)
        """
        B = x.size(0)

        # ─── Step 1: CNN backbone → feature map ──────────────────────────────
        features = self.backbone(x)  # [B, backbone_out, H/8, W/8]

        # ─── Step 2: Project to d_model ──────────────────────────────────────
        features = self.input_proj(features)  # [B, d_model, H/8, W/8]

        # ─── Step 3: Flatten feature map to sequence ─────────────────────────
        B, C, Hf, Wf = features.shape
        # [B, d_model, Hf*Wf] -> [B, Hf*Wf, d_model]
        tokens = features.flatten(2).transpose(1, 2)

        # ─── Step 4: Add positional encoding ─────────────────────────────────
        tokens = self.positional_encoding(tokens)

        # ─── Step 5: Transformer encoder ─────────────────────────────────────
        memory, _ = self.encoder(tokens)  # [B, Hf*Wf, d_model]

        # ─── Step 6: Object queries ───────────────────────────────────────────
        # [num_queries, d_model] -> [B, num_queries, d_model]
        query_ids = torch.arange(self.num_queries, device=x.device)
        queries = self.query_embeddings(query_ids).unsqueeze(0).expand(B, -1, -1)

        # ─── Step 7: Transformer decoder ─────────────────────────────────────
        # queries attend to encoded visual tokens (memory)
        decoder_output, _ = self.decoder(queries, memory)  # [B, num_queries, d_model]

        # ─── Step 8: Prediction heads ─────────────────────────────────────────
        class_logits = self.class_head(decoder_output)  # [B, num_queries, num_classes+1]
        boxes = self.box_head(decoder_output)            # [B, num_queries, 4]

        return class_logits, boxes

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
