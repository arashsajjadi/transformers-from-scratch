"""
Transformer for time-series forecasting.

This shows that Transformers are not limited to text or images.
Given the last `input_length` time steps, predict the next `forecast_length` steps.

Architecture:
    past values (float)
    -> linear projection to d_model
    -> sinusoidal positional encoding
    -> Transformer encoder (N blocks)
    -> take all encoder output tokens
    -> linear regression head
    -> forecast values
"""

from typing import Tuple

import torch
import torch.nn as nn

from src.models.transformer_encoder import TransformerEncoder
from src.models.positional_encoding import SinusoidalPositionalEncoding


class TimeSeriesTransformer(nn.Module):
    """Transformer encoder for time-series regression/forecasting.

    Args:
        input_size: Number of input features per time step (default 1).
        forecast_length: Number of future time steps to predict.
        d_model: Model embedding dimension.
        num_heads: Attention heads.
        num_layers: Encoder blocks.
        dim_feedforward: FFN hidden size.
        max_len: Maximum sequence length for positional encoding.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_size: int = 1,
        forecast_length: int = 12,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        max_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.forecast_length = forecast_length

        # Project each time step's features to d_model dimensions
        # This is the analog of token embedding in NLP
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding to distinguish time steps
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        # Transformer encoder processes the sequence
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, dim_feedforward, dropout)

        # Regression head: predict future values from encoder output
        # We take all input tokens and project to forecast_length output values
        # Option: use a small MLP on the mean-pooled representation
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, forecast_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast future values.

        Args:
            x: [batch, input_length, input_size] — past time series values.

        Returns:
            forecast: [batch, forecast_length, 1] — predicted future values.
        """
        # Step 1: Project each timestep to d_model — [batch, input_length, d_model]
        x = self.input_projection(x)

        # Step 2: Add positional encoding
        x = self.positional_encoding(x)

        # Step 3: Encode — [batch, input_length, d_model]
        x, _ = self.encoder(x)

        # Step 4: Mean pool over time dimension — [batch, d_model]
        pooled = x.mean(dim=1)

        # Step 5: Regression head — [batch, forecast_length]
        forecast = self.regression_head(pooled)

        # Step 6: Add a channel dimension — [batch, forecast_length, 1]
        return forecast.unsqueeze(-1)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
