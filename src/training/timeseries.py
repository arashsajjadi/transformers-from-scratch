"""
Loss and metric functions for time-series forecasting.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def timeseries_loss_fn(model: nn.Module, batch, device: torch.device) -> torch.Tensor:
    """MSE loss for time-series forecasting.

    Batch format: (x, y) where x = past values, y = future values.

    Args:
        model: TimeSeriesTransformer.
        batch: (x [B, input_len, 1], y [B, forecast_len, 1]).
        device: Target device.

    Returns:
        scalar MSE loss.
    """
    x, y = batch
    x = x.to(device)
    y = y.to(device)

    pred = model(x)  # [B, forecast_len, 1]
    return F.mse_loss(pred, y)


def timeseries_metric_fn(model: nn.Module, batch, device: torch.device) -> Dict[str, float]:
    """Compute MSE and MAE for a batch.

    Args:
        model: TimeSeriesTransformer.
        batch: (x, y).
        device: Target device.

    Returns:
        dict with 'mse' and 'mae'.
    """
    x, y = batch
    x = x.to(device)
    y = y.to(device)

    pred = model(x)

    mse = F.mse_loss(pred, y).item()
    mae = F.l1_loss(pred, y).item()

    return {"mse": mse, "mae": mae}
