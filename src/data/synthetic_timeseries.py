"""
Synthetic time-series dataset.

Generates sine waves with optional noise, multiple frequencies, and trend.
This is used in notebook 08 and script run_timeseries.py.

The task:
  Given the last `input_length` time steps, predict the next `forecast_length` steps.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticTimeSeriesDataset(Dataset):
    """Dataset of synthetic sine-wave time series.

    Each sample is a sliding window over a long generated signal.
    The model receives `input_length` steps and must predict the next
    `forecast_length` steps.

    Args:
        n_samples: Number of sliding-window samples to generate.
        input_length: Number of past time steps given as input.
        forecast_length: Number of future time steps to predict.
        mode: One of 'clean', 'noisy', 'multi_freq', 'trend'.
        noise_std: Standard deviation of Gaussian noise (used in 'noisy' mode).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        input_length: int = 48,
        forecast_length: int = 12,
        mode: str = "noisy",
        noise_std: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.forecast_length = forecast_length
        window = input_length + forecast_length

        rng = np.random.RandomState(seed)

        # Generate a signal long enough for n_samples windows
        total_length = n_samples + window
        t = np.linspace(0, 4 * np.pi * (total_length / 100), total_length)

        if mode == "clean":
            signal = np.sin(t)
        elif mode == "noisy":
            signal = np.sin(t) + rng.normal(0, noise_std, total_length)
        elif mode == "multi_freq":
            signal = (
                np.sin(t)
                + 0.5 * np.sin(2 * t)
                + 0.25 * np.sin(4 * t)
                + rng.normal(0, noise_std, total_length)
            )
        elif mode == "trend":
            trend = 0.002 * np.arange(total_length)
            seasonal = np.sin(t) + 0.3 * np.sin(3 * t)
            signal = trend + seasonal + rng.normal(0, noise_std, total_length)
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from 'clean', 'noisy', 'multi_freq', 'trend'.")

        # Normalize to [-1, 1] range
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)

        # Create sliding windows
        self.inputs = []
        self.targets = []
        for i in range(n_samples):
            self.inputs.append(signal[i : i + input_length])
            self.targets.append(signal[i + input_length : i + input_length + forecast_length])

        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shape: [input_length, 1] and [forecast_length, 1]
        x = torch.tensor(self.inputs[idx]).unsqueeze(-1)   # [input_length, 1]
        y = torch.tensor(self.targets[idx]).unsqueeze(-1)  # [forecast_length, 1]
        return x, y


def load_timeseries_data(
    n_train: int = 800,
    n_val: int = 200,
    input_length: int = 48,
    forecast_length: int = 12,
    mode: str = "noisy",
    noise_std: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and val DataLoaders for synthetic time-series.

    Args:
        n_train: Number of training samples.
        n_val: Number of validation samples.
        input_length: Number of input time steps.
        forecast_length: Number of steps to forecast.
        mode: Signal type. See SyntheticTimeSeriesDataset.
        noise_std: Noise standard deviation.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        seed: Random seed.

    Returns:
        (train_loader, val_loader)
    """
    train_ds = SyntheticTimeSeriesDataset(
        n_samples=n_train,
        input_length=input_length,
        forecast_length=forecast_length,
        mode=mode,
        noise_std=noise_std,
        seed=seed,
    )
    val_ds = SyntheticTimeSeriesDataset(
        n_samples=n_val,
        input_length=input_length,
        forecast_length=forecast_length,
        mode=mode,
        noise_std=noise_std,
        seed=seed + 1,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
