"""
Notebook-specific helpers.

These small functions are designed to be called from Jupyter notebooks.
They set up the environment, select the device, and display info tables.
"""

import sys
import platform
from pathlib import Path
from typing import Optional

import torch

from src.utils.device import get_best_device, print_device_info
from src.utils.seed import seed_everything
from src.utils.cleanup import prepare_run_dir
from src.utils.reporting import save_metrics_csv


def notebook_setup(
    run_dir: Path,
    seed: int = 42,
    clean: bool = True,
) -> torch.device:
    """Standard notebook setup: seed, device, run directory.

    Call this at the top of every notebook.

    Args:
        run_dir: The runs/ subdirectory for this notebook.
        seed: Random seed.
        clean: If True, clean old outputs from run_dir.

    Returns:
        torch.device: The selected device.
    """
    seed_everything(seed)
    device = get_best_device()

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    prepare_run_dir(run_dir, clean=clean)
    print_device_info(device)

    return device


def save_environment_csv(output_path: Path) -> None:
    """Save environment info (Python, PyTorch, platform) to a CSV file.

    Args:
        output_path: Path to save device_info.csv.
    """
    try:
        import torchvision
        tv_version = torchvision.__version__
    except ImportError:
        tv_version = "not installed"

    device = get_best_device()

    info = {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "torchvision_version": tv_version,
        "platform": platform.system(),
        "architecture": platform.machine(),
        "device": str(device),
        "cuda_available": str(torch.cuda.is_available()),
        "mps_available": str(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }
    save_metrics_csv(info, output_path)
    print(f"[notebook] Environment info saved to {output_path}")
