"""
Device selection utilities.

Priority order:
  1. CUDA  — NVIDIA GPU on Linux or Windows
  2. MPS   — Apple Silicon GPU on macOS
  3. CPU   — fallback for any machine

Usage:
    device = get_best_device()
    model = model.to(device)
    batch = move_batch_to_device(batch, device)
"""

import sys
import platform
import torch


def get_best_device() -> torch.device:
    """Return the fastest available device.

    Returns:
        torch.device: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device_arg: str) -> torch.device:
    """Resolve a device string from a command-line argument.

    Args:
        device_arg: One of 'auto', 'cuda', 'mps', 'cpu'.

    Returns:
        torch.device: the resolved device.

    If the requested device is unavailable, prints a warning and falls back to CPU.
    """
    if device_arg == "auto":
        return get_best_device()

    requested = torch.device(device_arg)

    if device_arg == "cuda" and not torch.cuda.is_available():
        print(f"[WARNING] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    if device_arg == "mps":
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_ok:
            print(f"[WARNING] MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")

    return requested


def move_batch_to_device(batch, device: torch.device):
    """Move a batch of data to the target device.

    The batch can be:
    - a single tensor
    - a list or tuple of tensors (and other objects)
    - a dict mapping strings to tensors

    Non-tensor items are left unchanged.

    Args:
        batch: A tensor, list, tuple, or dict.
        device: Target device.

    Returns:
        The same structure with all tensors moved to `device`.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        moved = [move_batch_to_device(item, device) for item in batch]
        return type(batch)(moved)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(val, device) for key, val in batch.items()}
    # Leave non-tensor items unchanged (e.g. strings, ints)
    return batch


def get_num_workers(is_notebook: bool = False) -> int:
    """Return a safe number of DataLoader workers for the current platform.

    Windows does not support num_workers > 0 in most notebook environments,
    so notebooks always use 0. Scripts use 2 on non-Windows.

    Args:
        is_notebook: True when calling from a Jupyter notebook.

    Returns:
        int: number of DataLoader workers.
    """
    if is_notebook:
        return 0
    # Windows multiprocessing with DataLoader can cause issues
    if platform.system() == "Windows":
        return 0
    return 2


def get_pin_memory(device: torch.device) -> bool:
    """Return True only when using CUDA.

    pin_memory=True speeds up host-to-GPU transfers but is not supported
    on MPS and has no effect on CPU.

    Args:
        device: The selected device.

    Returns:
        bool: True only for CUDA.
    """
    return device.type == "cuda"


def print_device_info(device: torch.device) -> None:
    """Print a summary of the selected device and system.

    Args:
        device: The selected device.
    """
    print(f"Platform  : {platform.system()} {platform.machine()}")
    print(f"Python    : {sys.version.split()[0]}")
    print(f"PyTorch   : {torch.__version__}")
    print(f"Device    : {device}")
    if device.type == "cuda":
        print(f"GPU       : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM      : {mem:.1f} GB")
    elif device.type == "mps":
        print(f"GPU       : Apple Silicon MPS")
    else:
        print(f"GPU       : not used (running on CPU)")
