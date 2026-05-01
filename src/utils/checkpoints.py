"""
Checkpoint saving and loading utilities.

Checkpoints are not saved by default (controlled by the --save-checkpoint flag).
The .pt files are gitignored to keep the repository small.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: Path,
) -> None:
    """Save a model checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer state to save.
        epoch: Current epoch number.
        metrics: Dictionary of current metric values.
        save_path: File path ending in .pt or .pth.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(state, save_path)
    print(f"[checkpoint] Saved to {save_path}")


def load_checkpoint(
    model: torch.nn.Module,
    load_path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a model checkpoint.

    Args:
        model: The model to load weights into.
        load_path: Path to the .pt file.
        optimizer: If provided, load optimizer state as well.
        device: Device to map the checkpoint to.

    Returns:
        Dict containing 'epoch' and 'metrics' from the checkpoint.
    """
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    map_location = device if device is not None else "cpu"
    state = torch.load(load_path, map_location=map_location)

    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    print(f"[checkpoint] Loaded from {load_path} (epoch {state.get('epoch', '?')})")

    return {"epoch": state.get("epoch", 0), "metrics": state.get("metrics", {})}
