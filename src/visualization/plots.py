"""
General plotting utilities: training curves, figures, tables.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import matplotlib.pyplot as plt
import numpy as np


def save_figure(fig: plt.Figure, path: Path, dpi: int = 120) -> None:
    """Save a matplotlib figure to disk and close it.

    Args:
        fig: The matplotlib Figure object.
        path: Output file path.
        dpi: Image resolution.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Path,
    title: str = "Training Curves",
) -> None:
    """Plot training and validation loss curves, plus any extra metrics.

    Args:
        history: Dict with keys like 'train_loss', 'val_loss', 'accuracy', etc.
        save_path: Where to save the figure.
        title: Plot title.
    """
    # Separate loss curves from metric curves
    loss_keys = [k for k in history if "loss" in k]
    metric_keys = [k for k in history if "loss" not in k]

    n_plots = 1 + (1 if metric_keys else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    # Loss plot
    ax = axes[0]
    for key in loss_keys:
        values = history[key]
        epochs = range(1, len(values) + 1)
        label = key.replace("_", " ").title()
        ax.plot(epochs, values, label=label, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Metric plot
    if metric_keys:
        ax = axes[1]
        for key in metric_keys:
            values = history[key]
            epochs = range(1, len(values) + 1)
            label = key.replace("_", " ").title()
            ax.plot(epochs, values, label=label, linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title("Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plot a confusion matrix as a heatmap.

    Args:
        cm: [num_classes, num_classes] integer numpy array.
        class_names: List of class name strings.
        save_path: Output path.
        title: Plot title.
    """
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(4, n), max(3, n)))

    # Normalize for display
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate each cell with the raw count
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=10)

    plt.tight_layout()
    save_figure(fig, save_path)


def show_image_grid(
    images: np.ndarray,
    labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    ncols: int = 8,
    title: str = "",
    cmap: Optional[str] = None,
) -> None:
    """Show a grid of images with optional labels.

    Args:
        images: [N, H, W] or [N, H, W, C] numpy array, float in [0, 1].
        labels: Optional list of N label strings.
        save_path: Where to save the figure.
        ncols: Number of columns in the grid.
        title: Figure title.
        cmap: Colormap. Use 'gray' for grayscale.
    """
    N = len(images)
    nrows = (N + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    if nrows == 1:
        axes = [axes] if ncols == 1 else axes
        axes = np.array(axes).reshape(1, -1)
    else:
        axes = np.array(axes).reshape(nrows, ncols)

    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        if i < N:
            img = images[i]
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = img.transpose(1, 2, 0)
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            if labels is not None:
                ax.set_title(str(labels[i]), fontsize=7)
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)
    else:
        plt.show()


def save_dataframe_table_as_markdown(
    rows: List[dict],
    path: Path,
    title: Optional[str] = None,
) -> None:
    """Save a list of dicts as a Markdown table.

    Args:
        rows: List of row dicts.
        path: Output file path.
        title: Optional title.
    """
    from src.utils.markdown import save_dataframe_table_as_markdown as _save
    _save(rows, path, title)
