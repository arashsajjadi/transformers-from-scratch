"""
Patch visualization for Vision Transformer.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.visualization.plots import save_figure


def show_patches(
    image: np.ndarray,
    patch_size: int,
    save_path: Optional[Path] = None,
    title: str = "Image Patches",
) -> None:
    """Show an image divided into patches.

    The image is displayed with grid lines overlaid to show patch boundaries.

    Args:
        image: [H, W] or [H, W, C] numpy array in [0, 1].
        patch_size: Size of each square patch in pixels.
        save_path: Where to save the figure.
        title: Plot title.
    """
    H = image.shape[0]
    W = image.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: original image
    ax = axes[0]
    cmap = "gray" if image.ndim == 2 else None
    ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
    ax.set_title("Original Image")
    ax.axis("off")

    # Right: image with patch grid
    ax = axes[1]
    ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
    # Draw horizontal lines
    for y in range(0, H + 1, patch_size):
        ax.axhline(y - 0.5, color="red", linewidth=0.8, alpha=0.7)
    # Draw vertical lines
    for x in range(0, W + 1, patch_size):
        ax.axvline(x - 0.5, color="red", linewidth=0.8, alpha=0.7)
    ax.set_title(f"Patches ({patch_size}x{patch_size} pixels each)")
    ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)
    else:
        plt.show()


def show_patch_grid(
    image: np.ndarray,
    patch_size: int,
    save_path: Optional[Path] = None,
    title: str = "Patch Grid",
) -> None:
    """Show all individual patches extracted from an image in a grid.

    Args:
        image: [H, W] or [H, W, C] numpy array in [0, 1].
        patch_size: Size of each patch.
        save_path: Where to save.
        title: Figure title.
    """
    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    H, W, C = image.shape
    n_h = H // patch_size
    n_w = W // patch_size

    fig, axes = plt.subplots(n_h, n_w, figsize=(n_w * 1.2, n_h * 1.2))

    for i in range(n_h):
        for j in range(n_w):
            patch = image[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            ax = axes[i][j] if n_h > 1 else axes[j]
            if C == 1:
                ax.imshow(patch[:, :, 0], cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(patch, vmin=0, vmax=1)
            ax.axis("off")

    fig.suptitle(f"{title} — {n_h*n_w} patches")
    plt.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)
    else:
        plt.show()
