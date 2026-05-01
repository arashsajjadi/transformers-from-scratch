"""
Segmentation visualization: overlays and mask grids.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.visualization.plots import save_figure


# Color palette for up to 8 classes
SEGMENTATION_COLORS = np.array([
    [0.85, 0.85, 0.85],  # 0 = background (light gray)
    [0.85, 0.30, 0.30],  # 1 = circle (red)
    [0.30, 0.55, 0.85],  # 2 = square (blue)
    [0.30, 0.75, 0.40],  # 3 = triangle (green)
    [0.90, 0.70, 0.20],  # 4 = yellow
    [0.70, 0.30, 0.75],  # 5 = purple
    [0.25, 0.75, 0.75],  # 6 = teal
    [0.90, 0.50, 0.25],  # 7 = orange
], dtype=np.float32)


def mask_to_rgb(mask: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """Convert an integer class mask to an RGB color image.

    Args:
        mask: [H, W] integer numpy array.
        num_classes: Number of classes (for color selection).

    Returns:
        [H, W, 3] float32 RGB array in [0, 1].
    """
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(min(num_classes, len(SEGMENTATION_COLORS))):
        rgb[mask == c] = SEGMENTATION_COLORS[c]
    return rgb


def overlay_segmentation_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    save_path: Optional[Path] = None,
    title: str = "Segmentation Overlay",
    class_names: Optional[List[str]] = None,
) -> None:
    """Overlay a segmentation mask on an image.

    Args:
        image: [H, W, C] or [H, W] numpy float array in [0, 1].
        mask: [H, W] integer numpy array with class IDs.
        alpha: Blend factor (0 = only image, 1 = only mask).
        save_path: Where to save.
        title: Figure title.
        class_names: Optional list of class name strings for the legend.
    """
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[0] == 1 or image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] == 1:
        image = np.concatenate([image] * 3, axis=-1)

    num_classes = int(mask.max()) + 1
    mask_rgb = mask_to_rgb(mask, num_classes)
    overlay = (1 - alpha) * image + alpha * mask_rgb
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image, vmin=0, vmax=1)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask_rgb, vmin=0, vmax=1)
    axes[1].set_title("Mask")
    axes[1].axis("off")

    axes[2].imshow(overlay, vmin=0, vmax=1)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    # Add legend
    if class_names is not None:
        patches = [
            mpatches.Patch(color=SEGMENTATION_COLORS[i], label=class_names[i])
            for i in range(min(len(class_names), len(SEGMENTATION_COLORS)))
        ]
        fig.legend(handles=patches, loc="lower center", ncol=len(patches), fontsize=9)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)
    else:
        plt.show()


def plot_segmentation_examples(
    images: np.ndarray,
    gt_masks: np.ndarray,
    pred_masks: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Segmentation Examples",
    n_examples: int = 4,
) -> None:
    """Show side-by-side images, GT masks, and predicted masks.

    Args:
        images: [N, C, H, W] numpy array, float [0, 1].
        gt_masks: [N, H, W] integer numpy array.
        pred_masks: [N, H, W] integer numpy array.
        save_path: Where to save.
        title: Figure title.
        n_examples: Number of examples to show.
    """
    n = min(n_examples, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(9, n * 3))
    if n == 1:
        axes = [axes]

    for i in range(n):
        img = images[i]
        if img.shape[0] in (1, 3):
            img = img.transpose(1, 2, 0)
        if img.shape[-1] == 1:
            img = img[:, :, 0]

        axes[i][0].imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1)
        axes[i][0].set_title("Image" if i == 0 else "")
        axes[i][0].axis("off")

        num_classes = max(int(gt_masks[i].max()), int(pred_masks[i].max())) + 1
        axes[i][1].imshow(mask_to_rgb(gt_masks[i], num_classes), vmin=0, vmax=1)
        axes[i][1].set_title("GT Mask" if i == 0 else "")
        axes[i][1].axis("off")

        axes[i][2].imshow(mask_to_rgb(pred_masks[i], num_classes), vmin=0, vmax=1)
        axes[i][2].set_title("Predicted" if i == 0 else "")
        axes[i][2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)
    else:
        plt.show()
