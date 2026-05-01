"""
Detection visualization: draw bounding boxes on images.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from src.visualization.plots import save_figure

# Colors for each class
DETECTION_COLORS = ["red", "blue", "green", "orange", "purple", "cyan"]
CLASS_NAMES = ["circle", "square", "triangle"]


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    class_ids: Optional[List[int]] = None,
    scores: Optional[List[float]] = None,
    color: str = "red",
    class_names: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    """Draw bounding boxes on an image axes.

    Args:
        image: [H, W, C] or [H, W] numpy float array in [0, 1].
        boxes: [N, 4] in (x_min, y_min, x_max, y_max) pixel coordinates.
        class_ids: Optional list of integer class IDs per box.
        scores: Optional list of confidence scores per box.
        color: Default box color if class_ids is not provided.
        class_names: Optional list of class name strings.
        ax: Matplotlib axes to draw on. Creates a new figure if None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        created_fig = True
    else:
        created_fig = False

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[0] in (1, 3):
        image = image.transpose(1, 2, 0)
    if image.shape[-1] == 1:
        image = np.concatenate([image] * 3, axis=-1)

    ax.imshow(image, vmin=0, vmax=1)

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        w = x_max - x_min
        h = y_max - y_min

        if class_ids is not None and i < len(class_ids):
            c = class_ids[i] % len(DETECTION_COLORS)
            box_color = DETECTION_COLORS[c]
        else:
            box_color = color

        rect = patches.Rectangle(
            (x_min, y_min), w, h,
            linewidth=2, edgecolor=box_color, facecolor="none"
        )
        ax.add_patch(rect)

        # Label
        label_parts = []
        if class_names is not None and class_ids is not None and i < len(class_ids):
            cid = class_ids[i]
            if 0 <= cid < len(class_names):
                label_parts.append(class_names[cid])
        if scores is not None and i < len(scores):
            label_parts.append(f"{scores[i]:.2f}")
        if label_parts:
            label = " ".join(label_parts)
            ax.text(
                x_min, y_min - 2, label,
                color=box_color, fontsize=8, weight="bold",
                bbox=dict(facecolor="white", alpha=0.5, pad=1, edgecolor="none"),
            )

    ax.axis("off")


def plot_detection_examples(
    images: np.ndarray,
    gt_boxes_list: List[np.ndarray],
    gt_class_ids_list: List[List[int]],
    pred_boxes_list: Optional[List[np.ndarray]] = None,
    pred_class_ids_list: Optional[List[List[int]]] = None,
    pred_scores_list: Optional[List[List[float]]] = None,
    save_path: Optional[Path] = None,
    title: str = "Detection Examples",
    n_examples: int = 4,
    class_names: Optional[List[str]] = None,
) -> None:
    """Plot ground-truth and predicted bounding boxes side by side.

    Args:
        images: [N, C, H, W] numpy array.
        gt_boxes_list: List of [M, 4] xyxy pixel boxes per image.
        gt_class_ids_list: List of class ID lists per image.
        pred_boxes_list: Optional predicted boxes.
        pred_class_ids_list: Optional predicted class IDs.
        pred_scores_list: Optional predicted scores.
        save_path: Where to save.
        title: Figure title.
        n_examples: Number of examples.
        class_names: Class name strings.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    n = min(n_examples, len(images))
    has_preds = pred_boxes_list is not None
    ncols = 2 if has_preds else 1
    fig, axes = plt.subplots(n, ncols, figsize=(ncols * 4, n * 4))

    if n == 1 and ncols == 1:
        axes = [[axes]]
    elif n == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    for i in range(n):
        img = images[i].copy()

        draw_bounding_boxes(
            img,
            gt_boxes_list[i] if len(gt_boxes_list) > i else np.zeros((0, 4)),
            gt_class_ids_list[i] if len(gt_class_ids_list) > i else [],
            class_names=class_names,
            ax=axes[i][0],
        )
        axes[i][0].set_title("Ground Truth" if i == 0 else "")

        if has_preds and len(axes[i]) > 1:
            draw_bounding_boxes(
                img,
                pred_boxes_list[i] if len(pred_boxes_list) > i else np.zeros((0, 4)),
                pred_class_ids_list[i] if (pred_class_ids_list and len(pred_class_ids_list) > i) else [],
                pred_scores_list[i] if (pred_scores_list and len(pred_scores_list) > i) else None,
                class_names=class_names,
                ax=axes[i][1],
            )
            axes[i][1].set_title("Predictions" if i == 0 else "")

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)
    else:
        plt.show()
