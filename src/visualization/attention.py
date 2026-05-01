"""
Attention visualization: heatmaps for single-head and multi-head attention.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.visualization.plots import save_figure


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    save_path: Optional[Path] = None,
    title: str = "Attention Weights",
    cmap: str = "viridis",
) -> None:
    """Plot a single attention weight matrix as a heatmap.

    Args:
        attention_weights: [seq_q, seq_k] numpy array of attention weights.
        row_labels: Labels for the query (row) dimension.
        col_labels: Labels for the key (column) dimension.
        save_path: Where to save the figure.
        title: Plot title.
        cmap: Colormap.
    """
    fig, ax = plt.subplots(figsize=(max(4, len(col_labels) * 0.6), max(3, len(row_labels) * 0.5)))

    im = ax.imshow(attention_weights, cmap=cmap, vmin=0, vmax=attention_weights.max())
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Keys (attending to)")
    ax.set_ylabel("Queries (attending from)")
    ax.set_title(title)

    plt.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)
    else:
        plt.show()


def plot_multi_head_attention(
    attention_weights: torch.Tensor,
    row_labels: List[str],
    col_labels: List[str],
    save_path: Optional[Path] = None,
    title: str = "Multi-Head Attention",
) -> None:
    """Plot one heatmap per attention head.

    Args:
        attention_weights: [num_heads, seq_q, seq_k] tensor or numpy array.
        row_labels: Labels for rows (query tokens).
        col_labels: Labels for columns (key tokens).
        save_path: Where to save.
        title: Figure title.
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    num_heads = attention_weights.shape[0]
    ncols = min(num_heads, 4)
    nrows = (num_heads + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * max(3, len(col_labels) * 0.5), nrows * max(2.5, len(row_labels) * 0.4))
    )

    # Flatten axes for easy indexing
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

    for h in range(num_heads):
        ax = axes_flat[h]
        im = ax.imshow(attention_weights[h], cmap="viridis", vmin=0, aspect="auto")
        ax.set_xticks(range(len(col_labels)))
        ax.set_yticks(range(len(row_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_title(f"Head {h + 1}", fontsize=9)

    # Hide extra subplots
    for h in range(num_heads, len(axes_flat)):
        axes_flat[h].axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)
    else:
        plt.show()
