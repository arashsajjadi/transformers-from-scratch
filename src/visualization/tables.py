"""
Table visualization: render data tables as matplotlib figures.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.visualization.plots import save_figure


def render_table_as_figure(
    rows: List[dict],
    save_path: Optional[Path] = None,
    title: str = "",
    max_rows: int = 20,
    col_width: float = 2.5,
    row_height: float = 0.4,
    font_size: int = 9,
) -> None:
    """Render a list of dicts as a matplotlib table figure.

    Args:
        rows: List of row dicts. All dicts must have the same keys.
        save_path: Where to save the figure.
        title: Table title.
        max_rows: Maximum rows to display.
        col_width: Width per column in inches.
        row_height: Height per row in inches.
        font_size: Font size for table cells.
    """
    if not rows:
        return

    rows = rows[:max_rows]
    columns = list(rows[0].keys())
    cell_data = [[str(row.get(col, "")) for col in columns] for row in rows]

    fig_width = len(columns) * col_width
    fig_height = (len(rows) + 2) * row_height

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_data,
        colLabels=columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.auto_set_column_width(range(len(columns)))

    # Style header row
    for j in range(len(columns)):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", weight="bold")

    # Alternating row colors
    for i in range(1, len(rows) + 1):
        color = "#F2F2F2" if i % 2 == 0 else "white"
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(color)

    if title:
        ax.set_title(title, fontsize=font_size + 2, pad=10)

    plt.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)
    else:
        plt.show()
