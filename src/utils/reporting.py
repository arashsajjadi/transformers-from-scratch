"""
Reporting utilities: save metrics, update summaries, and generate session reports.

Every notebook and script uses these functions to save consistent outputs.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.utils.paths import RUNS_SUMMARY_MD, REPO_ROOT


def save_metrics_json(metrics: Dict[str, Any], path: Path) -> None:
    """Save a metrics dictionary to a JSON file.

    Args:
        metrics: Dictionary of metric names to values.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def save_metrics_csv(metrics: Dict[str, Any], path: Path) -> None:
    """Save a metrics dictionary to a two-column CSV (name, value).

    Args:
        metrics: Dictionary of metric names to values.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, val in metrics.items():
            writer.writerow([key, val])


def save_table_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Save a list of dictionaries to a CSV file.

    Args:
        rows: List of dicts, each representing one row.
        path: Output file path.
    """
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_markdown_report(
    title: str,
    summary: str,
    metrics: Dict[str, Any],
    figures: List[str],
    tables: List[str],
    output_path: Path,
    device: str = "unknown",
    hyperparams: Optional[Dict[str, Any]] = None,
    architecture: Optional[str] = None,
    loss_fn: Optional[str] = None,
) -> None:
    """Generate and save a session report in Markdown format.

    Args:
        title: Session name, e.g. "Text Classification".
        summary: One or two sentence summary of the session.
        metrics: Dictionary of final metric values.
        figures: List of figure filenames (relative to the report's directory).
        tables: List of table filenames.
        output_path: Where to save the report (e.g. runs/text_classification/session_report.md).
        device: Device string, e.g. "cuda" or "cpu".
        hyperparams: Optional dict of hyperparameter values.
        architecture: Optional short text description of the model.
        loss_fn: Optional name of the loss function used.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# Session Report: {title}",
        "",
        f"**Date:** {timestamp}  ",
        f"**Device:** {device}  ",
        "",
        "## Summary",
        "",
        summary,
        "",
    ]

    if architecture:
        lines += ["## Architecture", "", "```", architecture, "```", ""]

    if loss_fn:
        lines += [f"**Loss function:** {loss_fn}", ""]

    if hyperparams:
        lines += ["## Hyperparameters", "", "| Parameter | Value |", "|-----------|-------|"]
        for k, v in hyperparams.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    if metrics:
        lines += ["## Metrics", "", "| Metric | Value |", "|--------|-------|"]
        for k, v in metrics.items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(f"| {k} | {val_str} |")
        lines.append("")

    if figures:
        lines += ["## Figures", ""]
        for fig in figures:
            lines.append(f"![{fig}]({fig})")
        lines.append("")

    if tables:
        lines += ["## Tables", ""]
        for tbl in tables:
            lines.append(f"- [{tbl}]({tbl})")
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def update_runs_summary(
    session_name: str,
    report_path: Path,
    metrics: Dict[str, Any],
    figures: List[str],
) -> None:
    """Update the central runs/SUMMARY.md file with results from a session.

    This function appends or replaces the session's row in the summary table.

    Args:
        session_name: Short name for the session, e.g. "text_classification".
        report_path: Path to the session's report markdown file.
        metrics: Dictionary of metric values for this session.
        figures: List of figure filenames produced.
    """
    summary_path = RUNS_SUMMARY_MD
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Format metrics as a short string
    metrics_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in list(metrics.items())[:3])
    figures_str = ", ".join(figures[:2]) if figures else "—"

    try:
        rel_report = report_path.relative_to(RUNS_SUMMARY_MD.parent)
    except ValueError:
        rel_report = report_path

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_row = (
        f"| {session_name} | {timestamp} | {metrics_str} | {figures_str} | "
        f"[report]({rel_report}) |"
    )

    # Read existing content or start fresh
    if summary_path.exists():
        content = summary_path.read_text()
    else:
        content = (
            "# Runs Summary\n\n"
            "This file is updated automatically after every notebook or script run.\n\n"
            "| Session | Last run | Metrics | Figures | Report |\n"
            "|---------|----------|---------|---------|--------|\n"
        )

    # Replace existing row for this session or append a new one
    lines = content.splitlines()
    new_lines = []
    replaced = False
    for line in lines:
        if line.startswith(f"| {session_name} |"):
            new_lines.append(new_row)
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        new_lines.append(new_row)

    summary_path.write_text("\n".join(new_lines) + "\n")
