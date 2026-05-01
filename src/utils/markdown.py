"""
Markdown utilities: update specific sections of a Markdown file.

Used to update the README's AUTO-GENERATED-RESULTS section and other
structured parts of documentation files.
"""

from pathlib import Path
from typing import List, Optional


def append_or_replace_section(
    markdown_path: Path,
    section_title: str,
    content: str,
) -> None:
    """Replace or append a section in a Markdown file.

    Finds the section by its heading and replaces everything until the
    next heading of the same or higher level. If the section is not found,
    it is appended at the end.

    Args:
        markdown_path: Path to the Markdown file.
        section_title: The section heading text (without #), e.g. "Results".
        content: New content to put inside the section (without the heading line).
    """
    markdown_path = Path(markdown_path)
    if not markdown_path.exists():
        markdown_path.write_text(f"## {section_title}\n\n{content}\n")
        return

    text = markdown_path.read_text()
    lines = text.splitlines(keepends=True)

    # Find the heading line
    heading_prefix = None
    heading_idx = None
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        for level in range(1, 7):
            prefix = "#" * level + " "
            if stripped == f"{prefix}{section_title}":
                heading_prefix = prefix
                heading_idx = i
                break
        if heading_idx is not None:
            break

    if heading_idx is None:
        # Section not found: append it
        new_text = text.rstrip() + f"\n\n## {section_title}\n\n{content}\n"
        markdown_path.write_text(new_text)
        return

    # Find the end of this section (next heading of same or higher level)
    level = len(heading_prefix.rstrip())
    end_idx = len(lines)
    for i in range(heading_idx + 1, len(lines)):
        stripped = lines[i].rstrip()
        if stripped.startswith("#"):
            current_level = len(stripped) - len(stripped.lstrip("#"))
            if current_level <= level:
                end_idx = i
                break

    # Rebuild the file
    before = lines[:heading_idx]
    after = lines[end_idx:]
    new_section = [f"{heading_prefix}{section_title}\n", "\n"] + [
        line + "\n" for line in content.splitlines()
    ] + ["\n"]

    markdown_path.write_text("".join(before + new_section + after))


def update_auto_generated_section(markdown_path: Path, new_content: str) -> None:
    """Update the AUTO-GENERATED-RESULTS block in a Markdown file.

    Replaces the content between:
      <!-- AUTO-GENERATED-RESULTS:START -->
    and
      <!-- AUTO-GENERATED-RESULTS:END -->

    If the markers are not found, does nothing.

    Args:
        markdown_path: Path to the Markdown file (e.g. README.md).
        new_content: New content to put between the markers.
    """
    markdown_path = Path(markdown_path)
    if not markdown_path.exists():
        return

    text = markdown_path.read_text()
    start_marker = "<!-- AUTO-GENERATED-RESULTS:START -->"
    end_marker = "<!-- AUTO-GENERATED-RESULTS:END -->"

    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        return  # Markers not found; do not modify

    before = text[: start_idx + len(start_marker)]
    after = text[end_idx:]
    markdown_path.write_text(f"{before}\n{new_content}\n{after}")


def save_dataframe_table_as_markdown(
    rows: List[dict],
    path: Path,
    title: Optional[str] = None,
) -> None:
    """Save a list of dicts as a Markdown table file.

    Args:
        rows: List of dicts with the same keys.
        path: Output file path.
        title: Optional title to put above the table.
    """
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = list(rows[0].keys())
    lines = []
    if title:
        lines.append(f"# {title}\n")
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = [str(row.get(col, "")) for col in columns]
        lines.append("| " + " | ".join(values) + " |")

    path.write_text("\n".join(lines) + "\n")
