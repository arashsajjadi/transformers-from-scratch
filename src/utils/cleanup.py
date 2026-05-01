"""
Run directory cleanup utilities.

These functions manage outputs in the runs/ folder. Before each run,
old output files are removed so stale results do not accumulate.

Safety rules:
- Never delete .gitkeep files.
- Never delete directories.
- Never operate outside the runs/ folder.
- Check that the path is inside REPO_ROOT before deleting anything.
"""

from pathlib import Path
from src.utils.paths import REPO_ROOT, RUNS_DIR

# File extensions that are safe to delete from run directories
CLEANABLE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".csv", ".json", ".md", ".txt"}


def _is_safe_to_clean(run_dir: Path) -> bool:
    """Return True if run_dir is inside the runs/ folder.

    This is a safety check to prevent accidentally deleting files outside
    the repository's runs/ directory.

    Args:
        run_dir: The directory to check.

    Returns:
        bool: True if safe to clean.
    """
    try:
        run_dir.resolve().relative_to(RUNS_DIR.resolve())
        return True
    except ValueError:
        return False


def clean_run_dir(run_dir: Path, keep_gitkeep: bool = True) -> int:
    """Delete old output files from a run directory.

    Only deletes files with extensions in CLEANABLE_EXTENSIONS.
    Never deletes .gitkeep, subdirectories, or files outside runs/.

    Args:
        run_dir: The run directory to clean. Must be inside runs/.
        keep_gitkeep: If True, preserve .gitkeep files. Default True.

    Returns:
        int: Number of files deleted.

    Raises:
        ValueError: If run_dir is not inside the runs/ folder.
    """
    run_dir = Path(run_dir)

    if not _is_safe_to_clean(run_dir):
        raise ValueError(
            f"Safety check failed: {run_dir} is not inside {RUNS_DIR}. "
            f"Cleanup aborted to protect repository files."
        )

    if not run_dir.exists():
        return 0

    deleted = 0
    for file_path in run_dir.iterdir():
        if file_path.is_dir():
            # Never delete directories
            continue
        if keep_gitkeep and file_path.name == ".gitkeep":
            continue
        if file_path.suffix.lower() in CLEANABLE_EXTENSIONS:
            file_path.unlink()
            deleted += 1

    return deleted


def prepare_run_dir(run_dir: Path, clean: bool = True) -> Path:
    """Create a run directory and optionally clean old outputs.

    This function is called at the start of every notebook and script.

    Args:
        run_dir: The run directory path. Must be inside runs/.
        clean: If True, delete old output files before starting.

    Returns:
        Path: The prepared run directory.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if clean:
        n_deleted = clean_run_dir(run_dir)
        if n_deleted > 0:
            print(f"[cleanup] Removed {n_deleted} old file(s) from {run_dir.relative_to(REPO_ROOT)}")

    return run_dir
