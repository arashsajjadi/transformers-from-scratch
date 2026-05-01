"""
Path constants for the repository.

All paths are relative to the repository root and use pathlib.Path so they
work on Linux, macOS, and Windows without modification.
"""

from pathlib import Path

# The repository root is two levels above this file (src/utils/paths.py -> repo root)
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Top-level directories
DATA_DIR: Path = REPO_ROOT / "data"
RUNS_DIR: Path = REPO_ROOT / "runs"
ASSETS_DIR: Path = REPO_ROOT / "assets"
NOTEBOOKS_DIR: Path = REPO_ROOT / "notebooks"
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"
DOCS_DIR: Path = REPO_ROOT / "docs"

# Data subdirectories
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
DATA_TINY_DIR: Path = DATA_DIR / "tiny"

# Runs subdirectories
RUNS_SETUP_DIR: Path = RUNS_DIR / "setup"
RUNS_ATTENTION_DIR: Path = RUNS_DIR / "attention"
RUNS_TEXT_CLS_DIR: Path = RUNS_DIR / "text_classification"
RUNS_TRANSLATION_DIR: Path = RUNS_DIR / "translation"
RUNS_TIMESERIES_DIR: Path = RUNS_DIR / "timeseries"
RUNS_VIT_CLS_DIR: Path = RUNS_DIR / "vit_classification"
RUNS_SEGMENTATION_DIR: Path = RUNS_DIR / "segmentation"
RUNS_DETECTION_DIR: Path = RUNS_DIR / "detection"

# Tiny dataset files
TINY_SENTIMENT_CSV: Path = DATA_TINY_DIR / "tiny_sentiment.csv"
TINY_TRANSLATION_CSV: Path = DATA_TINY_DIR / "tiny_translation_en_fr.csv"

# Summary file
RUNS_SUMMARY_MD: Path = RUNS_DIR / "SUMMARY.md"
