# src/athena_analysis/config.py

import os
from pathlib import Path

# Path to the root of the repository (two levels above this file)
PKG_DIR = Path(__file__).resolve().parent  # src/athena_analysis
REPO_ROOT = PKG_DIR.parent.parent          # athena-analysis/

# Default data directory (can be overridden by an environment variable)
DATA_DIR = Path(os.getenv("ATHENA_COLLAPSE_ANALYSIS_DATA", REPO_ROOT / "data"))


# Subdirectories
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PLOTS_DIR = DATA_DIR / "plots"

# Ensure they exist
for d in [RAW_DIR, PROCESSED_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
