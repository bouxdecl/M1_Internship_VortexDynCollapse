"""
athena_collapse_analysis — tools for running and analysing Athena++ simulations
"""

from importlib.metadata import version

__all__ = ["io", "setup", "analysis", "utils"]
__version__ = "0.1.0"

# Optional: expose common utilities here for convenience
from .config import DATA_DIR, RAW_DIR, PROCESSED_DIR, PLOTS_DIR
