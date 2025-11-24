"""
athena_collapse_analysis — tools for running and analysing Athena++ simulations
"""

from importlib.metadata import version

# Import subpackages
from . import io
from . import setup
from . import analysis
from . import utils

__all__ = ["io", "setup", "analysis", "utils"]
__version__ = "0.1.0"
