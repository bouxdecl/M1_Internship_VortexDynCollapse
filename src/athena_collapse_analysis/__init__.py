"""
athena_collapse_analysis — tools for running and analysing Athena++ simulations
"""

from importlib.metadata import version

# Import subpackages
from . import io
from . import analysis
from . import visu

__all__ = ["io", "analysis", "visu"]
__version__ = "0.1.0"
