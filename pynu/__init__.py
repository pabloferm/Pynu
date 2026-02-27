"""Pynu - Python Neutrino Fitting Framework."""

from .PyNuFit import PyNuFit

# Optional imports - Report requires pylatex which may not be installed
try:
    from .Report import Report
except ImportError:
    Report = None

try:
    from .Plot import Plot
except ImportError:
    Plot = None

__version__ = "1.0.0"
__all__ = ["PyNuFit", "Report", "Plot"]
