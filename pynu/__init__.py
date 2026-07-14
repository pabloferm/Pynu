"""Pynu - Python Neutrino Fitting Framework."""

__version__ = "1.0.0"
__all__ = ["PyNuFit", "Report", "Plot"]

# Lazy top-level surface (PEP 562, Track T / T4). The former eager
# ``from .PyNuFit import PyNuFit`` pulled pandas + the full framework into
# EVERY ``import pynu`` — including config-only consumers (analysis_reader /
# binned_dials / binned_config), which are stdlib+numpy. Attribute access is
# unchanged: ``from pynu import PyNuFit`` / ``pynu.PyNuFit`` resolve the class
# exactly as before (cached in module globals on first access); ``Report`` /
# ``Plot`` keep their optional-dependency semantics (None when pylatex etc.
# are absent).
_LAZY = {"PyNuFit": ".PyNuFit", "Report": ".Report", "Plot": ".Plot"}
_OPTIONAL = {"Report", "Plot"}


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    try:
        value = getattr(importlib.import_module(target, __name__), name)
    except ImportError:
        if name in _OPTIONAL:
            value = None
        else:
            raise
    globals()[name] = value   # cache: subsequent access skips __getattr__
    return value


def __dir__():
    return sorted(set(list(globals()) + __all__))
