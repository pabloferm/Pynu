"""pynu.binned — native SK binned-tensor forward model + fit, behind a
default-OFF ``<BinnedEngine>`` XML toggle.

This subpackage is self-contained (json / numpy / scipy only) and does NOT import
the rest of pynu, so importing it cannot pull in the heavy event pipeline
(nuSQuIDS, event MC). PyNuFit imports it lazily, and only ever reaches the
forward model when an analysis XML actually declares a ``<BinnedEngine>`` block.

To keep "toggle-OFF = zero code executed" literally true, only the lightweight,
stdlib-only config symbols load eagerly; the engine / interpolator / adapter load
lazily (PEP 562) on first access. So a PyNuFit construction that merely checks a
toggle-free XML (``parse_binned_config`` -> ``{}``) runs no forward-model code.

``sk_binned_engine`` / ``interp_engine`` are verbatim vendored snapshots — see
``PROVENANCE.md`` for source commits, sha256, and the resync protocol.
"""
from .config import BinnedConfig, parse_binned_config

__all__ = [
    "SKBinnedEngine",
    "resolve_nuisance_spec",
    "PhiInterpolator",
    "detect_grid",
    "BinnedConfig",
    "parse_binned_config",
    "BinnedEngineAdapter",
]

# name -> defining submodule; imported on first attribute access only.
_LAZY = {
    "SKBinnedEngine": ".sk_binned_engine",
    "resolve_nuisance_spec": ".sk_binned_engine",
    "PhiInterpolator": ".interp_engine",
    "detect_grid": ".interp_engine",
    "BinnedEngineAdapter": ".adapter",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(module, __name__), name)


def __dir__():
    return sorted(__all__)
