"""pynu.binned — native SK binned-tensor forward model + fit, behind a
default-OFF ``<BinnedEngine>`` XML toggle.

This subpackage is self-contained (json / numpy / scipy only) and does NOT import
the rest of pynu, so importing it cannot pull in the heavy event pipeline
(nuSQuIDS, event MC). PyNuFit imports it lazily, and only ever reaches the
forward model when an analysis XML actually declares a ``<BinnedEngine>`` block.

To keep "toggle-OFF = zero code executed" literally true, only the lightweight,
stdlib-only config symbols load eagerly; the engine / interpolator / binding load
lazily (PEP 562) on first access. So a PyNuFit construction that merely checks a
toggle-free XML (``parse_binned_config`` -> ``{}``) runs no forward-model code.

The package is native (Track S de-vendoring complete at E6): the engine +
kernels + descriptor modules are owned code, and the SK dial values ship as
package-data value XMLs. ``PROVENANCE.md`` is the historical record of the former
vendoring era.
"""
from .config import BinnedConfig, parse_binned_config

__all__ = [
    "SKBinnedEngine",
    "resolve_nuisance_spec",
    "PhiInterpolator",
    "detect_grid",
    "BinnedConfig",
    "parse_binned_config",
    "BinnedBinding",
    "TensorStore",
]

# name -> defining submodule; imported on first attribute access only.
_LAZY = {
    "SKBinnedEngine": ".sk_binned_engine",
    "resolve_nuisance_spec": ".sk_binned_engine",
    "PhiInterpolator": ".interp_engine",
    "detect_grid": ".interp_engine",
    "BinnedBinding": ".engine_core",
    "TensorStore": ".engine_core",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(module, __name__), name)


def __dir__():
    return sorted(__all__)
