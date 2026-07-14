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
__all__ = [
    "SKBinnedEngine",
    "resolve_nuisance_spec",
    "PhiInterpolator",
    "detect_grid",
    "BinnedConfig",
    "parse_binned_config",
    "BinnedBinding",
    "TensorStore",
    "sample_rates",
    "detector_factors",
    "EScaleHistogramOperator",
    "bin_era_from_sample_table",
]

# name -> defining submodule; imported on first attribute access only.
# (S.F1) BinnedConfig / parse_binned_config moved to
# ``pynu.analysis_reader.binned_config`` (their functional home) and are
# re-exported here for back-compat via the same PEP 562 lazy map. (S.F3) the
# φ interpolator moved to ``pynu.fitter.inference.interp_engine`` and the φ
# tensor store + loaded-triple holder moved to
# ``pynu.fitter.minimizer.binned_fit`` — all re-exported here for back-compat.
# (S.F4) the descriptor detector-factor kernels moved to
# ``pynu.PhysicsTunes.Detector.detector`` (beside SKCombinedDetector) and the
# event-side energy-scale histogram-transfer operator moved to
# ``pynu.PhysicsTunes.Detector.escale_operator`` — both re-exported here for
# back-compat. Absolute module paths ('pynu...') are used for symbols that live
# outside this package; relative paths ('.mod') for in-package submodules.
_LAZY = {
    "SKBinnedEngine": "pynu.Experiments.sk_binned_engine",
    "resolve_nuisance_spec": "pynu.analysis_reader.binned_dials",
    "PhiInterpolator": "pynu.fitter.inference.interp_engine",
    "detect_grid": "pynu.fitter.inference.interp_engine",
    "BinnedBinding": "pynu.fitter.minimizer.binned_fit",
    "TensorStore": "pynu.fitter.minimizer.binned_fit",
    "BinnedConfig": "pynu.analysis_reader.binned_config",
    "parse_binned_config": "pynu.analysis_reader.binned_config",
    "sample_rates": "pynu.PhysicsTunes.Detector.detector",
    "detector_factors": "pynu.PhysicsTunes.Detector.detector",
    "EScaleHistogramOperator": "pynu.PhysicsTunes.Detector.escale_operator",
    "bin_era_from_sample_table": "pynu.PhysicsTunes.Detector.escale_operator",
}

# (Track T / T3) the resident modules physically moved out of this package
# (engine trio + builder -> pynu/Experiments/, cell-weight factor sourcing ->
# pynu/PhysicsTunes/TuneFactorSource.py). ``from pynu.binned import
# sk_binned_engine`` (the attribute form the historical gate/campaign scripts
# use) keeps resolving via this map — the whole MODULE is forwarded.
# The dotted form ``import pynu.binned.sk_binned_engine`` is NOT recoverable
# by PEP 562 and is gone; in-tree callers were rewired at T3.
_LAZY_MODULES = {
    "sk_binned_engine": "pynu.Experiments.sk_binned_engine",
    "engine_core": "pynu.Experiments.sk_binned_engine_core",
    "masks": "pynu.Experiments.sk_binned_masks",
    "builder": "pynu.Experiments.sk_binned_builder",
    "grid_experiment": "pynu.PhysicsTunes.TuneFactorSource",
}


def __getattr__(name):
    import importlib
    target = _LAZY_MODULES.get(name)
    if target is not None:
        return importlib.import_module(target)
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # relative ('.mod') resolves against this package; absolute passes package=None
    package = __name__ if module.startswith(".") else None
    return getattr(importlib.import_module(module, package), name)


def __dir__():
    return sorted(__all__)
