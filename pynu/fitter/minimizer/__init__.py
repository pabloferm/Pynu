"""pynu.fitter.minimizer — minimizer utilities + the binned per-point fit.

Intentionally does NOT eagerly import its submodules: the binned per-point fit
protocol + tensor-store / binding (``binned_fit``) load only on explicit import,
preserving the "toggle-OFF pulls no binned code" invariant (Gate C-0 / F-C2).
"""
