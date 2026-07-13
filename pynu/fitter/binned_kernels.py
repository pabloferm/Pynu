#!/usr/bin/env python3
"""Binned χ² kernels (Track S·F, Phase F3).

The two stateless statistics kernels of the SK binned fit — the plain Poisson
form and the Barlow-Beeston-lite form — re-homed here from
``pynu/binned/engine_core.py`` into the ``pynu.fitter`` package, their functional
home (the delegation target of ``PoissonLikelihood.set_engine`` already reads the
engine's ``poisson_chi2`` kernel). ZERO numerical change: both bodies are a
verbatim move of the former ``engine_core.bb_chi2`` / ``engine_core.poisson_chi2``
functions (every guard, ordering, epsilon and comment preserved).

Neither kernel takes the engine instance nor references any engine global — they
operate purely on the passed observation / model / variance arrays, so this is a
pure relocation. ``engine_core`` re-imports both names for its own delegates, and
``SKBinnedEngine`` exposes them as static-method passthroughs.

This module is NOT imported by ``pynu/fitter/__init__.py`` (kept out of the eager
likelihood-class imports), so importing ``pynu.fitter`` stays stdlib+numpy only
and the toggle-OFF event path pulls no binned code (Gate C-0 invariant).
"""
import numpy as np


def bb_chi2(obs, n_mod, var):
    """BarlowBeestonLikelihood.stats_only (BB-lite, no muons)."""
    tau = np.divide(var, n_mod ** 2, out=np.zeros_like(var), where=n_mod != 0)
    b = n_mod * tau - 1.0
    c = -obs * tau
    beta = 0.5 * (-b + np.sqrt(np.maximum(0, b * b - 4 * c)))
    beta = np.maximum(beta, 1e-9)
    beta_E = np.maximum(beta * n_mod, 1e-9)
    log_term = np.log(np.divide(obs, beta_E, out=np.ones_like(obs),
                                where=beta_E > 0))
    log_term[obs == 0] = 0
    poisson = np.sum(2 * (beta_E - obs + obs * log_term))
    bb_pen = np.sum(np.divide((beta - 1) ** 2, tau, out=np.zeros_like(tau),
                              where=tau > 0))
    return poisson + bb_pen, beta, tau


def poisson_chi2(obs, n_mod):
    """Plain Poisson chi2 (event engine's no-MC-variance fallback form)."""
    if np.any(n_mod <= 0):
        return 9e9
    log_term = np.log(np.divide(obs, n_mod, out=np.ones_like(obs),
                                where=n_mod > 0))
    log_term[obs == 0] = 0
    return float(2 * np.sum(n_mod - obs + obs * log_term))
