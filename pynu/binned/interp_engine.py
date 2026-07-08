#!/usr/bin/env python3
"""Continuous phi(Delta m^2, s23) interpolator + joint-fit objective for the SK
binned GLOBAL-MINIMUM search (HANDOFF_2026-06-28_mcmc_interp.md, steps 2 & 4).

Purpose (2026-06-28/29 framing): the grid scan still draws the CONTOUR; this
gives an ACCURATE GLOBAL MINIMUM (best-fit osc point + profiled nuisances), free
of the per-cell grid convergence scatter, by interpolating the pre-built osc
tensors so a continuous (Delta m^2, s23) optimiser / sampler never re-propagates
nuSQuIDS.

GATE (interp_gate.py + fit-level tests, 2026-06-28): CUBIC interpolation of the
15x15 osc-tensor grid is FIT-GRADE for the global minimum -- nuisance-profiled
|dChi2| <~ 0.6 at the conservative leave-one-out distance (~0.04 at the real
<=1/2-step), and the interpolant's safe-band minimum equals the node minimum
exactly (no aliasing wiggles). Constraints baked in here:
  * dCP is PROFILED on the 13 discrete nodes (fit_point does this) -- NOT
    interpolated (13 nodes carry power to the H6 Nyquist limit).
  * Delta m^2 is interpolated but TRUSTED only in SAFE_DM = [2.32, 2.857]e-3.
    Above ~2.86e-3 the per-cell phi(Delta m^2) aliases -> spurious LOW chi2 dips;
    the joint optimiser MUST stay bounded (the grid scan owns that region).
  * s23 is clean across the full [0.40, 0.80].

The interpolant is a separable natural cubic spline over ALL 15 grid nodes per
axis (the scheme the gate validated); queries are bounded to the trusted box by
JointSKFit, not by the interpolator (so the spline stencil keeps correct
neighbour nodes). phi enters the forward model linearly, so interpolating phi
then contracting == interpolating the observable -- verified at machine precision.
"""
import glob
import os
import numpy as np
from scipy.interpolate import CubicSpline

# Module DEFAULTS (original 15x15 build). The interpolator now reads the actual
# node axes from the tensor directory (detect_grid), so these are only fallbacks;
# the 49x17 fine grid is picked up automatically.
DM = np.linspace(2.0e-3, 3.5e-3, 15)
S23 = np.linspace(0.40, 0.80, 15)
SAFE_DM = (2.32e-3, 2.857e-3)        # OLD 15x15 trusted range; new grid uses its full extent
SAFE_S23 = (0.40, 0.80)


def detect_grid(tensors_dir):
    """Read the (Delta m^2, s23) node axes straight from the tensor files so the
    interpolator serves any build density/extent. Returns (DM, S23) sorted axes."""
    ij = {}
    for f in glob.glob(os.path.join(tensors_dir, "osc_tensor_*_*.npz")):
        b = os.path.basename(f)[len("osc_tensor_"):-len(".npz")]
        try:
            i, j = (int(x) for x in b.split("_"))
        except ValueError:
            continue
        ij[(i, j)] = f
    if not ij:
        return DM, S23
    ni = max(i for i, j in ij) + 1
    nj = max(j for i, j in ij) + 1
    dm = np.array([float(np.load(ij[(i, 0)])["dm231"]) for i in range(ni)])
    s23 = np.array([float(np.load(ij[(0, j)])["s23"]) for j in range(nj)])
    return dm, s23


class PhiInterpolator:
    """phi(dm231, s23) -> (13, 2, 3, nE, nZ) via separable cubic over the grid.

    Exploits the bicubic LINEARITY in the node data: a tensor-product cubic is
      phi(dm,s23) = sum_i w_dm[i](dm) * [s23-cubic of row i](s23)
                  = [s23-cubic of  sum_i w_dm[i](dm) * row_i ](s23),
    so for a FIXED dm (the per-row scan) we dm-combine the rows ONCE into 15
    effective col-tensors, build ONE s23 spline, and reuse it for every s23 query
    (`row_spline`). This is the exact global natural-cubic scheme the gate
    validated, just reorganised to avoid rebuilding 15 splines per call.

    Memory: raw tensors cached on load (all 15x15 ~ 2.25 GB for the global dm
    cubic); a row_spline transiently holds the 15 effective cols + one spline
    (~1.4 GB). phi is queried inside fit_point's re-contraction loop, but only
    ONE phi per (dm,s23) point, so interp cost (~3 s/row) is negligible.
    """

    def __init__(self, tensors_dir, rows=None, cache_raw=False):
        self.dir = tensors_dir
        # grid-aware: node axes read from the tensors (any density/extent)
        self.DM, self.S23 = detect_grid(tensors_dir)
        self.n_s23 = len(self.S23)
        self.rows = list(range(len(self.DM))) if rows is None else list(rows)
        self.cache_raw = cache_raw           # True for repeated use (e.g. a future MCMC); the
        self._raw = {}                       # per-row worker streams (one call)
        self._wdm_spline = None

    def _load(self, i, j):
        """Load phi[i,j] as float32 (cache optional). Promotion to float64 happens
        only in the transient dm-combination, so the cache stays at ~2.25 GB and a
        streaming (uncached) row_spline peaks at ~1.7 GB."""
        k = (i, j)
        if k in self._raw:
            return self._raw[k]
        p = os.path.join(self.dir, f"osc_tensor_{i:03d}_{j:03d}.npz")
        phi = np.load(p)["phi"]              # float32
        if self.cache_raw:
            self._raw[k] = phi
        return phi

    def _dm_weights(self, dm231):
        """Cubic-spline weights w[i] with f(dm) = sum_i w[i] f_i (linearity)."""
        if self._wdm_spline is None:
            self._wdm_spline = CubicSpline(self.DM[self.rows],
                                           np.eye(len(self.rows)), axis=0)
        return self._wdm_spline(dm231)                  # (nrows,)

    def row_spline(self, dm231):
        """s23-cubic spline of phi at fixed dm231 (reusable across s23)."""
        w = self._dm_weights(dm231)
        cols = []
        for c in range(self.n_s23):
            acc = None
            for wi, i in zip(w, self.rows):
                t = wi * self._load(i, c)
                acc = t if acc is None else acc + t
            cols.append(acc)
        return CubicSpline(self.S23, np.stack(cols), axis=0)

    def __call__(self, dm231, s23):
        return self.row_spline(dm231)(s23)


class JointSKFit:
    """Joint-fit objective: continuous (dm231, s23) -> profiled chi2.

    chi2(dm231, s23) = min over {13 dCP nodes, nuisances} of the binned Poisson
    chi2 -- i.e. fit_point on the interpolated phi. This is the profile likelihood
    in the osc plane; the inner fit_point already marginalises dCP + nuisances
    analytically/iteratively. The optimiser/sampler runs on this 2-D surface.

    Queries outside the trusted box return +inf (penalty) so a bounded optimiser
    cannot chase the high-Delta m^2 aliasing dips.
    """

    def __init__(self, engine, interp, theta0=None,
                 safe_dm=None, safe_s23=None):
        self.eng = engine
        self.interp = interp
        self.theta0 = engine.nominal.copy() if theta0 is None else np.asarray(theta0)
        # default the box to the interpolator's actual grid extent
        self.safe_dm = safe_dm or (float(interp.DM[0]), float(interp.DM[-1]))
        self.safe_s23 = safe_s23 or (float(interp.S23[0]), float(interp.S23[-1]))
        self._last_nuis = self.theta0.copy()     # warm-start chain across calls

    def in_box(self, dm231, s23):
        return (self.safe_dm[0] <= dm231 <= self.safe_dm[1] and
                self.safe_s23[0] <= s23 <= self.safe_s23[1])

    def chi2(self, dm231, s23, x0=None, warm=True, npolish=2):
        if not self.in_box(dm231, s23):
            return np.inf
        phi = self.interp(dm231, s23)
        seed = x0 if x0 is not None else (self._last_nuis if warm else self.theta0)
        c, dcp, nuis, _, _ = self.eng.fit_point(phi, x0=seed)
        for _ in range(npolish):                 # restart-polish (resets Hessian)
            pc, pd, pn, _, _ = self.eng.fit_point(phi, x0=nuis)
            if pc < c - 1e-3:
                c, dcp, nuis = pc, pd, pn
            else:
                break
        if warm:
            self._last_nuis = nuis               # propagate the good basin
        return c, dcp, nuis

    def __call__(self, x):
        """scipy objective: x = [dm231, s23] -> profiled chi2 (float)."""
        return self.chi2(float(x[0]), float(x[1]))[0]
