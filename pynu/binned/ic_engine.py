"""`ICBinnedEngine`: statistical chi^2 and analytic gradient for IceCube-DeepCore.

Built on `ic_cells.ICCells` (sparse populated-cell structure), `ic_dials` (the
Mode-keyed forms) and `dial_fields` (the flux and shared cross-section forms). Every dial
is evaluated once per populated cell, and the gradient comes from one adjoint bincount.

The model:

    E_b = SUM_cat  C_cat[b] * hist_cat[b]  +  mu_b

    hist_cat[b] = SUM_{entries whose cell is in cat}  entry_w * NORM * phi_cell * W_cell
    W_cell      = PROD_{cell dials} factor(dial, geom_cell, x)
    C_cat[b]    = intercept_cat[b] + SUM_s slope_cat,s[b] * (x_s - nominal_s)

  * pure Poisson, statistical only, no prior: the caller owns the prior.
  * NORM is applied here; the response stores the raw weight.
  * muon: a 200-bin constant, added after the hypersurface correction, zero gradient.
  * phi is the oscillated flux (flux x P), so NC cells keep their phi.

The hypersurface (HS) block is additive. The 5 HS dials act only through the per-category
linear multiplier above, so

    dE_b/dx_s = SUM_cat slope_cat,s[b] * hist_cat[b]

with slopes constant at fixed Delta m^2 (set once per lattice cell by `set_hs_slopes`).
The HS categories are pure class masks, so they partition the cell axis exactly.
"""
import numpy as np

from .ic_cells import ICCells
from . import ic_dials as icf
from . import dial_fields as bdf

BARRIER_CHI2 = 9e9                             # chi^2 when a fitted bin's model rate is <= 0
N_BINS = 200
HS_DIALS = list(icf.HS_DIALS)
HS_CATEGORIES = ("nc_nue_cc", "numu_cc", "nutau_cc")
HS_NOMINALS = {"dom_eff": 1.0, "hole_ice_p0": 0.1, "hole_ice_p1": -0.05,
               "bulk_ice_abs": 1.0, "bulk_ice_scatter": 1.0}


def poisson_chi2(obs, n_mod):
    lt = np.log(np.divide(obs, n_mod, out=np.ones_like(obs), where=n_mod > 0))
    lt[obs == 0] = 0
    return float(2 * np.sum(n_mod - obs + obs * lt))


class ICBinnedEngine:
    """Statistical chi^2 and analytic gradient on the IC 200-bin reco axis.

    Args
      response_npz : ic_response_modeaxis_*.npz (must carry the |Mode| axis)
      obs200, mu200: observed and static-muon 200-bin vectors
      nuisance_names: the arm's parameter order; the gradient is returned in exactly
                      this order, pinned slots written 0.0
      norm         : exposure normalization
      few          : (200,) bool mask of the fitted bins
      pinned       : names whose gradient slot is forced to 0.0
      fluxq        : path of the flux-ratio q table
    """

    def __init__(self, response_npz, obs200, mu200, nuisance_names, norm, few, pinned, fluxq):
        self.cells = ICCells(response_npz)
        self.obs = np.asarray(obs200, float)
        self.mu = np.asarray(mu200, float)
        self.norm = float(norm)
        self.names = list(nuisance_names)
        self.n_dials = len(self.names)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.pinned = tuple(p for p in pinned if p in self.idx)
        self.few = np.asarray(few, bool)

        g = self.cells.geom
        self.shared_geom = bdf.build_cell_geometry(g.E, g.cz, g.pdg, g.cc, fluxq)
        self.registry = dict(bdf.FIELDS)
        self.registry.update(icf.FIELDS)

        # cell dials = every parameter with a field, minus HS and pinned
        self.cell_dials = [n for n in self.names
                           if n in self.registry and n not in HS_DIALS
                           and n not in self.pinned]
        missing = [n for n in self.names
                   if n not in self.registry and n not in HS_DIALS
                   and n not in self.pinned]
        if missing:
            raise SystemExit(f"no cell field for manifest dials: {missing}")

        # HS category partition of the cell axis
        a = np.abs(g.pdg)
        self.cat_mask = {
            "nc_nue_cc": (g.cc == 0) | ((a == 12) & (g.cc == 1)),
            "numu_cc": (a == 14) & (g.cc == 1),
            "nutau_cc": (a == 16) & (g.cc == 1),
        }
        stack = np.stack([self.cat_mask[c] for c in HS_CATEGORIES])
        if not np.array_equal(stack.sum(0), np.ones(self.cells.n_cell, int)):
            raise SystemExit("HS categories do not partition the cell axis exactly")
        # entry -> category, so the contraction can be done per category
        cell_cat = np.full(self.cells.n_cell, -1, np.int64)
        for i, c in enumerate(HS_CATEGORIES):
            cell_cat[self.cat_mask[c]] = i
        self.cell_cat = cell_cat
        self.entry_cat = cell_cat[self.cells.entry_cell]

        self.hs_slopes = None
        self.hs_dm31 = None
        self.hs_names = [n for n in self.names if n in HS_DIALS]

    # ------------------------------------------------------------------ model
    def _geom_for(self, name):
        """IC fields read ICCellGeom; shared fields read the shared CellGeom."""
        return self.cells.geom if name in icf.FIELDS else self.shared_geom

    def cell_weights(self, theta):
        """W_cell = product of every cell dial's factor, and the factors. Scalars (the
        off-domain collapse) multiply through as scalars."""
        th = np.asarray(theta, float)
        W = np.ones(self.cells.n_cell)
        factors = {}
        for name in self.cell_dials:
            x = float(th[self.idx[name]])
            f = self.registry[name].factor_fn(self._geom_for(name), x)
            factors[name] = f
            W = W * f
        return W, factors

    def _hs_factor(self, theta):
        """C_cat[b] per category: intercept + sum slope*(x - nominal)."""
        th = np.asarray(theta, float)
        out = {}
        for c in HS_CATEGORIES:
            s = self.hs_slopes[c]
            v = np.array(s["intercept"], float)
            for n in self.hs_names:
                v = v + s[n] * (th[self.idx[n]] - HS_NOMINALS[n])
            out[c] = v
        return out

    def _hist_by_cat(self, phi_cells, W):
        """hist_cat[b]: the response contraction, split by HS category."""
        per_cell = np.asarray(phi_cells, float) * W
        ew = self.cells.entry_w * self.norm * per_cell[self.cells.entry_cell]
        return {c: np.bincount(self.cells.entry_bin,
                               weights=np.where(self.entry_cat == i, ew, 0.0),
                               minlength=N_BINS)
                for i, c in enumerate(HS_CATEGORIES)}

    def set_hs_slopes(self, slopes, dm31):
        """Set the HS slopes of a lattice cell and record the Delta m^2 they came from."""
        self.hs_slopes = slopes
        self.hs_dm31 = float(dm31)

    def _check_hs_ready(self):
        """The slopes depend on Delta m^2 and must be set for the cell being fitted."""
        if self.hs_slopes is None and self.hs_names:
            raise RuntimeError(
                f"hs_slopes is None but the parameters include the HS dials "
                f"{self.hs_names}. They depend on Delta m^2 and must be set per lattice "
                f"cell with set_hs_slopes.")

    def expectation(self, phi_point, theta, return_parts=False):
        self._check_hs_ready()
        phi_cells = self.cells.phi_cells(phi_point)
        W, factors = self.cell_weights(theta)
        hist = self._hist_by_cat(phi_cells, W)
        C = self._hs_factor(theta)
        E = self.mu.copy()
        for c in HS_CATEGORIES:
            E = E + C[c] * hist[c]
        if return_parts:
            return E, {"phi_cells": phi_cells, "W": W, "factors": factors,
                       "hist": hist, "C": C}
        return E

    # ------------------------------------------------------------------- chi^2
    def chi2(self, phi_point, theta):
        E = self.expectation(phi_point, theta)
        n = E[self.few]
        if np.any(n <= 0):
            return BARRIER_CHI2
        return poisson_chi2(self.obs[self.few], n)

    def chi2_and_grad(self, phi_point, theta):
        """(chi^2, grad), statistical only; grad is in the order of the parameters."""
        E, parts = self.expectation(phi_point, theta, return_parts=True)
        n = E[self.few]
        if np.any(n <= 0):
            # barrier with a zero statistical gradient; the caller adds the prior gradient
            return BARRIER_CHI2, np.zeros(self.n_dials)
        chi2 = poisson_chi2(self.obs[self.few], n)
        resid = np.zeros(N_BINS)
        resid[self.few] = 2.0 * (1.0 - self.obs[self.few] / E[self.few])
        return chi2, self._adjoint(theta, parts, resid)

    def _adjoint(self, theta, parts, v):
        th = np.asarray(theta, float)
        grad = np.zeros(self.n_dials)
        W, C, hist = parts["W"], parts["C"], parts["hist"]
        per_cell = parts["phi_cells"] * W

        # cell dials: one weighted bincount over entries gives the per-cell u.
        # dE_b/dW_cell carries the cell's own category factor C_cat[b], so the per-entry
        # adjoint weight is v[b] * C_{cat(cell)}[b].
        vC = np.stack([v * C[c] for c in HS_CATEGORIES])          # (3, 200)
        w_entry = (self.cells.entry_w * self.norm
                   * per_cell[self.cells.entry_cell]
                   * vC[self.entry_cat, self.cells.entry_bin])
        u = np.bincount(self.cells.entry_cell, weights=w_entry,
                        minlength=self.cells.n_cell)

        for name in self.cell_dials:
            x = float(th[self.idx[name]])
            g = self.registry[name].dlnw_fn(self._geom_for(name), x,
                                            parts["factors"].get(name))
            grad[self.idx[name]] = (float(np.dot(u, g)) if np.ndim(g)
                                    else float(g) * float(u.sum()))

        # HS dials: additive, dE_b/dx_s = sum_cat slope_cat,s[b]*hist_cat[b]
        for n in self.hs_names:
            tot = 0.0
            for c in HS_CATEGORIES:
                tot += float(np.dot(v, self.hs_slopes[c][n] * hist[c]))
            grad[self.idx[n]] = tot

        # pinned slots exist and are literal zeros
        for p in self.pinned:
            grad[self.idx[p]] = 0.0
        return grad
