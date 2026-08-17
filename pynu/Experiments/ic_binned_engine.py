"""`ICBinnedEngine` — stat-only χ² and analytic gradient for the IC DeepCore arm.

Sibling of `ORCABinnedEngine`, same contract, built on the two certified bricks:
`ic_cells.ICCells` (sparse populated-cell structure + contraction) and
`ic_dial_fields` (the 11 Mode-keyed forms), plus `binned_dial_fields` consumed
FROZEN from the ORCA track for the 19 flux + 4 shared xsec forms.

WHAT IT REPLACES. Production differentiates 38 IC dials by finite difference, and
stage 0 measured the FD loop at 88.2% of a cold task. This evaluates every dial
ONCE PER CELL (127,757 at L3) instead of once per event (396,843), and gets the
whole gradient from ONE adjoint bincount.

THE MODEL, and where each piece is pinned:

    E_b = SUM_cat  C_cat[b] * hist_cat[b]  +  mu_b

    hist_cat[b] = SUM_{entries whose cell is in cat}  entry_w * NORM * phi_cell * W_cell
    W_cell      = PROD_{cell dials} factor(dial, geom_cell, x)
    C_cat[b]    = intercept_cat[b] + SUM_s slope_cat,s[b] * (x_s - nominal_s)

  * pure Poisson, STAT-ONLY, no prior — the worker owns the single union prior
    (`combined_ic_orca_fit_worker.py:242-246`); an arm-internal Gaussian would
    double-count silently AND still pass an identity gate (ADDENDUM item 6).
  * `few` mask = obs > MIN_ENTRIES = 0.01 (`ICDeepCore.py:36`).
  * NORM applied at scan time; the response stores RAW weight
    (`ICDeepCore.py:190-191`).
  * muon: 200-bin constant, added AFTER the HS correction, ZERO gradient
    (`ic_divergence_scan.py:265`).
  * phi is oscillated FLUX (flux x P), NOT bare probability, and there is NO SK
    NC override — G-IC-3 is the detector for a violation.
  * `ccqe_shape_subgev_e_edges` stays None, so mu = -1.0164880658631577 (G-C4 /
    ADDENDUM item 1). Injecting the IC ladder edges would redefine the dial.

THE HS BLOCK IS ADDITIVE, NOT MULTIPLICATIVE-IN-LOG. The 5 hypersurface dials are
event-weight no-ops; their whole effect is the per-category linear multiplier
above, so

    dE_b/dx_s = SUM_cat slope_cat,s[b] * hist_cat[b]

with slopes CONSTANT at fixed Dm2 (`interpolate_hs` once per cell, design §2.4).
Writing them as a d(ln W) form would be wrong twice: C_cat can pass through zero,
and they multiply a category HISTOGRAM rather than a per-event weight. The HS
categories are pure class masks (`ICDeepCore.py:600-604`) so they partition the
CELL axis exactly.

CACHE NOTE. This engine never calls `Tune.Get`, so the coordinate-blind
`@cache_method` on the dispatcher (`PhysicsTunes.py:341-342`) that contaminated
the two-arm scan cannot affect it — by construction, not by discipline.

Local-testable: no pynu import. The HS slopes are passed in as data (the caller
gets them from `exp.interpolate_hs(dm31)`), so gates run without nuflux.

================================ INTEGRATION CONTRACT ========================
For the stage-5 worker binding. Certified by G-IC-3 at 4.236e-15 (L3) /
4.688e-15 (L1) relative vs the production IC term — summation-order identity.

    eng = ICBinnedEngine(
        response_npz  = ".../ic_response_modeaxis_L3.npz",   # see (1)
        obs200        = ic_divergence_scan.observed_200(data, data_dir),
        mu200         = ic_divergence_scan.muon_200(mc, data_dir),
        nuisance_names= list(fit.Analysis.NuisanceList),      # 39, manifest order
        norm          = FitExposure * SECONDS_PER_YEAR,       # see (3)
        hs_slopes     = exp.interpolate_hs(dm31_cell),        # see (2) — PER CELL
        pinned        = ("nunubar_ratio",))                   # see (4)

    chi2            = eng.chi2(phi[ipt], theta)
    chi2, grad39    = eng.chi2_and_grad(phi[ipt], theta)      # STAT-ONLY

FIVE THINGS THAT WILL BITE IF SKIPPED:

(1) RESPONSE MUST BE THE MODE-AXIS, NO-SNAP BUILD (`ic_response_modeaxis_L*`).
    The |Mode| class axis (47 classes) is required — without it the 11 Mode-keyed
    dials are unrepresentable, and the constructor REFUSES a 12-class response
    rather than producing quiet nonsense. It must ALSO stay unsnapped (nE 160) so
    the existing phi tensors index it; `phi_cells` hard-fails on a shape mismatch.

(2) ★ hs_slopes IS Delta-m^2 DEPENDENT AND MUST BE RESET PER CELL, from THAT
    cell's true grid Dm2 — `eng.hs_slopes = exp.interpolate_hs(dm31_cell)`.
    Reusing one cell's slopes across a patch, or interpolating at a Dm2 that came
    from anywhere other than the grid, silently fits against the wrong hypersurface
    while still converging. This is exactly the defect that invalidated the first
    G-IC-4 postfit run (labels from an independent centre+step arithmetic fed
    `interpolate_hs`); the measured footprint was a per-bin |dC|/C up to 6.2e-3 for
    a one-grid-step Dm2 error. Derive dm31_cell and the phi row `ipt` from the SAME
    (i_dm, i_s23) — `ipt = i_dm*grid_size + i_s23`.

(3) NORM: the response stores RAW weight (ICDeepCore.py:190-191), so NORM is
    applied here. Recover it from the LIVE experiment rather than recomputing, so
    a config change cannot desync the two sides.

(4) STAT-ONLY, and the gradient is len(nuisance_names) in MANIFEST ORDER with the
    pinned slot PRESENT and written 0.0 — no reindexing at the call site. The
    worker keeps sole ownership of the single union prior; adding an arm-internal
    Gaussian would double-count silently AND still pass an identity gate.

(5) phi is indexed by INTEGER cell indices only, never float-edge matching — a
    cluster rebuild reproduces e_true_edges to allclose (9.1e-13) but not bitwise.
    `assert_phi_grid(phi_npz)` is available for an explicit allclose check.

Change requests to this module rather than edits, mirroring how this track
consumes `binned_dial_fields` from the ORCA track.
=============================================================================
"""
import numpy as np

from .ic_binned_cells import ICCells
from . import ic_dial_fields as icf
from . import binned_dial_fields as bdf

BARRIER_CHI2 = 9e9                             # combined_3exp_fit_worker.py:848
MIN_ENTRIES = 0.01                             # ICDeepCore.py:36
N_BINS = 200
HS_DIALS = list(icf.HS_DIALS)
HS_CATEGORIES = ("nc_nue_cc", "numu_cc", "nutau_cc")   # ICDeepCore.py:486-490
HS_NOMINALS = {"dom_eff": 1.0, "hole_ice_p0": 0.1, "hole_ice_p1": -0.05,
               "bulk_ice_abs": 1.0, "bulk_ice_scatter": 1.0}   # :475-482
# Pinned in the union vector; the slot exists and is written 0.0 (worker pins it).
PINNED = ("nunubar_ratio",)


def poisson_chi2(obs, n_mod):
    lt = np.log(np.divide(obs, n_mod, out=np.ones_like(obs), where=n_mod > 0))
    lt[obs == 0] = 0
    return float(2 * np.sum(n_mod - obs + obs * lt))


class ICBinnedEngine:
    """Stat-only χ² + analytic gradient on the IC 200-bin reco axis.

    Args
      response_npz : ic_response_modeaxis_L*.npz (MUST carry the |Mode| axis)
      obs200, mu200: observed and static-muon 200-bin vectors
      nuisance_names: the arm's manifest order (39 long); the gradient is
                      returned in exactly this order, PINNED slots written 0.0
      norm         : FitExposure * SECONDS_PER_YEAR
      hs_slopes    : {cat: {"intercept": (200,), <dial>: (200,), ...}} from
                     `exp.interpolate_hs(dm31)` — CONSTANT for this Dm2 cell
      pinned       : names whose gradient slot is forced to 0.0
    """

    def __init__(self, response_npz, obs200, mu200, nuisance_names, norm,
                 hs_slopes=None, few=None, pinned=PINNED):
        self.cells = ICCells(response_npz)
        self.response_path = response_npz
        self.obs = np.asarray(obs200, float)
        self.mu = np.asarray(mu200, float)
        self.norm = float(norm)
        self.names = list(nuisance_names)
        self.n_dials = len(self.names)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.pinned = tuple(p for p in pinned if p in self.idx)
        self.few = (self.obs > MIN_ENTRIES) if few is None else np.asarray(few, bool)

        g = self.cells.geom
        # shared registry (FROZEN, ORCA-owned) + the IC-local 11, in one lookup
        self.shared_geom = bdf.build_cell_geometry(g.E, g.cz, g.pdg, g.cc)
        self.registry = dict(bdf.FIELDS)
        self.registry.update(icf.FIELDS)

        # cell dials = every manifest name with a field, minus HS and pinned
        self.cell_dials = [n for n in self.names
                           if n in self.registry and n not in HS_DIALS
                           and n not in self.pinned]
        missing = [n for n in self.names
                   if n not in self.registry and n not in HS_DIALS
                   and n not in self.pinned]
        if missing:
            raise SystemExit(f"no cell field for manifest dials: {missing}")

        # HS category partition of the CELL axis (ICDeepCore.py:600-604) — pure
        # class masks, so they are exact on cells.
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

        self.hs_slopes = hs_slopes
        self.hs_dm31 = None
        self.hs_names = [n for n in self.names if n in HS_DIALS]

    # ------------------------------------------------------------------ model
    def _geom_for(self, name):
        """IC-local fields read ICCellGeom; shared fields read the shared CellGeom."""
        return self.cells.geom if name in icf.FIELDS else self.shared_geom

    def cell_weights(self, theta, return_factors=False):
        """W_cell = product of every cell dial's factor. Scalars (the off-domain
        collapse) multiply through as scalars, exactly as the tunes intend."""
        th = np.asarray(theta, float)
        W = np.ones(self.cells.n_cell)
        factors = {}
        for name in self.cell_dials:
            x = float(th[self.idx[name]])
            f = self.registry[name].factor_fn(self._geom_for(name), x)
            if return_factors:
                factors[name] = f
            W = W * f
        return (W, factors) if return_factors else W

    def _hs_factor(self, theta):
        """C_cat[b] per category — intercept + sum slope*(x - nominal)."""
        if self.hs_slopes is None:
            return {c: np.ones(N_BINS) for c in HS_CATEGORIES}
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
        """hist_cat[b] — the response contraction, split by HS category."""
        per_cell = np.asarray(phi_cells, float) * W
        ew = self.cells.entry_w * self.norm * per_cell[self.cells.entry_cell]
        return {c: np.bincount(self.cells.entry_bin,
                               weights=np.where(self.entry_cat == i, ew, 0.0),
                               minlength=N_BINS)
                for i, c in enumerate(HS_CATEGORIES)}

    def set_hs_slopes(self, slopes, dm31=None):
        """Set the per-cell HS slopes, optionally recording the Dm2 they came
        from so a caller/gate can assert the engine is on the cell it thinks.

        The engine cannot check this itself — it never sees the grid — so the
        recorded value is the hook that makes a stale-slopes bug detectable
        instead of silent."""
        self.hs_slopes = slopes
        self.hs_dm31 = None if dm31 is None else float(dm31)

    def _check_hs_ready(self):
        """HS slopes absent while HS dials are in the manifest is a SILENT wrong
        answer twice over — C_cat falls back to 1 (wrong expectation, the real
        intercept is ~1.008) and every HS dial gets a zero gradient, so the fit
        never moves them and still converges. Refuse rather than compute."""
        if self.hs_slopes is None and self.hs_names:
            raise RuntimeError(
                f"hs_slopes is None but the manifest carries HS dials "
                f"{self.hs_names}. They are Dm2-DEPENDENT and must be set PER "
                f"CELL: eng.set_hs_slopes(exp.interpolate_hs(dm31_cell), dm31_cell). "
                f"Leaving them None would silently use C_cat = 1 AND return a zero "
                f"gradient for every HS dial.")

    def expectation(self, phi_point, theta, return_parts=False):
        self._check_hs_ready()
        phi_cells = self.cells.phi_cells(phi_point)
        W, factors = self.cell_weights(theta, return_factors=True)
        hist = self._hist_by_cat(phi_cells, W)
        C = self._hs_factor(theta)
        E = self.mu.copy()
        for c in HS_CATEGORIES:
            E = E + C[c] * hist[c]
        if return_parts:
            return E, {"phi_cells": phi_cells, "W": W, "factors": factors,
                       "hist": hist, "C": C}
        return E

    # -------------------------------------------------------------------- χ²
    def chi2(self, phi_point, theta):
        E = self.expectation(phi_point, theta)
        n = E[self.few]
        if np.any(n <= 0):
            return BARRIER_CHI2
        return poisson_chi2(self.obs[self.few], n)

    def chi2_and_grad(self, phi_point, theta):
        """(χ², grad) — STAT-ONLY, grad is len(names) in manifest order."""
        E, parts = self.expectation(phi_point, theta, return_parts=True)
        n = E[self.few]
        if np.any(n <= 0):
            # barrier with a ZERO stat-gradient; the worker's prior gradient and
            # BARRIER_MAX_RUN guard take over.
            return BARRIER_CHI2, np.zeros(self.n_dials)
        chi2 = poisson_chi2(self.obs[self.few], n)
        resid = np.zeros(N_BINS)
        resid[self.few] = 2.0 * (1.0 - self.obs[self.few] / E[self.few])
        return chi2, self._adjoint(theta, parts, resid)

    def model_jacobian_dot(self, phi_point, theta, v):
        """[sum_b v_b dE_b/dx_d]_d. chi2_and_grad is this with v = resid; exposed
        because it states the muon invariant cleanly — mu enters E additively with
        no theta dependence, so this is bitwise independent of it (G-G2)."""
        _E, parts = self.expectation(phi_point, theta, return_parts=True)
        return self._adjoint(theta, parts, np.asarray(v, float))

    def _adjoint(self, theta, parts, v):
        th = np.asarray(theta, float)
        grad = np.zeros(self.n_dials)
        W, C, hist = parts["W"], parts["C"], parts["hist"]
        per_cell = parts["phi_cells"] * W

        # ---- cell dials: ONE weighted bincount over entries -> per-cell u ----
        # dE_b/dW_cell carries the cell's OWN category factor C_cat[b], so the
        # per-entry adjoint weight is v[b] * C_{cat(cell)}[b].
        vC = np.stack([v * C[c] for c in HS_CATEGORIES])          # (3, 200)
        w_entry = (self.cells.entry_w * self.norm
                   * per_cell[self.cells.entry_cell]
                   * vC[self.entry_cat, self.cells.entry_bin])
        u = np.bincount(self.cells.entry_cell, weights=w_entry,
                        minlength=self.cells.n_cell)              # u_cell = uW

        for name in self.cell_dials:
            x = float(th[self.idx[name]])
            g = self.registry[name].dlnw_fn(self._geom_for(name), x,
                                            parts["factors"].get(name))
            grad[self.idx[name]] = (float(np.dot(u, g)) if np.ndim(g)
                                    else float(g) * float(u.sum()))

        # ---- HS dials: ADDITIVE, dE_b/dx_s = sum_cat slope_cat,s[b]*hist_cat[b]
        if self.hs_slopes is not None:
            for n in self.hs_names:
                tot = 0.0
                for c in HS_CATEGORIES:
                    tot += float(np.dot(v, self.hs_slopes[c][n] * hist[c]))
                grad[self.idx[n]] = tot

        # ---- pinned slots exist and are literal zeros -------------------------
        for p in self.pinned:
            grad[self.idx[p]] = 0.0
        return grad

    # ------------------------------------------------------------------- info
    def summary(self):
        return {"response": self.response_path, "grid": self.cells.grid_label,
                "n_cell": self.cells.n_cell, "n_entry": self.cells.n_entry,
                "n_dials": self.n_dials, "n_cell_dials": len(self.cell_dials),
                "hs_dials": self.hs_names, "pinned": list(self.pinned),
                "few_bins": int(self.few.sum()), "norm": self.norm,
                "mu_total": float(self.mu.sum()),
                "hs_slopes_set": self.hs_slopes is not None,
                "hs_dm31": self.hs_dm31,
                "ccqe_shape_subgev_mu": icf.CCQE_SHAPE_SUBGEV_MU}
