"""ORCABinnedEngine — the ORCA arm as a binned response-matrix engine with an
analytic dial gradient.

Spec of record: `SCOPE_orca_binned_track_2026-08-17.md` §2 (math, dial table,
discipline checklist), under `DESIGN_binned_icorca_gradients_2026-08-16.md` and
its 2026-08-17 ADDENDUM.

WHAT THIS REPLACES. Production computes the ORCA term by
`orca_exact_scan.binned_expectation`, which runs `StartNuisance()` +
`ApplyNuisanceWeights(theta)` over all 592,099 events and scatters the resulting
per-event weight into the 900 bins — once per finite-difference step, 28 times
per gradient. This engine evaluates the same dial algebra on the 17,236 POPULATED
true cells, contracts once, and gets the whole 30-long gradient from a single
adjoint pass.

WHY IT IS EXACT, NOT APPROXIMATE. Gate G-C2 measures exactly one distinct ETrue
and one distinct cos ZTrue per true bin over all 592,099 rows, so evaluating a
dial at the cell's true coordinate reproduces the per-event value to the same
float. Combined with cell-constant PhysicsWeight (G-ORCA-0, `orca_cell_phi`), the
port differs from production only by float summation order. G-ORCA-1 therefore
gates at 1e-9 relative, and A FAILURE THERE IS A BUG, NEVER "BINNING".

★★ THIS ARM RETURNS STAT-ONLY chi2 AND STAT-ONLY GRADIENT — NO PRIOR.
The worker owns the single union Gaussian (`combined_ic_orca_fit_worker.py:242-246`,
gradient at `combined_3exp_fit_worker.py:1154`). The SK engine's `chi2_and_grad`
includes its own internal Gaussian; copying that here would double-count the prior
and silently move every result, and it would NOT trip G-ORCA-1, which compares
against `combined_model_and_chi2`'s `chi2_orca` term alone. This is risk R2, and
`gates/gate_orca_grad.py --stat-only` is its dedicated test.

★★ E_shift IS PINNED AND HARD-ASSERTED. The precomputed R_b encodes
ENERGY_SCALE = 1, so a moved E_shift would be silently IGNORED rather than error
(risk R3). Its gradient slot is written as a literal 0.0.

DEVIATIONS FROM THE SCOPE'S PSEUDOCODE (both strictly reductions, no arithmetic
change; reported rather than silent):
  1. Scope §2.2 routes the forward pass through the class marginal n_pre[k,b]
     and sums over k. The adjoint gradient makes that marginal unnecessary — the
     class-N shortcut aggregates s_k on the CELL axis, not the bin axis — so the
     forward pass contracts straight to S_b (a 900-long bincount instead of a
     7,200-long one). `expectation(..., return_parts=True)` still builds the
     marginal via `binned_contract.contract_class` and asserts it sums to S_b.
  2. The two bin-axis gradients are computed as `sum_b resid_b * S_b` over the
     dial's 300-bin block rather than the scope's
     `(1/f) * sum_b resid_b * (E_b - mu_b)`. Algebraically identical
     (D_b S_b = E_b - mu_b and D_b = f on that block), one fewer division, and
     no cancellation against the muon block.
"""
import os

import numpy as np

from .orca_binned_support import (                     # noqa: E402
    poisson_chi2, N_BINS as N_BINS_ORCA, N_ERECO, N_CZRECO, N_PID,
)
from .binned_contract import contract, contract_class, adjoint_cells  # noqa: E402
from . import binned_dial_fields as bdf                               # noqa: E402
from .binned_dial_fields import DialField, build_cell_geometry        # noqa: E402

# ---------------------------------------------------------------------------
# Manifest — XML order of `ORCA_Atm_r2_fude_ccqe.xml`, which is the order
# `combined_ic_orca_fit_worker._project_theta` (:281-286) delivers theta in.
# 30 ACTIVE dials; `muon_norm` is <status> 0 </status> and absent from the list.
# ---------------------------------------------------------------------------
ORCA_MANIFEST = (
    list(bdf.SHARED_FLUX_19)                                  # 19 flux
    + list(bdf.SHARED_XSEC_4)                                 # 4 xsec
    + ["f_all", "f_HPT", "f_Shower", "f_tauCC", "f_NC", "f_HE", "E_shift"]   # 7 det
)
assert len(ORCA_MANIFEST) == 30

ORCA_PINNED = ("E_shift",)                 # combined_ic_orca_fit_worker.py:120
PINNED_UNION = ("nunubar_ratio",)          # combined_3exp_fit_worker.py:173
BIN_AXIS_DIALS = ("f_HPT", "f_Shower")     # keyed on b // 300, not on the class
MIGRATION_DIALS = ("E_shift",)             # reco migration; cannot be cell-constant

# 28 MOVABLE = 30 - E_shift - nunubar_ratio (scope §2.1).
MOVABLE = tuple(n for n in ORCA_MANIFEST
                if n not in ORCA_PINNED and n not in PINNED_UNION)
assert len(MOVABLE) == 28

# G-G2's zero-gradient set. FOUR dials, not the design's three: on ORCA the
# minimum true energy is 1.1220184543 GeV, so `normalization_below1GeV`'s
# E < 1 mask is EMPTY as well as the three sub-GeV bands (ADDENDUM item 7).
INERT_DIALS = ("flux_nuebar_subgev", "flux_flavor_subgev", "flux_numubar_subgev",
               "normalization_below1GeV")

BARRIER_CHI2 = 9e9                         # combined_3exp_fit_worker.py:848
BLOCK = N_ERECO * N_CZRECO                 # 300 — b // BLOCK == pid (gate G-C1)
SCHEMA_FLAT900 = "orca_binned_response_v2_flat900"


# ---------------------------------------------------------------------------
# ORCA-local detector dial fields (ORCADetector.py). Not in the shared registry:
# `binned_dial_fields` holds only the 19 flux + 4 xsec forms both manifests share.
# f_HPT / f_Shower are absent here — they are BIN-axis and handled in D_b.
# ---------------------------------------------------------------------------
def _orca_detector_fields(geom):
    """Build the 4 cell-axis ORCA detector fields as closures over cached masks."""
    he = ((geom.E > 500.0) & geom.m_cc) | ((geom.E > 100.0) & geom.m_nc)  # :160
    tau_cc = geom.m_abs16 & geom.m_cc                                     # :110
    nc = geom.m_nc                                                        # :135
    ones = np.ones(geom.n_cell)

    def f_all(g, x):                                          # ORCADetector.py:34-46
        return x * ones

    def d_all(g, x, w):                                       # ORCADetector.py:48-50
        return ones / x

    def _mask_field(mask):
        def f(g, x):
            w = np.ones(g.n_cell)
            w[mask] = x
            return w

        def d(g, x, w):
            return mask / x
        return f, d

    f_tau, d_tau = _mask_field(tau_cc)                        # ORCADetector.py:96-119
    f_nc, d_nc = _mask_field(nc)                              # ORCADetector.py:121-142
    f_he, d_he = _mask_field(he)                              # ORCADetector.py:144-169

    return {
        "f_all": DialField("f_all", "N", f_all, d_all, "ORCADetector.py:34-50",
                           mask_fn=lambda g: np.ones(g.n_cell, bool)),
        "f_tauCC": DialField("f_tauCC", "N", f_tau, d_tau, "ORCADetector.py:96-119",
                             mask_fn=lambda g: tau_cc),
        "f_NC": DialField("f_NC", "N", f_nc, d_nc, "ORCADetector.py:121-142",
                          mask_fn=lambda g: nc),
        # f_HE's mask keys on ETrue AND current, so it is NOT class-constant; the
        # engine measures that at load and routes it through the general dot.
        "f_HE": DialField("f_HE", "E", f_he, d_he, "ORCADetector.py:144-169",
                          mask_fn=lambda g: he),
    }


class ORCABinnedEngine:
    """Binned ORCA arm: stat-only pure-Poisson chi2 + analytic 30-long gradient.

    theta and grad are both in ORCA MANIFEST order (`nuisance_names` as passed).
    """

    def __init__(self, response_npz, obs900, mu900, few, nuisance_names, norm,
                 likelihood="poisson", osc=None):
        if likelihood != "poisson":
            raise ValueError(
                f"likelihood={likelihood!r}: the ORCA arm is PURE POISSON, not "
                "Barlow-Beeston (campaign convention since 2026-06-25; "
                "combined_ic_orca_fit_worker.py:274-275)")
        self.likelihood = likelihood

        # --- risk R1: averaging must be off, or PhysicsWeight is not cell-constant
        if osc is not None and getattr(osc, "osc_avg_scale", None) is not None:
            raise AssertionError(
                f"osc_avg_scale is {osc.osc_avg_scale!r}, must be None — with "
                "averaging ON the cell-phi gather silently becomes approximate "
                "(risk R1; production sets it to None at "
                "combined_ic_orca_fit_worker.py:365)")

        r = np.load(response_npz, allow_pickle=True) if isinstance(
            response_npz, (str, os.PathLike)) else response_npz
        self.response_path = str(response_npz)

        # --- risk R4: a stale v1 300-bin response must not load silently -------
        schema = str(np.asarray(r["schema_version"]).item()
                     if np.asarray(r["schema_version"]).ndim == 0
                     else np.asarray(r["schema_version"])[()])
        class_axis = [str(x) for x in np.asarray(r["class_axis"])]
        self.n_bins = int(r["n_bins"])
        if schema != SCHEMA_FLAT900:
            raise AssertionError(
                f"response schema {schema!r} != {SCHEMA_FLAT900!r} — this engine "
                "requires the flat900 layout (rebuild with "
                "orca_binned_builder.py --layout flat900)")
        if self.n_bins != N_BINS_ORCA:
            raise AssertionError(f"n_bins {self.n_bins} != {N_BINS_ORCA}")
        if class_axis != ["pdg", "current"]:
            raise AssertionError(f"class_axis {class_axis} != ['pdg', 'current']")

        self.Rk = np.asarray(r["R_k"], np.int64)
        self.Re = np.asarray(r["R_e"], np.int64)
        self.Rz = np.asarray(r["R_z"], np.int64)
        self.Rb = np.asarray(r["R_b"], np.int64)
        # NORM applied ONCE, at load: the response stores RAW parquet weight and
        # BaseWeight = Weight * NORM with NORM = FitExposure * 1e4 (Orca.py:180,183).
        self.norm = float(norm)
        self.Rvn = np.asarray(r["R_v"], float) * self.norm
        self.nnz = int(self.Rvn.size)

        self.classes = np.asarray(r["classes"], np.int64)      # (n_cls, 2) pdg,current
        self.n_cls = int(self.classes.shape[0])
        self.n_etrue = int(r["n_etrue"])
        self.n_cztrue = int(r["n_cztrue"])
        e_c = np.asarray(r["e_true_centers"], float)
        z_c = np.asarray(r["cz_true_centers"], float)

        # --- populated cell list (design §3.3): unique (class, ie, iz) ---------
        key = (self.Rk * self.n_etrue + self.Re) * self.n_cztrue + self.Rz
        uniq, self.cell_of_nnz = np.unique(key, return_inverse=True)
        self.cell_of_nnz = np.asarray(self.cell_of_nnz, np.int64).ravel()
        self.n_cell = int(uniq.size)
        c_k = (uniq // (self.n_etrue * self.n_cztrue)).astype(np.int64)
        rem = uniq % (self.n_etrue * self.n_cztrue)
        c_ie = (rem // self.n_cztrue).astype(np.int64)
        c_iz = (rem % self.n_cztrue).astype(np.int64)
        self.c_k, self.c_ie, self.c_iz = c_k, c_ie, c_iz
        c_pdg = self.classes[c_k, 0]
        c_cc = self.classes[c_k, 1]
        self.c_pdg, self.c_cc = c_pdg, c_cc
        # SK's phi index convention (sk_binned_engine.py:215-216), shared with
        # ic_divergence_scan.nu_index.
        self.c_ntype = (c_pdg < 0).astype(np.int64)
        self.c_flavor = (np.abs(c_pdg) // 2 - 6).astype(np.int64)

        self.geom = build_cell_geometry(e_c[c_ie], z_c[c_iz], c_pdg, c_cc)
        self.e_true_centers, self.cz_true_centers = e_c, z_c

        # --- dial registry: shared 23 + the 4 ORCA-local cell-axis detector ----
        self.registry = dict(bdf.FIELDS)
        self.registry.update(_orca_detector_fields(self.geom))

        self.names = list(nuisance_names)
        if set(self.names) != set(ORCA_MANIFEST):
            missing = sorted(set(ORCA_MANIFEST) - set(self.names))
            extra = sorted(set(self.names) - set(ORCA_MANIFEST))
            raise AssertionError(
                f"nuisance_names is not the ORCA manifest: missing {missing}, "
                f"extra {extra}")
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.n_dials = len(self.names)
        self.i_eshift = self.idx["E_shift"]
        self.i_hpt = self.idx["f_HPT"]
        self.i_shower = self.idx["f_Shower"]
        self.cell_dials = [n for n in self.names
                           if n not in BIN_AXIS_DIALS and n not in MIGRATION_DIALS]
        assert len(self.cell_dials) == 27, len(self.cell_dials)   # 28 movable - 2 + nunubar

        # --- observation / muon / few -----------------------------------------
        self.obs = np.asarray(obs900, float)
        self.mu = np.asarray(mu900, float)
        self.few = np.asarray(few, bool)
        for nm, a in (("obs900", self.obs), ("mu900", self.mu), ("few", self.few)):
            if a.shape != (self.n_bins,):
                raise AssertionError(f"{nm} shape {a.shape} != ({self.n_bins},)")
        self.n_few = int(self.few.sum())

        # --- bin-axis blocks: b // 300 == pid (gate G-C1 floor) ----------------
        b_block = np.arange(self.n_bins) // BLOCK
        assert int(b_block.max()) == N_PID - 1
        self.m_hpt_bins = b_block == 1                 # ORCADetector f_HPT, Sample==1
        self.m_shower_bins = b_block == 0              # f_Shower, Sample==0
        # pid == 2 (the third block) receives neither dial: D_b == 1 there.

        # --- class-N fast path: VALIDATED, not declared ------------------------
        # A dial is fast-pathed only if its cell mask is MEASURED constant within
        # every class on THIS response; otherwise it falls back to the general
        # 17k-length dot. (normalization_above1GeV's mask is E > 1, which is all
        # cells on ORCA, and normalization_below1GeV's is empty — both trivially
        # class-constant here, but that is a property of the MC, not of the form.)
        self.class_mask = {}
        first_of_class = np.zeros(self.n_cls, np.int64)
        for k in range(self.n_cls):
            first_of_class[k] = int(np.flatnonzero(c_k == k)[0])
        for name in self.cell_dials:
            fld = self.registry[name]
            if fld.algebra != "N" or fld.mask_fn is None:
                continue
            m = np.asarray(fld.mask_fn(self.geom), bool)
            km = m[first_of_class]
            if np.array_equal(m, km[c_k]):
                self.class_mask[name] = km
        self.fast_dials = sorted(self.class_mask)

        self._chk = {"schema": schema, "class_axis": class_axis}

    # ------------------------------------------------------------------ utils
    def _theta_checked(self, theta):
        th = np.asarray(theta, float).ravel()
        if th.size != self.n_dials:
            raise AssertionError(f"theta length {th.size} != {self.n_dials}")
        # risk R3: the precomputed R_b encodes ENERGY_SCALE = 1, so a moved
        # E_shift would be IGNORED rather than error. Refuse it loudly.
        if th[self.i_eshift] != 1.0:
            raise AssertionError(
                f"E_shift = {th[self.i_eshift]!r} but the binned response encodes "
                "ENERGY_SCALE = 1 exactly; a moved E_shift would be silently "
                "ignored. E_shift is ORCA_PINNED in production "
                "(combined_ic_orca_fit_worker.py:120).")
        return th

    def _phi_cells(self, phi):
        p = np.asarray(phi, float)
        want = (2, 3, self.n_etrue, self.n_cztrue)
        if p.shape != want:
            raise AssertionError(f"phi shape {p.shape} != {want}")
        return p[self.c_ntype, self.c_flavor, self.c_ie, self.c_iz]

    # ------------------------------------------------------------------- model
    def cell_weights(self, phi, theta, return_factors=False):
        """W_c = phi(cell) * prod over the 27 cell-axis dials of f_d(c).

        NO SK NC OVERRIDE. SK forces P = 1 on NC classes because SK's phi is bare
        survival probability; ORCA's phi is oscillated FLUX (InitialFlux x P_osc,
        `orca_exact_scan.binned_expectation`), so applying the SK override here
        would drop the flux on every NC cell (design §5.1 — the single most likely
        silent factor-of-flux bug in the whole port).
        """
        th = self._theta_checked(theta)
        W = self._phi_cells(phi).astype(float, copy=True)
        factors = {}
        for name in self.cell_dials:
            f = self.registry[name].factor_fn(self.geom, float(th[self.idx[name]]))
            factors[name] = f
            W = W * f
        return (W, factors) if return_factors else W

    def _detector_bin_factor(self, th):
        """D_b — the per-bin morphology factor, keyed on b // 300 == pid."""
        D = np.ones(self.n_bins)
        D[self.m_hpt_bins] = th[self.i_hpt]
        D[self.m_shower_bins] = th[self.i_shower]
        return D

    def expectation(self, phi, theta, return_parts=False, class_marginal=False):
        """E_b = D_b * S_b + mu_b, the 900-long model rate.

        `class_marginal=True` additionally builds n_pre[k, b] via
        `binned_contract.contract_class` and asserts it sums over k to S_b (the
        class axis is a disjoint partition). That is a diagnostic, kept OUT of
        the hot path: the adjoint gradient needs only S_b.
        """
        th = self._theta_checked(theta)
        W, factors = self.cell_weights(phi, th, return_factors=True)
        Wflat = W[self.cell_of_nnz]
        S = contract(self.Rb, self.Rvn, Wflat, self.n_bins)
        D = self._detector_bin_factor(th)
        E = D * S + self.mu
        if not (return_parts or class_marginal):
            return E
        parts = {"S": S, "D": D, "W": W, "factors": factors, "mu": self.mu}
        if class_marginal:
            n_pre = contract_class(self.Rk, self.Rb, self.Rvn, Wflat,
                                   self.n_cls, self.n_bins)
            assert np.allclose(n_pre.sum(axis=0), S, rtol=1e-12,
                               atol=1e-9 * max(1.0, float(np.abs(S).max())))
            parts["n_pre"] = n_pre
        return E, parts

    # --------------------------------------------------------------------- chi2
    def chi2(self, phi, theta):
        """STAT-ONLY pure-Poisson chi2 on the `few` mask. NO PRIOR (see module doc)."""
        E = self.expectation(phi, theta)
        n_mod = E[self.few]
        if np.any(n_mod <= 0):                 # barrier BEFORE any log
            return BARRIER_CHI2
        return float(poisson_chi2(self.obs[self.few], n_mod))

    def chi2_and_grad(self, phi, theta):
        """(chi2, grad) — both STAT-ONLY; grad is 30-long in manifest order.

        The gradient is first-order in the SK sense: `resid` and `D` are held
        fixed while the cell-axis dials are differentiated
        (`sk_binned_engine.py:2508, :2520-2522`).
        """
        th = self._theta_checked(theta)
        E, parts = self.expectation(phi, th, return_parts=True)
        n_mod = E[self.few]
        if np.any(n_mod <= 0):
            # Barrier: return the barrier value with a ZERO stat-gradient; the
            # worker adds the prior gradient and its BARRIER_MAX_RUN guard takes
            # over (combined_3exp_fit_worker.py:848-849, 1216-1228).
            return BARRIER_CHI2, np.zeros(self.n_dials)

        chi2 = float(poisson_chi2(self.obs[self.few], n_mod))

        resid = np.zeros(self.n_bins)
        resid[self.few] = 2.0 * (1.0 - self.obs[self.few] / E[self.few])
        grad = self._adjoint(th, parts, resid)
        return chi2, grad

    # ----------------------------------------------------------------- adjoint
    def model_jacobian_dot(self, phi, theta, v):
        """[sum_b v_b * dE_b/dx_d]_d — the vector-Jacobian product, 30-long.

        `chi2_and_grad` is exactly this with v = resid. Exposed because it is the
        clean way to state the muon invariant: dE_b/dx carries no muon term, so
        this result is BITWISE independent of mu900 (gate G-G2c) even though the
        chi2 and its residual are not.
        """
        th = self._theta_checked(theta)
        _E, parts = self.expectation(phi, th, return_parts=True)
        return self._adjoint(th, parts, np.asarray(v, float))

    def _adjoint(self, th, parts, v):
        W, D, S = parts["W"], parts["D"], parts["S"]
        factors = parts["factors"]

        # ONE bincount, O(nnz) — the whole reason each cell dial is then a dot.
        u = adjoint_cells(self.cell_of_nnz, self.Rb, self.Rvn, v * D, self.n_cell)
        uW = u * W

        grad = np.zeros(self.n_dials)

        # class-N dials: 8 scalars, computed once, make these effectively free.
        s_k = np.bincount(self.c_k, weights=uW, minlength=self.n_cls)

        for name in self.cell_dials:
            x = float(th[self.idx[name]])
            km = self.class_mask.get(name)
            if km is not None:
                # sum over an empty class set is 0.0 exactly -> the inert dials
                # (normalization_below1GeV) get a BITWISE zero gradient.
                grad[self.idx[name]] = float(s_k[km].sum()) / x
            else:
                g = self.registry[name].dlnw_fn(self.geom, x, factors[name])
                grad[self.idx[name]] = float(np.dot(uW, g)) if np.ndim(g) else \
                    float(g) * float(uW.sum())

        # bin-axis dials: dE_b/df = S_b on the dial's own 300-bin block, 0 elsewhere.
        # (== the scope's (1/f) * sum resid_b * (E_b - mu_b), one division fewer.)
        vS = v * S
        grad[self.i_hpt] = float(vS[self.m_hpt_bins].sum())
        grad[self.i_shower] = float(vS[self.m_shower_bins].sum())

        # mu900 enters E_b additively with no theta dependence -> zero gradient.
        # E_shift is pinned and its slot is a literal 0.0.
        grad[self.i_eshift] = 0.0
        return grad

    # -------------------------------------------------------------------- info
    def summary(self):
        return {
            "response": self.response_path, "schema": self._chk["schema"],
            "nnz": self.nnz, "n_cell": self.n_cell, "n_cls": self.n_cls,
            "n_bins": self.n_bins, "n_few": self.n_few, "norm": self.norm,
            "n_dials": self.n_dials, "n_cell_dials": len(self.cell_dials),
            "fast_class_N_dials": self.fast_dials,
            "sum_Rvn": float(self.Rvn.sum()),
        }
