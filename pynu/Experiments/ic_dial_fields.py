"""IC-local per-cell dial fields — the 11 Mode-keyed xsec forms, plus the
hypersurface block's contract.

Companion to `binned_dial_fields.py`, which the ORCA track OWNS and this module
CONSUMES FROZEN (scope §4.3): the 19 flux + 4 shared xsec forms live there and are
gate-certified against the live tunes at 1e-12. This file adds only what is
IC-specific. Change requests go to the ORCA track; nothing here edits that module.

PROVENANCE. Every form is TRANSCRIBED from
`external/Pynu-upstream/pynu/PhysicsTunes/CrossSection/WaterXSection.py`, which
ships an analytic `diff_*` companion for each, so these are transcriptions of
verified code rather than new derivations. `--selftest` re-checks all 11 forward
factors and all 11 derivatives against the LIVE tune objects, cell by cell, at
1e-12 — runnable locally, since the tune modules import without nuflux.

TWO TRANSCRIPTION TRAPS, both live on IC and neither present on ORCA:

 1. `CCQENuBarNu` keys on the SIGNED mode (`experiment.Mode == -1`,
    WaterXSection.py:~150), NOT on |Mode|. The response class axis stores
    `absmode` (unsigned) because the sign is sign(pdg), already a class column
    (ic_binned_builder.py:139-140). So the cell predicate is
    `absmode == 1 AND pdg < 0`, and a naive |Mode|==1 transcription would apply
    the dial to neutrinos as well as antineutrinos.
 2. `DIS` is `|Mode| > 25 * CC`. CC is 1/0, so the threshold is 25 on CC events
    but **0 on NC events** — every NC cell with |Mode| > 0 is selected. Reading it
    as a flat ">25" would silently drop the entire NC population.

CLASS-AXIS SUFFICIENCY. All 11 are functions of (|Mode|, pdg, CC) and — for three
of them — of true energy through a THRESHOLD at 1.33 GeV or a smooth power/log of
E. The first group is exactly representable on the (pdg, current, |Mode|) class
axis of `ic_response_modeaxis_*.npz` (47 classes); the E-dependent group is
cell-exact only up to the cell-centering residue G-IC-4 measures, and the
threshold group additionally straddles unless the ladder is edge-snapped (B3).

mu FOR `xsec_ccqe_shape_subgev` IS THE FROZEN NON-INJECTED VALUE. Per design
ADDENDUM item 1, nothing in the repo assigns `ccqe_shape_subgev_e_edges`, so the
event path resolves mu from the WaterXSection class defaults (the SK production
grid), giving mu = -1.0164880658631577. The engine MUST reproduce that and must
NOT inject the IC ladder edges — injecting them would silently redefine the dial.
Gate G-C4 froze the value; `--selftest` re-asserts it to 1e-12.

    python ic_dial_fields.py --selftest        # local, no nuflux, no nuSQuIDS
"""
import argparse
import sys

import numpy as np

# Frozen by gate G-C4 (ic_stage1_gates.py); design ADDENDUM item 1.
CCQE_SHAPE_SUBGEV_MU = -1.0164880658631577
CCQE_SUBGEV_E = 1.33                      # WaterXSection._CCQE_SHAPE_SUBGEV_E

# Off-domain guard, transcribed from PhysicsTunes.Tune._unphysical_value
# (PhysicsTunes.py:366-381): return x < unphys_low or x > unphys_up.
UNPHYS_LOW, UNPHYS_UP = 0.0, 9999999.0
SUBGEV_UNPHYS_LOW = -9999999.0            # xsec_ccqe_shape_subgev passes this


def _unphys(x, low=UNPHYS_LOW, up=UNPHYS_UP):
    return bool(x < low or x > up)


class ICCellGeom:
    """Per-cell columns + the derived masks the 11 forms need.

    NOT a subclass of `binned_dial_fields.CellGeom`: that class uses `__slots__`,
    so IC-specific members cannot be attached to it (orca-impl's note). The two
    are used side by side — the shared registry gets the shared geometry, this
    one gets the IC extension.

    `mode` is the UNSIGNED |Mode| that the response class axis stores; the signed
    mode is reconstructed as sign(pdg) * |Mode| exactly as `ICDeepCore._NEUTMode`
    defines it (ICDeepCore.py:196-207).
    """

    def __init__(self, e_true, cz_true, pdg, cc, absmode):
        self.E = np.asarray(e_true, float)
        self.cz = np.asarray(cz_true, float)
        self.pdg = np.asarray(pdg, np.int64)
        self.cc = np.asarray(cc, np.int64)
        self.absmode = np.asarray(absmode, np.int64)
        self.n_cell = self.E.size

        a = np.abs(self.pdg)
        self.ln_E = np.log(self.E)
        self.m_ccqe = self.absmode == 1                      # _ccqe_mask
        self.m_nubar = self.pdg < 0
        # SIGNED Mode == -1  <=>  |Mode| == 1 AND pdg < 0   (trap 1)
        self.m_ccqe_nubar = self.m_ccqe & self.m_nubar
        self.m_ccqe_mu = self.m_ccqe & (a == 14)
        self.m_cc1pi = (self.absmode > 10) & (self.absmode < 17)
        self.m_cc1pi_nuebar = self.m_cc1pi & (self.pdg == -12)
        self.m_cc1pi_numubar = self.m_cc1pi & (self.pdg == -14)
        # |Mode| > 25*CC — threshold 25 on CC, 0 on NC              (trap 2)
        self.m_dis = self.absmode > 25 * self.cc
        self.m_multigev = self.E >= CCQE_SUBGEV_E
        self.m_ccqe_nue_hi = self.m_ccqe & (a == 12) & self.m_multigev
        self.m_ccqe_numu_hi = self.m_ccqe & (a == 14) & self.m_multigev
        # mean-zero sub-GeV log-E tilt; mu is the FROZEN non-injected constant
        self.sh_subgev = np.where(self.E < CCQE_SUBGEV_E,
                                  self.ln_E - CCQE_SHAPE_SUBGEV_MU, 0.0)


class ICDialField:
    """Forward factor and d(ln W)/dx for one IC-local dial, on the cell axis."""

    __slots__ = ("name", "algebra", "factor_fn", "dlnw_fn", "mask_fn", "source")

    def __init__(self, name, algebra, factor_fn, dlnw_fn, source, mask_fn=None):
        self.name, self.algebra = name, algebra
        self.factor_fn, self.dlnw_fn, self.mask_fn = factor_fn, dlnw_fn, mask_fn
        self.source = source


def _norm_field(name, mask_attr, source):
    """A pure class-mask normalization: w = 1 except x on the mask.

    algebra 'N'. These commute with binning EXACTLY when the mask is
    class-constant, which is why their binned-vs-event delta is bitwise zero —
    a result, not a defect (see the job 39685031 post-mortem).
    """
    def factor(g, x):
        if _unphys(x):
            return 1e-3                       # scalar collapse, verbatim
        w = np.ones(g.n_cell)
        w[getattr(g, mask_attr)] = x
        return w

    def dlnw(g, x, w=None):
        if _unphys(x) or x == 0:
            return 0.0
        d = np.zeros(g.n_cell)
        d[getattr(g, mask_attr)] = 1.0 / x    # diff/factor = 1/x on the mask
        return d

    return ICDialField(name, "N", factor, dlnw, source,
                       mask_fn=lambda g: getattr(g, mask_attr))


def _f_ccqe_shape(g, x):
    """w = E^(x-1) on CCQE, 1 elsewhere (WaterXSection.py:368-388)."""
    if _unphys(x):
        return 1e-3
    w = np.ones(g.n_cell)
    w[g.m_ccqe] = g.E[g.m_ccqe] ** (x - 1.0)
    return w


def _d_ccqe_shape(g, x, w=None):
    """diff = E^(x-1) ln E on CCQE, so diff/factor = ln E there, 0 elsewhere."""
    if _unphys(x):
        return 0.0
    d = np.zeros(g.n_cell)
    d[g.m_ccqe] = g.ln_E[g.m_ccqe]
    return d


def _f_ccqe_shape_subgev(g, x):
    """w = 1 + x*sh(E) on CCQE (WaterXSection.py:425-447). Note the widened
    lower off-domain bound — this dial is nominal-0 and legitimately negative."""
    if _unphys(x, low=SUBGEV_UNPHYS_LOW):
        return 1e-3
    w = np.ones(g.n_cell)
    w[g.m_ccqe] = 1.0 + x * g.sh_subgev[g.m_ccqe]
    return w


def _d_ccqe_shape_subgev(g, x, w=None):
    """diff = sh on CCQE, so diff/factor = sh / (1 + x*sh)."""
    if _unphys(x, low=SUBGEV_UNPHYS_LOW):
        return 0.0
    d = np.zeros(g.n_cell)
    m = g.m_ccqe
    d[m] = g.sh_subgev[m] / (1.0 + x * g.sh_subgev[m])
    return d


def _multigev(name, mask_attr, source):
    """w = x on (CCQE & flavour & E >= 1.33), 1 elsewhere. A class mask TIMES an
    energy threshold: exact on a snapped ladder, straddling otherwise (B3)."""
    return _norm_field(name, mask_attr, source)


FIELDS = {}
for _n, _m, _s in (
    ("DIS", "m_dis", "WaterXSection.py:97-103"),
    ("CCQE", "m_ccqe", "WaterXSection.py:~140"),
    ("CCQENuBarNu", "m_ccqe_nubar", "WaterXSection.py:~150 (SIGNED Mode == -1)"),
    ("CCQEMuE", "m_ccqe_mu", "WaterXSection.py:~160"),
    ("CC1PiProduction", "m_cc1pi", "WaterXSection.py:~172"),
    ("CC1Pi_NuBarNuE", "m_cc1pi_nuebar", "WaterXSection.py:~185"),
    ("CC1Pi_NuBarNuMu", "m_cc1pi_numubar", "WaterXSection.py:~200"),
):
    FIELDS[_n] = _norm_field(_n, _m, _s)

FIELDS["xsec_ccqe_shape"] = ICDialField(
    "xsec_ccqe_shape", "E", _f_ccqe_shape, _d_ccqe_shape,
    "WaterXSection.py:368-401")
FIELDS["xsec_ccqe_shape_subgev"] = ICDialField(
    "xsec_ccqe_shape_subgev", "E", _f_ccqe_shape_subgev, _d_ccqe_shape_subgev,
    "WaterXSection.py:403-447")
FIELDS["xsec_ccqe_multigev_nue"] = _multigev(
    "xsec_ccqe_multigev_nue", "m_ccqe_nue_hi", "WaterXSection.py:566-595 (pdg 12)")
FIELDS["xsec_ccqe_multigev_numu"] = _multigev(
    "xsec_ccqe_multigev_numu", "m_ccqe_numu_hi", "WaterXSection.py:566-595 (pdg 14)")

IC_ONLY_XSEC_11 = ["DIS", "CCQE", "CCQENuBarNu", "CCQEMuE", "CC1PiProduction",
                   "CC1Pi_NuBarNuE", "CC1Pi_NuBarNuMu", "xsec_ccqe_shape",
                   "xsec_ccqe_shape_subgev", "xsec_ccqe_multigev_nue",
                   "xsec_ccqe_multigev_numu"]
assert set(FIELDS) == set(IC_ONLY_XSEC_11), "IC-local field set != the 11 IC-only xsec dials"

# ---------------------------------------------------------------------------
# The 5 hypersurface dials are NOT cell fields
# ---------------------------------------------------------------------------
# dom_eff / hole_ice_p0 / hole_ice_p1 / bulk_ice_abs / bulk_ice_scatter are
# event-weight NO-OPS (ICDeepCoreDetector.py returns 1). Their entire effect is a
# HISTOGRAM-level, per-category, exactly LINEAR multiplier on the 200-bin
# expectation (ICDeepCore.apply_hs_correction:622-656):
#
#     C_cat = intercept_cat + SUM_s slope_cat,s * (theta_s - nominal_s)
#     n_bin = SUM_cat  C_cat * hist_cat(bin)                 + muon
#
# so d n_bin / d theta_s = SUM_cat slope_cat,s * hist_cat(bin) — ADDITIVE, with
# CONSTANT slopes at fixed Dm2 (design §2.4). It is NOT a d(ln W) form, and
# writing it as one would be wrong: C_cat can pass through zero, and the dial
# multiplies a category HISTOGRAM, not a per-event weight. The slopes are
# interpolated once per Dm2 cell (`interpolate_hs`) and then held fixed, which is
# what makes the gradient exact rather than finite-differenced.
HS_DIALS = ("dom_eff", "hole_ice_p0", "hole_ice_p1",
            "bulk_ice_abs", "bulk_ice_scatter")


def factor(name, geom, x, registry=None):
    reg = registry if registry is not None else FIELDS
    return reg[name].factor_fn(geom, x)


def dlnw(name, geom, x, w_cached=None, registry=None):
    reg = registry if registry is not None else FIELDS
    return reg[name].dlnw_fn(geom, x, w_cached)


# ---------------------------------------------------------------------------
# Local gate — all 11 forms + derivatives vs the LIVE tunes, cell by cell
# ---------------------------------------------------------------------------

def selftest(pynu_root, n=20000, tol=1e-12, seed=20260817):
    sys.path.insert(0, pynu_root)
    from pynu.PhysicsTunes.CrossSection.WaterXSection import WaterXSection

    print("=== IC dial fields — 11 Mode-keyed forms vs the LIVE tunes ===")
    xs = WaterXSection()

    # mu identity first: the engine must reproduce the NON-INJECTED value (G-C4).
    assert xs.ccqe_shape_subgev_e_edges is None, \
        "ccqe_shape_subgev_e_edges is injected — G-C4 / ADDENDUM item 1 violated"
    mu_live = xs._ccqe_shape_subgev_mu()
    d_mu = abs(mu_live - CCQE_SHAPE_SUBGEV_MU)
    print(f"GATE mu-identity: {'PASS' if d_mu <= 1e-12 else 'FAIL'} "
          f"mu={mu_live!r} vs frozen {CCQE_SHAPE_SUBGEV_MU!r} (|d|={d_mu:.3e})")
    ok = d_mu <= 1e-12

    rng = np.random.default_rng(seed)
    E = np.exp(rng.uniform(np.log(1.02344806), np.log(9292.0), n))
    cz = rng.uniform(-1.0, 1.0, n)
    pdg = rng.choice([12, -12, 14, -14, 16, -16], n)
    cc = rng.integers(0, 2, n)
    absmode = rng.choice([0, 1, 11, 26, 31], n)          # the realized |Mode| set
    g = ICCellGeom(E, cz, pdg, cc, absmode)

    class MockExp:
        pass
    m = MockExp()
    m.ETrue, m.CosZTrue, m.nuPDG = E, cz, pdg
    m.CC, m.NumberOfEvents = cc, n
    # the live tunes read the SIGNED mode; the response stores |Mode| and the sign
    # is sign(pdg) (ICDeepCore._NEUTMode) — reconstructing it here is exactly the
    # transformation the engine performs, so the gate tests that too.
    m.Mode = np.where(pdg < 0, -absmode, absmode)

    xs_draws = [0.7, 0.9, 1.0, 1.1, 1.4]
    worst_f, worst_d, worst_fn, worst_dn = 0.0, 0.0, "", ""
    for name in IC_ONLY_XSEC_11:
        live_f = getattr(xs, name)
        live_d = getattr(xs, "diff_" + name)
        draws = ([-0.8, -0.3, 0.0, 0.5, 1.2] if name == "xsec_ccqe_shape_subgev"
                 else xs_draws)
        for x in draws:
            fw = np.asarray(factor(name, g, x), float)
            lw = np.asarray(live_f(m, x), float)
            if lw.size == 1:                       # off-domain scalar collapse
                fe = abs(float(np.atleast_1d(fw)[0]) - float(lw))
            else:
                fe = float(np.max(np.abs(fw - lw)))
            # derivative: compare dlnw against the live diff/factor
            dw = dlnw(name, g, x)
            ld = np.asarray(live_d(m, x), float)
            if ld.size == 1 or lw.size == 1:
                de = abs(float(np.atleast_1d(dw)[0]) - 0.0)
            else:
                ref = np.where(lw != 0, ld / np.where(lw != 0, lw, 1.0), 0.0)
                de = float(np.max(np.abs(np.asarray(dw, float) - ref)))
            if fe > worst_f:
                worst_f, worst_fn = fe, f"{name}@{x}"
            if de > worst_d:
                worst_d, worst_dn = de, f"{name}@{x}"
    ok &= worst_f <= tol and worst_d <= tol
    print(f"GATE forward-factor: {'PASS' if worst_f <= tol else 'FAIL'} "
          f"worst |d| = {worst_f:.3e} at {worst_fn} (threshold {tol:g})")
    print(f"GATE dlnw-vs-diff/factor: {'PASS' if worst_d <= tol else 'FAIL'} "
          f"worst |d| = {worst_d:.3e} at {worst_dn} (threshold {tol:g})")

    # The two transcription traps, asserted explicitly rather than trusted.
    sgn_ok = np.array_equal(g.m_ccqe_nubar, m.Mode == -1)
    print(f"GATE trap1 signed-mode (CCQENuBarNu == Mode==-1): {sgn_ok}")
    dis_ok = np.array_equal(g.m_dis, np.abs(m.Mode) > 25 * m.CC)
    n_nc = int((g.m_dis & (cc == 0)).sum())
    print(f"GATE trap2 DIS mask (|Mode| > 25*CC, NC threshold 0): {dis_ok} "
          f"({n_nc} NC cells selected — a flat '>25' would drop all of them)")
    ok &= bool(sgn_ok and dis_ok and n_nc > 0)

    print(f"IC DIAL FIELDS: {'ALL PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pynu-root", default="external/Pynu-upstream")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.pynu_root) else 1)
    print(f"{len(FIELDS)} IC-local dial fields: {sorted(FIELDS)}")
