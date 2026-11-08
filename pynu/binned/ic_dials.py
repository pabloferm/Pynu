"""IC per-cell dial fields: the Mode-keyed cross-section forms, on top of the flux and
shared cross-section forms in ``dial_fields.py``.

Two details of the masks:

 1. `CCQENuBarNu` keys on the signed mode (Mode == -1). The response class axis stores
    the unsigned |Mode|, the sign being sign(pdg), so the cell predicate is
    `absmode == 1 AND pdg < 0`.
 2. `DIS` is `|Mode| > 25 * CC`. CC is 1/0, so the threshold is 25 on CC events but 0 on
    NC events: every NC cell with |Mode| > 0 is selected.
"""

import numpy as np

# Centring constant of the sub-GeV CCQE shape (the mean of ln E over SK's sub-GeV cells).
CCQE_SHAPE_SUBGEV_MU = -1.0164880658631577
CCQE_SUBGEV_E = 1.33                      # GeV, SK sub-GeV/multi-GeV boundary

# xsec_ccqe_shape_multigev, with the SK engine's form:
#
#     sh_mg(E) = 1[E >= 1.33] * ( ln E  -  <ln E>_{grid cells with E >= 1.33} )
#     w        = 1 + x * sh_mg(E)        on the CCQE class, 1 elsewhere
#     dlnW/dx  = sh_mg / (1 + x * sh_mg) on the CCQE class, 0 elsewhere
#
# The mean runs over this response's own true-energy grid, so the tilt is mean-zero over
# the multi-GeV cells here. x = 0 is a no-op.
CCQE_SHAPE_MULTIGEV_NAME = "xsec_ccqe_shape_multigev"

# Off-domain guard: x < low or x > up.
UNPHYS_LOW, UNPHYS_UP = 0.0, 9999999.0
SUBGEV_UNPHYS_LOW = -9999999.0            # for the nominal-0 CCQE shapes


def _unphys(x, low=UNPHYS_LOW, up=UNPHYS_UP):
    return bool(x < low or x > up)


class ICCellGeom:
    """Per-cell columns and the derived masks of the IC forms.

    `absmode` is the unsigned |Mode| that the response class axis stores; the signed mode
    is sign(pdg) * |Mode|.
    """

    def __init__(self, e_true, cz_true, pdg, cc, absmode, e_grid):
        self.E = np.asarray(e_true, float)
        self.cz = np.asarray(cz_true, float)
        self.pdg = np.asarray(pdg, np.int64)
        self.cc = np.asarray(cc, np.int64)
        self.absmode = np.asarray(absmode, np.int64)
        self.n_cell = self.E.size

        a = np.abs(self.pdg)
        self.ln_E = np.log(self.E)
        self.m_ccqe = self.absmode == 1
        self.m_nubar = self.pdg < 0
        # signed Mode == -1  <=>  |Mode| == 1 AND pdg < 0
        self.m_ccqe_nubar = self.m_ccqe & self.m_nubar
        self.m_ccqe_mu = self.m_ccqe & (a == 14)
        self.m_cc1pi = (self.absmode > 10) & (self.absmode < 17)
        self.m_cc1pi_nuebar = self.m_cc1pi & (self.pdg == -12)
        self.m_cc1pi_numubar = self.m_cc1pi & (self.pdg == -14)
        # |Mode| > 25*CC: threshold 25 on CC, 0 on NC
        self.m_dis = self.absmode > 25 * self.cc
        self.m_multigev = self.E >= CCQE_SUBGEV_E
        self.m_ccqe_nue_hi = self.m_ccqe & (a == 12) & self.m_multigev
        self.m_ccqe_numu_hi = self.m_ccqe & (a == 14) & self.m_multigev
        # sub-GeV log-E tilt with the fixed centring constant
        self.sh_subgev = np.where(self.E < CCQE_SUBGEV_E,
                                  self.ln_E - CCQE_SHAPE_SUBGEV_MU, 0.0)
        # multi-GeV log-E tilt; the mean runs over the true-energy grid, one entry per
        # energy, not over the cell list, which repeats energies
        grid = np.asarray(e_grid, float)
        _mg = grid >= CCQE_SUBGEV_E
        self.ccqe_shape_multigev_mu = (
            float(np.log(grid[_mg]).mean()) if _mg.any() else 0.0)
        self.sh_multigev = np.where(self.E >= CCQE_SUBGEV_E,
                                    self.ln_E - self.ccqe_shape_multigev_mu, 0.0)


class ICDialField:
    """Forward factor and d(ln W)/dx for one IC dial, on the cell axis."""

    __slots__ = ("name", "factor_fn", "dlnw_fn")

    def __init__(self, name, factor_fn, dlnw_fn):
        self.name = name
        self.factor_fn, self.dlnw_fn = factor_fn, dlnw_fn


def _norm_field(name, mask_attr):
    """A pure class-mask normalization: w = 1 except x on the mask."""
    def factor(g, x):
        if _unphys(x):
            return 1e-3                       # scalar collapse
        w = np.ones(g.n_cell)
        w[getattr(g, mask_attr)] = x
        return w

    def dlnw(g, x, w=None):
        if _unphys(x) or x == 0:
            return 0.0
        d = np.zeros(g.n_cell)
        d[getattr(g, mask_attr)] = 1.0 / x    # diff/factor = 1/x on the mask
        return d

    return ICDialField(name, factor, dlnw)


def _f_ccqe_shape(g, x):
    """w = E^(x-1) on CCQE, 1 elsewhere."""
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
    """w = 1 + x*sh(E) on CCQE. The lower off-domain bound is widened because this dial
    is nominal-0 and legitimately negative."""
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


def _f_ccqe_shape_multigev(g, x):
    """w = 1 + x*sh_mg(E) on CCQE, with the same widened lower off-domain bound."""
    if _unphys(x, low=SUBGEV_UNPHYS_LOW):
        return 1e-3
    w = np.ones(g.n_cell)
    w[g.m_ccqe] = 1.0 + x * g.sh_multigev[g.m_ccqe]
    return w


def _d_ccqe_shape_multigev(g, x, w=None):
    """diff/factor = sh_mg / (1 + x*sh_mg) on CCQE, 0 elsewhere."""
    if _unphys(x, low=SUBGEV_UNPHYS_LOW):
        return 0.0
    d = np.zeros(g.n_cell)
    m = g.m_ccqe
    sh = g.sh_multigev[m]
    fac = 1.0 + x * sh
    d[m] = np.divide(sh, fac, out=np.zeros_like(sh), where=fac != 0)
    return d


FIELDS = {}
for _n, _m in (
    ("DIS", "m_dis"),
    ("CCQE", "m_ccqe"),
    ("CCQENuBarNu", "m_ccqe_nubar"),
    ("CCQEMuE", "m_ccqe_mu"),
    ("CC1PiProduction", "m_cc1pi"),
    ("CC1Pi_NuBarNuE", "m_cc1pi_nuebar"),
    ("CC1Pi_NuBarNuMu", "m_cc1pi_numubar"),
    ("xsec_ccqe_multigev_nue", "m_ccqe_nue_hi"),
    ("xsec_ccqe_multigev_numu", "m_ccqe_numu_hi"),
):
    FIELDS[_n] = _norm_field(_n, _m)

FIELDS["xsec_ccqe_shape"] = ICDialField(
    "xsec_ccqe_shape", _f_ccqe_shape, _d_ccqe_shape)
FIELDS["xsec_ccqe_shape_subgev"] = ICDialField(
    "xsec_ccqe_shape_subgev", _f_ccqe_shape_subgev, _d_ccqe_shape_subgev)
FIELDS[CCQE_SHAPE_MULTIGEV_NAME] = ICDialField(
    CCQE_SHAPE_MULTIGEV_NAME, _f_ccqe_shape_multigev, _d_ccqe_shape_multigev)

# The 5 hypersurface dials are not cell fields. Their whole effect is a histogram-level,
# per-category, exactly linear multiplier on the 200-bin expectation:
#
#     C_cat = intercept_cat + SUM_s slope_cat,s * (theta_s - nominal_s)
#     n_bin = SUM_cat  C_cat * hist_cat(bin)                 + muon
#
# so d n_bin / d theta_s = SUM_cat slope_cat,s * hist_cat(bin): additive, with constant
# slopes at fixed Delta m^2. C_cat can pass through zero and the dial multiplies a
# category histogram, not a per-event weight, so it is not a d(ln W) form.
HS_DIALS = ("dom_eff", "hole_ice_p0", "hole_ice_p1",
            "bulk_ice_abs", "bulk_ice_scatter")
