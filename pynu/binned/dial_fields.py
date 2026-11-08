"""Per-cell dial fields shared by the IC and ORCA engines: 19 flux and 4 cross-section
forms, each with its forward factor and its d(ln W)/dx.

A dial is evaluated at the true coordinate of the cell. Dial functions take the
per-cell geometry and the dial value x.

Off-domain values. A form guards an unphysical dial value by returning a scalar, which
collapses the weight to a uniform global factor, with derivative 0. The scalar is 1e-3,
except for ``nunubar_ratio`` and ``flavor_ratio``, where it is 0. ``dlnw_fn`` returns a
literal 0.0 on every off-domain branch, because diff/factor would be 0/0 where the
factor is 0.
"""
import numpy as np

# Off-domain guard: x < low or x > up.
UNPHYS_LOW = 0.0
UNPHYS_UP = 9999999.0
KPI_UNPHYS_LOW = -9999999.0          # for the nominal-0 kpi_ratio and flux_horizvert


def _unphys(x, low=UNPHYS_LOW, up=UNPHYS_UP):
    return bool(x < low or x > up)


# ---------------------------------------------------------------------------
# Per-cell geometry: every theta-independent shape array, precomputed once
# ---------------------------------------------------------------------------
class CellGeom:
    """Theta-independent per-cell geometry and shape arrays, over the populated cells."""

    __slots__ = ("n_cell", "E", "cz", "pdg", "cc",
                 "ln_E_over_10", "log10_E", "exp_mE3", "kpi_ramp",
                 "tanh2_cz", "hv_g",
                 "m_below1", "m_above1", "m_up", "m_down",
                 "m_pdg_neg", "m_abs12", "m_abs16", "m_nc", "m_cc",
                 "bands", "legs",
                 "fr_q")            # flux-ratio dial -> (n_cell,) q

    def __init__(self, c_E, c_cz, c_pdg, c_cc):
        E = np.asarray(c_E, float)
        cz = np.asarray(c_cz, float)
        pdg = np.asarray(c_pdg, np.int64)
        cc = np.asarray(c_cc, np.int64)
        if not (E.shape == cz.shape == pdg.shape == cc.shape) or E.ndim != 1:
            raise ValueError("build_cell_geometry: all inputs must be 1-D, same length")
        self.n_cell = int(E.size)
        self.E, self.cz, self.pdg, self.cc = E, cz, pdg, cc

        # shape arrays
        self.ln_E_over_10 = np.log(E / 10.0)
        self.log10_E = np.log10(E)
        self.exp_mE3 = np.exp(-E / 3.0)
        self.kpi_ramp = np.maximum(0.0, np.log10(E / 3.0))
        self.tanh2_cz = np.tanh(cz) ** 2
        self.hv_g = 0.5 * (1.0 - 3.0 * cz ** 2)

        # masks
        self.m_below1 = E < 1.0
        self.m_above1 = E > 1.0
        self.m_up = cz < 0.0
        self.m_down = cz >= 0.0
        self.m_pdg_neg = pdg < 0
        self.m_abs12 = np.abs(pdg) == 12
        self.m_abs16 = np.abs(pdg) == 16
        self.m_nc = cc == 0
        self.m_cc = cc == 1

        # flux-ratio bands and pdg legs
        self.bands = {"sub": E < 1.0,
                      "mid": (E >= 1.0) & (E < 10.0),
                      "high": E >= 10.0}
        self.legs = {"nuebar": pdg == -12, "nue": pdg == 12,
                     "e": np.abs(pdg) == 12, "mu": np.abs(pdg) == 14,
                     "numubar": pdg == -14, "numu": pdg == 14}

        self.fr_q = {}


def build_cell_geometry(c_E, c_cz, c_pdg, c_cc, fluxq):
    """Build the per-cell geometry the dial fields consume. ``fluxq`` is the path of the
    site's flux-ratio q table."""
    g = CellGeom(c_E, c_cz, c_pdg, c_cc)
    resolve_flux_ratio_state(g, fluxq)
    return g


class DialField:
    """One dial's forward factor and log-derivative on the cell axis.

    algebra:  'N' per-class multiplicative norm (mask/x form; the ORCA engine fast-paths
              these to a sum over class scalars if the mask is measured to be
              class-constant on the loaded response)
              'E' shape in true energy, 'Z' shape in true cos-zenith.
    mask_fn:  for 'N' forms, the cell mask whose factor is x (None otherwise).
    """

    __slots__ = ("name", "algebra", "factor_fn", "dlnw_fn", "mask_fn")

    def __init__(self, name, algebra, factor_fn, dlnw_fn, mask_fn=None):
        self.name, self.algebra = name, algebra
        self.factor_fn, self.dlnw_fn, self.mask_fn = factor_fn, dlnw_fn, mask_fn


# ===========================================================================
# Flux
# ===========================================================================

def _mask_norm_factor(mask, x):
    """x on the mask, 1 elsewhere."""
    w = np.ones(mask.size)
    w[mask] = x
    return w


def _f_norm_below1(g, x):
    if _unphys(x):
        return 1e-3
    return _mask_norm_factor(g.m_below1, x)


def _d_norm_below1(g, x, w):
    if _unphys(x):
        return 0.0
    return g.m_below1 / x                                    # diff/factor


def _f_norm_above1(g, x):
    if _unphys(x):
        return 1e-3
    return _mask_norm_factor(g.m_above1, x)


def _d_norm_above1(g, x, w):
    if _unphys(x):
        return 0.0
    return g.m_above1 / x


def _f_tilt(g, x):
    return (g.E / 10.0) ** x                                 # pivot 10 GeV, no guard


def _d_tilt(g, x, w):
    return g.ln_E_over_10                                    # diff = w*ln(E/10)


def _f_nunubar(g, x):
    if _unphys(x):
        return 0
    return _mask_norm_factor(g.m_pdg_neg, x)


def _d_nunubar(g, x, w):
    if _unphys(x):
        return 0.0
    return g.m_pdg_neg / x


def _f_flavor_ratio(g, x):
    if _unphys(x):
        return 0
    return _mask_norm_factor(g.m_abs12, x)


def _d_flavor_ratio(g, x, w):
    if _unphys(x):
        return 0.0
    return g.m_abs12 / x


def _f_zenith_up(g, x):
    w = np.ones(g.n_cell)
    w[g.m_up] = 1.0 - x * g.tanh2_cz[g.m_up]
    return w


def _d_zenith_up(g, x, w):
    d = np.zeros(g.n_cell)
    d[g.m_up] = -g.tanh2_cz[g.m_up]
    return d / w


def _f_zenith_down(g, x):
    w = np.ones(g.n_cell)
    w[g.m_down] = 1.0 - x * g.tanh2_cz[g.m_down]
    return w


def _d_zenith_down(g, x, w):
    d = np.zeros(g.n_cell)
    d[g.m_down] = -g.tanh2_cz[g.m_down]
    return d / w


def _f_solar(g, x):
    return 1.0 - x * 0.08 * g.exp_mE3


def _d_solar(g, x, w):
    return (-0.08 * g.exp_mE3) / w


def _f_kpi(g, x):
    if _unphys(x, low=KPI_UNPHYS_LOW):
        return 1e-3
    return 1.0 + x * g.kpi_ramp


def _d_kpi(g, x, w):
    if _unphys(x, low=KPI_UNPHYS_LOW):
        return 0.0
    return g.kpi_ramp / w


def _f_horizvert(g, x):
    if _unphys(x, low=KPI_UNPHYS_LOW):
        return 1e-3
    return 1.0 + x * g.hv_g


def _d_horizvert(g, x, w):
    if _unphys(x, low=KPI_UNPHYS_LOW):
        return 0.0
    return g.hv_g / w


# ===========================================================================
# The nine energy-banded flux-ratio dials, with the same algebra as the SK engine:
#
#     f_h = 1 + band(E) * ( x*(1+q) / (1 + q*x)  -  1 )        heavy leg
#     f_l = 1 + band(E) * (   (1+q) / (1 + q*x)  -  1 )        light leg
#
#     dh  =      (1+q) / (1 + q*x)^2          d/dx [ x(1+q)/(1+qx) ]
#     dl  = -q * (1+q) / (1 + q*x)^2          d/dx [   (1+q)/(1+qx) ]
#     dlnW/dx = band*dh/f_h on the heavy leg,  band*dl/f_l on the light leg
#
# q is the heavy/light flux ratio of the two legs at that true energy. With it the
# construction conserves the band total per true-energy cell:
# Phi_h*f_h + Phi_l*f_l = Phi_h + Phi_l when q = Phi_h/Phi_l. x = 1 is a no-op for every q.
FLUX_RATIO_SPEC = {
    "flux_nuebar_subgev":  ("sub", "nuebar", "nue"),
    "flux_nuebar_mid":     ("mid", "nuebar", "nue"),
    "flux_nuebar_high":    ("high", "nuebar", "nue"),
    "flux_flavor_subgev":  ("sub", "mu", "e"),
    "flux_flavor_mid":     ("mid", "mu", "e"),
    "flux_flavor_high":    ("high", "mu", "e"),
    "flux_numubar_subgev": ("sub", "numubar", "numu"),
    "flux_numubar_mid":    ("mid", "numubar", "numu"),
    "flux_numubar_high":   ("high", "numubar", "numu"),
}


# Dial family -> q-table key.
FR_FAM_KEY = {"flux_flavor": "q_flavor",
              "flux_nuebar": "q_nuebar",
              "flux_numubar": "q_numubar"}


def _fr_family(name):
    """`flux_flavor_mid` -> `flux_flavor`."""
    return next((k for k in FR_FAM_KEY if name.startswith(k)), None)


def resolve_flux_ratio_state(g, fluxq):
    """Fill `g.fr_q`: the q table interpolated onto each cell's true energy, in ln E
    (exact at a table node)."""
    table = dict(np.load(fluxq))

    g.fr_q = {}
    for name in FLUX_RATIO_SPEC:
        fam = _fr_family(name)
        if fam is None or FR_FAM_KEY[fam] not in table:
            raise ValueError(
                f"q table {fluxq!r} has no key {FR_FAM_KEY.get(fam)!r} for dial "
                f"{name!r}; keys present: {sorted(k for k in table if k != 'provenance')}")
        key = FR_FAM_KEY[fam]
        qt = np.asarray(table[key], float)
        ec = np.asarray(table["e_c"], float)
        if qt.shape != ec.shape:
            raise ValueError(f"q table {key} size {qt.size} != e_c size {ec.size}")
        g.fr_q[name] = np.interp(np.log(g.E), np.log(ec), qt)


def _f_flux_ratio(name):
    def f(g, x):
        if _unphys(x):
            return 1e-3
        band_name, hv, lt = FLUX_RATIO_SPEC[name]
        band = g.bands[band_name]
        heavy, light = g.legs[hv], g.legs[lt]
        q = g.fr_q[name]
        fh = 1.0 + band * (x * (1.0 + q) / (1.0 + q * x) - 1.0)
        fl = 1.0 + band * ((1.0 + q) / (1.0 + q * x) - 1.0)
        w = np.ones(g.n_cell)
        w[heavy] = np.broadcast_to(fh, (g.n_cell,))[heavy]
        w[light] = np.broadcast_to(fl, (g.n_cell,))[light]
        return w
    return f


def _d_flux_ratio(name):
    def d(g, x, w):
        if _unphys(x):
            return 0.0
        band_name, hv, lt = FLUX_RATIO_SPEC[name]
        band = g.bands[band_name]
        heavy, light = g.legs[hv], g.legs[lt]
        out = np.zeros(g.n_cell)
        q = g.fr_q[name]
        den = (1.0 + q * x) ** 2
        dh = (1.0 + q) / den
        dl = -q * (1.0 + q) / den
        fh = 1.0 + band * (x * (1.0 + q) / (1.0 + q * x) - 1.0)
        fl = 1.0 + band * ((1.0 + q) / (1.0 + q * x) - 1.0)
        gh = np.broadcast_to(band * dh / fh, (g.n_cell,))
        gl = np.broadcast_to(band * dl / fl, (g.n_cell,))
        out[heavy] = gh[heavy]
        out[light] = gl[light]
        return out
    return d


# ===========================================================================
# Cross section: the 4 forms shared by both engines
# ===========================================================================

def _f_xsecnutau(g, x):
    if _unphys(x):
        return 1e-3
    return _mask_norm_factor(g.m_abs16, x)


def _d_xsecnutau(g, x, w):
    if _unphys(x):
        return 0.0
    return g.m_abs16 / x


def _f_nc(g, x):
    if _unphys(x):
        return 1e-3
    return _mask_norm_factor(g.m_nc, x)


def _d_nc(g, x, w):
    if _unphys(x):
        return 0.0
    return g.m_nc / x


def _f_axialmass(g, x):
    if _unphys(x):
        return 1e-3
    w = np.ones(g.n_cell)
    w[g.m_cc] = 1.0 + 0.042 * (x - 1.0) * 1.05 * g.log10_E[g.m_cc]
    return w


def _d_axialmass(g, x, w):
    if _unphys(x):
        return 0.0
    d = np.zeros(g.n_cell)
    d[g.m_cc] = 0.042 * 1.05 * g.log10_E[g.m_cc]
    return d / w


# ===========================================================================
# The registry
# ===========================================================================
def _reg(fields, name, algebra, ffn, dfn, mask_fn=None):
    fields[name] = DialField(name, algebra, ffn, dfn, mask_fn=mask_fn)


FIELDS = {}
_reg(FIELDS, "normalization_below1GeV", "N", _f_norm_below1, _d_norm_below1,
     mask_fn=lambda g: g.m_below1)
_reg(FIELDS, "normalization_above1GeV", "N", _f_norm_above1, _d_norm_above1,
     mask_fn=lambda g: g.m_above1)
_reg(FIELDS, "tilt", "E", _f_tilt, _d_tilt)
_reg(FIELDS, "nunubar_ratio", "N", _f_nunubar, _d_nunubar,
     mask_fn=lambda g: g.m_pdg_neg)
_reg(FIELDS, "flavor_ratio", "N", _f_flavor_ratio, _d_flavor_ratio,
     mask_fn=lambda g: g.m_abs12)
_reg(FIELDS, "zenith_up", "Z", _f_zenith_up, _d_zenith_up)
_reg(FIELDS, "zenith_down", "Z", _f_zenith_down, _d_zenith_down)
_reg(FIELDS, "solar_activity", "E", _f_solar, _d_solar)
_reg(FIELDS, "kpi_ratio", "E", _f_kpi, _d_kpi)
_reg(FIELDS, "flux_horizvert", "Z", _f_horizvert, _d_horizvert)
for _n in FLUX_RATIO_SPEC:
    _reg(FIELDS, _n, "E", _f_flux_ratio(_n), _d_flux_ratio(_n))
_reg(FIELDS, "XSecNuTau", "N", _f_xsecnutau, _d_xsecnutau,
     mask_fn=lambda g: g.m_abs16)
_reg(FIELDS, "NCoverCC", "N", _f_nc, _d_nc,
     mask_fn=lambda g: g.m_nc)
# NCHad applies the same factor to the same mask as NCoverCC, differing only in prior
# width, so the two are exactly degenerate in the likelihood.
_reg(FIELDS, "NCHad", "N", _f_nc, _d_nc,
     mask_fn=lambda g: g.m_nc)
_reg(FIELDS, "AxialMass", "E", _f_axialmass, _d_axialmass)

SHARED_FLUX_19 = ["normalization_below1GeV", "normalization_above1GeV", "tilt",
                  "nunubar_ratio", "flavor_ratio", "zenith_up", "zenith_down",
                  "solar_activity", "kpi_ratio", "flux_horizvert"] + list(FLUX_RATIO_SPEC)
SHARED_XSEC_4 = ["XSecNuTau", "NCoverCC", "NCHad", "AxialMass"]
assert len(SHARED_FLUX_19) == 19 and len(SHARED_XSEC_4) == 4
assert set(FIELDS) == set(SHARED_FLUX_19) | set(SHARED_XSEC_4), sorted(FIELDS)
