"""Shared per-cell dial fields — the 19 flux + 4 xsec forms common to the ORCA and
IC manifests, each with its forward factor and its d(ln W)/dx.

PROVENANCE. Every form here is TRANSCRIBED from the live tunes in the tree the
combined worker imports, and every one of them already ships an analytic `diff_*`
companion there, so these are transcriptions of verified code, not new
derivations. Sources (paths per design ADDENDUM item 8):

    external/Pynu-upstream/pynu/PhysicsTunes/Flux/AtmoFlux.py
    external/Pynu-upstream/pynu/PhysicsTunes/CrossSection/WaterXSection.py

Each field cites the `file:line` block it mirrors. `binned_arms/gates/
gate_orca_grad.py --dial-fields` re-checks all 23 forward factors and all 23
derivatives against the LIVE tune objects, event by event, at 1e-12.

WHY THIS FILE IS SHARED. The 19 flux + 4 xsec blocks are byte-identical in both
manifests. Per scope §4.3 the ORCA track OWNS this module and the IC track
consumes it FROZEN, adding its 11 Mode-keyed xsec dials and 5 hypersurface dials
in an IC-local module. The reason ORCA owns it: ORCA exercises all 23 forms
against an EXACT reference (G-ORCA-1 at 1e-9), so a transcription error is caught
unambiguously, whereas on IC the same error would be indistinguishable from the
Jensen residue that G-IC-4 is there to measure. Change requests, not edits.

THE ONE APPROXIMATION-FREE PREMISE. Evaluating a dial at the CELL's true
coordinate reproduces the per-event value identically only because the ORCA MC
true side is exactly quantized — one distinct ETrue and one distinct cos ZTrue
per true bin (gate G-C2, measured 1 and 1). On IC that premise does not hold and
the same fields carry a Jensen residue; that is an IC-side question, not a
property of these forms.

OFF-DOMAIN (`_unphysical_value`) DISCIPLINE. The tunes guard unphysical dial
values by returning a SCALAR, which collapses NuisanceWeight to a uniform global
factor, with derivative 0. That is reproduced verbatim, including the two cases
where the punished value is 0 rather than 1e-3 (`nunubar_ratio`,
`flavor_ratio` — design §6 says "1e-3 everywhere" and is wrong about these two;
ADDENDUM item 7). Neither can fire inside the ORCA fit box, so this is
faithfulness rather than a live path, but if the cliffs are reproduced at all
they must be reproduced at the right value.

`dlnw` returns d(ln W)/dx = diff/factor, and returns a literal 0.0 on every
off-domain branch. That short circuit matters: `nunubar_ratio` punishes with
factor 0 AND derivative 0, so a naive diff/factor would be 0/0 = nan. Inside the
fit box (nominal +/- 5 sigma, positive-nominal floored at 0.01) no denominator of
any form here vanishes.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Off-domain guard — transcribed from PhysicsTunes.Tune._unphysical_value
# (`external/Pynu-upstream/pynu/PhysicsTunes/PhysicsTunes.py:366-381`):
#     return x < unphys_low or x > unphys_up      # defaults 0, 9999999
# ---------------------------------------------------------------------------
UNPHYS_LOW = 0.0
UNPHYS_UP = 9999999.0
KPI_UNPHYS_LOW = -9999999.0          # AtmoFlux.py:393, :400, :422, :429


def _unphys(x, low=UNPHYS_LOW, up=UNPHYS_UP):
    return bool(x < low or x > up)


# ---------------------------------------------------------------------------
# Per-cell geometry — every theta-independent shape array, precomputed once
# ---------------------------------------------------------------------------
class CellGeom:
    """Theta-independent per-cell geometry + shape arrays.

    Built once per response load over the POPULATED cell list (design §3.3), so
    every dial field below is a cheap elementwise expression on arrays of length
    n_cell (ORCA flat900: 17,236) rather than of length nnz (592,099).
    """

    __slots__ = ("n_cell", "E", "cz", "pdg", "cc",
                 "ln_E_over_10", "log10_E", "exp_mE3", "kpi_ramp",
                 "tanh2_cz", "hv_g",
                 "m_below1", "m_above1", "m_up", "m_down",
                 "m_pdg_neg", "m_abs12", "m_abs16", "m_nc", "m_cc",
                 "bands", "legs")

    def __init__(self, c_E, c_cz, c_pdg, c_cc):
        E = np.asarray(c_E, float)
        cz = np.asarray(c_cz, float)
        pdg = np.asarray(c_pdg, np.int64)
        cc = np.asarray(c_cc, np.int64)
        if not (E.shape == cz.shape == pdg.shape == cc.shape) or E.ndim != 1:
            raise ValueError("build_cell_geometry: all inputs must be 1-D, same length")
        self.n_cell = int(E.size)
        self.E, self.cz, self.pdg, self.cc = E, cz, pdg, cc

        # shape arrays (AtmoFlux.py:137, WaterXSection.py:69, AtmoFlux.py:14/395/424)
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

        # flux-ratio bands (AtmoFlux.py:462-471) and pdg legs (:474-489)
        self.bands = {"sub": E < 1.0,
                      "mid": (E >= 1.0) & (E < 10.0),
                      "high": E >= 10.0}
        self.legs = {"nuebar": pdg == -12, "nue": pdg == 12,
                     "e": np.abs(pdg) == 12, "mu": np.abs(pdg) == 14,
                     "numubar": pdg == -14, "numu": pdg == 14}


def build_cell_geometry(c_E, c_cz, c_pdg, c_cc):
    """Build the per-cell geometry struct the dial fields consume."""
    return CellGeom(c_E, c_cz, c_pdg, c_cc)


class DialField:
    """One dial's forward factor and log-derivative on the cell axis.

    algebra:  'N' per-class multiplicative norm (mask/x form; the engine
              fast-paths these to a sum over 8 class scalars IF the mask is
              measured to be class-constant on the loaded response)
              'E' shape in true energy, 'Z' shape in true cos-zenith.
    axis:     'C' — every field here keys on the true-CELL axis. (ORCA's
              f_HPT/f_Shower are the only bin-axis dials and they are ORCA-local,
              defined in orca_binned_engine.)
    mask_fn:  for 'N' forms, the cell mask whose factor is x (None otherwise).
    """

    __slots__ = ("name", "algebra", "axis", "factor_fn", "dlnw_fn", "mask_fn", "source")

    def __init__(self, name, algebra, factor_fn, dlnw_fn, source,
                 mask_fn=None, axis="C"):
        self.name, self.algebra, self.axis = name, algebra, axis
        self.factor_fn, self.dlnw_fn, self.mask_fn = factor_fn, dlnw_fn, mask_fn
        self.source = source


# ===========================================================================
# FLUX — AtmoFlux.py
# ===========================================================================

def _mask_norm_factor(mask, x):
    """`ones(N); ones[mask] = x` — the tune's per-mask normalization shape."""
    w = np.ones(mask.size)
    w[mask] = x
    return w


def _f_norm_below1(g, x):                                    # AtmoFlux.py:54-70
    if _unphys(x):
        return 1e-3
    return _mask_norm_factor(g.m_below1, x)


def _d_norm_below1(g, x, w):                                 # AtmoFlux.py:72-88
    if _unphys(x):
        return 0.0
    return g.m_below1 / x                                    # diff/factor


def _f_norm_above1(g, x):                                    # AtmoFlux.py:90-105
    if _unphys(x):
        return 1e-3
    return _mask_norm_factor(g.m_above1, x)


def _d_norm_above1(g, x, w):                                 # AtmoFlux.py:107-123
    if _unphys(x):
        return 0.0
    return g.m_above1 / x


def _f_tilt(g, x):                                           # AtmoFlux.py:125-139
    return (g.E / 10.0) ** x                                 # E0Gam = 10 GeV, no guard


def _d_tilt(g, x, w):                                        # AtmoFlux.py:141-155
    return g.ln_E_over_10                                    # diff = w*ln(E/10)


def _f_nunubar(g, x):                                        # AtmoFlux.py:157-172
    if _unphys(x):
        return 0                                             # NOT 1e-3 (ADDENDUM 7)
    return _mask_norm_factor(g.m_pdg_neg, x)


def _d_nunubar(g, x, w):                                     # AtmoFlux.py:174-190
    if _unphys(x):
        return 0.0
    return g.m_pdg_neg / x


def _f_flavor_ratio(g, x):                                   # AtmoFlux.py:192-207
    if _unphys(x):
        return 0                                             # NOT 1e-3 (ADDENDUM 7)
    return _mask_norm_factor(g.m_abs12, x)


def _d_flavor_ratio(g, x, w):                                # AtmoFlux.py:209-225
    if _unphys(x):
        return 0.0
    return g.m_abs12 / x


def _f_zenith_up(g, x):                                      # AtmoFlux.py:227-246
    w = np.ones(g.n_cell)
    w[g.m_up] = 1.0 - x * g.tanh2_cz[g.m_up]
    return w


def _d_zenith_up(g, x, w):                                   # AtmoFlux.py:248-265
    d = np.zeros(g.n_cell)
    d[g.m_up] = -g.tanh2_cz[g.m_up]
    return d / w


def _f_zenith_down(g, x):                                    # AtmoFlux.py:267-286
    w = np.ones(g.n_cell)
    w[g.m_down] = 1.0 - x * g.tanh2_cz[g.m_down]
    return w


def _d_zenith_down(g, x, w):                                 # AtmoFlux.py:288-305
    d = np.zeros(g.n_cell)
    d[g.m_down] = -g.tanh2_cz[g.m_down]
    return d / w


def _f_solar(g, x):                                          # AtmoFlux.py:12-15
    # NOTE the tune allocates ones and then REBINDS w, so the factor applies to
    # EVERY event, not to a mask. Transcribed as written.
    return 1.0 - x * 0.08 * g.exp_mE3


def _d_solar(g, x, w):                                       # AtmoFlux.py:17-20
    return (-0.08 * g.exp_mE3) / w


def _f_kpi(g, x):                                            # AtmoFlux.py:377-396
    if _unphys(x, low=KPI_UNPHYS_LOW):
        return 1e-3
    return 1.0 + x * g.kpi_ramp


def _d_kpi(g, x, w):                                         # AtmoFlux.py:398-402
    if _unphys(x, low=KPI_UNPHYS_LOW):
        return 0.0
    return g.kpi_ramp / w


def _f_horizvert(g, x):                                      # AtmoFlux.py:404-425
    if _unphys(x, low=KPI_UNPHYS_LOW):
        return 1e-3
    return 1.0 + x * g.hv_g


def _d_horizvert(g, x, w):                                   # AtmoFlux.py:427-431
    if _unphys(x, low=KPI_UNPHYS_LOW):
        return 0.0
    return g.hv_g / w


# --- the nine energy-banded, rate-conserving symmetric flux-ratio dials ------
# spec AtmoFlux.py:449-459; weight :491-509; derivative :511-528.
#   heavy leg *= 1 + band*(2x/(1+x) - 1)      light leg *= 1 + band*(2/(1+x) - 1)
# with r == the DIAL VALUE x itself (not a recomputed BaseWeight rate ratio).
# x = 1 is an exact no-op. The heavy and light legs are disjoint in all nine
# triples, but the tune writes heavy first then light, and so does this.
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


def _f_flux_ratio(name):
    def f(g, x):                                             # AtmoFlux.py:491-509
        if _unphys(x):
            return 1e-3
        band_name, hv, lt = FLUX_RATIO_SPEC[name]
        band = g.bands[band_name]
        heavy, light = g.legs[hv], g.legs[lt]
        fh = 1.0 + band * (2.0 * x / (1.0 + x) - 1.0)
        fl = 1.0 + band * (2.0 / (1.0 + x) - 1.0)
        w = np.ones(g.n_cell)
        w[heavy] = fh[heavy]
        w[light] = fl[light]
        return w
    return f


def _d_flux_ratio(name):
    def d(g, x, w):                                          # AtmoFlux.py:511-528
        if _unphys(x):
            return 0.0
        band_name, hv, lt = FLUX_RATIO_SPEC[name]
        band = g.bands[band_name]
        heavy, light = g.legs[hv], g.legs[lt]
        dh = 2.0 / (1.0 + x) ** 2
        dl = -2.0 / (1.0 + x) ** 2
        out = np.zeros(g.n_cell)
        out[heavy] = (band * dh)[heavy]
        out[light] = (band * dl)[light]
        return out / w
    return d


# ===========================================================================
# CROSS SECTION — WaterXSection.py (the 4 forms shared by both manifests)
# ===========================================================================

def _f_xsecnutau(g, x):                                      # WaterXSection.py:17-32
    if _unphys(x):
        return 1e-3
    return _mask_norm_factor(g.m_abs16, x)


def _d_xsecnutau(g, x, w):                                   # WaterXSection.py:34-49
    if _unphys(x):
        return 0.0
    return g.m_abs16 / x


def _f_nc(g, x):                                     # WaterXSection.py:51-56 / :83-88
    if _unphys(x):
        return 1e-3
    return _mask_norm_factor(g.m_nc, x)


def _d_nc(g, x, w):                                  # WaterXSection.py:58-63 / :90-95
    if _unphys(x):
        return 0.0
    return g.m_nc / x


def _f_axialmass(g, x):                                      # WaterXSection.py:65-72
    if _unphys(x):
        return 1e-3
    w = np.ones(g.n_cell)
    w[g.m_cc] = 1.0 + 0.042 * (x - 1.0) * 1.05 * g.log10_E[g.m_cc]
    return w


def _d_axialmass(g, x, w):                                   # WaterXSection.py:74-81
    if _unphys(x):
        return 0.0
    d = np.zeros(g.n_cell)
    d[g.m_cc] = 0.042 * 1.05 * g.log10_E[g.m_cc]
    return d / w


# ===========================================================================
# The registry
# ===========================================================================
def _reg(fields, name, algebra, ffn, dfn, source, mask_fn=None):
    fields[name] = DialField(name, algebra, ffn, dfn, source, mask_fn=mask_fn)


FIELDS = {}
_reg(FIELDS, "normalization_below1GeV", "N", _f_norm_below1, _d_norm_below1,
     "AtmoFlux.py:54-88", mask_fn=lambda g: g.m_below1)
_reg(FIELDS, "normalization_above1GeV", "N", _f_norm_above1, _d_norm_above1,
     "AtmoFlux.py:90-123", mask_fn=lambda g: g.m_above1)
_reg(FIELDS, "tilt", "E", _f_tilt, _d_tilt, "AtmoFlux.py:125-155")
_reg(FIELDS, "nunubar_ratio", "N", _f_nunubar, _d_nunubar,
     "AtmoFlux.py:157-190", mask_fn=lambda g: g.m_pdg_neg)
_reg(FIELDS, "flavor_ratio", "N", _f_flavor_ratio, _d_flavor_ratio,
     "AtmoFlux.py:192-225", mask_fn=lambda g: g.m_abs12)
_reg(FIELDS, "zenith_up", "Z", _f_zenith_up, _d_zenith_up, "AtmoFlux.py:227-265")
_reg(FIELDS, "zenith_down", "Z", _f_zenith_down, _d_zenith_down, "AtmoFlux.py:267-305")
_reg(FIELDS, "solar_activity", "E", _f_solar, _d_solar, "AtmoFlux.py:12-20")
_reg(FIELDS, "kpi_ratio", "E", _f_kpi, _d_kpi, "AtmoFlux.py:377-402")
_reg(FIELDS, "flux_horizvert", "Z", _f_horizvert, _d_horizvert, "AtmoFlux.py:404-431")
for _n in FLUX_RATIO_SPEC:
    _reg(FIELDS, _n, "E", _f_flux_ratio(_n), _d_flux_ratio(_n),
         "AtmoFlux.py:491-528")
_reg(FIELDS, "XSecNuTau", "N", _f_xsecnutau, _d_xsecnutau,
     "WaterXSection.py:17-49", mask_fn=lambda g: g.m_abs16)
_reg(FIELDS, "NCoverCC", "N", _f_nc, _d_nc,
     "WaterXSection.py:51-63", mask_fn=lambda g: g.m_nc)
# NCHad applies the SAME factor to the SAME mask as NCoverCC, differing only in
# prior width (sigma 0.10 vs 0.20). Exactly degenerate in the likelihood — true of
# the event path too, so this is faithful, not a port bug (design B6 / §2.2).
_reg(FIELDS, "NCHad", "N", _f_nc, _d_nc,
     "WaterXSection.py:83-95", mask_fn=lambda g: g.m_nc)
_reg(FIELDS, "AxialMass", "E", _f_axialmass, _d_axialmass, "WaterXSection.py:65-81")

SHARED_FLUX_19 = ["normalization_below1GeV", "normalization_above1GeV", "tilt",
                  "nunubar_ratio", "flavor_ratio", "zenith_up", "zenith_down",
                  "solar_activity", "kpi_ratio", "flux_horizvert"] + list(FLUX_RATIO_SPEC)
SHARED_XSEC_4 = ["XSecNuTau", "NCoverCC", "NCHad", "AxialMass"]
assert len(SHARED_FLUX_19) == 19 and len(SHARED_XSEC_4) == 4
assert set(FIELDS) == set(SHARED_FLUX_19) | set(SHARED_XSEC_4), sorted(FIELDS)


def factor(name, geom, x, registry=None):
    """Forward multiplicative factor of dial `name` at value `x`.

    Returns an (n_cell,) array, or a SCALAR on the tune's off-domain branch (the
    tunes punish with a scalar, which collapses the weight to a uniform global
    factor; reproduced verbatim).
    """
    return (registry or FIELDS)[name].factor_fn(geom, float(x))


def dlnw(name, geom, x, w_cached=None, registry=None):
    """d(ln W)/dx of dial `name` at `x` — an (n_cell,) array, or literal 0.0 on
    the off-domain branch (where the tune's derivative is 0).

    `w_cached` is the dial's own forward factor; pass it to avoid recomputing.
    """
    fld = (registry or FIELDS)[name]
    x = float(x)
    if w_cached is None:
        w_cached = fld.factor_fn(geom, x)
    return fld.dlnw_fn(geom, x, w_cached)
