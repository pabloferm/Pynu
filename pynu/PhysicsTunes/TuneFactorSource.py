#!/usr/bin/env python3
"""GridExperiment shim for the binned engine's cell weights (Track S, Phase E5a).

De-vendors the hand-inlined flux/xsec formulas in ``SKBinnedEngine.cell_weights``:
the multiplicative per-dial FACTORS are now sourced from the REAL pynu
PhysicsTunes methods (AtmoFlux / WaterXSection) instead of formulas transcribed a
second time into the engine. ``cell_weights_via_tunes`` reassembles those factors
with the ENGINE'S EXACT association (same reduced-axis shapes, same operator
order as the frozen product) so the result is BYTE-IDENTICAL to the frozen
``cell_weights`` — the binding E1 kernel-parity gate is preserved.

Why factor-sourcing and not a per-event product: the frozen weight is an
axis-factored product — ``f_e`` on (nE,), ``f_cls`` on (n_cls,), ``F_ez`` on
(nE,nZ), ``A_ke``/``FR_ke``/``XX_ke`` on (n_cls,nE) — combined by broadcasting.
Float multiply is non-associative, so a per-event GridExperiment product
(everything flattened to (n_cls*nE*nZ,), combined in method order) would differ
from the frozen product at ~1 ulp on the sensitive dials and break byte-parity
(CLEANUP_MAP §3.1). Instead each real method is evaluated on a MINIMAL
pseudo-experiment sized to that factor's native axis (nE events at the E-cell
centres for E-only factors; n_cls events for class-only factors; nE*nZ for the
zenith block), so the method returns the SAME reduced-axis array the inlined
formula produced, and the engine's own assembly is reused verbatim.

The gradient (``_flux_dlnw`` / ``chi2_and_grad``) is NOT touched here: it keeps
the engine-side per-class dlnw kernels (map binding constraint — the ``diff_``
twins certify the formulas via the E5a weight gate but are not used in the
gradient assembly, to preserve association order / byte-parity).
"""
import numpy as np

# Dial vocabulary from the leaf (Track T / T1+T3) — this module lives in
# PhysicsTunes now (O-1 ruling) and consumes vocabulary, never the engine.
from ..analysis_reader.binned_dials import (
    MASK_TUNES,
    SUBGEV_NUE_NORM,
    MULTIGEV_CCQE_NORM,
)


class _AxisExperiment:
    """Minimal Experiment-shaped object exposing exactly the attributes the
    flux/xsec PhysicsTunes methods read (ETrue, CosZTrue, nuPDG, Mode, CC,
    NumberOfEvents). Used to evaluate a method on a reduced axis (one "event"
    per E-cell / per class / per (E,cz)) so its output matches the reduced-axis
    factor the frozen cell_weights builds."""

    __slots__ = ("ETrue", "CosZTrue", "nuPDG", "Mode", "CC", "NumberOfEvents")

    def __init__(self, ETrue, CosZTrue=None, nuPDG=None, Mode=None, CC=None):
        self.ETrue = np.asarray(ETrue, float)
        n = self.ETrue.size
        self.CosZTrue = (np.zeros(n) if CosZTrue is None
                         else np.asarray(CosZTrue, float))
        self.nuPDG = (np.zeros(n, int) if nuPDG is None
                      else np.asarray(nuPDG))
        self.Mode = (np.zeros(n, int) if Mode is None else np.asarray(Mode))
        self.CC = (np.ones(n, bool) if CC is None else np.asarray(CC, bool))
        self.NumberOfEvents = n


def _cz_only_row(field_ez, dial):
    """Collapse a cz-only (nE, nZ) factor field to its (nZ,) row, ASSERTING it is
    column-constant across E first (Track S review SF-1). zenith_up / zenith_down
    / flux_horizvert key on CosZTrue ONLY, so every E-row is identical and the
    engine's assembly keeps just the (nZ,) factor. This guard makes that
    assumption LOUD: if a future AtmoFlux revision makes one of these dials
    E-dependent, the silent drop of the E-dependence (which byte-parity's frozen
    baseline would not re-catch) becomes an immediate, dial-named failure instead
    of a latent physics/MC discrepancy."""
    if not np.array_equal(field_ez, np.broadcast_to(field_ez[0], field_ez.shape)):
        raise ValueError(
            f"cz-only flux dial {dial!r} produced an E-DEPENDENT (nE,nZ) factor "
            "field; the binned engine assembles it as a pure cosZ (nZ,) factor "
            "(row 0). The dial's PhysicsTunes method is no longer a pure function "
            "of CosZTrue — the (nE,nZ) coupling would be silently dropped. Re-derive "
            "the binned assembly for this dial before use.")
    return field_ez[0, :]


def cell_weights_via_tunes(eng, phi, theta, flux, xsec):
    """Byte-identical re-implementation of ``eng.cell_weights(phi, theta)`` whose
    per-dial factors come from the REAL ``flux``/``xsec`` PhysicsTunes methods.

    Assembly association is IDENTICAL to sk_binned_engine.cell_weights so the
    output is bit-for-bit equal (E1 byte-parity). ``flux`` = AtmosphericFlux,
    ``xsec`` = WaterXSection instances.
    """
    t = dict(zip(eng.nuisance_names, theta))

    # physics: gather per class, NC -> 1  (unchanged; not a tune)
    P = phi[eng.cls_type, eng.cls_flavor]
    P = np.where(eng.cls_cc[:, None, None] == 1, P, 1.0)

    # ---- reduced-axis pseudo-experiments (cell centres) ----
    e_exp = _AxisExperiment(eng.e_c)                     # nE events at E-cells
    # zenith block acts on (nE,nZ): build an (nE*nZ) grid experiment for the
    # methods whose factor the frozen code assembles at (nE,nZ) (barr uses both
    # E and cz; zenith_up/down + horizvert use cz only, E-independent).
    E2, Z2 = np.meshgrid(eng.e_c, eng.z_c, indexing="ij")   # (nE,nZ)
    ez_exp = _AxisExperiment(E2.ravel(), CosZTrue=Z2.ravel())
    # class-axis experiment for the per-class flux scalars (nunubar/flavor)
    cls_exp = _AxisExperiment(np.ones(eng.n_cls), nuPDG=eng.cls_pdg)

    # ---- flux field on (nE,) — real AtmoFlux methods, engine association ----
    # frozen: f_e = where(below1, below, 1) * where(above1, above, 1); then
    #         f_e *= (e_c/10)**tilt; then optional solar, kpi.
    f_below = flux.normalization_below1GeV(e_exp, t["normalization_below1GeV"])
    f_above = flux.normalization_above1GeV(e_exp, t["normalization_above1GeV"])
    f_e = f_below * f_above
    f_e = f_e * flux.tilt(e_exp, t["tilt"])
    if "solar_activity" in t:
        f_e = f_e * flux.solar_activity(e_exp, t["solar_activity"])
    if "kpi_ratio" in t:
        f_e = f_e * flux.kpi_ratio(e_exp, t["kpi_ratio"])

    # ---- zenith block -> F_ez on (nE,nZ) ----
    # barr_zenith is E,cz-coupled and carries the r<=0 punishment (whole-field
    # 1e-3). It matches the engine's F_ez = f_e[:,None] * barr[:,None]**tanh3z.
    if "barr_zenith" in t:
        barr = 1.0 + eng.barr_env * t["barr_zenith"]
        if np.any(barr <= 0):
            F_ez = 1e-3 * np.ones((eng.nE, eng.nZ))
        else:
            F_ez = f_e[:, None] * barr[:, None] ** eng.tanh3z[None, :]
    else:
        F_ez = f_e[:, None] * np.ones((eng.nE, eng.nZ))
    if "zenith_up" in t:
        zu = flux.zenith_up(ez_exp, t["zenith_up"]).reshape(eng.nE, eng.nZ)
        F_ez = F_ez * _cz_only_row(zu, "zenith_up")[None, :]
    if "zenith_down" in t:
        zd = flux.zenith_down(ez_exp, t["zenith_down"]).reshape(eng.nE, eng.nZ)
        F_ez = F_ez * _cz_only_row(zd, "zenith_down")[None, :]
    if "flux_horizvert" in t:
        hv = flux.flux_horizvert(ez_exp, t["flux_horizvert"]).reshape(eng.nE, eng.nZ)
        F_ez = F_ez * _cz_only_row(hv, "flux_horizvert")[None, :]

    # ---- per-class flux scalars (nunubar/flavor) on (n_cls,) ----
    f_cls = (flux.nunubar_ratio(cls_exp, t["nunubar_ratio"])
             * flux.flavor_ratio(cls_exp, t["flavor_ratio"]))

    # ---- xsec: 12 mask tunes -> per-class scalar product ----
    # frozen: X_cls = where(cls_bits, xs, 1).prod(axis=1), xs in MASK_TUNES order.
    xs = np.array([t[n] for n in MASK_TUNES])
    X_cls = np.where(eng.cls_bits, xs[None, :], 1.0).prod(axis=1)

    # ---- AxialMass: CC only, continuous in log10 ETrue -> A_ke on (n_cls,nE) ----
    ax = 1.0 + 0.042 * (t["AxialMass"] - 1.0) * 1.05 * eng.log10e
    A_ke = np.where(eng.cls_cc[:, None] == 1, ax[None, :], 1.0)

    # ---- optional flux ratios -> FR_ke on (n_cls,nE) (engine assembly) ----
    FR_ke = np.ones((eng.n_cls, eng.nE))
    for name, (band, hv, lt) in eng.fr_resolved.items():
        r = t[name]
        fh = 1.0 + band[None, :] * (2.0 * r / (1.0 + r) - 1.0)
        fl = 1.0 + band[None, :] * (2.0 / (1.0 + r) - 1.0)
        FR_ke = FR_ke * np.where(hv[:, None], fh, 1.0) \
                      * np.where(lt[:, None], fl, 1.0)

    # ---- optional sub-GeV xsec dials -> XX_ke on (n_cls,nE) (engine assembly) ----
    XX_ke = np.ones((eng.n_cls, eng.nE))
    if eng.active_xsec_extra:
        if "xsec_ccqe_shape" in t:
            b = t["xsec_ccqe_shape"] - 1.0
            tilt = eng.e_c[None, :] ** b
            XX_ke = XX_ke * np.where(eng.ccqe_cls[:, None], tilt, 1.0)
        if "xsec_ccqe_shape_subgev" in t:
            fac = 1.0 + t["xsec_ccqe_shape_subgev"] * eng.ccqe_shape_subgev
            XX_ke = XX_ke * np.where(eng.ccqe_cls[:, None], fac[None, :], 1.0)
        for _dial, _mattr in SUBGEV_NUE_NORM.items():
            if _dial in t:
                r = t[_dial]
                fac = 1.0 + eng.e_below1[None, :] * (r - 1.0)
                XX_ke = XX_ke * np.where(getattr(eng, _mattr)[:, None], fac, 1.0)
    if eng.active_multigev_ccqe:
        for _dial, _mattr in MULTIGEV_CCQE_NORM.items():
            if _dial in t:
                r = t[_dial]
                fac = 1.0 + eng.e_multigev[None, :] * (r - 1.0)
                XX_ke = XX_ke * np.where(getattr(eng, _mattr)[:, None], fac, 1.0)

    W = P * F_ez[None, :, :] * (f_cls * X_cls)[:, None, None] \
        * A_ke[:, :, None] * FR_ke[:, :, None] * XX_ke[:, :, None]
    return W
