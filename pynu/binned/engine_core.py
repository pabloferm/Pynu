#!/usr/bin/env python3
"""Native structural kernels of the SK binned engine (Track S, Phase E1).

Mechanical extraction of the structural/numerical kernels out of
``sk_binned_engine.py`` — contraction, energy-scale migration, direction
smearing, per-era theta view, expectation assembly, the analytic gradient
assembly, the flux-gradient fields, and the per-bin diagnostics report. Each
function takes the engine instance ``eng`` as its first argument (explicit state)
and is called by a one-line delegate on ``SKBinnedEngine``. ZERO numerical
change: every guard, ordering, epsilon and comment is preserved verbatim from the
moved code.

This module deliberately does NOT own the dial tables, ``resolve_nuisance_spec``,
``__init__``, ``cell_weights``, or ``detector_factors`` — those remain on the
engine (their port is Phase E2–E5). The kernels here reference engine attributes
(``eng.Rb``, ``eng.few``, ``eng.nuisance_names`` …) and call back into engine
methods (``eng.cell_weights``, ``eng.detector_factors``, ``eng.expectation`` …)
exactly as the original in-class code did through ``self``.

Track S·F / Phase F3: the two stateless χ² kernels (``poisson_chi2``,
``bb_chi2``) re-homed to ``pynu.fitter.binned_kernels``; the per-point fit
protocol (``fit_point``) + φ tensor lookup (``TensorStore``) + the loaded-triple
holder (``BinnedBinding``) re-homed to ``pynu.fitter.minimizer.binned_fit``. All
five are re-imported here (below) so the engine's delegates and the
``pynu.binned`` back-compat re-exports keep resolving; ZERO numerical change.
"""
import numpy as np

# module-level names the moved code references from the engine module's globals.
from .sk_binned_engine import (
    ERA_TAGS,
    MASK_TUNES,
    XSEC_VECTOR_NAMES,
    SUBGEV_NUE_NORM,
    MULTIGEV_CCQE_NORM,
    DIR_SMEAR_NAME,
    SOLAR_AMP,
    SOLAR_SCALE,
)


# ---------------- contractions ----------------
def contract(eng, W):
    """n_pre[b] = R contracted with cell weights W."""
    return np.bincount(eng.Rb, weights=eng.Rv * W.ravel()[eng.R_widx],
                       minlength=eng.n_bins)


def contract_var(eng, Wsq):
    """sum of BaseWeight^2 * W^2 per bin (pre-detector)."""
    return np.bincount(eng.S2b, weights=eng.S2v * Wsq.ravel()[eng.S2_widx],
                       minlength=eng.n_bins)


def contract_era(eng, W):
    """n_pre[era, b] = R contracted with cell weights W, split by SK era.
    Sums over era to contract(W) exactly (era is a disjoint partition)."""
    return np.bincount(eng.R_eb, weights=eng.Rv * W.ravel()[eng.R_widx],
                       minlength=eng.n_era * eng.n_bins
                       ).reshape(eng.n_era, eng.n_bins)


def contract_var_era(eng, Wsq):
    """Per-era pre-detector BaseWeight^2 * W^2 sum (era, b)."""
    return np.bincount(eng.S2_eb, weights=eng.S2v * Wsq.ravel()[eng.S2_widx],
                       minlength=eng.n_era * eng.n_bins
                       ).reshape(eng.n_era, eng.n_bins)


def escale_migrate(eng, arr_e, deltas, var=False):
    """Per-era energy-scale reco-E migration of a (n_era, n_bins) array.
    Linear, rate-conserving within each (sample, reco-cz) column:
      N'(ie) = N(ie) + d*( N(ie-1)*[ie>0] - N(ie)*[ie<ne-1] ).
    var=True propagates BB variances (independent-bin squared coefficients)."""
    out = np.empty_like(arr_e)
    for e in range(eng.n_era):
        d = deltas[e]
        N = arr_e[e]
        below = np.where(eng.es_below >= 0, N[eng.es_below], 0.0)   # N(ie-1)
        if not var:
            out[e] = N + d * (below * eng.es_has_below - N * eng.es_has_above)
        else:
            c_self = 1.0 - d * eng.es_has_above
            c_below = d * eng.es_has_below
            out[e] = c_self * c_self * N + c_below * c_below * below
    return out


def dir_smear_apply(eng, vec, s):
    """Apply the reco-cz migration operator A = I + s*(M - I) to a (n_bins,)
    vector, block-diagonal per sample x reco-momentum row. E'_i = sum_j A[i,j] E_j;
    at s=1, E' = M @ E per zenith row. Identity (z1bins) samples are untouched."""
    out = vec.copy()
    for off, ne_, nz, M in eng._ds_blocks:
        blk = vec[off:off + ne_ * nz].reshape(ne_, nz)
        sm = blk @ M.T                          # (M @ E_row) for every ie row
        out[off:off + ne_ * nz] = (blk + s * (sm - blk)).ravel()
    return out


def dir_smear_apply_T(eng, vec, s):
    """Apply A^T = I + s*(M^T - I) block-diagonally (used to pull the likelihood
    residual back through the smearing for the OTHER dials' gradient)."""
    out = vec.copy()
    for off, ne_, nz, M in eng._ds_blocks:
        blk = vec[off:off + ne_ * nz].reshape(ne_, nz)
        smT = blk @ M                           # (M^T @ r_row) = r_row @ M
        out[off:off + ne_ * nz] = (blk + s * (smT - blk)).ravel()
    return out


def era_theta(eng, t, e):
    """Per-era view of the nuisance dict: era-split detector stems take their
    era-e dial value; everything else is shared. No-op for legacy specs.

    The up/down energy-scale set is era-split too, but its dials are not in
    DET_ERA_STEMS, so it is routed here into the base key 'updown_escale' that
    detector_factors reads (mirrors the DET_ERA_STEMS remap)."""
    if not eng.det_split_stems and not eng.active_ude:
        return t
    te = dict(t)
    tag = ERA_TAGS[e]
    for stem in eng.det_split_stems:
        te[stem] = t[f"{stem}_{tag}"]
    if eng.active_ude:
        te["updown_escale"] = t[f"updown_escale_{tag}"]
    return te


# ---------------- expectation + chi2 ----------------
def expectation(eng, phi, theta, return_parts=False):
    """Full binned expectation (930) + variance, replicating the event chain.

    Era-aware: E_b = sum_era D_era[b] * n_pre_era[b]. Detector factors are
    evaluated per era (era-split dials take their era value; migration ratios
    use per-era rates). For a single-era response (n_era=1) this reduces
    exactly to the legacy n_pre * D path.
    """
    t = dict(zip(eng.nuisance_names, theta))
    if eng.solar_mix_f is None:
        W = eng.cell_weights(phi, theta)
        Wd = None
        n_pre_e = eng.contract_era(W)             # (n_era, n_bins)
        var_e = eng.contract_var_era(W * W)       # (n_era, n_bins)
    else:
        # solar-mix pair: phi = (phi_solmin, phi_solmax). W is affine in phi
        # (NC classes constant), so W_era = W_a + f_era*(W_b - W_a) exactly.
        f = eng.solar_mix_f[:, None]
        W = eng.cell_weights(phi[0], theta)
        Wd = eng.cell_weights(phi[1], theta) - W
        n_pre_e = eng.contract_era(W) + f * eng.contract_era(Wd)
        # var uses W_era^2: exact quadratic expansion in f.
        var_e = (eng.contract_var_era(W * W)
                 + 2.0 * f * eng.contract_var_era(W * Wd)
                 + f * f * eng.contract_var_era(Wd * Wd))
    # energy-scale: bin-level reco-E migration of the pre-detector rates (so
    # detector factors ride on top). Linear + rate-conserving; no-op at x=1.
    n_pre0_es = es_deltas = None
    if eng.active_energy_scale:
        es_deltas = np.array([t[f"energy_scale_{ERA_TAGS[e]}"] - 1.0
                              for e in range(eng.n_era)])
        n_pre0_es = n_pre_e                      # unmigrated (for the gradient)
        n_pre_e = eng._escale_migrate(n_pre_e, es_deltas)
        var_e = eng._escale_migrate(var_e, es_deltas, var=True)
    if eng.migration_mode == "rawcount":
        # migration ratios are physics-independent raw counts; no phys rates
        nphys_e = [None] * eng.n_era
    elif eng.solar_mix_f is None:
        nphys_e = eng.contract_era(eng.cell_weights_physics_only(phi))
    else:
        Pa = eng.cell_weights_physics_only(phi[0])
        Pd = eng.cell_weights_physics_only(phi[1]) - Pa
        nphys_e = (eng.contract_era(Pa)
                   + eng.solar_mix_f[:, None] * eng.contract_era(Pd))

    n_nu = np.zeros(eng.n_bins)
    var = np.zeros(eng.n_bins)
    parts_e = []
    for e in range(eng.n_era):
        t_e = eng._era_theta(t, e)
        np_e = None if eng.migration_mode == "rawcount" else nphys_e[e]
        rates_e = None if np_e is None else eng.sample_rates(np_e)
        D_e, dlnD_e = eng.detector_factors(t_e, rates_e, n_phys=np_e)
        n_nu = n_nu + n_pre_e[e] * D_e
        var = var + var_e[e] * D_e * D_e
        if return_parts:
            parts_e.append(dict(D=D_e, dlnD=dlnD_e, n_pre=n_pre_e[e],
                                rates=rates_e, n_phys=np_e))
    # direction-smearing: reco-cz migration of the FINAL reco expectation
    # E' = E + s*(M - I) @ E, applied POST-detector, POST-era-sum. M is era-common and
    # the migration is linear, so M(sum_e D_e n_pre_e) = sum_e M(D_e n_pre_e); and for
    # R2DS every active detector factor is per-sample-constant in zenith (and the
    # energy-scale migration is on the orthogonal reco-E axis), so M commutes with D
    # and this equals the per-era pre-detector application EXACTLY -- no ordering
    # approximation for R2DS (a zenith-varying detector dial, e.g. UDE/UBS, would add
    # one; R2DS excludes those). var is left unsmeared (production uses poisson; the
    # analytic gradient holds it fixed like the BB beta). s=0 -> exact no-op (guarded).
    ds_raw = None
    if eng.active_dir_smear:
        s_ds = t[DIR_SMEAR_NAME]
        ds_raw = n_nu                          # pre-smear reco expectation (for grad)
        if s_ds != 0.0:
            n_nu = eng._dir_smear_apply(n_nu, s_ds)
    if return_parts:
        out = dict(W=W, n_pre_e=n_pre_e, var_e=var_e, parts_e=parts_e)
        if ds_raw is not None:
            out.update(dir_smear_raw=ds_raw, dir_smear_s=t[DIR_SMEAR_NAME])
        if Wd is not None:
            out["W_delta"] = Wd
        if eng.active_energy_scale:
            out.update(n_pre0_es=n_pre0_es, es_deltas=es_deltas)
        if eng.n_era == 1:        # legacy keys for the single-era analytic grad
            p0 = parts_e[0]
            out.update(n_pre=n_pre_e[0], D=p0["D"], dlnD=p0["dlnD"],
                       rates=p0["rates"], n_phys=p0["n_phys"])
        return n_nu, var, out
    return n_nu, var


def cell_weights_physics_only(eng, phi):
    P = phi[eng.cls_type, eng.cls_flavor]
    return np.where(eng.cls_cc[:, None, None] == 1, P, 1.0)


def chi2(eng, phi, theta):
    n_nu, var = eng.expectation(phi, theta)
    if eng.likelihood == "poisson":
        stat = eng.poisson_chi2(eng.obs_f, n_nu[eng.few])
    else:
        stat, _, _ = eng.bb_chi2(eng.obs_f, n_nu[eng.few],
                                 var[eng.few])
    pen = np.sum((theta - eng.nominal) ** 2 / eng.sigma ** 2)
    return stat + pen


# ---------------- analytic gradient ----------------
def chi2_and_grad(eng, phi, theta):
    """f, g with the event engine's first-order convention (beta fixed,
    migration r fixed, Jacobian at the current point).

    Era-aware: physics (flux/xsec) params enter the era-independent cell
    weights W, so dE_b = sum_era D_era[b] * contract_era(W * dlnW/dp)[era][b];
    era-split detector dials route to their per-era dial and era-independent
    detector dials accumulate over eras. Reduces to the single-era gradient
    exactly when n_era==1.
    """
    n_nu, var, parts = eng.expectation(phi, theta, return_parts=True)
    m = eng.few
    obs, E, V = eng.obs_f, n_nu[m], var[m]
    pen = np.sum((theta - eng.nominal) ** 2 / eng.sigma ** 2)
    if eng.likelihood == "poisson":
        stat = eng.poisson_chi2(obs, E)
        if stat >= 9e9:                       # unphysical model region
            return stat + pen, 2 * (theta - eng.nominal) / eng.sigma ** 2
        f = stat + pen
        resid = 2 * (1 - obs / np.maximum(E, 1e-9))          # dchi2/dE_b
    else:
        stat, beta, tau = eng.bb_chi2(obs, E, V)
        f = stat + pen
        beta_E = np.maximum(beta * E, 1e-9)
        resid = 2 * (1 - obs / beta_E) * beta                # dchi2/dE_b

    g = 2 * (theta - eng.nominal) / eng.sigma ** 2         # penalty grad
    t = dict(zip(eng.nuisance_names, theta))
    W = parts["W"]
    parts_e = parts["parts_e"]
    D_stack = np.stack([pe["D"] for pe in parts_e])          # (n_era, n_bins)
    # energy-scale migration, held FIXED in the gradient (like migration r / BB
    # beta): flux & xsec grads ride through it (it is linear in n_pre), and the
    # dial's own grad is added below.
    es_deltas = parts.get("es_deltas")                       # (n_era,) or None

    # direction-smearing residual pullback: chi2 depends on E' = A @ E_raw
    # (A = I + s*(M - I)), so for EVERY other dial dChi2/dp = sum_b resid_b (A dE)_b
    # = sum_c (A^T resid)_c dE_c. When dir_smear is inactive or s=0 (A=I), acc(field)
    # is `sum(resid * field[m])` bit-for-bit (the pre-dir_smear code path). When
    # active, resid is pulled back through A^T over ALL bins (smearing leaks across
    # the FewEntries boundary). The dir_smear dial's OWN gradient is added at the end.
    ds_s = t[DIR_SMEAR_NAME] if eng.active_dir_smear else 0.0
    if eng.active_dir_smear and ds_s != 0.0:
        _resid_full = np.zeros(eng.n_bins)
        _resid_full[m] = resid
        _resid_eff = eng._dir_smear_apply_T(_resid_full, ds_s)

        def acc(field):
            return np.sum(_resid_eff * field)
    else:
        def acc(field):
            return np.sum(resid * field[m])

    # physics (flux/xsec) params live in the era-independent cell weights W:
    # dE_b = sum_era D_era[b] * migrate(contract_era(W * dlnW/dp))[era][b].
    def dE_phys(Wg):
        ce = eng.contract_era(Wg)
        if es_deltas is not None:
            ce = eng._escale_migrate(ce, es_deltas)
        return (D_stack * ce).sum(0)                         # (n_bins,)

    # solar-mix aware dE for a d-ln-W field g: dW_era/dp = (W + f_era*Wd)*g
    # (the dial fields are phi-independent). Wd is None on the single-phi path.
    Wd = parts.get("W_delta")

    def dE_W(gfield):
        if Wd is None:
            return dE_phys(W * gfield)
        ce = eng.contract_era(W * gfield) \
            + eng.solar_mix_f[:, None] * eng.contract_era(Wd * gfield)
        if es_deltas is not None:
            ce = eng._escale_migrate(ce, es_deltas)
        return (D_stack * ce).sum(0)                         # (n_bins,)

    # flux tunes: dW/W fields on cells -> dE_b = D * contract(W * g_field)
    # iterate the ACTIVE flux dials (zenith block depends on nuisance_spec)
    for name in eng.flux_names:
        gfield = eng._flux_dlnw(name, t)                    # (n_cls,nE,nZ) or None
        if gfield is None:
            continue
        dE = dE_W(gfield)
        g[eng.nuisance_names.index(name)] += acc(dE)

    # xsec mask tunes: g = bit/x per class
    for name in XSEC_VECTOR_NAMES:
        i = eng.nuisance_names.index(name)
        if name == "AxialMass":
            x = t[name]
            num = 0.042 * 1.05 * eng.log10e                 # d/dx of (1+0.042(x-1)1.05 log10E)
            den = 1.0 + 0.042 * (x - 1.0) * 1.05 * eng.log10e
            gf = np.where(eng.cls_cc[:, None] == 1, num / den, 0.0)
            dE = dE_W(gf[:, :, None])
        else:
            x = t[name]
            j = MASK_TUNES.index(name)
            gcls = np.where(eng.cls_bits[:, j], 1.0 / x, 0.0)
            dE = dE_W(gcls[:, None, None])
        g[i] += acc(dE)

    # optional flux ratios: per-class d ln W / d r, per-dial energy band.
    # heavy f_h=1+band(2r/(1+r)-1), light f_l=1+band(2/(1+r)-1). Generic over
    # the resolved registry (sub-GeV absorbers + energy-banded extensions).
    for name, (band, hv, lt) in eng.fr_resolved.items():
        i = eng.nuisance_names.index(name)
        r = t[name]
        dh = 2.0 / (1.0 + r) ** 2           # d/dr [2r/(1+r)]
        dl = -2.0 / (1.0 + r) ** 2          # d/dr [2/(1+r)]
        fh = 1.0 + band * (2.0 * r / (1.0 + r) - 1.0)
        fl = 1.0 + band * (2.0 / (1.0 + r) - 1.0)
        gh = band * dh / fh                 # (nE,)
        gl = band * dl / fl
        gfield = np.where(hv[:, None], gh[None, :], 0.0) \
            + np.where(lt[:, None], gl[None, :], 0.0)       # (n_cls,nE)
        dE = dE_W(gfield[:, :, None])
        g[i] += acc(dE)

    # optional sub-GeV xsec dials: d ln W / d r.
    if "xsec_ccqe_shape" in t and "xsec_ccqe_shape" in eng.nuisance_names:
        i = eng.nuisance_names.index("xsec_ccqe_shape")
        # W has CCQE *= E^(r-1); d ln/dr = ln(E) on CCQE cells.
        gf = np.where(eng.ccqe_cls[:, None], np.log(eng.e_c)[None, :], 0.0)
        dE = dE_W(gf[:, :, None])
        g[i] += acc(dE)
    # sub-GeV-localized CCQE shape: W has CCQE *= 1 + x*sh(E); d ln W/dx = sh/(1+x*sh).
    if "xsec_ccqe_shape_subgev" in t and "xsec_ccqe_shape_subgev" in eng.nuisance_names:
        i = eng.nuisance_names.index("xsec_ccqe_shape_subgev")
        sh = eng.ccqe_shape_subgev
        fac = 1.0 + t["xsec_ccqe_shape_subgev"] * sh
        dlnw = np.divide(sh, fac, out=np.zeros_like(sh), where=fac != 0)   # (nE,)
        gf = np.where(eng.ccqe_cls[:, None], dlnw[None, :], 0.0)
        dE = dE_W(gf[:, :, None])
        g[i] += acc(dE)
    # sub-GeV nu_e norm dials (lumped surrogate + 1p1h/2p2h split): d ln W/d r
    # = 1[e<1]/fac on the dial's class mask. Generic over SUBGEV_NUE_NORM.
    for _dial, _mattr in SUBGEV_NUE_NORM.items():
        if _dial in t and _dial in eng.nuisance_names:
            i = eng.nuisance_names.index(_dial)
            r = t[_dial]
            fac = 1.0 + eng.e_below1 * (r - 1.0)              # (nE,)
            # d ln W / dr = 1[e<1] / fac; fac->0 as r->0 on sub-GeV cells
            # (where W itself ->0, so those cells contribute 0) — guard the
            # singular divide instead of emitting inf into the gradient.
            dlnw = np.divide(eng.e_below1, fac, out=np.zeros_like(fac),
                             where=fac != 0)
            gf = np.where(getattr(eng, _mattr)[:, None], dlnw[None, :], 0.0)
            dE = dE_W(gf[:, :, None])
            g[i] += acc(dE)
    # multi-GeV CCQE flavor norms: d ln W/dr = 1[E>=1.33]/fac on the flavor mask
    # (mirrors the sub-GeV nu_e norm gradient with the complementary energy mask).
    for _dial, _mattr in MULTIGEV_CCQE_NORM.items():
        if _dial in t and _dial in eng.nuisance_names:
            i = eng.nuisance_names.index(_dial)
            r = t[_dial]
            fac = 1.0 + eng.e_multigev * (r - 1.0)            # (nE,)
            dlnw = np.divide(eng.e_multigev, fac, out=np.zeros_like(fac),
                             where=fac != 0)
            gf = np.where(getattr(eng, _mattr)[:, None], dlnw[None, :], 0.0)
            dE = dE_W(gf[:, :, None])
            g[i] += acc(dE)

    # detector tunes: dE_b = n_pre_era[b] * D_era[b] * dlnD_era[name][b]
    # (migration r fixed). Era-split stems route to their per-era dial; era-
    # independent detector dials (fiducial_volume, neutron-tag, ntag-split
    # bands) accumulate over eras. Reduces to the single-era loop at n_era==1.
    split_set = set(eng.det_split_stems)
    if eng.active_ude:            # 'updown_escale' base -> updown_escale_<era>
        split_set = split_set | {"updown_escale"}
    name_set = set(eng.nuisance_names)
    for e in range(eng.n_era):
        pe = parts_e[e]
        npD = pe["n_pre"] * pe["D"]
        for name, d in pe["dlnD"].items():
            if name in split_set:
                idx = eng.nuisance_names.index(f"{name}_{ERA_TAGS[e]}")
            elif name in name_set:
                idx = eng.nuisance_names.index(name)
            else:
                continue
            g[idx] += acc(npD * d)

    # energy-scale dials: dN'_e/dx_e = N(ie-1)*[ie>0] - N(ie)*[ie<ne-1] on the
    # UNMIGRATED per-era rates (migration is linear in delta=x-1, so this is exact
    # and delta-independent), propagated through the detector factor D_e.
    if eng.active_energy_scale:
        n_pre0_es = parts["n_pre0_es"]
        for e in range(eng.n_era):
            N = n_pre0_es[e]
            below = np.where(eng.es_below >= 0, N[eng.es_below], 0.0)
            dN = below * eng.es_has_below - N * eng.es_has_above
            dE = parts_e[e]["D"] * dN
            g[eng._es_idx[e]] += acc(dE)

    # direction-smearing OWN gradient: n_nu = E_raw + s*((M - I) @ E_raw), so
    # dn_nu/ds = (M - I) @ E_raw (constant in s -> exact, delta-independent), and
    # dChi2/ds = sum_few resid_b * ((M - I) @ E_raw)_b. E_raw is the pre-smear reco
    # expectation captured in expectation(). Added even at s=0 (the gradient is
    # nonzero there); the other dials' grads above are unchanged at s=0 (acc no-op).
    if eng.active_dir_smear:
        e_raw = parts["dir_smear_raw"]
        dE_ds = eng._dir_smear_apply(e_raw, 1.0) - e_raw    # (M - I) @ E_raw
        g[eng.nuisance_names.index(DIR_SMEAR_NAME)] += np.sum(resid * dE_ds[m])

    return f, g


def flux_dlnw(eng, name, t):
    if name == "normalization_below1GeV":
        gf = np.where(eng.e_below1, 1.0 / t[name], 0.0)
        return np.broadcast_to(gf[None, :, None],
                               (eng.n_cls, eng.nE, eng.nZ))
    if name == "normalization_above1GeV":
        gf = np.where(eng.e_above1, 1.0 / t[name], 0.0)
        return np.broadcast_to(gf[None, :, None],
                               (eng.n_cls, eng.nE, eng.nZ))
    if name == "tilt":
        gf = np.log(eng.e_c / 10.0)
        return np.broadcast_to(gf[None, :, None],
                               (eng.n_cls, eng.nE, eng.nZ))
    if name == "nunubar_ratio":
        gcls = np.where(eng.cls_pdg < 0, 1.0 / t[name], 0.0)
        return np.broadcast_to(gcls[:, None, None],
                               (eng.n_cls, eng.nE, eng.nZ))
    if name == "flavor_ratio":
        gcls = np.where(np.abs(eng.cls_pdg) == 12, 1.0 / t[name], 0.0)
        return np.broadcast_to(gcls[:, None, None],
                               (eng.n_cls, eng.nE, eng.nZ))
    if name == "barr_zenith":
        x = t[name]
        r = 1.0 + eng.barr_env * x
        gf = eng.tanh3z[None, :] * (eng.barr_env / r)[:, None]  # (nE,nZ)
        return np.broadcast_to(gf[None, :, :],
                               (eng.n_cls, eng.nE, eng.nZ))
    if name == "zenith_up":                          # w = 1 - x*tanh^2, z<0
        w = 1.0 - t[name] * eng.tanhz2
        gf = np.where(eng.z_c < 0, -eng.tanhz2 / w, 0.0)        # (nZ,)
        return np.broadcast_to(gf[None, None, :],
                               (eng.n_cls, eng.nE, eng.nZ))
    if name == "zenith_down":                        # w = 1 - x*tanh^2, z>=0
        w = 1.0 - t[name] * eng.tanhz2
        gf = np.where(eng.z_c >= 0, -eng.tanhz2 / w, 0.0)       # (nZ,)
        return np.broadcast_to(gf[None, None, :],
                               (eng.n_cls, eng.nE, eng.nZ))
    if name == "flux_horizvert":                 # w = 1 + x*g(cz), g=(1-3cz^2)/2
        w = 1.0 + t[name] * eng.horizvert_shape
        gf = eng.horizvert_shape / w                             # (nZ,)
        return np.broadcast_to(gf[None, None, :],
                               (eng.n_cls, eng.nE, eng.nZ))
    if name == "solar_activity":                     # w = 1 - x*A*exp(-E/L)
        s = SOLAR_AMP * np.exp(-eng.e_c / SOLAR_SCALE)           # (nE,)
        gf = -s / (1.0 - t[name] * s)                            # d ln w / dx, (nE,)
        return np.broadcast_to(gf[None, :, None],
                               (eng.n_cls, eng.nE, eng.nZ))
    if name == "kpi_ratio":                          # w = 1 + x*kpi_shape(E)
        gf = eng.kpi_shape / (1.0 + t[name] * eng.kpi_shape)   # d ln w / dx, (nE,)
        return np.broadcast_to(gf[None, :, None],
                               (eng.n_cls, eng.nE, eng.nZ))
    return None


# ---------------- per-bin diagnostics (pull extraction) ----------------
def per_bin_report(eng, phi, theta):
    """Per-bin decomposition of the stat chi2 at (phi, theta), over the
    FewEntries bins (self.few, obs>5). Returns a dict of arrays aligned to
    the masked bins, plus their GLOBAL bin index (0..n_bins-1) so callers
    can join with the SK release BinInfo (sample / momentum / zenith).

    The per-bin BB-lite chi2 is the exact addend-by-addend split of
    bb_chi2's two sums (Poisson term + Barlow-Beeston penalty), so
    sum(chi2_bin) == bb_chi2(...)[0] to floating precision. The primary
    'pull' is the signed sqrt of that per-bin chi2 (its square sums to the
    stat chi2); 'resid_std' is the conventional (obs - model)/sqrt(model +
    var) standardized residual for cross-reference.

    Only valid for likelihood='bb' (the production form used by this
    diagnostic); raises otherwise.
    """
    if eng.likelihood != "bb":
        raise ValueError("per_bin_report requires likelihood='bb'")
    n_nu, var = eng.expectation(phi, theta)
    m = eng.few
    obs = eng.obs_f
    E = n_nu[m]
    V = var[m]
    stat, beta, tau = eng.bb_chi2(obs, E, V)
    beta_E = np.maximum(beta * E, 1e-9)
    log_term = np.log(np.divide(obs, beta_E, out=np.ones_like(obs),
                                where=beta_E > 0))
    log_term[obs == 0] = 0
    poisson_b = 2.0 * (beta_E - obs + obs * log_term)
    bbpen_b = np.divide((beta - 1.0) ** 2, tau,
                        out=np.zeros_like(tau), where=tau > 0)
    chi2_b = poisson_b + bbpen_b
    pull = np.sign(obs - beta_E) * np.sqrt(np.maximum(chi2_b, 0.0))
    sigma_eff = np.sqrt(np.maximum(E + V, 1e-300))
    resid_std = (obs - E) / sigma_eff
    return dict(
        bin_index=np.nonzero(m)[0].astype(int),
        sample=eng.bin_sample[m].astype(int),
        obs=obs.astype(float), model=E.astype(float),
        beta=beta.astype(float), beta_model=beta_E.astype(float),
        var=V.astype(float), sigma_mc=np.sqrt(V).astype(float),
        tau=tau.astype(float),
        chi2_bin=chi2_b.astype(float),
        poisson_bin=poisson_b.astype(float),
        bbpen_bin=bbpen_b.astype(float),
        pull=pull.astype(float),
        resid_std=resid_std.astype(float),
        stat_total=float(stat),
    )


# --------------------------------------------------------------------------- #
#  Track S·F / Phase F3 — re-homed objects, re-imported for back-compat.
# --------------------------------------------------------------------------- #
# The χ² kernels moved to ``pynu.fitter.binned_kernels`` and the per-point fit
# protocol + φ tensor store + loaded-triple holder moved to
# ``pynu.fitter.minimizer.binned_fit`` (their functional homes). They are
# re-imported here so ``SKBinnedEngine``'s delegates (``poisson_chi2``,
# ``bb_chi2``, ``fit_point``) and the ``pynu.binned`` PEP 562 re-exports
# (``TensorStore``, ``BinnedBinding``) keep resolving off ``engine_core``
# unchanged. Bottom-of-module import: ``binned_fit`` imports the fit-time box
# dicts from ``sk_binned_engine`` (defined above), so importing it here (after
# this module is itself imported at the bottom of ``sk_binned_engine``) resolves
# without a circular import. ZERO numerical change.
from ..fitter.binned_kernels import bb_chi2, poisson_chi2   # noqa: E402,F401
from ..fitter.minimizer.binned_fit import (                 # noqa: E402,F401
    fit_point,
    TensorStore,
    BinnedBinding,
)
