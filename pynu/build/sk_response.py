"""SK binned-forward-model response matrices from the event-by-event MC.

One pass over the MC through the SuperK_2023 experiment class (NC weight fix, NORM,
CC-mask encoding inherited). Outputs: COO response R[k, c_E, c_Z, b, era] (BaseWeight),
R_plus/R_minus (EReco x1.02 / x0.98), sumw2 (BaseWeight^2), class table from the actual
xsec tunes at x=2, true grid (log-E, one edge snapped to 1.0 GeV), reco sample table,
unfiltered observed data vector (930).
"""
import json
import os

import numpy as np

from .events import load_pynufit

XSEC_TUNES = [
    "XSecNuTau", "NCoverCC", "NCHad", "DIS",
    "CC_2p2h", "CC_2p2hNuBarNu", "CC_2p2hMuE",
    "CCQE", "CCQENuBarNu",
    "CCQEMuE", "CC1Pi_Pi0Pi", "CC1Pi_NuBarNuE", "CC1Pi_NuBarNuMu",
    "CC1PiProduction", "CohPiProduction",
]
# The axial-mass dial is continuous in log10 ETrue for CC events: the engine evaluates it
# at the true-cell centres, so it needs no class bit.

# The true grid is pinned to this range, which the SK tensors share. The MC spans
# [0.0800, 59481.65] GeV, so the sub-0.1 GeV events are clipped into the first cell (they
# keep their reco bin).
PINNED_ETRUE_RANGE = (0.1000172462187625, 466048.7104907757)
N_ETRUE, N_CZTRUE = 400, 80
N_ERA = 4


def make_true_grid(emin, emax, n_e, n_z):
    e_edges = np.geomspace(emin, emax, n_e + 1)
    # snap the nearest edge to exactly 1.0 GeV (step tunes norm_below/above1GeV)
    i = np.argmin(np.abs(np.log(e_edges) - np.log(1.0)))
    e_edges[i] = 1.0
    z_edges = np.linspace(-1.0, 1.0, n_z + 1)
    return e_edges, z_edges


def reco_bin_index(exp, scale=1.0):
    """Per-event flat reco-bin index replicating BinIt_MC_2D ordering
    (per sample in exp.Samples order: C-order (E, cz) flatten). -1 = out of range."""
    E = exp.EReco * scale
    cz = exp.CosZReco
    idx = np.full(exp.NumberOfEvents, -1, dtype=np.int64)
    offsets = {}
    off = 0
    for s in exp.Samples:
        eb = np.asarray(exp.EnergyBins[s], float)
        zb = np.asarray(exp.CTBins[s], float)
        ne, nz = eb.size - 1, zb.size - 1
        offsets[int(s)] = (off, ne, nz)
        m = exp.Sample == s
        ie = np.digitize(E[m], eb) - 1
        iz = np.digitize(cz[m], zb) - 1
        # histogram fill includes the upper edge in the last bin; digitize doesn't
        ie[E[m] == eb[-1]] = ne - 1
        iz[cz[m] == zb[-1]] = nz - 1
        ok = (ie >= 0) & (ie < ne) & (iz >= 0) & (iz < nz)
        flat = np.where(ok, off + ie * nz + iz, -1)
        idx[m] = flat
        off += ne * nz
    return idx, offsets, off


def build_sk_response(event_xml, sk_mc, sk_data, out_path):
    n_etrue, n_cztrue = N_ETRUE, N_CZTRUE

    _, _, exp, pt = load_pynufit(event_xml, {"SK_MC": sk_mc, "SK_DATA": sk_data})
    N = exp.NumberOfEvents
    print(f"[sk-response] {N} events")

    # ---- class signatures from the actual xsec tunes (x=2 -> factor 2 marks mask)
    xsec_obj = pt.XSectionTunes
    bits = np.zeros((N, len(XSEC_TUNES)), dtype=np.int8)
    for j, name in enumerate(XSEC_TUNES):
        w = getattr(xsec_obj, name)(exp, 2.0)
        bits[:, j] = (np.asarray(w) != 1.0).astype(np.int8)
    cc = np.asarray(exp.CC, dtype=np.int8)
    pdg = np.asarray(exp.nuPDG, dtype=np.int64)

    sig = np.column_stack([pdg, cc, bits])
    classes, class_inv = np.unique(sig, axis=0, return_inverse=True)
    n_cls = classes.shape[0]
    print(f"[sk-response] {n_cls} event classes")

    # ---- true grid
    emin, emax = PINNED_ETRUE_RANGE
    if not (emin <= exp.Etrue_min and exp.Etrue_max <= emax):
        n_lo = int(np.sum(exp.ETrue < emin))
        n_hi = int(np.sum(exp.ETrue > emax))
        print(f"[sk-response] {n_lo} events below emin and {n_hi} above emax are "
              f"CLIPPED into the edge cells "
              f"({(n_lo + n_hi) / exp.NumberOfEvents:.4%} of events)")
    print(f"[sk-response] true grid PINNED to [{emin}, {emax}] "
          f"(MC range [{exp.Etrue_min}, {exp.Etrue_max}])")
    e_edges, z_edges = make_true_grid(emin, emax, n_etrue, n_cztrue)
    ie = np.clip(np.digitize(exp.ETrue, e_edges) - 1, 0, n_etrue - 1)
    iz = np.clip(np.digitize(exp.CosZTrue, z_edges) - 1, 0, n_cztrue - 1)

    # ---- reco bin indices (nominal and +-2% energy scale)
    b0, offsets, n_bins = reco_bin_index(exp, 1.0)
    bp, _, _ = reco_bin_index(exp, 1.02)
    bm, _, _ = reco_bin_index(exp, 0.98)

    w = np.asarray(exp.BaseWeight, float)
    w2 = w * w

    # ---- SK run-period era group (sk_phase 1..5 -> 0=SK-I, 1=SK-II, 2=SK-III,
    #      3=SK-IV+V); an all-zero sk_phase collapses to era 0.
    skphase = np.asarray(exp.SKPhase, dtype=np.int64)
    era = np.minimum(np.maximum(skphase, 1), 4) - 1

    def coo(bidx, weights):
        m = bidx >= 0
        # key digits slowest->fastest: class, ie, iz, recobin, era
        key = (((class_inv[m].astype(np.int64) * n_etrue + ie[m])
                * n_cztrue + iz[m]) * n_bins + bidx[m]) * N_ERA + era[m]
        uniq, inv = np.unique(key, return_inverse=True)
        val = np.bincount(inv, weights=weights[m], minlength=uniq.size)
        er = uniq % N_ERA
        rem = uniq // N_ERA
        b = rem % n_bins
        rem = rem // n_bins
        cz_ = rem % n_cztrue
        rem = rem // n_cztrue
        ce = rem % n_etrue
        k = rem // n_etrue
        return (k.astype(np.int32), ce.astype(np.int32), cz_.astype(np.int32),
                b.astype(np.int32), er.astype(np.int8), val)

    Rk, Re, Rz, Rb, Rera, Rv = coo(b0, w)
    Pk, Pe, Pz, Pb, Pera, Pv = coo(bp, w)
    Mk, Me, Mz, Mb, Mera, Mv = coo(bm, w)
    Sk_, Se, Sz, Sb, Sera, Sv = coo(b0, w2)
    print(f"[sk-response] R: {Rv.size} nonzeros, total weight {Rv.sum():.1f}")

    # ---- observed data vector (unfiltered, release order)
    obs = np.asarray(exp.BinData(), float)
    assert obs.size == n_bins

    # ---- self-check: R summed over (k, cells) == BinMC(ones)
    binned_base = np.asarray(exp.BinMC(np.ones(N)))
    rsum = np.zeros(n_bins)
    np.add.at(rsum, Rb, Rv)
    resid = np.max(np.abs(rsum - binned_base))
    ok = resid < 1e-6 * max(binned_base.max(), 1)
    print(f"[sk-response] self-check R vs BinMC(ones): max |diff| = {resid:.3e} "
          f"({'PASS' if ok else 'FAIL'})")
    if not ok:
        raise RuntimeError("R does not reproduce BinMC(ones) -- bin replication bug")

    sample_table = json.dumps({int(s): offsets[int(s)] for s in exp.Samples})
    sample_counts = {int(s): int(np.sum(exp.Sample == s)) for s in exp.Samples}
    np.savez_compressed(
        out_path,
        sample_event_counts=json.dumps(sample_counts),
        classes=classes, xsec_tune_names=np.array(XSEC_TUNES),
        R_k=Rk, R_e=Re, R_z=Rz, R_b=Rb, R_era=Rera, R_v=Rv,
        Rp_k=Pk, Rp_e=Pe, Rp_z=Pz, Rp_b=Pb, Rp_era=Pera, Rp_v=Pv,
        Rm_k=Mk, Rm_e=Me, Rm_z=Mz, Rm_b=Mb, Rm_era=Mera, Rm_v=Mv,
        S2_k=Sk_, S2_e=Se, S2_z=Sz, S2_b=Sb, S2_era=Sera, S2_v=Sv,
        n_era=np.int64(N_ERA),
        e_edges=e_edges, z_edges=z_edges, n_bins=np.int64(n_bins),
        observed=obs, sample_table=sample_table,
        meta=json.dumps({
            "config": os.path.basename(event_xml), "n_events": int(N),
            "n_classes": int(n_cls),
            "etrue_range": [float(emin), float(emax)],
            "etrue_range_mc": [float(exp.Etrue_min), float(exp.Etrue_max)],
            "n_etrue": n_etrue, "n_cztrue": n_cztrue,
            "binned_base_total": float(binned_base.sum()),
        }),
    )
    print(f"[sk-response] wrote {out_path}")
    return out_path
