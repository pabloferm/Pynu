#!/usr/bin/env python3
"""Build the SK binned-forward-model response matrices from the event-by-event MC.

One pass over the 3.8M-event MC THROUGH the SuperK_2023 experiment class (so all
conventions — NC w_no fix, NORM, WMC, CC-mask encoding — are inherited, never
re-implemented). Outputs `sk_response.npz` containing:

  - Sparse COO response R[k, c_E, c_Z, b] = sum of BaseWeight over events of
    class k in true cell (c_E, c_Z) landing in reco bin b. BaseWeight because
    BinIt_MC_2D multiplies the binned array by BaseWeight internally
    (Experiment.py:142-160) — the engine contracts R with physics x nuisance
    weights evaluated on the true grid.
  - R_plus / R_minus: same events re-binned with EReco*1.02 / *0.98 (the
    ENERGY_SCALE path of BinIt_MC_2D) for the Fij energy-scale response.
  - sumw2: same COO with BaseWeight^2 (optional Barlow-Beeston mode).
  - Class table: per class, (nuPDG, CC) + the 12 xsec-tune mask bits, computed
    by EVALUATING THE ACTUAL TUNES (WaterXSection methods at x=2) so the masks
    can never drift from production code (incl. the DIS |Mode|>25*CC quirk
    that tags all Mode!=0 NC events).
  - True grid edges (log-E with an edge snapped to exactly 1.0 GeV so the
    normalization_below/above1GeV step tunes are exact on the grid), reco bin
    structure (sample offsets), unfiltered observed data vector (930).

Usage (cluster, nuSQuIDS env):
    python build_sk_response.py --config <xsec_barr_ntag.xml> --output sk_response.npz \
        [--n-etrue 200 --n-cztrue 40]
"""
import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))  # scripts/ for the worker module
from run_sk_datafit_row_worker import setup_pynufit_datafit  # noqa: E402

XSEC_TUNES = [
    "XSecNuTau", "NCoverCC", "NCHad", "DIS", "CCQE", "CCQENuBarNu",
    "CCQEMuE", "CC1Pi_Pi0Pi", "CC1Pi_NuBarNuE", "CC1Pi_NuBarNuMu",
    "CC1PiProduction", "CohPiProduction",
]
# AxialMass handled separately (continuous in log10 ETrue, CC only).


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default="sk_response.npz")
    ap.add_argument("--n-etrue", type=int, default=200)
    ap.add_argument("--n-cztrue", type=int, default=40)
    args = ap.parse_args()

    pynufit, n_data, _ = setup_pynufit_datafit(args.config)
    exp_key = list(pynufit.Experiments.keys())[0]
    exp = pynufit.Experiments[exp_key]
    pt = pynufit.physics_tunes[exp_key]

    N = exp.NumberOfEvents
    print(f"[build] {N} events; n_data={n_data:.1f}")

    # ---- class signatures from the ACTUAL xsec tunes (x=2 -> factor 2 marks mask)
    xsec_obj = pt.XSectionTunes if hasattr(pt, "XSectionTunes") else None
    bits = np.zeros((N, len(XSEC_TUNES)), dtype=np.int8)
    for j, name in enumerate(XSEC_TUNES):
        if xsec_obj is not None and hasattr(xsec_obj, name):
            w = getattr(xsec_obj, name)(exp, 2.0)
        else:  # fall back through the dispatcher
            w = pt.get_xsection(name, 2.0)
        bits[:, j] = (np.asarray(w) != 1.0).astype(np.int8)
    cc = np.asarray(exp.CC, dtype=np.int8)
    pdg = np.asarray(exp.nuPDG, dtype=np.int64)

    sig = np.column_stack([pdg, cc, bits])
    classes, class_inv = np.unique(sig, axis=0, return_inverse=True)
    n_cls = classes.shape[0]
    print(f"[build] {n_cls} event classes (pdg x CC x 12 xsec-mask bits)")

    # ---- true grid
    e_edges, z_edges = make_true_grid(exp.Etrue_min, exp.Etrue_max,
                                      args.n_etrue, args.n_cztrue)
    ie = np.clip(np.digitize(exp.ETrue, e_edges) - 1, 0, args.n_etrue - 1)
    iz = np.clip(np.digitize(exp.CosZTrue, z_edges) - 1, 0, args.n_cztrue - 1)

    # ---- reco bin indices (nominal and +-2% energy scale)
    b0, offsets, n_bins = reco_bin_index(exp, 1.0)
    bp, _, _ = reco_bin_index(exp, 1.02)
    bm, _, _ = reco_bin_index(exp, 0.98)
    print(f"[build] {n_bins} reco bins; out-of-range fractions: "
          f"nominal {np.mean(b0 < 0):.4%}, +2% {np.mean(bp < 0):.4%}, -2% {np.mean(bm < 0):.4%}")

    w = np.asarray(exp.BaseWeight, float)
    w2 = w * w

    def coo(bidx, weights):
        m = bidx >= 0
        key = ((class_inv[m].astype(np.int64) * args.n_etrue + ie[m]) * args.n_cztrue
               + iz[m]) * n_bins + bidx[m]
        uniq, inv = np.unique(key, return_inverse=True)
        val = np.bincount(inv, weights=weights[m], minlength=uniq.size)
        k = uniq // (args.n_etrue * args.n_cztrue * n_bins)
        rem = uniq % (args.n_etrue * args.n_cztrue * n_bins)
        ce = rem // (args.n_cztrue * n_bins)
        rem = rem % (args.n_cztrue * n_bins)
        cz_ = rem // n_bins
        b = rem % n_bins
        return (k.astype(np.int32), ce.astype(np.int32), cz_.astype(np.int32),
                b.astype(np.int32), val)

    print("[build] aggregating R (nominal) ...")
    Rk, Re, Rz, Rb, Rv = coo(b0, w)
    print(f"[build]   {Rv.size} nonzeros, total weight {Rv.sum():.1f}")
    print("[build] aggregating R_plus / R_minus (energy scale) ...")
    Pk, Pe, Pz, Pb, Pv = coo(bp, w)
    Mk, Me, Mz, Mb, Mv = coo(bm, w)
    print("[build] aggregating sumw2 ...")
    Sk_, Se, Sz, Sb, Sv = coo(b0, w2)

    # ---- observed data vector (unfiltered, release order)
    obs = np.asarray(exp.BinData(), float)
    print(f"[build] data total {obs.sum():.1f} over {obs.size} bins")
    assert obs.size == n_bins

    # ---- self-check: R summed over (k, cells) == BinMC(ones)
    binned_base = np.asarray(exp.BinMC(np.ones(N)))
    rsum = np.zeros(n_bins)
    np.add.at(rsum, Rb, Rv)
    resid = np.max(np.abs(rsum - binned_base))
    print(f"[build] self-check R vs BinMC(ones): max |diff| = {resid:.3e} "
          f"({'PASS' if resid < 1e-6 * max(binned_base.max(), 1) else 'FAIL'})")
    if resid >= 1e-6 * max(binned_base.max(), 1):
        raise RuntimeError("R does not reproduce BinMC(ones) — bin replication bug")

    sample_table = json.dumps({int(s): offsets[int(s)] for s in exp.Samples})
    # raw (unweighted) event counts per sample — some detector tunes (e.g.
    # fcpc_separation) use np.sum(mask) raw counts, which the binned engine
    # must replicate verbatim
    sample_counts = {int(s): int(np.sum(exp.Sample == s)) for s in exp.Samples}
    np.savez_compressed(
        args.output,
        sample_event_counts=json.dumps(sample_counts),
        classes=classes, xsec_tune_names=np.array(XSEC_TUNES),
        R_k=Rk, R_e=Re, R_z=Rz, R_b=Rb, R_v=Rv,
        Rp_k=Pk, Rp_e=Pe, Rp_z=Pz, Rp_b=Pb, Rp_v=Pv,
        Rm_k=Mk, Rm_e=Me, Rm_z=Mz, Rm_b=Mb, Rm_v=Mv,
        S2_k=Sk_, S2_e=Se, S2_z=Sz, S2_b=Sb, S2_v=Sv,
        e_edges=e_edges, z_edges=z_edges, n_bins=np.int64(n_bins),
        observed=obs, sample_table=sample_table,
        meta=json.dumps({
            "config": args.config, "n_events": int(N), "n_classes": int(n_cls),
            "etrue_range": [float(exp.Etrue_min), float(exp.Etrue_max)],
            "n_etrue": args.n_etrue, "n_cztrue": args.n_cztrue,
            "binned_base_total": float(binned_base.sum()),
        }),
    )
    print(f"[build] wrote {args.output}")


if __name__ == "__main__":
    main()
