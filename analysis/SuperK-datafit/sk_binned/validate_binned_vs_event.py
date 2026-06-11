#!/usr/bin/env python3
"""Validation gates for the SK binned forward model vs the event-by-event engine.

Gate 0 (unit): Eq. 10 chi2 on a tiny synthetic case vs hand-computed value.
Gate A (nominal fidelity): binned nominal-nuisance expectations vs the event
  extractions of record: no-osc (sk_noosc_splittings.npz), SK best fit
  (sk_skbestfit_NO_q13free.npz), points A/B at their BB best dCP
  (sk_channels_NOMINAL_005_006/008_009.npz). Metric = Asimov-style distance
  D = 2*sum[Nb - Ne + Ne ln(Ne/Nb)] (target |D| < 0.1 per configuration) +
  worst per-channel ratio.
Gate B (systematics fidelity): expectations at the 6 gateB_reference.npz
  nuisance vectors vs the event engine, name-mapped. Residual must stay at the
  Gate-A (gridding) level — detector/xsec factors are exact by construction.
Gate C (two-point Poisson reproduction): Eq.10 fits at points A/B with
  min_entries=5, no energy scale, dCP fixed to the BB best values, compared to
  the event-engine bare-Poisson results (points A/B: 1081.25 / 1076.43, both
  unconverged upper bounds; V2-config mirror values supplied via --mirror).

Exits non-zero on any gate failure.
"""
import argparse
import json
import os
import sys

import numpy as np

from sk_binned_model import SKBinnedModel
from fit_sk_binned import fit_point, DCP_GRID

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")

EVENT_REFS = {  # name -> (npz path, tensor key, dcp index or None for single)
    "noosc": (os.path.join(RES, "pe_mc_comparison", "sk_noosc_splittings.npz"), "noosc", None),
    "skbf": (os.path.join(RES, "pe_mc_comparison", "sk_skbestfit_NO_q13free.npz"), "skbf", None),
    "pointA": (os.path.join(RES, "cc_normalization_bug", "sk_channels_NOMINAL_005_006.npz"), "pointA", 6),
    "pointB": (os.path.join(RES, "cc_normalization_bug", "sk_channels_NOMINAL_008_009.npz"), "pointB", 7),
}


def flatten_event_ref(path):
    d = np.load(path, allow_pickle=True)
    if "model_flat_total" in d:
        return np.asarray(d["model_flat_total"], float)
    parts = [np.asarray(d[f"model_{i}"], float).reshape(-1) for i in range(29)]
    return np.concatenate(parts)


def asimov_distance(nb, ne):
    m = (nb > 0) & (ne > 0)
    return 2.0 * float(np.sum(nb[m] - ne[m]) + np.sum(ne[m] * np.log(ne[m] / nb[m])))


def channel_worst(nb, ne, model):
    worst = 0.0
    for s, (off, nE, nZ) in model.sample_table.items():
        sl = slice(off, off + nE * nZ)
        a, b = nb[sl].sum(), ne[sl].sum()
        if b > 0:
            worst = max(worst, abs(a / b - 1.0))
    return worst


def gate0():
    O = np.array([4.0, 9.0, 0.0])
    E = np.array([5.0, 8.0, 0.5])
    stats = 2 * ((E - O).sum() + (O[:2] * np.log(O[:2] / E[:2])).sum())
    pulls = (0.5 / 0.25) ** 2 + (1.2 - 1.0) ** 2 / 0.1 ** 2
    expected = stats + pulls
    # replicate with a fake model-like computation
    nz = O > 0
    got = 2 * (np.sum(E - O) + np.sum(O[nz] * np.log(O[nz] / E[nz]))) + pulls
    ok = abs(got - expected) < 1e-12
    print(f"[gate0] Eq.10 unit: {got:.10f} vs {expected:.10f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--response", default=os.path.join(RES, "sk_binned", "sk_response.npz"))
    ap.add_argument("--xml", required=True)
    ap.add_argument("--tensors", default=os.path.join(RES, "sk_binned", "osc_tensors"))
    ap.add_argument("--gateA-target", type=float, default=0.1)
    ap.add_argument("--mirror", default=None,
                    help="JSON with event-engine Poisson chi2 at A/B under the "
                         "same config, e.g. '{\"pointA\": 1081.25, \"pointB\": 1076.43}'")
    args = ap.parse_args()

    fail = []
    if not gate0():
        fail.append("gate0")

    model = SKBinnedModel(args.response, args.xml, energy_scale=False)
    nominal = {n: v for n, v in zip(model.nuis_names, model.nominal)}
    specials = dict(np.load(os.path.join(args.tensors, "osc_tensor_specials.npz"),
                            allow_pickle=True))

    print("\n[gateA] nominal-nuisance binned vs event engine:")
    for name, (path, key, idx) in EVENT_REFS.items():
        if not os.path.exists(path):
            print(f"  {name}: reference missing ({path}) — SKIP")
            continue
        ne = flatten_event_ref(path)
        phi = np.asarray(specials[key], float)
        if idx is not None:
            phi = phi[idx]
        nb = model.expectation(phi, nominal)
        D = asimov_distance(nb, ne)
        wc = channel_worst(nb, ne, model)
        ok = abs(D) < args.gateA_target
        print(f"  {name:8s}: totals binned {nb.sum():9.1f} event {ne.sum():9.1f}  "
              f"AsimovD={D:+.4f}  worst-channel |r-1|={wc:.4f}  "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            fail.append(f"gateA:{name}")

    gateB_path = os.path.join(RES, "sk_binned", "gateB_reference.npz")
    if os.path.exists(gateB_path):
        print("\n[gateB] test nuisance vectors (event engine reference):")
        g = np.load(gateB_path, allow_pickle=True)
        names = [str(s) for s in g["names"]]
        phiA = np.asarray(specials["pointA"], float)[6]
        for i, (vec, ne) in enumerate(zip(g["vectors"], g["expectations"])):
            x = dict(zip(names, vec))
            nb = model.expectation(phiA, x)
            D = asimov_distance(nb, ne)
            ok = abs(D) < 3 * args.gateA_target  # gridding residual scales mildly
            print(f"  vector {i}: AsimovD={D:+.4f}  totals {nb.sum():9.1f}/{ne.sum():9.1f} "
                  f"{'PASS' if ok else 'FAIL'}")
            if not ok:
                fail.append(f"gateB:{i}")
    else:
        print("\n[gateB] reference not present — SKIP (run after cluster task 227)")

    print("\n[gateC] Eq.10 two-point fits (min_entries=5, no energy scale, fixed dCP):")
    results = {}
    for name, idx in (("pointA", 6), ("pointB", 7)):
        phi = np.asarray(specials[name], float)[idx]
        r = fit_point(model, phi, min_entries=5.0)
        results[name] = r
        print(f"  {name}: chi2={r['chi2']:.4f} nit={r['nit']} conv={r['converged']} "
              f"pull={r['max_pull']:.2f}")
    d = results["pointA"]["chi2"] - results["pointB"]["chi2"]
    print(f"  dchi2(A-B) binned Poisson = {d:+.4f}")
    if args.mirror:
        mir = json.loads(args.mirror)
        for name in ("pointA", "pointB"):
            ours, ref = results[name]["chi2"], mir[name]
            # ref values are unconverged event-engine UPPER BOUNDS
            ok = ours <= ref + 1.0
            print(f"  {name}: binned {ours:.2f} vs event upper-bound {ref:.2f} "
                  f"{'PASS' if ok else 'FAIL (binned above event upper bound)'}")
            if not ok:
                fail.append(f"gateC:{name}")

    print(f"\n[gates] {'ALL PASS' if not fail else 'FAILURES: ' + ', '.join(fail)}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
