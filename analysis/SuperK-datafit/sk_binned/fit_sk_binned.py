#!/usr/bin/env python3
"""Grid-scan fitter for the SK binned forward model with the SK-official
likelihood (Eq. 10: Poisson LLR + Gaussian pulls, no Barlow-Beeston).

Mirrors the published SK procedure: chi2 minimized over nuisance pulls at each
fixed oscillation grid point, dCP profiled over the 13 precomputed values,
best fit = lowest grid point. Runs locally on the precomputed response
(sk_response.npz) + oscillation tensors (osc_tensors/).

Modes:
  --points A B ...   fit only named special/grid points (e.g. pointA pointB skbf)
  --grid             full 15x15 scan (all available osc_tensor_*.npz)
Output: per-point JSON rows + assembled npz + text summary.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
from scipy.optimize import minimize

from sk_binned_model import SKBinnedModel

DCP_GRID = np.linspace(0.0, 2.0 * np.pi, 13, endpoint=False)


def make_bounds(model):
    lo, hi = [], []
    for name, nom, sig in model.nuis:
        if nom == 0.0:
            lo.append(nom - 6 * sig); hi.append(nom + 6 * sig)
        else:
            lo.append(max(nom - 6 * sig, 0.05)); hi.append(nom + 6 * sig)
    return list(zip(lo, hi))


def fit_point(model, phi_dcp, min_entries=-1.0, x0=None, profile_dcp=True):
    """phi_dcp: [n_dcp, 2, 3, nE, nZ] (or [2,3,nE,nZ] for a single dcp).
    Returns dict with best chi2 / dcp / nuisance vector."""
    bounds = make_bounds(model)
    if phi_dcp.ndim == 4:
        phi_dcp = phi_dcp[None]
        dcps = [None]
    else:
        dcps = DCP_GRID[: phi_dcp.shape[0]]
    best = dict(chi2=np.inf)
    x_warm = model.nominal.copy() if x0 is None else np.asarray(x0, float)
    order = range(len(dcps)) if profile_dcp else [0]
    for i in order:
        phi = phi_dcp[i]
        res = minimize(lambda v: model.chi2_and_grad(phi, v, min_entries), x_warm,
                       method="L-BFGS-B", jac=True, bounds=bounds,
                       options=dict(maxiter=500, ftol=1e-7, gtol=1e-5))
        if res.fun < best["chi2"]:
            best = dict(chi2=float(res.fun),
                        dcp=(float(dcps[i]) if dcps[i] is not None else None),
                        nuisance=res.x.tolist(), nit=int(res.nit),
                        converged=bool(res.success))
        x_warm = res.x  # warm-start the next dcp
    best["max_pull"] = float(np.max(np.abs(
        (np.array(best["nuisance"]) - model.nominal) / model.sigma)))
    return best


def main():
    ap = argparse.ArgumentParser()
    base = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--response", default=os.path.join(base, "results", "sk_response.npz"))
    ap.add_argument("--xml", required=True)
    ap.add_argument("--tensors", default=os.path.join(base, "results", "osc_tensors"))
    ap.add_argument("--points", nargs="+", default=None,
                    help="special names (pointA pointB skbf noosc) and/or i_j grid ids")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--min-entries", type=float, default=-1.0,
                    help="STRICT cut obs>min_entries (production: 5); -1 keeps all 930")
    ap.add_argument("--no-energy-scale", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    model = SKBinnedModel(args.response, args.xml,
                          energy_scale=not args.no_energy_scale)
    print(f"[fit] {len(model.nuis)} nuisances: {model.nuis_names}")
    print(f"[fit] bins: {model.n_bins}, min_entries={args.min_entries}, "
          f"data total {model.observed.sum():.1f}")

    specials = {}
    sp_path = os.path.join(args.tensors, "osc_tensor_specials.npz")
    if os.path.exists(sp_path):
        specials = dict(np.load(sp_path, allow_pickle=True))

    rows = []
    t0 = time.time()
    if args.points:
        for name in args.points:
            if name in ("pointA", "pointB"):
                phi = specials[name]
            elif name in ("skbf", "noosc"):
                phi = specials[name]
            else:
                i, j = name.split("_")
                phi = np.load(os.path.join(
                    args.tensors, f"osc_tensor_{int(i):03d}_{int(j):03d}.npz"))["phi"]
            r = fit_point(model, np.asarray(phi, float), args.min_entries)
            r["point"] = name
            rows.append(r)
            print(f"[fit] {name}: chi2={r['chi2']:.4f} dcp={r['dcp']} "
                  f"nit={r['nit']} conv={r['converged']} pull={r['max_pull']:.2f} "
                  f"({time.time()-t0:.1f}s elapsed)")
    if args.grid:
        files = sorted(glob.glob(os.path.join(args.tensors, "osc_tensor_[0-9]*.npz")))
        print(f"[fit] grid scan over {len(files)} points")
        x_warm = None
        for f in files:
            tag = os.path.basename(f).replace("osc_tensor_", "").replace(".npz", "")
            d = np.load(f)
            r = fit_point(model, np.asarray(d["phi"], float), args.min_entries,
                          x0=x_warm)
            x_warm = np.array(r["nuisance"])
            r.update(point=tag, dm231=float(d["dm231"]), s23=float(d["s23"]))
            rows.append(r)
            print(f"[fit] {tag}: dm={d['dm231']:.4e} s23={d['s23']:.4f} "
                  f"chi2={r['chi2']:.4f} ({time.time()-t0:.1f}s)")

    out = args.output or os.path.join(base, "results",
                                      "binned_fit_results.json")
    with open(out, "w") as f:
        json.dump(dict(rows=rows, min_entries=args.min_entries,
                       nuis_names=model.nuis_names,
                       elapsed_s=time.time() - t0), f, indent=1)
    print(f"[fit] wrote {out} ({time.time()-t0:.1f}s total)")
    if args.grid and rows:
        g = [r for r in rows if "dm231" in r]
        best = min(g, key=lambda r: r["chi2"])
        print(f"[fit] GRID BEST: {best['point']} dm231={best['dm231']:.4e} "
              f"s23={best['s23']:.4f} chi2={best['chi2']:.4f}")


if __name__ == "__main__":
    main()
