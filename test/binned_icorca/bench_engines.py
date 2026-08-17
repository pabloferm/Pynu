"""Micro-benchmark: per-evaluation cost of both binned engines.

Answers "what does one analytic gradient cost, in units of the event-path FD
evaluations it replaces". LOCAL — both engines are pure numpy, so no nuflux and no
nuSQuIDS. φ is synthetic; timing does not depend on physics validity, only on
shapes and occupancy, which are the real ones from the real responses.

★ LOCAL CPU IS NOT CLUSTER CPU. Absolute milliseconds here are indicative only.
The RATIO (analytic gradient / event-path FD evaluation) is the transferable
number, and even it assumes the two scale together across machines. The definitive
figure is the cluster's own `fd_stats` line from the G-E2E run.

FASRC event-path baselines, stage 0 (job 39678828, two cells, consistent):
  0.3431 s per FD model evaluation
  22.64 s per gradient (66 evaluations: 2 x 33 dials, central difference)

    python test/binned_icorca/bench_engines.py \
        --ic-response <ic_response_modeaxis_L3.npz> \
        --orca-response <orca_response_flat900.npz> [--n 200]
"""
import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = str(Path(__file__).resolve().parents[2])   # repo root: test/binned_icorca/..
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

FD_EVAL_S = 0.3431          # stage 0, FASRC
FD_GRAD_S = 22.64           # stage 0, FASRC (66 evals)
FD_N_EVAL = 66


def timeit(fn, n, warmup=20):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)      # ms
    ts.sort()
    return {"median": statistics.median(ts),
            "p10": ts[int(0.10 * len(ts))],
            "p90": ts[int(0.90 * len(ts))],
            "min": ts[0]}


def bench_ic(path, n):
    from pynu.Experiments.ic_binned_engine import (
        ICBinnedEngine, HS_CATEGORIES, HS_DIALS, N_BINS)
    from gate_ic_engine import NAMES, nominal_theta
    rng = np.random.default_rng(11)
    boot = ICBinnedEngine(path, np.ones(N_BINS), np.zeros(N_BINS), NAMES, norm=1.0)
    c = boot.cells
    phi = rng.uniform(0.4, 1.6, (2, 3, c.nE, c.nZ))
    hs = {cat: {"intercept": np.full(N_BINS, 1.0)} for cat in HS_CATEGORIES}
    for cat in HS_CATEGORIES:
        for d in HS_DIALS:
            hs[cat][d] = rng.normal(0.0, 0.05, N_BINS)
    mu = np.full(N_BINS, 512.166 / N_BINS)
    th0 = nominal_theta()
    e0 = ICBinnedEngine(path, np.ones(N_BINS), mu, NAMES, norm=1.0, hs_slopes=hs)
    obs = e0.expectation(phi, th0)
    eng = ICBinnedEngine(path, obs, mu, NAMES, norm=1.0, hs_slopes=hs)
    thetas = [("nominal", th0)] + [(f"rand{i}", th0 + rng.normal(0, 0.02, len(NAMES)))
                                   for i in range(3)]
    return eng, phi, thetas, len(NAMES), c.n_cell, c.n_entry


def bench_orca(path, n):
    from pynu.Experiments.orca_binned_engine import ORCABinnedEngine, ORCA_MANIFEST
    # The dev tree took this list from the private 3-exp worker as
    # `SHARED_FLUX + SHARED_XSEC + ORCA_DET`; that module does not ship. The
    # engine's own ORCA_MANIFEST is the SAME 30 dials in the XML's order rather
    # than the union-vector's (`tilt` sits 3rd instead of 5th). The benchmark is
    # permutation-invariant — the per-name nominal lookup below is unaffected and
    # the timing depends only on shapes — so this substitution changes no number.
    names = list(ORCA_MANIFEST)
    r = np.load(path, allow_pickle=True)
    n_bins = int(r["n_bins"])
    nE, nZ = int(r["n_etrue"]), int(r["n_cztrue"])
    rng = np.random.default_rng(12)
    phi = rng.uniform(0.4, 1.6, (2, 3, nE, nZ))
    mu = np.full(n_bins, 206.3429219139202 / n_bins)
    few = np.ones(n_bins, bool)
    # additive dials sit at 0; everything else at 1. E_shift is PINNED at 1.0 —
    # the engine asserts it, because the flat900 response encodes ENERGY_SCALE = 1
    # exactly and a moved E_shift would be silently ignored.
    nom = {"tilt": 0.0, "zenith_up": 0.0, "zenith_down": 0.0, "solar_activity": 0.0,
           "kpi_ratio": 0.0, "flux_horizvert": 0.0}
    th0 = np.array([nom.get(x, 1.0) for x in names], float)
    i_eshift = names.index("E_shift")
    e0 = ORCABinnedEngine(path, np.ones(n_bins), mu, few, names, norm=1.0)
    obs = e0.expectation(phi, th0)
    eng = ORCABinnedEngine(path, obs, mu, few, names, norm=1.0)
    thetas = [("nominal", th0)]
    for i in range(3):
        t = th0 + rng.normal(0, 0.02, len(names))
        t[i_eshift] = 1.0                       # stays pinned in every draw
        thetas.append((f"rand{i}", t))
    return eng, phi, thetas, len(names), eng.n_cell, int(r["R_v"].size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic-response", required=True,
                    help="ic_response_modeaxis_L3.npz — BUILD PRODUCT of "
                         "analysis/IC-binned-datafit/build_ic_binned_response.py")
    ap.add_argument("--orca-response", required=True,
                    help="orca_response_flat900.npz — BUILD PRODUCT of "
                         "analysis/ORCA-binned-datafit/build_orca_binned_response.py")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()

    print("=== binned-engine micro-benchmark (LOCAL cpu; ratios indicative) ===")
    print(f"    {a.n} timed calls per point, 20 warmup; numpy "
          f"{np.__version__}\n")
    rows = []
    for label, loader, path in (("IC", bench_ic, a.ic_response),
                                ("ORCA", bench_orca, a.orca_response)):
        eng, phi, thetas, n_dials, n_cell, n_entry = loader(path, a.n)
        print(f"--- {label}: {n_cell} cells, {n_entry} entries, {n_dials} dials ---")
        for tname, th in thetas:
            s_c = timeit(lambda: eng.chi2(phi, th), a.n)
            s_g = timeit(lambda: eng.chi2_and_grad(phi, th), a.n)
            rows.append((label, tname, n_dials, s_c, s_g))
            print(f"  {tname:8s}  chi2 {s_c['median']:7.3f} ms "
                  f"[p10 {s_c['p10']:6.3f}, p90 {s_c['p90']:6.3f}]   "
                  f"chi2_and_grad {s_g['median']:7.3f} ms "
                  f"[p10 {s_g['p10']:6.3f}, p90 {s_g['p90']:6.3f}]")
        # cheap breakout: the model build vs the adjoint on top of it
        th = thetas[0][1]
        s_w = timeit(lambda: eng.cell_weights(th) if label == "IC"
                     else eng.cell_weights(phi, th), a.n)
        s_e = timeit(lambda: eng.expectation(phi, th), a.n)
        med_c = [r for r in rows if r[0] == label and r[1] == "nominal"][0][3]["median"]
        med_g = [r for r in rows if r[0] == label and r[1] == "nominal"][0][4]["median"]
        print(f"  breakout (nominal): cell_weights {s_w['median']:.3f} ms | "
              f"expectation {s_e['median']:.3f} ms | "
              f"adjoint-only {(med_g - med_c):+.3f} ms (grad minus chi2)\n")

    print("=== TABLE: cost of ONE analytic gradient vs the event-path FD it replaces")
    print(f"    FASRC baselines: {FD_EVAL_S:.4f} s / FD model eval; "
          f"{FD_GRAD_S:.2f} s / {FD_N_EVAL}-eval gradient (stage 0, job 39678828)\n")
    hdr = (f"{'engine':6s} {'theta':8s} {'chi2 ms':>9s} {'grad ms':>9s} "
           f"{'grad / FD-eval':>15s} {'grad / FD-grad':>15s} {'speedup x':>10s}")
    print(hdr); print("-" * len(hdr))
    flags = []
    for label, tname, nd, s_c, s_g in rows:
        g_s = s_g["median"] / 1e3
        eq_eval = g_s / FD_EVAL_S
        eq_grad = g_s / FD_GRAD_S
        print(f"{label:6s} {tname:8s} {s_c['median']:9.3f} {s_g['median']:9.3f} "
              f"{eq_eval:15.5f} {eq_grad:15.7f} {1.0/eq_grad:10.0f}")
        if g_s >= FD_EVAL_S:
            flags.append(f"{label}/{tname}: gradient {g_s:.4f} s >= one FD eval "
                         f"{FD_EVAL_S:.4f} s")
    print()
    if flags:
        print("*** BOUND VIOLATED — the analytic gradient must cost far LESS than a")
        print("    single event-path FD evaluation; something is wrong:")
        for f in flags:
            print(f"      {f}")
    else:
        print("BOUND OK: every analytic gradient costs << one event-path FD model")
        print("evaluation (0.3431 s), which is the minimum the port had to achieve.")
    print("\nCAVEAT: local CPU != cluster CPU. Treat absolute ms as indicative and")
    print("the ratio as approximate; the definitive number is the G-E2E fd_stats line.")


if __name__ == "__main__":
    main()
