"""Gates for `ICBinnedEngine` — G-G1 (per-dial gradient), G-G2 (zero invariants),
plus construction checks. Runs LOCALLY: the engine takes its HS slopes as data,
so no pynu experiment and no nuflux are needed.

G-G1 uses a RICHARDSON-EXTRAPOLATED central difference, not a plain one. The ORCA
track's lesson applies here unchanged: a plain central difference on a χ² of order
1e3 has truncation error O(h²·f''') and roundoff O(eps·f/h), and at a single fixed
h the two together can sit near 1e-6 relative — indistinguishable from a genuinely
wrong derivative. Richardson, D = (4·D(h/2) − D(h))/3, is O(h⁴) and moves the FD
noise floor far below the gate threshold, so a failure means the ANALYTIC side is
wrong rather than the reference being noisy.

    python test/binned_icorca/gate_ic_engine.py \
        --response <ic_response_modeaxis_L3.npz>
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = str(Path(__file__).resolve().parents[2])   # repo root: test/binned_icorca/..
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)
from pynu.Experiments.ic_binned_engine import (                             # noqa: E402
    ICBinnedEngine, HS_CATEGORIES, HS_NOMINALS, HS_DIALS, N_BINS, BARRIER_CHI2)
from pynu.Experiments import ic_dial_fields as icf                          # noqa: E402
from pynu.Experiments import binned_dial_fields as bdf                      # noqa: E402

# IC manifest order (IC_DeepCore_r2_fude_ccqe.xml; 39 dials).
FLUX = ["normalization_below1GeV", "normalization_above1GeV", "nunubar_ratio",
        "flavor_ratio", "tilt", "zenith_up", "zenith_down", "solar_activity",
        "kpi_ratio", "flux_horizvert",
        "flux_nuebar_subgev", "flux_nuebar_mid", "flux_nuebar_high",
        "flux_flavor_subgev", "flux_flavor_mid", "flux_flavor_high",
        "flux_numubar_subgev", "flux_numubar_mid", "flux_numubar_high"]
XSEC = ["XSecNuTau", "NCoverCC", "NCHad", "AxialMass",
        "DIS", "CCQE", "CCQENuBarNu", "CCQEMuE", "CC1PiProduction",
        "CC1Pi_NuBarNuE", "CC1Pi_NuBarNuMu", "xsec_ccqe_shape",
        "xsec_ccqe_shape_subgev", "xsec_ccqe_multigev_nue",
        "xsec_ccqe_multigev_numu"]
NAMES = FLUX + XSEC + HS_DIALS
NOMINAL = {"nunubar_ratio": 1.0, "flavor_ratio": 1.0, "normalization_below1GeV": 1.0,
           "normalization_above1GeV": 1.0, "tilt": 0.0, "zenith_up": 0.0,
           "zenith_down": 0.0, "solar_activity": 0.0, "kpi_ratio": 0.0,
           "flux_horizvert": 0.0, "XSecNuTau": 1.0, "NCoverCC": 1.0, "NCHad": 1.0,
           "AxialMass": 1.0, "xsec_ccqe_shape_subgev": 0.0}
INERT_SUBGEV = ("flux_nuebar_subgev", "flux_flavor_subgev", "flux_numubar_subgev")


def nominal_theta():
    t = np.array([NOMINAL.get(n, 1.0) for n in NAMES], float)
    for i, n in enumerate(NAMES):
        if n in HS_NOMINALS:
            t[i] = HS_NOMINALS[n]
    return t


def synth_inputs(eng, seed=20260817):
    """Synthetic but STRUCTURALLY faithful phi / obs / HS slopes."""
    rng = np.random.default_rng(seed)
    c = eng.cells
    phi = rng.uniform(0.4, 1.6, (2, 3, c.nE, c.nZ))
    hs = {cat: {"intercept": np.full(N_BINS, 1.0)} for cat in HS_CATEGORIES}
    for cat in HS_CATEGORIES:
        for d in HS_DIALS:
            hs[cat][d] = rng.normal(0.0, 0.05, N_BINS)
    return phi, hs


def richardson_grad(f, x0, i, h):
    def D(step):
        xp, xm = x0.copy(), x0.copy()
        xp[i] += step
        xm[i] -= step
        return (f(xp) - f(xm)) / (2.0 * step)
    return (4.0 * D(h / 2.0) - D(h)) / 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--response", required=True,
                    help="ic_response_modeaxis_L3.npz — BUILD PRODUCT of "
                         "analysis/IC-binned-datafit/build_ic_binned_response.py")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    print("=== ICBinnedEngine gates (G-G1 / G-G2 / construction) ===")
    # bootstrap: build once with unit obs to get the cell structure, then make a
    # self-consistent observed vector from the nominal model (chi2 ~ 0 at truth).
    boot = ICBinnedEngine(a.response, np.ones(N_BINS), np.zeros(N_BINS),
                          NAMES, norm=1.0)
    phi, hs = synth_inputs(boot)
    mu = np.full(N_BINS, 512.166 / N_BINS)
    th0 = nominal_theta()
    e0 = ICBinnedEngine(a.response, np.ones(N_BINS), mu, NAMES, norm=1.0,
                        hs_slopes=hs)
    obs = e0.expectation(phi, th0)
    eng = ICBinnedEngine(a.response, obs, mu, NAMES, norm=1.0, hs_slopes=hs)
    s = eng.summary()
    print(f"  {s['grid']}: {s['n_cell']} cells, {s['n_entry']} entries, "
          f"{s['n_dials']} dials ({s['n_cell_dials']} cell + {len(s['hs_dials'])} HS "
          f"+ {len(s['pinned'])} pinned), few={s['few_bins']}/200")
    ok = True

    # --- construction ------------------------------------------------------
    c0 = eng.chi2(phi, th0)
    print(f"GATE construct-selfconsistency: {'PASS' if abs(c0) < 1e-6 else 'FAIL'} "
          f"chi2(obs==model) = {c0:.3e}")
    ok &= abs(c0) < 1e-6
    mu_ok = abs(s["ccqe_shape_subgev_mu"] - icf.CCQE_SHAPE_SUBGEV_MU) <= 1e-12
    print(f"GATE mu-identity: {'PASS' if mu_ok else 'FAIL'} "
          f"{s['ccqe_shape_subgev_mu']!r} (edges NOT injected)")
    ok &= mu_ok

    # --- G-G1 : per-dial analytic vs Richardson FD --------------------------
    rng = np.random.default_rng(7)
    worst, worst_name, rows = 0.0, "", []
    thetas = [th0] + [th0 + rng.normal(0, 0.02, len(NAMES)) for _ in range(2)]
    for ti, th in enumerate(thetas):
        chi2, g = eng.chi2_and_grad(phi, th)
        for i, n in enumerate(NAMES):
            if n in eng.pinned:
                continue
            scale = max(abs(th[i]), 1.0)
            fd = richardson_grad(lambda t: eng.chi2(phi, t), th, i, 1e-4 * scale)
            rel = abs(g[i] - fd) / max(1.0, abs(fd))
            rows.append({"theta": ti, "dial": n, "analytic": float(g[i]),
                         "fd": float(fd), "rel": float(rel)})
            if rel > worst:
                worst, worst_name = rel, f"{n}@theta{ti}"
    ok_g1 = worst <= a.tol
    print(f"GATE G-G1 per-dial grad vs Richardson FD: {'PASS' if ok_g1 else 'FAIL'} "
          f"worst rel = {worst:.3e} at {worst_name} (threshold {a.tol:g}); "
          f"{len(rows)} dial-points over {len(thetas)} theta")
    if not ok_g1:
        for r in sorted(rows, key=lambda r: -r["rel"])[:6]:
            print(f"    {r['dial']:28s} theta{r['theta']} analytic={r['analytic']:+.6e} "
                  f"fd={r['fd']:+.6e} rel={r['rel']:.3e}")
    ok &= ok_g1

    # --- G-G2 : zero invariants --------------------------------------------
    _c, g = eng.chi2_and_grad(phi, th0)
    inert = {n: float(g[NAMES.index(n)]) for n in INERT_SUBGEV}
    inert_ok = all(v == 0.0 for v in inert.values())
    print(f"GATE G-G2 inert *_subgev bitwise zero: {'PASS' if inert_ok else 'FAIL'} "
          f"{inert}")
    ok &= inert_ok

    pin = {p: float(g[NAMES.index(p)]) for p in eng.pinned}
    pin_ok = all(v == 0.0 for v in pin.values())
    print(f"GATE G-G2 pinned slot present and zero: {'PASS' if pin_ok else 'FAIL'} "
          f"{pin} (slot exists in a {len(g)}-long vector)")
    ok &= pin_ok and len(g) == len(NAMES)

    # muon invariance: mu enters E additively, so the vector-Jacobian product must
    # be BITWISE independent of it even though chi2 is not.
    eng2 = ICBinnedEngine(a.response, obs, mu * 3.0, NAMES, norm=1.0, hs_slopes=hs)
    v = np.linspace(0.3, 1.7, N_BINS)
    j1 = eng.model_jacobian_dot(phi, th0, v)
    j2 = eng2.model_jacobian_dot(phi, th0, v)
    mu_inv = np.array_equal(j1, j2)
    print(f"GATE G-G2 muon zero-gradient (Jacobian bitwise mu-independent): "
          f"{'PASS' if mu_inv else 'FAIL'} {mu_inv}")
    ok &= mu_inv

    # --- barrier ------------------------------------------------------------
    thb = th0.copy()
    thb[NAMES.index("XSecNuTau")] = -5.0            # off-domain -> scalar collapse
    cb, gb = eng.chi2_and_grad(phi, thb)
    bar_ok = (cb == BARRIER_CHI2 and np.all(gb == 0.0)) or cb < BARRIER_CHI2
    print(f"GATE barrier: {'PASS' if bar_ok else 'FAIL'} chi2={cb:.6g} "
          f"(BARRIER={BARRIER_CHI2:g} with zero grad, or finite if the collapse "
          f"stays positive)")
    ok &= bar_ok

    print(f"IC ENGINE GATES: {'ALL PASS' if ok else 'FAIL'}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"all_pass": bool(ok), "worst_g1_rel": worst,
                       "worst_g1_dial": worst_name, "summary": s,
                       "inert": inert, "pinned": pin,
                       "muon_jacobian_invariant": bool(mu_inv),
                       "rows": rows}, f, indent=2)
        print(f"  wrote {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
