"""G-IC-3 — `ICBinnedEngine` vs the PRODUCTION IC term at NOMINAL dials, 5 cells.

FASRC-ONLY (constructing the IC experiment imports nuflux, `ICDeepCore.py:7`).

WHY NOMINAL DIALS IS THE RIGHT PROBE, and not a weaker one. At nominal every IC
dial is an exact no-op (tilt x=0, AxialMass x=1, kpi_ratio x=0, ...), so the
per-event vs per-cell dial distinction VANISHES and the dial-side residue G-IC-4
measures drops out of the comparison entirely. What remains is exactly the thing
this gate exists to test: the φ convention, the NC treatment, NORM, the muon add,
the HS correction, and the response contraction. A discrepancy here is a real bug,
never "binning".

Expected agreement is SUMMATION-ORDER, not merely close. Both sides sum the same
per-event weights against the same cell-constant φ; the engine just sums them into
cells first. So the target is ~1e-12 relative, and the gate says so rather than
hiding behind a loose tolerance.

THE FAILURE IT IS DESIGNED TO CATCH: a factor-of-flux error on the NC classes
(design §5.1). φ is the oscillated FLUX (flux x P), NOT a bare probability, and
there is NO SK-style NC override on IC. Either mistake is O(1) on the NC block and
shows up here immediately.

    python test/binned_icorca/gate_ic_identity.py \
        --config <IC_DeepCore_r2_fude_ccqe.xml> \
        --response <ic_response_modeaxis_L3.npz> --phi <ic_phi_L3.npz> \
        --out gate_ic_identity_L3.json
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
from pynu.Experiments.ic_binned_engine import ICBinnedEngine, N_BINS   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="IC analysis XML (IC_DeepCore_r2_fude_ccqe.xml); no "
                         "repo-relative default — it is not a repo file")
    ap.add_argument("--mc", default=os.path.join(
        _REPO, "data", "IceCube", "IC_MC.parquet"),
        help="IC MC parquet (default: data/IceCube/IC_MC.parquet)")
    ap.add_argument("--data", default=os.path.join(
        _REPO, "data", "IceCube", "IC_data.parquet"),
        help="IC data parquet (default: data/IceCube/IC_data.parquet)")
    ap.add_argument("--data-dir", default=os.path.join(_REPO, "data", "IceCube"),
                    help="dir holding the reco bin-edge .npy files "
                         "(default: data/IceCube)")
    ap.add_argument("--response", required=True,
                    help="ic_response_modeaxis_L3.npz — BUILD PRODUCT of "
                         "analysis/IC-binned-datafit/build_ic_binned_response.py")
    ap.add_argument("--phi", required=True,
                    help="ic_phi_L3.npz — BUILD PRODUCT of "
                         "analysis/IC-binned-datafit/build_ic_osc_tensors.py")
    ap.add_argument("--hs-dir", default=os.path.join(_REPO, "data", "IceCube"),
                    help="dir holding the three hypersurface CSVs "
                         "(default: data/IceCube)")
    ap.add_argument("--pynu-root", default=_REPO,
                    help="repo root providing `pynu` (default: this clone)")
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--out", default="gate_ic_identity.json")
    a = ap.parse_args()

    _root = os.path.abspath(a.pynu_root)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from pynu.Experiments import ic_binned_support as H
    from pynu import PyNuFit
    from pynu.Experiments.ICDeepCore import ICDeepCore_Atm

    print("=== GATE G-IC-3 — engine vs production, NOMINAL dials ===")
    fit = PyNuFit(a.config, verbosity=False)
    exp_name = list(fit.Experiments.keys())[0]
    exp = fit.Experiments[exp_name]
    osc = fit.physics_tunes[exp_name].OscillationTunes
    if isinstance(exp, ICDeepCore_Atm):
        exp.load_hypersurfaces(a.hs_dir)
    names = list(fit.Analysis.NuisanceList)
    nominal = np.array(fit.Analysis.NuisNominalList, float)

    obs = H.observed_200(a.data, a.data_dir)
    mu200 = H.muon_200(a.mc, a.data_dir)
    _b, ie, iz, ntype, flavor = H.nu_index(a.mc, a.data_dir, a.response)

    # NORM: response stores RAW weight, NORM applied at scan time
    # (ICDeepCore.py:190-191). Recover it from the live experiment rather than
    # recomputing it, so a config change cannot silently desync the two sides.
    base = np.asarray(exp.BaseWeight, float)
    raw = np.asarray(exp.Weight, float) if hasattr(exp, "Weight") else None
    norm = float(np.median(base / raw)) if raw is not None else float(exp.NORM)
    print(f"    NORM = {norm!r}")

    phi = np.load(a.phi, allow_pickle=True)["phi"]
    osc.osc_avg_scale = None
    eng = ICBinnedEngine(a.response, obs, mu200, names, norm=norm,
                         hs_slopes=None)      # HS slopes set per point below
    print(f"    {eng.summary()['grid']}: {eng.summary()['n_cell']} cells, "
          f"{len(names)} dials")

    rows, worst = [], 0.0
    for ip, (dm, s23) in enumerate(H.POINTS):
        # production reference: cell-phi, nominal dials, same HS + muon
        phys_cell = np.asarray(phi[ip][ntype, flavor, ie, iz], float)
        fit.StartNuisance()
        fit.ApplyNuisanceWeights(nominal)
        nuis = np.asarray(exp.NuisanceWeight, float)
        hs_params = H._hs_params_from_theta(nominal, names, exp.HS_SLOPE_NAMES)
        corr = H._hs_correction_factor(exp, dm, hs_params)
        n_prod = H._corrected_expectation(exp, phys_cell * nuis, corr, mu200)
        c_prod = H.poisson_chi2(obs[eng.few], n_prod[eng.few])

        # engine: same HS slopes, same nominal theta
        eng.hs_slopes = exp.interpolate_hs(dm)
        n_eng = eng.expectation(phi[ip], nominal)
        c_eng = eng.chi2(phi[ip], nominal)

        d_bin = np.max(np.abs(n_eng - n_prod)
                       / np.where(n_prod > 0, n_prod, np.nan))
        d_chi2 = abs(c_eng - c_prod) / max(1.0, abs(c_prod))
        worst = max(worst, float(np.nanmax([d_bin, d_chi2])))
        rows.append({"point": ip, "dm31": dm, "s23": s23,
                     "chi2_production": c_prod, "chi2_engine": c_eng,
                     "rel_chi2": float(d_chi2),
                     "max_rel_per_bin": float(d_bin)})
        print(f"  pt{ip} dm={dm:.4e} s23={s23:.3f}: chi2 prod={c_prod:.6f} "
              f"eng={c_eng:.6f}  rel={d_chi2:.3e}  max/bin={d_bin:.3e}")

    ok = worst <= a.tol
    print(f"GATE G-IC-3: {'PASS' if ok else 'FAIL'} worst rel = {worst:.3e} "
          f"(threshold {a.tol:g}, summation-order expected)")
    with open(a.out, "w") as f:
        json.dump({"gate": "G-IC-3", "pass": bool(ok), "worst_rel": worst,
                   "tol": a.tol, "norm": norm,
                   "response": os.path.abspath(a.response),
                   "phi": os.path.abspath(a.phi), "rows": rows}, f, indent=2)
    print(f"  wrote {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
