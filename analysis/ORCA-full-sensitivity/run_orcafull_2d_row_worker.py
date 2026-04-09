#!/usr/bin/env python3
"""
ORCA-Full 2D Sensitivity Scan — Row Worker

Scans one Dm231 row over Sin2Theta23, minimizing over nuisance parameters.
Works for both matrix-based and event-by-event MC (set via --config).

Usage:
    python run_orcafull_2d_row_worker.py --row-idx 0 --config CONFIG --output-dir DIR
"""

import sys
import os
import json
import numpy as np
import copy
from datetime import datetime
from scipy.optimize import minimize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYNU_DIR = os.path.join(SCRIPT_DIR, "..", "..")
PYNU_DIR = os.path.abspath(PYNU_DIR)
if PYNU_DIR not in sys.path:
    sys.path.insert(0, PYNU_DIR)

from pynu import PyNuFit


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--row-idx", type=int, required=True)
    parser.add_argument("--n-dm", type=int, default=41)
    parser.add_argument("--n-s23", type=int, default=41)
    parser.add_argument("--dm-min", type=float, default=2.0e-3)
    parser.add_argument("--dm-max", type=float, default=3.0e-3)
    parser.add_argument("--s23-min", type=float, default=0.40)
    parser.add_argument("--s23-max", type=float, default=0.65)
    parser.add_argument("--exposure", type=float, default=5.0)
    parser.add_argument("--no-mc-variance", action="store_true",
                        help="Disable BB MC variance (standard Poisson)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dm_grid = np.linspace(args.dm_min, args.dm_max, args.n_dm)
    s23_grid = np.linspace(args.s23_min, args.s23_max, args.n_s23)

    row_idx = args.row_idx
    dm_val = dm_grid[row_idx]

    print(f"=== 2D Row Worker: row {row_idx}/{args.n_dm}, Dm231={dm_val:.4e} ===")
    print(f"S23: {args.n_s23} pts in [{args.s23_min}, {args.s23_max}]")
    print(f"Config: {args.config}")
    print(f"MC variance: {'disabled (Poisson)' if args.no_mc_variance else 'enabled (BB)'}")
    print(f"Started: {datetime.now().isoformat()}")

    # Initialize
    pynufit = PyNuFit(args.config, verbosity=False)

    # Override exposure
    for exp_name, exp in pynufit.Experiments.items():
        mc_exp = exp.TotalMCexposure
        exp.FitExposure = args.exposure
        exp.NORM = (args.exposure / mc_exp) * 1e4 * exp.SECONDS_PER_YEAR
        exp.BaseWeight = exp.Weight * exp.NORM
        print(f"  {exp_name}: NORM={exp.NORM:.3e} ({args.exposure}yr / {mc_exp}yr MC)")

    # Generate Asimov at truth
    pynufit.StartPhysics()
    pynufit.StartNuisance()
    pynufit.ApplyOscillations("Physics")
    pynufit.ApplyNuisanceWeights(pynufit.Analysis.NuisNominalList)
    pynufit.SetExpectedWeights()
    pynufit.SetBinnedExpectedEvents()
    pynufit.SetBinnedMCVariance()
    pynufit.SetMuonBackground()

    asimov = copy.deepcopy(pynufit.Expectation)
    for k, v in asimov.items():
        if k in pynufit.MuonBackground and pynufit.MuonBackground[k] is not None:
            mu_counts, _ = pynufit.MuonBackground[k]
            asimov[k] = v + mu_counts
    total_events = sum(v.sum() for v in asimov.values())
    print(f"  Total Asimov events: {total_events:.1f}")

    # Likelihood
    pynufit.set_likelihood("BarlowBeestonLikelihood")
    for k in asimov:
        pynufit.LLH.observation[k] = asimov[k]
    pynufit.LLH.set_muon_background(pynufit.MuonBackground)
    if not args.no_mc_variance:
        pynufit.LLH.set_mc_variance(pynufit.MCVariance)

    # Nuisance setup
    nominal = np.array(pynufit.Analysis.NuisNominalList)
    sigma = np.array(pynufit.Analysis.NuisSigmaList)
    lower = nominal - 5 * sigma
    upper = nominal + 5 * sigma
    for k in range(len(lower)):
        if nominal[k] > 0 and lower[k] < 0.01:
            lower[k] = 0.01
    bounds = list(zip(lower, upper))

    mc_var = None if args.no_mc_variance else pynufit.MCVariance

    chi2_row = np.full(args.n_s23, np.nan)
    converged_row = np.zeros(args.n_s23, dtype=bool)
    x0 = nominal.copy()

    t_start = datetime.now()

    for j, s23 in enumerate(s23_grid):
        pynufit.StartPhysics()
        pynufit.StartNuisance()

        for name, pt in pynufit.physics_tunes.items():
            pt.OscillationTunes.Parameters["Dm231"] = dm_val
            pt.OscillationTunes.Parameters["Sin2Theta23"] = s23
            if hasattr(pt.OscillationTunes, 'reset_cache'):
                pt.OscillationTunes.reset_cache()

        pynufit.ApplyOscillations("Physics")
        pynufit.ApplyNuisanceWeights(nominal)
        pynufit.SetExpectedWeights()
        pynufit.SetBinnedExpectedEvents()
        if not args.no_mc_variance:
            pynufit.SetBinnedMCVariance()
            mc_var = pynufit.MCVariance
        pynufit.ComputeBinnedDiffExpectation()

        def objective(nuisance):
            pynufit.StartNuisance()
            pynufit.ApplyNuisanceWeights(nuisance)
            pynufit.SetExpectedWeights()
            pynufit.SetBinnedExpectedEvents()
            return pynufit.LLH.stats_and_systematics(
                pynufit.Expectation, nuisance, mc_var
            )

        def gradient(nuisance):
            return pynufit.LLH.gradient(
                pynufit.Expectation, pynufit.DiffExpectation, nuisance, mc_var
            )

        result = minimize(
            objective, x0,
            method='L-BFGS-B', jac=gradient, bounds=bounds,
            options={'ftol': 1e-5, 'gtol': 1e-5, 'maxiter': 200}
        )

        chi2_row[j] = result.fun
        converged_row[j] = result.success
        if result.success:
            x0 = result.x.copy()

    total_time = (datetime.now() - t_start).total_seconds()

    row_data = {
        'row_idx': row_idx,
        'dm231': float(dm_val),
        'chi2': chi2_row.tolist(),
        'converged': converged_row.tolist(),
        'total_time_s': total_time,
    }

    row_file = os.path.join(args.output_dir, f'row_{row_idx:03d}.json')
    with open(row_file, 'w') as f:
        json.dump(row_data, f, indent=2)

    print(f"Row {row_idx} done: min_chi2={np.nanmin(chi2_row):.4f}, "
          f"conv={converged_row.sum()}/{args.n_s23}, time={total_time:.1f}s")
    print(f"Saved: {row_file}")


if __name__ == "__main__":
    main()
