#!/usr/bin/env python3
"""
ORCA-Full 4D Sensitivity Scan — Row Worker

Scans Dm231 x Sin2Theta23, profiling (minimizing) over Sin2Theta13 and dCP grids,
plus nuisance parameters.

Each SLURM array task handles one Dm231 row.
Works for both matrix-based and event-by-event MC (set via --config).

Usage:
    python run_orcafull_4d_row_worker.py --row-idx 0 --config CONFIG --output-dir DIR
"""

import sys
import os
import json
import numpy as np
import copy
from datetime import datetime
from scipy.optimize import minimize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
PYNU_PARENT = os.path.join(BASE_DIR, 'Pynu')
if PYNU_PARENT not in sys.path:
    sys.path.insert(0, PYNU_PARENT)

from pynu import PyNuFit


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--row-idx", type=int, required=True)
    # Dm231 x S23 grid
    parser.add_argument("--n-dm", type=int, default=41)
    parser.add_argument("--n-s23", type=int, default=41)
    parser.add_argument("--dm-min", type=float, default=2.0e-3)
    parser.add_argument("--dm-max", type=float, default=3.0e-3)
    parser.add_argument("--s23-min", type=float, default=0.40)
    parser.add_argument("--s23-max", type=float, default=0.65)
    # Profiling grids
    parser.add_argument("--n-theta13", type=int, default=11)
    parser.add_argument("--theta13-min", type=float, default=0.018)
    parser.add_argument("--theta13-max", type=float, default=0.026)
    parser.add_argument("--n-dcp", type=int, default=12)
    parser.add_argument("--dcp-min", type=float, default=0.0)
    parser.add_argument("--dcp-max", type=float, default=5.8853)  # ~2pi - step/2
    # Exposure
    parser.add_argument("--exposure", type=float, default=5.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    TRUTH_DM = 2.511e-3
    TRUTH_S23 = 0.572

    dm_grid = np.linspace(args.dm_min, args.dm_max, args.n_dm)
    s23_grid = np.linspace(args.s23_min, args.s23_max, args.n_s23)
    theta13_grid = np.linspace(args.theta13_min, args.theta13_max, args.n_theta13)
    dcp_grid = np.linspace(args.dcp_min, args.dcp_max, args.n_dcp)

    row_idx = args.row_idx
    dm_val = dm_grid[row_idx]

    print(f"=== 4D Row Worker: row {row_idx}/{args.n_dm}, Dm231={dm_val:.4e} ===")
    print(f"S23: {args.n_s23} pts, Theta13: {args.n_theta13} pts, dCP: {args.n_dcp} pts")
    print(f"Total evals this row: {args.n_s23 * args.n_theta13 * args.n_dcp}")
    print(f"Config: {args.config}")
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

    # Output arrays: profiled chi2 (min over theta13, dcp) for each s23
    chi2_profiled = np.full(args.n_s23, np.nan)
    converged_all = np.zeros(args.n_s23, dtype=bool)
    best_theta13 = np.full(args.n_s23, np.nan)
    best_dcp = np.full(args.n_s23, np.nan)

    # Also save the full 3D chi2 for this row: (n_s23, n_theta13, n_dcp)
    chi2_full = np.full((args.n_s23, args.n_theta13, args.n_dcp), np.nan)

    x0 = nominal.copy()
    t_start = datetime.now()

    for j, s23 in enumerate(s23_grid):
        best_chi2_this_s23 = np.inf
        any_converged = False

        for it, theta13 in enumerate(theta13_grid):
            for id, dcp in enumerate(dcp_grid):
                # Set physics
                pynufit.StartPhysics()
                pynufit.StartNuisance()

                for name, pt in pynufit.physics_tunes.items():
                    pt.OscillationTunes.Parameters["Dm231"] = dm_val
                    pt.OscillationTunes.Parameters["Sin2Theta23"] = s23
                    pt.OscillationTunes.Parameters["Sin2Theta13"] = theta13
                    pt.OscillationTunes.Parameters["dCP"] = dcp
                    if hasattr(pt.OscillationTunes, 'reset_cache'):
                        pt.OscillationTunes.reset_cache()

                pynufit.ApplyOscillations("Physics")
                pynufit.ApplyNuisanceWeights(nominal)
                pynufit.SetExpectedWeights()
                pynufit.SetBinnedExpectedEvents()
                pynufit.SetBinnedMCVariance()
                pynufit.ComputeBinnedDiffExpectation()

                def objective(nuisance):
                    pynufit.StartNuisance()
                    pynufit.ApplyNuisanceWeights(nuisance)
                    pynufit.SetExpectedWeights()
                    pynufit.SetBinnedExpectedEvents()
                    return pynufit.LLH.stats_and_systematics(
                        pynufit.Expectation, nuisance, pynufit.MCVariance
                    )

                def gradient(nuisance):
                    return pynufit.LLH.gradient(
                        pynufit.Expectation, pynufit.DiffExpectation, nuisance, pynufit.MCVariance
                    )

                result = minimize(
                    objective, x0,
                    method='L-BFGS-B', jac=gradient, bounds=bounds,
                    options={'ftol': 1e-5, 'gtol': 1e-5, 'maxiter': 200}
                )

                chi2_full[j, it, id] = result.fun

                if result.success:
                    x0 = result.x.copy()
                    any_converged = True

                if result.fun < best_chi2_this_s23:
                    best_chi2_this_s23 = result.fun
                    best_theta13[j] = theta13
                    best_dcp[j] = dcp

        chi2_profiled[j] = best_chi2_this_s23
        converged_all[j] = any_converged

        elapsed = (datetime.now() - t_start).total_seconds()
        eta = elapsed / (j + 1) * (args.n_s23 - j - 1)
        print(f"  S23={s23:.4f}: profiled_chi2={best_chi2_this_s23:.4f}, "
              f"best_th13={best_theta13[j]:.4f}, best_dcp={best_dcp[j]:.3f}, "
              f"ETA={eta:.0f}s")

    total_time = (datetime.now() - t_start).total_seconds()

    # Save row results
    row_data = {
        'row_idx': row_idx,
        'dm231': float(dm_val),
        'chi2_profiled': chi2_profiled.tolist(),
        'converged': converged_all.tolist(),
        'best_theta13': best_theta13.tolist(),
        'best_dcp': best_dcp.tolist(),
        'total_time_s': total_time,
    }

    row_file = os.path.join(args.output_dir, f'row_{row_idx:03d}.json')
    with open(row_file, 'w') as f:
        json.dump(row_data, f, indent=2)

    # Also save the full 3D chi2 for this row
    np.save(os.path.join(args.output_dir, f'chi2_full_row_{row_idx:03d}.npy'), chi2_full)

    print(f"\nRow {row_idx} complete: min_profiled={np.nanmin(chi2_profiled):.4f}, "
          f"time={total_time:.1f}s")
    print(f"Saved: {row_file}")


if __name__ == "__main__":
    main()
