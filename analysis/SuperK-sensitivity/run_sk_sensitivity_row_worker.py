#!/usr/bin/env python3
"""
Super-Kamiokande 2023 Standard Sensitivity Scan Row Worker (for SLURM array jobs)

Each worker handles one row of the 2D grid:
- Fixed Dm231 value (determined by --row-idx)
- Scans all Sin2Theta23 values
- Saves results to a per-row JSON file

Uses analytical gradient for L-BFGS-B minimization.
Warm starting: uses previous point's best-fit nuisance as x0 for the next.

Grid: Dm231 x Sin2Theta23

Usage:
    python run_sk_sensitivity_row_worker.py --row-idx 0 --n-dm 41 --n-s23 41 --output-dir DIR
"""

import sys
import os
import argparse
import numpy as np
import copy
import json
from datetime import datetime
from scipy.optimize import minimize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYNU_DIR = os.path.join(SCRIPT_DIR, "..", "..")
PYNU_DIR = os.path.abspath(PYNU_DIR)
if PYNU_DIR not in sys.path:
    sys.path.insert(0, PYNU_DIR)

from pynu import PyNuFit

# Grid parameters
DM_MIN = 2.0e-3
DM_MAX = 3.0e-3
S23_MIN = 0.40
S23_MAX = 0.65

# Truth values (Asimov)
TRUTH_DM = 2.511e-3
TRUTH_S23 = 0.572


def setup_pynufit(config):
    """Initialize PyNuFit with Asimov observation at truth parameters."""
    pynufit = PyNuFit(config, verbosity=False)

    # Set truth parameters
    for name, pt in pynufit.physics_tunes.items():
        pt.OscillationTunes.Parameters["Sin2Theta23"] = TRUTH_S23
        pt.OscillationTunes.Parameters["Dm231"] = TRUTH_DM
        # Set Dm231_bar = Dm231 for CPT symmetry (no-op in 3-Osc mode)
        if "Dm231_bar" in pt.OscillationTunes.Parameters:
            pt.OscillationTunes.Parameters["Dm231_bar"] = TRUTH_DM
        if hasattr(pt.OscillationTunes, 'reset_cache'):
            pt.OscillationTunes.reset_cache()

    # Generate expectation at truth
    pynufit.StartPhysics()
    pynufit.StartNuisance()
    pynufit.ApplyOscillations("Physics")
    pynufit.ApplyNuisanceWeights(pynufit.Analysis.NuisNominalList)
    pynufit.SetExpectedWeights()
    pynufit.SetBinnedExpectedEvents()
    pynufit.SetBinnedMCVariance()
    pynufit.SetMuonBackground()

    # Build Asimov = neutrino expectation + muon background (if any)
    asimov = copy.deepcopy(pynufit.Expectation)
    for exp_name in asimov:
        if exp_name in pynufit.MuonBackground and pynufit.MuonBackground[exp_name] is not None:
            muon_counts, _ = pynufit.MuonBackground[exp_name]
            asimov[exp_name] = asimov[exp_name] + muon_counts

    # Set up likelihood
    pynufit.set_likelihood("BarlowBeestonLikelihood")
    for exp_name in asimov:
        pynufit.LLH.observation[exp_name] = asimov[exp_name]
    pynufit.LLH.set_muon_background(pynufit.MuonBackground)
    pynufit.LLH.set_mc_variance(pynufit.MCVariance)

    return pynufit


def run_one_point(pynufit, dm231, sin2theta23, x0, sigma, bounds):
    """Run minimization at a single (Dm231, Sin2Theta23) grid point.

    Uses analytical gradient for L-BFGS-B.
    x0: initial guess for nuisance parameters (warm starting).
    """
    pynufit.StartPhysics()
    pynufit.StartNuisance()

    for name, pt in pynufit.physics_tunes.items():
        pt.OscillationTunes.Parameters["Dm231"] = dm231
        if "Dm231_bar" in pt.OscillationTunes.Parameters:
            pt.OscillationTunes.Parameters["Dm231_bar"] = dm231
        pt.OscillationTunes.Parameters["Sin2Theta23"] = sin2theta23
        if hasattr(pt.OscillationTunes, 'reset_cache'):
            pt.OscillationTunes.reset_cache()

    pynufit.ApplyOscillations("Physics")
    pynufit.ApplyNuisanceWeights(x0)
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

    chi2_stat = pynufit.LLH.stats_only(pynufit.Expectation, pynufit.MCVariance)
    tol_adj = max(1e-5, np.sqrt(max(chi2_stat, 0)) * 1e-5)

    result = minimize(
        objective, x0,
        method='L-BFGS-B', jac=gradient, bounds=bounds,
        options={'ftol': tol_adj, 'gtol': 1e-5, 'maxiter': 200}
    )

    return result.fun, result.x, result.nit, result.success


def main():
    parser = argparse.ArgumentParser(description="SK sensitivity scan row worker")
    parser.add_argument("--row-idx", type=int, required=True, help="Row index (Dm231 index)")
    parser.add_argument("--n-dm", type=int, default=41, help="Number of Dm231 grid points")
    parser.add_argument("--n-s23", type=int, default=41, help="Number of Sin2Theta23 grid points")
    parser.add_argument("--output-dir", required=True, help="Output directory for row results")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    PROJECT_DIR = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit'
    if args.config is None:
        args.config = f"{PROJECT_DIR}/Pynu/examples/AnalysisFiles/SK2023_Atm.xml"

    dm_grid = np.linspace(DM_MIN, DM_MAX, args.n_dm)
    s23_grid = np.linspace(S23_MIN, S23_MAX, args.n_s23)
    dm231 = dm_grid[args.row_idx]

    print(f"[Worker {args.row_idx}] Dm231={dm231:.5e}, scanning {args.n_s23} Sin2Theta23 points")
    print(f"  Sin2Theta23 range: [{S23_MIN}, {S23_MAX}]")
    print(f"  Started: {datetime.now().isoformat()}")

    # Initialize
    pynufit = setup_pynufit(args.config)

    # Set up minimization bounds
    nominal = np.array(pynufit.Analysis.NuisNominalList)
    sigma = np.array(pynufit.Analysis.NuisSigmaList)
    lower = nominal - 5 * sigma
    upper = nominal + 5 * sigma
    for k in range(len(lower)):
        if nominal[k] > 0 and lower[k] < 0.01:
            lower[k] = 0.01
    bounds = list(zip(lower, upper))

    # Run all Sin2Theta23 points for this Dm231
    # Warm starting: carry over previous best-fit nuisance
    results = []
    x0 = nominal.copy()
    for j, sin2theta23 in enumerate(s23_grid):
        chi2, nuisance, nit, converged = run_one_point(
            pynufit, dm231, sin2theta23, x0, sigma, bounds
        )
        # Warm start: use this result as starting point for next
        x0 = nuisance.copy()

        pull_max = np.max(np.abs((nuisance - nominal) / sigma))
        results.append({
            'i': args.row_idx, 'j': j,
            'dm231': float(dm231), 'sin2theta23': float(sin2theta23),
            'chi2': float(chi2), 'nit': int(nit),
            'converged': bool(converged), 'max_pull': float(pull_max),
            'nuisance': nuisance.tolist()
        })
        print(f"  [{args.row_idx},{j:2d}] s23={sin2theta23:.4f}: chi2={chi2:8.4f}, "
              f"iter={nit:3d}, conv={converged}, pull={pull_max:.3f}")

    # Save row results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"row_{args.row_idx:03d}.json")
    with open(out_path, 'w') as f:
        json.dump({
            'row_idx': args.row_idx,
            'dm231': float(dm231),
            'n_dm': args.n_dm,
            'n_s23': args.n_s23,
            'dm_range': [float(DM_MIN), float(DM_MAX)],
            's23_range': [float(S23_MIN), float(S23_MAX)],
            'truth_dm': float(TRUTH_DM),
            'truth_s23': float(TRUTH_S23),
            'nuisance_names': pynufit.Analysis.NuisanceList,
            'points': results
        }, f, indent=2)

    n_conv = sum(1 for r in results if r['converged'])
    print(f"  Convergence: {n_conv}/{len(results)}")
    print(f"  Saved: {out_path}")
    print(f"  Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
