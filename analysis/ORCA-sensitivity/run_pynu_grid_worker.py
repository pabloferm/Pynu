#!/usr/bin/env python3
"""
Pynu Grid Scan Worker (for SLURM array jobs)

Each worker handles one row of the grid (all Dm231 values for one Sin2Theta23).
Properly includes muons in observation to match NTOA.

Usage:
    python run_pynu_grid_worker.py --row-idx 0 --n-theta 20 --n-dm 20 --output-dir DIR
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

# Grid matching NTOA sensitivity check (0116_sensitivity_check)
THETA_MIN, THETA_MAX = 0.3, 0.7
DM_MIN, DM_MAX = 0.0015, 0.003


def setup_pynufit(config):
    """Initialize PyNuFit and set up observation WITH muons (matching NTOA)."""
    pynufit = PyNuFit(config, verbosity=False)

    # Compute MC variance and muon background at nominal
    pynufit.SetBinnedMCVariance()
    pynufit.SetMuonBackground()

    # Build observation = neutrino expectation + muon background
    # This matches NTOA's BinEvents(include_muons=True)
    observation = copy.deepcopy(pynufit.Observation)
    n_muon = 0
    for exp_name in observation:
        if pynufit.MuonBackground[exp_name] is not None:
            muon_counts, _ = pynufit.MuonBackground[exp_name]
            observation[exp_name] = observation[exp_name] + muon_counts
            n_muon += np.sum(muon_counts)

    # Set up Barlow-Beeston likelihood with corrected observation
    pynufit.set_likelihood("BarlowBeestonLikelihood")
    for exp_name in observation:
        pynufit.LLH.observation[exp_name] = observation[exp_name]
    pynufit.LLH.set_muon_background(pynufit.MuonBackground)
    pynufit.LLH.set_mc_variance(pynufit.MCVariance)

    return pynufit, n_muon


def run_one_point(pynufit, theta, dm, nominal, sigma, bounds):
    """Run minimization at a single grid point. Returns (chi2, nuisance, nit, converged)."""
    # Reset weights
    pynufit.StartPhysics()
    pynufit.StartNuisance()

    # Set oscillation parameters
    for name, pt in pynufit.physics_tunes.items():
        pt.OscillationTunes.Parameters["Sin2Theta23"] = theta
        pt.OscillationTunes.Parameters["Dm231"] = dm
        pt.OscillationTunes.Parameters["Dm231_bar"] = dm
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

    chi2_stat = pynufit.LLH.stats_only(pynufit.Expectation, pynufit.MCVariance)
    tol_adj = max(1e-5, np.sqrt(max(chi2_stat, 0)) * 1e-5)

    result = minimize(
        objective, nominal,
        method='L-BFGS-B', jac=gradient, bounds=bounds,
        options={'ftol': tol_adj, 'gtol': 1e-5, 'maxiter': 200}
    )

    return result.fun, result.x, result.nit, result.success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-idx", type=int, required=True, help="Row index (theta index)")
    parser.add_argument("--n-theta", type=int, default=20)
    parser.add_argument("--n-dm", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    if args.config is None:
        args.config = os.path.join(PYNU_DIR, "examples/AnalysisFiles/ORCA_Atm_CPT_real.xml")

    theta_grid = np.linspace(THETA_MIN, THETA_MAX, args.n_theta)
    dm_grid = np.linspace(DM_MIN, DM_MAX, args.n_dm)
    theta = theta_grid[args.row_idx]

    print(f"[Worker {args.row_idx}] theta={theta:.4f}, {args.n_dm} dm points")
    print(f"  Started: {datetime.now().isoformat()}")

    # Initialize
    pynufit, n_muon = setup_pynufit(args.config)
    print(f"  Muon events in observation: {n_muon:.1f}")

    nominal = np.array(pynufit.Analysis.NuisNominalList)
    sigma = np.array(pynufit.Analysis.NuisSigmaList)
    lower = nominal - 5 * sigma
    upper = nominal + 5 * sigma
    for k in range(len(lower)):
        if nominal[k] > 0 and lower[k] < 0.01:
            lower[k] = 0.01
    bounds = list(zip(lower, upper))

    # Run all dm points for this theta
    results = []
    for j, dm in enumerate(dm_grid):
        chi2, nuisance, nit, converged = run_one_point(pynufit, theta, dm, nominal, sigma, bounds)
        pull_max = np.max(np.abs((nuisance - nominal) / sigma))
        results.append({
            'i': args.row_idx, 'j': j,
            'theta': float(theta), 'dm': float(dm),
            'chi2': float(chi2), 'nit': int(nit), 'converged': bool(converged),
            'nuisance': nuisance.tolist()
        })
        print(f"  [{args.row_idx},{j:2d}] dm={dm:.5f}: chi2={chi2:7.2f}, "
              f"iter={nit:3d}, pull={pull_max:.2f}s")

    # Save row results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"row_{args.row_idx:03d}.json")
    with open(out_path, 'w') as f:
        json.dump({
            'row_idx': args.row_idx,
            'theta': float(theta),
            'n_dm': args.n_dm,
            'nuisance_names': pynufit.Analysis.NuisanceList,
            'points': results
        }, f, indent=2)

    print(f"  Saved: {out_path}")
    print(f"  Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
