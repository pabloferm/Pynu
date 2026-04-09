#!/usr/bin/env python3
"""
ORCA Data Fit Row Worker (for SLURM array jobs)

Standard (non-CPT) 2D grid scan over (Dm231, Sin2Theta23) with real ORCA data.
Each worker handles one row (fixed Dm231, scanning sin²θ₂₃).
Uses analytical gradient (L-BFGS-B).
No hypersurface corrections (ORCA uses event-level detector systematics).

Usage:
    python run_orca_datafit_row_worker.py --row-idx 0 --n-dm 41 --n-s23 41 --output-dir DIR
"""

import sys
import os
import argparse
import numpy as np
import json
from datetime import datetime
from scipy.optimize import minimize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
PYNU_PARENT = os.path.join(BASE_DIR, 'Pynu')
if PYNU_PARENT not in sys.path:
    sys.path.insert(0, PYNU_PARENT)

from pynu import PyNuFit

# Grid parameters
DM_MIN = 2.0e-3
DM_MAX = 3.0e-3
S23_MIN = 0.40
S23_MAX = 0.65

# Truth values (for metadata only)
TRUTH_DM = 2.511e-3
TRUTH_S23 = 0.572


def setup_pynufit_datafit(config):
    """Initialize PyNuFit with real ORCA data as observation."""
    pynufit = PyNuFit(config, verbosity=False)

    # Set initial physics params
    for name, pt in pynufit.physics_tunes.items():
        pt.OscillationTunes.Parameters["Sin2Theta23"] = TRUTH_S23
        pt.OscillationTunes.Parameters["Dm231"] = TRUTH_DM
        pt.OscillationTunes.Parameters["Dm231_bar"] = TRUTH_DM
        if hasattr(pt.OscillationTunes, 'reset_cache'):
            pt.OscillationTunes.reset_cache()

    # Generate nominal expectation
    pynufit.StartPhysics()
    pynufit.StartNuisance()
    pynufit.ApplyOscillations("Physics")
    pynufit.ApplyNuisanceWeights(pynufit.Analysis.NuisNominalList)
    pynufit.SetExpectedWeights()
    pynufit.SetBinnedExpectedEvents()
    pynufit.SetBinnedMCVariance()
    pynufit.SetMuonBackground()

    # Set observation from real data
    pynufit.Observation = {}
    for exp_name, exp in pynufit.Experiments.items():
        exp.SetObservedBinned()
        pynufit.Observation[exp_name] = exp.GetObservedBinned()

    # Set up likelihood
    pynufit.set_likelihood("BarlowBeestonLikelihood")
    for exp_name in pynufit.Observation:
        pynufit.LLH.observation[exp_name] = pynufit.Observation[exp_name]
    pynufit.LLH.set_muon_background(pynufit.MuonBackground)
    pynufit.LLH.set_mc_variance(pynufit.MCVariance)

    # Report stats
    n_data = sum(np.sum(obs) for obs in pynufit.Observation.values())
    n_muon = 0
    for exp_name in pynufit.MuonBackground:
        if pynufit.MuonBackground[exp_name] is not None:
            muon_counts, _ = pynufit.MuonBackground[exp_name]
            n_muon += np.sum(muon_counts)

    return pynufit, n_data, n_muon


def run_one_point(pynufit, dm231, sin2theta23, nominal, sigma, bounds, x0=None):
    """Run minimization at a single grid point.

    Args:
        x0: Initial guess for nuisance parameters. If None, uses nominal.
            Warm starting from previous point's best-fit improves convergence
            and produces smoother chi2 surfaces (matches PyNuFit.run_2d_profile_scan).
    """
    if x0 is None:
        x0 = nominal

    pynufit.StartPhysics()
    pynufit.StartNuisance()

    for name, pt in pynufit.physics_tunes.items():
        pt.OscillationTunes.Parameters["Dm231"] = dm231
        pt.OscillationTunes.Parameters["Dm231_bar"] = dm231
        pt.OscillationTunes.Parameters["Sin2Theta23"] = sin2theta23
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
        objective, x0,
        method='L-BFGS-B', jac=gradient, bounds=bounds,
        options={'ftol': tol_adj, 'gtol': 1e-5, 'maxiter': 200}
    )

    return result.fun, result.x, result.nit, result.success


def main():
    parser = argparse.ArgumentParser(description="ORCA data fit row worker (Dm231 x Sin2Theta23)")
    parser.add_argument("--row-idx", type=int, required=True, help="Row index (Dm231 index)")
    parser.add_argument("--n-dm", type=int, default=41, help="Number of Dm231 points")
    parser.add_argument("--n-s23", type=int, default=41, help="Number of Sin2Theta23 points")
    parser.add_argument("--output-dir", required=True, help="Output directory for row results")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dm-min", type=float, default=DM_MIN, help="Dm231 min")
    parser.add_argument("--dm-max", type=float, default=DM_MAX, help="Dm231 max")
    parser.add_argument("--s23-min", type=float, default=S23_MIN, help="Sin2Theta23 min")
    parser.add_argument("--s23-max", type=float, default=S23_MAX, help="Sin2Theta23 max")
    args = parser.parse_args()

    PROJECT_DIR = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit'
    if args.config is None:
        args.config = f"{PROJECT_DIR}/Pynu/examples/AnalysisFiles/ORCA_Atm_datafit_muon_norm.xml"

    dm_grid = np.linspace(args.dm_min, args.dm_max, args.n_dm)
    s23_grid = np.linspace(args.s23_min, args.s23_max, args.n_s23)
    dm231 = dm_grid[args.row_idx]

    print(f"[ORCA DataFit Worker {args.row_idx}] Dm231={dm231:.5e}, scanning {args.n_s23} Sin2Theta23 points")
    print(f"  Sin2Theta23 range: [{args.s23_min}, {args.s23_max}]")
    print(f"  Started: {datetime.now().isoformat()}")

    pynufit, n_data, n_muon = setup_pynufit_datafit(args.config)
    print(f"  Data events in observation: {n_data:.1f}")
    print(f"  Muon background: {n_muon:.1f}")
    print(f"  Nuisance parameters: {pynufit.Analysis.NuisanceList}")

    nominal = np.array(pynufit.Analysis.NuisNominalList)
    sigma = np.array(pynufit.Analysis.NuisSigmaList)
    lower = nominal - 5 * sigma
    upper = nominal + 5 * sigma
    for k in range(len(lower)):
        if nominal[k] > 0 and lower[k] < 0.01:
            lower[k] = 0.01
    bounds = list(zip(lower, upper))

    results = []
    warm_x0 = None  # Warm start: use previous point's best-fit nuisance
    for j, sin2theta23 in enumerate(s23_grid):
        chi2, nuisance, nit, converged = run_one_point(
            pynufit, dm231, sin2theta23, nominal, sigma, bounds, x0=warm_x0
        )
        warm_x0 = nuisance.copy()  # Pass to next point
        pull_max = np.max(np.abs((nuisance - nominal) / sigma))
        results.append({
            'i': args.row_idx, 'j': j,
            'dm231': float(dm231), 'sin2theta23': float(sin2theta23),
            'chi2': float(chi2), 'nit': int(nit),
            'converged': bool(converged), 'max_pull': float(pull_max),
            'nuisance': nuisance.tolist()
        })
        print(f"  [{args.row_idx},{j:2d}] s23={sin2theta23:.4f}: chi2={chi2:8.4f}, "
              f"iter={nit:3d}, conv={converged}, pull={pull_max:.3f}s")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"row_{args.row_idx:03d}.json")
    with open(out_path, 'w') as f:
        json.dump({
            'row_idx': args.row_idx,
            'dm231': float(dm231),
            'n_dm': args.n_dm,
            'n_s23': args.n_s23,
            'dm_range': [float(args.dm_min), float(args.dm_max)],
            's23_range': [float(args.s23_min), float(args.s23_max)],
            'truth_dm': float(TRUTH_DM),
            'truth_s23': float(TRUTH_S23),
            'mode': 'datafit',
            'n_data_events': float(n_data),
            'nuisance_names': pynufit.Analysis.NuisanceList,
            'points': results
        }, f, indent=2)

    n_conv = sum(1 for r in results if r['converged'])
    print(f"  Convergence: {n_conv}/{len(results)}")
    print(f"  Saved: {out_path}")
    print(f"  Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
