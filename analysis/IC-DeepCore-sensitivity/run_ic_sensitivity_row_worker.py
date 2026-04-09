#!/usr/bin/env python3
"""
IceCube DeepCore Sensitivity Scan Row Worker (for SLURM array jobs)
— Hypersurface (HS) detector systematics version.

Each worker handles one row of the 2D grid:
- Fixed Dm231 value (determined by --row-idx)
- Scans all Sin2Theta23 values
- Saves results to a per-row JSON file

HS corrections are applied at the binned-histogram level:
  corrected[cat][bin] = uncorrected[cat][bin] × (intercept + Σ slope × (param − nominal))

Grid: Dm231 x Sin2Theta23

Usage:
    python run_ic_sensitivity_row_worker.py --row-idx 0 --n-dm 41 --n-s23 41 --output-dir DIR
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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
PYNU_PARENT = os.path.join(BASE_DIR, 'Pynu')
if PYNU_PARENT not in sys.path:
    sys.path.insert(0, PYNU_PARENT)

from pynu import PyNuFit
from pynu.Experiments.ICDeepCore import ICDeepCore_Atm

# Grid parameters
DM_MIN = 2.0e-3
DM_MAX = 3.0e-3
S23_MIN = 0.40
S23_MAX = 0.65

# Truth values (Asimov)
TRUTH_DM = 2.511e-3
TRUTH_S23 = 0.572

# Default HS directory (cluster path)
DEFAULT_HS_DIR = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit/Pynu/data/IceCube'


def setup_pynufit(config, hs_dir):
    """Initialize PyNuFit with Asimov observation at truth parameters.

    The Asimov observation includes HS corrections at truth Δm²₃₁ with
    nominal HS parameters, ensuring chi²=0 at the truth point.
    """
    pynufit = PyNuFit(config, verbosity=False)

    # Load hypersurfaces for each ICDeepCore experiment
    for exp_name, exp in pynufit.Experiments.items():
        if isinstance(exp, ICDeepCore_Atm):
            exp.load_hypersurfaces(hs_dir)

    # Set truth parameters
    for name, pt in pynufit.physics_tunes.items():
        pt.OscillationTunes.Parameters["Sin2Theta23"] = TRUTH_S23
        pt.OscillationTunes.Parameters["Dm231"] = TRUTH_DM
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

    # Apply HS correction at truth with nominal params → corrected expectation
    for exp_name, exp in pynufit.Experiments.items():
        if isinstance(exp, ICDeepCore_Atm):
            corrected = exp.apply_hs_correction(TRUTH_DM, ICDeepCore_Atm.HS_NOMINALS)
            pynufit.Expectation[exp_name] = corrected

    # Build Asimov = HS-corrected neutrino expectation + muon background
    asimov = copy.deepcopy(pynufit.Expectation)
    n_muon = 0
    for exp_name in asimov:
        if exp_name in pynufit.MuonBackground and pynufit.MuonBackground[exp_name] is not None:
            muon_counts, _ = pynufit.MuonBackground[exp_name]
            # Muon background is already masked, so it matches the masked expectation
            asimov[exp_name] = asimov[exp_name] + muon_counts
            n_muon += np.sum(muon_counts)

    # Set up likelihood
    pynufit.set_likelihood("BarlowBeestonLikelihood")
    for exp_name in asimov:
        pynufit.LLH.observation[exp_name] = asimov[exp_name]
    pynufit.LLH.set_muon_background(pynufit.MuonBackground)
    pynufit.LLH.set_mc_variance(pynufit.MCVariance)

    # Build HS parameter index map (nuisance vector position → HS param name)
    hs_names = ICDeepCore_Atm.HS_SLOPE_NAMES
    hs_indices = {}
    for name in hs_names:
        if name in pynufit.Analysis.NuisanceList:
            hs_indices[name] = pynufit.Analysis.NuisanceList.index(name)
    print(f"  HS parameter indices: {hs_indices}")

    return pynufit, n_muon, hs_indices


def run_one_point(pynufit, dm231, sin2theta23, nominal, sigma, bounds, hs_indices):
    """Run minimization at a single (Dm231, Sin2Theta23) grid point.

    Uses HS corrections on binned histograms. The objective function:
    1. Applies flux nuisance weights at event level (HS params are no-ops)
    2. Bins events via SetBinnedExpectedEvents (uncorrected total)
    3. Overwrites Expectation with HS-corrected histograms
    4. Computes chi² including prior penalties for all params

    Analytical gradient is dropped — uses L-BFGS-B with finite differences.
    """
    pynufit.StartPhysics()
    pynufit.StartNuisance()

    for name, pt in pynufit.physics_tunes.items():
        pt.OscillationTunes.Parameters["Dm231"] = dm231
        pt.OscillationTunes.Parameters["Dm231_bar"] = dm231  # CPT symmetric
        pt.OscillationTunes.Parameters["Sin2Theta23"] = sin2theta23
        if hasattr(pt.OscillationTunes, 'reset_cache'):
            pt.OscillationTunes.reset_cache()

    pynufit.ApplyOscillations("Physics")

    def objective(nuisance):
        # 1. Apply nuisance weights (flux params modify event weights; HS are no-ops)
        pynufit.StartNuisance()
        pynufit.ApplyNuisanceWeights(nuisance)
        pynufit.SetExpectedWeights()
        # 2. Bin events (uncorrected)
        pynufit.SetBinnedExpectedEvents()

        # 3. Build HS param dict from nuisance vector
        hs_params = {}
        for hs_name, idx in hs_indices.items():
            hs_params[hs_name] = nuisance[idx]

        # 4. Overwrite expectation with HS-corrected histograms
        for exp_name, exp in pynufit.Experiments.items():
            if isinstance(exp, ICDeepCore_Atm):
                corrected = exp.apply_hs_correction(dm231, hs_params)
                pynufit.Expectation[exp_name] = corrected

        # 5. Compute chi² (stats + systematics penalty)
        return pynufit.LLH.stats_and_systematics(
            pynufit.Expectation, nuisance, pynufit.MCVariance
        )

    # No analytical gradient — use finite differences
    result = minimize(
        objective, nominal,
        method='L-BFGS-B', jac=None, bounds=bounds,
        options={'ftol': 1e-5, 'gtol': 1e-5, 'maxiter': 200}
    )

    return result.fun, result.x, result.nit, result.success


def main():
    parser = argparse.ArgumentParser(description="IC DeepCore sensitivity scan row worker (HS version)")
    parser.add_argument("--row-idx", type=int, required=True, help="Row index (Dm231 index)")
    parser.add_argument("--n-dm", type=int, default=41, help="Number of Dm231 grid points")
    parser.add_argument("--n-s23", type=int, default=41, help="Number of Sin2Theta23 grid points")
    parser.add_argument("--output-dir", required=True, help="Output directory for row results")
    parser.add_argument("--config", default=None)
    parser.add_argument("--hs-dir", default=DEFAULT_HS_DIR,
                        help="Path to directory containing hs_*.csv files")
    args = parser.parse_args()

    PROJECT_DIR = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit'
    if args.config is None:
        args.config = f"{PROJECT_DIR}/Pynu/examples/AnalysisFiles/IC_Atm.xml"

    dm_grid = np.linspace(DM_MIN, DM_MAX, args.n_dm)
    s23_grid = np.linspace(S23_MIN, S23_MAX, args.n_s23)
    dm231 = dm_grid[args.row_idx]

    print(f"[Worker {args.row_idx}] Dm231={dm231:.5e}, scanning {args.n_s23} Sin2Theta23 points")
    print(f"  Sin2Theta23 range: [{S23_MIN}, {S23_MAX}]")
    print(f"  HS dir: {args.hs_dir}")
    print(f"  Started: {datetime.now().isoformat()}")

    # Initialize
    pynufit, n_muon, hs_indices = setup_pynufit(args.config, args.hs_dir)
    print(f"  Muon events in observation: {n_muon:.1f}")

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
    results = []
    for j, sin2theta23 in enumerate(s23_grid):
        chi2, nuisance, nit, converged = run_one_point(
            pynufit, dm231, sin2theta23, nominal, sigma, bounds, hs_indices
        )
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
            'hs_indices': hs_indices,
            'points': results
        }, f, indent=2)

    n_conv = sum(1 for r in results if r['converged'])
    print(f"  Convergence: {n_conv}/{len(results)}")
    print(f"  Saved: {out_path}")
    print(f"  Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
