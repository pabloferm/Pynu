#!/usr/bin/env python3
"""
IceCube DeepCore Data Fit Row Worker (for SLURM array jobs)
— Hypersurface (HS) detector systematics version.

Uses real IceCube data as observation (from IC_data.parquet).
Each worker handles one row of the 2D grid (Dm231 x Sin2Theta23).

Usage:
    python run_ic_datafit_row_worker.py --row-idx 0 --n-dm 41 --n-s23 41 --output-dir DIR
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
from pynu.Experiments.ICDeepCore import ICDeepCore_Atm

# Grid parameters
DM_MIN = 2.0e-3
DM_MAX = 3.0e-3
S23_MIN = 0.40
S23_MAX = 0.65

# Truth values (for metadata only — not used for observation)
TRUTH_DM = 2.511e-3
TRUTH_S23 = 0.572

# Default HS directory (cluster path)
DEFAULT_HS_DIR = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit/Pynu/data/IceCube'


def setup_pynufit_datafit(config, hs_dir):
    """Initialize PyNuFit with real data as observation.

    For data fitting:
    - Observation = binned real data (from IC_data.parquet, DataFiles status=1)
    - FewEntries mask is set from the data observation
    - No HS correction on observation (data already contains detector effects)
    """
    pynufit = PyNuFit(config, verbosity=False)

    # Load hypersurfaces
    for exp_name, exp in pynufit.Experiments.items():
        if isinstance(exp, ICDeepCore_Atm):
            exp.load_hypersurfaces(hs_dir)

    # Set initial physics params (needed for SetObservedBinned in Asimov mode,
    # but for data fit it just uses BinData)
    for name, pt in pynufit.physics_tunes.items():
        pt.OscillationTunes.Parameters["Sin2Theta23"] = TRUTH_S23
        pt.OscillationTunes.Parameters["Dm231"] = TRUTH_DM
        pt.OscillationTunes.Parameters["Dm231_bar"] = TRUTH_DM
        if hasattr(pt.OscillationTunes, 'reset_cache'):
            pt.OscillationTunes.reset_cache()

    # Generate nominal expectation (needed for FewEntries if not data fit,
    # but with DataFit=True, SetObservedBinned uses BinData)
    pynufit.StartPhysics()
    pynufit.StartNuisance()
    pynufit.ApplyOscillations("Physics")
    pynufit.ApplyNuisanceWeights(pynufit.Analysis.NuisNominalList)
    pynufit.SetExpectedWeights()
    pynufit.SetBinnedExpectedEvents()
    pynufit.SetBinnedMCVariance()
    pynufit.SetMuonBackground()

    # Set observation from real data
    # SetObservedBinned will call BinData() since DataFit=True
    # This also sets FewEntries mask from data
    pynufit.Observation = {}
    for exp_name, exp in pynufit.Experiments.items():
        exp.SetObservedBinned()
        pynufit.Observation[exp_name] = exp.GetObservedBinned()

    # Set up likelihood with data observation
    pynufit.set_likelihood("BarlowBeestonLikelihood")
    for exp_name in pynufit.Observation:
        pynufit.LLH.observation[exp_name] = pynufit.Observation[exp_name]
    pynufit.LLH.set_muon_background(pynufit.MuonBackground)
    pynufit.LLH.set_mc_variance(pynufit.MCVariance)

    # Report observation stats
    n_data = sum(np.sum(obs) for obs in pynufit.Observation.values())
    n_muon = 0
    for exp_name in pynufit.MuonBackground:
        if pynufit.MuonBackground[exp_name] is not None:
            muon_counts, _ = pynufit.MuonBackground[exp_name]
            n_muon += np.sum(muon_counts)

    # Build HS parameter index map
    hs_names = ICDeepCore_Atm.HS_SLOPE_NAMES
    hs_indices = {}
    for name in hs_names:
        if name in pynufit.Analysis.NuisanceList:
            hs_indices[name] = pynufit.Analysis.NuisanceList.index(name)
    print(f"  HS parameter indices: {hs_indices}")

    return pynufit, n_data, n_muon, hs_indices


def run_one_point(pynufit, dm231, sin2theta23, nominal, sigma, bounds, hs_indices):
    """Run minimization at a single grid point with HS corrections."""
    pynufit.StartPhysics()
    pynufit.StartNuisance()

    for name, pt in pynufit.physics_tunes.items():
        pt.OscillationTunes.Parameters["Dm231"] = dm231
        pt.OscillationTunes.Parameters["Dm231_bar"] = dm231
        pt.OscillationTunes.Parameters["Sin2Theta23"] = sin2theta23
        if hasattr(pt.OscillationTunes, 'reset_cache'):
            pt.OscillationTunes.reset_cache()

    pynufit.ApplyOscillations("Physics")

    def objective(nuisance):
        pynufit.StartNuisance()
        pynufit.ApplyNuisanceWeights(nuisance)
        pynufit.SetExpectedWeights()
        pynufit.SetBinnedExpectedEvents()

        hs_params = {}
        for hs_name, idx in hs_indices.items():
            hs_params[hs_name] = nuisance[idx]

        for exp_name, exp in pynufit.Experiments.items():
            if isinstance(exp, ICDeepCore_Atm):
                corrected = exp.apply_hs_correction(dm231, hs_params)
                pynufit.Expectation[exp_name] = corrected

        return pynufit.LLH.stats_and_systematics(
            pynufit.Expectation, nuisance, pynufit.MCVariance
        )

    result = minimize(
        objective, nominal,
        method='L-BFGS-B', jac=None, bounds=bounds,
        options={'ftol': 1e-5, 'gtol': 1e-5, 'maxiter': 200}
    )

    return result.fun, result.x, result.nit, result.success


def main():
    parser = argparse.ArgumentParser(description="IC DeepCore data fit row worker (HS version)")
    parser.add_argument("--row-idx", type=int, required=True)
    parser.add_argument("--n-dm", type=int, default=41)
    parser.add_argument("--n-s23", type=int, default=41)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--hs-dir", default=DEFAULT_HS_DIR)
    args = parser.parse_args()

    PROJECT_DIR = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit'
    if args.config is None:
        args.config = f"{PROJECT_DIR}/Pynu/examples/AnalysisFiles/IC_Atm_datafit.xml"

    dm_grid = np.linspace(DM_MIN, DM_MAX, args.n_dm)
    s23_grid = np.linspace(S23_MIN, S23_MAX, args.n_s23)
    dm231 = dm_grid[args.row_idx]

    print(f"[DataFit Worker {args.row_idx}] Dm231={dm231:.5e}, scanning {args.n_s23} Sin2Theta23 points")
    print(f"  Sin2Theta23 range: [{S23_MIN}, {S23_MAX}]")
    print(f"  HS dir: {args.hs_dir}")
    print(f"  Started: {datetime.now().isoformat()}")

    pynufit, n_data, n_muon, hs_indices = setup_pynufit_datafit(args.config, args.hs_dir)
    print(f"  Data events in observation: {n_data:.1f}")
    print(f"  Muon background: {n_muon:.1f}")

    nominal = np.array(pynufit.Analysis.NuisNominalList)
    sigma = np.array(pynufit.Analysis.NuisSigmaList)
    lower = nominal - 5 * sigma
    upper = nominal + 5 * sigma
    for k in range(len(lower)):
        if nominal[k] > 0 and lower[k] < 0.01:
            lower[k] = 0.01
    bounds = list(zip(lower, upper))

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
            'mode': 'datafit',
            'n_data_events': float(n_data),
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
