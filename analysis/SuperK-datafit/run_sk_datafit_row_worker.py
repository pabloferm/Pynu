#!/usr/bin/env python3
"""
Super-Kamiokande 2023 Data Fit Row Worker

Standard oscillation data fit over Dm231 x Sin2Theta23 grid with:
- 32 nuisance parameters (7 flux + 4 xsec + 21 detector) with analytical gradient
- dCP profiled via discrete scan over N_DCP values in [0, 2pi)
- Sin2Theta13, Sin2Theta12, Dm221 fixed at nominal (strong external priors +
  negligible atmospheric sensitivity — confirmed by 36-param run where they
  didn't move from nominal)
- L-BFGS-B with analytical gradient for fast nuisance minimization
- Duo start: at each point, minimize from BOTH nominal and warm start, keep best
  (avoids local minima from either starting point alone)

Each worker handles one row (fixed Dm231), scanning all Sin2Theta23 values.

Usage:
    python run_sk_datafit_row_worker.py --row-idx 0 --n-dm 41 --n-s23 41 --output-dir DIR
"""

import sys
import os
import argparse
import numpy as np
import json
from datetime import datetime
from scipy.optimize import minimize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
PYNU_PARENT = os.path.join(BASE_DIR, 'Pynu')
if PYNU_PARENT not in sys.path:
    sys.path.insert(0, PYNU_PARENT)

from pynu import PyNuFit

# Grid parameters
DM_MIN = 2.0e-3
DM_MAX = 3.0e-3
S23_MIN = 0.40
S23_MAX = 0.65

# Nominal oscillation parameter values
NOM_DM = 2.511e-3
NOM_S23 = 0.572
NOM_DCP = 1.36
NOM_S13 = 0.022
NOM_S12 = 0.303
NOM_DM221 = 7.41e-5


def setup_pynufit_datafit(config):
    """Initialize PyNuFit with real SK data as observation."""
    pynufit = PyNuFit(config, verbosity=False)

    # Set initial physics params at nominal
    for name, pt in pynufit.physics_tunes.items():
        pt.OscillationTunes.Parameters["Sin2Theta23"] = NOM_S23
        pt.OscillationTunes.Parameters["Dm231"] = NOM_DM
        pt.OscillationTunes.Parameters["dCP"] = NOM_DCP
        pt.OscillationTunes.Parameters["Sin2Theta13"] = NOM_S13
        pt.OscillationTunes.Parameters["Sin2Theta12"] = NOM_S12
        pt.OscillationTunes.Parameters["Dm221"] = NOM_DM221
        if "Dm231_bar" in pt.OscillationTunes.Parameters:
            pt.OscillationTunes.Parameters["Dm231_bar"] = NOM_DM
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

    # Set observation from real data (DataFiles enabled in XML)
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


def set_physics_params(pynufit, dm231, sin2theta23, dcp):
    """Set oscillation parameters and recompute weights."""
    for name, pt in pynufit.physics_tunes.items():
        pt.OscillationTunes.Parameters["Dm231"] = dm231
        if "Dm231_bar" in pt.OscillationTunes.Parameters:
            pt.OscillationTunes.Parameters["Dm231_bar"] = dm231
        pt.OscillationTunes.Parameters["Sin2Theta23"] = sin2theta23
        pt.OscillationTunes.Parameters["dCP"] = dcp
        if hasattr(pt.OscillationTunes, 'reset_cache'):
            pt.OscillationTunes.reset_cache()

    pynufit.ApplyOscillations("Physics")


def minimize_nuisance(pynufit, x0, sigma, bounds):
    """Minimize chi2 over nuisance parameters at current physics point."""
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

    # Prepare differential expectation for gradient
    pynufit.StartNuisance()
    pynufit.ApplyNuisanceWeights(x0)
    pynufit.SetExpectedWeights()
    pynufit.SetBinnedExpectedEvents()
    pynufit.SetBinnedMCVariance()
    pynufit.ComputeBinnedDiffExpectation()

    chi2_stat = pynufit.LLH.stats_only(pynufit.Expectation, pynufit.MCVariance)
    tol_adj = max(1e-5, np.sqrt(max(chi2_stat, 0)) * 1e-5)

    result = minimize(
        objective, x0,
        method='L-BFGS-B', jac=gradient, bounds=bounds,
        options={'ftol': tol_adj, 'gtol': 1e-5, 'maxiter': 200}
    )

    return result.fun, result.x, result.nit, result.success


def run_one_point_single(pynufit, dm231, sin2theta23, x0, sigma, bounds, n_dcp=13):
    """Run dCP-profiled minimization from a single starting point.

    Scans dCP over n_dcp values in [0, 2pi), minimizing nuisance at each.
    Returns the minimum chi2 over all dCP values.
    """
    dcp_grid = np.linspace(0, 2 * np.pi, n_dcp, endpoint=False)
    best_chi2 = np.inf
    best_dcp = dcp_grid[0]
    best_nuisance = x0.copy()
    best_nit = 0
    best_converged = False

    for dcp in dcp_grid:
        # Set all physics params and compute oscillation weights
        pynufit.StartPhysics()
        pynufit.StartNuisance()
        set_physics_params(pynufit, dm231, sin2theta23, dcp)

        # Minimize over nuisance parameters
        chi2, nuisance, nit, converged = minimize_nuisance(
            pynufit, x0, sigma, bounds
        )

        if chi2 < best_chi2:
            best_chi2 = chi2
            best_dcp = dcp
            best_nuisance = nuisance.copy()
            best_nit = nit
            best_converged = converged

    return best_chi2, best_dcp, best_nuisance, best_nit, best_converged


def run_one_point(pynufit, dm231, sin2theta23, nominal, warm_x0, sigma, bounds, n_dcp=13, duo=False):
    """Minimization with warm start (+ optional duo-start from nominal).

    - Warm start: uses previous point's best-fit nuisance as starting point
    - Duo start (optional): also tries from nominal, keeps best result
    """
    # Use warm start if available, otherwise nominal
    x0 = warm_x0 if (warm_x0 is not None) else nominal

    chi2_best, dcp_best, nuis_best, nit_best, conv_best = run_one_point_single(
        pynufit, dm231, sin2theta23, x0, sigma, bounds, n_dcp
    )
    winner = 'warm' if (warm_x0 is not None) else 'nominal'

    # Optional: also try from nominal and keep the better result
    if duo and warm_x0 is not None and not np.array_equal(warm_x0, nominal):
        chi2_nom, dcp_nom, nuis_nom, nit_nom, conv_nom = run_one_point_single(
            pynufit, dm231, sin2theta23, nominal, sigma, bounds, n_dcp
        )
        if chi2_nom < chi2_best:
            chi2_best, dcp_best, nuis_best, nit_best, conv_best = \
                chi2_nom, dcp_nom, nuis_nom, nit_nom, conv_nom
            winner = 'nominal'

    return chi2_best, dcp_best, nuis_best, nit_best, conv_best, winner


def main():
    parser = argparse.ArgumentParser(description="SK data fit row worker (dCP profiled + 32 nuisance)")
    parser.add_argument("--row-idx", type=int, required=True, help="Row index (Dm231 index)")
    parser.add_argument("--n-dm", type=int, default=41, help="Number of Dm231 grid points")
    parser.add_argument("--n-s23", type=int, default=41, help="Number of Sin2Theta23 grid points")
    parser.add_argument("--n-dcp", type=int, default=13, help="Number of dCP scan points in [0, 2pi)")
    parser.add_argument("--output-dir", required=True, help="Output directory for row results")
    parser.add_argument("--config", default=None)
    parser.add_argument("--duo", action="store_true", help="Enable duo-start (nominal + warm). Slower but more robust.")
    args = parser.parse_args()

    PROJECT_DIR = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit'
    if args.config is None:
        args.config = f"{PROJECT_DIR}/Pynu/examples/AnalysisFiles/SK2023_Atm_datafit.xml"

    dm_grid = np.linspace(DM_MIN, DM_MAX, args.n_dm)
    s23_grid = np.linspace(S23_MIN, S23_MAX, args.n_s23)
    dm231 = dm_grid[args.row_idx]

    print(f"[SK DataFit Worker {args.row_idx}] Dm231={dm231:.5e}, scanning {args.n_s23} Sin2Theta23 points")
    print(f"  Sin2Theta23 range: [{S23_MIN}, {S23_MAX}]")
    print(f"  dCP profiling: {args.n_dcp} points in [0, 2pi)")
    print(f"  32 nuisance params with analytical gradient")
    print(f"  Sin2Theta13={NOM_S13}, Sin2Theta12={NOM_S12}, Dm221={NOM_DM221} (fixed, strong priors)")
    print(f"  Started: {datetime.now().isoformat()}")

    # Initialize with real data
    pynufit, n_data, n_muon = setup_pynufit_datafit(args.config)
    print(f"  Data events in observation: {n_data:.1f}")
    print(f"  Muon background: {n_muon:.1f}")

    n_nuisance = len(pynufit.Analysis.NuisNominalList)
    print(f"  Nuisance parameters ({n_nuisance}): {pynufit.Analysis.NuisanceList}")

    # Set up minimization bounds (10 sigma for data fit — wider than sensitivity
    # to avoid optimizer getting stuck at bounds)
    nominal = np.array(pynufit.Analysis.NuisNominalList)
    sigma = np.array(pynufit.Analysis.NuisSigmaList)
    lower = nominal - 10 * sigma
    upper = nominal + 10 * sigma
    for k in range(len(lower)):
        if nominal[k] > 0 and lower[k] < 0.01:
            lower[k] = 0.01
    bounds = list(zip(lower, upper))

    # Run all Sin2Theta23 points for this Dm231
    # Duo start: at each point, minimize from both nominal and warm start, keep best
    results = []
    warm_x0 = None  # No warm start for first point
    for j, sin2theta23 in enumerate(s23_grid):
        t0 = datetime.now()
        chi2, best_dcp, nuisance, nit, converged, winner = run_one_point(
            pynufit, dm231, sin2theta23, nominal, warm_x0, sigma, bounds,
            n_dcp=args.n_dcp, duo=args.duo
        )
        elapsed = (datetime.now() - t0).total_seconds()

        pull_max = np.max(np.abs((nuisance - nominal) / sigma))

        # Always update warm start with best result for next point
        warm_x0 = nuisance.copy()

        results.append({
            'i': args.row_idx, 'j': j,
            'dm231': float(dm231), 'sin2theta23': float(sin2theta23),
            'chi2': float(chi2), 'best_dcp': float(best_dcp),
            'nit': int(nit), 'converged': bool(converged),
            'max_pull': float(pull_max),
            'elapsed_s': float(elapsed),
            'winner': winner,
            'nuisance': nuisance.tolist()
        })
        print(f"  [{args.row_idx},{j:2d}] s23={sin2theta23:.4f}: chi2={chi2:8.4f}, "
              f"dCP={best_dcp:.3f}, iter={nit:3d}, conv={converged}, "
              f"pull={pull_max:.3f}, {winner}, {elapsed:.1f}s")

    # Save row results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"row_{args.row_idx:03d}.json")
    with open(out_path, 'w') as f:
        json.dump({
            'row_idx': args.row_idx,
            'dm231': float(dm231),
            'n_dm': args.n_dm,
            'n_s23': args.n_s23,
            'n_dcp': args.n_dcp,
            'dm_range': [float(DM_MIN), float(DM_MAX)],
            's23_range': [float(S23_MIN), float(S23_MAX)],
            'truth_dm': float(NOM_DM),
            'truth_s23': float(NOM_S23),
            'truth_dcp': float(NOM_DCP),
            'mode': 'datafit',
            'n_data_events': float(n_data),
            'n_nuisance': n_nuisance,
            'nuisance_names': pynufit.Analysis.NuisanceList,
            'fixed_osc_params': {
                'Sin2Theta13': NOM_S13,
                'Sin2Theta12': NOM_S12,
                'Dm221': NOM_DM221
            },
            'points': results
        }, f, indent=2)

    n_conv = sum(1 for r in results if r['converged'])
    chi2_min = min(r['chi2'] for r in results)
    print(f"\n  Convergence: {n_conv}/{len(results)}")
    print(f"  Min chi2 in row: {chi2_min:.4f}")
    print(f"  Saved: {out_path}")
    print(f"  Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
