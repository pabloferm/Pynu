#!/usr/bin/env python3
"""
Assemble Super-Kamiokande Sensitivity Scan Results from Array Job Row Files

Collects per-row JSON files from the array job workers and assembles them
into the final grid arrays (chi2_grid.npy, etc.) compatible with the
plotting script.

Usage:
    python assemble_sk_sensitivity.py --input-dir DIR [--output-dir DIR]
"""

import sys
import os
import argparse
import numpy as np
import json
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Assemble SK sensitivity grid from row results")
    parser.add_argument("--input-dir", required=True, help="Directory with row_*.json files")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: parent of input-dir)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input_dir)

    print("=" * 70)
    print("ASSEMBLING SUPER-KAMIOKANDE SENSITIVITY SCAN RESULTS")
    print("=" * 70)

    # Find all row files
    row_files = sorted([f for f in os.listdir(args.input_dir) if f.startswith("row_") and f.endswith(".json")])
    print(f"Found {len(row_files)} row files")

    if len(row_files) == 0:
        print("ERROR: No row files found!")
        return 1

    # Load first row to get grid parameters
    with open(os.path.join(args.input_dir, row_files[0]), 'r') as f:
        first_row = json.load(f)

    n_dm = first_row['n_dm']
    n_s23 = first_row['n_s23']
    dm_min, dm_max = first_row['dm_range']
    s23_min, s23_max = first_row['s23_range']
    truth_dm = first_row['truth_dm']
    truth_s23 = first_row['truth_s23']
    nuisance_names = first_row['nuisance_names']
    n_nuisance = len(nuisance_names)

    dm_grid = np.linspace(dm_min, dm_max, n_dm)
    s23_grid = np.linspace(s23_min, s23_max, n_s23)

    print(f"Grid: {n_dm} (Dm231) x {n_s23} (Sin2Theta23)")
    print(f"Dm231 range: [{dm_min:.2e}, {dm_max:.2e}]")
    print(f"Sin2Theta23 range: [{s23_min:.2f}, {s23_max:.2f}]")
    print(f"Truth: Dm231={truth_dm:.4e}, Sin2Theta23={truth_s23:.4f}")
    print(f"Nuisance parameters: {n_nuisance}")
    print()

    # Initialize arrays
    chi2_grid = np.full((n_dm, n_s23), np.nan)
    nuisance_grid = np.full((n_dm, n_s23, n_nuisance), np.nan)
    converged_grid = np.zeros((n_dm, n_s23), dtype=bool)
    n_iterations_grid = np.zeros((n_dm, n_s23), dtype=int)

    # Load all rows
    missing_rows = []
    for row_idx in range(n_dm):
        fname = f"row_{row_idx:03d}.json"
        fpath = os.path.join(args.input_dir, fname)
        if not os.path.exists(fpath):
            missing_rows.append(row_idx)
            continue

        with open(fpath, 'r') as f:
            row_data = json.load(f)

        for pt in row_data['points']:
            i, j = pt['i'], pt['j']
            chi2_grid[i, j] = pt['chi2']
            nuisance_grid[i, j, :] = pt['nuisance']
            converged_grid[i, j] = pt['converged']
            n_iterations_grid[i, j] = pt['nit']

    if missing_rows:
        print(f"WARNING: Missing {len(missing_rows)} rows: {missing_rows}")
    else:
        print(f"All {n_dm} rows loaded successfully")

    # Compute statistics
    valid = ~np.isnan(chi2_grid)
    min_chi2 = np.nanmin(chi2_grid)
    best_idx = np.unravel_index(np.nanargmin(chi2_grid), chi2_grid.shape)
    best_dm231 = dm_grid[best_idx[0]]
    best_s23 = s23_grid[best_idx[1]]
    delta_chi2 = chi2_grid - min_chi2

    n_converged = np.sum(converged_grid)
    n_total = np.sum(valid)

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Valid points: {n_total}/{n_dm * n_s23}")
    print(f"Converged: {n_converged}/{n_total} ({100*n_converged/max(n_total,1):.1f}%)")
    print(f"Min chi2: {min_chi2:.6f}")
    print(f"Best-fit Dm231: {best_dm231:.6e} eV2")
    print(f"Best-fit Sin2Theta23: {best_s23:.6f}")
    print(f"Distance from truth: |dDm231|={abs(best_dm231-truth_dm):.2e}, |dS23|={abs(best_s23-truth_s23):.4f}")
    print()

    # Confidence levels
    print("Confidence contour levels (2 DOF):")
    cl_levels = {'1sigma (68.3%)': 2.30, '90%': 4.61, '2sigma (95.5%)': 6.18, '3sigma (99.7%)': 11.83}
    for label, level in cl_levels.items():
        n_inside = np.sum(delta_chi2[valid] <= level)
        print(f"  Delta chi2 = {level:.2f} ({label}): {n_inside} points inside contour")

    # 1D profile summaries
    print()
    print("1D Profile Summaries:")

    dm_profile = np.nanmin(chi2_grid, axis=1)
    dm_dchi2 = dm_profile - np.nanmin(dm_profile)
    dm_1sigma = dm_grid[dm_dchi2 <= 1.0]
    if len(dm_1sigma) > 0:
        print(f"  Dm231 1sigma range: [{dm_1sigma[0]*1e3:.4f}, {dm_1sigma[-1]*1e3:.4f}] x10^-3 eV2")
        print(f"  Dm231 1sigma width: {(dm_1sigma[-1] - dm_1sigma[0])*1e3:.4f} x10^-3 eV2")

    s23_profile = np.nanmin(chi2_grid, axis=0)
    s23_dchi2 = s23_profile - np.nanmin(s23_profile)
    s23_1sigma = s23_grid[s23_dchi2 <= 1.0]
    if len(s23_1sigma) > 0:
        print(f"  Sin2Theta23 1sigma range: [{s23_1sigma[0]:.4f}, {s23_1sigma[-1]:.4f}]")
        print(f"  Sin2Theta23 1sigma width: {s23_1sigma[-1] - s23_1sigma[0]:.4f}")

    # Save assembled results
    os.makedirs(args.output_dir, exist_ok=True)

    np.save(os.path.join(args.output_dir, "chi2_grid.npy"), chi2_grid)
    np.save(os.path.join(args.output_dir, "delta_chi2_grid.npy"), delta_chi2)
    np.save(os.path.join(args.output_dir, "dm_grid.npy"), dm_grid)
    np.save(os.path.join(args.output_dir, "s23_grid.npy"), s23_grid)
    np.save(os.path.join(args.output_dir, "nuisance_grid.npy"), nuisance_grid)
    np.save(os.path.join(args.output_dir, "converged_grid.npy"), converged_grid)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "Super-Kamiokande 2023",
        "grid_size": {"n_dm": n_dm, "n_s23": n_s23},
        "dm_range": [float(dm_min), float(dm_max)],
        "s23_range": [float(s23_min), float(s23_max)],
        "truth": {"dm231": truth_dm, "sin2theta23": truth_s23},
        "min_chi2": float(min_chi2),
        "best_fit": {
            "dm231": float(best_dm231),
            "sin2theta23": float(best_s23),
            "index": [int(best_idx[0]), int(best_idx[1])]
        },
        "nuisance_names": nuisance_names,
        "n_converged": int(n_converged),
        "total_points": int(n_total),
        "missing_rows": missing_rows
    }

    with open(os.path.join(args.output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print(f"Saved to: {args.output_dir}")
    print(f"  - chi2_grid.npy ({n_dm}x{n_s23})")
    print(f"  - delta_chi2_grid.npy")
    print(f"  - dm_grid.npy ({n_dm} points)")
    print(f"  - s23_grid.npy ({n_s23} points)")
    print(f"  - nuisance_grid.npy")
    print(f"  - converged_grid.npy")
    print(f"  - metadata.json")

    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
