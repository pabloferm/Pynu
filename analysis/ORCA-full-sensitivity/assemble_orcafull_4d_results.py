#!/usr/bin/env python3
"""
Assemble 4D scan row JSONs into profiled 2D grid arrays.
"""

import sys
import os
import json
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-dm", type=int, default=41)
    parser.add_argument("--n-s23", type=int, default=41)
    parser.add_argument("--dm-min", type=float, default=2.0e-3)
    parser.add_argument("--dm-max", type=float, default=3.0e-3)
    parser.add_argument("--s23-min", type=float, default=0.40)
    parser.add_argument("--s23-max", type=float, default=0.65)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir

    dm_grid = np.linspace(args.dm_min, args.dm_max, args.n_dm)
    s23_grid = np.linspace(args.s23_min, args.s23_max, args.n_s23)

    chi2_grid = np.full((args.n_dm, args.n_s23), np.nan)
    converged_grid = np.zeros((args.n_dm, args.n_s23), dtype=bool)
    best_theta13_grid = np.full((args.n_dm, args.n_s23), np.nan)
    best_dcp_grid = np.full((args.n_dm, args.n_s23), np.nan)

    found = 0
    total_time = 0
    for i in range(args.n_dm):
        row_file = os.path.join(args.input_dir, f'row_{i:03d}.json')
        if not os.path.exists(row_file):
            print(f"  Missing row {i}")
            continue
        with open(row_file) as f:
            row = json.load(f)
        chi2_grid[i, :] = row['chi2_profiled']
        converged_grid[i, :] = row['converged']
        best_theta13_grid[i, :] = row['best_theta13']
        best_dcp_grid[i, :] = row['best_dcp']
        total_time += row.get('total_time_s', 0)
        found += 1

    print(f"Assembled {found}/{args.n_dm} rows")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, 'chi2_grid_4d.npy'), chi2_grid)
    np.save(os.path.join(args.output_dir, 'converged_grid_4d.npy'), converged_grid)
    np.save(os.path.join(args.output_dir, 'dm_grid.npy'), dm_grid)
    np.save(os.path.join(args.output_dir, 's23_grid.npy'), s23_grid)
    np.save(os.path.join(args.output_dir, 'best_theta13_grid.npy'), best_theta13_grid)
    np.save(os.path.join(args.output_dir, 'best_dcp_grid.npy'), best_dcp_grid)

    # Metadata
    meta = {
        'n_dm': args.n_dm, 'n_s23': args.n_s23,
        'dm_min': args.dm_min, 'dm_max': args.dm_max,
        's23_min': args.s23_min, 's23_max': args.s23_max,
        'profiled_over': ['Sin2Theta13', 'dCP'],
        'rows_found': found,
        'convergence_rate': float(converged_grid.sum()) / (args.n_dm * args.n_s23),
        'min_chi2': float(np.nanmin(chi2_grid)),
        'total_time_s': total_time,
    }

    bf_idx = np.unravel_index(np.nanargmin(chi2_grid), chi2_grid.shape)
    meta['best_fit_dm'] = float(dm_grid[bf_idx[0]])
    meta['best_fit_s23'] = float(s23_grid[bf_idx[1]])
    meta['best_fit_theta13'] = float(best_theta13_grid[bf_idx])
    meta['best_fit_dcp'] = float(best_dcp_grid[bf_idx])

    with open(os.path.join(args.output_dir, 'metadata_4d.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Summary
    print(f"\nMin chi²: {meta['min_chi2']:.4f}")
    print(f"Best-fit: Dm231={meta['best_fit_dm']:.4e}, S23={meta['best_fit_s23']:.4f}")
    print(f"  theta13={meta['best_fit_theta13']:.4f}, dCP={meta['best_fit_dcp']:.3f}")
    print(f"Convergence: {meta['convergence_rate']*100:.1f}%")
    print(f"Total time: {total_time/3600:.1f} hr")

    # 1sigma widths
    dchi2 = chi2_grid - np.nanmin(chi2_grid)
    dm_prof = np.nanmin(dchi2, axis=1)
    s23_prof = np.nanmin(dchi2, axis=0)
    dm_1s = dm_grid[dm_prof < 1.0]
    s23_1s = s23_grid[s23_prof < 1.0]
    if len(dm_1s) > 0:
        print(f"1σ Dm231: [{dm_1s[0]*1e3:.4f}, {dm_1s[-1]*1e3:.4f}] "
              f"(width {(dm_1s[-1]-dm_1s[0])*1e3:.4f}×10⁻³)")
    if len(s23_1s) > 0:
        print(f"1σ S23:   [{s23_1s[0]:.4f}, {s23_1s[-1]:.4f}] "
              f"(width {s23_1s[-1]-s23_1s[0]:.4f})")

    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
