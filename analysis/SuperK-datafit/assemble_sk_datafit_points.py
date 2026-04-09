#!/usr/bin/env python3
"""
Assemble SK datafit row JSONs into grid arrays + plot contours.

Usage:
    python assemble_sk_datafit_points.py --input-dir DIR --output-dir DIR [--n-dm 41] [--n-s23 41]
"""

import argparse
import json
import glob
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def assemble(args):
    """Read row JSONs and assemble into grid arrays."""
    files = sorted(glob.glob(os.path.join(args.input_dir, "row_*.json")))
    print(f"Found {len(files)} row files (expected {args.n_dm})")

    if len(files) == 0:
        print("ERROR: No row files found!")
        return None

    # Read grid params from first file
    with open(files[0]) as f:
        d0 = json.load(f)
    dm_range = d0['dm_range']
    s23_range = d0['s23_range']
    n_dcp = d0.get('n_dcp', 13)

    dm_grid = np.linspace(dm_range[0], dm_range[1], args.n_dm)
    s23_grid = np.linspace(s23_range[0], s23_range[1], args.n_s23)

    chi2 = np.full((args.n_dm, args.n_s23), np.nan)
    dcp = np.full((args.n_dm, args.n_s23), np.nan)
    converged = np.zeros((args.n_dm, args.n_s23), dtype=bool)
    max_pull = np.full((args.n_dm, args.n_s23), np.nan)
    elapsed = np.full((args.n_dm, args.n_s23), np.nan)

    n_nuis = None
    nuisance_all = None
    nuisance_names = None

    for fpath in files:
        with open(fpath) as f:
            d = json.load(f)
        for pt in d['points']:
            i, j = pt['i'], pt['j']
            chi2[i, j] = pt['chi2']
            dcp[i, j] = pt['best_dcp']
            converged[i, j] = pt['converged']
            max_pull[i, j] = pt['max_pull']
            elapsed[i, j] = pt['elapsed_s']

            if n_nuis is None:
                n_nuis = len(pt['nuisance'])
                nuisance_all = np.full((args.n_dm, args.n_s23, n_nuis), np.nan)
                nuisance_names = d.get('nuisance_names', [])
            nuisance_all[i, j, :] = pt['nuisance']

    filled = np.sum(~np.isnan(chi2))
    total = args.n_dm * args.n_s23
    print(f"Filled: {filled}/{total} ({100*filled/total:.1f}%)")

    valid = ~np.isnan(chi2)
    if np.any(valid):
        min_chi2 = np.nanmin(chi2)
        bf = np.unravel_index(np.nanargmin(chi2), chi2.shape)
        n_conv = np.sum(converged[valid])
        print(f"Min chi2: {min_chi2:.4f} at Dm231={dm_grid[bf[0]]:.5e}, s23={s23_grid[bf[1]]:.4f}")
        print(f"Convergence: {n_conv}/{filled} ({100*n_conv/filled:.1f}%)")
        print(f"Mean max_pull: {np.nanmean(max_pull):.2f}")
        print(f"Mean elapsed: {np.nanmean(elapsed):.1f}s")
    else:
        print("No valid points!")
        return None

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, 'chi2_grid.npy'), chi2)
    np.save(os.path.join(args.output_dir, 'dcp_grid.npy'), dcp)
    np.save(os.path.join(args.output_dir, 'dm_grid.npy'), dm_grid)
    np.save(os.path.join(args.output_dir, 's23_grid.npy'), s23_grid)
    np.save(os.path.join(args.output_dir, 'converged_grid.npy'), converged)
    np.save(os.path.join(args.output_dir, 'max_pull_grid.npy'), max_pull)
    if nuisance_all is not None:
        np.save(os.path.join(args.output_dir, 'nuisance_grid.npy'), nuisance_all)

    # Save metadata
    metadata = {
        'grid_size': {'n_dm': args.n_dm, 'n_s23': args.n_s23},
        'dm_range': dm_range, 's23_range': s23_range,
        'n_dcp': n_dcp,
        'min_chi2': float(min_chi2),
        'best_fit': {'dm231': float(dm_grid[bf[0]]), 'sin2theta23': float(s23_grid[bf[1]])},
        'n_converged': int(n_conv), 'n_total': int(filled),
        'nuisance_names': nuisance_names,
        'mode': 'datafit'
    }
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved to {args.output_dir}")
    return {
        'chi2': chi2, 'dcp': dcp, 'converged': converged, 'max_pull': max_pull,
        'dm_grid': dm_grid, 's23_grid': s23_grid, 'nuisance': nuisance_all,
        'nuisance_names': nuisance_names, 'bf': bf, 'min_chi2': min_chi2
    }


def plot_contours(data, output_dir):
    """Plot 2D contours and 1D profiles."""
    chi2 = data['chi2']
    dm_grid = data['dm_grid'] * 1e3  # convert to 10^-3 eV^2
    s23_grid = data['s23_grid']
    bf = data['bf']
    min_chi2 = data['min_chi2']

    dchi2 = chi2 - min_chi2

    # 2D contour plot
    fig, ax = plt.subplots(figsize=(8, 7))
    S23, DM = np.meshgrid(s23_grid, dm_grid)
    levels = [2.30, 6.18, 11.83]  # 1σ, 2σ, 3σ for 2 DOF
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = ['1σ (68.3%)', '2σ (95.5%)', '3σ (99.7%)']

    cs = ax.contour(S23, DM, dchi2, levels=levels, colors=colors, linewidths=2)
    ax.contourf(S23, DM, dchi2, levels=[0, levels[0]], colors=[colors[0]], alpha=0.15)

    ax.plot(s23_grid[bf[1]], dm_grid[bf[0]], 'r*', markersize=15, label=f'Best fit (χ²={min_chi2:.1f})')
    ax.set_xlabel(r'$\sin^2\theta_{23}$', fontsize=14)
    ax.set_ylabel(r'$\Delta m^2_{31}$ [$\times 10^{-3}$ eV²]', fontsize=14)
    ax.set_title('SK 2023 Data Fit (post NC-fix): Δχ² Contours', fontsize=14)

    for i, (level, label) in enumerate(zip(levels, labels)):
        ax.plot([], [], color=colors[i], linewidth=2, label=f'{label} (Δχ²={level:.2f})')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sk_datafit_contours.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: sk_datafit_contours.png")
    plt.close()

    # 1D profiles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Dm231 profile (minimize over s23)
    dchi2_dm = np.nanmin(dchi2, axis=1)
    ax1.plot(dm_grid, dchi2_dm, 'b-', linewidth=2)
    ax1.axhline(1.0, color='gray', ls='--', alpha=0.5, label='1σ')
    ax1.axhline(4.0, color='gray', ls=':', alpha=0.5, label='2σ')
    ax1.axvline(dm_grid[bf[0]], color='r', ls='--', alpha=0.5)
    ax1.set_xlabel(r'$\Delta m^2_{31}$ [$\times 10^{-3}$ eV²]', fontsize=13)
    ax1.set_ylabel(r'$\Delta\chi^2$', fontsize=13)
    ax1.set_title('1D Profile: Δm²₃₁', fontsize=13)
    ax1.legend()
    ax1.set_ylim(-0.5, 15)
    ax1.grid(alpha=0.3)

    # s23 profile (minimize over dm)
    dchi2_s23 = np.nanmin(dchi2, axis=0)
    ax2.plot(s23_grid, dchi2_s23, 'b-', linewidth=2)
    ax2.axhline(1.0, color='gray', ls='--', alpha=0.5, label='1σ')
    ax2.axhline(4.0, color='gray', ls=':', alpha=0.5, label='2σ')
    ax2.axvline(s23_grid[bf[1]], color='r', ls='--', alpha=0.5)
    ax2.set_xlabel(r'$\sin^2\theta_{23}$', fontsize=13)
    ax2.set_ylabel(r'$\Delta\chi^2$', fontsize=13)
    ax2.set_title(r'1D Profile: $\sin^2\theta_{23}$', fontsize=13)
    ax2.legend()
    ax2.set_ylim(-0.5, 15)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sk_datafit_1d_profiles.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: sk_datafit_1d_profiles.png")
    plt.close()

    # Convergence and pull maps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    im1 = ax1.pcolormesh(s23_grid, dm_grid, data['max_pull'], cmap='YlOrRd')
    ax1.set_xlabel(r'$\sin^2\theta_{23}$')
    ax1.set_ylabel(r'$\Delta m^2_{31}$ [$\times 10^{-3}$ eV²]')
    ax1.set_title('Max pull (σ)')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.pcolormesh(s23_grid, dm_grid, data['converged'].astype(float), cmap='RdYlGn', vmin=0, vmax=1)
    ax2.set_xlabel(r'$\sin^2\theta_{23}$')
    ax2.set_ylabel(r'$\Delta m^2_{31}$ [$\times 10^{-3}$ eV²]')
    ax2.set_title('Convergence')
    plt.colorbar(im2, ax=ax2)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sk_datafit_diagnostics.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: sk_datafit_diagnostics.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Assemble SK datafit row results")
    parser.add_argument("--input-dir", required=True, help="Directory with row_*.json files")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--n-dm", type=int, default=41)
    parser.add_argument("--n-s23", type=int, default=41)
    args = parser.parse_args()

    data = assemble(args)
    if data is not None:
        print("\nGenerating plots...")
        plot_contours(data, args.output_dir)
        print("Done!")


if __name__ == "__main__":
    main()
