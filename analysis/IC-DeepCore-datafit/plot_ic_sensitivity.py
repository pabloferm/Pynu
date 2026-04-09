#!/usr/bin/env python3
"""
Plot IceCube DeepCore Sensitivity Contours from 2D Profile Scan

Generates:
1. 2D contour plot in (sin²θ₂₃, Δm²₃₁) plane with CL contours
2. 1D Δm²₃₁ profile (marginalized over sin²θ₂₃)
3. 1D sin²θ₂₃ profile (marginalized over Δm²₃₁)

Usage:
    python plot_ic_sensitivity.py --input-dir DIR [--output-dir DIR]
"""

import sys
import os
import argparse
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot IC DeepCore sensitivity contours")
    parser.add_argument("--input-dir", required=True, help="Directory with grid scan results")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots (default: input-dir)")
    parser.add_argument("--title", default=None, help="Plot title (default: auto-detect)")
    parser.add_argument("--no-truth", action="store_true", help="Do not plot truth marker (for data fits)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir

    # Load data
    chi2_grid = np.load(os.path.join(args.input_dir, "chi2_grid.npy"))
    dm_grid = np.load(os.path.join(args.input_dir, "dm_grid.npy"))
    s23_grid = np.load(os.path.join(args.input_dir, "s23_grid.npy"))
    converged_grid = np.load(os.path.join(args.input_dir, "converged_grid.npy"))

    with open(os.path.join(args.input_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)

    min_chi2 = np.nanmin(chi2_grid)
    delta_chi2 = chi2_grid - min_chi2

    best_idx = np.unravel_index(np.nanargmin(chi2_grid), chi2_grid.shape)
    best_dm231 = dm_grid[best_idx[0]]
    best_s23 = s23_grid[best_idx[1]]

    truth_dm = metadata.get("truth", {}).get("dm231", 2.511e-3)
    truth_s23 = metadata.get("truth", {}).get("sin2theta23", 0.572)
    n_converged = np.sum(converged_grid)
    n_dm = len(dm_grid)
    n_s23 = len(s23_grid)

    print(f"Grid: {n_dm}x{n_s23}")
    print(f"Min chi2: {min_chi2:.4f}")
    print(f"Best-fit: Dm231={best_dm231:.4e}, Sin2Theta23={best_s23:.4f}")
    print(f"Convergence: {n_converged}/{n_dm*n_s23} ({100*n_converged/(n_dm*n_s23):.1f}%)")

    # Scale Dm231 to 10^-3 for plotting
    dm_plot = dm_grid * 1e3

    # Confidence levels for 2 DOF
    cl_levels = {
        r'68.3% (1$\sigma$)': 2.30,
        '90%': 4.61,
        r'95.5% (2$\sigma$)': 6.18,
        r'99.7% (3$\sigma$)': 11.83,
    }

    # =========================================================================
    # Plot 1: 2D contour plot — x=sin²θ₂₃, y=Δm²₃₁
    # =========================================================================
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # chi2_grid shape is (n_dm, n_s23), so for contour with x=s23, y=dm:
    # contour(X, Y, Z) where X=s23, Y=dm, Z=delta_chi2
    levels_fill = np.array([0, 2.30, 4.61, 6.18, 11.83, 25, 50])
    cf = ax.contourf(s23_grid, dm_plot, delta_chi2, levels=levels_fill,
                     cmap='Blues_r', extend='max')
    cbar = plt.colorbar(cf, ax=ax, label=r'$\Delta\chi^2$')

    # Draw contour lines at CL levels
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    for (label, level), color in zip(cl_levels.items(), colors):
        cs = ax.contour(s23_grid, dm_plot, delta_chi2, levels=[level],
                        colors=[color], linewidths=2)
        ax.plot([], [], color=color, linewidth=2, label=f'{label} ({level:.2f})')

    # Mark best-fit point
    ax.plot(best_s23, best_dm231*1e3, 'r*', markersize=15,
            label='Best fit', zorder=10)

    # Mark truth
    if not args.no_truth:
        ax.plot(truth_s23, truth_dm*1e3, 'k+', markersize=12, markeredgewidth=2,
                label=f'Truth ({truth_s23:.3f}, {truth_dm*1e3:.3f})', zorder=10)

    ax.set_xlabel(r'$\sin^2\theta_{23}$', fontsize=14)
    ax.set_ylabel(r'$\Delta m^2_{31}$ [$\times 10^{-3}$ eV$^2$]', fontsize=14)
    if args.title:
        plot_title = args.title
    elif 'orca' in args.input_dir.lower():
        plot_title = 'ORCA Data Fit'
    else:
        plot_title = 'IceCube DeepCore 9yr Sensitivity (Asimov)'
    ax.set_title(plot_title, fontsize=16)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "ic_sensitivity_contours.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)

    # =========================================================================
    # Plot 2: 1D Δm²₃₁ profile (marginalized over sin²θ₂₃)
    # =========================================================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))

    dm_profile = np.nanmin(chi2_grid, axis=1)  # minimize over s23 columns
    dm_dchi2 = dm_profile - np.nanmin(dm_profile)

    ax2.plot(dm_plot, dm_dchi2, 'b-', linewidth=2)
    if not args.no_truth:
        ax2.axvline(truth_dm*1e3, color='r', linestyle='--', alpha=0.5, label='Truth')

    # 1 DOF CL levels
    cl_1d = {r'1$\sigma$ (1.00)': 1.0, '90% (2.71)': 2.71, r'2$\sigma$ (4.00)': 4.0}
    colors_1d = ['green', 'orange', 'red']
    for (label, level), color in zip(cl_1d.items(), colors_1d):
        ax2.axhline(level, color=color, linestyle=':', alpha=0.6, label=label)

    ax2.set_xlabel(r'$\Delta m^2_{31}$ [$\times 10^{-3}$ eV$^2$]', fontsize=14)
    ax2.set_ylabel(r'$\Delta\chi^2$', fontsize=14)
    ax2.set_title(r'1D Profile: $\Delta m^2_{31}$ (marginalized over $\sin^2\theta_{23}$)', fontsize=16)
    ax2.legend(fontsize=10)
    ax2.set_ylim(bottom=-0.5)
    ax2.tick_params(labelsize=12)

    plt.tight_layout()
    out_path2 = os.path.join(args.output_dir, "ic_1d_profile_dm231.png")
    fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path2}")
    plt.close(fig2)

    # =========================================================================
    # Plot 3: 1D sin²θ₂₃ profile (marginalized over Δm²₃₁)
    # =========================================================================
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5))

    s23_profile = np.nanmin(chi2_grid, axis=0)  # minimize over dm rows
    s23_dchi2 = s23_profile - np.nanmin(s23_profile)

    ax3.plot(s23_grid, s23_dchi2, 'b-', linewidth=2)
    if not args.no_truth:
        ax3.axvline(truth_s23, color='r', linestyle='--', alpha=0.5, label='Truth')

    for (label, level), color in zip(cl_1d.items(), colors_1d):
        ax3.axhline(level, color=color, linestyle=':', alpha=0.6, label=label)

    ax3.set_xlabel(r'$\sin^2\theta_{23}$', fontsize=14)
    ax3.set_ylabel(r'$\Delta\chi^2$', fontsize=14)
    ax3.set_title(r'1D Profile: $\sin^2\theta_{23}$ (marginalized over $\Delta m^2_{31}$)', fontsize=16)
    ax3.legend(fontsize=10)
    ax3.set_ylim(bottom=-0.5)
    ax3.tick_params(labelsize=12)

    plt.tight_layout()
    out_path3 = os.path.join(args.output_dir, "ic_1d_profile_sin2theta23.png")
    fig3.savefig(out_path3, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path3}")
    plt.close(fig3)

    # Print contour extent info
    print("\n" + "=" * 60)
    print("CONTOUR EXTENT SUMMARY")
    print("=" * 60)
    for label, level in cl_levels.items():
        mask = delta_chi2 <= level
        if mask.any():
            i_in, j_in = np.where(mask)
            dm_range = (dm_grid[i_in.min()]*1e3, dm_grid[i_in.max()]*1e3)
            s23_range = (s23_grid[j_in.min()], s23_grid[j_in.max()])
            print(f"  {label}:")
            print(f"    Dm231 range: [{dm_range[0]:.4f}, {dm_range[1]:.4f}] x10^-3 eV2")
            print(f"    Sin2Theta23 range: [{s23_range[0]:.4f}, {s23_range[1]:.4f}]")
        else:
            print(f"  {label}: No points within contour")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
