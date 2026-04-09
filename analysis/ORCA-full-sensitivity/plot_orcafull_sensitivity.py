#!/usr/bin/env python3
"""
Plot ORCAFull Standard Oscillation Sensitivity Contours

Generates contour plots in the (sin²θ₂₃, Δm²₃₁) plane from a 2D grid scan.

Usage:
    python plot_orcafull_sensitivity.py --input-dir DIR [--output-dir DIR]
"""

import sys
import os
import argparse
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--title", default="ORCA-Full 5yr Sensitivity (Asimov)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir

    # Load data
    chi2_grid = np.load(os.path.join(args.input_dir, "chi2_grid.npy"))
    dm_grid = np.load(os.path.join(args.input_dir, "dm_grid.npy"))
    s23_grid = np.load(os.path.join(args.input_dir, "s23_grid.npy"))
    converged_grid = np.load(os.path.join(args.input_dir, "converged_grid.npy"))

    with open(os.path.join(args.input_dir, "metadata.json"), 'r') as f:
        meta = json.load(f)

    min_chi2 = np.nanmin(chi2_grid)
    dchi2 = chi2_grid - min_chi2
    bf_idx = np.unravel_index(np.nanargmin(chi2_grid), chi2_grid.shape)
    bf_dm = dm_grid[bf_idx[0]]
    bf_s23 = s23_grid[bf_idx[1]]

    truth_dm = meta.get('truth_dm', 2.511e-3)
    truth_s23 = meta.get('truth_s23', 0.572)
    n_conv = converged_grid.sum()
    n_total = converged_grid.size

    print(f"Grid: {len(dm_grid)}x{len(s23_grid)}")
    print(f"Min chi2: {min_chi2:.4f}")
    print(f"Best-fit: Dm231={bf_dm:.4e}, S23={bf_s23:.4f}")
    print(f"Convergence: {n_conv}/{n_total} ({100*n_conv/n_total:.1f}%)")

    # 2 DOF confidence levels
    cl_levels = {
        r'1$\sigma$ (68.3%)': 2.30,
        '90% CL': 4.61,
        r'2$\sigma$ (95.5%)': 5.99,
        r'3$\sigma$ (99.7%)': 11.83,
    }
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd']

    # ---- Plot 1: 2D contour ----
    fig, ax = plt.subplots(figsize=(8, 7))

    S23, DM = np.meshgrid(s23_grid, dm_grid * 1e3)

    # Filled contour background
    levels_fill = [0, 2.30, 4.61, 5.99, 11.83, np.nanmax(dchi2) + 1]
    cmap = plt.cm.Blues_r
    cf = ax.contourf(S23, DM, dchi2, levels=levels_fill, cmap=cmap, alpha=0.4)

    # Contour lines
    for (label, level), color in zip(cl_levels.items(), colors):
        cs = ax.contour(S23, DM, dchi2, levels=[level], colors=[color], linewidths=2)
        ax.plot([], [], color=color, linewidth=2, label=f'{label}')

    # Best-fit and truth markers
    ax.plot(bf_s23, bf_dm * 1e3, 'r*', markersize=15, zorder=10,
            label=f'Best fit ({bf_s23:.3f}, {bf_dm*1e3:.3f})')
    ax.plot(truth_s23, truth_dm * 1e3, 'k+', markersize=12, markeredgewidth=2,
            zorder=10, label=f'Truth ({truth_s23}, {truth_dm*1e3:.3f})')

    ax.set_xlabel(r'$\sin^2\theta_{23}$', fontsize=14)
    ax.set_ylabel(r'$\Delta m^2_{31}$ [$\times 10^{-3}$ eV$^2$]', fontsize=14)
    ax.set_title(args.title, fontsize=16)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    out1 = os.path.join(args.output_dir, "orcafull_sensitivity_contours.png")
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"Saved: {out1}")
    plt.close(fig)

    # ---- Plot 2: 1D profile in Dm231 (profiled over S23) ----
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

    dm_profile = np.nanmin(dchi2, axis=1)
    ax2a.plot(dm_grid * 1e3, dm_profile, 'b-', linewidth=2)
    ax2a.axvline(truth_dm * 1e3, color='r', linestyle='--', alpha=0.5, label='Truth')

    cl_1d = {r'1$\sigma$': 1.0, '90%': 2.71, r'2$\sigma$': 4.0}
    colors_1d = ['green', 'orange', 'red']
    for (label, level), c in zip(cl_1d.items(), colors_1d):
        ax2a.axhline(level, color=c, linestyle=':', alpha=0.6, label=f'{label} ({level})')

    ax2a.set_xlabel(r'$\Delta m^2_{31}$ [$\times 10^{-3}$ eV$^2$]', fontsize=14)
    ax2a.set_ylabel(r'$\Delta\chi^2$', fontsize=14)
    ax2a.set_title(r'$\Delta m^2_{31}$ Profile (profiled over $\sin^2\theta_{23}$)', fontsize=13)
    ax2a.legend(fontsize=10)
    ax2a.set_ylim(bottom=-0.5)
    ax2a.tick_params(labelsize=12)

    # 1D profile in S23 (profiled over Dm231)
    s23_profile = np.nanmin(dchi2, axis=0)
    ax2b.plot(s23_grid, s23_profile, 'b-', linewidth=2)
    ax2b.axvline(truth_s23, color='r', linestyle='--', alpha=0.5, label='Truth')

    for (label, level), c in zip(cl_1d.items(), colors_1d):
        ax2b.axhline(level, color=c, linestyle=':', alpha=0.6, label=f'{label} ({level})')

    ax2b.set_xlabel(r'$\sin^2\theta_{23}$', fontsize=14)
    ax2b.set_ylabel(r'$\Delta\chi^2$', fontsize=14)
    ax2b.set_title(r'$\sin^2\theta_{23}$ Profile (profiled over $\Delta m^2_{31}$)', fontsize=13)
    ax2b.legend(fontsize=10)
    ax2b.set_ylim(bottom=-0.5)
    ax2b.tick_params(labelsize=12)

    plt.tight_layout()
    out2 = os.path.join(args.output_dir, "orcafull_sensitivity_profiles.png")
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out2}")
    plt.close(fig2)

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print("CONTOUR EXTENT SUMMARY (1 DOF profiled)")
    print("=" * 60)

    dm_1sig = dm_grid[dm_profile < 1.0]
    s23_1sig = s23_grid[s23_profile < 1.0]
    if len(dm_1sig) > 0:
        print(f"  1σ Dm231: [{dm_1sig[0]*1e3:.4f}, {dm_1sig[-1]*1e3:.4f}] x10^-3 "
              f"(width {(dm_1sig[-1]-dm_1sig[0])*1e3:.4f})")
    if len(s23_1sig) > 0:
        print(f"  1σ S23:   [{s23_1sig[0]:.4f}, {s23_1sig[-1]:.4f}] "
              f"(width {s23_1sig[-1]-s23_1sig[0]:.4f})")

    dm_2sig = dm_grid[dm_profile < 4.0]
    s23_2sig = s23_grid[s23_profile < 4.0]
    if len(dm_2sig) > 0:
        print(f"  2σ Dm231: [{dm_2sig[0]*1e3:.4f}, {dm_2sig[-1]*1e3:.4f}] x10^-3 "
              f"(width {(dm_2sig[-1]-dm_2sig[0])*1e3:.4f})")
    if len(s23_2sig) > 0:
        print(f"  2σ S23:   [{s23_2sig[0]:.4f}, {s23_2sig[-1]:.4f}] "
              f"(width {s23_2sig[-1]-s23_2sig[0]:.4f})")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
