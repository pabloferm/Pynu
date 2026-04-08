#!/usr/bin/env python3
"""
Reformat IceCube DeepCore 9-year data release into Pynu-compatible parquet files.

Input:  IC_data_release/ directory with CSV files from IceCube data release
        (Phys. Rev. D 108, 012014, 2023)

Output:
  - IC_MC.parquet          (neutrino MC events + fake muon events at bin centers)
  - IC_data.parquet        (fake data events at bin centers with weight=count)
  - _E_reco_bins.npy       (energy bin edges)
  - _cosT_reco_bins.npy    (cos-zenith bin edges)
  - hs_nu_nc_nue_cc.csv    (hypersurface slopes: NC + CC νe)
  - hs_numu_cc.csv         (hypersurface slopes: CC νμ)
  - hs_nutau_cc.csv        (hypersurface slopes: CC ντ)

Column mapping (IceCube CSV → Pynu parquet):
  true_energy   → true_energy    (keep, GeV)
  true_coszen   → true_zenith    (arccos, radians)
  reco_energy   → reco_energy    (keep, GeV)
  reco_coszen   → reco_zenith    (arccos, radians)
  pdg           → pdg            (keep, int)
  type          → current_type   (0=NC, 1=CC)
  interaction   → interaction_type (0-4, NEUT interaction mode for xsec syst)
  pid           → pid            (<0.75 → 0=mixed, ≥0.75 → 1=tracks)
  weight        → weight         (keep, GeV cm² sr)
  (computed)    → weight_variance (weight², Poisson assumption)
  (added)       → MC_type        (1=neutrino, -1=muon)

Muon background (mc_mu.csv) is pre-binned; converted to fake events at bin centers.
Data (data.csv) is pre-binned; converted to fake events at bin centers.
"""

import numpy as np
import pandas as pd
import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Reformat IceCube DeepCore 9yr data release to Pynu parquet'
    )
    parser.add_argument('--input-dir', default='IC_data_release',
                        help='Input directory with IceCube CSV files')
    parser.add_argument('--output-dir', default='data/IceCube',
                        help='Output directory for Pynu parquet files')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # 1. Process MC neutrino events (4 CSV files → combined DataFrame)
    # =========================================================================
    print("=== Processing MC neutrino events ===")

    mc_files = ['mc_nu_nc', 'mc_nue_cc', 'mc_numu_cc', 'mc_nutau_cc']
    mc_dfs = []
    for f in mc_files:
        path = os.path.join(input_dir, f + '.csv')
        df = pd.read_csv(path)
        print(f"  {f}: {len(df)} events")
        mc_dfs.append(df)

    mc_nu = pd.concat(mc_dfs, ignore_index=True)
    print(f"  Total neutrino MC events: {len(mc_nu)}")

    # Map columns to Pynu schema
    mc_pynu = pd.DataFrame()
    mc_pynu['true_energy'] = mc_nu['true_energy'].values
    mc_pynu['true_zenith'] = np.arccos(np.clip(mc_nu['true_coszen'].values, -1, 1))
    mc_pynu['reco_energy'] = mc_nu['reco_energy'].values
    mc_pynu['reco_zenith'] = np.arccos(np.clip(mc_nu['reco_coszen'].values, -1, 1))
    mc_pynu['pdg'] = mc_nu['pdg'].values.astype(int)
    mc_pynu['current_type'] = mc_nu['type'].values.astype(int)  # 0=NC, 1=CC
    mc_pynu['interaction_type'] = mc_nu['interaction'].values.astype(int)  # 0-4 interaction mode

    # Map PID: continuous → integer
    # [0.55, 0.75) → 0 (mixed/cascade-like)
    # [0.75, 1.0]  → 1 (track-like)
    mc_pynu['pid'] = (mc_nu['pid'].values >= 0.75).astype(int)

    mc_pynu['weight'] = mc_nu['weight'].values
    mc_pynu['weight_variance'] = mc_nu['weight'].values ** 2  # Poisson assumption
    mc_pynu['MC_type'] = 1  # neutrino

    print(f"  PDG codes: {sorted(mc_pynu['pdg'].unique())}")
    print(f"  Current types: {sorted(mc_pynu['current_type'].unique())}")
    print(f"  PID (mapped): {sorted(mc_pynu['pid'].unique())}")
    print(f"  Weight range: [{mc_pynu['weight'].min():.6e}, {mc_pynu['weight'].max():.6e}]")

    # =========================================================================
    # 2. Process muon background (pre-binned → fake events at bin centers)
    # =========================================================================
    print("\n=== Processing muon background ===")

    mc_mu = pd.read_csv(os.path.join(input_dir, 'mc_mu.csv'))
    print(f"  Muon bins: {len(mc_mu)}")
    print(f"  Total muon count: {mc_mu['count'].sum():.1f}")

    mu_pynu = pd.DataFrame()
    mu_pynu['true_energy'] = mc_mu['reco_energy'].values  # dummy: use reco
    mu_pynu['true_zenith'] = np.arccos(np.clip(mc_mu['reco_coszen'].values, -1, 1))
    mu_pynu['reco_energy'] = mc_mu['reco_energy'].values
    mu_pynu['reco_zenith'] = np.arccos(np.clip(mc_mu['reco_coszen'].values, -1, 1))
    mu_pynu['pdg'] = 13  # muon PDG (dummy, not used for muon bkg)
    mu_pynu['current_type'] = 0  # dummy
    mu_pynu['interaction_type'] = -1  # dummy (muon, not used)
    mu_pynu['pid'] = (mc_mu['pid'].values >= 0.75).astype(int)
    mu_pynu['weight'] = mc_mu['count'].values  # already in event counts
    mu_pynu['weight_variance'] = mc_mu['abs_uncertainty'].values ** 2  # variance
    mu_pynu['MC_type'] = -1  # muon flag

    # =========================================================================
    # 3. Combine neutrino + muon and save MC parquet
    # =========================================================================
    mc_all = pd.concat([mc_pynu, mu_pynu], ignore_index=True)
    mc_path = os.path.join(output_dir, 'IC_MC.parquet')
    mc_all.to_parquet(mc_path, index=False)
    print(f"\n  Saved MC parquet: {mc_path}")
    print(f"  Total rows: {len(mc_all)} (nu: {len(mc_pynu)}, mu: {len(mu_pynu)})")

    # =========================================================================
    # 4. Process data (pre-binned → fake events at bin centers)
    # =========================================================================
    print("\n=== Processing data ===")

    data = pd.read_csv(os.path.join(input_dir, 'data.csv'))
    print(f"  Data bins: {len(data)}")
    print(f"  Total data events: {data['count'].sum()}")

    data_pynu = pd.DataFrame()
    data_pynu['reco_energy'] = data['reco_energy'].values
    data_pynu['reco_zenith'] = np.arccos(np.clip(data['reco_coszen'].values, -1, 1))
    data_pynu['pid'] = (data['pid'].values >= 0.75).astype(int)
    data_pynu['weight'] = data['count'].values.astype(float)

    data_path = os.path.join(output_dir, 'IC_data.parquet')
    data_pynu.to_parquet(data_path, index=False)
    print(f"  Saved data parquet: {data_path}")
    print(f"  Total rows: {len(data_pynu)}")

    # =========================================================================
    # 5. Save binning arrays for the experiment class
    # =========================================================================
    print("\n=== Saving binning arrays ===")

    # Energy bins: 10 bins (last bin double-width for statistics)
    # From readme: [6.31, 8.459, ..., 88.199, 158.49]
    E_edges = np.array([
        6.31, 8.45862141, 11.33887101, 15.19987592, 20.37559363,
        27.3136977, 36.61429921, 49.08185342, 65.79474104,
        88.19854278, 158.49
    ])

    # Cos(zenith) bins: 10 bins
    cosZ_edges = np.array([
        -1., -0.89, -0.78, -0.67, -0.56, -0.45,
        -0.34, -0.23, -0.12, -0.01, 0.1
    ])

    np.save(os.path.join(output_dir, '_E_reco_bins.npy'), E_edges)
    np.save(os.path.join(output_dir, '_cosT_reco_bins.npy'), cosZ_edges)
    print(f"  Energy bins: {len(E_edges)-1} bins, [{E_edges[0]:.2f}, {E_edges[-1]:.2f}] GeV")
    print(f"  CosZ bins: {len(cosZ_edges)-1} bins, [{cosZ_edges[0]:.2f}, {cosZ_edges[-1]:.2f}]")

    # =========================================================================
    # 6. Copy hypersurface CSV files (detector systematics)
    # =========================================================================
    print("\n=== Copying hypersurface files ===")

    import shutil
    hs_files = ['hs_nu_nc_nue_cc.csv', 'hs_numu_cc.csv', 'hs_nutau_cc.csv']
    for hs_file in hs_files:
        src = os.path.join(input_dir, hs_file)
        dst = os.path.join(output_dir, hs_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied: {hs_file}")
        else:
            print(f"  WARNING: {src} not found — skipping")

    # =========================================================================
    # 7. Summary
    # =========================================================================
    print("\n=== Summary ===")
    print(f"  MC file: {mc_path}")
    print(f"    Neutrino events: {len(mc_pynu)}")
    print(f"    Muon 'events' (pre-binned): {len(mu_pynu)}")
    print(f"  Data file: {data_path}")
    print(f"    Data 'events' (pre-binned): {len(data_pynu)}")
    print(f"  Binning: _E_reco_bins.npy, _cosT_reco_bins.npy")
    print(f"  Hypersurfaces: {', '.join(hs_files)}")
    print(f"  Output directory: {output_dir}")
    print("\nDone!")


if __name__ == '__main__':
    main()
