#!/usr/bin/env python3
"""
Generate ORCA-Full meta-event parquet from response matrices.

Creates a parquet file in the same column format as ORCA-6 data release,
where each row represents one (true_E, true_cz, channel) -> (reco_E, reco_cz, pid)
response matrix cell with a combined weight.

Weight formula:
  weight = Aeff[i_E, ch] * P_pid[pid, i_E, ch] * R_E[i_E, j_E, topo(pid)] * R_cz[m, k, i_E, res_group(pid, sign)]

Output: parquet file with columns matching ORCA-6 format.
"""

import argparse
import os
import numpy as np
import pandas as pd


# ============================================================
# Channel definitions (must match build_response_matrices.py)
# ============================================================

CHANNEL_NAMES = ['nueCC', 'nuebarCC', 'numuCC', 'numubarCC',
                 'nutauCC', 'nutaubarCC', 'nuNC', 'nubarNC']

# Channel -> (pdg, current_type)
CHANNEL_PDG_CURRENT = {
    0: (12, 1),    # nueCC
    1: (-12, 1),   # nuebarCC
    2: (14, 1),    # numuCC
    3: (-14, 1),   # numubarCC
    4: (16, 1),    # nutauCC
    5: (-16, 1),   # nutaubarCC
    6: (14, 0),    # nuNC (pdg arbitrary for NC)
    7: (-14, 0),   # nubarNC
}


def pid_to_topo(pid):
    """Map PID class to energy migration topology index.
    cascade(0)->0, track(1)->1, intermediate(2)->-1 (50/50 mix).

    The event MC (MCGenerator.py) uses np.random.randint(2) for intermediate,
    giving 50% cascade and 50% track energy migration. We encode this as -1
    and handle the averaging in the weight calculation."""
    if pid == 0:
        return 0
    elif pid == 1:
        return 1
    else:
        return -1  # intermediate: 50/50 mix of cascade and track


def res_group(pid, pdg):
    """Map (PID, nu/nubar sign) to zenith resolution group index.
    cascade+nu:0, cascade+nubar:1, track+nu:2, track+nubar:3,
    intermediate+nu:4, intermediate+nubar:5."""
    is_nubar = 1 if pdg < 0 else 0
    if pid == 0:  # cascade
        return is_nubar  # 0 or 1
    elif pid == 1:  # track
        return 2 + is_nubar  # 2 or 3
    else:  # intermediate
        return 4 + is_nubar  # 4 or 5


def generate_meta_parquet(response_file, output_dir):
    """Generate meta-event parquet from response matrices."""

    print("Loading response matrices...")
    data = np.load(response_file, allow_pickle=True)
    R_E = data['R_E']            # (N_Etrue, N_Ereco, 2)
    R_cz = data['R_cz']          # (N_cz, N_cz, N_Etrue, N_groups)
    P_pid = data['P_pid']        # (3, N_Etrue, 8)
    Aeff = data['Aeff']          # (N_Etrue, 8)
    E_true_edges = data['E_true_edges']
    E_reco_edges = data['E_reco_edges']
    cz_edges = data['cz_true_edges']  # same for true and reco

    N_Etrue = len(E_true_edges) - 1
    N_Ereco = len(E_reco_edges) - 1
    N_cz = len(cz_edges) - 1
    N_channels = 8
    N_pids = 3

    # Bin centers
    E_true_centers = np.sqrt(E_true_edges[:-1] * E_true_edges[1:])
    E_reco_centers = np.sqrt(E_reco_edges[:-1] * E_reco_edges[1:])
    cz_centers = 0.5 * (cz_edges[:-1] + cz_edges[1:])

    print(f"  R_E shape: {R_E.shape}")
    print(f"  R_cz shape: {R_cz.shape}")
    print(f"  P_pid shape: {P_pid.shape}")
    print(f"  Aeff shape: {Aeff.shape}")
    print(f"  N_Etrue={N_Etrue}, N_Ereco={N_Ereco}, N_cz={N_cz}")

    # Check R_cz has 6 groups
    n_groups = R_cz.shape[3]
    if n_groups != 6:
        print(f"  WARNING: R_cz has {n_groups} groups, expected 6. "
              f"Re-run build_response_matrices.py first.")
        if n_groups == 4:
            print("  Falling back: intermediate will use cascade resolution")

    # Build meta-events
    rows = []
    weight_threshold = 1e-30  # skip negligible weights

    for ch in range(N_channels):
        pdg, current_type = CHANNEL_PDG_CURRENT[ch]

        for i_E in range(N_Etrue):
            aeff = Aeff[i_E, ch]
            if aeff <= 0:
                continue

            for m_cz in range(N_cz):
                for pid in range(N_pids):
                    p_pid = P_pid[pid, i_E, ch]
                    if p_pid <= 0:
                        continue

                    topo = pid_to_topo(pid)
                    rg = res_group(pid, pdg)
                    # Clamp rg to available groups
                    if rg >= n_groups:
                        rg = rg - 4  # fall back to cascade/track group

                    for j_E in range(N_Ereco):
                        if topo >= 0:
                            p_E = R_E[i_E, j_E, topo]
                        else:
                            # Intermediate: 50/50 average of cascade and track
                            p_E = 0.5 * (R_E[i_E, j_E, 0] + R_E[i_E, j_E, 1])
                        if p_E <= 0:
                            continue

                        for k_cz in range(N_cz):
                            p_cz = R_cz[m_cz, k_cz, i_E, rg]
                            if p_cz <= 0:
                                continue

                            w = aeff * p_pid * p_E * p_cz
                            if w < weight_threshold:
                                continue

                            # Convert cz to zenith angle
                            true_zen = np.arccos(cz_centers[m_cz])
                            reco_zen = np.arccos(cz_centers[k_cz])

                            rows.append({
                                'true_energy': E_true_centers[i_E],
                                'true_zenith': true_zen,
                                'reco_energy': E_reco_centers[j_E],
                                'reco_zenith': reco_zen,
                                'pdg': pdg,
                                'current_type': current_type,
                                'pid': pid,
                                'weight': w,
                                'weight_variance': w * w,  # deterministic: 1 effective MC event
                                'MC_type': 1,
                                'interaction_type': current_type,  # 0=NC, 1=CC
                            })

        print(f"  Channel {CHANNEL_NAMES[ch]}: {len(rows)} rows so far")

    print(f"\nTotal meta-events: {len(rows)}")

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Ensure correct dtypes
    df['pdg'] = df['pdg'].astype(np.int32)
    df['current_type'] = df['current_type'].astype(np.int32)
    df['pid'] = df['pid'].astype(np.int32)
    df['MC_type'] = df['MC_type'].astype(np.int32)
    df['interaction_type'] = df['interaction_type'].astype(np.int32)

    # Save parquet
    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, 'ORCA_full_MC.parquet')
    df.to_parquet(parquet_path, index=False)
    print(f"\nSaved parquet: {parquet_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Save bin edge files (for Pynu Binning() to load)
    np.save(os.path.join(output_dir, '_E_reco_bins.npy'), E_reco_edges)
    np.save(os.path.join(output_dir, '_cosT_reco_bins.npy'), cz_edges)
    print(f"  Saved _E_reco_bins.npy ({len(E_reco_edges)} edges)")
    print(f"  Saved _cosT_reco_bins.npy ({len(cz_edges)} edges)")

    # Summary statistics
    print(f"\n=== Summary ===")
    total_weight = df['weight'].sum()
    print(f"  Total weight (sum of Aeff*response): {total_weight:.6e}")
    for pid in range(3):
        pid_w = df[df['pid'] == pid]['weight'].sum()
        pid_names = ['cascade', 'track', 'intermediate']
        print(f"  {pid_names[pid]}: {pid_w:.6e} ({100*pid_w/total_weight:.1f}%)")

    print(f"\n  Unique true_energy values: {df['true_energy'].nunique()}")
    print(f"  Unique reco_energy values: {df['reco_energy'].nunique()}")
    print(f"  Unique true_zenith values: {df['true_zenith'].nunique()}")
    print(f"  Unique reco_zenith values: {df['reco_zenith'].nunique()}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Generate ORCA-Full meta-event parquet')
    parser.add_argument('--response-file', type=str, required=True,
                        help='Path to orca_full_response.npz')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for parquet and bin files')
    args = parser.parse_args()

    generate_meta_parquet(args.response_file, args.output_dir)
    print("\nDone!")


if __name__ == '__main__':
    main()
