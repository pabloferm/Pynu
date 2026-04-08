#!/usr/bin/env python3
"""
Compute the effective PID probabilities using the IC→ORCA morphing logic
from MCGenerator.assign_topology(), and compare with:
  1. Direct ORCA topology probs (what P_pid currently uses)
  2. Event MC topology fractions (ground truth)

The morphing is a ratio-based transformation:
  Given IC fracs (p_IC_track, p_IC_cascade) and ORCA probs (p_ORCA_track, p_ORCA_cas):
  - With restricted_rand_morph=True, inter_to_tracks=False:
    The conditional assignment probabilities are computed per-event based on
    original IC topology, then averaged over the IC population.

This script derives the effective P(cascade|channel,E), P(track|channel,E),
P(intermediate|channel,E) analytically.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_probs_from_csv(filepath):
    """Load ORCA topology histogram from CSV (single column)."""
    df = pd.read_csv(filepath, header=None, usecols=[1])
    res = np.array(df.iloc[:, 0])
    if len(res) < 30:
        padding = np.zeros(30 - len(res))
        res = np.concatenate((padding, res), axis=0)
    return res


def compute_morphed_probs(p_ic_track, p_ic_cas, p_orca_track, p_orca_cas):
    """
    Compute effective topology fractions after morphing.

    Given IC topology fractions and ORCA topology probabilities,
    compute the resulting (cascade, track, intermediate) fractions
    using the assign_topology logic with restricted_rand_morph=True.

    Returns: (p_eff_cascade, p_eff_track, p_eff_intermediate)
    """
    # Handle edge cases
    if p_ic_track < 1e-10 and p_ic_cas < 1e-10:
        return 0, 0, 0

    # Case 1: IC track >= ORCA track AND IC cascade >= ORCA cascade
    if p_ic_track >= p_orca_track and p_ic_cas >= p_orca_cas:
        # IC cascade events: new_track_p=0, new_cas_p=p_orca_cas/p_ic_cas
        # → assigned as: P(cas)=p_orca_cas/p_ic_cas, P(track)=0, P(interm)=1-p_orca_cas/p_ic_cas
        if p_ic_cas > 0:
            cas_from_cas = (p_orca_cas / p_ic_cas) * p_ic_cas  # = p_orca_cas
            trk_from_cas = 0
            int_from_cas = (1 - p_orca_cas / p_ic_cas) * p_ic_cas  # = p_ic_cas - p_orca_cas
        else:
            cas_from_cas = trk_from_cas = int_from_cas = 0

        # IC track events: new_track_p=p_orca_track/p_ic_track, new_cas_p=0
        if p_ic_track > 0:
            trk_from_trk = (p_orca_track / p_ic_track) * p_ic_track  # = p_orca_track
            cas_from_trk = 0
            int_from_trk = (1 - p_orca_track / p_ic_track) * p_ic_track  # = p_ic_track - p_orca_track
        else:
            trk_from_trk = cas_from_trk = int_from_trk = 0

    # Case 2: IC track >= ORCA track AND IC cascade < ORCA cascade
    elif p_ic_track >= p_orca_track and p_ic_cas < p_orca_cas:
        # IC cascade events: new_track_p=0, new_cas_p=1
        cas_from_cas = p_ic_cas
        trk_from_cas = 0
        int_from_cas = 0

        # IC track events (restricted): new_track_p=p_orca_track/p_ic_track, new_cas_p=0
        # (restricted_rand_morph=True means we DON'T add (o_cas-i_cas)/i_track to cascade)
        if p_ic_track > 0:
            trk_from_trk = (p_orca_track / p_ic_track) * p_ic_track  # = p_orca_track
            cas_from_trk = 0  # restricted: no cross-topology borrowing
            int_from_trk = (1 - p_orca_track / p_ic_track) * p_ic_track  # = p_ic_track - p_orca_track
        else:
            trk_from_trk = cas_from_trk = int_from_trk = 0

    # Case 3: IC track < ORCA track AND IC cascade >= ORCA cascade
    elif p_ic_track < p_orca_track and p_ic_cas >= p_orca_cas:
        # IC cascade events (restricted): new_track_p=0, new_cas_p=p_orca_cas/p_ic_cas
        if p_ic_cas > 0:
            cas_from_cas = (p_orca_cas / p_ic_cas) * p_ic_cas  # = p_orca_cas
            trk_from_cas = 0  # restricted: no cross-topology borrowing
            int_from_cas = (1 - p_orca_cas / p_ic_cas) * p_ic_cas  # = p_ic_cas - p_orca_cas
        else:
            cas_from_cas = trk_from_cas = int_from_cas = 0

        # IC track events: new_track_p=1, new_cas_p=0
        trk_from_trk = p_ic_track
        cas_from_trk = 0
        int_from_trk = 0

    else:
        # Case 4: IC track < ORCA track AND IC cascade < ORCA cascade
        # (shouldn't happen if IC fractions sum to ~1 and ORCA probs are reasonable)
        cas_from_cas = p_ic_cas
        trk_from_cas = 0
        int_from_cas = 0
        trk_from_trk = p_ic_track
        cas_from_trk = 0
        int_from_trk = 0

    p_eff_cas = cas_from_cas + cas_from_trk
    p_eff_trk = trk_from_trk + trk_from_cas
    p_eff_int = int_from_cas + int_from_trk

    return p_eff_cas, p_eff_trk, p_eff_int


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orca-results-dir', required=True)
    parser.add_argument('--event-mc', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    energy_bins = np.logspace(np.log10(2), np.log10(50), 31)
    nbins = 30
    ecenters = np.sqrt(energy_bins[:-1] * energy_bins[1:])

    # Channel definitions
    channels = [
        ('nue_CC', 12, 1, 'nue', True),
        ('nuebarCC', -12, 1, 'nuebar', True),
        ('numu_CC', 14, 1, 'numu', True),
        ('numubarCC', -14, 1, 'numubar', True),
        ('nutau_CC', 16, 1, 'nutau', True),
        ('nutaubarCC', -16, 1, 'nutaubar', True),
        ('nu_NC', 12, 0, 'nu', True),     # NC: use nu
        ('nubar_NC', -12, 0, 'nubar', True),  # NC: use nubar
    ]

    # ORCA topology files mapping
    orca_files = {
        ('nue_CC', 'track'): 'track_nue_CC.csv',
        ('nue_CC', 'cascade'): 'cascade_nue_CC.csv',
        ('nuebarCC', 'track'): 'track_nuebar_CC.csv',
        ('nuebarCC', 'cascade'): 'cascade_nuebar_CC.csv',
        ('numu_CC', 'track'): 'track_numu_CC.csv',
        ('numu_CC', 'cascade'): 'cascade_numu_CC.csv',
        ('numubarCC', 'track'): 'track_numubar_CC.csv',
        ('numubarCC', 'cascade'): 'cascade_numubar_CC.csv',
        ('nutau_CC', 'track'): 'track_nutau_CC.csv',
        ('nutau_CC', 'cascade'): 'cascade_nutau_CC.csv',
        ('nutaubarCC', 'track'): 'track_nutaubar_CC.csv',
        ('nutaubarCC', 'cascade'): 'cascade_nutaubar_CC.csv',
        ('nu_NC', 'track'): 'track_nu_NC.csv',
        ('nu_NC', 'cascade'): 'cascade_nu_NC.csv',
        ('nubar_NC', 'track'): 'track_nubar_NC.csv',
        ('nubar_NC', 'cascade'): 'cascade_nubar_NC.csv',
    }

    # IC topology files mapping
    ic_files = {
        'nue_CC': ('nue_CC_Topology_Fraction', 'nu'),
        'nuebarCC': ('nue_CC_Topology_Fraction', 'nubar'),
        'numu_CC': ('numu_CC_Topology_Fraction', 'nu'),
        'numubarCC': ('numu_CC_Topology_Fraction', 'nubar'),
        'nutau_CC': ('nutau_CC_Topology_Fraction', 'nu'),
        'nutaubarCC': ('nutau_CC_Topology_Fraction', 'nubar'),
        'nu_NC': ('nu_NC_Topology_Fraction', 'nu'),
        'nubar_NC': ('nu_NC_Topology_Fraction', 'nubar'),
    }

    # Compute morphed PID for each channel and energy bin
    results = {}
    for ch_name, pdg, current, label, _ in channels:
        # Load ORCA probs
        orca_track = get_probs_from_csv(
            os.path.join(args.orca_results_dir, orca_files[(ch_name, 'track')]))
        orca_cascade = get_probs_from_csv(
            os.path.join(args.orca_results_dir, orca_files[(ch_name, 'cascade')]))

        # ORCA returns: track[idx], 1 - cascade[idx]
        # So p_orca_track = track[idx], p_orca_cas = 1 - cascade[idx]
        p_orca_track = orca_track
        p_orca_cas = 1 - orca_cascade  # note: get_ORCA_topology_prob returns (track, 1-cascade)

        # Load IC fracs
        ic_filename, nutype = ic_files[ch_name]
        ic_df = pd.read_csv(os.path.join(args.orca_results_dir, ic_filename))
        if nutype == 'nu':
            p_ic_track = ic_df['nu_track'].values
            p_ic_cas = ic_df['nu_cas'].values
        else:
            p_ic_track = ic_df['nubar_track'].values
            p_ic_cas = ic_df['nubar_cas'].values

        # Compute morphed probs for each energy bin
        morphed_cas = np.zeros(nbins)
        morphed_trk = np.zeros(nbins)
        morphed_int = np.zeros(nbins)

        for i in range(nbins):
            c, t, inter = compute_morphed_probs(
                p_ic_track[i], p_ic_cas[i],
                p_orca_track[i], p_orca_cas[i])
            morphed_cas[i] = c
            morphed_trk[i] = t
            morphed_int[i] = inter

        results[ch_name] = {
            'morphed_cas': morphed_cas,
            'morphed_trk': morphed_trk,
            'morphed_int': morphed_int,
            'orca_track': p_orca_track,
            'orca_cas': p_orca_cas,
            'ic_track': p_ic_track,
            'ic_cas': p_ic_cas,
        }

    # Load event MC for comparison
    print("Loading event MC...")
    mc = pd.read_csv(args.event_mc)
    print(f"  {len(mc)} events")

    mc_fracs = {}
    for ch_name, pdg, current, label, _ in channels:
        if current == 0:  # NC
            if pdg > 0:
                mask = (mc['pdg'].values > 0) & (mc['current_type'].values == 0)
            else:
                mask = (mc['pdg'].values < 0) & (mc['current_type'].values == 0)
        else:
            mask = (mc['pdg'].values == pdg) & (mc['current_type'].values == current)

        sub = mc[mask]
        E = sub['true_energy'].values
        pids = sub['pid'].values.astype(int)
        weights = sub['weight'].values

        all_hist, _ = np.histogram(E, bins=energy_bins, weights=weights)
        trk_hist, _ = np.histogram(E, bins=energy_bins, weights=weights * (pids == 1))
        cas_hist, _ = np.histogram(E, bins=energy_bins, weights=weights * (pids == 0))
        int_hist, _ = np.histogram(E, bins=energy_bins, weights=weights * (pids == 2))

        with np.errstate(divide='ignore', invalid='ignore'):
            trk_frac = np.where(all_hist > 0, trk_hist / all_hist, 0)
            cas_frac = np.where(all_hist > 0, cas_hist / all_hist, 0)
            int_frac = np.where(all_hist > 0, int_hist / all_hist, 0)

        mc_fracs[ch_name] = {
            'track': trk_frac,
            'cascade': cas_frac,
            'intermediate': int_frac,
        }

    # ================================================================
    # Plot: Morphed PID vs ORCA-direct vs Event MC — per channel
    # ================================================================
    channel_pairs = [
        ('nue_CC', 'nuebarCC', r'$\nu_e$ CC'),
        ('numu_CC', 'numubarCC', r'$\nu_\mu$ CC'),
        ('nutau_CC', 'nutaubarCC', r'$\nu_\tau$ CC'),
        ('nu_NC', 'nubar_NC', r'NC'),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(18, 24))

    for row, (nu_ch, nubar_ch, title) in enumerate(channel_pairs):
        for col, (ch_name, nutype_label) in enumerate([(nu_ch, r'$\nu$'), (nubar_ch, r'$\bar{\nu}$')]):
            ax = axes[row, col]
            r = results[ch_name]
            mc_f = mc_fracs[ch_name]

            # Track fraction
            ax.step(energy_bins[:-1], mc_f['track'], where='post',
                    color='blue', linewidth=2, label='Track (Event MC)')
            ax.step(energy_bins[:-1], r['morphed_trk'], where='post',
                    color='blue', linewidth=1.5, linestyle='--',
                    label='Track (Morphed)')
            ax.step(energy_bins[:-1], r['orca_track'][:nbins], where='post',
                    color='blue', linewidth=1, linestyle=':',
                    label='Track (ORCA direct)')

            # Cascade fraction
            ax.step(energy_bins[:-1], mc_f['cascade'], where='post',
                    color='red', linewidth=2, label='Cascade (Event MC)')
            ax.step(energy_bins[:-1], r['morphed_cas'], where='post',
                    color='red', linewidth=1.5, linestyle='--',
                    label='Cascade (Morphed)')
            ax.step(energy_bins[:-1], r['orca_cas'][:nbins], where='post',
                    color='red', linewidth=1, linestyle=':',
                    label='Cascade (ORCA direct)')

            # Intermediate fraction
            ax.step(energy_bins[:-1], mc_f['intermediate'], where='post',
                    color='green', linewidth=2, label='Intermediate (Event MC)')
            ax.step(energy_bins[:-1], r['morphed_int'], where='post',
                    color='green', linewidth=1.5, linestyle='--',
                    label='Intermediate (Morphed)')

            ax.set_xscale('log')
            ax.set_xlabel(r'$E_\nu$ [GeV]')
            ax.set_ylabel('Fraction')
            ax.set_xlim(2, 50)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f'{title} {nutype_label}', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    outpath = os.path.join(args.output_dir, 'morphed_pid_comparison.png')
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved {outpath}")

    # ================================================================
    # Print numerical summary
    # ================================================================
    print("\n=== Morphed PID vs Event MC (energy-averaged) ===")
    print(f"{'Channel':<15} | {'Topo':<12} | {'Event MC':>10} | {'Morphed':>10} | {'ORCA direct':>12} | {'Morph/MC':>10} | {'ORCA/MC':>10}")
    print("-" * 95)

    for ch_name, pdg, current, label, _ in channels:
        r = results[ch_name]
        mc_f = mc_fracs[ch_name]

        for topo, mc_key, morph_key, orca_key in [
            ('track', 'track', 'morphed_trk', 'orca_track'),
            ('cascade', 'cascade', 'morphed_cas', 'orca_cas'),
            ('intermediate', 'intermediate', 'morphed_int', None),
        ]:
            mc_avg = np.mean(mc_f[mc_key])
            morph_avg = np.mean(r[morph_key])
            if orca_key:
                orca_avg = np.mean(r[orca_key][:nbins])
            else:
                # ORCA direct doesn't have intermediate
                orca_avg = np.mean(1 - r['orca_track'][:nbins] - r['orca_cas'][:nbins])

            r_morph = morph_avg / mc_avg if mc_avg > 0 else 0
            r_orca = orca_avg / mc_avg if mc_avg > 0 else 0
            flag = " ***" if abs(r_morph - 1) > 0.15 else ""
            print(f"{ch_name:<15} | {topo:<12} | {mc_avg:10.4f} | {morph_avg:10.4f} | "
                  f"{orca_avg:12.4f} | {r_morph:10.3f} | {r_orca:10.3f}{flag}")

    print("\nDone!")


if __name__ == '__main__':
    main()
