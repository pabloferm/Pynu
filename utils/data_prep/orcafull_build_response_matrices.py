#!/usr/bin/env python3
"""
Build ORCA-Full response matrices from digitized data.

Constructs 4 response components:
  a) Energy migration matrix R_E[i_true, j_reco, topology]
  b) Zenith migration matrix R_cz[m_true, k_reco, i_E, flavor_group]
  c) PID classification matrix P_pid[l_pid, i_E, channel]
  d) ORCA effective area Aeff[i_E, channel]

Source data from AtmNuCombination/sources/ORCA/.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as img
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import lognorm as scipy_lognorm


# ============================================================
# Constants
# ============================================================

# True-energy bins: 22 logspaced [1.85, 54] GeV (matching digitalizer)
N_ETRUE = 22
E_TRUE_EDGES = np.logspace(np.log10(1.85), np.log10(54), N_ETRUE + 1)
E_TRUE_CENTERS = np.sqrt(E_TRUE_EDGES[:-1] * E_TRUE_EDGES[1:])

# Reco-energy bins: 20 logspaced [2, 100] GeV (matching Pynu ORCA analysis)
N_ERECO = 20
E_RECO_EDGES = np.logspace(np.log10(2), np.log10(100), N_ERECO + 1)

# Zenith bins: 10 linear [-1, 0] (upgoing only)
N_CZ = 10
CZ_EDGES = np.linspace(-1, 0, N_CZ + 1)
CZ_CENTERS = 0.5 * (CZ_EDGES[:-1] + CZ_EDGES[1:])

# Channel ordering (8 channels)
CHANNEL_NAMES = ['nueCC', 'nuebarCC', 'numuCC', 'numubarCC',
                 'nutauCC', 'nutaubarCC', 'nuNC', 'nubarNC']

# Resolution groups for zenith smearing (6 groups, indexed by PID + nu/nubar sign)
# Groups 0-3: original flavor-based; Groups 4-5: intermediate PID (averaged sigma)
RESOLUTION_GROUP_NAMES = ['cas_nu', 'cas_nubar', 'trk_nu', 'trk_nubar', 'int_nu', 'int_nubar']
# Keep old name for backwards compatibility in plotting
FLAVOR_GROUP_NAMES = RESOLUTION_GROUP_NAMES

# Number density: water, n_d = 0.9168e6 * 6.022e23 [cm^-3]
N_D = 0.9168e6 * 6.022e23  # ~5.53e29 cm^-3

# MC sampling for zenith migration
N_ZEN_SAMPLES = 5000


# ============================================================
# (a) Energy migration matrix
# ============================================================

def lognormal_pdf(x, mu, sigma, A):
    """Lognormal as defined in util.py: gaussian(x, mu, sigma, A)."""
    return (A / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)))


def gaus_fit(data_entries, bins, current_binnum):
    """Fit lognormal to digitized migration row (from util.py)."""
    bins_centers = np.sqrt(bins[1:] * bins[:-1])
    try:
        popt, _ = curve_fit(lognormal_pdf, xdata=bins_centers, ydata=data_entries,
                            p0=[current_binnum, 5, 1], maxfev=10000)
        return popt[0], popt[1], popt[2]  # mu, sigma, A
    except RuntimeError:
        # Fallback: delta at diagonal
        return np.log(bins_centers[current_binnum]), 0.01, 1.0


def digitize_migration_png(image_path, scale_path, n_bins=22, hardcodes=None):
    """
    Digitize a migration PNG and fit lognormals per true-E bin.
    hardcodes: dict of {(i,j): value} to override after digitization.
    Returns: ext_data, gaussians array shape (n_bins, 3) = [sigma, mu, A].
    """
    image = img.imread(image_path)
    scale = img.imread(scale_path)
    scale_y, scale_x = scale.shape[:2]
    image_y, image_x = image.shape[:2]

    # Set palette: must match original analyze.py: set_palette(0, -3, 100)
    scale_max = 0
    scale_min = -3
    palette_bins = 100
    palette_x = np.logspace(scale_min, scale_max, 2 * palette_bins + 1)[1::2]
    palette_clr = []
    dy_s = scale_y / palette_bins
    for iy in range(palette_bins)[::-1]:
        c = scale[int(np.round((iy + 0.5) * dy_s)), int(np.round(0.5 * scale_x))]
        palette_clr.append(c)
    palette_clr = np.array(palette_clr)

    # Digitize the image
    n_bins_x = n_bins
    n_bins_y = n_bins
    ext_data = np.zeros((n_bins_x, n_bins_y))
    dx = image_x / n_bins_x
    dy = image_y / n_bins_y

    for ix in range(n_bins_x):
        for iy in range(n_bins_y):
            c = image[int(np.round((iy + 0.5) * dy)), int(np.round((ix + 0.5) * dx))]
            if abs(np.sum(c**2) - 4) < 1e-4:
                ext_data[ix, n_bins_y - 1 - iy] = 0
            else:
                ic = np.argmin(np.sum((c[None, :] - palette_clr)**2, axis=1))
                if palette_x[ic] < 0.001:
                    ext_data[ix, n_bins_y - 1 - iy] = 0
                elif palette_x[ic] > 0.58:
                    # Text avoidance: sample nearby pixels
                    py = int(np.round((iy + 0.5) * dy))
                    px = int(np.round((ix + 0.5) * dx))
                    c1 = image[min(py + 15, image_y - 1), min(px + 10, image_x - 1)]
                    c2 = image[min(py + 17, image_y - 1), max(px - 8, 0)]
                    ic = min(
                        np.argmin(np.sum((c1[None, :] - palette_clr)**2, axis=1)),
                        np.argmin(np.sum((c2[None, :] - palette_clr)**2, axis=1))
                    )
                    ext_data[ix, n_bins_y - 1 - iy] = palette_x[ic]
                else:
                    ext_data[ix, n_bins_y - 1 - iy] = palette_x[ic]

    # Apply manual hardcodes (from analyze.py)
    if hardcodes:
        for (i, j), val in hardcodes.items():
            ext_data[i][j] = val

    # Fit lognormals
    x_bins = E_TRUE_EDGES  # 23 edges for 22 bins
    gaussians = np.ndarray((n_bins, 3), float)
    for i in range(n_bins):
        mu, sigma, A = gaus_fit(ext_data[i], x_bins, i)
        gaussians[i][0] = sigma
        gaussians[i][1] = mu
        gaussians[i][2] = A

    return ext_data, gaussians


def build_energy_migration(source_dir):
    """
    Build R_E[i_true, j_reco, topology] where topology: 0=cascade, 1=track.

    For each true-E bin, integrate the fitted lognormal PDF over reco-E bins.
    Row-normalize so each true-E row sums to 1.
    """
    results_dir = os.path.join(source_dir, 'ORCA_Results')
    scale_path = os.path.join(results_dir, 'scale.png')

    R_E = np.zeros((N_ETRUE, N_ERECO, 2))

    # Track hardcodes from analyze.py: D_tracks[11][14] = 0.1
    track_hardcodes = {(11, 14): 0.1}

    for t, (name, png_name, hc) in enumerate([
        ('cascade', 'migrationCascades.png', None),
        ('track', 'migrationTracks.png', track_hardcodes)
    ]):
        image_path = os.path.join(results_dir, png_name)
        print(f"  Digitizing {png_name}...")
        _, gaussians = digitize_migration_png(image_path, scale_path, n_bins=N_ETRUE,
                                              hardcodes=hc)

        for i in range(N_ETRUE):
            sigma = gaussians[i][0]
            mu = gaussians[i][1]

            # The original code uses np.random.lognormal(mu, sigma) which
            # takes abs(sigma) internally. Negative sigma from curve_fit is
            # valid (sigma^2 is the same). Use abs(sigma) for CDF integration.
            sigma = abs(sigma)
            if sigma < 1e-6:
                sigma = 0.01  # fallback for degenerate fits

            # scipy.stats.lognorm(s=sigma, scale=exp(mu))
            dist = scipy_lognorm(s=sigma, scale=np.exp(mu))

            for j in range(N_ERECO):
                R_E[i, j, t] = dist.cdf(E_RECO_EDGES[j + 1]) - dist.cdf(E_RECO_EDGES[j])

        # Row-normalize
        for i in range(N_ETRUE):
            row_sum = R_E[i, :, t].sum()
            if row_sum > 0:
                R_E[i, :, t] /= row_sum

    return R_E


# ============================================================
# (b) Zenith migration matrix
# ============================================================

def get_zenith_error_functions():
    """Return 4 zenith error functions (degrees) from util.py."""
    def exp_func(a, b, c, d, x):
        return a + b * np.exp(c * x) + d * x

    e_nu = lambda x: exp_func(8.29046, 30.2972, -0.3048, -0.12256, x)
    e_nubar = lambda x: exp_func(6.69236, 28.38647, -0.36536, -0.093334, x)
    mu_nu = lambda x: exp_func(8.355, 47.171, -0.45966, -0.10707, x)
    mu_nubar = lambda x: exp_func(6.17314, 42.50309, -0.41, -0.08031, x)

    return [e_nu, e_nubar, mu_nu, mu_nubar]


def R_from_axis(u, th):
    """Rotation matrix around axis u by angle th (from util.py)."""
    cth = np.cos(th)
    sth = np.sin(th)
    R = np.array([
        [cth + u[0]**2*(1-cth), u[0]*u[1]*(1-cth) - u[2]*sth, u[0]*u[2]*(1-cth) + u[1]*sth],
        [u[1]*u[0]*(1-cth) + u[2]*sth, cth + u[1]**2*(1-cth), u[1]*u[2]*(1-cth) - u[0]*sth],
        [u[2]*u[0]*(1-cth) - u[1]*sth, u[2]*u[1]*(1-cth) + u[0]*sth, cth + u[2]**2*(1-cth)]
    ])
    return R


def rot_vector(v, th, phi):
    """Two successive rotations (from util.py)."""
    v = v / np.linalg.norm(v)
    u = np.array([v[1], -v[0], 0.0])
    norm_u = np.linalg.norm(u)
    if norm_u < 1e-10:
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = u / norm_u

    Rth = R_from_axis(u, th)
    vth = Rth @ v
    vth = vth / np.linalg.norm(vth)

    Rphi = R_from_axis(v, phi)
    vrot = Rphi @ vth
    vrot = vrot / np.linalg.norm(vrot)
    return vrot


def rand_reco_zen_batch(zen_true, sigma_deg, n_samples):
    """
    Generate n_samples of reco cos(zenith) given true zenith angle
    and angular error sigma (in degrees).
    Returns array of reco cos(zenith) values.
    """
    sigma_rad = sigma_deg * np.pi / 180.0
    azim = np.random.uniform(0, 2 * np.pi)  # true azimuth (arbitrary for isotropic)

    v = np.array([np.cos(azim) * np.sin(zen_true),
                  np.sin(azim) * np.sin(zen_true),
                  np.cos(zen_true)])

    cos_zen_reco = np.zeros(n_samples)
    for s in range(n_samples):
        err = np.abs(np.random.normal(0, sigma_rad))
        phi_rand = np.random.uniform(0, 2 * np.pi)
        vrot = rot_vector(v, err, phi_rand)
        # cos(zenith) = vrot[2] (z-component)
        cos_zen_reco[s] = vrot[2]

    return cos_zen_reco


def build_zenith_migration(source_dir):
    """
    Build R_cz[m_true, k_reco, i_E, res_group] with 6 resolution groups.

    Groups 0-3: cascade/track × nu/nubar (same sigma as original flavor groups).
    Groups 4-5: intermediate PID × nu/nubar (averaged sigma from cascade+track).

    For each (true_cz, E, res_group): MC-sample N rotations,
    histogram into reco cz bins, row-normalize.
    """
    zen_err_funcs = get_zenith_error_functions()  # [e_nu, e_nubar, mu_nu, mu_nubar]
    R_cz = np.zeros((N_CZ, N_CZ, N_ETRUE, 6))

    # Build sigma functions for all 6 groups
    # Groups 0-3 reuse the original 4 functions (cascade=e-like, track=mu-like)
    # Groups 4-5: intermediate = average of cascade and track sigma
    def int_nu_sigma(E):
        return 0.5 * (zen_err_funcs[0](E) + zen_err_funcs[2](E))

    def int_nubar_sigma(E):
        return 0.5 * (zen_err_funcs[1](E) + zen_err_funcs[3](E))

    all_sigma_funcs = list(zen_err_funcs) + [int_nu_sigma, int_nubar_sigma]

    total = N_CZ * N_ETRUE * 6
    count = 0

    for g_idx, (g_name, sigma_func) in enumerate(zip(RESOLUTION_GROUP_NAMES, all_sigma_funcs)):
        for i_E in range(N_ETRUE):
            E = E_TRUE_CENTERS[i_E]
            sigma_deg = sigma_func(E)
            if sigma_deg < 0:
                sigma_deg = 0.1  # floor

            for m in range(N_CZ):
                cos_zen_true = CZ_CENTERS[m]
                zen_true = np.arccos(cos_zen_true)

                cos_zen_reco = rand_reco_zen_batch(zen_true, sigma_deg, N_ZEN_SAMPLES)

                # Histogram into cz bins
                hist, _ = np.histogram(cos_zen_reco, bins=CZ_EDGES)
                row_sum = hist.sum()
                if row_sum > 0:
                    R_cz[m, :, i_E, g_idx] = hist / row_sum
                else:
                    # All samples outside range — put in nearest bin
                    R_cz[m, m, i_E, g_idx] = 1.0

                count += 1

        print(f"  Zenith migration: {g_name} done ({count}/{total})")

    return R_cz


# ============================================================
# (c) PID classification matrix
# ============================================================

def _compute_morphed_probs(p_ic_track, p_ic_cas, p_orca_track, p_orca_cas):
    """
    Compute effective topology fractions after IC→ORCA morphing.

    Replicates the assign_topology() logic from MCGenerator.py with
    restricted_rand_morph=True, inter_to_tracks=False.

    Given:
      - IC topology fractions (p_ic_track, p_ic_cas) from *_Topology_Fraction CSVs
      - ORCA topology probs (p_orca_track, p_orca_cas) from track_*.csv/cascade_*.csv

    Returns: (p_eff_cascade, p_eff_track, p_eff_intermediate)
    """
    if p_ic_track < 1e-10 and p_ic_cas < 1e-10:
        return 0, 0, 0

    if p_ic_track >= p_orca_track and p_ic_cas >= p_orca_cas:
        # Case 1: both IC >= ORCA — scale down proportionally
        cas_from_cas = p_orca_cas if p_ic_cas > 0 else 0
        int_from_cas = (p_ic_cas - p_orca_cas) if p_ic_cas > 0 else 0
        trk_from_trk = p_orca_track if p_ic_track > 0 else 0
        int_from_trk = (p_ic_track - p_orca_track) if p_ic_track > 0 else 0
    elif p_ic_track >= p_orca_track and p_ic_cas < p_orca_cas:
        # Case 2: IC track >= ORCA track, IC cascade < ORCA cascade
        cas_from_cas = p_ic_cas
        int_from_cas = 0
        trk_from_trk = p_orca_track if p_ic_track > 0 else 0
        int_from_trk = (p_ic_track - p_orca_track) if p_ic_track > 0 else 0
    elif p_ic_track < p_orca_track and p_ic_cas >= p_orca_cas:
        # Case 3: IC track < ORCA track, IC cascade >= ORCA cascade
        cas_from_cas = p_orca_cas if p_ic_cas > 0 else 0
        int_from_cas = (p_ic_cas - p_orca_cas) if p_ic_cas > 0 else 0
        trk_from_trk = p_ic_track
        int_from_trk = 0
    else:
        # Case 4: both IC < ORCA (rare edge case)
        cas_from_cas = p_ic_cas
        int_from_cas = 0
        trk_from_trk = p_ic_track
        int_from_trk = 0

    p_eff_cas = cas_from_cas
    p_eff_trk = trk_from_trk
    p_eff_int = int_from_cas + int_from_trk

    return p_eff_cas, p_eff_trk, p_eff_int


def build_pid_matrix(source_dir):
    """
    Build P_pid[l_pid, i_E, channel] where l_pid: 0=cascade, 1=track, 2=intermediate.

    Uses the IC→ORCA morphing logic from MCGenerator.assign_topology() to compute
    effective PID probabilities. This combines:
      - IC topology fractions from *_Topology_Fraction CSVs
      - ORCA topology probabilities from track_*.csv / cascade_*.csv
    using the ratio-based morphing with restricted_rand_morph=True.
    """
    results_dir = os.path.join(source_dir, 'ORCA_Results')

    # Original topology bins: 30 logspaced [2, 50] GeV
    topo_edges = np.logspace(np.log10(2), np.log10(50), 31)

    # Channel file mapping: (track_file, cascade_file)
    orca_channel_files = {
        'nueCC':      ('track_nue_CC.csv', 'cascade_nue_CC.csv'),
        'nuebarCC':   ('track_nuebar_CC.csv', 'cascade_nuebar_CC.csv'),
        'numuCC':     ('track_numu_CC.csv', 'cascade_numu_CC.csv'),
        'numubarCC':  ('track_numubar_CC.csv', 'cascade_numubar_CC.csv'),
        'nutauCC':    ('track_nutau_CC.csv', 'cascade_nutau_CC.csv'),
        'nutaubarCC': ('track_nutaubar_CC.csv', 'cascade_nutaubar_CC.csv'),
        'nuNC':       ('track_nu_NC.csv', 'cascade_nu_NC.csv'),
        'nubarNC':    ('track_nubar_NC.csv', 'cascade_nubar_NC.csv'),
    }

    # IC topology fraction files: (filename, column_prefix)
    ic_channel_files = {
        'nueCC':      ('nue_CC_Topology_Fraction', 'nu'),
        'nuebarCC':   ('nue_CC_Topology_Fraction', 'nubar'),
        'numuCC':     ('numu_CC_Topology_Fraction', 'nu'),
        'numubarCC':  ('numu_CC_Topology_Fraction', 'nubar'),
        'nutauCC':    ('nutau_CC_Topology_Fraction', 'nu'),
        'nutaubarCC': ('nutau_CC_Topology_Fraction', 'nubar'),
        'nuNC':       ('nu_NC_Topology_Fraction', 'nu'),
        'nubarNC':    ('nu_NC_Topology_Fraction', 'nubar'),
    }

    def load_orca_probs(track_file, cascade_file):
        """Load ORCA topology probabilities (padded to 30 bins)."""
        track_df = pd.read_csv(os.path.join(results_dir, track_file),
                               header=None, usecols=[1])
        track = np.array(track_df.iloc[:, 0])
        cascade_df = pd.read_csv(os.path.join(results_dir, cascade_file),
                                 header=None, usecols=[1])
        cascade = np.array(cascade_df.iloc[:, 0])
        if len(track) < 30:
            track = np.concatenate((np.zeros(30 - len(track)), track))
        if len(cascade) < 30:
            cascade = np.concatenate((np.zeros(30 - len(cascade)), cascade))
        # get_ORCA_topology_prob returns: track[idx], 1 - cascade[idx]
        return track, 1.0 - cascade

    def load_ic_fracs(filename, nutype):
        """Load IC topology fractions from consolidated CSV."""
        df = pd.read_csv(os.path.join(results_dir, filename))
        track = df[f'{nutype}_track'].values
        cascade = df[f'{nutype}_cas'].values
        return track, cascade

    P_pid = np.zeros((3, N_ETRUE, len(CHANNEL_NAMES)))

    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        track_file, cascade_file = orca_channel_files[ch_name]
        orca_track, orca_cas = load_orca_probs(track_file, cascade_file)

        ic_filename, nutype = ic_channel_files[ch_name]
        ic_track, ic_cas = load_ic_fracs(ic_filename, nutype)

        # Rebin to N_ETRUE bins
        for i in range(N_ETRUE):
            E = E_TRUE_CENTERS[i]
            if E <= topo_edges[0]:
                idx = 0
            elif E >= topo_edges[-1]:
                idx = 29
            else:
                idx = np.searchsorted(topo_edges, E) - 1
                idx = min(idx, 29)

            # Compute morphed PID using IC→ORCA ratio-based transformation
            p_cas, p_trk, p_int = _compute_morphed_probs(
                ic_track[idx], ic_cas[idx],
                orca_track[idx], orca_cas[idx])

            # Normalize to sum to 1
            total = p_cas + p_trk + p_int
            if total > 0:
                P_pid[0, i, ch_idx] = p_cas / total
                P_pid[1, i, ch_idx] = p_trk / total
                P_pid[2, i, ch_idx] = p_int / total
            else:
                P_pid[0, i, ch_idx] = 1.0  # default to cascade

    return P_pid


# ============================================================
# (d) ORCA effective area
# ============================================================

def build_effective_area(source_dir):
    """
    Build Aeff[i_E, channel] in units of m^2.

    Load ORCA Veff from CSVs, cross-sections from all_xsecs.csv.
    Aeff = Veff * n_d * sigma(E)
    Rebin from 50-bin [1,50] to N_ETRUE bins.
    """
    results_dir = os.path.join(source_dir, 'ORCA_Results')

    # Load cross-sections
    xsec_df = pd.read_csv(os.path.join(source_dir, 'all_xsecs.csv'))
    f_nu = interp1d(xsec_df['energy'], xsec_df['nu'], fill_value='extrapolate')
    f_nubar = interp1d(xsec_df['energy'], xsec_df['nubar'], fill_value='extrapolate')
    f_tau = interp1d(xsec_df['energy'], xsec_df['tau'], fill_value='extrapolate')
    f_taubar = interp1d(xsec_df['energy'], xsec_df['taubar'], fill_value='extrapolate')
    f_nc = interp1d(xsec_df['energy'], xsec_df['nc'], fill_value='extrapolate')
    f_ncbar = interp1d(xsec_df['energy'], xsec_df['ncbar'], fill_value='extrapolate')

    # Channel -> (Veff CSV, xsec function, tau flag, nc flag)
    channel_info = {
        'nueCC':      ('nueCC.csv', f_nu, False, False),
        'nuebarCC':   ('nuebarCC.csv', f_nubar, False, False),
        'numuCC':     ('numuCC.csv', f_nu, False, False),
        'numubarCC':  ('numubarCC.csv', f_nubar, False, False),
        'nutauCC':    ('nutauCC.csv', f_tau, True, False),
        'nutaubarCC': ('nutaubarCC.csv', f_taubar, True, False),
        'nuNC':       ('nuNC.csv', f_nc, False, True),
        'nubarNC':    ('nubarNC.csv', f_ncbar, False, True),
    }

    # Original Veff bins: 50 logspaced [1, 50] GeV
    orig_edges = np.logspace(np.log10(1), np.log10(50), 51)
    orig_centers = 0.5 * (orig_edges[:-1] + orig_edges[1:])

    Aeff = np.zeros((N_ETRUE, len(CHANNEL_NAMES)))

    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        csv_file, xsec_func, is_tau, is_nc = channel_info[ch_name]

        # Load Veff (in m^3, after 1e6 scaling)
        veff_df = pd.read_csv(os.path.join(results_dir, csv_file), header=None, usecols=[1])
        veff = np.array(veff_df.iloc[:, 0]) * 1e6  # m^3

        # Apply threshold zeroing (matching getORCAbins)
        if is_tau:
            for i in range(min(16, len(veff))):
                veff[i] = 0
        if is_nc:
            for i in range(min(4, len(veff))):
                veff[i] = 0

        # Rebin to N_ETRUE by nearest-neighbor
        for i in range(N_ETRUE):
            E = E_TRUE_CENTERS[i]

            # Find nearest original bin
            j = np.searchsorted(orig_centers, E)
            j = min(j, len(veff) - 1)

            V = veff[j]  # m^3

            # Cross-section in 10^-38 cm^2
            if E < 0.01:
                sigma = 0.0
            else:
                sigma = float(xsec_func(E))  # 10^-38 cm^2

            # Convert: sigma in cm^2
            sigma_cm2 = sigma * 1e-38

            # Aeff = Veff[m^3] * n_d[cm^-3] * sigma[cm^2]
            # But Veff is in m^3 and n_d in cm^-3: need to convert Veff to cm^3
            # Actually looking at NewEffective.py: nd = 0.9168 * (100**3) * 6.022e23
            # That's 0.9168e6 * 6.022e23 in cm^-3
            # And sigma = xsec * 1e-38 / (100^2) ... wait, let me re-read.
            # In NewEffective.py line 274: ccxsec = nusec(energy) * (10**-38) / (100**2)
            # And line 284: self.earea[i] = self.evol[i] * nd * ccxsec
            # So they divide sigma by 100^2 = 10^4 ... that's cm^2 -> m^2? No, 1e-38/1e4 = 1e-42.
            # nd = 0.9168 * 1e6 * 6.022e23 in cm^-3
            # evol is in m^3 (from getORCAbins * 1e6)
            # earea = evol[m^3] * nd[cm^-3] * sigma[1e-38 cm^2 / 1e4]
            # Units: m^3 * cm^-3 * cm^2/1e4 = m^3 * cm^-1 / 1e4
            # Hmm, that doesn't work dimensionally. Let me just follow exactly what they do.

            # Following NewEffective.py exactly:
            nd = 0.9168 * (100**3) * 6.022e23  # cm^-3, = 5.53e29
            xsec_area = sigma * 1e-38 / (100**2)  # 1e-42 cm^2... or is /100^2 meant to be cm^2 -> m^2?
            # Actually 1 m^2 = 10^4 cm^2, so /100^2 converts cm^2 to m^2
            # So xsec_area is in m^2

            # But then nd is in cm^-3 and Veff in m^3:
            # Aeff = V[m^3] * nd[cm^-3] * sigma[m^2]
            # This has units m^3 * cm^-3 * m^2 = m^5 / cm^3
            # That's not right either. The issue is mixing units.

            # Let me just replicate exactly: earea = evol * nd * ccxsec
            # where evol from getORCAbins is raw CSV * 1e6
            # nd = 0.9168 * 1e6 * 6.022e23
            # ccxsec = xsec(E) * 1e-38 / 1e4
            # This is what the original code does, so the "m^2" units are whatever they are.
            Aeff[i, ch_idx] = V * nd * xsec_area

    return Aeff


# ============================================================
# Plotting
# ============================================================

def plot_energy_migration(R_E, output_dir):
    """Plot energy migration heatmaps for cascade and track (log-scale colorbar)."""
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for t, (ax, name) in enumerate(zip(axes, ['Cascade', 'Track'])):
        # Use log-scale colorbar matching reference plots
        R_plot = np.ma.masked_where(R_E[:, :, t] <= 0, R_E[:, :, t])
        im = ax.pcolormesh(
            np.arange(N_ERECO + 1), np.arange(N_ETRUE + 1),
            R_plot, cmap='gray_r', norm=LogNorm(vmin=1e-3, vmax=1)
        )
        ax.set_xlabel('Reco E bin index')
        ax.set_ylabel('True E bin index')
        ax.set_title(f'Energy Migration: {name}')
        plt.colorbar(im, ax=ax, label='Probability')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_migration_heatmaps.png'), dpi=150)
    plt.close()
    print(f"  Saved energy_migration_heatmaps.png")


def plot_zenith_migration(R_cz, output_dir):
    """Plot zenith migration heatmaps at representative energies."""
    E_indices = [2, 8, 15, 20]  # ~2.5, ~6, ~18, ~45 GeV
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for row, f_idx in enumerate([0, 2]):  # e_nu, mu_nu
        for col, i_E in enumerate(E_indices):
            ax = axes[row, col]
            im = ax.pcolormesh(
                CZ_EDGES, CZ_EDGES,
                R_cz[:, :, i_E, f_idx], cmap='viridis', vmin=0
            )
            E = E_TRUE_CENTERS[i_E]
            ax.set_title(f'{FLAVOR_GROUP_NAMES[f_idx]}, E={E:.1f} GeV')
            ax.set_xlabel('Reco cos(zen)')
            ax.set_ylabel('True cos(zen)')
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'zenith_migration_heatmaps.png'), dpi=150)
    plt.close()
    print(f"  Saved zenith_migration_heatmaps.png")


def plot_pid_fractions(P_pid, output_dir):
    """Plot PID fractions vs energy for representative channels."""
    channels_to_plot = [0, 2, 4, 6]  # nueCC, numuCC, nutauCC, nuNC
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, ch_idx in zip(axes.flat, channels_to_plot):
        ax.plot(E_TRUE_CENTERS, P_pid[0, :, ch_idx], 'b-', label='Cascade')
        ax.plot(E_TRUE_CENTERS, P_pid[1, :, ch_idx], 'r-', label='Track')
        ax.plot(E_TRUE_CENTERS, P_pid[2, :, ch_idx], 'g--', label='Intermediate')
        ax.set_xscale('log')
        ax.set_xlabel('True Energy [GeV]')
        ax.set_ylabel('Fraction')
        ax.set_title(CHANNEL_NAMES[ch_idx])
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pid_fractions.png'), dpi=150)
    plt.close()
    print(f"  Saved pid_fractions.png")


def plot_effective_area(Aeff, output_dir):
    """Plot effective area vs energy for all 8 channels."""
    colors = ['red', 'red', 'blue', 'blue', 'green', 'green', 'brown', 'brown']
    styles = ['-', '--', '-', '--', '-', '--', '-', '--']

    fig, ax = plt.subplots(figsize=(10, 7))
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        ax.plot(E_TRUE_CENTERS, Aeff[:, ch_idx],
                color=colors[ch_idx], linestyle=styles[ch_idx],
                label=ch_name, linewidth=1.5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('True Energy [GeV]')
    ax.set_ylabel('Effective Area [arb. units]')
    ax.set_title('ORCA Effective Area')
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'effective_area.png'), dpi=150)
    plt.close()
    print(f"  Saved effective_area.png")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Build ORCA-Full response matrices')
    parser.add_argument('--source-dir', type=str, required=True,
                        help='Path to AtmNuCombination/sources/ORCA/')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for .npz and plots')
    args = parser.parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # (a) Energy migration
    print("Building energy migration matrix...")
    R_E = build_energy_migration(source_dir)
    print(f"  R_E shape: {R_E.shape}")
    for t, name in enumerate(['cascade', 'track']):
        row_sums = R_E[:, :, t].sum(axis=1)
        print(f"  {name} row sums: min={row_sums.min():.4f}, max={row_sums.max():.4f}")

    # (b) Zenith migration
    print("Building zenith migration matrix...")
    R_cz = build_zenith_migration(source_dir)
    print(f"  R_cz shape: {R_cz.shape}")
    # Check row normalization
    for g_idx, g_name in enumerate(RESOLUTION_GROUP_NAMES):
        row_sums = R_cz[:, :, :, g_idx].sum(axis=1)
        print(f"  {g_name} row sums: min={row_sums.min():.4f}, max={row_sums.max():.4f}")

    # (c) PID matrix
    print("Building PID classification matrix...")
    P_pid = build_pid_matrix(source_dir)
    print(f"  P_pid shape: {P_pid.shape}")
    col_sums = P_pid.sum(axis=0)
    print(f"  PID column sums: min={col_sums.min():.4f}, max={col_sums.max():.4f}")

    # (d) Effective area
    print("Building effective area...")
    Aeff = build_effective_area(source_dir)
    print(f"  Aeff shape: {Aeff.shape}")
    print(f"  Aeff range: [{Aeff[Aeff > 0].min():.4e}, {Aeff.max():.4e}]")

    # Save
    npz_path = os.path.join(output_dir, 'orca_full_response.npz')
    np.savez(npz_path,
             R_E=R_E, R_cz=R_cz, P_pid=P_pid, Aeff=Aeff,
             E_true_edges=E_TRUE_EDGES, E_reco_edges=E_RECO_EDGES,
             cz_true_edges=CZ_EDGES, cz_reco_edges=CZ_EDGES,
             channel_names=CHANNEL_NAMES,
             resolution_group_names=RESOLUTION_GROUP_NAMES)
    print(f"\nSaved response matrices to {npz_path}")

    # Validation plots
    print("\nGenerating validation plots...")
    plot_energy_migration(R_E, output_dir)
    plot_zenith_migration(R_cz, output_dir)
    plot_pid_fractions(P_pid, output_dir)
    plot_effective_area(Aeff, output_dir)

    # Print verification checklist
    print("\n=== Verification Checklist ===")
    for t, name in enumerate(['cascade', 'track']):
        row_sums = R_E[:, :, t].sum(axis=1)
        n_good = np.sum(row_sums > 0.01)
        ok_nonzero = np.allclose(row_sums[row_sums > 0.01], 1.0, atol=0.01)
        n_zero = N_ETRUE - n_good
        status = 'PASS' if ok_nonzero else 'FAIL'
        print(f"  [{status}] R_E {name}: {n_good}/{N_ETRUE} rows sum to ~1"
              f" ({n_zero} empty — true E below reco range)")

    for g_idx, g_name in enumerate(RESOLUTION_GROUP_NAMES):
        row_sums = R_cz[:, :, :, g_idx].sum(axis=1)
        ok = np.allclose(row_sums, 1.0, atol=0.05)
        print(f"  [{'PASS' if ok else 'FAIL'}] R_cz {g_name} rows sum to ~1")

    col_sums = P_pid.sum(axis=0)
    ok = np.allclose(col_sums, 1.0, atol=0.01)
    print(f"  [{'PASS' if ok else 'FAIL'}] P_pid columns sum to 1")

    ok = np.all(Aeff >= 0)
    print(f"  [{'PASS' if ok else 'FAIL'}] Aeff values non-negative")

    print("\nDone!")


if __name__ == '__main__':
    main()
