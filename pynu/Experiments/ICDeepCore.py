# Class for IceCube DeepCore atmospheric neutrinos
# Based on Phys. Rev. D 108, 012014 (2023)

import os

import numpy as np
import pandas as pd
import nuflux
from .Experiment import Experiment


class ICDeepCore_Atm(Experiment):
    """
    IceCube DeepCore atmospheric neutrino experiment class for Pynu.

    Handles:
    - Parquet file input format (converted from IceCube CSV data release)
    - 2 event morphologies (0: mixed/cascade-like, 1: tracks)
    - 2D binning in reconstructed energy and cos(zenith)

    Weight convention:
    - MC weights are in (GeV cm² sr) = effective area × generation normalization
    - Oscillated flux from nuSQuIDS is in (GeV⁻¹ cm⁻² sr⁻¹ s⁻¹)
    - NORM = livetime_seconds = FitExposure_years × seconds_per_year
    - Expected events = Weight × NORM × oscillated flux
    """

    SECONDS_PER_YEAR = 3.15576e7  # Julian year

    def __init__(self, dict_of_details):
        # Initialize parent class (calls Reader())
        super(ICDeepCore_Atm, self).__init__(dict_of_details)

        self.MIN_ENTRIES = 0.01

        # IceCube DeepCore: 2 PID samples
        # 0: mixed/cascade-like (PID bin [0.55, 0.75])
        # 1: tracks (PID bin [0.75, 1.0])
        self.Samples = [0, 1]
        self.NumberOfSamples = 2

        # Read and process MC
        self.MCVariables()

        # Set up binning
        self.Binning()
        self.SetBinner_2D()

    def Reader(self):
        """Read the neutrino rows (MC_type != -1) of the parquet MC."""
        df = pd.read_parquet(self.MCFiles[0])
        df_nu = df[df['MC_type'] != -1]
        self.MC = {col: df_nu[col].values for col in df_nu.columns}

    def MCVariables(self):
        """
        Extract and process MC variables for IceCube DeepCore.

        Expected columns in parquet:
        - true_energy, reco_energy (GeV)
        - true_zenith, reco_zenith (radians, converted to cos internally)
        - pdg (neutrino PDG code)
        - pid (0=mixed, 1=tracks)
        - weight (GeV cm² sr)
        """
        # Apply quality cuts
        d_Etrue = self.MC.get('true_energy')
        condition = (d_Etrue >= 0) & (d_Etrue < 1e5)

        # Reconstructed quantities
        self.EReco = self.MC['reco_energy'][condition]
        self.CosZReco = np.cos(self.MC['reco_zenith'][condition])

        # True quantities (for flux/oscillation calculations)
        self.ETrue = self.MC['true_energy'][condition]
        self.CosZTrue = np.cos(self.MC['true_zenith'][condition])

        # Particle identification
        self.nuPDG = self.MC['pdg'][condition].astype(int)

        # Event classification (morphology/sample)
        self.Sample = self.MC['pid'][condition].astype(int)

        # Weights
        self.Weight = self.MC['weight'][condition]

        # Event counts
        self.NumberOfEvents = len(self.EReco)

        # Energy and zenith ranges for the nuSQuIDS nodes
        self.Etrue_min = 0.1
        self.Etrue_max = 1e4
        self.Z_edges = [-1, 1]  # Full zenith range for oscillation calculation

        # IceCube weight convention:
        # weight in (GeV cm² sr), flux in (GeV⁻¹ cm⁻² sr⁻¹ s⁻¹)
        # events = weight × flux × livetime_seconds
        self.NORM = self.FitExposure * self.SECONDS_PER_YEAR
        self.BaseWeight = self.Weight * self.NORM

        # Clean up raw MC dict to save memory
        del self.MC

    def SetInitialFlux(self, energy_nodes, cth_nodes, neutrino_flavors):
        """
        Set up the atmospheric neutrino initial flux using nuflux.

        Args:
            energy_nodes: Array of energy values [GeV]
            cth_nodes: Array of cos(zenith) values
            neutrino_flavors: Number of neutrino flavors

        Returns:
            AtmInitialFlux: 4D array [cth, energy, nu/nubar, flavor]
        """
        flux = nuflux.makeFlux('IPhonda2014_spl_solmin')

        AtmInitialFlux = np.zeros(
            (len(cth_nodes), len(energy_nodes), 2, neutrino_flavors)
        )

        for ic, nu_cos_zenith in enumerate(cth_nodes):
            for ie, nu_energy in enumerate(energy_nodes):
                # Neutrinos (index 0)
                AtmInitialFlux[ic][ie][0][0] = flux.getFlux(
                    nuflux.NuE, nu_energy, nu_cos_zenith)
                AtmInitialFlux[ic][ie][0][1] = flux.getFlux(
                    nuflux.NuMu, nu_energy, nu_cos_zenith)
                AtmInitialFlux[ic][ie][0][2] = 0.0  # nutau (no atmospheric)

                # Anti-neutrinos (index 1)
                AtmInitialFlux[ic][ie][1][0] = flux.getFlux(
                    nuflux.NuEBar, nu_energy, nu_cos_zenith)
                AtmInitialFlux[ic][ie][1][1] = flux.getFlux(
                    nuflux.NuMuBar, nu_energy, nu_cos_zenith)
                AtmInitialFlux[ic][ie][1][2] = 0.0  # nutaubar

        return AtmInitialFlux

    def Binning(self):
        """2D binning in (E_reco, cos_theta_reco), the same for both samples, from the
        edge files next to the MC."""
        bin_dir = os.path.dirname(self.MCFiles[0])
        erec = np.load(os.path.join(bin_dir, '_E_reco_bins.npy'))
        cz_bins = np.load(os.path.join(bin_dir, '_cosT_reco_bins.npy'))

        # Same binning for both samples
        self.EnergyBins = {s: erec for s in self.Samples}
        self.CTBins = {s: cz_bins for s in self.Samples}

    def BinMC(self, array):
        """
        Bin MC events with given weights.

        Args:
            array: Weight array to apply (e.g., oscillation weights)

        Returns:
            Flattened array of binned events [sample0_bins..., sample1_bins...]
        """
        self.CosThetaReco = self.CosZReco
        return self.BinIt_MC_2D(array)

    # =====================================================================
    # Hypersurface (HS) detector systematics
    # =====================================================================

    HS_NOMINALS = {
        'dom_eff': 1.0,
        'hole_ice_p0': 0.1,
        'hole_ice_p1': -0.05,
        'bulk_ice_abs': 1.0,
        'bulk_ice_scatter': 1.0,
    }

    HS_SLOPE_NAMES = ['dom_eff', 'hole_ice_p0', 'hole_ice_p1',
                      'bulk_ice_abs', 'bulk_ice_scatter']

    HS_CATEGORIES = {
        'nc_nue_cc': 'hs_nu_nc_nue_cc.csv',
        'numu_cc':   'hs_numu_cc.csv',
        'nutau_cc':  'hs_nutau_cc.csv',
    }

    # PID value → sample index mapping
    HS_PID_MAP = {0.65: 0, 0.88: 1}

    def load_hypersurfaces(self, hs_dir):
        """
        Load the 3 HS CSV files and build indexed arrays.

        For each category and each deltam31 slice, stores slope arrays
        of shape (N_bins=200,) where bins are ordered as
        [sample0_flat, sample1_flat] with energy-slow, coszen-fast.

        Args:
            hs_dir: Path to directory containing hs_*.csv files
        """
        self._hs_data = {}
        self._hs_dm31_grid = None

        n_e = len(self.EnergyBins[0]) - 1
        n_cz = len(self.CTBins[0]) - 1
        n_bins_per_sample = n_e * n_cz  # 10 * 10 = 100
        n_bins_total = n_bins_per_sample * self.NumberOfSamples  # 200

        for cat_name, csv_file in self.HS_CATEGORIES.items():
            filepath = os.path.join(hs_dir, csv_file)
            df = pd.read_csv(filepath)

            dm31_values = np.array(sorted(df['deltam31'].unique()))
            if self._hs_dm31_grid is None:
                self._hs_dm31_grid = dm31_values
            n_dm = len(dm31_values)

            # Pre-allocate: (n_dm, n_bins_total) for intercept and each slope
            cat_data = {
                'intercept': np.zeros((n_dm, n_bins_total)),
            }
            for sname in self.HS_SLOPE_NAMES:
                cat_data[sname] = np.zeros((n_dm, n_bins_total))

            for idm, dm_val in enumerate(dm31_values):
                df_dm = df[df['deltam31'] == dm_val]

                for pid_val, sample_idx in self.HS_PID_MAP.items():
                    df_pid = df_dm[df_dm['pid'] == pid_val]
                    # Sort by (energy, coszen) → energy slow, coszen fast
                    df_pid = df_pid.sort_values(['reco_energy', 'reco_coszen'])

                    offset = sample_idx * n_bins_per_sample
                    cat_data['intercept'][idm, offset:offset + n_bins_per_sample] = df_pid['intercept'].values
                    for sname in self.HS_SLOPE_NAMES:
                        cat_data[sname][idm, offset:offset + n_bins_per_sample] = df_pid[sname].values

            self._hs_data[cat_name] = cat_data
