# Class for the atmospheric neutrinos in ORCA (KM3NeT)

import os

import numpy as np
import pandas as pd
import nuflux
from .Experiment import Experiment


class ORCA_Atm(Experiment):
    """
    ORCA atmospheric neutrino experiment class for Pynu.

    Handles:
    - Parquet file input format
    - 3 event morphologies (shower, HPT, track)
    - 2D binning in reconstructed energy and cos(zenith)
    """

    def __init__(self, dict_of_details):
        # Initialize parent class
        super(ORCA_Atm, self).__init__(dict_of_details)

        self.MIN_ENTRIES = 0.01

        # ORCA-specific parameters
        self.Samples = [0, 1, 2]  # 0: shower, 1: HPT, 2: track
        self.NumberOfSamples = 3

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
        Extract and process MC variables for ORCA.

        Expected columns in parquet:
        - true_energy, reco_energy
        - true_zenith, reco_zenith
        - pdg (neutrino PDG code)
        - pid (morphology: 0=shower, 1=HPT, 2=track)
        - weight (event weight)
        """
        # Apply quality cuts
        d_Etrue = self.MC['true_energy']
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

        # events = weight * flux * livetime * 1e4, with livetime = FitExposure in years
        self.NORM = self.FitExposure * 1e4

        # Base weights for histogram filling
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
        """2D binning in (E_reco, cos_theta_reco), the same for each morphology, from
        the edge files next to the MC."""
        bin_dir = os.path.dirname(self.MCFiles[0])
        erec = np.load(os.path.join(bin_dir, '_E_reco_bins.npy'))
        cz_bins = np.load(os.path.join(bin_dir, '_cosT_reco_bins.npy'))

        # Same binning for all morphologies
        self.EnergyBins = {s: erec for s in self.Samples}
        self.CTBins = {s: cz_bins for s in self.Samples}

    def BinMC(self, array):
        """
        Bin MC events with given weights.

        Args:
            array: Weight array to apply (e.g., oscillation weights)

        Returns:
            Flattened array of binned events [morph0_bins..., morph1_bins..., morph2_bins...]
        """
        self.CosThetaReco = self.CosZReco
        return self.BinIt_MC_2D(array)
