# Class for IceCube Upgrade atmospheric neutrinos
# Based on IceCubeUpgradeNeutrinoMCDataRelease-2 public MC
# Modeled on ICDeepCore.py with simplifications for the data release

import numpy as np
import nuflux
from .Experiment import Experiment


class ICUpgrade_Atm(Experiment):
    """
    IceCube Upgrade atmospheric neutrino experiment class for Pynu.

    Handles:
    - CSV file input (neutrino_mc.csv from data release)
    - 2 PID samples (0: cascade, 1: track)
    - No muon background (negligible per data release README)
    - No detector systematics (no ice gradient files in release)
    - 2D binning in reconstructed energy and cos(zenith)

    Weight convention:
    - MC weights are in m^2 (effective area x generation normalization)
    - Oscillated flux from nuSQuIDS is in (GeV^-1 cm^-2 sr^-1 s^-1)
    - NORM = (exposure/mc_exposure) * seconds_per_year * 1e4 (m^2 -> cm^2)
    - Expected events = Weight * NORM * PhysicsWeight
    """

    SECONDS_PER_YEAR = 3.15576e7  # Julian year

    def __init__(self, dict_of_details, scenario):
        # Initialize parent class (calls Reader() for CSV loading)
        super(ICUpgrade_Atm, self).__init__(dict_of_details)

        self.MIN_ENTRIES = 0.01

        self.Detector = "ICUpgrade"
        self.Target = "Water"
        self.SOURCE = "Atmospheric"
        self.SCENARIO = scenario

        self.SetDefinition()

        # 2 PID samples: 0=cascade, 1=track
        self.NumberOfMorphologies = 2
        self.Samples = [0, 1]
        self.NumberOfSamples = 2

        # Read and process MC
        self.MCVariables()

        # Set up binning
        self.Binning()
        self.SetBinner_2D()

        # Handle data if in data-fitting mode
        if self.DataFit:
            self.DataVariables()

        # No muon background in ICUp data release
        self.MuonVariables()

    def MCVariables(self):
        """
        Extract and process MC variables for IceCube Upgrade.

        Expected columns in CSV (from data release):
        - true_energy, reco_energy (GeV)
        - true_zenith, reco_zenith (radians)
        - true_azimuth (radians)
        - pdg (neutrino PDG code)
        - current_type (0=NC, 1=CC)
        - pid (0=cascade, 1=track)
        - weight (m^2)
        """
        d_Etrue = self.MC.get('true_energy')
        condition = (d_Etrue >= 0) & (d_Etrue < 1e5)

        # Reconstructed quantities
        self.EReco = self.MC['reco_energy'][condition]
        self.CosZReco = np.cos(self.MC['reco_zenith'][condition])

        # True quantities (for flux/oscillation calculations)
        self.ETrue = self.MC['true_energy'][condition]
        self.CosZTrue = np.cos(self.MC['true_zenith'][condition])

        # Azimuth (available in ICUp data release)
        self.AziTrue = self.MC['true_azimuth'][condition]

        # Particle identification
        self.nuPDG = self.MC['pdg'][condition].astype(int)
        self.CC = self.MC['current_type'][condition]  # 0=NC, 1=CC

        # Event classification (PID: 0=cascade, 1=track)
        self.Sample = self.MC['pid'][condition].astype(int)

        # Weights (m^2 units)
        self.Weight = self.MC['weight'][condition]

        # Interaction mode (NEUT convention) for xsec systematics
        if 'interaction_type' in self.MC:
            self.Mode = self._NEUTMode(self.MC['interaction_type'], self.MC['pdg'])[condition]
        else:
            self.Mode = np.zeros(condition.sum(), dtype=int)

        # MC statistical variance (Poisson fallback; no variance column in release)
        self.WeightVariance = self.Weight**2

        # Event counts
        self.NumberOfEvents = len(self.EReco)

        # Energy and zenith ranges for oscillation grid
        self.Erec_min = 1.0
        self.Erec_max = 100.0
        self.Etrue_min = 0.1
        self.Etrue_max = 1e4
        self.E_edges = [self.Erec_min, self.Erec_max]
        self.Z_edges = [-1, 1]

        # ICUp weight convention:
        # weight in m^2, flux in (GeV^-1 cm^-2 sr^-1 s^-1)
        # events = weight * 1e4 (m2->cm2) * flux * livetime_seconds
        # NORM = (exposure/mc_exposure) * seconds_per_year * 1e4
        self.NORM *= self.SECONDS_PER_YEAR * 1e4
        self.BaseWeight = self.Weight * self.NORM

        # Clean up raw MC dict to save memory
        del self.MC

    @staticmethod
    def _NEUTMode(interaction_type, pdg):
        """Convert interaction_type (0-4) to NEUT mode codes, signed by nu/nubar.

        Mapping: 0→31(DIS), 1→1(CCQE), 2→11(1π), 3→26(DIS/multi-π), 4→16(coh π).
        Sign: positive for neutrinos (pdg>0), negative for antineutrinos (pdg<0).
        """
        itype = np.asarray(interaction_type, dtype=int)
        sign = np.where(np.asarray(pdg) > 0, 1, -1)
        neut_map = {0: 31, 1: 1, 2: 11, 3: 26, 4: 16}
        mode = np.array([neut_map.get(v, 0) for v in itype])
        return sign * mode

    def MuonVariables(self):
        """No muon background in ICUp data release (negligible per README)."""
        self.HasMuonBkg = False
        self.MuonEReco = None
        self.MuonCosZReco = None
        self.MuonSample = None
        self.MuonWeight = None
        self.MuonWeightVariance = None

    def DataVariables(self):
        """Process data variables for data-fitting mode."""
        self.dEReco = self.Data['reco_energy']
        self.dCosZReco = np.cos(self.Data['reco_zenith'])
        self.dSample = self.Data['pid'].astype(int)
        self.dSamples = self.Samples

        if 'weight' in self.Data:
            self.dWeight = self.Data['weight']
        else:
            self.dWeight = np.ones_like(self.dEReco)

        self.dNumberOfEvents = len(self.dEReco)

        del self.Data

    def SetInitialFlux(self, energy_nodes, cth_nodes, neutrino_flavors):
        """
        Set up the atmospheric neutrino initial flux using nuflux.

        Args:
            energy_nodes: Array of energy values [GeV]
            cth_nodes: Array of cos(zenith) values
            neutrino_flavors: Number of neutrino flavors (3 or 4)

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
                AtmInitialFlux[ic][ie][0][2] = 0.0  # nutau

                # Anti-neutrinos (index 1)
                AtmInitialFlux[ic][ie][1][0] = flux.getFlux(
                    nuflux.NuEBar, nu_energy, nu_cos_zenith)
                AtmInitialFlux[ic][ie][1][1] = flux.getFlux(
                    nuflux.NuMuBar, nu_energy, nu_cos_zenith)
                AtmInitialFlux[ic][ie][1][2] = 0.0  # nutaubar

                # Sterile (if applicable)
                if neutrino_flavors > 3:
                    AtmInitialFlux[ic][ie][0][3] = 0.0
                    AtmInitialFlux[ic][ie][1][3] = 0.0

        return AtmInitialFlux

    def Binning(self, bin_dir=None):
        """
        Set up IceCube Upgrade binning scheme.

        Default: 20 log-spaced energy bins [1, 100] GeV, 10 linear coszen [-1, 1].
        Tries to load from numpy files in the data directory first.
        """
        import os

        if bin_dir is None and len(self.MCFiles) > 0:
            bin_dir = os.path.dirname(self.MCFiles[0])

        erec = None
        cz_bins = None

        # Try to load from numpy files
        if bin_dir:
            erec_file = os.path.join(bin_dir, '_E_reco_bins.npy')
            cz_file = os.path.join(bin_dir, '_cosT_reco_bins.npy')

            if os.path.exists(erec_file):
                try:
                    erec = np.load(erec_file)
                    print(f"[ICUpgrade] Loaded energy bins from {erec_file}")
                except Exception as e:
                    print(f"[ICUpgrade] Warning: Could not load {erec_file}: {e}")

            if os.path.exists(cz_file):
                try:
                    cz_bins = np.load(cz_file)
                    print(f"[ICUpgrade] Loaded zenith bins from {cz_file}")
                except Exception as e:
                    print(f"[ICUpgrade] Warning: Could not load {cz_file}: {e}")

        # Fall back to default ICUp binning
        if erec is None:
            NErec = 20
            erec = np.logspace(np.log10(1.0), np.log10(100.0), NErec + 1, endpoint=True)
            print("[ICUpgrade] Using default 20 log-spaced energy bins [1, 100] GeV")

        self.NErec = len(erec) - 1

        if cz_bins is None:
            cz_bins = np.linspace(-1.0, 1.0, 11)
            print("[ICUpgrade] Using default 10 linear coszen bins [-1, 1]")

        # Same binning for both samples
        self.EnergyBins = {s: erec for s in self.Samples}
        self.CTBins = {s: cz_bins for s in self.Samples}

    def BinMC(self, array):
        """Bin MC events with given weights."""
        self.CosThetaReco = self.CosZReco
        return self.BinIt_MC_2D(array)

    def BinData(self, entries=None):
        """Bin data events using weights."""
        self.dCosThetaReco = self.dCosZReco
        if entries is None and hasattr(self, 'dWeight'):
            entries = self.dWeight
        return self.BinIt_Data_2D(entries)

    def GetMuonBackground(self):
        """No muon background for ICUp."""
        return None, None

    def GetMCVariance(self, array):
        """
        Compute MC statistical variance for given weights.

        For weighted histograms, variance = sum(w^2 * variance_per_event)
        """
        safe_weight_sq = np.where(self.Weight != 0, self.Weight**2, 1.0)
        relative_var = np.where(self.Weight != 0, self.WeightVariance / safe_weight_sq, 0.0)
        var_weights = (array * self.Weight * self.NORM)**2 * relative_var

        variance_binned = []
        for m in self.Samples:
            mask = self.Sample == m
            E_sample = self.EReco[mask]
            cz_sample = self.CosZReco[mask]
            w_sample = var_weights[mask]

            hist, _, _ = np.histogram2d(
                E_sample, cz_sample,
                bins=[self.EnergyBins[m], self.CTBins[m]],
                weights=w_sample
            )
            variance_binned.append(hist.flatten())

        return np.concatenate(variance_binned)


# Alias for MCReader compatibility
ICUpgrade = ICUpgrade_Atm
