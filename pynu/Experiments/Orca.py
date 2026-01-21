# Class for the atmospheric neutrinos in ORCA (KM3NeT)
# Adapted from the NeutrinoTelescopeOscillationAnalysis workflow

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
    - Atmospheric muon background
    - MC statistical uncertainties (weight_variance)
    - 2D binning in reconstructed energy and cos(zenith)
    """
    
    def __init__(self, dict_of_details, scenario):
        # Initialize parent class
        super(ORCA_Atm, self).__init__(dict_of_details)
        
        # Match NTOA bin selection threshold for ORCA (was MIN_ENTRIES=5 in parent class)
        self.MIN_ENTRIES = 0.01
        
        # Experiment identifiers for PhysicsTunes lookup
        self.Detector = "ORCA"
        self.Target = "Water"
        self.SOURCE = "Atmospheric"
        self.SCENARIO = scenario
        
        self.SetDefinition()
        
        # ORCA-specific parameters
        self.NumberOfMorphologies = 3  # 0: shower, 1: HPT, 2: track
        self.Samples = [0, 1, 2]
        self.NumberOfSamples = 3
        
        # Read and process MC
        self.MCVariables()
        
        # Set up binning
        self.Binning()
        self.SetBinner_2D()
        
        # Handle data if in data-fitting mode
        if self.DataFit:
            self.DataVariables()
        
        # Handle muon background if present
        self.MuonVariables()
    
    def Reader(self):
        """
        Override parent Reader to handle parquet files for ORCA.
        """
        self.MC = {}
        for i, f in enumerate(self.MCFiles):
            if f.endswith('.parquet'):
                df = pd.read_parquet(f)
                # Filter to neutrino events only (MC_type != -1)
                df_nu = df[df['MC_type'] != -1]
                newdata = {col: df_nu[col].values for col in df_nu.columns}
            else:
                # Fall back to parent reader for other formats
                from . import MCReader
                newdata = MCReader.reader(f)
            
            if i == 0:
                self.MC = newdata
            else:
                for key, value in newdata.items():
                    if key in self.MC:
                        self.MC[key] = np.append(self.MC[key], value)
        
        # Store muon MC separately
        self.MuonMC = {}
        for i, f in enumerate(self.MCFiles):
            if f.endswith('.parquet'):
                df = pd.read_parquet(f)
                df_mu = df[df['MC_type'] == -1]
                if len(df_mu) > 0:
                    newdata = {col: df_mu[col].values for col in df_mu.columns}
                    if i == 0:
                        self.MuonMC = newdata
                    else:
                        for key, value in newdata.items():
                            if key in self.MuonMC:
                                self.MuonMC[key] = np.append(self.MuonMC[key], value)
        
        # Handle data files
        if self.DataFit:
            self.Data = {}
            for i, f in enumerate(self.DataFiles):
                if f.endswith('.parquet'):
                    df = pd.read_parquet(f)
                    newdata = {col: df[col].values for col in df.columns}
                else:
                    from . import MCReader
                    newdata = MCReader.reader(f)
                
                if i == 0:
                    self.Data = newdata
                else:
                    for key, value in newdata.items():
                        if key in self.Data:
                            self.Data[key] = np.append(self.Data[key], value)
    
    def MCVariables(self):
        """
        Extract and process MC variables for ORCA.
        
        Expected columns in parquet:
        - true_energy, reco_energy
        - true_zenith, reco_zenith
        - pdg (neutrino PDG code)
        - current_type (1=CC, 0=NC)
        - pid (morphology: 0=shower, 1=HPT, 2=track)
        - weight (event weight)
        - weight_variance (MC statistical variance)
        """
        # Apply quality cuts
        d_Etrue = self.MC.get('true_energy', self.MC.get('Etrue', None))
        condition = (d_Etrue >= 0) & (d_Etrue < 1e5)
        
        # Reconstructed quantities
        self.EReco = self.MC['reco_energy'][condition]
        self.CosZReco = np.cos(self.MC['reco_zenith'][condition])
        
        # True quantities (for flux/oscillation calculations)
        self.ETrue = self.MC['true_energy'][condition]
        self.CosZTrue = np.cos(self.MC['true_zenith'][condition])
        
        # Particle identification
        self.nuPDG = self.MC['pdg'][condition].astype(int)
        self.CC = self.MC['current_type'][condition]  # 1=CC, 0=NC
        
        # Event classification (morphology/sample)
        self.Sample = self.MC['pid'][condition].astype(int)
        
        # Weights
        self.Weight = self.MC['weight'][condition]
        
        # MC statistical variance (for Barlow-Beeston)
        if 'weight_variance' in self.MC:
            self.WeightVariance = self.MC['weight_variance'][condition]
        else:
            self.WeightVariance = self.Weight**2  # Fallback: Poisson
        
        # Azimuth (if available)
        if 'true_azimuth' in self.MC:
            self.AziTrue = self.MC['true_azimuth'][condition]
        else:
            self.AziTrue = np.zeros_like(self.ETrue)
        
        # Event counts
        self.NumberOfEvents = len(self.EReco)
        
        # Energy and zenith ranges for flux
        self.Erec_min = 1.0
        self.Erec_max = 100.0
        self.Etrue_min = 0.1
        self.Etrue_max = 1e4
        self.E_edges = [self.Erec_min, self.Erec_max]
        self.Z_edges = [-1, 1]  # Full zenith range for oscillation calculation  # ORCA sees upgoing events only
        
        # Normalization: match original workflow convention
        # Original uses: weighted_rate = unweighted_rate * mc_weights * livetime * unit_norm
        # where livetime = FitExposure (in years), unit_norm = 1e4
        # Override parent's NORM to match this convention
        self.NORM = self.FitExposure * 1e4  # livetime * unit_norm
        
        # Base weights for histogram filling
        self.BaseWeight = self.Weight * self.NORM
        
        # Clean up raw MC dict to save memory
        del self.MC
    
    def MuonVariables(self):
        """
        Process atmospheric muon background.
        """
        if not hasattr(self, 'MuonMC') or len(self.MuonMC) == 0:
            self.HasMuonBkg = False
            self.MuonEReco = None
            self.MuonCosZReco = None
            self.MuonSample = None
            self.MuonWeight = None
            self.MuonWeightVariance = None
            return
        
        self.HasMuonBkg = True
        self.MuonEReco = self.MuonMC['reco_energy']
        self.MuonCosZReco = np.cos(self.MuonMC['reco_zenith'])
        self.MuonSample = self.MuonMC['pid'].astype(int)
        self.MuonWeight = self.MuonMC['weight']
        
        if 'weight_variance' in self.MuonMC:
            self.MuonWeightVariance = self.MuonMC['weight_variance']
        else:
            self.MuonWeightVariance = self.MuonWeight**2
        
        # Pre-compute binned muon background
        self._compute_muon_background()
        
        del self.MuonMC
    
    def _compute_muon_background(self):
        """
        Pre-bin the atmospheric muon background.
        """
        if not self.HasMuonBkg:
            self.MuonBkgBinned = None
            self.MuonBkgVarBinned = None
            return
        
        binned_events = []
        binned_variance = []
        
        for m in self.Samples:
            mask = self.MuonSample == m
            E_sample = self.MuonEReco[mask]
            cz_sample = self.MuonCosZReco[mask]
            w_sample = self.MuonWeight[mask]
            var_sample = self.MuonWeightVariance[mask]
            
            hist, _, _ = np.histogram2d(
                E_sample, cz_sample,
                bins=[self.EnergyBins[m], self.CTBins[m]],
                weights=w_sample
            )
            var_hist, _, _ = np.histogram2d(
                E_sample, cz_sample,
                bins=[self.EnergyBins[m], self.CTBins[m]],
                weights=var_sample
            )
            
            binned_events.append(hist.flatten())
            binned_variance.append(var_hist.flatten())
        
        self.MuonBkgBinned = np.concatenate(binned_events)
        self.MuonBkgVarBinned = np.concatenate(binned_variance)
    
    def DataVariables(self):
        """
        Process data variables for data-fitting mode.
        """
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
                AtmInitialFlux[ic][ie][0][2] = 0.0  # nutau (no atmospheric)
                
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
        Set up ORCA binning scheme.

        Tries to load binning from numpy files in bin_dir (or same directory as MC files).
        Falls back to default binning if files not found.

        Uses 2D binning in (E_reco, cos_theta_reco) for each morphology.
        """
        import os

        # Try to find binning files in the data directory
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
                    print(f"[ORCA] Loaded energy bins from {erec_file}")
                except Exception as e:
                    print(f"[ORCA] Warning: Could not load {erec_file}: {e}")

            if os.path.exists(cz_file):
                try:
                    cz_bins = np.load(cz_file)
                    print(f"[ORCA] Loaded zenith bins from {cz_file}")
                except Exception as e:
                    print(f"[ORCA] Warning: Could not load {cz_file}: {e}")

        # Fall back to default binning if files not found
        if erec is None:
            self.NErec = 20
            erec = np.logspace(np.log10(1.0), np.log10(100.0), self.NErec + 1)
            print("[ORCA] Using default energy binning")
        else:
            self.NErec = len(erec) - 1

        if cz_bins is None:
            cz_bins = np.linspace(-1, 0, 11)
            print("[ORCA] Using default zenith binning")

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
    
    def BinData(self, entries=None):
        """
        Bin data events using weights.
        
        Args:
            entries: Optional weight array (defaults to self.dWeight)
        
        Returns:
            Flattened array of binned data
        """
        self.dCosThetaReco = self.dCosZReco
        # Use data weights by default
        if entries is None and hasattr(self, 'dWeight'):
            entries = self.dWeight
        return self.BinIt_Data_2D(entries)
    
    def GetMuonBackground(self):
        """
        Return the pre-computed muon background.
        
        Returns:
            Tuple of (binned_events, binned_variance) or (None, None)
        """
        if self.HasMuonBkg:
            return self.MuonBkgBinned, self.MuonBkgVarBinned
        return None, None
    
    def GetMCVariance(self, array):
        """
        Compute MC statistical variance for given weights.
        
        For weighted histograms, variance = sum(w^2 * variance_per_event)
        
        Args:
            array: Weight array
        
        Returns:
            Flattened array of variances per bin
        """
        # Compute variance weights: (weight * array)^2 * relative_variance
        # Handle division by zero - where Weight is zero, set variance to 0
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
Orca = ORCA_Atm
