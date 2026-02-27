# Class for IceCube DeepCore 9-year atmospheric neutrinos
# Based on Phys. Rev. D 108, 012014 (2023)
# Modeled on ORCA_Atm for parquet input, muon background, weight_variance support

import numpy as np
import pandas as pd
import nuflux
from .Experiment import Experiment


class ICDeepCore_Atm(Experiment):
    """
    IceCube DeepCore 9-year atmospheric neutrino experiment class for Pynu.

    Handles:
    - Parquet file input format (converted from IceCube CSV data release)
    - 2 event morphologies (0: mixed/cascade-like, 1: tracks)
    - Atmospheric muon background (pre-binned from KDE)
    - MC statistical uncertainties (weight_variance)
    - 2D binning in reconstructed energy and cos(zenith)

    Weight convention:
    - MC weights are in (GeV cm² sr) = effective area × generation normalization
    - Oscillated flux from nuSQuIDS is in (GeV⁻¹ cm⁻² sr⁻¹ s⁻¹)
    - NORM = livetime_seconds = FitExposure_years × seconds_per_year
    - Expected events = Weight × NORM × PhysicsWeight
    """

    SECONDS_PER_YEAR = 3.15576e7  # Julian year

    def __init__(self, dict_of_details, scenario):
        # Initialize parent class (calls Reader())
        super(ICDeepCore_Atm, self).__init__(dict_of_details)

        # Low threshold for sparse bins (matches ORCA convention)
        self.MIN_ENTRIES = 0.01

        # Experiment identifiers for PhysicsTunes lookup
        self.Detector = "IceCube-DeepCore"
        self.Target = "Water"  # H₂O ice — same cross-sections as water
        self.SOURCE = "Atmospheric"
        self.SCENARIO = scenario

        self.SetDefinition()

        # IceCube DeepCore: 2 PID samples
        # 0: mixed/cascade-like (PID bin [0.55, 0.75])
        # 1: tracks (PID bin [0.75, 1.0])
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

        # Handle muon background if present
        self.MuonVariables()

    def Reader(self):
        """
        Override parent Reader to handle parquet files.
        Separates neutrino (MC_type != -1) from muon (MC_type == -1) events.
        """
        self.MC = {}
        for i, f in enumerate(self.MCFiles):
            if f.endswith('.parquet'):
                df = pd.read_parquet(f)
                # Filter to neutrino events only
                df_nu = df[df['MC_type'] != -1]
                newdata = {col: df_nu[col].values for col in df_nu.columns}
            else:
                # Fall back to parent reader for CSV/HDF5
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
        Extract and process MC variables for IceCube DeepCore.

        Expected columns in parquet:
        - true_energy, reco_energy (GeV)
        - true_zenith, reco_zenith (radians, converted to cos internally)
        - pdg (neutrino PDG code)
        - current_type (1=CC, 0=NC)
        - pid (0=mixed, 1=tracks)
        - weight (GeV cm² sr)
        - weight_variance (Poisson: weight²)
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
        self.CC = self.MC['current_type'][condition]  # 1=CC, 0=NC

        # Event classification (morphology/sample)
        self.Sample = self.MC['pid'][condition].astype(int)

        # Weights
        self.Weight = self.MC['weight'][condition]

        # Interaction mode (NEUT convention) for xsec systematics
        if 'interaction_type' in self.MC:
            self.Mode = self._NEUTMode(self.MC['interaction_type'], self.MC['pdg'])[condition]
        elif 'type' in self.MC:
            self.Mode = self._NEUTMode(self.MC['type'], self.MC['pdg'])[condition]
        else:
            self.Mode = np.zeros(condition.sum(), dtype=int)

        # MC statistical variance (for Barlow-Beeston)
        if 'weight_variance' in self.MC:
            self.WeightVariance = self.MC['weight_variance'][condition]
        else:
            self.WeightVariance = self.Weight**2  # Fallback: Poisson

        # Azimuth (not available in IceCube data release)
        self.AziTrue = np.zeros_like(self.ETrue)

        # Event counts
        self.NumberOfEvents = len(self.EReco)

        # Energy and zenith ranges for oscillation grid
        self.Erec_min = 1.0
        self.Erec_max = 200.0
        self.Etrue_min = 0.1
        self.Etrue_max = 1e4
        self.E_edges = [self.Erec_min, self.Erec_max]
        self.Z_edges = [-1, 1]  # Full zenith range for oscillation calculation

        # IceCube weight convention:
        # weight in (GeV cm² sr), flux in (GeV⁻¹ cm⁻² sr⁻¹ s⁻¹)
        # events = weight × flux × livetime_seconds
        self.NORM = self.FitExposure * self.SECONDS_PER_YEAR
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
        """
        Process atmospheric muon background.
        For IceCube, muon background is pre-binned (from KDE) and stored
        as fake events at bin centers in the parquet file.
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
        For IceCube, the muon "events" are already at bin centers with
        weight = count, so histogramming reproduces the original binned data.
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
        For IceCube, data "events" are at bin centers with weight = count.
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
        Set up IceCube DeepCore binning scheme.

        Default binning from the IceCube DeepCore 9-year analysis:
        - Energy: 10 log-spaced bins [6.31, 158.49] GeV (last bin double-width)
        - Cos(zenith): 10 linear bins [-1.0, 0.1]

        Tries to load from numpy files in the data directory first.
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
                    print(f"[ICDeepCore] Loaded energy bins from {erec_file}")
                except Exception as e:
                    print(f"[ICDeepCore] Warning: Could not load {erec_file}: {e}")

            if os.path.exists(cz_file):
                try:
                    cz_bins = np.load(cz_file)
                    print(f"[ICDeepCore] Loaded zenith bins from {cz_file}")
                except Exception as e:
                    print(f"[ICDeepCore] Warning: Could not load {cz_file}: {e}")

        # Fall back to default IceCube DeepCore binning
        if erec is None:
            erec = np.array([
                6.31, 8.45862141, 11.33887101, 15.19987592, 20.37559363,
                27.3136977, 36.61429921, 49.08185342, 65.79474104,
                88.19854278, 158.49
            ])
            print("[ICDeepCore] Using default IceCube energy binning")

        self.NErec = len(erec) - 1

        if cz_bins is None:
            cz_bins = np.array([
                -1., -0.89, -0.78, -0.67, -0.56, -0.45,
                -0.34, -0.23, -0.12, -0.01, 0.1
            ])
            print("[ICDeepCore] Using default IceCube zenith binning")

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

    def BinData(self, entries=None):
        """
        Bin data events using weights.

        Args:
            entries: Optional weight array (defaults to self.dWeight)

        Returns:
            Flattened array of binned data
        """
        self.dCosThetaReco = self.dCosZReco
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
        Load the 3 HS CSV files and build indexed arrays for interpolation.

        For each category and each deltam31 slice, stores slope arrays
        of shape (N_bins=200,) where bins are ordered as
        [sample0_flat, sample1_flat] with energy-slow, coszen-fast.

        Args:
            hs_dir: Path to directory containing hs_*.csv files
        """
        import os

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

        print(f"[ICDeepCore] Loaded hypersurfaces: {n_dm} deltam31 slices, "
              f"{n_bins_total} bins/category, 3 categories")

    def interpolate_hs(self, dm31):
        """
        Piecewise linear interpolation of HS slopes at given Δm²₃₁.

        Args:
            dm31: Δm²₃₁ value in eV²

        Returns:
            dict: {category: {intercept: arr(200), dom_eff: arr(200), ...}}
        """
        grid = self._hs_dm31_grid

        # Clamp to grid bounds
        if dm31 <= grid[0]:
            idx_lo, idx_hi, frac = 0, 0, 0.0
        elif dm31 >= grid[-1]:
            idx_lo, idx_hi, frac = len(grid) - 1, len(grid) - 1, 0.0
        else:
            idx_hi = np.searchsorted(grid, dm31)
            idx_lo = idx_hi - 1
            frac = (dm31 - grid[idx_lo]) / (grid[idx_hi] - grid[idx_lo])

        result = {}
        for cat_name, cat_data in self._hs_data.items():
            interp = {}
            for key in ['intercept'] + self.HS_SLOPE_NAMES:
                arr_lo = cat_data[key][idx_lo]
                arr_hi = cat_data[key][idx_hi]
                interp[key] = arr_lo + frac * (arr_hi - arr_lo)
            result[cat_name] = interp

        return result

    def bin_by_flavor(self, weight_array):
        """
        Bin events into the 3 HS flavor categories using histogram2d.

        Categories (matching HS CSV files):
          - nc_nue_cc: NC (all flavors) + CC νe
          - numu_cc:   CC νμ
          - nutau_cc:  CC ντ

        Args:
            weight_array: Per-event weight array (e.g., ExpectedWeight)

        Returns:
            dict: {category: arr(200)} of binned histograms
        """
        abs_pdg = np.abs(self.nuPDG)

        masks = {
            'nc_nue_cc': (self.CC == 0) | ((abs_pdg == 12) & (self.CC == 1)),
            'numu_cc':   (abs_pdg == 14) & (self.CC == 1),
            'nutau_cc':  (abs_pdg == 16) & (self.CC == 1),
        }

        result = {}
        for cat_name, mask in masks.items():
            binned_samples = []
            for m in self.Samples:
                sample_mask = mask & (self.Sample == m)
                hist, _, _ = np.histogram2d(
                    self.EReco[sample_mask],
                    self.CosZReco[sample_mask],
                    bins=[self.EnergyBins[m], self.CTBins[m]],
                    weights=(weight_array * self.BaseWeight)[sample_mask]
                )
                binned_samples.append(hist.flatten())
            result[cat_name] = np.concatenate(binned_samples)

        return result

    def apply_hs_correction(self, dm31, hs_params):
        """
        Apply hypersurface corrections to get the corrected expectation.

        Pipeline:
        1. Bin events by flavor category using current ExpectedWeight
        2. Interpolate HS slopes at given dm31
        3. Compute per-category correction factors
        4. Multiply each category histogram by its correction
        5. Sum categories and apply FewEntries mask

        Args:
            dm31: Δm²₃₁ value in eV² (for HS interpolation)
            hs_params: dict {param_name: value} for 5 HS parameters

        Returns:
            Masked corrected expectation array
        """
        # 1. Bin by flavor category
        flavor_hists = self.bin_by_flavor(self.ExpectedWeight)

        # 2. Get interpolated HS slopes
        hs_slopes = self.interpolate_hs(dm31)

        # 3-4. Apply correction per category and sum
        total = np.zeros_like(flavor_hists['nc_nue_cc'])
        for cat_name in self.HS_CATEGORIES:
            slopes = hs_slopes[cat_name]
            correction = slopes['intercept'].copy()
            for sname in self.HS_SLOPE_NAMES:
                correction += slopes[sname] * (hs_params[sname] - self.HS_NOMINALS[sname])
            total += flavor_hists[cat_name] * correction

        # 5. Apply FewEntries mask
        return total[self.FewEntries]


# Alias for MCReader compatibility
ICDeepCore = ICDeepCore_Atm
