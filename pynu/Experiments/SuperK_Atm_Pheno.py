# Class for the atmospheric neutrinos in Super-Kamiokande using the MC
# simulation developed with public information

import numpy as np
import nuflux
import os
import pandas as pd
from .Experiment import Experiment


class SuperK(Experiment):
    def __init__(self, dict_of_details, scenario):
        super(SuperK, self).__init__(dict_of_details)
        self.Detector = "SuperK_Pheno"
        self.SOURCE = "Atmospheric"
        self.Target = "Water"
        self.SCENARIO = scenario

        self.SetDefinition()

        self.MCVariables()

        self.Binning()
        self.SetBinner_2D()

        if self.DataFit:
            self.DataVariables()
            self.BinData()

    def MCVariables(self):
        d_itype = self.MC["itype"]
        condition = d_itype > -1
        self.EReco = self.MC["evis"][condition]
        self.CosZReco = self.MC["recodirZ"][condition]
        self.CosZTrue = self.MC["dirnuZ"][condition]
        self.AziTrue = self.MC["azi"][condition]
        self.Mode = self.MC["mode"][condition]
        self.CC = np.abs(self.Mode) < 30
        self.nuPDG = self.MC["ipnu"][condition]
        self.ETrue = self.MC["pnu"][condition]
        self.Weight = self.MC["weightReco"][condition] * self.MC["weightSim"][condition]
        self.Sample = self.MC["itype"][condition]  # Sample of each event
        self.DecayE = self.MC["muedk"][condition]

        self.NumberOfEvents = self.Sample.size
        self.Samples = np.unique(self.Sample)  # Samples in the analysis
        # self.NumberOfSamples = 1 + np.amax(self.Samples)
        self.NumberOfSamples = 16
        self.Erec_max = 4e2
        self.Erec_min = 0.1
        self.Etrue_min = 0.08
        self.Etrue_max = 4.5e4
        self.E_edges = [self.Erec_min, self.Erec_max]
        self.Z_edges = [-1, 1]

        self.NORM *= 1

        self.BaseWeight = self.Weight * self.NORM

        del self.MC

    def SetInitialFlux(self, energy_nodes, cth_nodes, neutrino_flavors):
        AtmInitialFlux = np.zeros(
            (len(cth_nodes), len(energy_nodes), 2, neutrino_flavors)
        )

        low_energy_nodes = energy_nodes[energy_nodes < 0.1]
        mid_energy_nodes = energy_nodes[(energy_nodes >= 0.1) & (energy_nodes <= 1e4)]
        hi_energy_nodes = energy_nodes[energy_nodes > 1e4]

        _mid_flux = nuflux.makeFlux("IPhonda2014_sk_solmin")
        _hi_flux = nuflux.makeFlux("honda2006")

        _low_fluka_flux = pd.read_csv(
            f"{os.environ['PYNU']}/../data/Kamioka_SolAvg_FLUKA_noerr.dat", sep=", ", engine="python",
        )
        _energy = _low_fluka_flux["E (GeV)"]

        for ic, nu_cos_zenith in enumerate(cth_nodes):
            for ie, nu_energy in enumerate(energy_nodes):
                if nu_energy < 1e-1:
                    AtmInitialFlux[ic][ie][0][0] = (
                        np.interp(nu_energy, _energy, _low_fluka_flux["NuE"])
                        / (4 * np.pi)
                        / 1000
                        / 9
                    )  # nue
                    AtmInitialFlux[ic][ie][1][0] = (
                        np.interp(nu_energy, _energy, _low_fluka_flux["NuEBar"])
                        / (4 * np.pi)
                        / 1000
                        / 9
                    )  # nue bar
                    AtmInitialFlux[ic][ie][0][1] = (
                        np.interp(nu_energy, _energy, _low_fluka_flux["NuMu"])
                        / (4 * np.pi)
                        / 1000
                        / 9
                    )  # numu
                    AtmInitialFlux[ic][ie][1][1] = (
                        np.interp(nu_energy, _energy, _low_fluka_flux["NuMuBar"])
                        / (4 * np.pi)
                        / 1000
                        / 9
                    )  # numu bar
                elif nu_energy > 1e4:
                    AtmInitialFlux[ic][ie][0][0] = _hi_flux.getFlux(
                        nuflux.NuE, nu_energy, nu_cos_zenith
                    )  # nue
                    AtmInitialFlux[ic][ie][1][0] = _hi_flux.getFlux(
                        nuflux.NuEBar, nu_energy, nu_cos_zenith
                    )  # nue bar
                    AtmInitialFlux[ic][ie][0][1] = _hi_flux.getFlux(
                        nuflux.NuMu, nu_energy, nu_cos_zenith
                    )  # numu
                    AtmInitialFlux[ic][ie][1][1] = _hi_flux.getFlux(
                        nuflux.NuMuBar, nu_energy, nu_cos_zenith
                    )  # numu bar
                else:
                    AtmInitialFlux[ic][ie][0][0] = _mid_flux.getFlux(
                        nuflux.NuE, nu_energy, nu_cos_zenith
                    )  # nue
                    AtmInitialFlux[ic][ie][1][0] = _mid_flux.getFlux(
                        nuflux.NuEBar, nu_energy, nu_cos_zenith
                    )  # nue bar
                    AtmInitialFlux[ic][ie][0][1] = _mid_flux.getFlux(
                        nuflux.NuMu, nu_energy, nu_cos_zenith
                    )  # numu
                    AtmInitialFlux[ic][ie][1][1] = _mid_flux.getFlux(
                        nuflux.NuMuBar, nu_energy, nu_cos_zenith
                    )  # numu bar
                AtmInitialFlux[ic][ie][0][2] = 0.0  # nutau
                AtmInitialFlux[ic][ie][1][2] = 0.0  # nutau bar

        del _mid_flux
        del _hi_flux
        del _low_fluka_flux
        return AtmInitialFlux

    def DataVariables(self):
        d_itype = self.Data["itype"]
        condition = (d_itype < 16) * (d_itype > -1)
        self.dEReco = self.Data["evis"][condition]
        self.dCosZReco = self.Data["recodirZ"][condition]
        self.dSample = self.Data["itype"][condition]  # Sample of each event
        self.dDecayE = self.Data["muedk"][condition]
        self.dNumberOfEvents = self.Sample.size

        del self.Data

    def BinMC(self, array, shift_E=1, bias_E=0):
        self.CosThetaReco = self.CosZReco  # redundant
        self.set_energy_bias(bias_E)
        self.set_energy_scale(shift_E)
        return self.BinIt_MC_2D(array)

    def BinData(self):
        self.dCosThetaReco = self.dCosZReco
        return self.BinIt_Data_2D()

    def Binning(self):
        sge_ebins = np.array([0.1, 0.25, 0.4, 0.63, 1.0, 1.33])
        sgm_ebins = np.array([0.1, 0.25, 0.4, 0.63, 1.0, 1.33])
        sgsrpi0ebins = np.array([0.1, 0.25, 0.4, 0.63, 1.33])
        sgmrpi0ebins = np.array([0.1, 0.15, 0.25, 0.4, 0.63, 1.33])
        mge_ebins = np.array([1.3, 2.5, 5.0, 10.0, 500.0])
        mgm_ebins = np.array([1.3, 3.0, 500.0])
        mre_ebins = np.array([1.3, 2.5, 5.0, 500.0])
        mrm_ebins = np.array([0.6, 1.3, 2.5, 5.0, 500.0])
        mro_ebins = np.array([1.3, 2.5, 5.0, 10.0, 500.0])
        pcs_ebins = np.array([0.1, 10.0, 1.0e5])
        pct_ebins = np.array([0.1, 10.0, 50.0, 1.0e5])
        z10bins = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        z1bins = np.array([-1.0, 1.0])
        self.EnergyBins = {
            0: sge_ebins,
            1: sge_ebins,
            2: sgsrpi0ebins,
            3: sgm_ebins,
            4: sgm_ebins,
            5: sgm_ebins,
            6: sgmrpi0ebins,
            7: mge_ebins,
            8: mge_ebins,
            9: mgm_ebins,
            10: mre_ebins,
            11: mre_ebins,
            12: mrm_ebins,
            13: mro_ebins,
            14: pcs_ebins,
            15: pct_ebins,
        }
        self.CTBins = {
            0: z10bins,
            1: z1bins,
            2: z1bins,
            3: z10bins,
            4: z10bins,
            5: z1bins,
            6: z1bins,
            7: z10bins,
            8: z10bins,
            9: z10bins,
            10: z10bins,
            11: z10bins,
            12: z10bins,
            13: z10bins,
            14: z10bins,
            15: z10bins,
        }


class SuperK_Htag(SuperK):
    def __init__(self, dict_of_details, scenario):
        super(SuperK_Htag, self).__init__(dict_of_details, scenario)

        self.Detector = "SuperK_Htag_Pheno"

        self.SetDefinition()

        self.Binning()

        self.SetBinner_2D()

        if self.DataFit:
            self.DataVariables()
            self.BinData()

    def Binning(self):
        sge_ebins = np.array([0.1, 0.25, 0.4, 0.63, 1.0, 1.33])
        sgm_ebins = np.array([0.1, 0.25, 0.4, 0.63, 1.0, 1.33])
        sgsrpi0ebins = np.array([0.1, 0.25, 0.4, 0.63, 1.33])
        sgmrpi0ebins = np.array([0.1, 0.15, 0.25, 0.4, 0.63, 1.33])
        mge_ebins = np.array([1.3, 2.5, 5.0, 10.0, 500.0])
        mgm_ebins = np.array([1.3, 3.0, 500.0])
        mre_ebins = np.array([1.3, 2.5, 5.0, 500.0])
        mrm_ebins = np.array([0.6, 1.3, 2.5, 5.0, 500.0])
        mro_ebins = np.array([1.3, 2.5, 5.0, 10.0, 500.0])
        pcs_ebins = np.array([0.1, 10.0, 1.0e5])
        pct_ebins = np.array([0.1, 10.0, 50.0, 1.0e5])
        z10bins = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        z1bins = np.array([-1, 1.0])
        self.EnergyBins = {
            0: sge_ebins,
            1: sge_ebins,
            2: sge_ebins,
            3: sgsrpi0ebins,
            4: sgm_ebins,
            5: sgm_ebins,
            6: sgmrpi0ebins,
            7: mge_ebins,
            8: mge_ebins,
            9: mge_ebins,
            10: mgm_ebins,
            11: mgm_ebins,
            12: mre_ebins,
            13: mre_ebins,
            14: mrm_ebins,
            15: mro_ebins,
            16: pcs_ebins,
            17: pct_ebins,
        }
        self.CzBins = {
            0: z1bins,
            1: z10bins,
            2: z10bins,
            3: z1bins,
            4: z10bins,
            5: z10bins,
            6: z1bins,
            7: z10bins,
            8: z10bins,
            9: z10bins,
            10: z10bins,
            11: z10bins,
            12: z10bins,
            13: z10bins,
            14: z10bins,
            15: z10bins,
            16: z10bins,
            17: z10bins,
        }


class SuperK_Gdtag(SuperK_Htag):
    def __init__(self, dict_of_details, scenario):
        super(SuperK_Gdtag, self).__init__(dict_of_details, scenario)
        self.Detector = "SuperK_Gdtag_Pheno"
        self.SetDefinition()


class SuperK_2023(SuperK):
    def __init__(self, dict_of_details, scenario):
        super(SuperK_2023, self).__init__(dict_of_details, scenario)
        self.Detector = "SuperK_pheno_2023"
        self.SetDefinition()

    def MCVariables(self):
        sample_condition = (self.MC["itype"] > -1) & (self.MC["itype"] < 300)
        #sample_condition = (sample_condition) & (self.MC["itype"] > -1) & (self.MC["w_no"]<1e9)
        d_itype = self.MC["itype"]#[sample_condition]
        self.EReco = self.MC["evis"]#[sample_condition]
        self.CosZReco = self.MC["recodirZ"]#[sample_condition]
        self.CosZTrue = self.MC["dirnuZ"]#[sample_condition]
        #self.AziTrue = self.MC["azi"]#[sample_condition]
        #self.Mode = self.MC["mode"]#[sample_condition]
        #self.CC = np.abs(self.Mode) < 30
        self.current = self.MC["current"].astype(str) #[sample_condition]
        self.nuPDG = self.MC["ipnu"]#[sample_condition]
        self.ETrue = self.MC["pnu"]#[sample_condition]
        self.Weight = self.MC["inv_flux"]#[sample_condition]
        self.WMC = self.MC["weight_genMC"]#[sample_condition] * self.MC["weight_tune"]#[sample_condition]
        self.Sample = self.MC["itype"]#[sample_condition]  # Sample of each event
        #self.DecayE = self.MC["muedk"]#[sample_condition]
        self.Bin = self.MC["bin_number"]#[sample_condition]
        self.wno = self.MC["w_no"]#[sample_condition]

        self.NumberOfEvents = self.Sample.size
        self.Samples = np.unique(self.Sample)  # Samples in the analysis
        self.NumberOfSamples = self.Samples.size
        #self.NumberOfSamples = self.NumberOfSamples.astype(int)
        self.Erec_max = max(self.EReco)
        self.Erec_min = min(self.EReco)
        self.Etrue_min = min(self.ETrue)
        self.Etrue_max = max(self.ETrue)
        self.E_edges = [self.Erec_min, self.Erec_max]
        self.Z_edges = [-1, 1]

        self.CC = np.full(self.NumberOfEvents, True)
        self.CC[self.current != "CC"] = False
        # No mode/interaction_type in this h5 — set zeros for safety
        self.Mode = np.zeros(self.NumberOfEvents, dtype=int)

        # NC weight fix: use w_no for NC events instead of inv_flux.
        # inv_flux is the raw inverse generation flux — it only gives physical
        # results when multiplied by nuSQuIDS oscillation probabilities, which
        # for NC events don't correctly reproduce the flavor-sum=1 identity.
        # w_no is the pre-computed weight at NO best-fit (=1.0 for all NC events).
        # Using w_no for NC events gives NC rates matching the SK data release.
        self.Weight[~self.CC] = self.wno[~self.CC]

        # MC statistical variance (for Barlow-Beeston)
        self.WeightVariance = self.Weight**2

        self.BaseWeight = self.Weight * self.NORM * self.WMC
        #self.BaseWeight = np.ones(self.NumberOfEvents) * self.NORM #* self.WMC

        del self.MC

    def sample_index(self, sample_name_array):
        self.sample_names = {
            0: "sk1-3_fc_subgev_1ring_elike_0decaye",
            1: "sk1-3_fc_subgev_1ring_elike_1decaye",
            2: "sk1-5_fc_1ring_ncpi0",
            3: "sk1-3_fc_subgev_1ring_mulike_0decaye",
            4: "sk1-3_fc_subgev_1ring_mulike_1decaye",
            5: "sk1-3_fc_subgev_1ring_mulike_2decaye",
            6: "sk1-5_fc_2ring_ncpi0",
            7: "sk1-3_fc_multigev_1ring_nuelike",
            8: "sk1-3_fc_multigev_1ring_nuebarlike",
            9: "sk1-3_fc_multigev_1ring_mulike",
            10: "sk1-5_fc_multigev_multiring_nuelike",
            11: "sk1-5_fc_multigev_multiring_nuebarlike",
            12: "sk1-5_fc_multigev_multiring_mulike",
            13: "sk1-5_fc_multigev_multiring_other",
            14: "sk1-5_pc_stop",
            15: "sk1-5_pc_thru",
            16: "sk1-5_upmu_stop",
            17: "sk1-5_upmu_thru_nonshowering",
            18: "sk1-5_upmu_thru_showering",
            19: "sk4-5_fc_subgev_1ring_nuelike",
            20: "sk4-5_fc_subgev_1ring_nuebarlike_0neutron",
            21: "sk4-5_fc_subgev_1ring_nuebarlike_1neutron",
            22: "sk4-5_fc_subgev_1ring_numulike",
            23: "sk4-5_fc_subgev_1ring_numubarlike",
            24: "sk4-5_fc_multigev_1ring_nuelike",
            25: "sk4-5_fc_multigev_1ring_nuebarlike_0neutron",
            26: "sk4-5_fc_multigev_1ring_nuebarlike_1neutron",
            27: "sk4-5_fc_multigev_1ring_numulike",
            28: "sk4-5_fc_multigev_1ring_numubarlike",
        }
        index = np.zeros_like(sample_name_array)
        inverted_sample_names = dict(
            zip(self.sample_names.values(), self.sample_names.keys())
        )
        for i, sample in enumerate(sample_name_array):
            index[i] = inverted_sample_names[sample]
        return index

    def DataVariables(self):
        self.dEReco = 0.5 * (self.Data["E_reco(up)"] + self.Data["E_reco(low)"])
        self.dCosZReco = 0.5 * (self.Data["Cz_reco(up)"] + self.Data["Cz_reco(low)"])
        self.dSample = self.sample_index(self.Data["Sample"])  # Sample of each event
        self.dEntries = self.Data["entries"]
        self.dNumberOfEvents = np.sum(self.dEntries)
        self.dBin = self.Data["Bin"]

        del self.Data

    def UpdatePhysicsWeights(self, w):
        """Override: keep NC events at PhysicsWeight=1.

        NC interactions are flavor-blind, so the total oscillation probability
        summed over all final flavors is identically 1. However, nuSQuIDS
        returns per-flavor probabilities that don't sum to exactly 1 for NC
        events in this MC. Force NC PhysicsWeight=1 to ensure correct NC rates.
        """
        super().UpdatePhysicsWeights(w)
        if hasattr(self, 'CC') and hasattr(self.PhysicsWeight, '__len__'):
            self.PhysicsWeight[~self.CC] = 1.0

    def BinMC(self, array, shift_E=1, bias_E=0):
        self.CosThetaReco = self.CosZReco  # redundant
        self.set_energy_bias(bias_E)
        self.set_energy_scale(shift_E)
        return self.BinIt_MC_2D(array)

    def GetMCVariance(self, array):
        """Compute MC statistical variance for Barlow-Beeston."""
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

    def BinData(self):
        self.dCosThetaReco = self.dCosZReco
        return self.BinIt_Data_2D(entries=self.dEntries)

    def BinIt_Data_2D(self, entries=None):  # 2D energy and cos(angle) binning
        dentries = np.zeros(930)
        for s in range(930):
            dentries[s] = np.sum(entries[self.dBin==s])
        return dentries


    def Binning(self):
        sg_ebins = np.array(
            [
                0.1,
                0.25118864315095796,
                0.3981071705534973,
                0.630957344480193,
                1.0,
                1.584893192461114,
            ]
        )
        sg_1_ebins = np.array(
            [
                0.1,
                0.15848931924611143,
                0.25118864315095796,
                0.3981071705534973,
                0.630957344480193,
                1.584893192461114,
            ]
        )
        mg_4_ebins = np.array([1.0, 2.5118864315095797, 5.011872336272725, 10.0, 100.0])
        mg_2_ebins = np.array([1.3, 2.5118864315095797, 100.0])
        mr_3_ebins = np.array([1.0, 2.5118864315095797, 5.011872336272725, 100.0])
        mr_4_ebins = np.array(
            [0.1, 1.3299998745408388, 2.5118864315095797, 5.011872336272725, 100.0]
        )
        pcs_ebins = np.array([0.1, 2.5118864315095797, 100.0])
        pct_ebins = np.array(
            [0.1, 1.32739445772974, 2.5118864315095797, 5.011872336272725, 100.0]
        )
        upmus_ebins = np.array(
            [1.584893192461114, 2.4945947269429536, 4.9888448746001215, 100000.0]
        )
        upmut_ebins = np.array([0.1, 100000.0])
        z10bins = np.array(
            [-1.0, -0.839, -0.644, -0.448, -0.224, 0.0, 0.224, 0.448, 0.644, 0.839, 1.0]
        )
        z10bins_up = np.array(
            [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0]
        )
        z1bins = np.array([-1.0, 1.0])

        self.EnergyBins = {
            0: sg_ebins,
            1: sg_ebins,
            2: sg_ebins,
            3: sg_ebins,
            4: sg_ebins,
            5: sg_ebins,
            6: sg_1_ebins,
            7: mg_4_ebins,
            8: mg_4_ebins,
            9: mg_2_ebins,
            10: mr_3_ebins,
            11: mr_3_ebins,
            12: mr_4_ebins,
            13: mg_4_ebins,
            14: pcs_ebins,
            15: pct_ebins,
            16: upmus_ebins,
            17: upmut_ebins,
            18: upmut_ebins,
            19: sg_ebins,
            20: sg_ebins,
            21: sg_ebins,
            22: sg_ebins,
            23: sg_ebins,
            24: mg_4_ebins,
            25: mg_4_ebins,
            26: mg_4_ebins,
            27: mg_2_ebins,
            28: mg_2_ebins,
        }
        self.CTBins = {
            0: z10bins,
            1: z1bins,
            2: z1bins,
            3: z10bins,
            4: z10bins,
            5: z1bins,
            6: z1bins,
            7: z10bins,
            8: z10bins,
            9: z10bins,
            10: z10bins,
            11: z10bins,
            12: z10bins,
            13: z10bins,
            14: z10bins,
            15: z10bins,
            16: z10bins_up,
            17: z10bins_up,
            18: z10bins_up,
            19: z10bins,
            20: z10bins,
            21: z10bins,
            22: z10bins,
            23: z10bins,
            24: z10bins,
            25: z10bins,
            26: z10bins,
            27: z10bins,
            28: z10bins,
        }


class SuperK_2023_NoUpMu(SuperK_2023):
    """SuperK_2023 with upward-going muon samples (16, 17, 18) excluded.

    Up-mu samples have different morphology (upward-going high-energy muons
    measured via Cherenkov light in the rock below the detector) that can
    cause fitting issues. This variant excludes them for cleaner fits.
    """

    UPMU_SAMPLES = {16, 17, 18}

    def __init__(self, dict_of_details, scenario):
        super().__init__(dict_of_details, scenario)
        self.Detector = "SuperK_pheno_2023_noupmu"
        self.SetDefinition()

    def MCVariables(self):
        super().MCVariables()
        # Filter out up-mu events
        keep = ~np.isin(self.Sample, list(self.UPMU_SAMPLES))
        for attr in ['EReco', 'CosZReco', 'CosZTrue', 'current', 'nuPDG',
                      'ETrue', 'Weight', 'WMC', 'Sample', 'Bin', 'wno',
                      'CC', 'Mode', 'WeightVariance', 'BaseWeight']:
            setattr(self, attr, getattr(self, attr)[keep])

        self.NumberOfEvents = self.Sample.size
        self.Samples = np.unique(self.Sample)
        self.NumberOfSamples = self.Samples.size

    def SetBinner_2D(self):
        """Use actual sample IDs as keys (handles non-contiguous sample indices)."""
        import boost_histogram as bh
        self.Binner = [
            bh.Histogram(
                bh.axis.Variable(self.EnergyBins[s]),
                bh.axis.Variable(self.CTBins[s])
            )
            for s in self.Samples
        ]

    def DataVariables(self):
        super().DataVariables()
        # Filter out up-mu data entries
        keep = ~np.isin(self.dSample, list(self.UPMU_SAMPLES))
        self.dEReco = self.dEReco[keep]
        self.dCosZReco = self.dCosZReco[keep]
        self.dSample = self.dSample[keep]
        self.dEntries = self.dEntries[keep]
        self.dBin = self.dBin[keep]
        self.dNumberOfEvents = np.sum(self.dEntries)

    def _upmu_bin_indices(self):
        """Compute bin indices (in the full 930-bin scheme) belonging to up-mu samples."""
        upmu_indices = []
        offset = 0
        for s in range(29):  # all original 29 samples
            n_e = len(self.EnergyBins[s]) - 1
            n_cz = len(self.CTBins[s]) - 1
            n_bins = n_e * n_cz
            if s in self.UPMU_SAMPLES:
                upmu_indices.extend(range(offset, offset + n_bins))
            offset += n_bins
        return set(upmu_indices)

    def BinIt_Data_2D(self, entries=None):
        """Bin data excluding up-mu bins."""
        # Build full 930-bin array
        dentries = np.zeros(930)
        for s in range(930):
            dentries[s] = np.sum(entries[self.dBin == s])
        # Remove up-mu bins
        upmu_bins = sorted(self._upmu_bin_indices())
        keep_mask = np.ones(930, dtype=bool)
        keep_mask[list(upmu_bins)] = False
        return dentries[keep_mask]
