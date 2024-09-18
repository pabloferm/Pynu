# Class for the atmospheric neutrinos in Super-Kamiokande using the MC
# simulation developed with public information

import numpy as np
import nuflux
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
        self.Etrue_min = 0.1
        self.Etrue_max = 1e3
        self.E_edges = [self.Erec_min, self.Erec_max]
        self.Z_edges = [-1, 1]

        self.NORM *= 1

        self.BaseWeight = self.Weight * self.NORM

        del self.MC

    def SetInitialFlux(self, energy_nodes, cth_nodes, neutrino_flavors):
        flux = nuflux.makeFlux("IPhonda2014_sk_solmin")

        AtmInitialFlux = np.zeros(
            (len(cth_nodes), len(energy_nodes), 2, neutrino_flavors)
        )

        for ic, nu_cos_zenith in enumerate(cth_nodes):
            for ie, nu_energy in enumerate(energy_nodes):
                AtmInitialFlux[ic][ie][0][0] = flux.getFlux(
                    nuflux.NuE, nu_energy, nu_cos_zenith
                )  # nue
                AtmInitialFlux[ic][ie][1][0] = flux.getFlux(
                    nuflux.NuEBar, nu_energy, nu_cos_zenith
                )  # nue bar
                AtmInitialFlux[ic][ie][0][1] = flux.getFlux(
                    nuflux.NuMu, nu_energy, nu_cos_zenith
                )  # numu
                AtmInitialFlux[ic][ie][1][1] = flux.getFlux(
                    nuflux.NuMuBar, nu_energy, nu_cos_zenith
                )  # numu bar
                AtmInitialFlux[ic][ie][0][2] = 0.0  # nutau
                AtmInitialFlux[ic][ie][1][2] = 0.0  # nutau bar
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
        z10bins = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        z1bins = np.array([-1, 1.0])
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
        d_itype = self.MC["itype"]
        # condition = d_itype > -1
        self.EReco = self.MC["evis"]
        self.CosZReco = self.MC["recodirZ"]
        self.CosZTrue = self.MC["dirnuZ"]
        self.AziTrue = self.MC["azi"]
        self.Mode = self.MC["mode"]
        self.CC = np.abs(self.Mode) < 30
        self.nuPDG = self.MC["ipnu"]
        self.ETrue = self.MC["pnu"]
        self.Weight = self.MC["tune_weights"] * self.MC["inv_flux"]
        self.Sample = self.MC["itype"]  # Sample of each event
        self.DecayE = self.MC["muedk"]

        self.NumberOfEvents = self.Sample.size
        self.Samples = np.unique(self.Sample)  # Samples in the analysis
        self.NumberOfSamples = 1 + np.amax(self.Samples)
        self.NumberOfSamples = self.NumberOfSamples.astype(int)
        self.Erec_max = max(self.EReco)
        self.Erec_min = min(self.EReco)
        self.Etrue_min = min(self.ETrue)
        self.Etrue_max = max(self.ETrue)
        self.E_edges = [self.Erec_min, self.Erec_max]
        self.Z_edges = [-1, 1]

        self.BaseWeight = self.Weight * self.NORM

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

        del self.Data

    def BinData(self):
        self.dCosThetaReco = self.dCosZReco
        return self.BinIt_Data_2D(entries=self.dEntries)

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
            [-1, -0.839, -0.644, -0.448, -0.224, 0.0, 0.224, 0.448, 0.644, 0.839, 1.0]
        )
        z10bins_up = np.array(
            [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0]
        )
        z1bins = np.array([-1, 1.0])

        self.EnergyBins = {
            0: sg_ebins,
            1: sg_ebins,
            2: sg_ebins,
            3: sg_ebins,
            4: sg_ebins,
            5: sg_ebins,
            6: sg_ebins,
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
