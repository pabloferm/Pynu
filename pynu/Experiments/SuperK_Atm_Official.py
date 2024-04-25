# Class for the atmospheric neutrinos in Super-Kamiokande using the
# official MC simulation

import numpy as np
import nuflux
from .Experiment import Experiment


class SuperK_I(Experiment):
    def __init__(self, dict_of_details, scenario):
        super(SuperK_I, self).__init__(dict_of_details)
        self.Detector = "SuperK_I"
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
        self.MC["pnu"]
        condition = d_itype > -1
        self.EReco = self.MC["amom"][condition].astype(np.float64) * 1e-3
        self.CosZReco = self.MC["dir"][:, 2][condition]
        self.CosZTrue = self.MC["dirnu"][:, 2][condition].astype(np.float64)
        # self.AziTrue = self.MC['azi']d_azi[condition]
        self.Mode = self.MC["mode"][condition]
        self.CC = np.abs(self.Mode) < 30
        self.nuPDG = self.MC["ipnu"][condition]
        self.ETrue = self.MC["pnu"][condition].astype(np.float64)
        self.Sample = self.MC["itype"][condition]  # Sample of each event
        self.DecayE = self.MC["muedk"][condition]
        self.Wall = self.MC["wall"][condition]
        self.MCWeight = self.MC["flxho"]
        self.Weight = self.SKFluxWeight()

        self.NumberOfEvents = self.Sample.size
        self.Samples = np.unique(self.Sample)  # Samples in the analysis
        self.First_Sample = np.amin(self.Samples)
        self.Samples -= self.First_Sample
        self.Sample -= self.First_Sample
        self.NumberOfSamples = 1 + np.amax(self.Samples) - np.amin(self.Samples)
        self.Erec_max = 1e5
        self.Erec_min = 0.1
        self.Etrue_min = np.amin(self.ETrue)
        self.Etrue_max = np.amax(self.ETrue)
        self.E_edges = [self.Erec_min, self.Erec_max]
        self.Z_edges = [-1, 1]

        self.BaseWeight = self.NORM * self.Weight

        # del self.MC

    def SKFluxWeight(self):
        flux = nuflux.makeFlux("IPhonda2014_sk_solmin")
        nus = {
            12: nuflux.NuE,
            -12: nuflux.NuEBar,
            14: nuflux.NuMu,
            -14: nuflux.NuMuBar,
            16: nuflux.NuMu,
            -16: nuflux.NuMuBar,
        }

        flux_weight = np.array(
            [
                flux.getFlux(nus[v], E, cz)
                for v, E, cz in zip(self.nuPDG, self.ETrue, self.CosZTrue)
            ],
            dtype=np.double,
        )

        inv_flux_weight = np.zeros_like(flux_weight)
        inv_flux_weight[flux_weight > 0] = 1 / flux_weight[flux_weight > 0]

        # for here, there in zip(inv_flux_weight,self.MCWeight):
        #     print(here, there)

        return np.array(inv_flux_weight)

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
        condition = d_itype > -1
        self.dEReco = self.Data["evis"][condition].astype(np.float64) * 1e-3
        self.dCosZReco = self.Data["recodirZ"][condition]
        self.dSample = self.Data["itype"][condition]  # Sample of each event
        self.dSample -= self.First_Sample
        self.dDecayE = self.Data["muedk"][condition]
        self.dWall = self.Data["wall"][condition]
        self.dNumberOfEvents = self.Sample.size

        # del self.Data

    def BinMC(self, array):
        self.CosThetaReco = self.CosZReco
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
        umstop_ebins = np.array([1.0, 8.0, 20.0, 1.0e5])
        pcs_ebins = np.array([0.1, 10.0, 1.0e5])
        pct_ebins = np.array([0.1, 10.0, 50.0, 1.0e5])
        um_ebins = np.array([1.0, 1.0e5])
        z10bins = np.linspace(-1, 1, 10 + 1)
        z10bins_up = np.linspace(-1, 0, 10 + 1)
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
            16: umstop_ebins,
            17: um_ebins,
            18: um_ebins,
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
        }


class SuperK_II(SuperK_I):
    def __init__(self, dict_of_details, scenario):
        super(SuperK_II, self).__init__(dict_of_details, scenario)
        self.Detector = "SuperK_II"

        self.SetDefinition()


class SuperK_III(SuperK_I):
    def __init__(self, dict_of_details, scenario):
        super(SuperK_III, self).__init__(dict_of_details, scenario)
        self.Detector = "SuperK_III"

        self.SetDefinition()


class SuperK_IV_noNtag(SuperK_I):
    def __init__(self, dict_of_details, scenario):
        super(SuperK_IV_noNtag, self).__init__(dict_of_details, scenario)
        self.Detector = "SuperK_IV_noNtag"

        self.SetDefinition()


class SuperK_IV(SuperK_I):
    def __init__(self, dict_of_details, scenario):
        super(SuperK_IV, self).__init__(dict_of_details, scenario)

        self.Detector = "SuperK_IV"

        self.SetDefinition()

        self.AddMCVariables()

        # self.set_KDE_1D()

        self.DeleteBinner()
        self.Binning()
        self.SetBinner_2D()

        if self.DataFit:
            self.AddDataVariables()
            self.DataVariables()
            self.BinData()

        # del self.MC

    def AddMCVariables(self):
        d_itype = self.MC["itype"]
        condition = d_itype > -1
        self.NN = self.MC["nn"][condition]
        self.trueNN = self.MC["nn_mctruth"][condition]

    def AddDataVariables(self):
        d_itype = self.Data["itype"]
        condition = d_itype > -1
        self.dNN = self.Data["nn"][condition]
        self.dtrueNN = self.Data["nn_mctruth"][condition]

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
        umstop_ebins = np.array([1.0, 8.0, 20.0, 1.0e5])
        um_ebins = np.array([1.0, 1.0e5])
        z10bins = np.linspace(-1, 1, 10 + 1)
        z10bins_up = np.linspace(-1, 0, 10 + 1)
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
            18: umstop_ebins,
            19: um_ebins,
            20: um_ebins,
        }
        self.CTBins = {
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
            18: z10bins_up,
            19: z10bins_up,
            20: z10bins_up,
        }


class SuperK_V(SuperK_IV):
    def __init__(self, dict_of_details, scenario):
        super(SuperK_V, self).__init__(dict_of_details, scenario)

        self.Detector = "SuperK_V"

        self.SetDefinition()


class SuperK_VI(SuperK_IV):
    def __init__(self, dict_of_details, scenario) -> None:
        super(SuperK_VI, self).__init__(dict_of_details, scenario)

        self.Detector = "SuperK_VI"

        self.SetDefinition()
