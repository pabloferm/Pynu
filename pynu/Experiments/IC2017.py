# Class for the atmospheric neutrinos in IceCube-Upgrade

import numpy as np
import nuflux
from .Experiment import Experiment


class IC2017(Experiment):
    def __init__(self, dict_of_details, scenario):
        super(IC2017, self).__init__(dict_of_details)

        self.Detector = "IceCube-2017"
        self.Target = "Water"
        self.SOURCE = "Atmospheric"
        self.SCENARIO = scenario

        self.SetDefinition()

        self.MCVariables()

        self.Binning()
        self.SetBinner_2D()

        if self.DataFit:
            self.DataVariables()
            self.BinData()

    def MCVariables(self):
        d_itype = self.MC["pid"]
        d_Etrue = self.MC["true_energy"]
        self.EReco = self.MC["reco_energy"]
        self.CosZReco = np.cos(self.MC["reco_coszen"])
        self.CosZTrue = np.cos(self.MC["true_coszen"])
        # self.AziTrue = self.MC['true_azimuth']
        self.CC = self.MC["type"] > 0
        self.NC = self.MC["type"] == 0
        self.nuPDG = np.int_(self.MC["pdg"])
        self.ETrue = self.MC["true_energy"]
        self.Weight = self.MC["weight"]
        self.Sample = self.MC["pid"]  # Sample of each event
        self.Mode = self.NEUTMode()

        self.NumberOfEvents = self.Sample.size
        self.Samples = np.unique(self.Sample)  # Samples in the analysis
        # Number of samples in the analysis
        self.NumberOfSamples = 1 + np.amax(self.Samples)
        self.Erec_min = 5.623413
        self.Erec_max = 56.23413
        self.Etrue_min = np.amin(self.ETrue)
        self.Etrue_max = np.amax(self.ETrue)
        self.E_edges = [5.623413, 56.23413]
        # self.E_edges = [self.Erec_min, self.Erec_max]
        self.Z_edges = [-1, 1]

        self.NORM *= 1006 * 24 * 60 * 60 * 1e4

        self.BaseWeight = self.Weight * self.NORM

        del self.MC

    def SetInitialFlux(self, energy_nodes, cth_nodes, neutrino_flavors):
        flux = nuflux.makeFlux('IPhonda2014_spl_solmin')

        AtmInitialFlux = np.zeros(
            (len(cth_nodes), len(energy_nodes), 2, neutrino_flavors))

        for ic, nu_cos_zenith in enumerate(cth_nodes):
            for ie, nu_energy in enumerate(energy_nodes):
                AtmInitialFlux[ic][ie][0][0] = flux.getFlux(
                    nuflux.NuE, nu_energy, nu_cos_zenith)  # nue
                AtmInitialFlux[ic][ie][1][0] = flux.getFlux(
                    nuflux.NuEBar, nu_energy, nu_cos_zenith)  # nue bar
                AtmInitialFlux[ic][ie][0][1] = flux.getFlux(
                    nuflux.NuMu, nu_energy, nu_cos_zenith)  # numu
                AtmInitialFlux[ic][ie][1][1] = flux.getFlux(
                    nuflux.NuMuBar, nu_energy, nu_cos_zenith)  # numu bar
                AtmInitialFlux[ic][ie][0][2] = 0.  # nutau
                AtmInitialFlux[ic][ie][1][2] = 0.  # nutau bar
        return AtmInitialFlux

    def NEUTMode(self):
        noNEUTmode = self.MC["type"]
        nuPDG = self.MC["pdg"]
        c_mode = np.logical_and(nuPDG > 0, noNEUTmode == 0)
        noNEUTmode[c_mode] = 31
        c_mode = np.logical_and(nuPDG > 0, noNEUTmode == 1)
        noNEUTmode[c_mode] = 1
        c_mode = np.logical_and(nuPDG > 0, noNEUTmode == 2)
        noNEUTmode[c_mode] = 11
        c_mode = np.logical_and(nuPDG > 0, noNEUTmode == 3)
        noNEUTmode[c_mode] = 26
        c_mode = np.logical_and(nuPDG > 0, noNEUTmode == 4)
        noNEUTmode[c_mode] = 16
        c_mode = np.logical_and(nuPDG < 0, noNEUTmode == 0)
        noNEUTmode[c_mode] = -31
        c_mode = np.logical_and(nuPDG < 0, noNEUTmode == 1)
        noNEUTmode[c_mode] = -1
        c_mode = np.logical_and(nuPDG < 0, noNEUTmode == 2)
        noNEUTmode[c_mode] = -11
        c_mode = np.logical_and(nuPDG < 0, noNEUTmode == 3)
        noNEUTmode[c_mode] = -26
        c_mode = np.logical_and(nuPDG < 0, noNEUTmode == 4)
        noNEUTmode[c_mode] = -16
        return noNEUTmode

    def DataVariables(self):
        self.Data["pid"]
        self.dEReco = self.Data["reco_energy"]
        self.dCosZReco = np.cos(self.Data["reco_coszen"])
        self.dSample = self.Data["pid"]  # Sample of each event
        self.dNumberOfEvents = self.Sample.size

        del self.Data

    def BinMC(self, array):
        self.CosThetaReco = self.CosZReco
        return self.BinIt_MC_2D(array)

    def BinData(self):
        self.dCosThetaReco = self.dCosZReco
        return self.BinIt_Data_2D()

    def Binning(self):
        NErec = 8
        # erec = np.logspace(
        #    np.log10(
        #        self.Erec_min), np.log10(
        #        self.Erec_max), NErec + 1, endpoint=True)
        erec = np.array(
            [
                5.623413,
                7.498942,
                10.0,
                13.335215,
                17.782795,
                23.713737,
                31.622776,
                42.16965,
                56.23413,
            ]
        )

        z10bins = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])

        self.EnergyBins = {0: erec, 1: erec}
        self.CTBins = {0: z10bins, 1: z10bins}
