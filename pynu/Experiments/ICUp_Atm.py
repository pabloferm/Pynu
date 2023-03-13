# Class for the atmospheric neutrinos in IceCube-Upgrade

import numpy as np
import pandas as pd
import h5py
import nuflux
from .Experiment import Experiment


class ICUp_Atm(Experiment):
    def __init__(self, dict_of_details, scenario):
        super(ICUp_Atm, self).__init__(dict_of_details)

        self.Detector = 'IceCube-Upgrade'
        self.Target = 'Water'
        self.Source = 'Atmospheric'
        self.Scenario = scenario

        self.Definition()

        self.MCVariables()

        self.Binning()
        self.SetBinner_2D()

        if self.DataFit:
            self.DataVariables()
            self.BinData()

    def MCVariables(self):
        d_itype = self.MC['pid']
        d_Etrue = self.MC['true_energy']
        condition = (d_itype < 16) * (d_itype > -1) * (d_Etrue > 1)
        self.EReco = self.MC['reco_energy'][condition]
        self.CosZReco = np.cos(self.MC['reco_zenith'][condition])
        self.CosZTrue = np.cos(self.MC['true_zenith'][condition])
        self.AziTrue = self.MC['true_azimuth'][condition]
        self.CC = self.MC['current_type'][condition]
        self.nuPDG = np.int_(self.MC['pdg'][condition])
        self.ETrue = self.MC['true_energy'][condition]
        self.Weight = self.MC['weight'][condition]
        self.Sample = self.MC['pid'][condition]  # Sample of each event
        self.Mode = self.NEUTMode()[condition]

        self.NumberOfEvents = self.Sample.size
        self.Samples = np.unique(self.Sample)  # Samples in the analysis
        # Number of samples in the analysis
        self.NumberOfSamples = 1 + np.amax(self.Samples)
        self.Erec_min = 1
        self.Erec_max = 1e3
        self.Etrue_min = 1
        self.Etrue_max = 1e3
        self.E_edges = [self.Erec_min, self.Erec_max]
        self.Z_edges = [-1, 1]

        self.Norm = 365 * 24 * 60 * 60 * 1e4 * self.FitExposure

        self.NominalWeight = self.Weight
        self.BaseWeight = self.Weight
        self.BaseAndPhysicsWeight = self.Weight
        self.ExpectedWeight = self.Weight

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
        noNEUTmode = self.MC['interaction_type']
        nuPDG = self.MC['pdg']
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
        d_itype = self.Data['pid']
        condition = (d_itype < 16) * (d_itype > -1)
        self.dEReco = self.Data['reco_energy'][condition]
        self.dCosZReco = np.cos(self.Data['reco_zenith'][condition])
        self.dSample = self.Data['pid'][condition]  # Sample of each event
        self.dNumberOfEvents = self.Sample.size

    def BinMC(self, array, shift_E=1, bias_E=0):
        self.CosThetaReco = self.CosZReco
        return self.BinIt_MC_2D(array, shift_E=1, bias_E=0)

    def BinData(self):
        self.dCosThetaReco = self.dCosZReco
        return self.BinIt_Data_2D()

    def Binning(self):
        NErec = 40
        erec = np.logspace(
            np.log10(
                self.Erec_min), np.log10(
                self.Erec_max), NErec + 1, endpoint=True)
        z10bins = np.array([-1, -0.8, -0.6, -0.4, -0.2,
                           0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        self.EnergyBins = {0: erec, 1: erec}
        self.CTBins = {0: z10bins, 1: z10bins}
