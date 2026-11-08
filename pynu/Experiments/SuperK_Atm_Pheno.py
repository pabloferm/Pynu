# Class for the atmospheric neutrinos in Super-Kamiokande using the MC
# simulation developed with public information

import numpy as np
import nuflux
import os
import pandas as pd
from .Experiment import Experiment


class SuperK(Experiment):
    def __init__(self, dict_of_details):
        super(SuperK, self).__init__(dict_of_details)

        self.MCVariables()

        self.Binning()
        self.SetBinner_2D()

        self.DataVariables()

    def SetInitialFlux(self, energy_nodes, cth_nodes, neutrino_flavors):
        AtmInitialFlux = np.zeros(
            (len(cth_nodes), len(energy_nodes), 2, neutrino_flavors)
        )

        _mid_flux = nuflux.makeFlux("IPhonda2014_sk_solmin")
        _hi_flux = nuflux.makeFlux("honda2006")

        _low_fluka_flux = pd.read_csv(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data",
                         "SuperK", "Kamioka_SolAvg_FLUKA_noerr.dat"), sep=", ", engine="python",
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


class SuperK_2023(SuperK):
    def MCVariables(self):
        self.EReco = self.MC["evis"]
        self.CosZReco = self.MC["recodirZ"]
        self.CosZTrue = self.MC["dirnuZ"]
        self.current = self.MC["current"].astype(str)
        self.nuPDG = self.MC["ipnu"]
        self.ETrue = self.MC["pnu"]
        self.Weight = self.MC["inv_flux"]
        # The generation weights are optional columns. In an MC without them the event
        # multiplicity carries the rate, and the oversampling factor is <MCexposure>.
        try:
            weight_tune = self.MC["weight_tune"]
        except (KeyError, ValueError):
            weight_tune = 1.0
        try:
            self.WMC = self.MC["weight_genMC"] * weight_tune
        except (KeyError, ValueError):
            self.WMC = np.ones(self.nuPDG.size)
        self.Sample = self.MC["itype"]  # Sample of each event
        try:
            self.wno = self.MC["w_no"]
        except (KeyError, ValueError):
            # Only used by the NC weight fix below; 1.0 for every NC event.
            self.wno = np.ones(self.nuPDG.size)

        self.NumberOfEvents = self.Sample.size
        self.Samples = np.unique(self.Sample)  # Samples in the analysis
        self.NumberOfSamples = self.Samples.size
        self.Etrue_min = min(self.ETrue)
        self.Etrue_max = max(self.ETrue)
        self.Z_edges = [-1, 1]

        # Handle both |S3 ('CC'/'NC') and |S1 ('C'/'N') current-type encodings
        cc_mask = np.array([c.startswith("C") for c in self.current])
        self.CC = cc_mask
        # NEUT mode (signed); zeros if the MC has no mode column.
        try:
            self.Mode = self.MC["mode"].astype(int)
        except (KeyError, ValueError):
            self.Mode = np.zeros(self.NumberOfEvents, dtype=int)

        # SK run-period phase (sk_phase: 1..5 for SK-I..V); zeros if the MC has no
        # such column, which the response builder treats as a single era.
        try:
            self.SKPhase = self.MC["sk_phase"].astype(int)
        except (KeyError, ValueError):
            self.SKPhase = np.zeros(self.NumberOfEvents, dtype=int)

        # NC weight fix: use w_no for NC events instead of inv_flux.
        # inv_flux is the raw inverse generation flux. It only gives physical
        # results when multiplied by nuSQuIDS oscillation probabilities, which
        # for NC events don't correctly reproduce the flavor-sum=1 identity.
        # w_no is the pre-computed weight at NO best-fit (=1.0 for all NC events).
        # Using w_no for NC events gives NC rates matching the SK data release.
        self.Weight[~self.CC] = self.wno[~self.CC]

        self.BaseWeight = self.Weight * self.NORM * self.WMC

        del self.MC

    def DataVariables(self):
        self.dEntries = self.Data["entries"]
        self.dBin = self.Data["Bin"]

        del self.Data

    def BinMC(self, array):
        self.CosThetaReco = self.CosZReco
        return self.BinIt_MC_2D(array)

    def BinData(self):
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
