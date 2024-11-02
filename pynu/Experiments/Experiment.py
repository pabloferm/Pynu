# General experiment class

from .MCReader import reader
import numpy as np
import boost_histogram as bh
from KDEpy import FFTKDE


class Experiment:
    def __init__(self, dict_of_details):
        self.Detector = None
        self.Target = None
        self.SOURCE = None
        self.SCENARIO = None

        self.TotalMCexposure = dict_of_details["TotalMCexposure"]
        self.FitExposure = dict_of_details["Exposure"]
        self.FewEntries = None
        self.NORM = self.FitExposure / self.TotalMCexposure
        self.MCFiles = dict_of_details["MCFiles"]
        self.DataFiles = dict_of_details["DataFiles"]

        self.DataFit = len(self.DataFiles) > 0
        self.Reader()

        self.FewEntries = []
        self.EReco = 0
        self.CosThetaReco = 0
        self.Sample = 0
        self.EnergyBins = []
        self.CTBins = []
        self.ExpectedWeight = 1

        self.PhysicsWeight = 1
        self.BaseWeight = 1
        self.NuisanceWeight = 1
        self.NominalWeight = 1

        self.ENERGY_BIAS = 0
        self.ENERGY_SCALE = 1

    def SetDefinition(self):
        self.Definition = {
            self.Detector: "Detector",
            self.Target: "XSection",
            self.SOURCE: "Flux",
            self.SCENARIO: "Osc",
        }

    def MCVariables(self):
        pass

    def Reader(self):
        self.MC = {}
        for i, f in enumerate(self.MCFiles):
            newdata = reader(f)
            if i == 0:
                self.MC = newdata
            else:
                for key, value in newdata.items():
                    if key in self.MC:
                        self.MC[key] = np.append(self.MC[key], value)
                    else:
                        print(
                            "Warning: MC files have not the same variables, it may produce errors."
                        )
        if self.DataFit:
            self.Data = {}
            for i, f in enumerate(self.DataFiles):
                newdata = reader(f)
                if i == 0:
                    self.Data = newdata
                else:
                    for key, value in newdata.items():
                        if key in self.Data:
                            self.Data[key] = np.append(self.Data[key], value)
                        else:
                            print(
                                "Notice: Data files have not the same variables, it might produce errors."
                            )

    def set_KDE_1D(self):
        self.KDEer = []
        kde = FFTKDE(bw="silverman", kernel="gaussian")
        data = self.EReco[self.Sample == 0]
        norm = data.size
        x, y = kde.fit(data)(2**10)
        import matplotlib.pyplot as plt

        y *= norm / np.sum(y)
        plt.plot(x, y, label="FFTKDE")
        plt.hist(data, bins=15)
        plt.show()

    def SetBinner_1D(self):  # 1D energy binning
        self.Binner = [
            bh.Histogram(bh.axis.Variable(self.EnergyBins[s]))
            for s in range(self.NumberOfSamples)
        ]

    def SetBinner_2D(self):  # 2D energy binning
        self.Binner = [
            bh.Histogram(
                bh.axis.Variable(self.EnergyBins[s]), bh.axis.Variable(self.CTBins[s])
            )
            for s in range(self.NumberOfSamples)
        ]

    def DeleteBinner(self):
        self.Binner = []

    def set_energy_bias(self, bias_E):
        self.ENERGY_BIAS = bias_E

    def set_energy_scale(self, scale_E):
        self.ENERGY_SCALE = scale_E

    def BinIt_MC_1D(self, array):  # 1D energy binning
        for hist in self.Binner:
            hist.reset()

        if self.ENERGY_SCALE == 1 and self.ENERGY_BIAS == 0:
            E = self.EReco
        else:
            E = self.EReco * self.ENERGY_SCALE + self.ENERGY_BIAS

        v_list = [None] * self.NumberOfSamples
        for i, hist in enumerate(self.Binner):
            sample_mask = self.Sample == i
            E_sample = E[sample_mask]
            weight_sample = array[sample_mask] * self.BaseWeight[sample_mask]

            hist.fill(E_sample, weight=weight_sample)
            v_list[i] = hist.values().reshape(-1)

        return np.concatenate(v_list)

    # 2D energy and cos(angle) binning
    def BinIt_MC_2D(self, array):
        for hist in self.Binner:
            hist.reset()

        if self.ENERGY_SCALE == 1 and self.ENERGY_BIAS == 0:
            E = self.EReco
        else:
            E = self.EReco * self.ENERGY_SCALE + self.ENERGY_BIAS

        v_list = [None] * self.NumberOfSamples
        for i, hist in enumerate(self.Binner):
            sample_mask = self.Sample == i
            E_sample = E[sample_mask]
            CosThetaReco_sample = self.CosThetaReco[sample_mask]
            weight_sample = array[sample_mask] * self.BaseWeight[sample_mask]

            hist.fill(E_sample, CosThetaReco_sample, weight=weight_sample)
            v_list[i] = hist.values().reshape(-1)

        return np.concatenate(v_list)

    def BinIt_Data_1D(self):  # 1D energy binning
        for hist in self.Binner:
            hist.reset()
        v_list = [None] * self.NumberOfSamples
        for i, hist in enumerate(self.Binner):
            sample_mask = self.dSample == i
            E_sample = self.dEReco[sample_mask]

            hist.fill(E_sample)
            v_list[i] = hist.values().reshape(-1)

        return np.concatenate(v_list)

    def BinIt_Data_2D(self, entries=None):  # 2D energy and cos(angle) binning
        for hist in self.Binner:
            hist.reset()
        v_list = [None] * self.NumberOfSamples
        # if np.any(entries):
        if entries is None:
            for i, hist in enumerate(self.Binner):
                sample_mask = self.dSample == i
                E_sample = self.dEReco[sample_mask]
                CosThetaReco_sample = self.dCosThetaReco[sample_mask]
                hist.fill(E_sample, CosThetaReco_sample)
                v_list[i] = hist.values().reshape(-1)

        else:
            for i, hist in enumerate(self.Binner):
                sample_mask = self.dSample == i
                E_sample = self.dEReco[sample_mask]
                CosThetaReco_sample = self.dCosThetaReco[sample_mask]
                weight_sample = entries[self.dSample == i]
                hist.fill(E_sample, CosThetaReco_sample, weight=weight_sample)
                v_list[i] = hist.values().reshape(-1)

        return np.concatenate(v_list)

    # Contains all default weights of the analysis
    def StartPhysicsWeights(self):
        """Start physics weights from scratch, i.e. equal to 1"""
        self.PhysicsWeight = 1

    def UpdatePhysicsWeights(self, w):
        """Update physics weights for the experiment by mutiplying the existing weights with the input vector `w`"""
        self.PhysicsWeight = self.PhysicsWeight * w

    # Contains all non-changing weights of the analysis, i.e. fixed
    def UpdateBaseWeights(self, w):
        self.BaseWeight = self.BaseWeight * w

    # Contains all weights of the analysis except for those relative to
    # nuisance parameters
    # Starts expected weights with fixed values
    def StartNuisanceWeights(self):
        self.NuisanceWeight = 1

    def UpdateNuisanceWeights(self, w):
        self.NuisanceWeight *= w

    # Contains all non-changing weights of the analysis, i.e. fixed
    def UpdateNominalWeights(self, w):
        self.NominalWeight *= w

    def SetExpectedWeight(self):
        self.ExpectedWeight = self.PhysicsWeight * self.NuisanceWeight

    def SetExpectedBinned(self):
        self.ExpectedBinned = self.BinMC(self.ExpectedWeight)
        self.RemoveFewEntries("Expected")

    def SetObservedBinned(self):
        if self.DataFit:
            self.ObservedBinned = self.BinData()
        else:
            self.ObservedBinned = self.BinMC(self.NominalWeight)

        self.FewEntries = self.ObservedBinned > 4
        self.RemoveFewEntries("Observed")

    def GetObservedBinned(self):
        return self.ObservedBinned

    def GetExpectedBinned(self):
        return self.ExpectedBinned

    def RemoveFewEntries(self, which):
        if which == "Observed":
            self.ObservedBinned = self.ObservedBinned[self.FewEntries]
        # elif which == 'Nominal':
        # 	self.NominalBinned = self.NominalBinned[self.FewEntries]
        elif which == "Expected":
            self.ExpectedBinned = self.ExpectedBinned[self.FewEntries]
        else:
            print(
                "Warning: No valid item to remove entries with few bins, please select Observed, Nominal or Expected."
            )
