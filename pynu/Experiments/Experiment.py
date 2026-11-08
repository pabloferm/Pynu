# General experiment class

from .MCReader import reader
import numpy as np
import boost_histogram as bh


class Experiment:
    def __init__(self, dict_of_details):
        self.MIN_ENTRIES = 5  # bins with no more observed entries than this are masked

        self.TotalMCexposure = dict_of_details["TotalMCexposure"]
        self.FitExposure = dict_of_details["Exposure"]
        self.NORM = self.FitExposure / self.TotalMCexposure
        self.MCFiles = dict_of_details["MCFiles"]
        self.DataFiles = dict_of_details["DataFiles"]

        self.DataFit = len(self.DataFiles) > 0
        self.Reader()

        self.NominalWeight = 1

    def Reader(self):
        self.MC = reader(self.MCFiles[0])
        self.Data = reader(self.DataFiles[0])

    def SetBinner_2D(self):  # 2D energy binning
        self.Binner = [
            bh.Histogram(
                bh.axis.Variable(self.EnergyBins[s]), bh.axis.Variable(self.CTBins[s])
            )
            for s in range(self.NumberOfSamples)
        ]

    # 2D energy and cos(angle) binning
    def BinIt_MC_2D(self, array):
        for hist in self.Binner:
            hist.reset()

        E = self.EReco

        v_list = [None] * self.NumberOfSamples
        for i, hist in enumerate(self.Binner):
            sample_mask = self.Sample == self.Samples[i]
            E_sample = E[sample_mask]
            CosThetaReco_sample = self.CosThetaReco[sample_mask]
            weight_sample = array[sample_mask] * self.BaseWeight[sample_mask]
            hist.fill(E_sample, CosThetaReco_sample, weight=weight_sample)
            v_list[i] = hist.values().reshape(-1)

        return np.concatenate(v_list)

    def UpdateNominalWeights(self, w):
        self.NominalWeight *= w

    def SetObservedBinned(self):
        if self.DataFit:
            self.ObservedBinned = self.BinData()
        else:
            self.ObservedBinned = self.BinMC(self.NominalWeight)

        self.FewEntries = self.ObservedBinned > self.MIN_ENTRIES
