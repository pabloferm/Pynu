# General experiment class

import pathlib
from .MCReader import reader
import numpy as np
import boost_histogram as bh


class Experiment:
    def __init__(self, dict_of_details):
        self.Detector = None
        self.Target = None
        self.Source = None
        self.Scenario = None

        self.TotalMCexposure = dict_of_details['TotalMCexposure']
        self.FitExposure = dict_of_details['Exposure']
        self.FewEntries = None
        self.Norm = self.FitExposure / self.TotalMCexposure
        self.MCFiles = dict_of_details['MCFiles']
        self.DataFiles = dict_of_details['DataFiles']

        if len(self.DataFiles) > 0:
            self.DataFit = True
        else:
            self.DataFit = False

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

    def SetDefinition(self):
        self.Definition = {
            self.Detector: 'Detector',
            self.Target: 'XSection',
            self.Source: 'Flux',
            self.Scenario: 'Osc'}

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
                            'Warning: MC files have not the same variables, it may produce errors.')

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
                                'Warning: Data files have not the same variables, it may produce errors.')

    def SetBinner_1D(self):  # 1D energy binning
        self.Binner = [
            bh.Histogram(bh.axis.Variable(self.EnergyBins[s]))
            for s in range(self.NumberOfSamples)]

    def SetBinner_2D(self):  # 1D energy binning
        self.Binner = [
            bh.Histogram(
                bh.axis.Variable(self.EnergyBins[s]),
                bh.axis.Variable(self.CTBins[s]))
            for s in range(self.NumberOfSamples)]

    def DeleteBinner(self):
        self.Binner = []

    def BinIt_MC_1D(self, array, shift_E=1, bias_E=0):  # 1D energy binning
        for hist in self.Binner:
            hist.reset()

        if shift_E == 1 and bias_E == 0:
            E = self.EReco
        else:
            E = self.EReco * shift_E + bias_E

        v = np.array([])
        for i,hist in enumerate(self.Binner):
            v = np.hstack((v, hist.fill(
                E[self.Sample == i],
                weight=array[self.Sample == i] * self.BaseWeight[self.Sample == i]).values().reshape(-1)))

        return v

    def BinIt_MC_2D(self, array, shift_E=1, bias_E=0):  # 2D energy and cos(angle) binning
        for hist in self.Binner:
            hist.reset()

        if shift_E == 1 and bias_E == 0:
            E = self.EReco
        else:
            E = self.EReco * shift_E + bias_E
        # v = np.ndarray([hist.fill(E[self.Sample == i],
        #                self.CosThetaReco[self.Sample == i],
        #                weight=array[self.Sample == i] * self.BaseWeight[self.Sample == i]).values() for i,
        #      hist in enumerate(self.Binner)], dtype=object)
        v = np.array([])
        for i,hist in enumerate(self.Binner):
            v = np.hstack((v,hist.fill(E[self.Sample == i], self.CosThetaReco[self.Sample == i], weight=array[self.Sample == i] * self.BaseWeight[self.Sample == i]).values().reshape(-1)))

        return v

    def BinIt_Data_1D(self):  # 1D energy binning
        v = np.array([])
        for i,hist in enumerate(self.Binner):
            v = np.hstack((v, hist.fill(self.dEReco[self.dSample == i]).values().reshape(-1)))
        return v

    def BinIt_Data_2D(self):  # 2D energy and cos(angle) binning
        v = np.array([])
        for i,hist in enumerate(self.Binner):
            v = np.hstack((v,hist.fill(
                self.dEReco[self.dSample == i],
                self.dCosThetaReco[self.dSample == i]).values().reshape(-1)))
        return v

    # Contains all default weights of the analysis

    def StartPhysicsWeights(self):  # Starts expected weights with fixed values
        self.PhysicsWeight = 1

    def UpdatePhysicsWeights(self, w):
        self.PhysicsWeight = w * self.PhysicsWeight

    # Contains all non-changing weights of the analysis, i.e. fixed
    def UpdateBaseWeights(self, w):
        self.BaseWeight = w * self.BaseWeight

    # Contains all weights of the analysis except for those relative to
    # nuisance parameters
    def StartNuisanceWeights(self):  # Starts expected weights with fixed values
        self.NuisanceWeight = 1

    def UpdateNuisanceWeights(self, w):
        self.NuisanceWeight = w * self.NuisanceWeight

    # Contains all non-changing weights of the analysis, i.e. fixed
    def UpdateNominalWeights(self, w):
        self.NominalWeight = w * self.NominalWeight

    def SetExpectedWeight(self):
        self.ExpectedWeight = self.PhysicsWeight * self.NuisanceWeight

    def SetExpectedBinned(self):
        self.ExpectedBinned = self.BinMC(self.ExpectedWeight)
        self.RemoveFewEntries('Expected')

    def SetObservedBinned(self):
        if self.DataFit:
            self.ObservedBinned = self.BinData()
        else:
            self.ObservedBinned = self.BinMC(self.NominalWeight)
        self.FewEntries = self.ObservedBinned > 4
        self.RemoveFewEntries('Observed')

    def GetObservedBinned(self):
        return self.ObservedBinned

    def GetExpectedBinned(self):
        return self.ExpectedBinned

    def RemoveFewEntries(self, which):
        if which == 'Observed':
            self.ObservedBinned = self.ObservedBinned[self.FewEntries]
        # elif which == 'Nominal':
        # 	self.NominalBinned = self.NominalBinned[self.FewEntries]
        elif which == 'Expected':
            self.ExpectedBinned = self.ExpectedBinned[self.FewEntries]
        else:
            print('Warning: No valid item to remove entries with few bins, please select Observed, Nominal or Expected.')
