# General experiment class

import pathlib
from .MCReader import reader
import numpy as np

class Experiment:
	def __init__(self, dict_of_details):
		self.Detector = None
		self.Target = None
		self.Source = None
		self.Scenario = None
		
		self.TotalMCexposure = dict_of_details['TotalMCexposure']
		self.FitExposure = dict_of_details['Exposure']
		self.FewEntries = None
		# self.Norm = self.FitExposure / self.TotalMCexposure
		self.Norm = 1
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

	def Definition(self):
		self.Definition = {self.Detector:'Detector', self.Target:'XSection', self.Source:'Flux', self.Scenario:'Osc'}

	def MCVariables(self):
		pass

	def Reader(self):
		self.MC = {}
		for i,f in enumerate(self.MCFiles):
			newdata = reader(f)
			if i==0:
				self.MC = newdata
			else:
				for key, value in newdata.items():
					if key in self.MC:
						self.MC[key] = np.append(self.MC[key],value)
					else:
						print('Warning: MC files have not the same variables, it may produce errors.')

		if self.DataFit:
			self.Data = {}
			for i,f in enumerate(self.DataFiles):
				newdata = reader(f)
				if i==0:
					self.Data = newdata
				else:
					for key, value in newdata.items():
						if key in self.Data:
							self.Data[key] = np.append(self.Data[key],value)
						else:
							print('Warning: Data files have not the same variables, it may produce errors.')

	# def Binning(self):
	# 	pass

	# def MakeInitialFlux(self):
	# 	pass


	def BinIt_MC_1D(self, array, shift_E=1, bias_E=0): # 1D energy binning
		v = np.array([])
		E = self.EReco * shift_E
		for s, sample in enumerate(self.Samples):
			cond = self.Sample==s
			dummy_w = array[cond]
			Obs, __ = np.histogram1d(E[cond], bins=self.EnergyBins[s], weights=dummy_w*self.Norm)
			# Obs, __ = np.histogram1d(E[cond], bins=self.EnergyBins[s], weights=dummy_w)
			v = np.append(v,Obs)
		return v.reshape(-1)

	def BinIt_MC_2D(self, array, shift_E=1, bias_E=0): # 2D energy and cos(angle) binning
		v = np.array([])
		E = self.EReco * shift_E
		for s, sample in enumerate(self.Samples):
			cond = self.Sample==s
			dummy_w = array[cond]
			Obs, __, __ = np.histogram2d(E[cond], self.CosThetaReco[cond], bins=(self.EnergyBins[s], self.CTBins[s]), weights=dummy_w*self.Norm)
			# Obs, __, __ = np.histogram2d(E[cond], self.CosThetaReco[cond], bins=(self.EnergyBins[s], self.CTBins[s]), weights=dummy_w)
			v = np.append(v,Obs)
		return v.reshape(-1)

	def BinIt_Data_1D(self): # 1D energy binning
		v = np.array([])
		for s, sample in enumerate(self.Samples):
			cond = self.dSample==s
			Obs, __ = np.histogram1d(self.dEReco[cond], bins=self.EnergyBins[s])
			v = np.append(v,Obs)
		return v.reshape(-1)

	def BinIt_Data_2D(self): # 2D energy and cos(angle) binning
		v = np.array([])
		for s, sample in enumerate(self.Samples):
			cond = self.dSample==s
			Obs, __, __ = np.histogram2d(self.dEReco[cond], self.dCosThetaReco[cond], bins=(self.EnergyBins[s], self.CTBins[s]))
			v = np.append(v,Obs)
		return v.reshape(-1)

	def UpdateExpectedWeights(self,w): # Contains all default weights of the analysis
		self.ExpectedWeight = w * self.ExpectedWeight

	def UpdateBaseWeights(self,w): # Contains all non-changing weights of the analysis, i.e. fixed
		self.BaseWeight = w * self.BaseWeight

	def StartExpectedWeights(self): # Starts expected weights with fixed values
		self.ExpectedWeight = self.BaseWeight

	def UpdateObservedWeights(self,w): # Contains all non-changing weights of the analysis, i.e. fixed
		self.NominalWeight = w * self.NominalWeight

	def BinNominalWeights(self):
		self.NominalBinned = self.BinMC(self.NominalWeight)

	def SetExpectedBinned(self):
		self.ExpectedBinned = self.BinMC(self.ExpectedWeight)
		self.RemoveFewEntries('Expected')

	def SetObservedBinned(self):
		if self.DataFit:
			self.ObservedBinned = self.BinData()
		else:
			self.BinNominalWeights()
			self.ObservedBinned = self.NominalBinned
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