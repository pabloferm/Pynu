# General experiment class

import pathlib
from .MCReader import reader
import numpy as np

class Experiment:
	def __init__(self, dict_of_details):
		self.TotalMCexposure = dict_of_details['TotalMCexposure']
		self.FitExposure = dict_of_details['Exposure']
		self.FewEntries = None
		self.Norm = self.FitExposure / self.TotalMCexposure
		self.MCFiles = dict_of_details['MCFiles']
		self.DataFiles = dict_of_details['DataFiles']
		self.DataFit = False
		if len(self.DataFiles) > 0: 
			self.DataFit = True
		else:
			print('Sensitivity study.')

		self.Reader()

		self.FewEntries = []
		self.EReco = 0
		self.CosThetaReco = 0
		self.Sample = 0
		self.EnergyBins = []
		self.CTBins = []

	# def Detector(self):

	def Reader(self):
		self.MC = {}
		for f in self.MCFiles:
			# print(f)
			newdata = reader(f)
			self.MC =  {x: self.MC.get(x, 0) + newdata.get(x, 0) for x in set(self.MC).union(newdata)}

		if self.DataFit:
			self.Data = {}
			for f in self.DataFiles:
				newdata = reader(f)
				self.Data =  {x: self.Data.get(x, 0) + newdata.get(x, 0) for x in set(self.Data).union(newdata)}


	# def Binning(self):
	# 	pass

	# def MakeInitialFlux(self):
	# 	pass

	def wBinIt(self,array,shift_E=1, bias_E=0):
		return self.BinIt(array*self.Weight*self.Norm,shift_E=shift_E, bias_E=bias_E)

	def BinIt_1D(self, array, shift_E=1, bias_E=0): # 1D energy binning
		v = np.array([])
		E = self.EReco * shift_E
		for s, sample in enumerate(self.Samples):
			cond = self.Sample==s
			dummy_w = array[cond]
			Obs, __ = np.histogram1d(E[cond], bins=self.EnergyBins[s], weights=dummy_w)
			v = np.append(v,Obs)
		v = v.reshape(-1)
		if len(self.FewEntries) > 0:
			v = v[self.FewEntries]
		return v

	def BinIt_2D(self, array, shift_E=1, bias_E=0): # 2D energy and cos(angle) binning
		v = np.array([])
		E = self.EReco * shift_E
		for s, sample in enumerate(self.Samples):
			cond = self.Sample==s
			dummy_w = array[cond]
			Obs, __, __ = np.histogram2d(E[cond], self.CosThetaReco[cond], bins=(self.EnergyBins[s], self.CTBins[s]), weights=dummy_w)
			v = np.append(v,Obs)
		v = v.reshape(-1)
		if len(self.FewEntries) > 0:
			v = v[self.FewEntries]
		return v

	def RemoveLowBins(self):
		self.FewEntries = self.ExpectedBinned>4
