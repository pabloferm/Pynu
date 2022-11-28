# Class for the atmospheric neutrinos in Super-Kamiokande

import numpy as np
import pandas as pd
import h5py
from .Experiment import Experiment

class SuperK_Atm(Experiment):
	self.Experiment = 'SuperK'
	self.Source = 'Atmospheric'
	self.Target = 'Water'

	self.MCVariables()

	if self.DataFit: 
		self.DataVariables()
		self.BinData()

	def MCVariables(self):
		d_itype = self.MC['itype']
		condition = (d_itype<16) * (d_itype>-1)
		self.EReco = self.MC['evis'][condition]
		self.CosZReco = self.MC['recodirZ'][condition]
		self.CosZTrue = self.MC['dirnuZ'][condition]
		self.AziTrue = self.MC['azi']d_azi[condition]
		self.Mode = self.MC['mode'][condition]
		self.CC = np.abs(self.Mode) < 30
		self.nuPDG = self.MC['ipnu'][condition]
		self.ETrue = self.MC['pnu'][condition]
		self.Weight = self.MC['weightReco'][condition] * self.MC['weightSim'][condition]
		self.Sample = self.MC['itype'][condition] # Sample of each event
		self.DecayE = self.MC['muedk'][condition]

		self.NumberOfEvents = self.Sample.size
		self.Samples = np.unique(self.Sample) # Samples in the analysis
		self.Erec_min = 0.1
		self.E_edges = [0.1,1e3]
		self.Z_edges = [-1,1]

	def DataVariables(self):
		d_itype = self.Data['itype']
		condition = (d_itype<16) * (d_itype>-1)
		self.dEReco = self.Data['evis'][condition]
		self.dCosZReco = self.Data['recodirZ'][condition]
		self.dSample = self.Data['itype'][condition] # Sample of each event
		self.dDecayE = self.Data['muedk'][condition]
		self.dNumberOfEvents = self.Sample.size

	def BinIt(self, array, shift_E=1, bias_E=0):
		self.CosThetaReco = self.CosZReco
		return self.BinIt_2D(array, shift_E=1, bias_E=0)

	def Expectation(self, weights): # fixed, nominal, true --> Compute elsewhere
		self.ExpectedBinned = self.wBinIt(weights)
		self.RemoveLowBins()

	def BinData(self): 
		self.ExpectedBinned = self.BinIt(1)
		self.DataBinned = self.ExpectedBinned

	def RemoveLowBins(self):
		self.FewEntries = self.ExpectedBinned>4
