# Class for the atmospheric neutrinos in IceCube-Upgrade

import numpy as np
import pandas as pd
import h5py
from .Experiment import Experiment

class ICUp_Atm(Experiment):
	def __init__(self, dict_of_details):
		super(ICUp_Atm, self).__init__( dict_of_details )
		self.Experiment = 'IceCube-Upgrade'
		self.Source = 'Atmospheric'
		self.Target = 'Water'

		self.MCVariables()

		if self.DataFit: 
			self.DataVariables()
			self.BinData()

	def MCVariables(self):
		d_itype = self.MC['pid']
		condition = (d_itype<16) * (d_itype>-1)
		self.EReco = self.MC['reco_energy'][condition]
		self.CosZReco = self.MC['reco_zenith'][condition]
		self.CosZTrue = self.MC['true_zenith'][condition]
		self.AziTrue = self.MC['true_azimuth'][condition]
		self.CC = self.MC['current_type']
		self.nuPDG = self.MC['pdg'][condition]
		self.ETrue = self.MC['true_energy'][condition]
		self.Weight = self.MC['weight'][condition]
		self.Sample = self.MC['pid'][condition] # Sample of each event
		self.Mode = self.NEUTMode()[condition]

		self.NumberOfEvents = self.Sample.size
		self.Samples = np.unique(self.Sample) # Samples in the analysis
		self.Erec_min = 1
		self.E_edges = [1,1e3]
		self.Z_edges = [-1,1]

	def NEUTMode(self):
		noNEUTmode = self.MC['interaction_type']
		c_mode = np.logical_and(self.nuPDG>0, noNEUTmode==0)
		noNEUTmode[c_mode] = 31
		c_mode = np.logical_and(self.nuPDG>0, noNEUTmode==1)
		noNEUTmode[c_mode] = 1
		c_mode = np.logical_and(self.nuPDG>0, noNEUTmode==2)
		noNEUTmode[c_mode] = 11
		c_mode = np.logical_and(self.nuPDG>0, noNEUTmode==3)
		noNEUTmode[c_mode] = 26
		c_mode = np.logical_and(self.nuPDG>0, noNEUTmode==4)
		noNEUTmode[c_mode] = 16
		c_mode = np.logical_and(self.nuPDG<0, noNEUTmode==0)
		noNEUTmode[c_mode] = -31
		c_mode = np.logical_and(self.nuPDG<0, noNEUTmode==1)
		noNEUTmode[c_mode] = -1
		c_mode = np.logical_and(self.nuPDG<0, noNEUTmode==2)
		noNEUTmode[c_mode] = -11
		c_mode = np.logical_and(self.nuPDG<0, noNEUTmode==3)
		noNEUTmode[c_mode] = -26
		c_mode = np.logical_and(self.nuPDG<0, noNEUTmode==4)
		noNEUTmode[c_mode] = -16
		return noNEUTmode

	def DataVariables(self):
		d_itype = self.Data['pid']
		condition = (d_itype<16) * (d_itype>-1)
		self.dEReco = self.Data['reco_energy'][condition]
		self.dCosZReco = self.Data['reco_zenith'][condition]
		self.dSample = self.Data['pid'][condition] # Sample of each event
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

