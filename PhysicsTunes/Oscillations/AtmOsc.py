import numpy as np
import nuSQuIDS as nsq
from math import asin, sqrt, pi
from .Oscillations import Parameters

####################
# Atmospheric flux #
####################

class AtmosphericOscillations:
	def __init__(self, scenario, neutrino_flavors, exp):
		self.experiment = exp
		self.neutrino_flavors = neutrino_flavors
		self.units = nsq.Const()
		interactions = False
		rel_error = 1e-4
		abs_error = 1e-4

		E_nodes = 100
		energy_nodes = nsq.logspace(exp.E_edges[0],exp.E_edges[1],E_nodes)
		Z_nodes = 40
		cth_nodes = nsq.linspace(exp.Z_edges[0],exp.Z_edges[1],Z_nodes)
		self.Osc = nsq.nuSQUIDSAtm(cth_nodes,energy_nodes*self.units.GeV,self.neutrino_flavors,nsq.NeutrinoType.both,interactions)
		self.Osc.Set_rel_error(rel_error)
		self.Osc.Set_abs_error(abs_error)
		self.neutype = np.zeros(exp.NumberOfEvents)
		self.neutype[exp.nuPDG<0] = 1
		self.neutype = self.neutype.astype(np.uint32).tolist()
		self.neuflavor = 0.5*np.abs(exp.nuPDG)-6
		self.neuflavor = self.neuflavor.astype(np.uint32).tolist()
		
	def SetParameters(self, **kwpars):
		parameters = Parameters(self.neutrino_flavors,**kwpars)
		for i in range(1,self.neutrino_flavors):
			for j in range(i):
				s_theta = 't'+str(j+1)+str(i+1)
				if s_theta in parameters:
					theta = parameters[s_theta]
					self.Osc.Set_MixingAngle(j,i,asin(sqrt(theta)))
			s_dm = 'Dm2'+str(i+1)+'1'
		if s_dm in parameters:
			dm = parameters[s_dm]
			if Ordering != 'normal' and s_dm == 'Dm231':
				dm = parameters['Dm221'] - parameters['Dm231']
			self.Osc.Set_SquareMassDifference(i,dm)
		if 'dCP' in parameters:
			self.Osc.Set_CPPhase(0,2,parameters['dCP'])
		if self.neutrino_flavors > 3 and 'dCP2' in parameters:
			self.Osc.Set_CPPhase(0,3,parameters['dCP2'])

		self.parameters = parameters

	def Oscillator(self):
		self.Osc.Set_initial_state(self.experiment.InitialFlux,nsq.Basis.flavor)
		self.Osc.EvolveState()
		w = list(map(self.Osc.EvalFlavor, self.neuflavor, self.experiment.CosZTrue, self.experiment.ETrue*self.units.GeV, self.neutype, repeat(True)))
		return np.array(w)

	def Sin2Theta13(self, x):
		self.Osc.Set_MixingAngle(0,2,asin(sqrt(x)))
		self.parameters['Sin2Theta13'] = x
		w = self.Oscillator()
		return self.experiment.Exp_wBinIt(w) / self.experiment.weightOscBF_binned - 1
	def Diff_Sin2Theta13(self, x): # Numerical derivation
		h0 = x+1e-2
		h1 = x-1e-2
		w0 = self.Sin2Theta13(h0)
		w1 = self.Sin2Theta13(h1)
		dw = ((w0 - w1) / (h0 - h1)) / experiment.weightOscBF_binned
		return dw

	def Sin2Theta12(self, x):
		self.Osc.Set_MixingAngle(0,1,asin(sqrt(x)))
		self.parameters['Sin2Theta12'] = x
		w = self.Oscillator()
		return self.experiment.Exp_wBinIt(w) / self.experiment.weightOscBF_binned - 1
	def Diff_Sin2Theta12(self, x): # Numerical derivation
		h0 = x+1e-2
		h1 = x-1e-2
		w0 = self.Sin2Theta12(h0)
		w1 = self.Sin2Theta12(h1)
		dw = ((w0 - w1) / (h0 - h1)) / experiment.weightOscBF_binned
		return dw

	def Sin2Theta23(self, x):
		self.Osc.Set_MixingAngle(1,2,asin(sqrt(x)))
		self.parameters['Sin2Theta23'] = x
		w = self.Oscillator()
		return self.experiment.Exp_wBinIt(w) / self.experiment.weightOscBF_binned - 1
	def Diff_Sin2Theta23(self, x): # Numerical derivation
		h0 = x+1e-2
		h1 = x-1e-2
		w0 = self.Sin2Theta23(h0)
		w1 = self.Sin2Theta23(h1)
		dw = ((w0 - w1) / (h0 - h1)) / experiment.weightOscBF_binned
		return dw

	def dCP(self, x):
		self.Osc.Set_CPPhase(0,2,x)
		self.parameters['dCP'] = x
		w = self.Oscillator()
		return self.experiment.Exp_wBinIt(w) / self.experiment.weightOscBF_binned - 1
	def Diff_dCP(self, x): # Numerical derivation
		h0 = x+1e-2
		h1 = x-1e-2
		w0 = self.dCP(h0)
		w1 = self.dCP(h1)
		dw = ((w0 - w1) / (h0 - h1)) / experiment.weightOscBF_binned
		return dw

	def Dm221(self, x):
		self.Osc.Set_SquareMassDifference(1,x)
		self.parameters['Dm221'] = x
		w = self.Oscillator()
		return self.experiment.Exp_wBinIt(w) / self.experiment.weightOscBF_binned - 1
	def Diff_Dm221(self, x): # Numerical derivation
		h0 = x+1e-2
		h1 = x-1e-2
		w0 = self.Dm221(h0)
		w1 = self.Dm221(h1)
		dw = ((w0 - w1) / (h0 - h1)) / experiment.weightOscBF_binned
		return dw

	def Dm231(self, x):
		self.Osc.Set_SquareMassDifference(2,x)
		self.parameters['Dm231'] = x
		w = self.Oscillator()
		return self.experiment.Exp_wBinIt(w) / self.experiment.weightOscBF_binned - 1
	def Diff_Dm231(self, x): # Numerical derivation
		h0 = x+1e-2
		h1 = x-1e-2
		w0 = self.Dm231(h0)
		w1 = self.Dm231(h1)
		dw = ((w0 - w1) / (h0 - h1)) / experiment.weightOscBF_binned
		return dw