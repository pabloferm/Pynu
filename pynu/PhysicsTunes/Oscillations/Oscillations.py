import sys
sys.path.append('../')
from PhysicsTunes import Tune
import nuSQuIDS as nsq
import numpy as np


# General oscillator

class Oscillator(Tune):
	def __init__(self, scenario, neutrino_flavors):
		super().__init__()
		self.Scenario = scenario
		self.NeutrinoFlavors = neutrino_flavors
		self.units = nsq.Const()
		self.interactions = False
		self.rel_error = 1e-4
		self.abs_error = 1e-4
		self.E_nodes = 100
		self.eps = 1e-2

		self.Osc = None

	def Oscillator(self):
		sys.exit('Oscillator not defined.')

	def Sin2Theta13(self, experiment, x):
		self.Osc.Set_MixingAngle(0,2,asin(sqrt(x)))
		self.Parameters['Sin2Theta13'] = x
		return self.Oscillator()
	def Diff_Sin2Theta13(self, experiment, x): # Numerical derivation
		h0 = x*(1+self.eps)
		h1 = x*(1-self.eps)
		w0 = self.Sin2Theta13(h0)
		w1 = self.Sin2Theta13(h1)
		dw = ((w0 - w1) / (h0 - h1)) / self.experiment.weightOscBF_binned
		return dw

	def Sin2Theta12(self, experiment, x):
		self.Osc.Set_MixingAngle(0,1,asin(sqrt(x)))
		self.Parameters['Sin2Theta12'] = x
		w = self.Oscillator()
		return self.experiment.Exp_wBinIt(w) / self.experiment.weightOscBF_binned - 1
	def Diff_Sin2Theta12(self, experiment, x): # Numerical derivation
		h0 = x*(1+self.eps)
		h1 = x*(1-self.eps)
		w0 = self.Sin2Theta12(h0)
		w1 = self.Sin2Theta12(h1)
		dw = ((w0 - w1) / (h0 - h1)) / self.experiment.weightOscBF_binned
		return dw

	def Sin2Theta23(self, experiment, x):
		self.Osc.Set_MixingAngle(1,2,asin(sqrt(x)))
		self.Parameters['Sin2Theta23'] = x
		w = self.Oscillator()
		return self.experiment.Exp_wBinIt(w) / self.experiment.weightOscBF_binned - 1
	def Diff_Sin2Theta23(self, experiment, x): # Numerical derivation
		h0 = x*(1+self.eps)
		h1 = x*(1-self.eps)
		w0 = self.Sin2Theta23(h0)
		w1 = self.Sin2Theta23(h1)
		dw = ((w0 - w1) / (h0 - h1)) / self.experiment.weightOscBF_binned
		return dw

	def dCP(self, experiment, x):
		self.Osc.Set_CPPhase(0,2,x)
		self.Parameters['dCP'] = x
		w = self.Oscillator()
		return self.experiment.Exp_wBinIt(w) / self.experiment.weightOscBF_binned - 1
	def Diff_dCP(self, experiment, x): # Numerical derivation
		h0 = x*(1+self.eps)
		h1 = x*(1-self.eps)
		w0 = self.dCP(h0)
		w1 = self.dCP(h1)
		dw = ((w0 - w1) / (h0 - h1)) / self.experiment.weightOscBF_binned
		return dw

	def Dm221(self, experiment, x):
		self.Osc.Set_SquareMassDifference(1,x)
		self.Parameters['Dm221'] = x
		w = self.Oscillator()
		return self.experiment.Exp_wBinIt(w) / self.experiment.weightOscBF_binned - 1
	def Diff_Dm221(self, experiment, x): # Numerical derivation
		h0 = x*(1+self.eps)
		h1 = x*(1-self.eps)
		w0 = self.Dm221(h0)
		w1 = self.Dm221(h1)
		dw = ((w0 - w1) / (h0 - h1)) / self.experiment.weightOscBF_binned
		return dw

	def Dm231(self, experiment, x):
		self.Osc.Set_SquareMassDifference(2,x)
		self.Parameters['Dm231'] = x
		w = self.Oscillator()
		return self.experiment.Exp_wBinIt(w) / self.experiment.weightOscBF_binned - 1
	def Diff_Dm231(self, experiment, x): # Numerical derivation
		h0 = x*(1+self.eps)
		h1 = x*(1-self.eps)
		w0 = self.Dm231(h0)
		w1 = self.Dm231(h1)
		dw = ((w0 - w1) / (h0 - h1)) / self.experiment.weightOscBF_binned
		return dw

	def Parameters(self, **kwpars):
		if self.NeutrinoFlavors <= 3:
			parameters = {'Sin2Theta12':0, 'Sin2Theta13':0, 'Sin2Theta23':0, 'Dm221':0, 'Dm231':0, 'dCP':0, 'Ordering':'normal'}
		else:
			# At least...
			parameters = {'Sin2Theta12':0, 'Sin2Theta13':0, 'Sin2Theta23':0, 'Dm221':0, 'Dm231':0, 'dCP':0, 'Ordering':'normal', 'Sin2Theta14':0, 'Sin2Theta24':0, 'Sin2Theta34':0, 'Dm241':0, 'dCP2':0}
		for par, value in kwpars.items():
			parameters[par] = value
		return parameters

	def UpdateParameters(self, **kwpars):
		for par, value in kwpars.items():
			self.Parameters[par] = value
		self.SetParameters(**self.Parameters)

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
			if 'inverted' in parameters['Ordering'] and s_dm == 'Dm231':
				dm = parameters['Dm221'] - parameters['Dm231']
			self.Osc.Set_SquareMassDifference(i,dm)
		if 'dCP' in parameters:
			self.Osc.Set_CPPhase(0,2,parameters['dCP'])
		if self.neutrino_flavors > 3 and 'dCP2' in parameters:
			self.Osc.Set_CPPhase(0,3,parameters['dCP2'])

		self.Parameters = parameters

	def NSQNeutrinoType(self, experiment):
		neutype = np.zeros(experiment.NumberOfEvents)
		neutype[experiment.nuPDG<0] = 1
		return neutype

	def NSQNeutrinoFlavor(self, experiment):
		neuflavor = 0.5*np.abs(experiment.nuPDG)-6
		neuflavor = neuflavor.astype(np.uint32).tolist()
		return neuflavor
