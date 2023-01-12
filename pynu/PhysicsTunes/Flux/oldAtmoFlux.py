import numpy as np
from PhysicsTunes import Tune

####################
# Atmospheric flux #
####################

class AtmosphericFlux(Tune):
		
	def FluxNormalization(self, exp, x):
		return x
	def Diff_FluxNormalization(self, exp, x):
		return 1

	def FluxNormalization_Below1GeV(self, exp, x):
		nev = np.ones(exp.NumberOfEvents)
		nev[exp.ETrue<1] = x
		return nev
	def Diff_FluxNormalization_Below1GeV(self, exp, x):
		nev = np.zeros(exp.NumberOfEvents)
		nev[exp.ETrue<1] = 1
		return nev

	def FluxNormalization_Above1GeV(self, exp, x):
		nev = np.ones(exp.NumberOfEvents)
		nev[exp.ETrue>1] = x
		return nev
	def Diff_FluxNormalization_Above1GeV(self, exp, x):
		nev = np.zeros(exp.NumberOfEvents)
		nev[exp.ETrue>1] = 1
		return nev

	def FluxTilt(self, exp, x):
		E0Gam = 10 # GeV
		nev = (exp.ETrue / E0Gam)**x
		return nev
	def Diff_FluxTilt(self, exp, x):
		E0Gam = 10 # GeV
		nev = (exp.ETrue / E0Gam)**x * np.log(exp.ETrue / E0Gam)
		return nev

	def NuNuBarRatio(self, exp, x):
		nnbar = np.ones(exp.NumberOfEvents)
		nnbar[exp.nuPDG<0] = x
		return nnbar
	def Diff_NuNuBarRatio(self, exp, x):
		nnbar = np.zeros(exp.NumberOfEvents)
		nnbar[exp.nuPDG<0] = 1
		return nnbar

	def FlavorRatio(self, exp, x):
		eovermu = np.ones(exp.NumberOfEvents)
		eovermu[np.abs(exp.nuPDG)==12] = x
		return eovermu
	def Diff_FlavorRatio(self, exp, x):
		eovermu = np.zeros(exp.NumberOfEvents)
		eovermu[abs(exp.nuPDG)==12] = 1
		return eovermu

	def ZenithFluxUp(self, exp, x):
		zenith = np.ones(exp.NumberOfEvents) 
		zenith[exp.CosZTrue>=0] = zenith[exp.CosZTrue>=0] - x * np.tanh(exp.CosZTrue[exp.CosZTrue>=0])**2
		return zenith
	def Diff_ZenithFluxUp(self, exp, x):
		zenith = np.zeros(exp.NumberOfEvents) 
		zenith[exp.CosZTrue>=0] = - np.tanh(exp.CosZTrue[exp.CosZTrue>=0])**2
		return zenith

	def ZenithFluxDown(self, exp, x):
		zenith = np.ones(exp.NumberOfEvents) 
		zenith[exp.CosZTrue<0] = zenith[exp.CosZTrue<0] - x * np.tanh(exp.CosZTrue[exp.CosZTrue<0])**2
		return zenith
	def Diff_ZenithFluxDown(self, exp, x):
		zenith = np.zeros(exp.NumberOfEvents) 
		zenith[exp.CosZTrue<0] = - np.tanh(exp.CosZTrue[exp.CosZTrue<0])**2
		return zenith