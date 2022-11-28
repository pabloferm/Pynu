import numpy as np


####################
# Atmospheric flux #
####################

class AtmosphericFlux:
	def __init__(self, experiment):
		self.experiment = experiment
		pass
		
	def FluxNormalization(self, x):
		return x - 1
	def Diff_FluxNormalization(self, x):
		return 1

	def FluxNormalization_Below1GeV(self, x):
		nev = np.ones(self.experiment.NumberOfEvents)
		nev[self.experiment.ETrue<1] = x
		return self.experiment.Exp_wBinIt(nev) / self.experiment.weightOscBF_binned - 1
	def Diff_FluxNormalization_Below1GeV(self, x):
		nev = np.zeros(self.experiment.NumberOfEvents)
		nev[self.experiment.ETrue<1] = 1
		return self.experiment.Exp_wBinIt(nev) / self.experiment.weightOscBF_binned

	def FluxNormalization_Above1GeV(self, x):
		nev = np.ones(self.experiment.NumberOfEvents)
		nev[self.experiment.ETrue>1] = x
		return self.experiment.Exp_wBinIt(nev) / self.experiment.weightOscBF_binned - 1
	def Diff_FluxNormalization_Above1GeV(self, x):
		nev = np.zeros(self.experiment.NumberOfEvents)
		nev[self.experiment.ETrue>1] = 1
		return self.experiment.Exp_wBinIt(nev) / self.experiment.weightOscBF_binned

	def FluxTilt(self, x):
		E0Gam = 10 # GeV
		nev = (self.experiment.ETrue / E0Gam)**x
		return self.experiment.Exp_wBinIt(nev) / self.experiment.weightOscBF_binned - 1
	def Diff_FluxTilt(self, x):
		E0Gam = 10 # GeV
		nev = (self.experiment.ETrue / E0Gam)**x * np.log(self.experiment.ETrue / E0Gam)
		return self.experiment.Exp_wBinIt(nev) / self.experiment.weightOscBF_binned

	def NuNuBarRatio(self, x):
		nnbar = np.ones(self.experiment.NumberOfEvents)
		nnbar[self.experiment.nuPDG<0] = x
		return self.experiment.Exp_wBinIt(nnbar) / self.experiment.weightOscBF_binned - 1
	def Diff_NuNuBarRatio(self, x):
		nnbar = np.zeros(self.experiment.NumberOfEvents)
		nnbar[self.experiment.nuPDG<0] = 1
		return self.experiment.Exp_wBinIt(nnbar) / self.experiment.weightOscBF_binned

	def FlavorRatio(self, x):
		eovermu = np.ones(self.experiment.NumberOfEvents)
		eovermu[np.abs(self.experiment.nuPDG)==12] = x
		return self.experiment.Exp_wBinIt(eovermu) / self.experiment.weightOscBF_binned - 1
	def Diff_FlavorRatio(self, x):
		eovermu = np.zeros(self.experiment.NumberOfEvents)
		eovermu[abs(self.experiment.nuPDG)==12] = 1
		return self.experiment.Exp_wBinIt(eovermu) / self.experiment.weightOscBF_binned

	def ZenithFluxUp(self, x):
		zenith = np.ones(self.experiment.NumberOfEvents) 
		zenith[self.experiment.CosZTrue>=0] = zenith[self.experiment.CosZTrue>=0] - x * np.tanh(self.experiment.CosZTrue[self.experiment.CosZTrue>=0])**2
		return self.experiment.Exp_wBinIt(zenith) / self.experiment.weightOscBF_binned - 1
	def Diff_ZenithFluxUp(self, x):
		zenith = np.zeros(self.experiment.NumberOfEvents) 
		zenith[self.experiment.CosZTrue>=0] = - np.tanh(self.experiment.CosZTrue[self.experiment.CosZTrue>=0])**2
		return self.experiment.Exp_wBinIt(zenith) / self.experiment.weightOscBF_binned

	def ZenithFluxDown(self, x):
		zenith = np.ones(self.experiment.NumberOfEvents) 
		zenith[self.experiment.CosZTrue<0] = zenith[self.experiment.CosZTrue<0] - x * np.tanh(self.experiment.CosZTrue[self.experiment.CosZTrue<0])**2
		return self.experiment.Exp_wBinIt(zenith) / self.experiment.weightOscBF_binned - 1
	def Diff_ZenithFluxDown(self, x):
		zenith = np.zeros(self.experiment.NumberOfEvents) 
		zenith[self.experiment.CosZTrue<0] = - np.tanh(self.experiment.CosZTrue[self.experiment.CosZTrue<0])**2
		return self.experiment.Exp_wBinIt(zenith) / self.experiment.weightOscBF_binned