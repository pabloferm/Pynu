import numpy as np

####################
# Atmospheric flux #
####################

def Diff_FluxNormalization(x, experiment):
	return 1

def Diff_FluxNormalization_Below1GeV(x, experiment):
	nev = np.zeros(experiment.NumberOfEvents)
	nev[experiment.ETrue<1] = 1
	return experiment.Exp_wBinIt(nev) / experiment.weightOscBF_binned

def Diff_FluxNormalization_Above1GeV(x, experiment):
	nev = np.zeros(experiment.NumberOfEvents)
	nev[experiment.ETrue>1] = 1
	return experiment.Exp_wBinIt(nev) / experiment.weightOscBF_binned

def Diff_FluxTilt(x, experiment):
	E0Gam = 10 # GeV
	nev = (experiment.ETrue / E0Gam)**x * np.log(experiment.ETrue / E0Gam)
	return experiment.Exp_wBinIt(nev) / experiment.weightOscBF_binned

def Diff_NuNuBarRatio(x, experiment):
	nnbar = np.zeros(experiment.NumberOfEvents)
	nnbar[experiment.nuPDG<0] = 1
	return experiment.Exp_wBinIt(nnbar) / experiment.weightOscBF_binned

def Diff_FlavorRatio(x,experiment):
	eovermu = np.zeros(experiment.NumberOfEvents)
	eovermu[abs(experiment.nuPDG)==12] = 1
	return experiment.Exp_wBinIt(eovermu) / experiment.weightOscBF_binned

def Diff_ZenithFluxUp(x, experiment):
	zenith = np.zeros(experiment.NumberOfEvents) 
	zenith[experiment.CosZTrue>=0] = - np.tanh(experiment.CosZTrue[experiment.CosZTrue>=0])**2
	return experiment.Exp_wBinIt(zenith) / experiment.weightOscBF_binned

def Diff_ZenithFluxDown(x, experiment):
	zenith = np.zeros(experiment.NumberOfEvents) 
	zenith[experiment.CosZTrue<0] = - np.tanh(experiment.CosZTrue[experiment.CosZTrue<0])**2
	return experiment.Exp_wBinIt(zenith) / experiment.weightOscBF_binned