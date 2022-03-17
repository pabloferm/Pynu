import numpy as np

####################
# Atmospheric flux #
####################

def FluxNormalization(x, experiment):
	return x - 1

def FluxNormalization_Below1GeV(x, experiment):
	nev = np.ones(experiment.NumberOfEvents)
	nev[experiment.ETrue<1] = x
	return experiment.Exp_wBinIt(nev) / experiment.weightOscBF_binned - 1

def FluxNormalization_Above1GeV(x, experiment):
	nev = np.ones(experiment.NumberOfEvents)
	nev[experiment.ETrue>1] = x
	return experiment.Exp_wBinIt(nev) / experiment.weightOscBF_binned - 1

def FluxTilt(x, experiment):
	E0Gam = 10 # GeV
	nev = (experiment.ETrue / E0Gam)**x
	return experiment.Exp_wBinIt(nev) / experiment.weightOscBF_binned - 1

def NuNuBarRatio(x, experiment):
	nnbar = np.ones(experiment.NumberOfEvents)
	nnbar[experiment.nuPDG<0] = x
	return experiment.Exp_wBinIt(nnbar) / experiment.weightOscBF_binned - 1

def FlavorRatio(x,experiment):
	eovermu = np.ones(experiment.NumberOfEvents)
	eovermu[np.abs(experiment.nuPDG)==12] = x
	return experiment.Exp_wBinIt(eovermu) / experiment.weightOscBF_binned - 1

def ZenithFluxUp(x, experiment):
	zenith = np.ones(experiment.NumberOfEvents) 
	zenith[experiment.CosZTrue>=0] = zenith[experiment.CosZTrue>=0] - x * np.tanh(experiment.CosZTrue[experiment.CosZTrue>=0])**2
	return experiment.Exp_wBinIt(zenith) / experiment.weightOscBF_binned - 1

def ZenithFluxDown(x, experiment):
	zenith = np.ones(experiment.NumberOfEvents) 
	zenith[experiment.CosZTrue<0] = zenith[experiment.CosZTrue<0] - x * np.tanh(experiment.CosZTrue[experiment.CosZTrue<0])**2
	return experiment.Exp_wBinIt(zenith) / experiment.weightOscBF_binned - 1
