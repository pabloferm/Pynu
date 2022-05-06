import numpy as np

####################
#  Cross-section   #
####################

def XSecNuTau(x, experiment):
	if x == 1: return 0 
	tau = np.ones(experiment.NumberOfEvents)
	tau[np.abs(experiment.nuPDG)==16] = x
	return experiment.Exp_wBinIt(tau) / experiment.weightOscBF_binned - 1

def NCoverCC(x, experiment):
	if x == 1: return 0 
	nc = np.ones(experiment.NumberOfEvents)
	nc[experiment.CC==0] = x 
	return experiment.Exp_wBinIt(nc) / experiment.weightOscBF_binned - 1

def AxialMass(x, experiment):
	if x == 1: return 0 
	cc = np.ones(experiment.NumberOfEvents)
	cc[experiment.CC==1] = 1+0.042*(x-1)*1.05*np.log10(experiment.ETrue[experiment.CC==1]) 
	return experiment.Exp_wBinIt(cc) / experiment.weightOscBF_binned - 1

def NCHad(x, experiment):
	if x == 1: return 0 
	nc = np.ones(experiment.NumberOfEvents)
	nc[experiment.CC==0] = x 
	return experiment.Exp_wBinIt(nc) / experiment.weightOscBF_binned - 1

def DIS(x,experiment):
	if x == 1: return 0 
	if experiment.Experiment == 'IceCube-Upgrade' or experiment.Experiment == 'IC' or experiment.Experiment == 'DeepCore': return 0
	dis = np.ones(experiment.NumberOfEvents)
	cond = np.abs(experiment.Mode)>25 * experiment.CC
	dis[cond] = x
	return experiment.Exp_wBinIt(dis) / experiment.weightOscBF_binned - 1

def CCQE(x,experiment):
	if x == 1: return 0 
	if experiment.Experiment == 'IceCube-Upgrade' or experiment.Experiment == 'IC' or experiment.Experiment == 'DeepCore': return 0
	ccqe = np.ones(experiment.NumberOfEvents)
	ccqe[np.abs(experiment.Mode)==1] = x
	return experiment.Exp_wBinIt(ccqe) / experiment.weightOscBF_binned - 1

def CCQENuBarNu(x,experiment):
	if x == 1: return 0 
	if experiment.Experiment == 'IceCube-Upgrade' or experiment.Experiment == 'IC' or experiment.Experiment == 'DeepCore': return 0
	ccqe = np.ones(experiment.NumberOfEvents)
	ccqe[experiment.Mode==-1] = x
	return experiment.Exp_wBinIt(ccqe) / experiment.weightOscBF_binned - 1

def CCQEMuE(x,experiment):
	if x == 1: return 0 
	if experiment.Experiment == 'IceCube-Upgrade' or experiment.Experiment == 'IC' or experiment.Experiment == 'DeepCore': return 0
	ccqe = np.ones(experiment.NumberOfEvents)
	cond = (np.abs(experiment.Mode)==1) * (np.abs(experiment.nuPDG)==14)
	ccqe[cond] = x
	return experiment.Exp_wBinIt(ccqe) / experiment.weightOscBF_binned - 1

def CC1Pi_Pi0Pi(x,experiment):
	if x == 1: return 0 
	if experiment.Experiment == 'IceCube-Upgrade' or experiment.Experiment == 'IC' or experiment.Experiment == 'DeepCore': return 0
	ccpi = np.ones(experiment.NumberOfEvents)
	ccpi[np.abs(experiment.Mode)==12] = x
	return experiment.Exp_wBinIt(ccpi) / experiment.weightOscBF_binned - 1

def CC1Pi_NuBarNuE(x,experiment):
	if x == 1: return 0 
	if experiment.Experiment == 'IceCube-Upgrade' or experiment.Experiment == 'IC' or experiment.Experiment == 'DeepCore': return 0
	ccpi = np.ones(experiment.NumberOfEvents)
	cond = (np.abs(experiment.Mode)>10) * (np.abs(experiment.Mode)<17) * (experiment.nuPDG==-12)
	ccpi[cond] = x
	return experiment.Exp_wBinIt(ccpi) / experiment.weightOscBF_binned - 1

def CC1Pi_NuBarNuMu(x,experiment):
	if x == 1: return 0 
	if experiment.Experiment == 'IceCube-Upgrade' or experiment.Experiment == 'IC' or experiment.Experiment == 'DeepCore': return 0
	ccpi = np.ones(experiment.NumberOfEvents)
	cond = (np.abs(experiment.Mode)>10) * (np.abs(experiment.Mode)<17) * (experiment.nuPDG==-14)
	ccpi[cond] = x
	return experiment.Exp_wBinIt(ccpi) / experiment.weightOscBF_binned - 1

def CC1PiProduction(x,experiment):
	if x == 1: return 0 
	if experiment.Experiment == 'IceCube-Upgrade' or experiment.Experiment == 'IC' or experiment.Experiment == 'DeepCore': return 0
	ccpi = np.ones(experiment.NumberOfEvents)
	cond = (np.abs(experiment.Mode)>10) * (np.abs(experiment.Mode)<17)
	ccpi[cond] = x
	return experiment.Exp_wBinIt(ccpi) / experiment.weightOscBF_binned - 1

def CohPiProduction(x,experiment):
	if x == 1: return 0 
	if experiment.Experiment == 'IceCube-Upgrade' or experiment.Experiment == 'IC' or experiment.Experiment == 'DeepCore': return 0
	ccpi = np.ones(experiment.NumberOfEvents)
	ccpi[np.abs(experiment.Mode)==16] = x
	return experiment.Exp_wBinIt(ccpi) / experiment.weightOscBF_binned - 1
