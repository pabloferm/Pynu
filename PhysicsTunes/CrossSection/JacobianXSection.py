import numpy as np

####################
#  Cross-section   #
####################

def Diff_XSecNuTau(x, experiment):
	tau = np.zeros(experiment.NumberOfEvents)
	tau[np.abs(experiment.nuPDG)==16] = 1
	return experiment.Exp_wBinIt(tau) / experiment.weightOscBF_binned

def Diff_NCoverCC(x, experiment):
	nc = np.zeros(experiment.NumberOfEvents)
	nc[experiment.CC==0] = 1
	return experiment.Exp_wBinIt(nc) / experiment.weightOscBF_binned

def Diff_AxialMass(x, experiment):
	cc = np.zeros(experiment.NumberOfEvents)
	cc[experiment.CC==1] = 0.042*1.05*np.log10(experiment.ETrue[experiment.CC==1]) 
	return experiment.Exp_wBinIt(cc) / experiment.weightOscBF_binned

def Diff_NCHad(x, experiment):
	nc = np.zeros(experiment.NumberOfEvents)
	nc[experiment.CC==0] = 1 
	return experiment.Exp_wBinIt(nc) / experiment.weightOscBF_binned

def Diff_DIS(x,experiment):
	if experiment.Experiment == 'IceCube-Upgrade' or experiment.Experiment == 'IC' or experiment.Experiment == 'DeepCore': return 0
	dis = np.zeros(experiment.NumberOfEvents)
	cond = np.abs(experiment.Mode)>25 * experiment.CC
	dis[cond] = 1
	return experiment.Exp_wBinIt(dis) / experiment.weightOscBF_binned

def Diff_CCQE(x,experiment):
	if experiment.Experiment == 'IceCube-Upgrade' or experiment.Experiment == 'IC' or experiment.Experiment == 'DeepCore': return 0
	ccqe = np.zeros(experiment.NumberOfEvents)
	ccqe[np.abs(experiment.Mode)==1] = 1
	return experiment.Exp_wBinIt(ccqe) / experiment.weightOscBF_binned

def Diff_CCQENuBarNu(x,experiment):
	if experiment.Experiment == 'IceCube-Upgrade' or experiment.Experiment == 'IC' or experiment.Experiment == 'DeepCore': return 0
	ccqe = np.zeros(experiment.NumberOfEvents)
	ccqe[experiment.Mode==-1] = 1
	return experiment.Exp_wBinIt(ccqe) / experiment.weightOscBF_binned

def Diff_CCQEMuE(x,experiment):
	if experiment.Experiment == 'IceCube-Upgrade' or experiment.Experiment == 'IC' or experiment.Experiment == 'DeepCore': return 0
	ccqe = np.zeros(experiment.NumberOfEvents)
	cond = (np.abs(experiment.Mode)==1) * (np.abs(experiment.nuPDG)==14)
	ccqe[cond] = 1
	return experiment.Exp_wBinIt(ccqe) / experiment.weightOscBF_binned

def Diff_CC1Pi_Pi0Pi(x,experiment):
	if experiment.Experiment == 'IceCube-Upgrade' or experiment.Experiment == 'IC' or experiment.Experiment == 'DeepCore': return 0
	ccpi = np.zeros(experiment.NumberOfEvents)
	ccpi[np.abs(experiment.Mode)==12] = 1
	return experiment.Exp_wBinIt(ccpi) / experiment.weightOscBF_binned

def Diff_CC1Pi_NuBarNuE(x,experiment):
	if experiment.Experiment == 'IceCube-Upgrade' or experiment.Experiment == 'IC' or experiment.Experiment == 'DeepCore': return 0
	ccpi = np.zeros(experiment.NumberOfEvents)
	cond = (np.abs(experiment.Mode)>10) * (np.abs(experiment.Mode)<17) * (experiment.nuPDG==-12)
	ccpi[cond] = 1
	return experiment.Exp_wBinIt(ccpi) / experiment.weightOscBF_binned

def Diff_CC1Pi_NuBarNuMu(x,experiment):
	if experiment.Experiment == 'IceCube-Upgrade' or experiment.Experiment == 'IC' or experiment.Experiment == 'DeepCore': return 0
	ccpi = np.zeros(experiment.NumberOfEvents)
	cond = (np.abs(experiment.Mode)>10) * (np.abs(experiment.Mode)<17) * (experiment.nuPDG==-14)
	ccpi[cond] = 1
	return experiment.Exp_wBinIt(ccpi) / experiment.weightOscBF_binned

def Diff_CC1PiProduction(x,experiment):
	if experiment.Experiment == 'IceCube-Upgrade' or experiment.Experiment == 'IC' or experiment.Experiment == 'DeepCore': return 0
	ccpi = np.zeros(experiment.NumberOfEvents)
	cond = (np.abs(experiment.Mode)>10) * (np.abs(experiment.Mode)<17)
	ccpi[cond] = 1
	return experiment.Exp_wBinIt(ccpi) / experiment.weightOscBF_binned

def Diff_CohPiProduction(x,experiment):
	if experiment.Experiment == 'IceCube-Upgrade' or experiment.Experiment == 'IC' or experiment.Experiment == 'DeepCore': return 0
	ccpi = np.zeros(experiment.NumberOfEvents)
	ccpi[np.abs(experiment.Mode)==16] = 1
	return experiment.Exp_wBinIt(ccpi) / experiment.weightOscBF_binned

