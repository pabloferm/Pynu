import numpy as np


def StatsOnly(Observation_dict, Expectation_dict):
	''' Compute statistics only binned chi-squared '''
	X2 = 0
	for O, E in zip(Observation_dict.values(), Expectation_dict.values()):
		X2 += 2 * np.sum(E-O+O*np.log(O/E))
	return X2


def AnalyticPriorsBounds(Observation_dict, Expectation_dict, DiffExpectation_dict, NominalNuisance_list, SigmaNuisance_list):
	''' First order analytic computation of values for parameters to be mariginalized '''
	number_of_nuisance = len(NominalNuisance_list)
	A = np.zeros(number_of_nuisance)
	B = np.zeros(number_of_nuisance)
	mu = np.array(NominalNuisance_list)
	sig = np.array(SigmaNuisance_list)

	# Experiments
	for i, dE in enumerate(DiffExpectation_dict.values()):
		for O, E, dEdx in zip(Observation_dict.values(), Expectation_dict.values(), dE.values()):
			A[i] += np.sum((O/E -1)*dEdx)
			B[i] += np.sum(O/E**2 * dEdx**2)

	priors = mu + 0.5 * A / (B + 1/sig**2)

	delta = np.minimum(2*np.abs(priors-mu), sig)
	delta[delta==0] = sig[delta==0]

	bounds = np.c_[priors - delta, priors + delta]
	bounds = tuple(map(tuple, bounds))

	return priors, bounds


# def SystsCombined(syst, analysis, Obs, experiments):
# 	''' Compute chi-squared value with systematics '''
# 	JX2 = [0] * len(syst)
# 	X2 = 0
# 	# Experiments
# 	for exp in experiments.values():
# 		# Binned tatistics
# 		E = exp.weightOscBF_binned
# 		O = Obs[exp.Experiment]
# 		#Systematics
# 		usedSysts = []
# 		dEdx = [0] * len(syst)
# 		wSys = 0
# 		dummywSys = 0
# 		thisSyst = analysis.Systematics[exp.Experiment] + analysis.Systematics[exp.Source] + analysis.Systematics[exp.Detector]
# 		for sys in thisSyst:
# 			index = np.where(analysis.SystematicsList==sys)[0]
# 			j = index[0]
# 			xFij = globals()[sys](syst[j],exp)
# 			wSys += xFij
# 			dEdx[j] = E * globals()['Diff_'+sys](syst[j],exp)
# 			usedSysts.append(j)
# 		Es = E * (1 + wSys)
# 		# Compute Chi^2
# 		if np.any(Es<=0):
# 			X2 = 1e6
# 		else:
# 			X2 += 2 * np.sum(Es-O+O*np.log(O/Es))
# 		# Compute Jacobian of Chi^2
# 		for i in usedSysts:
# 			JX2[i] += 2 * np.sum((1-O/Es)*dEdx[i])
# 	# Systematic's penalty terms
# 	for i,(x,mu,sig) in enumerate(zip(syst,analysis.SystNominalList,analysis.SystSigmaList)):
# 		X2 += ((x-mu) / sig)**2
# 		JX2[i] += 2 * (x-mu) / sig**2

# 	return (X2,JX2)