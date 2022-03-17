import numpy as np
import math
from scipy.special import gamma
from Systematics import *
	
def Chi2StatsCombined(neutrino_flavors, t12, t13, t23, dm21, dm31, dcp, Ordering, experiments, outfile):
	X2 = 0
	for exp in experiments:
		wOsc = experiments[exp].Oscillator(neutrino_flavors, t12, t13, t23, dm21, dm31, dcp, Ordering)
		for s in range(experiments[exp].NumberOfSamples):
			wBF = experiments[exp].weightOscBF[experiments[exp].Sample==s] * experiments[exp].Weight[experiments[exp].Sample==s] * experiments[exp].Norm
			w = wOsc[experiments[exp].Sample==s] * experiments[exp].Weight[experiments[exp].Sample==s] * experiments[exp].Norm
			Exp, dx, dy = np.histogram2d(experiments[exp].EReco[experiments[exp].Sample==s], experiments[exp].CosZReco[experiments[exp].Sample==s], bins=(experiments[exp].EnergyBins[s], experiments[exp].CzBins[s]), weights=wBF)
			Obs, dx, dy = np.histogram2d(experiments[exp].EReco[experiments[exp].Sample==s], experiments[exp].CosZReco[experiments[exp].Sample==s], bins=(experiments[exp].EnergyBins[s], experiments[exp].CzBins[s]), weights=w)
			for O,E in zip(np.ravel(Obs),np.ravel(Exp)):
				if O!=0 and E!=0:
					X2 = X2 + 2 * (E - O + O * math.log(O/E))

	# Writing output
	with open(outfile,'a') as f:
		f.write(f'{t12} {t13} {t23} {dm21} {dm31} {dcp} {Ordering} {X2}\n')
		f.flush()


def Chi2SystsCombined(syst, analysis, Obs, experiments):
	JX2 = [0] * len(syst)
	X2 = 0

	# Experiments
	for exp in experiments.values():

		# Binned tatistics
		E = exp.weightOscBF_binned
		O = Obs[exp.Experiment]

		#Systematics
		usedSysts = []
		dEdx = [0] * len(syst)
		wSys = 0
		dummywSys = 0
		thisSyst = analysis.Systematics[exp.Experiment] + analysis.Systematics[exp.Source] + analysis.Systematics[exp.Detector]
		for sys in thisSyst:
			index = np.where(analysis.SystematicsList==sys)[0]
			j = index[0]
			xFij = globals()[sys](syst[j],exp)
			wSys += xFij
			dEdx[j] = E * globals()['Diff_'+sys](syst[j],exp)
			usedSysts.append(j)

		Es = E * (1 + wSys)

		# Compute Chi^2
		if np.any(Es<=0):
			X2 = 1e6
		else:
			X2 += 2 * np.sum(Es-O+O*np.log(O/Es))

		# Compute Jacobian of Chi^2
		for i in usedSysts:
			JX2[i] += 2 * np.sum((1-O/Es)*dEdx[i])

	# Systematic's penalty terms
	for i,(x,mu,sig) in enumerate(zip(syst,analysis.SystNominalList,analysis.SystSigmaList)):
		X2 += ((x-mu) / sig)**2
		JX2[i] += 2 * (x-mu) / sig**2

	return (X2,JX2)

def BlockChi2SystsCombined(syst, analysis, Obs, experiments, source):
	JX2 = [0] * len(syst)
	X2 = 0

	if source in experiments.keys():
		BlockExperiment = {source:experiments[source]}
	else:
		BlockExperiment = experiments
	# Experiments
	for exp in BlockExperiment.values():

		# Binned tatistics
		E = exp.weightOscBF_binned
		O = Obs[exp.Experiment]

		#Systematics
		usedSysts = []
		dEdx = [0] * len(syst)
		wSys = 0
		dummywSys = 0
		thisSyst = analysis.Systematics[source]
		for j,sys in enumerate(thisSyst):
			xFij = globals()[sys](syst[j],exp)
			wSys += xFij
			dEdx[j] = E * globals()['Diff_'+sys](syst[j],exp)
			usedSysts.append(j)

		Es = E * (1 + wSys)

		# Compute Chi^2
		if np.any(Es<=0):
			X2 = 1e6
		else:
			X2 += 2 * np.sum(Es-O+O*np.log(O/Es))

		# Compute Jacobian of Chi^2
		for i in usedSysts:
			JX2[i] += 2 * np.sum((1-O/Es)*dEdx[i])

	# Systematic's penalty terms
	for i,(x,mu,sig) in enumerate(zip(syst,analysis.SystNominal[source],analysis.SystSigma[source])):
		X2 += ((x-mu) / sig)**2
		JX2[i] += 2 * (x-mu) / sig**2

	return (X2,JX2)

'''
Older versions
def Chi2SystsCombined(syst, analysis, t12, t13, t23, dm21, dm31, dcp, Ordering, experiments):
	X2 = 0
	# Systematic penalty terms
	for i,s in enumerate(syst):
		X2 = X2 + ((s-analysis.SystNominalList[i])/analysis.SystSigmaList[i])**2

	for exp in experiments:
		wOsc = experiments[exp].Oscillator(analysis.neutrinos, t12, t13, t23, dm21, dm31, dcp, Ordering)

		# Systematic weights
		wSys = np.ones(experiments[exp].NumberOfEvents)
		for i,source in enumerate(analysis.Systematics):
			if source==exp or source==experiments[exp].Source or source==experiments[exp].Detector:
				for sys in analysis.Systematics[source]:
					index = np.where(analysis.SystematicsList==sys)[0]
					j = index[0]
					print(f'Doing {sys}')
					wSys = wSys * globals()[sys](syst[j],experiments[exp])

		# Binned Statistics
		for s in range(experiments[exp].NumberOfSamples):
			wBF = experiments[exp].weightOscBF[experiments[exp].Sample==s] * experiments[exp].Weight[experiments[exp].Sample==s] * experiments[exp].Norm
			w = wSys[experiments[exp].Sample==s] * wOsc[experiments[exp].Sample==s] * experiments[exp].Weight[experiments[exp].Sample==s] * experiments[exp].Norm
			Exp, dx, dy = np.histogram2d(experiments[exp].EReco[experiments[exp].Sample==s], experiments[exp].CosZReco[experiments[exp].Sample==s], bins=(experiments[exp].EnergyBins[s], experiments[exp].CzBins[s]), weights=wBF)
			Obs, dx, dy = np.histogram2d(experiments[exp].EReco[experiments[exp].Sample==s], experiments[exp].CosZReco[experiments[exp].Sample==s], bins=(experiments[exp].EnergyBins[s], experiments[exp].CzBins[s]), weights=w)
			for O,E in zip(np.ravel(Obs),np.ravel(Exp)):
				if O>0 and E>0:
					X2 = X2 + 2 * (E - O + O * math.log(O/E))
				elif O<=0 and E>0:
					X2 = np.inf
					return (X2)
	return (X2)


def Chi2SystsCombined_and_Gradient(syst, analysis, t12, t13, t23, dm21, dm31, dcp, Ordering, experiments):
	JX2 = [0] * len(syst)
	X2 = 0

	# Systematic penalty terms
	for i,s in enumerate(syst):
		X2 = X2 + ((s-analysis.SystNominalList[i])/analysis.SystSigmaList[i])**2
		JX2[i] += 2 * (s-analysis.SystNominalList[i]) / analysis.SystSigmaList[i]**2

	# Experiments
	for exp in experiments:
		dWSys = [0] * len(syst)
		wOsc = experiments[exp].Oscillator(analysis.neutrinos, t12, t13, t23, dm21, dm31, dcp, Ordering)

		# Systematic weights
		wSys = np.ones(experiments[exp].NumberOfEvents)
		for i,source in enumerate(analysis.Systematics):
			if source==exp or source==experiments[exp].Source or source==experiments[exp].Detector:
				for sys in analysis.Systematics[source]:
					index = np.where(analysis.SystematicsList==sys)[0]
					j = index[0]
					Wj = globals()[sys](syst[j],experiments[exp])
					wSys = wSys * Wj
					diff_sys = 'Diff_'+sys
					dWSys[j] = np.ones(experiments[exp].NumberOfEvents) * globals()[diff_sys](syst[j],experiments[exp]) / Wj

		# Binned tatistics
		for s in range(experiments[exp].NumberOfSamples):
			wBF = experiments[exp].weightOscBF[experiments[exp].Sample==s] * experiments[exp].Weight[experiments[exp].Sample==s] * experiments[exp].Norm
			w = wSys[experiments[exp].Sample==s] * wOsc[experiments[exp].Sample==s] * experiments[exp].Weight[experiments[exp].Sample==s] * experiments[exp].Norm

			Exp, dx, dy = np.histogram2d(experiments[exp].EReco[experiments[exp].Sample==s], experiments[exp].CosZReco[experiments[exp].Sample==s], bins=(experiments[exp].EnergyBins[s], experiments[exp].CzBins[s]), weights=wBF)
			Obs, dx, dy = np.histogram2d(experiments[exp].EReco[experiments[exp].Sample==s], experiments[exp].CosZReco[experiments[exp].Sample==s], bins=(experiments[exp].EnergyBins[s], experiments[exp].CzBins[s]), weights=w)
			for O,E in zip(np.ravel(Obs),np.ravel(Exp)):
				if O>0 and E>0:
					X2 = X2 + 2 * (E - O + O * math.log(O/E))
				elif O<=0 and E>0:
					X2 = 99999.

			for i,source in enumerate(analysis.Systematics):
				if source==exp or source==experiments[exp].Source or source==experiments[exp].Detector:
					for sys in analysis.Systematics[source]:
						index = np.where(analysis.SystematicsList==sys)[0]
						j = index[0]
						dummy = dWSys[j]
						dw = w * dummy[experiments[exp].Sample==s]
						dObs, dx, dy = np.histogram2d(experiments[exp].EReco[experiments[exp].Sample==s], experiments[exp].CosZReco[experiments[exp].Sample==s], bins=(experiments[exp].EnergyBins[s], experiments[exp].CzBins[s]), weights=dw)
						for dO,O,E in zip(np.ravel(dObs),np.ravel(Obs),np.ravel(Exp)):
							if O>0 and E>0:
								JX2[j] += 2 * dO * (math.log(O/E))
							elif O<=0 and E>0:
								JX2[j] = 9999.

	return (X2,JX2)


def Chi2SystsCombined_and_Gradient_Binned(syst, analysis, wOsc, experiments):
	JX2 = [0] * len(syst)
	X2 = 0

	# Experiments
	for exp in experiments:
		usedSysts = []
		dWSys = [np.ones(experiments[exp].NumberOfEvents)] * len(syst)
		wSys = np.ones(experiments[exp].NumberOfEvents)
		for i,source in enumerate(analysis.Systematics):
			if source==exp or source==experiments[exp].Source or source==experiments[exp].Detector:
				for sys in analysis.Systematics[source]:
					index = np.where(analysis.SystematicsList==sys)[0]
					j = index[0]
					usedSysts.append(j)
					Wj = globals()[sys](syst[j],experiments[exp])
					diff_sys = 'Diff_'+sys
					wSys = wSys * Wj
					for k in range(len(syst)):
						if k==j:
							dWSys[k] = dWSys[k] * globals()[diff_sys](syst[j],experiments[exp])
						else:
							dWSys[k] = dWSys[k] * Wj

		# Binned tatistics
		Exp = experiments[exp].weightOscBF_binned
		c = Exp>5
		Exp = Exp[c]
		w = wOsc[exp] * experiments[exp].Weight * experiments[exp].Norm
		Obs = experiments[exp].BinIt(wSys * w)
		Obs = Obs[c]
		Obs[Obs<=0] = 1e-5

		X2 += 2 * np.sum(Exp-Obs+Obs*np.log(Obs/Exp))

		for i in usedSysts:
			dObs = experiments[exp].BinIt(dWSys[i] * w)
			dObs = dObs[c]
			JX2[i] += 2 * np.sum(dObs * np.log(Obs/Exp))

	# Systematic penalty terms
	for i,s in enumerate(syst):
		X2 = X2 + ((s-analysis.SystNominalList[i])/analysis.SystSigmaList[i])**2
		JX2[i] += 2 * (s-analysis.SystNominalList[i]) / analysis.SystSigmaList[i]**2


	return (X2,JX2)


def Chi2SystsCombined_and_Gradient_Binned_Next(syst, analysis, wOsc, experiments):
	JX2 = [0] * len(syst)
	X2 = 0

	# Experiments
	for exp in experiments:
		usedSysts = []
		dWSys = [np.ones(experiments[exp].NumberOfEvents)] * len(syst)
		wSys = np.ones(experiments[exp].NumberOfEvents)
		thisSyst = analysis.Systematics[exp] + analysis.Systematics[experiments[exp].Source] + analysis.Systematics[experiments[exp].Detector]
		for sys in thisSyst:
			index = np.where(analysis.SystematicsList==sys)[0]
			j = index[0]
			usedSysts.append(j)
			Wj = globals()[sys](syst[j],experiments[exp])
			diff_sys = 'Diff_'+sys
			wSys = wSys * Wj
			dWSys[j] = globals()[diff_sys](syst[j],experiments[exp]) / Wj

		# Binned tatistics
		Exp = experiments[exp].weightOscBF_binned
		c = Exp>5
		Exp = Exp[c]
		w = wOsc[exp] * experiments[exp].Weight * experiments[exp].Norm
		Obs = experiments[exp].BinIt(wSys * w)[c]
		Obs[Obs<=0] = 1e-5

		X2 += 2 * np.sum(Exp-Obs+Obs*np.log(Obs/Exp))

		for i in usedSysts:
			dObs = experiments[exp].BinIt(dWSys[i] * wSys * w)[c]
			JX2[i] += 2 * np.sum(dObs * np.log(Obs/Exp))

	# Systematic penalty terms
	for i,s in enumerate(syst):
		X2 = X2 + ((s-analysis.SystNominalList[i])/analysis.SystSigmaList[i])**2
		JX2[i] += 2 * (s-analysis.SystNominalList[i]) / analysis.SystSigmaList[i]**2


	return (X2,JX2)


'''