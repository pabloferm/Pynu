from scipy.optimize import minimize
import numpy as np
from .ChiSquared import Chi2StatsCombined, AnalyticPriorsBounds, Chi2SystsCombined

class sensitivity:
	def __init__(self, tunes, outfile, sigma, nominal, Osc):
		""" Compute physics weights and update Expected """
		self.sigma = sigma
		self.nominal = nominal
		self.PTs = tunes
		self.outfile = outfile
		self.OscScenario = Osc

		""" Are any oscillation parameters nuisance? """
		if len(self.nominal[self.OscScenario].keys())>0:
			self.OscNuis = True
		else:
			self.OscNuis = False

	def SetPhysicsPoint(self, physics):
		self.physics = physics
		self.ApplyPhysics()

	def ApplyPhysics(self):
		""" Make oscillations for current point """
		for val,s_array in self.PTs.items():
			for item in s_array:
				""" Oscillations: only apply new parameters """
				item['Osc'].UpdateParameters(**physics[self.OscScenario])
				if not self.OscNuis:
					w = item['Osc'].Oscillator()
					item['Osc'].experiment.UpdateExpectedWeights(w)

				for var,value in an.FixedValue[ExperimentClasses[exp][source].Source].items():
					for tune in ['Flux']: # ['Flux', 'XSec', 'Det']
						w = getattr(item[tune],var)(value)
						ExperimentClasses[exp][source].UpdateNominalWeights(w)



		pass
'''
		pass

		t12, t13, t23, dm21, dm31, dcp = x
		for i,par in enumerate(analysis.OscParametersEdges.values()):
			param = list(analysis.OscParametersEdges.keys())[i]
			if param=='Ordering':
				pass
			elif x[i]<par[0] or x[i]>par[1]:
				return -9999.
		# Compute expectation at BF point
		Obs = {}
		for exp in experiments:
			Obs[exp] = experiments[exp].BinOscillator(analysis.neutrinos, t12, t13, t23, dm21, dm31, dcp, Ordering)
# for each physics point in the grid make oscillations with nominals and points
		statX2 = Chi2StatsCombined(analysis, Obs, experiments)

		if not analysis.wSyst:
			# Writing output
			with open(outfile,'a') as f:
				f.write(f'{t12} {t13} {t23} {dm21} {dm31} {dcp} {Ordering} {statX2}\n')
				f.flush()

			return - 0.5 * statX2

		else:
			# Analytic estimate for priors and bounds
			analysis.SystPrior, bounds = AnalyticPriorsBounds(analysis, Obs, experiments)

			# Combined chi^2 minimization
			tol = max(1e-4,np.sqrt(statX2)*1e-5)
			res = minimize(Chi2SystsCombined, analysis.SystPrior, args=(analysis, Obs, experiments), method='L-BFGS-B', jac=True, bounds=bounds, options={'disp' : False, 'ftol' : tol, 'gtol': 1e-03})

			# Writing output
			sys_data = ' '.join(map(str,res.x))
			with open(outfile,'a') as f:
				f.write(f'{t12:.3f} {t13:.3f} {t23:.5f} {dm21:.7f} {dm31:.7f} {dcp:.3f} {Ordering} {sys_data} {res.fun:.2f}\n')
				f.flush()

			return - 0.5 * res.fun
'''

'''
	def Chi2StatsCombined(self, analysis, Obs, experiments):
		""" Compute statistics only chi-squared """
		X2 = 0
		for exp in experiments.values():
			# Binned tatistics
			E = exp.weightOscBF_binned
			O = Obs[exp.Experiment]
			X2 += 2 * np.sum(E-O+O*np.log(O/E))	

		return X2


	def AnalyticPriorsBounds(self, analysis, Obs, experiments):
		""" First order analytic computation of values for parameters to be mariginalized """
		priors = [0] * len(analysis.SystPrior)
		A = [0] * len(analysis.SystPrior)
		B = [0] * len(analysis.SystPrior)

		# Experiments
		for exp in experiments.values():
			# Binned statistics
			E = exp.weightOscBF_binned
			O = Obs[exp.Experiment]
			OmE = O-E
			#Systematics
			usedSysts = []
			dfdx = [0] * len(analysis.SystPrior)
			thisSyst = analysis.Systematics[exp.Experiment] + analysis.Systematics[exp.Source] + analysis.Systematics[exp.Detector]
			for sys in thisSyst:
				index = np.where(analysis.SystematicsList==sys)[0]
				j = index[0]
				dfdx[j] = globals()['Diff_'+sys](analysis.SystNominalList[j],exp)
				usedSysts.append(j)
			# Compute Jacobian of Chi^2
			for i in usedSysts:
				A[i] += np.sum(OmE * dfdx[i])
				B[i] += np.sum(O * dfdx[i] * dfdx[i])
		# Systematic's penalty terms
		bnds = []
		for i,(mu,sig) in enumerate(zip(analysis.SystNominalList,analysis.SystSigmaList)):
			pr = mu + A[i] / (B[i] + 1/sig**2)
			delta = min(np.abs(pr-mu), sig)
			priors[i] = 0.5*(mu+pr)
			if delta>0:
				bnds.append((priors[i]-delta, priors[i]+delta))
			else:
				bnds.append((priors[i]-sig, priors[i]+sig))
		return priors, tuple(bnds)



	def Chi2SystsCombined(self, syst, analysis, Obs, experiments):
		""" Compute chi-squared value with systematics """
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
'''