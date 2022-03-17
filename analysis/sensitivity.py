from scipy.optimize import minimize
import multiprocessing
from chiSquared import Chi2StatsCombined, Chi2SystsCombined, BlockChi2SystsCombined
import numpy as np

def sensitivity(analysis, t12, t13, t23, dm21, dm31, dcp, Ordering, experiments, outfile, queue):
	
	if analysis.NoSyst:
		Chi2StatsCombined(analysis.neutrinos, t12, t13, t23, dm21, dm31, dcp, Ordering, experiments, outfile)

	else:
		priors = np.array([])

		Obs = {}
		# Compute expectation at BF point
		for exp in experiments:
			Obs[exp] = experiments[exp].BinOscillator(analysis.neutrinos, t12, t13, t23, dm21, dm31, dcp, Ordering)

		# Minimize over systematic sources to feed the combined minimizer with priors
		if np.all(analysis.SystNominalList == analysis.SystPrior):
			print('Getting priors...')
			for source in analysis.Systematics:
				if len(analysis.Systematics[source])>0:
					bnds = []
					for prior,sigma in zip(analysis.SystNominal[source],analysis.SystSigma[source]):
						bnd_min = prior - 2*sigma
						bnd_max = prior + 2*sigma
						if prior==1:
							bnd_min = max(0.5,bnd_min)
							bnd_max = min(2,bnd_max)
						bnds.append((bnd_min,bnd_max))
					bounds = tuple(bnds)
					res = minimize(BlockChi2SystsCombined, analysis.SystNominal[source], args=(analysis, Obs, experiments, source), method='L-BFGS-B', jac=True, bounds=bounds, options={'disp' : False, 'ftol' : 1e-4, 'gtol': 1e-03})
					priors = np.append(priors, res.x)
			analysis.SystPrior = priors

		bnds = []
		for prior,sigma in zip(analysis.SystPrior,analysis.SystSigmaList):
			bnd_min = prior - 2*sigma
			bnd_max = prior + 2*sigma
			if prior==1:
				bnd_min = max(0.5,bnd_min)
				bnd_max = min(2,bnd_max)
			bnds.append((bnd_min,bnd_max))
		bounds = tuple(bnds)

		# Combined chi^2 minimization
		res = minimize(Chi2SystsCombined, analysis.SystPrior, args=(analysis, Obs, experiments), method='L-BFGS-B', jac=True, bounds=bounds, options={'disp' : False, 'ftol' : 1e-4, 'gtol': 1e-03})

		# type(analysis).SystPrior = res.x
		if type(queue) == multiprocessing.queues.Queue:
			queue.put(res.x)

		# Writing output
		sys_data = ' '.join(map(str,res.x))
		with open(outfile,'a') as f:
			f.write(f'{t12} {t13} {t23} {dm21} {dm31} {dcp} {Ordering} {sys_data} {res.fun}\n')
			f.flush()
