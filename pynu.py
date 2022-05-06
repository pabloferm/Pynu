import sys
import os.path
import numpy as np
from itertools import product
import argparse

from Experiments import * # contains rd class to read and setup each experiment
from AnalysisReader import * # contains parse class to read and setup the analysis
from Analysis import * # import Sensitivity # Computes sensitivity

# Read arguments
############################
input_arg = argparse.ArgumentParser()
input_arg.add_argument("xml_file", type=str, nargs='?', default='xmlAnalysis/AnalysisTemplate.xml', help='Input analysis file in xml format.')
input_arg.add_argument('-p', '--point', nargs='?', type=int, default=0, help='Specify analysis point to run. Only if \'cluster\' option is enabled.')
input_arg.add_argument('-lp', '--list_of_points', nargs='?', type=list, default=0, help='Specify set of analysis points to run.')
input_arg.add_argument('-o', '--outfile', nargs='?', type=str, default='out.dat', help='Analysis output file.')
input_arg.add_argument("--multi", dest='multiproc', default=False, action='store_true', help='Option for running the analysis with multiprocessing (recommended locally).') 
input_arg.add_argument("--cluster", dest='cluster', default=False, action='store_true', help='Option for submitting jobs to a cluster.')
input_arg.add_argument("--mcmc", dest='mcmc', default=False, action='store_true', help='Option for sampling parameter space using Markov Chain Monte Carlo.')
args = input_arg.parse_args()

# Setup running flags
############################
multiproc = args.multiproc
cluster = args.cluster
markov = args.mcmc
if cluster:
	if args.point is None:
		point=0
	else:
		point = args.point

# Setup analysis files
############################
analysis_xml_file = args.xml_file # input
outfile = args.outfile # output

# Setup analysis from xml file
############################
an = parse(analysis_xml_file)
an.readSources()
an.readExperiments()
an.readDetectors()
an.readPhysics()
an.readOscPar()
an.CheckSystematics()

# Setup all experiments
############################
mcList = {}
for s in an.sources:
	for i,(exp,fil,t) in enumerate(zip(an.experiments,an.mcFiles,an.Exposure)):
		mcList[exp] = rd(s,exp,t,fil)
		mcList[exp].Binning()		
		mcList[exp].InitialFlux() # Get unoscillated fluxes
		mcList[exp].BFOscillator(an.neutrinos,**an.OscParametersBest) # Set best fit value oscillations


# Write first line of output file
############################
if (cluster and (point==0 or not os.path.isfile(outfile))) or not cluster:
	with open(outfile,'w') as f:
		for par in an.parameters:
			f.write(par+' ')
		if an.NoSyst == 0:
			for sys in an.Systematics.values():
				for s in sys:
					f.write(s+' ')
		f.write('X2 ')
		f.write('\n')

print('=============================================================\n==================== Starting analysis ======================\n=============================================================')


# Main analysis loop
############################

if an.physics[0] == 'Three Flavour':
	# Osc. in parameters space
	param = []
	for oscpar in [*an.OscParametersGrid]:
		param.append(an.OscParametersGrid[oscpar])
	parametersGrid = product(*param)

	if cluster:
		element = list(parametersGrid)[point]
		print(f'Processing {element}')
		sensitivity(element[:-1], element[-1], an, mcList, outfile)

	elif multiproc:
		import multiprocessing
		cores = multiprocessing.cpu_count()
		if an.NoSyst:
			cores = 3*cores
			print('Analyzing with no systematics')

		if markov:
			''' Testing '''
			import emcee
			nwalkers = 2**4
			ndim = len(an.OscParametersEdges) - 1
			nsteps = 200
			initial = np.zeros((nwalkers,ndim))
			for i,par in enumerate(an.OscParametersEdges.values()):
				param = list(an.OscParametersEdges.keys())[i]
				if param=='Ordering':
					pass
				else:
					mu = an.OscParametersBest[param]
					sigma = min(abs(par[0]-mu),abs(par[1]-mu))
					initial[:,i] = np.random.uniform(par[0],par[1],nwalkers)
			mo = an.OscParametersBest['Ordering']
			with multiprocessing.Pool(processes=cores) as pool:
				sampler = emcee.EnsembleSampler(nwalkers, ndim, sensitivity, args=[mo, an, mcList, outfile], pool=pool)
				# state = sampler.run_mcmc(initial, 10, progress=True, skip_initial_state_check=True)
				# sampler.reset()
				sampler.run_mcmc(initial, nsteps, progress=True, skip_initial_state_check=True)

		else:
			with multiprocessing.Pool(processes=cores) as pool:
				for element in parametersGrid:
					print(f'Processing {element}')
					res = pool.apply_async(sensitivity, args=(element[:-1], element[-1], an, mcList, outfile))
				pool.close()
				pool.join()				

	else:
		for element in parametersGrid:
			print(f'Processing {element}')
			sensitivity(element[:-1], element[-1], an, mcList, outfile)