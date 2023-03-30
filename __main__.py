import sys
import os
import numpy as np
from itertools import product
import argparse

import AnalysisReader as AR  # contains parse class to read and setup the analysis
import Experiments as Exp  # contains rd class to read and setup each experiment
# contains everything to modify your simulations to help figuring out what
# you have measured
import PhysicsTunes as PT
import Fitter as FT  # does all the fitting calculations

import time

print('=============================================================\n===================== Analysis setup ========================\n=============================================================')

# Read arguments
############################
parse = argparse.ArgumentParser()
parse.add_argument(
    "xml_file",
    type=str,
    nargs='?',
    default=os.environ['PYNU'] +
    '/examples/AnalysisFiles/test.xml',
    help='Input analysis file in xml format.')
parse.add_argument(
    '-p', '--point', nargs='+', type=int, default=0,
    help='Specify analysis point or points (p0 p1 p2 p3) to run.')
parse.add_argument(
    '-rp',
    '--range_of_points',
    nargs='+',
    type=int,
    default=None,
    help='Specify range (start and end) of analysis points to run, p0 p3 = p0 p0+1 p0+2 ... p3-1 p3. Edges are included.')
parse.add_argument(
    '-o',
    '--outfile',
    nargs='?',
    type=str,
    default='outfile.dat',
    help='Analysis output file.')
parse.add_argument(
    "--multi",
    dest='multiproc',
    default=False,
    action='store_true',
    help='Option for running the analysis with multiprocessing (recommended locally).')
parse.add_argument(
    "--cluster",
    dest='cluster',
    default=False,
    action='store_true',
    help='Option for submitting jobs to a cluster.')
parse.add_argument(
    "--mcmc",
    dest='mcmc',
    default=False,
    action='store_true',
    help='Option for sampling parameter space using Markov Chain Monte Carlo.')
args = parse.parse_args()

# Setup running points
############################
if args.range_of_points is None:
    points = np.unique(args.point)
else:
    points = np.arange(
        int(args.range_of_points[0]),
        1 + int(args.range_of_points[-1]))

# Setup analysis from xml file
############################
an = AR.parse(args.xml_file, check=False)

# Setup all experiments and
# their physics tunes
############################
ExperimentClasses = {}
PhysicsTunesClasses = {}

# Loop over detectors
for exp in an.Experiments:
    ExperimentClasses[exp] = {}
    PhysicsTunesClasses[exp] = {}

    # Loop over neutrino sources
    for source in an.Experiments[exp]:

        # Experiment
        ExperimentClasses[exp][source] = Exp.Manager(
            exp, source, an.Experiments[exp][source])

        # Oscillations
        PhysicsTunesClasses[exp][source] = {}
        PhysicsTunesClasses[exp][source]['Osc'] = PT.Oscillations(
            an.OscScenario, source, an.Flavors, ExperimentClasses[exp][source])
        PhysicsTunesClasses[exp][source]['Osc'].SetParameters(
            **an.OscNominalParameters)
        w = PhysicsTunesClasses[exp][source]['Osc'].Oscillator()
        ExperimentClasses[exp][source].UpdateNominalWeights(w)
        if an.OscScenario not in an.Physics.keys(
        ) and an.OscScenario not in an.Nuisance.keys():
            ExperimentClasses[exp][source].UpdateBaseWeights(w)

        # Flux
        PhysicsTunesClasses[exp][source]['Flux'] = PT.Flux(
            source, ExperimentClasses[exp][source])
        # CrossSection
        # PhysicsTunesClasses[exp][source]['XSec'] = PT.CrossSection(source, ExperimentClasses[exp][source])
        # Detector
        # PhysicsTunesClasses[exp][source]['Det'] = PT.Detector(source, ExperimentClasses[exp][source])

        # Loop over fixed values (excluding oscillations, whose parameters need
        # to be fed all at once and was already done)
        for var, value in an.FixedValue[ExperimentClasses[exp][source].Source].items(
        ):
            w = getattr(PhysicsTunesClasses[exp][source]['Flux'], var)(value)
            ExperimentClasses[exp][source].UpdateNominalWeights(w)
            ExperimentClasses[exp][source].UpdateBaseWeights(w)

        # Loop over physics and nuisance true and nominal values (excluding
        # oscillations, whose parameters need to be fed all at once and was
        # already done)
        for var, value in an.PhysTrue[ExperimentClasses[exp][source].Source].items(
        ) | an.NuisNominal[ExperimentClasses[exp][source].Source].items():
            w = getattr(PhysicsTunesClasses[exp][source]['Flux'], var)(value)
            ExperimentClasses[exp][source].UpdateNominalWeights(w)


# ExperimentClasses[exp][source].BinNominalWeights()
# ExperimentClasses[exp][source].SetObservedBinned()
# ExperimentClasses[exp][source].GetObservedBinned()


# Write first line of output file
############################
if (args.cluster and not os.path.isfile(args.outfile)) or not args.cluster:
    with open(args.outfile, 'w') as f:
        for par in an.PhysicsList:
            f.write(par + ' ')
        if an.wSyst:
            for sys in an.NuisanceList:
                f.write(sys + ' ')
        f.write('X2 ')
        f.write('\n')

print('=============================================================\n==================== Starting analysis ======================\n=============================================================')


# I'm here
# Main analysis loop
############################

# Make grid of all points to be sampled from physics parameter space
param = []
for source, dsource in an.PhysGrid.items():
    for item, array in dsource.items():
        param.append(an.PhysGrid[source][item])
parametersGrid = product(*param)

# print(an.Physics)
# print(len(list(parametersGrid)[0]))
# print(an.NumberOfPhysPoints)
# print(an.NumberOfPhys)


Sens = FT.sensitivity(
    PhysicsTunesClasses,
    args.outfile,
    an.NuisSigma,
    an.NuisNominal,
    an.OscScenario)


if args.cluster:
    parametersGridList = list(parametersGrid)
    parametersPointDict = {}
    for p in points:
        element = parametersGridList[p]
        i = 0
        for s, a in an.Physics.items():
            parametersPointDict[s] = {}
            for v in a:
                parametersPointDict[s][v] = element[i]
                i = i + 1
        print(f'Processing point {p} --> {element}')
        print(parametersPointDict)
        Sens.SetPhysicsPoint(parametersPointDict)


'''

	elif args.multiproc:
		import multiprocessing
		cores = multiprocessing.cpu_count()
		if an.NoSyst:
			cores = 3*cores
			print('Analyzing with no systematics')

		if args.mcmc:
			# Testing
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
				sampler = emcee.EnsembleSampler(nwalkers, ndim, sensitivity, args=[mo, an, mcList, args.outfile], pool=pool)
				# state = sampler.run_mcmc(initial, 10, progress=True, skip_initial_state_check=True)
				# sampler.reset()
				sampler.run_mcmc(initial, nsteps, progress=True, skip_initial_state_check=True)

		else:
			with multiprocessing.Pool(processes=cores) as pool:
				for element in parametersGrid:
					print(f'Processing {element}')
					res = pool.apply_async(sensitivity, args=(element[:-1], element[-1], an, mcList, args.outfile))
				pool.close()
				pool.join()

	else:
		for element in parametersGrid:
			print(f'Processing {element}')
			sensitivity(element[:-1], element[-1], an, mcList, args.outfile)
'''
