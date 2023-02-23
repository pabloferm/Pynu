import sys
import os
import argparse

from PyNu import PyNu

import time
from itertools import product


print('=============================================================\n' +
	'===================== Analysis setup ========================\n' +
	'=============================================================')

# Read arguments
############################
parse = argparse.ArgumentParser()
parse.add_argument("xml_file", type=str, nargs='?', default=os.environ['PYNU']+'/examples/AnalysisFiles/test.xml', 
	help='Input analysis file in xml format.')
parse.add_argument('-p', '--point', nargs='+', type=int, default=None, 
	help='Specify analysis point or points (p0 p1 p2 p3) to run.')
parse.add_argument('-rp', '--range_of_points', nargs='+', type=int, default=None, 
	help='Specify range (start and end) of analysis points to run, p0 p3 = p0 p0+1 p0+2 ... p3-1 p3. Edges are included.')
parse.add_argument('-o', '--outfile', nargs='?', type=str, default='outfile.dat', 
	help='Analysis output file.')
parse.add_argument("--multi", dest='multiproc', default=False, action='store_true', 
	help='Option for running the analysis with multiprocessing (recommended locally).') 
parse.add_argument("--cluster", dest='cluster', default=False, action='store_true', 
	help='Option for submitting jobs to a cluster.')
parse.add_argument("--mcmc", dest='mcmc', default=False, action='store_true', 
	help='Option for sampling parameter space using Markov Chain Monte Carlo.')
args = parse.parse_args()

# Setup analysis from xml file
############################
pynu = PyNu(args.xml_file, verbosity=False)

# Setup running points
############################
if args.range_of_points is None and args.point is not None:
	points = [*set(args.point)]
	if points >= pynu.Analysis.NumberOfPhysPoints:
		sys.exit('Point out of range for this analysis.')
elif args.range_of_points is not None and args.point is None:
	points = list(range(int(args.range_of_points[0]), 1+int(args.range_of_points[-1])))
	if points[-1] >= pynu.Analysis.NumberOfPhysPoints:
		sys.exit('Point out of range for this analysis.')
else: # run over all analysis points
	points = range(0,pynu.Analysis.NumberOfPhysPoints)

print(points)


pynu.SetUpExperiments()
# print(pynu.Experiments)

pynu.SetUpPhysicsTunes()
# print(pynu.PhysicsTunes)

# Compute nominal weights for the analysis (aka Observation)
pynu.ApplyFixedWeights()
pynu.ApplyNominalWeights()
pynu.ApplyTrueWeights()
pynu.ApplyOscillations()
pynu.SetObservedEvents()
for key, val in pynu.Observation.items():
	val0 = val[100]

# Loop over specified points of analysis
for p in points:
	# Compute weights at a given point of the physics grid (fixed part for nuisance minimisation)
	pynu.StartExpectation()
	pynu.ApplyPhysicsWeights(p)
	pynu.ApplyNuisanceWeights(pynu.Analysis.NuisNominalList)
	pynu.ApplyOscillations(Expectation=True)
	pynu.SetExpectedEvents()
	for key, val in pynu.Expectation.items():
		val = val[100]
		print(val, val0)
	# Compute chi2-value
	# print(p, pynu.Sensitivity())


