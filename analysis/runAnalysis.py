import sys
import os.path
import numpy as np
import math
from math import asin, sqrt
from SimReader import Reader
import multiprocessing
from xmlReader import parseXML
from itertools import product
from Sensitivity import sensitivity
import argparse

# Read arguments
############################
parser = argparse.ArgumentParser()
parser.add_argument("xml_file", type=str, nargs='?', default='xmlAnalysis/AnalysisTemplate.xml', help='Input analysis file in xml format')
parser.add_argument('-p', '--point', nargs='?', type=int, default=0, help='Specify analysis point to run. Only if \'cluster\' option is enabled')
parser.add_argument('-o', '--outfile', nargs='?', type=str, default='out.dat', help='Analysis output file')
parser.add_argument("--multi", dest='multiproc', default=False, action='store_true', help='Option for running the analysis with multiprocessing (recommended locally)') 
parser.add_argument("--cluster", dest='cluster', default=False, action='store_true', help='Option for submitting jobs to a cluster')
args = parser.parse_args()

# Setup running flags
############################
# mcmc = args.mcmc
multiproc = args.multiproc
cluster = args.cluster
if cluster:
	if args.point is None:
		point=0
	else:
		point = args.point

# Setup analysis files
############################
analysis_xml_file = args.xml_file
outfile = args.outfile

# Setup analysis from xml file
############################
an = parseXML(analysis_xml_file)
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
		# print(s,exp,t,fil)
		mcList[exp] = Reader(s,exp,t,fil)
		mcList[exp].Binning()
		# Get unoscillated atm. fluxes
		mcList[exp].InitialFlux()
		# Set best fit value oscillations
		mcList[exp].BFOscillator(an.neutrinos,**an.OscParametersBest)


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

print('=============================================================')
print('==================== Starting analysis ======================')
print('=============================================================')

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
		sensitivity(an, *element, mcList, outfile, None)

	elif multiproc:
		cores = multiprocessing.cpu_count()
		if an.NoSyst:
			cores = 3*cores
			print('Analyzing with no systematics')
		processes = []
		jj = 0
		q = multiprocessing.Queue()
		for element in parametersGrid:
			p = multiprocessing.Process(target=sensitivity,args=(an, *element, mcList, outfile, q, ))
			if __name__ == "__main__":
				processes.append(p)
				p.start()
				jj = jj + 1
				print(f'{element} process started')
				if jj%cores==0:
					for i,p in enumerate(processes):
						p.join()
						processes = []
					an.SystPrior = q.get()
					print(an.SystPrior)

	else:
		for element in parametersGrid:
			print(f'Processing {element}')
			sensitivity(an, *element, mcList, outfile, None)
