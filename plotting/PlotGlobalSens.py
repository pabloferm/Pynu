import numpy as np
import pandas as pd
from plotting import cornerPlot, cornerPlotBothO
import sys

def X2readerGeneral(filename):	
	df = pd.read_csv(filename, sep=' ')
	rawOscPar = {}
	oscPar = {}
	for i in range(7):
		rawOscPar[i] = df.values[:,i]
		oscPar[i] = np.unique(rawOscPar[i])
	rawchi2 = np.array(df["X2"])
	minchi2 = np.amin(rawchi2)
	chi2 = rawchi2 - minchi2
	print(f'Minimum x2 = {minchi2}')

	fixPar = []
	fitPar = []
	switchPar = []
	for par in oscPar:
		if oscPar[par].size==1:
			fixPar.append(par)
		elif oscPar[par].size==2:
			switchPar.append(par)
		else:
			fitPar.append(par)

	if len(switchPar)>0:
		ChiSq = {}
		for switch in switchPar:
			for par1 in fitPar:
				for par2 in fitPar:
					if par1!=par2:
						for i,scenario in enumerate(oscPar[switch]):
							ChiSq[i] = np.array([])
							for val1 in oscPar[par1]:
								for val2 in oscPar[par2]:
									cut = (rawOscPar[par1]==val1) * (rawOscPar[par2]==val2) * (rawOscPar[switch]==scenario)
									chi2_slice = chi2[cut]
									marg = np.amin(chi2_slice)
									ChiSq[i] = np.append(ChiSq[i], marg)
						cornerPlotBothO(oscPar[par1],oscPar[par2],ChiSq[0],ChiSq[1])
							


	else:
		for par1 in fitPar:
			for par2 in fitPar:
				if par1!=par2:
					ChiSq = np.array([])
					for val1 in oscPar[par1]:
						for val2 in oscPar[par2]:
							cut = (rawOscPar[par1]==val1) * (rawOscPar[par2]==val2)
							chi2_slice = chi2[cut]
							marg = np.amin(chi2_slice)
							ChiSq = np.append(ChiSq, marg)
					cornerPlot(oscPar[par1],oscPar[par2],ChiSq)


def X2readerGeneralCombined(filename, filename2):	
	df = pd.read_csv(filename, sep=' ')
	rawOscPar = {}
	oscPar = {}
	for i in range(7):
		rawOscPar[i] = df.values[:,i]
		oscPar[i] = np.unique(rawOscPar[i])
	chi2 = np.array(df["X2"])
	rawOscPar2 = {}
	oscPar2 = {}
	df2 = pd.read_csv(filename2, sep=' ')
	for i in range(df2.columns.size-1):
		rawOscPar2[i] = df2.values[:,i]
		oscPar2[i] = np.unique(rawOscPar2[i])
	
	chi20 = np.array(df2["X2"])
	chi2 = chi2 + chi20
	i0,j0 = 0,0
	count=0
	cond = 1
	for i in range(rawOscPar[0].size):
		cond = 1
		for k in range(7):
			cond = cond * rawOscPar2[k][i]==rawOscPar[k][i]
		chi2[i0] = chi2[i0] + chi20[j0]


	fixPar = []
	fitPar = []
	switchPar = []
	for par in oscPar:
		if oscPar[par].size==1:
			fixPar.append(par)
		elif oscPar[par].size==2:
			switchPar.append(par)
		else:
			fitPar.append(par)

	if len(switchPar)>0:
		for switch in switchPar:
			for par1 in fitPar:
				for par2 in fitPar:
					if par1!=par2:
						for scenario in oscPar[switch]:
							ChiSq = np.array([])
							for val1 in oscPar[par1]:
								for val2 in oscPar[par2]:
									cut = (rawOscPar[par1]==val1) * (rawOscPar[par2]==val2) * (rawOscPar[switch]==scenario)
									chi2_slice = chi2[cut]
									marg = np.amin(chi2_slice)
									ChiSq = np.append(ChiSq, marg)
							cornerPlot(oscPar[par1],oscPar[par2],ChiSq)
	else:
		for par1 in fitPar:
			for par2 in fitPar:
				if par1!=par2:
					ChiSq = np.array([])
					for val1 in oscPar[par1]:
						for val2 in oscPar[par2]:
							cut = (rawOscPar[par1]==val1) * (rawOscPar[par2]==val2)
							chi2_slice = chi2[cut]
							marg = np.amin(chi2_slice)
							ChiSq = np.append(ChiSq, marg)
					cornerPlot(oscPar[par1],oscPar[par2],ChiSq)





experiment = str(sys.argv[1])
filename = str(sys.argv[2])
if len(sys.argv)==3 and (experiment=='SK' or experiment=='IC'):
	X2readerGeneral(filename)
elif len(sys.argv)==4 and (experiment=='SK' or experiment=='IC'):
	filename2 = sys.argv[3]
	X2readerGeneral(filename, filename2)
elif experiment=='SK+IC' or experiment=='IC+SK':
	if len(sys.argv)==3:
		X2readerGeneral(filename)
	elif len(sys.argv)==4:
		print('You are addind IC and SK X2 surfaces, make sure they share common points...')
		filename2 = sys.argv[3]
		X2readerGeneralCombined(filename,filename2)
