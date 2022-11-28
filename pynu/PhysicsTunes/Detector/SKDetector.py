import numpy as np

####################
# Super-Kamiokande #
####################

def SKEnergyScale(x, experiment):
	if np.abs(x-1)<5e-4: 
		return np.zeros(experiment.NumberOfBins)
	return experiment.Exp_wBinIt(1,shift_E=x) / experiment.weightOscBF_binned - 1

def FCPCSeparation(x,experiment):
	fcpc = np.ones(experiment.NumberOfEvents)
	if experiment.Experiment == 'SuperK-Gd' or experiment.Experiment == 'SKIV' or experiment.Experiment == 'SuperK_Htag' or experiment.Experiment == 'SuperK_Gdtag':
		fcpc[experiment.Sample<16] = x
		wFC = np.sum(experiment.Weight[experiment.Sample<16])
		wPC = np.sum(experiment.Weight[np.logical_or(experiment.Sample==16, experiment.Sample==17)])
		y = ((wPC+wFC)-x*wFC) / wPC
		fcpc[np.logical_or(experiment.Sample==16,experiment.Sample==17)] = y
	else:
		fcpc[experiment.Sample<14] = x
		wFC = np.sum(experiment.Weight[experiment.Sample<14])
		wPC = np.sum(experiment.Weight[np.logical_or(experiment.Sample==14, experiment.Sample==15)])
		y = ((wPC+wFC)-x*wFC) / wPC
		fcpc[np.logical_or(experiment.Sample==14,experiment.Sample==15)] = y
	return experiment.Exp_wBinIt(fcpc) / experiment.weightOscBF_binned - 1

def FCReduction(x,experiment):
	fc = np.ones(experiment.NumberOfEvents)
	if experiment.Experiment == 'SuperK-Gd' or experiment.Experiment == 'SKIV' or experiment.Experiment == 'SuperK_Htag' or experiment.Experiment == 'SuperK_Gdtag':
		fc[experiment.Sample<16] = x
	else:
		fc[experiment.Sample<14] = x
	return experiment.Exp_wBinIt(fc) / experiment.weightOscBF_binned - 1

def FiducialVolume(x,experiment):
	return (x-1)

def PCReduction(x,experiment):
	pc = np.ones(experiment.NumberOfEvents)
	if experiment.Experiment == 'SuperK-Gd' or experiment.Experiment == 'SKIV' or experiment.Experiment == 'SuperK_Htag' or experiment.Experiment == 'SuperK_Gdtag':
		pc[np.logical_or(experiment.Sample==16 , experiment.Sample==17)] = x
	else:
		pc[np.logical_or(experiment.Sample==14, experiment.Sample==15)] = x
	return experiment.Exp_wBinIt(pc) / experiment.weightOscBF_binned - 1

def SubGeV2ringPi0(x,experiment):
	pi02r = np.ones(experiment.NumberOfEvents)
	pi02r[experiment.Sample==6] = x
	return experiment.Exp_wBinIt(pi02r) / experiment.weightOscBF_binned - 1

def SubGeV1ringPi0(x,experiment):
	pi01r = np.ones(experiment.NumberOfEvents)
	if experiment.Experiment == 'SuperK-Gd' or experiment.Experiment == 'SKIV' or experiment.Experiment == 'SuperK_Htag' or experiment.Experiment == 'SuperK_Gdtag':
		pi01r[experiment.Sample==3] = x
	else:
		pi01r[experiment.Sample==2] = x
	return experiment.Exp_wBinIt(pi01r) / experiment.weightOscBF_binned - 1

def MultiRing_NuNuBarSeparation(x,experiment):
	if experiment.Experiment == 'SuperK-Gd' or experiment.Experiment == 'SKIV' or experiment.Experiment == 'SuperK_Htag' or experiment.Experiment == 'SuperK_Gdtag':
		nu = 12
		nub = 13
	else:
		nu = 10
		nub = 11
	mr = np.ones(experiment.NumberOfEvents)
	n0 = np.sum(experiment.Weight[experiment.Sample==nu])
	n1 = np.sum(experiment.Weight[experiment.Sample==nub])
	r = n0 / n1
	mr[experiment.Sample==nu] = x
	mr[experiment.Sample==nub] = 1+r-r*x
	return experiment.Exp_wBinIt(mr) / experiment.weightOscBF_binned - 1

def MultiRing_EMuSeparation(x,experiment):
	if experiment.Experiment == 'SuperK-Gd' or experiment.Experiment == 'SKIV' or experiment.Experiment == 'SuperK_Htag' or experiment.Experiment == 'SuperK_Gdtag':
		e0 = 12
		e1 = 13
		e2 = 15
		mu = 14
	else:
		e0 = 10
		e1 = 11
		e2 = 13
		mu = 12
	mr = np.ones(experiment.NumberOfEvents)
	n0 = np.sum(experiment.Weight[experiment.Sample==e0]) + np.sum(experiment.Weight[experiment.Sample==e1]) + np.sum(experiment.Weight[experiment.Sample==e2])
	n1 = np.sum(experiment.Weight[experiment.Sample==mu])
	r = n0 / n1
	mr[experiment.Sample==e0] = x
	mr[experiment.Sample==e1] = x
	mr[experiment.Sample==e2] = x
	mr[experiment.Sample==mu] = 1+r-r*x
	return experiment.Exp_wBinIt(mr) / experiment.weightOscBF_binned - 1

def MultiRing_EOtherSeparation(x,experiment):
	if experiment.Experiment == 'SuperK-Gd' or experiment.Experiment == 'SKIV' or experiment.Experiment == 'SuperK_Htag' or experiment.Experiment == 'SuperK_Gdtag':
		e0 = 12
		e1 = 13
		o0 = 15
	else:
		e0 = 10
		e1 = 11
		o0 = 13
	mr = np.ones(experiment.NumberOfEvents)
	n0 = np.sum(experiment.Weight[experiment.Sample==e0]) + np.sum(experiment.Weight[experiment.Sample==e1])
	n1 = np.sum(experiment.Weight[experiment.Sample==o0])
	r = n0 / n1
	mr[experiment.Sample==e0] = x
	mr[experiment.Sample==e1] = x
	mr[experiment.Sample==o0] = 1+r-r*x
	return experiment.Exp_wBinIt(mr) / experiment.weightOscBF_binned - 1

def PC_StopThruSeparation(x,experiment):
	if experiment.Experiment == 'SuperK-Gd' or experiment.Experiment == 'SKIV' or experiment.Experiment == 'SuperK_Htag' or experiment.Experiment == 'SuperK_Gdtag':
		pcs = 16
		pct = 17
	else:
		pcs = 14
		pct = 15
	mr = np.ones(experiment.NumberOfEvents)
	n0 = np.sum(experiment.Weight[experiment.Sample==pcs])
	n1 = np.sum(experiment.Weight[experiment.Sample==pct])
	r = n0 / n1
	mr[experiment.Sample==pcs] = x
	mr[experiment.Sample==pct] = 1+r-r*x
	return experiment.Exp_wBinIt(mr) / experiment.weightOscBF_binned - 1

def Pi0_RingSeparation(x,experiment):
	if experiment.Experiment == 'SuperK-Gd' or experiment.Experiment == 'SKIV' or experiment.Experiment == 'SuperK_Htag' or experiment.Experiment == 'SuperK_Gdtag':
		r1 = 3
		r2 = 6
	else:
		r1 = 2
		r2 = 6
	mr = np.ones(experiment.NumberOfEvents)
	n0 = np.sum(experiment.Weight[experiment.Sample==r1])
	n1 = np.sum(experiment.Weight[experiment.Sample==r2 ])
	r = n0 / n1
	mr[experiment.Sample==r1] = x
	mr[experiment.Sample==r2 ] = 1+r-r*x
	return experiment.Exp_wBinIt(mr) / experiment.weightOscBF_binned - 1

def E_RingSeparation(x,experiment):
	if experiment.Experiment == 'SuperK-Gd' or experiment.Experiment == 'SKIV' or experiment.Experiment == 'SuperK_Htag' or experiment.Experiment == 'SuperK_Gdtag':
		r1= [0,1,2,7,8,9]
		r2 = [12,13,14]
	else:
		r1= [0,1,2,7,8,9]
		r2 = [12,13,14]
	mr = np.ones(experiment.NumberOfEvents)
	n0 = 0
	n1 = 0
	for sample in r1 :
		n0 += np.sum(experiment.Weight[experiment.Sample==sample])
	for sample in r2 :
		n1 += np.sum(experiment.Weight[experiment.Sample==sample])
	r = n0 / n1
	for sample in r1 :
		mr[experiment.Sample==sample] = x
	for sample in r2 :
		mr[experiment.Sample==sample] = 1+r-r*x
	return experiment.Exp_wBinIt(mr) / experiment.weightOscBF_binned - 1

def Mu_RingSeparation(x,experiment):
	if experiment.Experiment == 'SuperK-Gd' or experiment.Experiment == 'SKIV' or experiment.Experiment == 'SuperK_Htag' or experiment.Experiment == 'SuperK_Gdtag':
		r1= [4,5,10,11]
		r2 = [14]
	else:
		r1= [3,4,5,9]
		r2 = [12]
	mr = np.ones(experiment.NumberOfEvents)
	n0 = 0
	n1 = 0
	for sample in r1 :
		n0 += np.sum(experiment.Weight[experiment.Sample==sample])
	for sample in r2 :
		n1 += np.sum(experiment.Weight[experiment.Sample==sample])
	r = n0 / n1
	for sample in r1 :
		mr[experiment.Sample==sample] = x
	for sample in r2 :
		mr[experiment.Sample==sample] = 1+r-r*x
	return experiment.Exp_wBinIt(mr) / experiment.weightOscBF_binned - 1

def SingleRing_PID(x,experiment):
	if experiment.Experiment == 'SuperK-Gd' or experiment.Experiment == 'SKIV' or experiment.Experiment == 'SuperK_Htag' or experiment.Experiment == 'SuperK_Gdtag':
		e = [0,1,2,3,7,8,9]
		mu = [4,5,10,11]
	else:
		e = [0,1,2,7,8]
		mu = [3,4,5,9]
	mr = np.ones(experiment.NumberOfEvents)
	n0 = 0
	n1 = 0
	for sample in e:
		n0 += np.sum(experiment.Weight[experiment.Sample==sample])
	for sample in mu:
		n1 += np.sum(experiment.Weight[experiment.Sample==sample])
	r = n0 / n1
	for sample in e:
		mr[experiment.Sample==sample] = x
	for sample in mu:
		mr[experiment.Sample==sample] = 1+r-r*x
	return experiment.Exp_wBinIt(mr) / experiment.weightOscBF_binned - 1

def MultiRing_PID(x,experiment):
	if experiment.Experiment == 'SuperK-Gd' or experiment.Experiment == 'SKIV' or experiment.Experiment == 'SuperK_Htag' or experiment.Experiment == 'SuperK_Gdtag':
		e = [12,13,15]
		mu = [14]
	else:
		e = [10,11,13]
		mu = [12]
	mr = np.ones(experiment.NumberOfEvents)
	n0 = 0
	n1 = 0
	for sample in e:
		n0 += np.sum(experiment.Weight[experiment.Sample==sample])
	for sample in mu:
		n1 += np.sum(experiment.Weight[experiment.Sample==sample])
	r = n0 / n1
	for sample in e:
		mr[experiment.Sample==sample] = x
	for sample in mu:
		mr[experiment.Sample==sample] = 1+r-r*x
	return experiment.Exp_wBinIt(mr) / experiment.weightOscBF_binned - 1

def NeutronTagging(x,experiment):
	nn = np.ones(experiment.NumberOfEvents)
	if experiment.Experiment == 'SuperK-Gd' or experiment.Experiment == 'SKIV' or experiment.Experiment == 'SuperK_Htag' or experiment.Experiment == 'SuperK_Gdtag':
		n0 = np.sum(experiment.Neutron==0)
		n1 = np.sum(experiment.Neutron>0)
		r = n0 / n1
		nn[experiment.Neutron==0] = x 
		nn[experiment.Neutron>0] = 1+r-r*x
		return experiment.Exp_wBinIt(nn) / experiment.weightOscBF_binned - 1
	else:
		return 0

def DecayETagging(x,experiment):
	mue = np.ones(experiment.NumberOfEvents)
	n0 = np.sum(experiment.DecayE==0)
	n1 = np.sum(experiment.DecayE==1)
	n2 = np.sum(experiment.DecayE>1)
	N = n0 + n1 + n2
	r0 = n0 / N
	r1 = n1 / N
	r2 = n2 / N
	rx1 = x*r1 + 2*(1-x)*r2
	rx2 = x*x*r2 + 2*(1-x)*r2
	rx0 = 1 - rx1 - rx2
	mue[experiment.DecayE==0] = rx0 / r0
	mue[experiment.DecayE==1] = rx1 / r1
	mue[experiment.DecayE>1] = rx2 / r2
	return experiment.Exp_wBinIt(mue) / experiment.weightOscBF_binned - 1
