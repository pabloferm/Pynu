import numpy as np

####################
#  IceCube Upgrade #
####################

def IceAbsorption(x, experiment):
	xx = x - 1
	d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
	experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
	experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
	experiment.ExpFracNC * experiment.ice_absorption['NC']
	return xx * d

def IceScattering(x, experiment):
	xx = x - 1
	d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
	experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
	experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
	experiment.ExpFracNC * experiment.ice_absorption['NC']
	return xx * d

def OffSet(x, experiment):
	xx = x
	d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
	experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
	experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
	experiment.ExpFracNC * experiment.ice_absorption['NC']
	return xx * d

def OptEffHeadon(x, experiment):
	xx = x
	d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
	experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
	experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
	experiment.ExpFracNC * experiment.ice_absorption['NC']
	return xx * d

def OptEffLateral(x, experiment):
	xx = x - 25
	d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
	experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
	experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
	experiment.ExpFracNC * experiment.ice_absorption['NC']
	return xx * d

def OptEffOverall(x, experiment):
	xx = x - 1
	d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
	experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
	experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
	experiment.ExpFracNC * experiment.ice_absorption['NC']
	return xx * d

def CoinFraction(x, experiment):
	xx = x
	d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
	experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
	experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
	experiment.ExpFracNC * experiment.ice_absorption['NC']
	return xx * d