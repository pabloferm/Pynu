import numpy as np

####################
#  IceCube Upgrade #
####################

def Diff_IceAbsorption(x, experiment):
	d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
	experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
	experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
	experiment.ExpFracNC * experiment.ice_absorption['NC']
	return d

def Diff_IceScattering(x, experiment):
	d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
	experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
	experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
	experiment.ExpFracNC * experiment.ice_absorption['NC']
	return d

def Diff_OffSet(x, experiment):
	d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
	experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
	experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
	experiment.ExpFracNC * experiment.ice_absorption['NC']
	return d

def Diff_OptEffHeadon(x, experiment):
	d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
	experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
	experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
	experiment.ExpFracNC * experiment.ice_absorption['NC']
	return d

def Diff_OptEffLateral(x, experiment):
	d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
	experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
	experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
	experiment.ExpFracNC * experiment.ice_absorption['NC']
	return d

def Diff_OptEffOverall(x, experiment):
	d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
	experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
	experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
	experiment.ExpFracNC * experiment.ice_absorption['NC']
	return d

def Diff_CoinFraction(x, experiment):
	d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
	experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
	experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
	experiment.ExpFracNC * experiment.ice_absorption['NC']
	return d