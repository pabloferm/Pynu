from Systematics.SKDetector import *

####################
###### SK-Gd #######
####################

def SKGd_SKEnergyScale(x, experiment):
	return SKEnergyScale(x,experiment)

def SKGd_FCPCSeparation(x,experiment):
	return FCPCSeparation(x,experiment)

def SKGd_FiducialVolume(x,experiment):
	return FiducialVolume(x,experiment)

def SKGd_FCReduction(x,experiment):
	return FCReduction(x,experiment)

def SKGd_PCReduction(x,experiment):
	return PCReduction(x,experiment)

def SKGd_SubGeV2ringPi0(x,experiment):
	return SubGeV2ringPi0(x,experiment)

def SKGd_SubGeV1ringPi0(x,experiment):
	return SubGeV1ringPi0(x,experiment)

def SKGd_MultiRing_NuNuBarSeparation(x,experiment):
	return MultiRing_NuNuBarSeparation(x,experiment)

def SKGd_MultiRing_EMuSeparation(x,experiment):
	return MultiRing_EMuSeparation(x,experiment)

def SKGd_MultiRing_EOtherSeparation(x,experiment):
	return MultiRing_EOtherSeparation(x,experiment)

def SKGd_PC_StopThruSeparation(x,experiment):
	return PC_StopThruSeparation(x,experiment)

def SKGd_Pi0_RingSeparation(x,experiment):
	return Pi0_RingSeparation(x,experiment)

def SKGd_E_RingSeparation(x,experiment):
	return E_RingSeparation(x,experiment)

def SKGd_Mu_RingSeparation(x,experiment):
	return Mu_RingSeparation(x,experiment)

def SKGd_SingleRing_PID(x,experiment):
	return SingleRing_PID(x,experiment)

def SKGd_MultiRing_PID(x,experiment):
	return MultiRing_PID(x,experiment)

def SKGd_NeutronTagging(x,experiment):
	return NeutronTagging(x,experiment)

def SKGd_DecayETagging(x,experiment):
	return DecayETagging(x,experiment)
