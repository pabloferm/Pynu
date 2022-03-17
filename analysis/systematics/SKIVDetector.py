from Systematics.SKDetector import *

####################
###### SK-IV #######
####################

def SKIV_SKEnergyScale(x, experiment):
	return SKEnergyScale(x,experiment)

def SKIV_FCPCSeparation(x,experiment):
	return FCPCSeparation(x,experiment)

def SKIV_FCReduction(x,experiment):
	return FCReduction(x,experiment)

def SKIV_PCReduction(x,experiment):
	return PCReduction(x,experiment)

def SKIV_FiducialVolume(x,experiment):
	return FiducialVolume(x,experiment)

def SKIV_SubGeV2ringPi0(x,experiment):
	return SubGeV2ringPi0(x,experiment)

def SKIV_SubGeV1ringPi0(x,experiment):
	return SubGeV1ringPi0(x,experiment)

def SKIV_MultiRing_NuNuBarSeparation(x,experiment):
	return MultiRing_NuNuBarSeparation(x,experiment)

def SKIV_MultiRing_EMuSeparation(x,experiment):
	return MultiRing_EMuSeparation(x,experiment)

def SKIV_MultiRing_EOtherSeparation(x,experiment):
	return MultiRing_EOtherSeparation(x,experiment)

def SKIV_PC_StopThruSeparation(x,experiment):
	return PC_StopThruSeparation(x,experiment)

def SKIV_Pi0_RingSeparation(x,experiment):
	return Pi0_RingSeparation(x,experiment)

def SKIV_E_RingSeparation(x,experiment):
	return E_RingSeparation(x,experiment)

def SKIV_Mu_RingSeparation(x,experiment):
	return Mu_RingSeparation(x,experiment)

def SKIV_SingleRing_PID(x,experiment):
	return SingleRing_PID(x,experiment)

def SKIV_MultiRing_PID(x,experiment):
	return MultiRing_PID(x,experiment)

def SKIV_NeutronTagging(x,experiment):
	return NeutronTagging(x,experiment)

def SKIV_DecayETagging(x,experiment):
	return DecayETagging(x,experiment)
