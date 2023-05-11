from .JacobianSKDetector import *

####################
###### SK-IV #######
####################


def diff_SKIV_SKEnergyScale(x, experiment):
    return diff_SKEnergyScale(x, experiment)


def diff_SKIV_FCPCSeparation(x, experiment):
    return diff_FCPCSeparation(x, experiment)


def diff_SKIV_FCReduction(x, experiment):
    return diff_FCReduction(x, experiment)


def diff_SKIV_PCReduction(x, experiment):
    return diff_PCReduction(x, experiment)


def diff_SKIV_SubGeV2ringPi0(x, experiment):
    return diff_SubGeV2ringPi0(x, experiment)


def diff_SKIV_FiducialVolume(x, experiment):
    return diff_FiducialVolume(x, experiment)


def diff_SKIV_SubGeV1ringPi0(x, experiment):
    return diff_SubGeV1ringPi0(x, experiment)


def diff_SKIV_MultiRing_NuNuBarSeparation(x, experiment):
    return diff_MultiRing_NuNuBarSeparation(x, experiment)


def diff_SKIV_MultiRing_EMuSeparation(x, experiment):
    return diff_MultiRing_EMuSeparation(x, experiment)


def diff_SKIV_MultiRing_EOtherSeparation(x, experiment):
    return diff_MultiRing_EOtherSeparation(x, experiment)


def diff_SKIV_PC_StopThruSeparation(x, experiment):
    return diff_PC_StopThruSeparation(x, experiment)


def diff_SKIV_Pi0_RingSeparation(x, experiment):
    return diff_Pi0_RingSeparation(x, experiment)


def diff_SKIV_E_RingSeparation(x, experiment):
    return diff_E_RingSeparation(x, experiment)


def diff_SKIV_Mu_RingSeparation(x, experiment):
    return diff_Mu_RingSeparation(x, experiment)


def diff_SKIV_SingleRing_PID(x, experiment):
    return diff_SingleRing_PID(x, experiment)


def diff_SKIV_MultiRing_PID(x, experiment):
    return diff_MultiRing_PID(x, experiment)


def diff_SKIV_NeutronTagging(x, experiment):
    return diff_NeutronTagging(x, experiment)


def diff_SKIV_DecayETagging(x, experiment):
    return diff_DecayETagging(x, experiment)
