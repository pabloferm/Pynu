from .JacobianSKDetector import *

####################
###### SK-Gd #######
####################


def diff_SKGd_SKEnergyScale(x, experiment):
    return diff_SKEnergyScale(x, experiment)


def diff_SKGd_FCPCSeparation(x, experiment):
    return diff_FCPCSeparation(x, experiment)


def diff_SKGd_FCReduction(x, experiment):
    return diff_FCReduction(x, experiment)


def diff_SKGd_PCReduction(x, experiment):
    return diff_PCReduction(x, experiment)


def diff_SKGd_FiducialVolume(x, experiment):
    return diff_FiducialVolume(x, experiment)


def diff_SKGd_SubGeV2ringPi0(x, experiment):
    return diff_SubGeV2ringPi0(x, experiment)


def diff_SKGd_SubGeV1ringPi0(x, experiment):
    return diff_SubGeV1ringPi0(x, experiment)


def diff_SKGd_MultiRing_NuNuBarSeparation(x, experiment):
    return diff_MultiRing_NuNuBarSeparation(x, experiment)


def diff_SKGd_MultiRing_EMuSeparation(x, experiment):
    return diff_MultiRing_EMuSeparation(x, experiment)


def diff_SKGd_MultiRing_EOtherSeparation(x, experiment):
    return diff_MultiRing_EOtherSeparation(x, experiment)


def diff_SKGd_PC_StopThruSeparation(x, experiment):
    return diff_PC_StopThruSeparation(x, experiment)


def diff_SKGd_Pi0_RingSeparation(x, experiment):
    return diff_Pi0_RingSeparation(x, experiment)


def diff_SKGd_E_RingSeparation(x, experiment):
    return diff_E_RingSeparation(x, experiment)


def diff_SKGd_Mu_RingSeparation(x, experiment):
    return diff_Mu_RingSeparation(x, experiment)


def diff_SKGd_SingleRing_PID(x, experiment):
    return diff_SingleRing_PID(x, experiment)


def diff_SKGd_MultiRing_PID(x, experiment):
    return diff_MultiRing_PID(x, experiment)


def diff_SKGd_NeutronTagging(x, experiment):
    return diff_NeutronTagging(x, experiment)


def diff_SKGd_DecayETagging(x, experiment):
    return diff_DecayETagging(x, experiment)
