from PhysicsTunes import Tune
import numpy as np

import sys
sys.path.append('../')

####################
# Atmospheric flux #
####################


class AtmosphericFlux(Tune):

    def FluxNormalization(self, experiment, x):
        return x

    def Diff_FluxNormalization(self, experiment, x):
        return 1

    def FluxNormalization_Below1GeV(self, experiment, x):
        nev = np.ones(experiment.NumberOfEvents)
        nev[experiment.ETrue < 1] = x
        return nev

    def Diff_FluxNormalization_Below1GeV(self, experiment, x):
        nev = np.zeros(experiment.NumberOfEvents)
        nev[experiment.ETrue < 1] = 1
        return nev

    def FluxNormalization_Above1GeV(self, experiment, x):
        nev = np.ones(experiment.NumberOfEvents)
        nev[experiment.ETrue > 1] = x
        return nev

    def Diff_FluxNormalization_Above1GeV(self, experiment, x):
        nev = np.zeros(experiment.NumberOfEvents)
        nev[experiment.ETrue > 1] = 1
        return nev

    def FluxTilt(self, experiment, x):
        E0Gam = 10  # GeV
        nev = (experiment.ETrue / E0Gam)**x
        return nev

    def Diff_FluxTilt(self, experiment, x):
        E0Gam = 10  # GeV
        nev = (experiment.ETrue / E0Gam)**x * np.log(experiment.ETrue / E0Gam)
        return nev

    def NuNuBarRatio(self, experiment, x):
        nnbar = np.ones(experiment.NumberOfEvents)
        nnbar[experiment.nuPDG < 0] = x
        return nnbar

    def Diff_NuNuBarRatio(self, experiment, x):
        nnbar = np.zeros(experiment.NumberOfEvents)
        nnbar[experiment.nuPDG < 0] = 1
        return nnbar

    def FlavorRatio(self, experiment, x):
        eovermu = np.ones(experiment.NumberOfEvents)
        eovermu[np.abs(experiment.nuPDG) == 12] = x
        return eovermu

    def Diff_FlavorRatio(self, experiment, x):
        eovermu = np.zeros(experiment.NumberOfEvents)
        eovermu[abs(experiment.nuPDG) == 12] = 1
        return eovermu

    def ZenithFluxUp(self, experiment, x):
        zenith = np.ones(experiment.NumberOfEvents)
        zenith[experiment.CosZTrue >= 0] = zenith[experiment.CosZTrue >=
                                                  0] - x * np.tanh(experiment.CosZTrue[experiment.CosZTrue >= 0])**2
        return zenith

    def Diff_ZenithFluxUp(self, experiment, x):
        zenith = np.zeros(experiment.NumberOfEvents)
        zenith[experiment.CosZTrue >= 0] = - \
            np.tanh(experiment.CosZTrue[experiment.CosZTrue >= 0])**2
        return zenith

    def ZenithFluxDown(self, experiment, x):
        zenith = np.ones(experiment.NumberOfEvents)
        zenith[experiment.CosZTrue < 0] = zenith[experiment.CosZTrue < 0] - \
            x * np.tanh(experiment.CosZTrue[experiment.CosZTrue < 0])**2
        return zenith

    def Diff_ZenithFluxDown(self, experiment, x):
        zenith = np.zeros(experiment.NumberOfEvents)
        zenith[experiment.CosZTrue < 0] = - \
            np.tanh(experiment.CosZTrue[experiment.CosZTrue < 0])**2
        return zenith
