from PhysicsTunes import Tune
import numpy as np

import sys
sys.path.append('../')

####################
#  IceCube Upgrade #
####################


class ICUpgrade(Tune):

    def IceAbsorption(self, experiment, x):
        xx = x - 1
        d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
            experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
            experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
            experiment.ExpFracNC * experiment.ice_absorption['NC']
        return xx * d

    def Diff_IceAbsorption(self, experiment, x):
        d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
            experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
            experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
            experiment.ExpFracNC * experiment.ice_absorption['NC']
        return d

    def IceScattering(self, experiment, x):
        xx = x - 1
        d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
            experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
            experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
            experiment.ExpFracNC * experiment.ice_absorption['NC']
        return xx * d

    def Diff_IceScattering(self, experiment, x):
        d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
            experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
            experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
            experiment.ExpFracNC * experiment.ice_absorption['NC']
        return d

    def OffSet(self, experiment, x):
        xx = x
        d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
            experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
            experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
            experiment.ExpFracNC * experiment.ice_absorption['NC']
        return xx * d

    def Diff_OffSet(self, experiment, x):
        d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
            experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
            experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
            experiment.ExpFracNC * experiment.ice_absorption['NC']
        return d

    def OptEffHeadon(self, experiment, x):
        xx = x
        d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
            experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
            experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
            experiment.ExpFracNC * experiment.ice_absorption['NC']
        return xx * d

    def Diff_OptEffHeadon(self, experiment, x):
        d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
            experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
            experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
            experiment.ExpFracNC * experiment.ice_absorption['NC']
        return d

    def OptEffLateral(self, experiment, x):
        xx = x - 25
        d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
            experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
            experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
            experiment.ExpFracNC * experiment.ice_absorption['NC']
        return xx * d

    def Diff_OptEffLateral(self, experiment, x):
        d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
            experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
            experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
            experiment.ExpFracNC * experiment.ice_absorption['NC']
        return d

    def OptEffOverall(self, experiment, x):
        xx = x - 1
        d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
            experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
            experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
            experiment.ExpFracNC * experiment.ice_absorption['NC']
        return xx * d

    def Diff_OptEffOverall(self, experiment, x):
        d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
            experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
            experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
            experiment.ExpFracNC * experiment.ice_absorption['NC']
        return d

    def CoinFraction(self, experiment, x):
        xx = x
        d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
            experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
            experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
            experiment.ExpFracNC * experiment.ice_absorption['NC']
        return xx * d

    def Diff_CoinFraction(self, experiment, x):
        d = experiment.ExpFracNuECC * experiment.ice_absorption['nueCC'] + \
            experiment.ExpFracNuMuCC * experiment.ice_absorption['numuCC'] + \
            experiment.ExpFracNuTauCC * experiment.ice_absorption['nutauCC'] + \
            experiment.ExpFracNC * experiment.ice_absorption['NC']
        return d
