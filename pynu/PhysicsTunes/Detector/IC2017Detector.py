from PhysicsTunes import Tune

import sys

sys.path.append("../")

####################
#  IceCube Upgrade #
####################


class IC2017(Tune):
    def IceAbsorption(self, experiment, x):
        xx = x - 1
        d = (
            experiment.ExpFracNuECC * experiment.ice_absorption["nueCC"]
            + experiment.ExpFracNuMuCC * experiment.ice_absorption["numuCC"]
            + experiment.ExpFracNuTauCC * experiment.ice_absorption["nutauCC"]
            + experiment.ExpFracNC * experiment.ice_absorption["NC"]
        )
        return xx * d * 100

    def diff_IceAbsorption(self, experiment, x):
        d = (
            experiment.ExpFracNuECC * experiment.ice_absorption["nueCC"]
            + experiment.ExpFracNuMuCC * experiment.ice_absorption["numuCC"]
            + experiment.ExpFracNuTauCC * experiment.ice_absorption["nutauCC"]
            + experiment.ExpFracNC * experiment.ice_absorption["NC"]
        )
        return d * 100

    def IceScattering(self, experiment, x):
        xx = x - 1
        d = (
            experiment.ExpFracNuECC * experiment.ice_absorption["nueCC"]
            + experiment.ExpFracNuMuCC * experiment.ice_absorption["numuCC"]
            + experiment.ExpFracNuTauCC * experiment.ice_absorption["nutauCC"]
            + experiment.ExpFracNC * experiment.ice_absorption["NC"]
        )
        return xx * d * 100

    def diff_IceScattering(self, experiment, x):
        d = (
            experiment.ExpFracNuECC * experiment.ice_absorption["nueCC"]
            + experiment.ExpFracNuMuCC * experiment.ice_absorption["numuCC"]
            + experiment.ExpFracNuTauCC * experiment.ice_absorption["nutauCC"]
            + experiment.ExpFracNC * experiment.ice_absorption["NC"]
        )
        return d * 100

    def OffSet(self, experiment, x):
        xx = x
        d = (
            experiment.ExpFracNuECC * experiment.ice_absorption["nueCC"]
            + experiment.ExpFracNuMuCC * experiment.ice_absorption["numuCC"]
            + experiment.ExpFracNuTauCC * experiment.ice_absorption["nutauCC"]
            + experiment.ExpFracNC * experiment.ice_absorption["NC"]
        )
        return xx * d

    def diff_OffSet(self, experiment, x):
        d = (
            experiment.ExpFracNuECC * experiment.ice_absorption["nueCC"]
            + experiment.ExpFracNuMuCC * experiment.ice_absorption["numuCC"]
            + experiment.ExpFracNuTauCC * experiment.ice_absorption["nutauCC"]
            + experiment.ExpFracNC * experiment.ice_absorption["NC"]
        )
        return d

    def OptEffHeadon(self, experiment, x):
        xx = x
        d = (
            experiment.ExpFracNuECC * experiment.ice_absorption["nueCC"]
            + experiment.ExpFracNuMuCC * experiment.ice_absorption["numuCC"]
            + experiment.ExpFracNuTauCC * experiment.ice_absorption["nutauCC"]
            + experiment.ExpFracNC * experiment.ice_absorption["NC"]
        )
        return xx * d

    def diff_OptEffHeadon(self, experiment, x):
        d = (
            experiment.ExpFracNuECC * experiment.ice_absorption["nueCC"]
            + experiment.ExpFracNuMuCC * experiment.ice_absorption["numuCC"]
            + experiment.ExpFracNuTauCC * experiment.ice_absorption["nutauCC"]
            + experiment.ExpFracNC * experiment.ice_absorption["NC"]
        )
        return d

    def OptEffLateral(self, experiment, x):
        xx = x * 10 + 25
        d = (
            experiment.ExpFracNuECC * experiment.ice_absorption["nueCC"]
            + experiment.ExpFracNuMuCC * experiment.ice_absorption["numuCC"]
            + experiment.ExpFracNuTauCC * experiment.ice_absorption["nutauCC"]
            + experiment.ExpFracNC * experiment.ice_absorption["NC"]
        )
        return xx * d

    def diff_OptEffLateral(self, experiment, x):
        d = (
            experiment.ExpFracNuECC * experiment.ice_absorption["nueCC"]
            + experiment.ExpFracNuMuCC * experiment.ice_absorption["numuCC"]
            + experiment.ExpFracNuTauCC * experiment.ice_absorption["nutauCC"]
            + experiment.ExpFracNC * experiment.ice_absorption["NC"]
        )
        return d * 10

    def OptEffOverall(self, experiment, x):
        xx = x
        d = (
            experiment.ExpFracNuECC * experiment.ice_absorption["nueCC"]
            + experiment.ExpFracNuMuCC * experiment.ice_absorption["numuCC"]
            + experiment.ExpFracNuTauCC * experiment.ice_absorption["nutauCC"]
            + experiment.ExpFracNC * experiment.ice_absorption["NC"]
        )
        return xx * d

    def diff_OptEffOverall(self, experiment, x):
        d = (
            experiment.ExpFracNuECC * experiment.ice_absorption["nueCC"]
            + experiment.ExpFracNuMuCC * experiment.ice_absorption["numuCC"]
            + experiment.ExpFracNuTauCC * experiment.ice_absorption["nutauCC"]
            + experiment.ExpFracNC * experiment.ice_absorption["NC"]
        )
        return d

    def NeutralC(self, experiment, x):
        xx = x
        d = experiment.ExpFracNC * experiment.ice_absorption["NC"]
        return d * xx

    def diff_NeutralC(self, experiment, x):
        d = experiment.ExpFracNC * experiment.ice_absorption["NC"]
        return d
