from PhysicsTunes import Tune
import numpy as np

import sys

sys.path.append("../")

##########################
#  Water Cross-section   #
##########################


class WaterXSection(Tune):
    r"""Class containing the tunes for the neutrino-water cross section. Note that there are some dependencies on the NEUT interaction mode definition."""

    def XSecNuTau(self, experiment, x):
        r"""Method for modifying the $\nu_\tau$ cross-section normalization.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if self._unphysical_value(x): return 1e3
        tau = np.ones(experiment.NumberOfEvents)
        tau[np.abs(experiment.nuPDG) == 16] = x
        return tau

    def diff_XSecNuTau(self, experiment, x):
        r"""Method for computing the derivative of the weights of the $\nu_\tau$ cross-section normalization w.r.t. the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.

        Returns:
            Numpy.array or float with the derivative of the `XSecNuTau` weights.
        """
        if self._unphysical_value(x): return 0
        tau = np.zeros(experiment.NumberOfEvents)
        tau[np.abs(experiment.nuPDG) == 16] = 1
        return tau

    def NCoverCC(self, experiment, x):
        if self._unphysical_value(x): return 1e3
        nc = np.ones(experiment.NumberOfEvents)
        nc[experiment.CC == 0] = x
        return nc

    def diff_NCoverCC(self, experiment, x):
        if self._unphysical_value(x): return 0
        nc = np.zeros(experiment.NumberOfEvents)
        nc[experiment.CC == 0] = 1
        return nc

    def AxialMass(self, experiment, x):
        if self._unphysical_value(x): return 1e3
        cc = np.ones(experiment.NumberOfEvents)
        cc[experiment.CC == 1] = 1 + 0.042 * (x - 1) * 1.05 * np.log10(
            experiment.ETrue[experiment.CC == 1]
        )
        return cc

    def diff_AxialMass(self, experiment, x):
        if self._unphysical_value(x): return 0
        cc = np.zeros(experiment.NumberOfEvents)
        cc[experiment.CC == 1] = (
            0.042 * 1.05 * np.log10(experiment.ETrue[experiment.CC == 1])
        )
        return cc

    def NCHad(self, experiment, x):
        if self._unphysical_value(x): return 1e3
        nc = np.ones(experiment.NumberOfEvents)
        nc[experiment.CC == 0] = x
        return nc

    def diff_NCHad(self, experiment, x):
        if self._unphysical_value(x): return 0
        nc = np.zeros(experiment.NumberOfEvents)
        nc[experiment.CC == 0] = 1
        return nc

    def DIS(self, experiment, x):
        if self._unphysical_value(x): return 1e3
        dis = np.ones(experiment.NumberOfEvents)
        cond = np.abs(experiment.Mode) > 25 * experiment.CC
        dis[cond] = x
        return dis

    def diff_DIS(self, experiment, x):
        if self._unphysical_value(x): return 0
        dis = np.zeros(experiment.NumberOfEvents)
        cond = np.abs(experiment.Mode) > 25 * experiment.CC
        dis[cond] = 1
        return dis

    def CCQE(self, experiment, x):
        if self._unphysical_value(x): return 1e3
        ccqe = np.ones(experiment.NumberOfEvents)
        ccqe[np.abs(experiment.Mode) == 1] = x
        return ccqe

    def diff_CCQE(self, experiment, x):
        if self._unphysical_value(x): return 0
        ccqe = np.zeros(experiment.NumberOfEvents)
        ccqe[np.abs(experiment.Mode) == 1] = 1
        return ccqe

    def CCQENuBarNu(self, experiment, x):
        if self._unphysical_value(x): return 1e3
        ccqe = np.ones(experiment.NumberOfEvents)
        ccqe[experiment.Mode == -1] = x
        return ccqe

    def diff_CCQENuBarNu(self, experiment, x):
        if self._unphysical_value(x): return 0
        ccqe = np.zeros(experiment.NumberOfEvents)
        ccqe[experiment.Mode == -1] = 1
        return ccqe

    def CCQEMuE(self, experiment, x):
        if self._unphysical_value(x): return 1e3
        ccqe = np.ones(experiment.NumberOfEvents)
        cond = (np.abs(experiment.Mode) == 1) * (np.abs(experiment.nuPDG) == 14)
        ccqe[cond] = x
        return ccqe

    def diff_CCQEMuE(self, experiment, x):
        if self._unphysical_value(x): return 0
        ccqe = np.zeros(experiment.NumberOfEvents)
        cond = (np.abs(experiment.Mode) == 1) * (np.abs(experiment.nuPDG) == 14)
        ccqe[cond] = 1
        return ccqe

    def CC1Pi_Pi0Pi(self, experiment, x):
        if self._unphysical_value(x): return 1e3
        ccpi = np.ones(experiment.NumberOfEvents)
        ccpi[np.abs(experiment.Mode) == 12] = x
        return ccpi

    def diff_CC1Pi_Pi0Pi(self, experiment, x):
        if self._unphysical_value(x): return 0
        ccpi = np.zeros(experiment.NumberOfEvents)
        ccpi[np.abs(experiment.Mode) == 12] = 1
        return ccpi

    def CC1Pi_NuBarNuE(self, experiment, x):
        if self._unphysical_value(x): return 1e3
        ccpi = np.ones(experiment.NumberOfEvents)
        cond = (
            (np.abs(experiment.Mode) > 10)
            * (np.abs(experiment.Mode) < 17)
            * (experiment.nuPDG == -12)
        )
        ccpi[cond] = x
        return ccpi

    def diff_CC1Pi_NuBarNuE(self, experiment, x):
        if self._unphysical_value(x): return 0
        ccpi = np.zeros(experiment.NumberOfEvents)
        cond = (
            (np.abs(experiment.Mode) > 10)
            * (np.abs(experiment.Mode) < 17)
            * (experiment.nuPDG == -12)
        )
        ccpi[cond] = 1
        return ccpi

    def CC1Pi_NuBarNuMu(self, experiment, x):
        if self._unphysical_value(x): return 1e3
        ccpi = np.ones(experiment.NumberOfEvents)
        cond = (
            (np.abs(experiment.Mode) > 10)
            * (np.abs(experiment.Mode) < 17)
            * (experiment.nuPDG == -14)
        )
        ccpi[cond] = x
        return ccpi

    def diff_CC1Pi_NuBarNuMu(self, experiment, x):
        if self._unphysical_value(x): return 0
        ccpi = np.zeros(experiment.NumberOfEvents)
        cond = (
            (np.abs(experiment.Mode) > 10)
            * (np.abs(experiment.Mode) < 17)
            * (experiment.nuPDG == -14)
        )
        ccpi[cond] = 1
        return ccpi

    def CC1PiProduction(self, experiment, x):
        if self._unphysical_value(x): return 1e3
        ccpi = np.ones(experiment.NumberOfEvents)
        cond = (np.abs(experiment.Mode) > 10) * (np.abs(experiment.Mode) < 17)
        ccpi[cond] = x
        return ccpi

    def diff_CC1PiProduction(self, experiment, x):
        if self._unphysical_value(x): return 0
        ccpi = np.zeros(experiment.NumberOfEvents)
        cond = (np.abs(experiment.Mode) > 10) * (np.abs(experiment.Mode) < 17)
        ccpi[cond] = 1
        return ccpi

    def CohPiProduction(self, experiment, x):
        if self._unphysical_value(x): return 1e3
        ccpi = np.ones(experiment.NumberOfEvents)
        ccpi[np.abs(experiment.Mode) == 16] = x
        return ccpi

    def diff_CohPiProduction(self, experiment, x):
        if self._unphysical_value(x): return 0
        ccpi = np.zeros(experiment.NumberOfEvents)
        ccpi[np.abs(experiment.Mode) == 16] = 1
        return ccpi
