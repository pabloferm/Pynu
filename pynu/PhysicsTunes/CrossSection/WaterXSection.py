import numpy as np

##########################
#  Water Cross-section   #
##########################


class WaterXSection:
    r"""Class containing the tunes for the neutrino-water cross section. Note that there are some dependencies on the NEUT interaction mode definition.

    Each tune returns a per-event factor: `x` for the events it acts on, 1 elsewhere.
    """

    def XSecNuTau(self, experiment, x):
        r"""Method for modifying the $\nu_\tau$ cross-section normalization.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        tau = np.ones(experiment.NumberOfEvents)
        tau[np.abs(experiment.nuPDG) == 16] = x
        return tau

    def NCoverCC(self, experiment, x):
        nc = np.ones(experiment.NumberOfEvents)
        nc[experiment.CC == 0] = x
        return nc

    def NCHad(self, experiment, x):
        nc = np.ones(experiment.NumberOfEvents)
        nc[experiment.CC == 0] = x
        return nc

    def DIS(self, experiment, x):
        w = np.ones(experiment.NumberOfEvents)
        cond = np.abs(experiment.Mode) > 25 * experiment.CC
        w[cond] = x
        return w

    # NEUT 2p2h (meson exchange current) is |Mode|==2; nu=+2, nubar=-2.
    def CC_2p2h(self, experiment, x):
        cc_2p2h = np.ones(experiment.NumberOfEvents)
        cc_2p2h[np.abs(experiment.Mode) == 2] = x
        return cc_2p2h

    def CC_2p2hNuBarNu(self, experiment, x):
        cc_2p2h = np.ones(experiment.NumberOfEvents)
        cc_2p2h[experiment.Mode == -2] = x
        return cc_2p2h

    def CC_2p2hMuE(self, experiment, x):
        cc_2p2h = np.ones(experiment.NumberOfEvents)
        cond = (np.abs(experiment.Mode) == 2) * (np.abs(experiment.nuPDG) == 14)
        cc_2p2h[cond] = x
        return cc_2p2h

    def CCQE(self, experiment, x):
        ccqe = np.ones(experiment.NumberOfEvents)
        ccqe[np.abs(experiment.Mode) == 1] = x
        return ccqe

    def CCQENuBarNu(self, experiment, x):
        ccqe = np.ones(experiment.NumberOfEvents)
        ccqe[experiment.Mode == -1] = x
        return ccqe

    def CCQEMuE(self, experiment, x):
        ccqe = np.ones(experiment.NumberOfEvents)
        cond = (np.abs(experiment.Mode) == 1) * (np.abs(experiment.nuPDG) == 14)
        ccqe[cond] = x
        return ccqe

    def CC1Pi_Pi0Pi(self, experiment, x):
        ccpi = np.ones(experiment.NumberOfEvents)
        ccpi[np.abs(experiment.Mode) == 12] = x
        return ccpi

    def CC1Pi_NuBarNuE(self, experiment, x):
        ccpi = np.ones(experiment.NumberOfEvents)
        cond = (
            (np.abs(experiment.Mode) > 10)
            * (np.abs(experiment.Mode) < 17)
            * (experiment.nuPDG == -12)
        )
        ccpi[cond] = x
        return ccpi

    def CC1Pi_NuBarNuMu(self, experiment, x):
        ccpi = np.ones(experiment.NumberOfEvents)
        cond = (
            (np.abs(experiment.Mode) > 10)
            * (np.abs(experiment.Mode) < 17)
            * (experiment.nuPDG == -14)
        )
        ccpi[cond] = x
        return ccpi

    def CC1PiProduction(self, experiment, x):
        ccpi = np.ones(experiment.NumberOfEvents)
        cond = (np.abs(experiment.Mode) > 10) * (np.abs(experiment.Mode) < 17)
        ccpi[cond] = x
        return ccpi

    def CohPiProduction(self, experiment, x):
        ccpi = np.ones(experiment.NumberOfEvents)
        ccpi[np.abs(experiment.Mode) == 16] = x
        return ccpi
