from PhysicsTunes import Tune

# from .SKDetector import SuperK
import numpy as np

import sys

sys.path.append("../")

############################################
###### Used for pheno combined MC-IV #######
############################################


class SuperK_Combined(Tune):
    def energy_scale(self, experiment, x):
        """See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.energy_scale`."""
        logging.info(f"Computing {experiment.Detector} energy scale tune.")
        return SuperK.energy_scale(experiment, x)

    def diff_energy_scale(self, experiment, x):
        """See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.diff_energy_scale`."""
        logging.info(f"Computing {experiment.Detector} energy scale tune derivative.")
        return SuperK.diff_energy_scale(experiment, x)

    def fiducial_volume(self, experiment, x):
        r"""Method changing the efficiency of the fiducial volume cut.
        NOTE: Currently, it applies a normalization factor on all events. More precise implementation coming soon.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        # # return 1e3 if self._unphysical_value(x) else x
        return x

    def diff_fiducial_volume(self, experiment, x):
        r"""Method for computing the derivative of the weights w.r.t. the tuning parameter of the fiducial volumen.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `fiducial_volume` weights.
        """
        # # return 0 if self._unphysical_value(x) else 1
        return 0

    def subgev_2ring_pi0(self, experiment, x):
        r"""Method changing the fraction of 2-ring $\pi^0$-like events.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        # if self._unphysical_value(x):
        #     return 1e3
        pi02r = np.ones(experiment.NumberOfEvents)
        pi02r[experiment.Sample == 6] = x
        return pi02r

    def diff_subgev_2ring_pi0(self, experiment, x):
        r"""Method for computing the derivative of the weights of the 2-ring $\pi^0$-like events w.r.t.
        the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `subgev_2ring_pi0` weights.
        """
        # if self._unphysical_value(x):
        #     return 0
        pi02r = np.zeros(experiment.NumberOfEvents)
        pi02r[experiment.Sample == 6] = 1
        return pi02r

    def fcpc_separation(self, experiment, x):
        r"""Method changing the efficiency of the fully and partially-contained events in SK.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        # logging.info(f"Entering {__name__}")
        # print(f"Entering {__name__} in combined")
        # if self._unphysical_value(x):
        #     return 1e3
        fcpc = np.ones(experiment.NumberOfEvents)

        pc = (experiment.Sample == 14) | (experiment.Sample == 15)
        um = (experiment.Sample >= 16) & (experiment.Sample <= 18)
        fc = np.logical_not((pc | um))

        fcpc[fc] = x

        wfc = np.sum(experiment.Weight[fc])
        wpc = np.sum(experiment.Weight[pc])
        y = ((wpc + wfc) - x * wfc) / wpc

        fcpc[pc] = y

        return fcpc

    def diff_fcpc_separation(self, experiment, x):
        r"""Method for computing the derivative of the weights of the fully and partially-contained events
        w.r.t. the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `fcpc_separation` weights.
        """
        # print(f"Entering {__name__}")
        # if self._unphysical_value(x):
        #     return 0
        fcpc = np.zeros(experiment.NumberOfEvents)

        pc = (experiment.Sample == 14) | (experiment.Sample == 15)
        um = (experiment.Sample >= 16) & (experiment.Sample <= 18)
        fc = np.logical_not((pc | um))

        fcpc[fc] = 1

        wfc = np.sum(experiment.Weight[fc])
        wpc = np.sum(experiment.Weight[pc])
        y = -wfc / wpc

        fcpc[pc] = y

        return fcpc

    def fc_reduction(self, experiment, x):
        r"""Method changing the efficiency of the fully-contained events reduction in SK.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        # if self._unphysical_value(x):
        #     return 1e3
        fc = np.ones(experiment.NumberOfEvents)
        pc = (experiment.Sample == 14) | (experiment.Sample == 15)
        um = (experiment.Sample >= 16) & (experiment.Sample <= 18)
        fc[np.logical_not((pc | um))] = x

        return fc

    def diff_fc_reduction(self, experiment, x):
        r"""Method for computing the derivative of the weights of the fully-contained events w.r.t.
        the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `fc_reduction` weights.
        """
        # if self._unphysical_value(x):
        #     return 0
        fc = np.zeros(experiment.NumberOfEvents)
        pc = (experiment.Sample == 14) | (experiment.Sample == 15)
        um = (experiment.Sample >= 16) & (experiment.Sample <= 18)
        fc[np.logical_not((pc | um))] = 1

        return fc

    def pc_reduction(self, experiment, x):
        r"""Method changing the efficiency of the partially-contained events reduction in SK.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        # if self._unphysical_value(x):
        #     return 1e3
        w = np.ones(experiment.NumberOfEvents)
        pc = (experiment.Sample == 14) | (experiment.Sample == 15)
        w[pc] = x
        return w

    def diff_pc_reduction(self, experiment, x):
        r"""Method for computing the derivative of the weights of the partially-contained events w.r.t.
        the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `pc_reduction` weights.
        """
        # if self._unphysical_value(x):
        #     return 0
        w = np.zeros(experiment.NumberOfEvents)
        pc = (experiment.Sample == 14) | (experiment.Sample == 15)
        w[pc] = 1
        return w

    def subgev_1ring_pi0(self, experiment, x):
        r"""Method changing the fraction of single-ring $\pi^0$-like events.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        # if self._unphysical_value(x):
        #     return 1e3
        pi01r = np.ones(experiment.NumberOfEvents)
        pi01r[experiment.Sample == 2] = x
        return pi01r

    def diff_subgev_1ring_pi0(self, experiment, x):
        r"""Method for computing the derivative of the weights of the single-ring $\pi^0$-like events w.r.t.
        the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `subgev_1ring_pi0` weights.
        """
        # if self._unphysical_value(x):
        #     return 0
        pi01r = np.zeros(experiment.NumberOfEvents)
        pi01r[experiment.Sample == 2] = 1
        return pi01r

    def mge_nonubkg(self, experiment, x):
        # if self._unphysical_value(x):
        #     return 1e3
        w = np.ones(experiment.NumberOfEvents)
        mge = (
            (experiment.Sample == 7)
            | (experiment.Sample == 8)
            | (experiment.Sample == 24)
            | (experiment.Sample == 25)
            | (experiment.Sample == 26)
        )
        w[mge] = x
        return w

    def diff_mge_nonubkg(self, experiment, x):
        # if self._unphysical_value(x):
        #     return 0
        w = np.zeros(experiment.NumberOfEvents)
        mge = (
            (experiment.Sample == 7)
            | (experiment.Sample == 8)
            | (experiment.Sample == 24)
            | (experiment.Sample == 25)
            | (experiment.Sample == 26)
        )
        w[mge] = 1
        return w

    def multiring_nunubar_separation(self, experiment, x):
        r"""Method changing the efficiency of neutrino-antineutrino separation in multi-ring events.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        # if self._unphysical_value(x):
        #     return 1e3
        mr = np.ones(experiment.NumberOfEvents)
        n0 = np.sum(experiment.Weight[experiment.Sample == 10])
        n1 = np.sum(experiment.Weight[experiment.Sample == 11])
        r = n0 / n1
        mr[experiment.Sample == 10] = x
        mr[experiment.Sample == 11] = 1 + r - r * x
        return mr

    def diff_multiring_nunubar_separation(self, experiment, x):
        r"""Method for computing the derivative of the weights of the multi-ring neutrino and antineutrino
        events w.r.t. the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `multiring_nunubar_separation` weights.
        """
        # if self._unphysical_value(x):
        #     return 0
        mr = np.zeros(experiment.NumberOfEvents)
        n0 = np.sum(experiment.Weight[experiment.Sample == 10])
        n1 = np.sum(experiment.Weight[experiment.Sample == 11])
        r = n0 / n1
        mr[experiment.Sample == 10] = 1
        mr[experiment.Sample == 11] = -r
        return mr

    def multiring_emu_separation(self, experiment, x):
        r"""Method changing the efficiency of electron-muon separation in multi-ring events.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        # if self._unphysical_value(x):
        #     return 1e3
        e0 = 10
        e1 = 11
        e2 = 13
        mu = 12
        mr = np.ones(experiment.NumberOfEvents)
        n0 = (
            np.sum(experiment.Weight[experiment.Sample == e0])
            + np.sum(experiment.Weight[experiment.Sample == e1])
            + np.sum(experiment.Weight[experiment.Sample == e2])
        )
        n1 = np.sum(experiment.Weight[experiment.Sample == mu])
        r = n0 / n1
        mr[experiment.Sample == e0] = x
        mr[experiment.Sample == e1] = x
        mr[experiment.Sample == e2] = x
        mr[experiment.Sample == mu] = 1 + r - r * x
        return mr

    def diff_multiring_emu_separation(self, experiment, x):
        r"""Method for computing the derivative of the weights of the multi-ring muon and electron (anti)neutrino
        events w.r.t. the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `multiring_emu_separation` weights.
        """
        # if self._unphysical_value(x):
        #     return 0
        e0 = 10
        e1 = 11
        e2 = 13
        mu = 12
        mr = np.zeros(experiment.NumberOfEvents)
        n0 = (
            np.sum(experiment.Weight[experiment.Sample == e0])
            + np.sum(experiment.Weight[experiment.Sample == e1])
            + np.sum(experiment.Weight[experiment.Sample == e2])
        )
        n1 = np.sum(experiment.Weight[experiment.Sample == mu])
        r = n0 / n1
        mr[experiment.Sample == e0] = 1
        mr[experiment.Sample == e1] = 1
        mr[experiment.Sample == e2] = 1
        mr[experiment.Sample == mu] = -r
        return mr

    def multiring_eother_separation(self, experiment, x):
        r"""Method changing the efficiency of electron neutrinos interacting charged-current and neutral-current
        interactions in multi-ring events.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        # if self._unphysical_value(x):
        #     return 1e3
        e0 = 10
        e1 = 11
        o0 = 13
        mr = np.ones(experiment.NumberOfEvents)
        n0 = np.sum(experiment.Weight[experiment.Sample == e0]) + np.sum(
            experiment.Weight[experiment.Sample == e1]
        )
        n1 = np.sum(experiment.Weight[experiment.Sample == o0])
        r = n0 / n1
        mr[experiment.Sample == e0] = x
        mr[experiment.Sample == e1] = x
        mr[experiment.Sample == o0] = 1 + r - r * x
        return mr

    def diff_multiring_eother_separation(self, experiment, x):
        r"""Method for computing the derivative of the weights of the multi-ring e-like events w.r.t. the
        tuning parameter separating between CC $\nu_e$ and NC $\nu$.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `multiring_eother_separation` weights.
        """
        # if self._unphysical_value(x):
        #     return 0
        e0 = 10
        e1 = 11
        o0 = 13
        mr = np.zeros(experiment.NumberOfEvents)
        n0 = np.sum(experiment.Weight[experiment.Sample == e0]) + np.sum(
            experiment.Weight[experiment.Sample == e1]
        )
        n1 = np.sum(experiment.Weight[experiment.Sample == o0])
        r = n0 / n1
        mr[experiment.Sample == e0] = 1
        mr[experiment.Sample == e1] = 1
        mr[experiment.Sample == o0] = -r
        return mr

    def pc_stopthru_separation(self, experiment, x):
        r"""Method changing the efficiency of pc-StopThru separation.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        # if self._unphysical_value(x):
        #     return 1e3
        pcs = 14
        pct = 15
        mr = np.ones(experiment.NumberOfEvents)
        n0 = np.sum(experiment.Weight[experiment.Sample == pcs])
        n1 = np.sum(experiment.Weight[experiment.Sample == pct])
        r = n0 / n1
        mr[experiment.Sample == pcs] = x
        mr[experiment.Sample == pct] = 1 + r - r * x
        return mr

    def diff_pc_stopthru_separation(self, experiment, x):
        r"""Method for computing the derivative of the weights of the pc and Stop Thru events w.r.t. the
        tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `pc_stopthru_separation` weights.
        """
        # if self._unphysical_value(x):
        #     return 0
        pcs = 14
        pct = 15
        mr = np.zeros(experiment.NumberOfEvents)
        n0 = np.sum(experiment.Weight[experiment.Sample == pcs])
        n1 = np.sum(experiment.Weight[experiment.Sample == pct])
        r = n0 / n1
        mr[experiment.Sample == pcs] = 1
        mr[experiment.Sample == pct] = -r
        return mr

    def pi0_ring_separation(self, experiment, x):
        r"""Method changing the efficiency of ring separation in the $\pi^0\rightarrow 2\gamma$ decay.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        # if self._unphysical_value(x):
        #     return 1e3
        r1 = 2
        r2 = 6
        mr = np.ones(experiment.NumberOfEvents)
        n0 = np.sum(experiment.Weight[experiment.Sample == r1])
        n1 = np.sum(experiment.Weight[experiment.Sample == r2])
        r = n0 / n1
        mr[experiment.Sample == r1] = x
        mr[experiment.Sample == r2] = 1 + r - r * x
        return mr

    def diff_pi0_ring_separation(self, experiment, x):
        r"""Method for computing the derivative of the weights of the events from $\pi^0\rightarrow 2\gamma$ decays
        w.r.t. the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `pi0_ring_separation` weights.
        """
        # if self._unphysical_value(x):
        #     return 0
        r1 = 2
        r2 = 6
        mr = np.zeros(experiment.NumberOfEvents)
        n0 = np.sum(experiment.Weight[experiment.Sample == r1])
        n1 = np.sum(experiment.Weight[experiment.Sample == r2])
        r = n0 / n1
        mr[experiment.Sample == r1] = 1
        mr[experiment.Sample == r2] = -r
        return mr

    def e_ring_separation(self, experiment, x):
        r"""Method changing the efficiency of detecting e-like rings.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        # if self._unphysical_value(x):
        #     return 1e3
        r1 = [0, 1, 7, 8, 19, 20, 21]
        r2 = [10, 11, 13, 24, 25, 26]
        mr = np.ones(experiment.NumberOfEvents)
        n0 = sum(
            np.sum(experiment.Weight[experiment.Sample == sample]) for sample in r1
        )
        n1 = sum(
            np.sum(experiment.Weight[experiment.Sample == sample]) for sample in r2
        )
        r = n0 / n1
        for sample in r1:
            mr[experiment.Sample == sample] = x
        for sample in r2:
            mr[experiment.Sample == sample] = 1 + r - r * x
        return mr

    def diff_e_ring_separation(self, experiment, x):
        r"""Method for computing the derivative of the weights of the e-like ring events w.r.+t. the
        tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `e_ring_separation` weights.
        """
        # if self._unphysical_value(x):
        #     return 0
        r1 = [0, 1, 7, 8, 19, 20, 21]
        r2 = [10, 11, 13, 24, 25, 26]
        mr = np.zeros(experiment.NumberOfEvents)
        n0 = sum(
            np.sum(experiment.Weight[experiment.Sample == sample]) for sample in r1
        )
        n1 = sum(
            np.sum(experiment.Weight[experiment.Sample == sample]) for sample in r2
        )
        r = n0 / n1
        for sample in r1:
            mr[experiment.Sample == sample] = 1
        for sample in r2:
            mr[experiment.Sample == sample] = -r
        return mr

    def mu_ring_separation(self, experiment, x):
        r"""Method changing the efficiency of detecting $\mu$-like rings.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        # if self._unphysical_value(x):
        #     return 1e3
        r1 = [3, 4, 5, 9, 22, 23, 27, 28]
        r2 = [12]
        mr = np.ones(experiment.NumberOfEvents)
        n0 = sum(
            np.sum(experiment.Weight[experiment.Sample == sample]) for sample in r1
        )
        n1 = sum(
            np.sum(experiment.Weight[experiment.Sample == sample]) for sample in r2
        )
        r = n0 / n1
        for sample in r1:
            mr[experiment.Sample == sample] = x
        for sample in r2:
            mr[experiment.Sample == sample] = 1 + r - r * x
        return mr

    def diff_mu_ring_separation(self, experiment, x):
        r"""Method for computing the derivative of the weights of the $\mu$-like ring events w.r.t. the
        tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `mu_ring_separation` weights.
        """
        # if self._unphysical_value(x):
        #     return 0
        r1 = [3, 4, 5, 9, 22, 23, 27, 28]
        r2 = [12]
        mr = np.zeros(experiment.NumberOfEvents)
        n0 = sum(
            np.sum(experiment.Weight[experiment.Sample == sample]) for sample in r1
        )
        n1 = sum(
            np.sum(experiment.Weight[experiment.Sample == sample]) for sample in r2
        )
        r = n0 / n1
        for sample in r1:
            mr[experiment.Sample == sample] = 1
        for sample in r2:
            mr[experiment.Sample == sample] = -r
        return mr

    def singlering_pid(self, experiment, x):
        r"""Method changing the particle identification efficiency of single-ring events.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        # if self._unphysical_value(x):
        #     return 1e3
        e = [0, 1, 7, 8, 19, 20, 21, 24, 25, 26]
        mu = [3, 4, 5, 9, 22, 23, 27, 28]
        mr = np.ones(experiment.NumberOfEvents)
        n0 = sum(np.sum(experiment.Weight[experiment.Sample == sample]) for sample in e)
        n1 = sum(
            np.sum(experiment.Weight[experiment.Sample == sample]) for sample in mu
        )
        r = n0 / n1
        for sample in e:
            mr[experiment.Sample == sample] = x
        for sample in mu:
            mr[experiment.Sample == sample] = 1 + r - r * x
        return mr

    def diff_singlering_pid(self, experiment, x):
        r"""Method for computing the derivative of the weights of the single-ring events w.r.t. the pid tuning
        parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `singlering_pid` weights.
        """
        # if self._unphysical_value(x):
        #     return 0
        e = [0, 1, 7, 8, 19, 20, 21, 24, 25, 26]
        mu = [3, 4, 5, 9, 22, 23, 27, 28]
        mr = np.zeros(experiment.NumberOfEvents)
        n0 = sum(np.sum(experiment.Weight[experiment.Sample == sample]) for sample in e)
        n1 = sum(
            np.sum(experiment.Weight[experiment.Sample == sample]) for sample in mu
        )
        r = n0 / n1
        for sample in e:
            mr[experiment.Sample == sample] = 1
        for sample in mu:
            mr[experiment.Sample == sample] = -r
        return mr

    def multiring_pid(self, experiment, x):
        r"""Method changing the particle identification efficiency of multi-ring events.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        # if self._unphysical_value(x):
        #     return 1e3
        e = [10, 11, 13]
        mu = [12]
        mr = np.ones(experiment.NumberOfEvents)
        n0 = sum(np.sum(experiment.Weight[experiment.Sample == sample]) for sample in e)
        n1 = sum(
            np.sum(experiment.Weight[experiment.Sample == sample]) for sample in mu
        )
        r = n0 / n1
        for sample in e:
            mr[experiment.Sample == sample] = x
        for sample in mu:
            mr[experiment.Sample == sample] = 1 + r - r * x
        return mr

    def diff_multiring_pid(self, experiment, x):
        r"""Method for computing the derivative of the weights of the multi-ring events w.r.t. the pid tuning
        parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `multiring_pid` weights.
        """
        # if self._unphysical_value(x):
        #     return 0
        e = [10, 11, 13]
        mu = [12]
        mr = np.zeros(experiment.NumberOfEvents)
        n0 = sum(np.sum(experiment.Weight[experiment.Sample == sample]) for sample in e)
        n1 = sum(
            np.sum(experiment.Weight[experiment.Sample == sample]) for sample in mu
        )
        r = n0 / n1
        for sample in e:
            mr[experiment.Sample == sample] = 1
        for sample in mu:
            mr[experiment.Sample == sample] = -r
        return mr

    def neutron_tagging(self, experiment, x):
        r"""Method changing the efficiency of neutron tagging.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        # if self._unphysical_value(x):
        #     return 1e3
        nn = np.ones(experiment.NumberOfEvents)
        nn0 = (
            (experiment.Sample == 20)
            | (experiment.Sample == 25)
            | (experiment.Sample == 22)
            | (experiment.Sample == 27)
        )
        nn1 = (
            (experiment.Sample == 21)
            | (experiment.Sample == 26)
            | (experiment.Sample == 23)
            | (experiment.Sample == 28)
        )
        n0 = np.sum(nn0)
        n1 = np.sum(nn1)
        r = n0 / n1
        nn[nn0] = x
        nn[nn1] = 1 + r - r * x
        return nn

    def diff_neutron_tagging(self, experiment, x):
        r"""Method for computing the derivative of the weights w.r.t. the neutron tagging efficiency tuning
        parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `neutron_tagging` weights.
        """
        # if self._unphysical_value(x):
        #     return 0
        nn = np.zeros(experiment.NumberOfEvents)
        nn0 = (
            (experiment.Sample == 20)
            | (experiment.Sample == 25)
            | (experiment.Sample == 22)
            | (experiment.Sample == 27)
        )
        nn1 = (
            (experiment.Sample == 21)
            | (experiment.Sample == 26)
            | (experiment.Sample == 23)
            | (experiment.Sample == 28)
        )
        n0 = np.sum(nn0)
        n1 = np.sum(nn1)
        r = n0 / n1
        nn[nn0] = 1
        nn[nn1] = -r
        return nn

    def decay_e_tagging(self, experiment, x):
        r"""Method changing the efficiency of decay electron tagging.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        # if self._unphysical_value(x):
        #     return 1e3
        mue = np.ones(experiment.NumberOfEvents)
        n0 = np.sum(experiment.DecayE < 1)
        n1 = np.sum((experiment.DecayE >= 1) & (experiment.DecayE < 2))
        n2 = np.sum(experiment.DecayE >= 2)
        N = n0 + n1 + n2
        r0 = n0 / N
        r1 = n1 / N
        r2 = n2 / N
        rx1 = x * r1 + 2 * (1 - x) * r2
        rx2 = x * x * r2 + 2 * (1 - x) * r2
        rx0 = 1 - rx1 - rx2
        mue[experiment.DecayE == 0] = rx0 / r0
        mue[experiment.DecayE == 1] = rx1 / r1
        mue[experiment.DecayE > 1] = rx2 / r2
        return mue

    def diff_decay_e_tagging(self, experiment, x):
        r"""Method for computing the derivative of the weights w.r.t. the decay electron tagging efficiency tuning
        parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `decay_e_tagging` weights.
        """
        # if self._unphysical_value(x):
        #     return 0
        mue = np.zeros(experiment.NumberOfEvents)
        n0 = np.sum(experiment.DecayE < 1)
        n1 = np.sum((experiment.DecayE >= 1) & (experiment.DecayE < 2))
        n2 = np.sum(experiment.DecayE >= 2)
        N = n0 + n1 + n2
        r0 = n0 / N
        r1 = n1 / N
        r2 = n2 / N
        rx1 = r1 - 2 * r2
        rx2 = 2 * x * r2 - 2 * r2
        rx0 = -rx1 - rx2
        mue[experiment.DecayE == 0] = rx0 / r0
        mue[experiment.DecayE == 1] = rx1 / r1
        mue[experiment.DecayE > 1] = rx2 / r2
        return mue

    def upmu_shower_separation(self, experiment, x):
        # if self._unphysical_value(x):
        #     return 1e3
        um = np.ones(experiment.NumberOfEvents)
        n0 = np.sum(experiment.Weight[experiment.Sample == 18])
        n1 = np.sum(experiment.Weight[experiment.Sample == 17])
        r = n0 / n1
        um[experiment.Sample == 18] = x
        um[experiment.Sample == 17] = 1 + r - r * x
        return um

    def diff_upmu_shower_separation(self, experiment, x):
        # if self._unphysical_value(x):
        #     return 0
        um = np.zeros(experiment.NumberOfEvents)
        n0 = np.sum(experiment.Weight[experiment.Sample == 18])
        n1 = np.sum(experiment.Weight[experiment.Sample == 17])
        r = n0 / n1
        um[experiment.Sample == 18] = 1
        um[experiment.Sample == 17] = -r
        return um

    def upmu_stop_bkg(self, experiment, x):
        # if self._unphysical_value(x):
        #     return 1e3
        um = np.ones(experiment.NumberOfEvents)
        um[experiment.Sample == 16] = x
        return um

    def diff_upmu_stop_bkg(self, experiment, x):
        # if self._unphysical_value(x):
        #     return 0
        um = np.zeros(experiment.NumberOfEvents)
        um[experiment.Sample == 16] = 1
        return um

    def upmu_showering_bkg(self, experiment, x):
        # if self._unphysical_value(x):
        #     return 1e3
        um = np.ones(experiment.NumberOfEvents)
        um[experiment.Sample == 18] = x
        return um

    def diff_upmu_showering_bkg(self, experiment, x):
        # if self._unphysical_value(x):
        #     return 0
        um = np.zeros(experiment.NumberOfEvents)
        um[experiment.Sample == 18] = 1
        return um

    def upmu_nonshowering_bkg(self, experiment, x):
        # if self._unphysical_value(x):
        #     return 1e3
        um = np.ones(experiment.NumberOfEvents)
        um[experiment.Sample == 17] = x
        return um

    def diff_upmu_nonshowering_bkg(self, experiment, x):
        # if self._unphysical_value(x):
        #     return 0
        um = np.zeros(experiment.NumberOfEvents)
        um[experiment.Sample == 17] = 1
        return um
