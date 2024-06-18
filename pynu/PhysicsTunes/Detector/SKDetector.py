from PhysicsTunes import Tune
import numpy as np

# import logging
from LoggingDecorator import logd
import sys

sys.path.append("../")

####################
# Super-Kamiokande #
####################


class SuperK(Tune):
    r"""Class containing general implementation of a Super-Kamiokande like detectors."""

    def attenuation_length(self, experiment, x):
        r"""Method for modifying the energy scale of the simulation by multiplying by x the
        reconstructed energy.

        Args:
            x (float): Variation of absorption lenght w.r.t. nominal.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information
            of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        # energy bias
        nueCC = np.abs(experiment.nuPDG == 12) & experiment.CC
        bias = 0.1 * x
        experiment.set_energy_bias(bias, nueCC)

        numuCC = np.abs(experiment.nuPDG == 14) & experiment.CC
        bias = 0.02 * (experiment.ETrue_lepton - 1.0) / (0.2 - 1.0) - 0.1 * (
            experiment.ETrue_lepton - 1.0
        ) / (0.2 - 1.0)
        experiment.set_energy_bias(bias, numuCC)

        # energy scale
        pass

        # PID
        pass

    @logd(file=False, logging_level="debug")
    def energy_scale(self, experiment, x):
        r"""Method for modifying the energy scale of the simulation by multiplying by x the
        reconstructed energy.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if np.abs(x - 1) > 5e-4:
            experiment.set_energy_scale(x)
        return 1

    @logd(file=False, logging_level="debug")
    def diff_energy_scale(self, experiment, x):
        r"""Method for computing the derivative of the weights of the energy scale w.r.t. the
        tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `energy_scale` weights.
        """
        pass

    @logd(file=False, logging_level="debug")
    def FCPC_separation(self, experiment, x):
        r"""Method changing the efficiency of the fully and partially-contained events in SK.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        # logging.info(f"Entering {__name__}")
        print(f"Entering {__name__}")
        fcpc = np.ones(experiment.NumberOfEvents)
        if (
            experiment.Detector == "SuperK-Gd"
            or experiment.Detector == "SKIV"
            or experiment.Detector == "SuperK_Htag"
            or experiment.Detector == "SuperK_Gdtag"
        ):
            fcpc[experiment.Sample < 16] = x
            wFC = np.sum(experiment.Weight[experiment.Sample < 16])
            wPC = np.sum(
                experiment.Weight[
                    np.logical_or(experiment.Sample == 16, experiment.Sample == 17)
                ]
            )
            y = ((wPC + wFC) - x * wFC) / wPC
            fcpc[np.logical_or(experiment.Sample == 16, experiment.Sample == 17)] = y
        else:
            fcpc[experiment.Sample < 14] = x
            wFC = np.sum(experiment.Weight[experiment.Sample < 14])
            wPC = np.sum(
                experiment.Weight[
                    np.logical_or(experiment.Sample == 14, experiment.Sample == 15)
                ]
            )
            y = ((wPC + wFC) - x * wFC) / wPC
            fcpc[np.logical_or(experiment.Sample == 14, experiment.Sample == 15)] = y
        return fcpc

    @logd(file=False, logging_level="debug")
    def diff_FCPC_separation(self, experiment, x):
        r"""Method for computing the derivative of the weights of the fully and partially-contained events
        w.r.t. the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `FCPC_separation` weights.
        """
        print(f"Entering {__name__}")
        fcpc = np.zeros(experiment.NumberOfEvents)
        if (
            experiment.Experiment == "SuperK-Gd"
            or experiment.Experiment == "SKIV"
            or experiment.Experiment == "SuperK_Htag"
            or experiment.Experiment == "SuperK_Gdtag"
        ):
            fcpc[experiment.Sample < 16] = 1
            wFC = np.sum(experiment.Weight[experiment.Sample < 16])
            wPC = np.sum(
                experiment.Weight[
                    np.logical_or(experiment.Sample == 16, experiment.Sample == 17)
                ]
            )
            y = (-wFC) / wPC
            fcpc[np.logical_or(experiment.Sample == 16, experiment.Sample == 17)] = y
        else:
            fcpc[experiment.Sample < 14] = 1
            wFC = np.sum(experiment.Weight[experiment.Sample < 14])
            wPC = np.sum(
                experiment.Weight[
                    np.logical_or(experiment.Sample == 14, experiment.Sample == 15)
                ]
            )
            y = (-wFC) / wPC
            fcpc[np.logical_or(experiment.Sample == 14, experiment.Sample == 15)] = y
        return fcpc

    @logd(file=False, logging_level="debug")
    def FC_reduction(self, experiment, x):
        r"""Method changing the efficiency of the fully-contained events reduction in SK.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        fc = np.ones(experiment.NumberOfEvents)
        if (
            experiment.Detector == "SuperK-Gd"
            or experiment.Detector == "SKIV"
            or experiment.Detector == "SuperK_Htag"
            or experiment.Detector == "SuperK_Gdtag"
        ):
            fc[experiment.Sample < 16] = x
        else:
            fc[experiment.Sample < 14] = x
        return fc

    @logd(file=False, logging_level="debug")
    def diff_FC_reduction(self, experiment, x):
        r"""Method for computing the derivative of the weights of the fully-contained events w.r.t.
        the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `FC_reduction` weights.
        """
        fc = np.zeros(experiment.NumberOfEvents)
        if (
            experiment.Experiment == "SuperK-Gd"
            or experiment.Experiment == "SKIV"
            or experiment.Experiment == "SuperK_Htag"
            or experiment.Experiment == "SuperK_Gdtag"
        ):
            fc[experiment.Sample < 16] = 1
        else:
            fc[experiment.Sample < 14] = 1
        return fc

    @logd(file=False, logging_level="debug")
    def fiducial_volume(self, experiment, x):
        r"""Method changing the efficiency of the fiducial volume cut.
        NOTE: Currently, it applies a normalization factor on all events. More precise implementation coming soon.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        print("hello!!!!!!!!!!!")
        print(x)
        return x

    @logd(file=False, logging_level="debug")
    def diff_fiducial_volume(self, experiment, x):
        r"""Method for computing the derivative of the weights w.r.t. the tuning parameter of the fiducial volumen.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `fiducial_volume` weights.
        """
        return 1

    @logd(file=False, logging_level="debug")
    def PC_reduction(self, experiment, x):
        r"""Method changing the efficiency of the partially-contained events reduction in SK.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        pc = np.ones(experiment.NumberOfEvents)
        if (
            experiment.Detector == "SuperK-Gd"
            or experiment.Detector == "SKIV"
            or experiment.Detector == "SuperK_Htag"
            or experiment.Detector == "SuperK_Gdtag"
        ):
            pc[np.logical_or(experiment.Sample == 16, experiment.Sample == 17)] = x
        else:
            pc[np.logical_or(experiment.Sample == 14, experiment.Sample == 15)] = x
        return pc

    @logd(file=False, logging_level="debug")
    def diff_PC_reduction(self, experiment, x):
        r"""Method for computing the derivative of the weights of the partially-contained events w.r.t.
        the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `PC_reduction` weights.
        """
        pc = np.zeros(experiment.NumberOfEvents)
        if (
            experiment.Experiment == "SuperK-Gd"
            or experiment.Experiment == "SKIV"
            or experiment.Experiment == "SuperK_Htag"
            or experiment.Experiment == "SuperK_Gdtag"
        ):
            pc[np.logical_or(experiment.Sample == 16, experiment.Sample == 17)] = 1
        else:
            pc[np.logical_or(experiment.Sample == 14, experiment.Sample == 15)] = 1
        return pc

    @logd(file=False, logging_level="debug")
    def subgev_2ring_pi0(self, experiment, x):
        r"""Method changing the fraction of 2-ring $\pi^0$-like events.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        pi02r = np.ones(experiment.NumberOfEvents)
        pi02r[experiment.Sample == 6] = x
        return pi02r

    @logd(file=False, logging_level="debug")
    def diff_subgev_2ring_pi0(self, experiment, x):
        r"""Method for computing the derivative of the weights of the 2-ring $\pi^0$-like events w.r.t.
        the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `subgev_2ring_pi0` weights.
        """
        pi02r = np.zeros(experiment.NumberOfEvents)
        pi02r[experiment.Sample == 6] = 1
        return pi02r

    @logd(file=False, logging_level="debug")
    def subgev_1ring_pi0(self, experiment, x):
        r"""Method changing the fraction of single-ring $\pi^0$-like events.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        pi01r = np.ones(experiment.NumberOfEvents)
        if (
            experiment.Detector == "SuperK-Gd"
            or experiment.Detector == "SKIV"
            or experiment.Detector == "SuperK_Htag"
            or experiment.Detector == "SuperK_Gdtag"
        ):
            pi01r[experiment.Sample == 3] = x
        else:
            pi01r[experiment.Sample == 2] = x
        return pi01r

    @logd(file=False, logging_level="debug")
    def diff_subgev_1ring_pi0(self, experiment, x):
        r"""Method for computing the derivative of the weights of the single-ring $\pi^0$-like events w.r.t.
        the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `subgev_1ring_pi0` weights.
        """
        pi01r = np.zeros(experiment.NumberOfEvents)
        if (
            experiment.Experiment == "SuperK-Gd"
            or experiment.Experiment == "SKIV"
            or experiment.Experiment == "SuperK_Htag"
            or experiment.Experiment == "SuperK_Gdtag"
        ):
            pi01r[experiment.Sample == 3] = 1
        else:
            pi01r[experiment.Sample == 2] = 1
        return pi01r

    @logd(file=False, logging_level="debug")
    def multiring_nunubar_separation(self, experiment, x):
        r"""Method changing the efficiency of neutrino-antineutrino separation in multi-ring events.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if (
            experiment.Detector == "SuperK-Gd"
            or experiment.Detector == "SKIV"
            or experiment.Detector == "SuperK_Htag"
            or experiment.Detector == "SuperK_Gdtag"
        ):
            nu = 12
            nub = 13
        else:
            nu = 10
            nub = 11
        mr = np.ones(experiment.NumberOfEvents)
        n0 = np.sum(experiment.Weight[experiment.Sample == nu])
        n1 = np.sum(experiment.Weight[experiment.Sample == nub])
        r = n0 / n1
        mr[experiment.Sample == nu] = x
        mr[experiment.Sample == nub] = 1 + r - r * x
        return mr

    @logd(file=False, logging_level="debug")
    def diff_multiring_nunubar_separation(self, experiment, x):
        r"""Method for computing the derivative of the weights of the multi-ring neutrino and antineutrino
        events w.r.t. the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `multiring_nunubar_separation` weights.
        """
        if (
            experiment.Experiment == "SuperK-Gd"
            or experiment.Experiment == "SKIV"
            or experiment.Experiment == "SuperK_Htag"
            or experiment.Experiment == "SuperK_Gdtag"
        ):
            nu = 12
            nub = 13
        else:
            nu = 10
            nub = 11
        mr = np.zeros(experiment.NumberOfEvents)
        n0 = np.sum(experiment.Weight[experiment.Sample == nu])
        n1 = np.sum(experiment.Weight[experiment.Sample == nub])
        r = n0 / n1
        mr[experiment.Sample == nu] = 1
        mr[experiment.Sample == nub] = -r
        return mr

    @logd(file=False, logging_level="debug")
    def multiring_emu_separation(self, experiment, x):
        r"""Method changing the efficiency of electron-muon separation in multi-ring events.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if (
            experiment.Detector == "SuperK-Gd"
            or experiment.Detector == "SKIV"
            or experiment.Detector == "SuperK_Htag"
            or experiment.Detector == "SuperK_Gdtag"
        ):
            e0 = 12
            e1 = 13
            e2 = 15
            mu = 14
        else:
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

    @logd(file=False, logging_level="debug")
    def diff_multiring_emu_separation(self, experiment, x):
        r"""Method for computing the derivative of the weights of the multi-ring muon and electron (anti)neutrino
        events w.r.t. the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `multiring_emu_separation` weights.
        """
        if (
            experiment.Experiment == "SuperK-Gd"
            or experiment.Experiment == "SKIV"
            or experiment.Experiment == "SuperK_Htag"
            or experiment.Experiment == "SuperK_Gdtag"
        ):
            e0 = 12
            e1 = 13
            e2 = 15
            mu = 14
        else:
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

    @logd(file=False, logging_level="debug")
    def multiring_eother_separation(self, experiment, x):
        r"""Method changing the efficiency of electron neutrinos interacting charged-current and neutral-current
        interactions in multi-ring events.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if (
            experiment.Detector == "SuperK-Gd"
            or experiment.Detector == "SKIV"
            or experiment.Detector == "SuperK_Htag"
            or experiment.Detector == "SuperK_Gdtag"
        ):
            e0 = 12
            e1 = 13
            o0 = 15
        else:
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

    @logd(file=False, logging_level="debug")
    def diff_multiring_eother_separation(self, experiment, x):
        r"""Method for computing the derivative of the weights of the multi-ring e-like events w.r.t. the
        tuning parameter separating between CC $\nu_e$ and NC $\nu$.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `multiring_eother_separation` weights.
        """
        if (
            experiment.Experiment == "SuperK-Gd"
            or experiment.Experiment == "SKIV"
            or experiment.Experiment == "SuperK_Htag"
            or experiment.Experiment == "SuperK_Gdtag"
        ):
            e0 = 12
            e1 = 13
            o0 = 15
        else:
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

    @logd(file=False, logging_level="debug")
    def pc_stopthru_separation(self, experiment, x):
        r"""Method changing the efficiency of PC-StopThru separation.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if (
            experiment.Detector == "SuperK-Gd"
            or experiment.Detector == "SKIV"
            or experiment.Detector == "SuperK_Htag"
            or experiment.Detector == "SuperK_Gdtag"
        ):
            pcs = 16
            pct = 17
        else:
            pcs = 14
            pct = 15
        mr = np.ones(experiment.NumberOfEvents)
        n0 = np.sum(experiment.Weight[experiment.Sample == pcs])
        n1 = np.sum(experiment.Weight[experiment.Sample == pct])
        r = n0 / n1
        mr[experiment.Sample == pcs] = x
        mr[experiment.Sample == pct] = 1 + r - r * x
        return mr

    @logd(file=False, logging_level="debug")
    def diff_PC_StopThru_separation(self, experiment, x):
        r"""Method for computing the derivative of the weights of the PC and Stop Thru events w.r.t. the
        tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `pc_stopthru_separation` weights.
        """
        if (
            experiment.Experiment == "SuperK-Gd"
            or experiment.Experiment == "SKIV"
            or experiment.Experiment == "SuperK_Htag"
            or experiment.Experiment == "SuperK_Gdtag"
        ):
            pcs = 16
            pct = 17
        else:
            pcs = 14
            pct = 15
        mr = np.zeros(experiment.NumberOfEvents)
        n0 = np.sum(experiment.Weight[experiment.Sample == pcs])
        n1 = np.sum(experiment.Weight[experiment.Sample == pct])
        r = n0 / n1
        mr[experiment.Sample == pcs] = 1
        mr[experiment.Sample == pct] = -r
        return mr

    @logd(file=False, logging_level="debug")
    def pi0_ring_separation(self, experiment, x):
        r"""Method changing the efficiency of ring separation in the $\pi^0\rightarrow 2\gamma$ decay.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if (
            experiment.Detector == "SuperK-Gd"
            or experiment.Detector == "SKIV"
            or experiment.Detector == "SuperK_Htag"
            or experiment.Detector == "SuperK_Gdtag"
        ):
            r1 = 3
            r2 = 6
        else:
            r1 = 2
            r2 = 6
        mr = np.ones(experiment.NumberOfEvents)
        n0 = np.sum(experiment.Weight[experiment.Sample == r1])
        n1 = np.sum(experiment.Weight[experiment.Sample == r2])
        r = n0 / n1
        mr[experiment.Sample == r1] = x
        mr[experiment.Sample == r2] = 1 + r - r * x
        return mr

    @logd(file=False, logging_level="debug")
    def diff_pi0_ring_separation(self, experiment, x):
        r"""Method for computing the derivative of the weights of the events from $\pi^0\rightarrow 2\gamma$ decays
        w.r.t. the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `pi0_ring_separation` weights.
        """
        if (
            experiment.Experiment == "SuperK-Gd"
            or experiment.Experiment == "SKIV"
            or experiment.Experiment == "SuperK_Htag"
            or experiment.Experiment == "SuperK_Gdtag"
        ):
            r1 = 3
            r2 = 6
        else:
            r1 = 2
            r2 = 6
        mr = np.zeros(experiment.NumberOfEvents)
        n0 = np.sum(experiment.Weight[experiment.Sample == r1])
        n1 = np.sum(experiment.Weight[experiment.Sample == r2])
        r = n0 / n1
        mr[experiment.Sample == r1] = 1
        mr[experiment.Sample == r2] = -r
        return mr

    @logd(file=False, logging_level="debug")
    def e_ring_separation(self, experiment, x):
        r"""Method changing the efficiency of detecting e-like rings.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if (
            experiment.Detector == "SuperK-Gd"
            or experiment.Detector == "SKIV"
            or experiment.Detector == "SuperK_Htag"
            or experiment.Detector == "SuperK_Gdtag"
        ):
            r1 = [0, 1, 2, 7, 8, 9]
            r2 = [12, 13, 14]
        else:
            r1 = [0, 1, 2, 7, 8, 9]
            r2 = [12, 13, 14]
        mr = np.ones(experiment.NumberOfEvents)
        n0 = 0
        n1 = 0
        for sample in r1:
            n0 += np.sum(experiment.Weight[experiment.Sample == sample])
        for sample in r2:
            n1 += np.sum(experiment.Weight[experiment.Sample == sample])
        r = n0 / n1
        for sample in r1:
            mr[experiment.Sample == sample] = x
        for sample in r2:
            mr[experiment.Sample == sample] = 1 + r - r * x
        return mr

    @logd(file=False, logging_level="debug")
    def diff_e_ring_separation(self, experiment, x):
        r"""Method for computing the derivative of the weights of the e-like ring events w.r.t. the
        tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `e_ring_separation` weights.
        """
        if (
            experiment.Experiment == "SuperK-Gd"
            or experiment.Experiment == "SKIV"
            or experiment.Experiment == "SuperK_Htag"
            or experiment.Experiment == "SuperK_Gdtag"
        ):
            r1 = [0, 1, 2, 7, 8, 9]
            r2 = [12, 13, 14]
        else:
            r1 = [0, 1, 2, 7, 8, 9]
            r2 = [12, 13, 14]
        mr = np.zeros(experiment.NumberOfEvents)
        n0 = 0
        n1 = 0
        for sample in r1:
            n0 += np.sum(experiment.Weight[experiment.Sample == sample])
        for sample in r2:
            n1 += np.sum(experiment.Weight[experiment.Sample == sample])
        r = n0 / n1
        for sample in r1:
            mr[experiment.Sample == sample] = 1
        for sample in r2:
            mr[experiment.Sample == sample] = -r
        return mr

    @logd(file=False, logging_level="debug")
    def mu_ring_separation(self, experiment, x):
        r"""Method changing the efficiency of detecting $\mu$-like rings.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if (
            experiment.Detector == "SuperK-Gd"
            or experiment.Detector == "SKIV"
            or experiment.Detector == "SuperK_Htag"
            or experiment.Detector == "SuperK_Gdtag"
        ):
            r1 = [4, 5, 10, 11]
            r2 = [14]
        else:
            r1 = [3, 4, 5, 9]
            r2 = [12]
        mr = np.ones(experiment.NumberOfEvents)
        n0 = 0
        n1 = 0
        for sample in r1:
            n0 += np.sum(experiment.Weight[experiment.Sample == sample])
        for sample in r2:
            n1 += np.sum(experiment.Weight[experiment.Sample == sample])
        r = n0 / n1
        for sample in r1:
            mr[experiment.Sample == sample] = x
        for sample in r2:
            mr[experiment.Sample == sample] = 1 + r - r * x
        return mr

    @logd(file=False, logging_level="debug")
    def diff_mu_ring_separation(self, experiment, x):
        r"""Method for computing the derivative of the weights of the $\mu$-like ring events w.r.t. the
        tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `mu_ring_separation` weights.
        """
        if (
            experiment.Experiment == "SuperK-Gd"
            or experiment.Experiment == "SKIV"
            or experiment.Experiment == "SuperK_Htag"
            or experiment.Experiment == "SuperK_Gdtag"
        ):
            r1 = [4, 5, 10, 11]
            r2 = [14]
        else:
            r1 = [3, 4, 5, 9]
            r2 = [12]
        mr = np.zeros(experiment.NumberOfEvents)
        n0 = 0
        n1 = 0
        for sample in r1:
            n0 += np.sum(experiment.Weight[experiment.Sample == sample])
        for sample in r2:
            n1 += np.sum(experiment.Weight[experiment.Sample == sample])
        r = n0 / n1
        for sample in r1:
            mr[experiment.Sample == sample] = 1
        for sample in r2:
            mr[experiment.Sample == sample] = -r
        return mr

    @logd(file=False, logging_level="debug")
    def singlering_pid(self, experiment, x):
        r"""Method changing the particle identification efficiency of single-ring events.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if (
            experiment.Detector == "SuperK-Gd"
            or experiment.Detector == "SKIV"
            or experiment.Detector == "SuperK_Htag"
            or experiment.Detector == "SuperK_Gdtag"
        ):
            e = [0, 1, 2, 3, 7, 8, 9]
            mu = [4, 5, 10, 11]
        else:
            e = [0, 1, 2, 7, 8]
            mu = [3, 4, 5, 9]
        mr = np.ones(experiment.NumberOfEvents)
        n0 = 0
        n1 = 0
        for sample in e:
            n0 += np.sum(experiment.Weight[experiment.Sample == sample])
        for sample in mu:
            n1 += np.sum(experiment.Weight[experiment.Sample == sample])
        r = n0 / n1
        for sample in e:
            mr[experiment.Sample == sample] = x
        for sample in mu:
            mr[experiment.Sample == sample] = 1 + r - r * x
        return mr

    @logd(file=False, logging_level="debug")
    def diff_singlering_pid(self, experiment, x):
        r"""Method for computing the derivative of the weights of the single-ring events w.r.t. the pid tuning
        parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `singlering_pid` weights.
        """
        if (
            experiment.Experiment == "SuperK-Gd"
            or experiment.Experiment == "SKIV"
            or experiment.Experiment == "SuperK_Htag"
            or experiment.Experiment == "SuperK_Gdtag"
        ):
            e = [0, 1, 2, 3, 7, 8, 9]
            mu = [4, 5, 10, 11]
        else:
            e = [0, 1, 2, 7, 8]
            mu = [3, 4, 5, 9]
        mr = np.zeros(experiment.NumberOfEvents)
        n0 = 0
        n1 = 0
        for sample in e:
            n0 += np.sum(experiment.Weight[experiment.Sample == sample])
        for sample in mu:
            n1 += np.sum(experiment.Weight[experiment.Sample == sample])
        r = n0 / n1
        for sample in e:
            mr[experiment.Sample == sample] = 1
        for sample in mu:
            mr[experiment.Sample == sample] = -r
        return mr

    @logd(file=False, logging_level="debug")
    def multiring_pid(self, experiment, x):
        r"""Method changing the particle identification efficiency of multi-ring events.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if (
            experiment.Detector == "SuperK-Gd"
            or experiment.Detector == "SKIV"
            or experiment.Detector == "SuperK_Htag"
            or experiment.Detector == "SuperK_Gdtag"
        ):
            e = [12, 13, 15]
            mu = [14]
        else:
            e = [10, 11, 13]
            mu = [12]
        mr = np.ones(experiment.NumberOfEvents)
        n0 = 0
        n1 = 0
        for sample in e:
            n0 += np.sum(experiment.Weight[experiment.Sample == sample])
        for sample in mu:
            n1 += np.sum(experiment.Weight[experiment.Sample == sample])
        r = n0 / n1
        for sample in e:
            mr[experiment.Sample == sample] = x
        for sample in mu:
            mr[experiment.Sample == sample] = 1 + r - r * x
        return mr

    @logd(file=False, logging_level="debug")
    def diff_multiring_pid(self, experiment, x):
        r"""Method for computing the derivative of the weights of the multi-ring events w.r.t. the pid tuning
        parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `multiring_pid` weights.
        """
        if (
            experiment.Experiment == "SuperK-Gd"
            or experiment.Experiment == "SKIV"
            or experiment.Experiment == "SuperK_Htag"
            or experiment.Experiment == "SuperK_Gdtag"
        ):
            e = [12, 13, 15]
            mu = [14]
        else:
            e = [10, 11, 13]
            mu = [12]
        mr = np.zeros(experiment.NumberOfEvents)
        n0 = 0
        n1 = 0
        for sample in e:
            n0 += np.sum(experiment.Weight[experiment.Sample == sample])
        for sample in mu:
            n1 += np.sum(experiment.Weight[experiment.Sample == sample])
        r = n0 / n1
        for sample in e:
            mr[experiment.Sample == sample] = 1
        for sample in mu:
            mr[experiment.Sample == sample] = -r
        return mr

    @logd(file=False, logging_level="debug")
    def neutron_tagging(self, experiment, x):
        r"""Method changing the efficiency of neutron tagging.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        nn = np.ones(experiment.NumberOfEvents)
        if (
            experiment.Detector == "SuperK-Gd"
            or experiment.Detector == "SKIV"
            or experiment.Detector == "SuperK_Htag"
            or experiment.Detector == "SuperK_Gdtag"
        ):
            n0 = np.sum(experiment.Neutron == 0)
            n1 = np.sum(experiment.Neutron > 0)
            r = n0 / n1
            nn[experiment.Neutron == 0] = x
            nn[experiment.Neutron > 0] = 1 + r - r * x
            return nn
        else:
            return 0

    @logd(file=False, logging_level="debug")
    def diff_neutron_tagging(self, experiment, x):
        r"""Method for computing the derivative of the weights w.r.t. the neutron tagging efficiency tuning
        parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `neutron_tagging` weights.
        """
        nn = np.zeros(experiment.NumberOfEvents)
        if (
            experiment.Experiment == "SuperK-Gd"
            or experiment.Experiment == "SKIV"
            or experiment.Experiment == "SuperK_Htag"
            or experiment.Experiment == "SuperK_Gdtag"
        ):
            n0 = np.sum(experiment.Neutron == 0)
            n1 = np.sum(experiment.Neutron > 0)
            r = n0 / n1
            nn[experiment.Neutron == 0] = 1
            nn[experiment.Neutron > 0] = -r
            return nn
        else:
            return 0

    @logd(file=False, logging_level="debug")
    def decay_e_tagging(self, experiment, x):
        r"""Method changing the efficiency of decay electron tagging.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        mue = np.ones(experiment.NumberOfEvents)
        n0 = np.sum(experiment.DecayE == 0)
        n1 = np.sum(experiment.DecayE == 1)
        n2 = np.sum(experiment.DecayE > 1)
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

    @logd(file=False, logging_level="debug")
    def diff_decay_e_tagging(self, experiment, x):
        r"""Method for computing the derivative of the weights w.r.t. the decay electron tagging efficiency tuning
        parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `decay_e_tagging` weights.
        """
        mue = np.zeros(experiment.NumberOfEvents)
        n0 = np.sum(experiment.DecayE == 0)
        n1 = np.sum(experiment.DecayE == 1)
        n2 = np.sum(experiment.DecayE > 1)
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
