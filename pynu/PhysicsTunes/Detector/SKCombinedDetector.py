from ..PhysicsTunes import Tune

import numpy as np

import os
import sys

sys.path.append("../")

############################################
###### Used for pheno combined SK MC #######
############################################


class SuperK_Combined(Tune):
    #: Rate basis for the migration ratios r (and decay-e fractions):
    #:   "weighted" (default) -- expected rates, W = BaseWeight*PhysicsWeight
    #:                           (Newtrinos/PE `get_double_factor` convention)
    #:   "raw"                -- unweighted MC event counts (pre-2026 behavior)
    #: Select via the PYNU_SK_MIGRATION_BASIS environment variable (read at
    #: import time) or set SuperK_Combined.MIGRATION_BASIS = "raw" before the
    #: fit. Both bases are nuisance-independent, so the analytic diff_*
    #: derivatives stay exact either way.
    MIGRATION_BASIS = os.environ.get("PYNU_SK_MIGRATION_BASIS", "weighted")

    def _rate_weight(self, experiment):
        """Per-event rate basis for migration ratios, per MIGRATION_BASIS.

        Default "weighted": BaseWeight*PhysicsWeight, the physics-weighted,
        pre-detector-nuisance per-event expected rate -- rate-conserving
        migration ratios r in place of raw event counts. "raw": ones, which
        reduces every sum to an unweighted event count, reproducing the
        original implementation exactly. Either basis is independent of the
        detector nuisance vector, so the paired diff_* derivatives (which
        treat r as constant) remain exact.
        """
        if str(self.MIGRATION_BASIS).lower() == "raw":
            return np.ones(experiment.NumberOfEvents)
        return experiment.BaseWeight * experiment.PhysicsWeight

    def _migration_ratio(self, experiment, donor, acceptor):
        """Weighted-rate ratio r = rate(donor samples) / rate(acceptor samples).

        `donor`/`acceptor` may be a scalar sample id or a list of ids.
        """
        import numpy as np
        W = self._rate_weight(experiment)
        n0 = np.sum(W[np.isin(experiment.Sample, np.atleast_1d(donor))])
        n1 = np.sum(W[np.isin(experiment.Sample, np.atleast_1d(acceptor))])
        return n0 / n1

    def _mask_ratio(self, experiment, mask0, mask1):
        """Weighted-rate ratio r from boolean event masks (used by neutron_tagging)."""
        import numpy as np
        W = self._rate_weight(experiment)
        return np.sum(W[mask0]) / np.sum(W[mask1])

    def energy_scale(self, experiment, x):
        """See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.energy_scale`."""
        escale = np.ones(experiment.NumberOfEvents)
        for sample in experiment.Samples:
            ebins = experiment.EnergyBins[sample]
            if len(ebins)>1:
                for i in range(ebins.size):
                    if i==0 and x<0:
                        pass # do nothing if lowest bin and shift is negative
                    elif (i==ebins.size-1) and (x>0):
                        pass # do nothing if highest bin and shift is positive
                    else: # energy migration
                        bin_cut = (experiment.EReco >= ebins[i]) * (experiment.EReco >= ebins[i+1]) # select events with energy in bin
                        events_in_bin = np.sum(bin_cut)
                        escale[bin_cut] = 1 + x
                        if x>0: # shifts energy upwards and so does the event migration
                            bin_cut_above = (experiment.EReco >= ebins[i+1]) * (experiment.EReco >= ebins[i+2]) # select events with energy in bin
                            events_in_bin_above = np.sum(bin_cut_above)
                            escale[bin_cut_above] = 1 - x * events_in_bin / events_in_bin_above
                        elif x<=0: # shifts energy upwards and so does the event migration
                            bin_cut_below = (experiment.EReco >= ebins[i-1]) * (experiment.EReco >= ebins[i]) # select events with energy in bin
                            events_in_bin_below = np.sum(bin_cut_below)
                            escale[bin_cut_below] = 1 - x * events_in_bin / events_in_bin_below
        return escale
            
    def diff_energy_scale(self, experiment, x):
        """See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.diff_energy_scale`."""
        escale = np.zeros(experiment.NumberOfEvents)
        for sample in experiment.Samples:
            ebins = experiment.EnergyBins[sample]
            if len(ebins)>1:
                for i in range(ebins.size):
                    if i==0 and x<0:
                        pass # do nothing if lowest bin and shift is negative
                    elif (i==ebins.size-1) and (x>0):
                        pass # do nothing if highest bin and shift is positive
                    else: # energy migration
                        bin_cut = (experiment.EReco >= ebins[i]) * (experiment.EReco >= ebins[i+1]) # select events with energy in bin
                        events_in_bin = np.sum(bin_cut)
                        escale[bin_cut] = 1
                        if x>0: # shifts energy upwards and so does the event migration
                            bin_cut_above = (experiment.EReco >= ebins[i+1]) * (experiment.EReco >= ebins[i+2]) # select events with energy in bin
                            events_in_bin_above = np.sum(bin_cut_above)
                            escale[bin_cut_above] = events_in_bin / events_in_bin_above
                        elif x<=0: # shifts energy upwards and so does the event migration
                            bin_cut_below = (experiment.EReco >= ebins[i-1]) * (experiment.EReco >= ebins[i]) # select events with energy in bin
                            events_in_bin_below = np.sum(bin_cut_below)
                            escale[bin_cut_below] = events_in_bin / events_in_bin_below
        return escale

    def fiducial_volume(self, experiment, x):
        r"""Method changing the efficiency of the fiducial volume cut.
        NOTE: Currently, it applies a normalization factor on all events. More precise implementation coming soon.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if self._unphysical_value(x):
            return 1e-3
        return x

    def diff_fiducial_volume(self, experiment, x):
        r"""Method for computing the derivative of the weights w.r.t. the tuning parameter of the fiducial volumen.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the derivative of the `fiducial_volume` weights.
        """
        if self._unphysical_value(x):
            return 0
        return 1

    def subgev_2ring_pi0(self, experiment, x):
        r"""Method changing the fraction of 2-ring $\pi^0$-like events.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        pi02r = np.ones(experiment.NumberOfEvents)
        if self._unphysical_value(x):
            pi02r[experiment.Sample == 6] = 1e-3
        else:
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
        pi02r = np.zeros(experiment.NumberOfEvents)
        if self._unphysical_value(x):
            pi02r[experiment.Sample == 6] = 0
        else:
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
        fcpc = np.ones(experiment.NumberOfEvents)

        pc = (experiment.Sample == 14) | (experiment.Sample == 15)
        um = (experiment.Sample >= 16) & (experiment.Sample <= 18)
        fc = np.logical_not((pc | um))

        wfc = np.sum(fc)
        wpc = np.sum(pc)

        if self._unphysical_value(x):
            fcpc[fc] = 1e-3
            y = (wpc + wfc) / wpc
            fcpc[pc] = y
        else:
            fcpc[fc] = x
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
        fcpc = np.zeros(experiment.NumberOfEvents)

        pc = (experiment.Sample == 14) | (experiment.Sample == 15)
        um = (experiment.Sample >= 16) & (experiment.Sample <= 18)
        fc = np.logical_not((pc | um))

        wfc = np.sum(fc)
        wpc = np.sum(pc)

        if self._unphysical_value(x):
            fcpc[fc] = 0
            fcpc[pc] = 0
        else:
            fcpc[fc] = 1
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
        fc = np.ones(experiment.NumberOfEvents)
        pc = (experiment.Sample == 14) | (experiment.Sample == 15)
        um = (experiment.Sample >= 16) & (experiment.Sample <= 18)
        if self._unphysical_value(x):
            fc[np.logical_not((pc | um))] = 1e-3
        else:
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
        if self._unphysical_value(x):
            return 0
        fc = np.zeros(experiment.NumberOfEvents)
        pc = (experiment.Sample == 14) | (experiment.Sample == 15)
        um = (experiment.Sample >= 16) & (experiment.Sample <= 18)
        if self._unphysical_value(x):
            fc[np.logical_not((pc | um))] = 0
        else:
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
        w = np.ones(experiment.NumberOfEvents)
        pc = (experiment.Sample == 14) | (experiment.Sample == 15)
        if self._unphysical_value(x):
            w[pc] = 1e-3
        else:
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
        w = np.zeros(experiment.NumberOfEvents)
        pc = (experiment.Sample == 14) | (experiment.Sample == 15)
        if self._unphysical_value(x):
            w[pc] = 0
        else:
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
        pi01r = np.ones(experiment.NumberOfEvents)
        if self._unphysical_value(x):
            pi01r[experiment.Sample == 2] = 1e-3
        else:
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
        if self._unphysical_value(x):
            return 0
        pi01r = np.zeros(experiment.NumberOfEvents)
        pi01r[experiment.Sample == 2] = 1
        return pi01r

    def mre_nonubkg(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        w = np.ones(experiment.NumberOfEvents)
        mge = (
            (experiment.Sample == 10)
            | (experiment.Sample == 11)
            | (experiment.Sample == 12)
            | (experiment.Sample == 13)
        )
        w[mge] = x
        return w

    def diff_mre_nonubkg(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        w = np.zeros(experiment.NumberOfEvents)
        mge = (
            (experiment.Sample == 10)
            | (experiment.Sample == 11)
            | (experiment.Sample == 12)
            | (experiment.Sample == 13)
        )
        w[mge] = 1
        return w

    def mge_nonubkg(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
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
        if self._unphysical_value(x):
            return 0
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
        if self._unphysical_value(x):
            return 1e-3
        mr = np.ones(experiment.NumberOfEvents)
        r = self._migration_ratio(experiment, 10, 11)
        mr[experiment.Sample == 10] = x
        mr[experiment.Sample == 11] = 1 + r * (1 - x)
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
        if self._unphysical_value(x):
            return 0
        mr = np.zeros(experiment.NumberOfEvents)
        r = self._migration_ratio(experiment, 10, 11)
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
        if self._unphysical_value(x):
            return 1e-3
        e0 = 10
        e1 = 11
        e2 = 13
        mu = 12
        mr = np.ones(experiment.NumberOfEvents)
        r = self._migration_ratio(experiment, [e0, e1, e2], mu)
        mr[experiment.Sample == e0] = x
        mr[experiment.Sample == e1] = x
        mr[experiment.Sample == e2] = x
        mr[experiment.Sample == mu] = 1 + r * (1 - x)
        if self._unphysical_value(2 - x):
            return 1e-3
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
        if self._unphysical_value(x):
            return 0
        e0 = 10
        e1 = 11
        e2 = 13
        mu = 12
        mr = np.zeros(experiment.NumberOfEvents)
        r = self._migration_ratio(experiment, [e0, e1, e2], mu)
        mr[experiment.Sample == e0] = 1
        mr[experiment.Sample == e1] = 1
        mr[experiment.Sample == e2] = 1
        mr[experiment.Sample == mu] = -r
        if self._unphysical_value(1 + r * (1 - x)):
            return 0
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
        if self._unphysical_value(x):
            return 1e-3
        e0 = 10
        e1 = 11
        o0 = 13
        mr = np.ones(experiment.NumberOfEvents)
        r = self._migration_ratio(experiment, [e0, e1], o0)
        mr[experiment.Sample == e0] = x
        mr[experiment.Sample == e1] = x
        mr[experiment.Sample == o0] = 1 + r * (1 - x)
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
        if self._unphysical_value(x):
            return 0
        e0 = 10
        e1 = 11
        o0 = 13
        mr = np.zeros(experiment.NumberOfEvents)
        r = self._migration_ratio(experiment, [e0, e1], o0)
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
        if self._unphysical_value(x):
            return 1e-3
        pcs = 14
        pct = 15
        mr = np.ones(experiment.NumberOfEvents)
        r = self._migration_ratio(experiment, pcs, pct)
        mr[experiment.Sample == pcs] = x
        mr[experiment.Sample == pct] = 1 + r * (1 - x)
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
        if self._unphysical_value(x):
            return 0
        pcs = 14
        pct = 15
        mr = np.zeros(experiment.NumberOfEvents)
        r = self._migration_ratio(experiment, pcs, pct)
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
        if self._unphysical_value(x):
            return 1e-3
        r1 = 2
        r2 = 6
        mr = np.ones(experiment.NumberOfEvents)
        r = self._migration_ratio(experiment, r1, r2)
        mr[experiment.Sample == r1] = x
        mr[experiment.Sample == r2] = 1 + r * (1 - x)
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
        if self._unphysical_value(x):
            return 0
        r1 = 2
        r2 = 6
        mr = np.zeros(experiment.NumberOfEvents)
        r = self._migration_ratio(experiment, r1, r2)
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
        if self._unphysical_value(x):
            return 1e-3
        r1 = [0, 1, 7, 8, 19, 20, 21]
        r2 = [10, 11, 13, 24, 25, 26]
        mr = np.ones(experiment.NumberOfEvents)
        r = self._migration_ratio(experiment, r1, r2)
        for sample in r1:
            mr[experiment.Sample == sample] = x
        for sample in r2:
            mr[experiment.Sample == sample] = 1 + r * (1 - x)
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
        if self._unphysical_value(x):
            return 0
        r1 = [0, 1, 7, 8, 19, 20, 21]
        r2 = [10, 11, 13, 24, 25, 26]
        mr = np.zeros(experiment.NumberOfEvents)
        r = self._migration_ratio(experiment, r1, r2)
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
        if self._unphysical_value(x):
            return 1e-3
        r1 = [3, 4, 5, 9, 22, 23, 27, 28]
        r2 = [12]
        mr = np.ones(experiment.NumberOfEvents)
        r = self._migration_ratio(experiment, r1, r2)
        for sample in r1:
            mr[experiment.Sample == sample] = x
        for sample in r2:
            mr[experiment.Sample == sample] = 1 + r * (1 - x)
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
        if self._unphysical_value(x):
            return 0
        r1 = [3, 4, 5, 9, 22, 23, 27, 28]
        r2 = [12]
        mr = np.zeros(experiment.NumberOfEvents)
        r = self._migration_ratio(experiment, r1, r2)
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
        if self._unphysical_value(x):
            return 1e-3
        e = [0, 1, 7, 8, 19, 20, 21, 24, 25, 26]
        mu = [3, 4, 5, 9, 22, 23, 27, 28]
        mr = np.ones(experiment.NumberOfEvents)
        r = self._migration_ratio(experiment, e, mu)
        for sample in e:
            mr[experiment.Sample == sample] = x
        for sample in mu:
            mr[experiment.Sample == sample] = 1 + r * (1 - x)
        if self._unphysical_value(1 + r * (1 - x)):
            return 1e-3
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
        if self._unphysical_value(x):
            return 0
        if np.abs(1 - x) < 1e-4:
            x = 1
        e = [0, 1, 7, 8, 19, 20, 21, 24, 25, 26]
        mu = [3, 4, 5, 9, 22, 23, 27, 28]
        mr = np.zeros(experiment.NumberOfEvents)
        r = self._migration_ratio(experiment, e, mu)
        for sample in e:
            mr[experiment.Sample == sample] = 1
        for sample in mu:
            mr[experiment.Sample == sample] = -r
        if self._unphysical_value(1 + r * (1 - x)):
            return 0
        return mr

    def multiring_pid(self, experiment, x):
        r"""Method changing the particle identification efficiency of multi-ring events.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if self._unphysical_value(x):
            return 1e-3
        if np.abs(1 - x) < 1e-4:
            x = 1
        e = [10, 11, 13]
        mu = [12]
        mr = np.ones(experiment.NumberOfEvents)
        r = self._migration_ratio(experiment, e, mu)
        for sample in e:
            mr[experiment.Sample == sample] = x
        for sample in mu:
            mr[experiment.Sample == sample] = 1 + r * (1 - x)
        if self._unphysical_value(1 + r * (1 - x)):
            return 1e-3
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
        if self._unphysical_value(x):
            return 0
        if np.abs(1 - x) < 1e-4:
            x = 1
        e = [10, 11, 13]
        mu = [12]
        mr = np.zeros(experiment.NumberOfEvents)
        r = self._migration_ratio(experiment, e, mu)
        for sample in e:
            mr[experiment.Sample == sample] = 1
        for sample in mu:
            mr[experiment.Sample == sample] = -r
        if self._unphysical_value(1 + r * (1 - x)):
            return 0
        return mr

    def neutron_tagging(self, experiment, x):
        r"""Method changing the efficiency of neutron tagging.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if self._unphysical_value(x):
            return 1e-3
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
        r = self._mask_ratio(experiment, nn0, nn1)
        nn[nn0] = x
        nn[nn1] = 1 + r * (1 - x)
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
        if self._unphysical_value(x):
            return 0
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
        r = self._mask_ratio(experiment, nn0, nn1)
        nn[nn0] = 1
        nn[nn1] = -r
        return nn

    # --- Era-split neutron tagging (PE/Newtrinos.jl granularity) -------------
    # Newtrinos.jl uses two independent n-tag efficiency dofs
    # (sk_iv_v_subgev_neutron_tag_eff, sk_iv_v_multigev_neutron_tag_eff, both
    # N(1, 0.12); super_k.jl:319-320) where the shared `neutron_tagging` above
    # ties SubGeV+MGeV (and e-like+mu-like) to one parameter. Same
    # rate-conserving migration form as `neutron_tagging`, restricted per era.
    # Sample ids: SubGeV untagged {20 (ebar 0n), 22 (numu)} <-> tagged
    # {21 (ebar 1n), 23 (numubar)}; MGeV untagged {25, 27} <-> tagged {26, 28}.

    def _neutron_tagging_era(self, experiment, x, s0, s1, diff=False):
        nn0 = (experiment.Sample == s0[0]) | (experiment.Sample == s0[1])
        nn1 = (experiment.Sample == s1[0]) | (experiment.Sample == s1[1])
        r = self._mask_ratio(experiment, nn0, nn1)
        if diff:
            nn = np.zeros(experiment.NumberOfEvents)
            nn[nn0] = 1
            nn[nn1] = -r
        else:
            nn = np.ones(experiment.NumberOfEvents)
            nn[nn0] = x
            nn[nn1] = 1 + r * (1 - x)
        return nn

    def neutron_tagging_subgev(self, experiment, x):
        """SubGeV-only neutron-tagging efficiency (samples 20/22 <-> 21/23)."""
        if self._unphysical_value(x):
            return 1e-3
        return self._neutron_tagging_era(experiment, x, (20, 22), (21, 23))

    def diff_neutron_tagging_subgev(self, experiment, x):
        """Derivative of `neutron_tagging_subgev` w.r.t. the tuning parameter."""
        if self._unphysical_value(x):
            return 0
        return self._neutron_tagging_era(experiment, x, (20, 22), (21, 23), diff=True)

    def neutron_tagging_multigev(self, experiment, x):
        """MGeV-only neutron-tagging efficiency (samples 25/27 <-> 26/28)."""
        if self._unphysical_value(x):
            return 1e-3
        return self._neutron_tagging_era(experiment, x, (25, 27), (26, 28))

    def diff_neutron_tagging_multigev(self, experiment, x):
        """Derivative of `neutron_tagging_multigev` w.r.t. the tuning parameter."""
        if self._unphysical_value(x):
            return 0
        return self._neutron_tagging_era(experiment, x, (25, 27), (26, 28), diff=True)

    def decay_e_tagging(self, experiment, x):
        r"""Method changing the efficiency of decay electron tagging.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if self._unphysical_value(x):
            return 1e-3
        mue = np.ones(experiment.NumberOfEvents)
        W = self._rate_weight(experiment)
        n0 = np.sum(W[experiment.DecayE < 1])
        n1 = np.sum(W[(experiment.DecayE >= 1) & (experiment.DecayE < 2)])
        n2 = np.sum(W[experiment.DecayE >= 2])
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
        if self._unphysical_value(x):
            return 0
        mue = np.zeros(experiment.NumberOfEvents)
        W = self._rate_weight(experiment)
        n0 = np.sum(W[experiment.DecayE < 1])
        n1 = np.sum(W[(experiment.DecayE >= 1) & (experiment.DecayE < 2)])
        n2 = np.sum(W[experiment.DecayE >= 2])
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
        if self._unphysical_value(x):
            return 1e-3
        um = np.ones(experiment.NumberOfEvents)
        r = self._migration_ratio(experiment, 18, 17)
        if self._unphysical_value(1 + r * (1 - x)):
            return 1e-3
        um[experiment.Sample == 18] = x
        um[experiment.Sample == 17] = 1 + r * (1 - x)
        return um

    def diff_upmu_shower_separation(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        um = np.zeros(experiment.NumberOfEvents)
        r = self._migration_ratio(experiment, 18, 17)
        if self._unphysical_value(1 + r * (1 - x)):
            return 0
        um[experiment.Sample == 18] = 1
        um[experiment.Sample == 17] = -r
        return um

    def upmu_stop_bkg(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        um = np.ones(experiment.NumberOfEvents)
        um[experiment.Sample == 16] = x
        return um

    def diff_upmu_stop_bkg(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        um = np.zeros(experiment.NumberOfEvents)
        um[experiment.Sample == 16] = 1
        return um

    def upmu_showering_bkg(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        um = np.ones(experiment.NumberOfEvents)
        um[experiment.Sample == 18] = x
        return um

    def diff_upmu_showering_bkg(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        um = np.zeros(experiment.NumberOfEvents)
        um[experiment.Sample == 18] = 1
        return um

    def upmu_nonshowering_bkg(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        um = np.ones(experiment.NumberOfEvents)
        um[experiment.Sample == 17] = x
        return um

    def diff_upmu_nonshowering_bkg(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        um = np.zeros(experiment.NumberOfEvents)
        um[experiment.Sample == 17] = 1
        return um

    def subgev_numulike_sk45_mc(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        sgm = np.ones(experiment.NumberOfEvents)
        sgm[experiment.Sample == 27] = x
        return sgm

    def diff_subgev_numulike_sk45_mc(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        sgm = np.zeros(experiment.NumberOfEvents)
        sgm[experiment.Sample == 27] = 1
        return sgm