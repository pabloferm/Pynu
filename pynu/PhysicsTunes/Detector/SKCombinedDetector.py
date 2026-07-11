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
    #:                           (physics-weighted, rate-conserving convention)
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
        W = self._rate_weight(experiment)
        n0 = np.sum(W[np.isin(experiment.Sample, np.atleast_1d(donor))])
        n1 = np.sum(W[np.isin(experiment.Sample, np.atleast_1d(acceptor))])
        return n0 / n1

    def _mask_ratio(self, experiment, mask0, mask1):
        """Weighted-rate ratio r from boolean event masks (used by neutron_tagging)."""
        import numpy as np
        W = self._rate_weight(experiment)
        return np.sum(W[mask1]) / np.sum(W[mask0])

    def energy_scale(self, experiment, x):
        """See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.energy_scale`.

        Minimal bin-cut bugfix, keeping the existing weight-emulation approach.
        Fixes: (1) bin membership is half-open [ebins[i], ebins[i+1]) — the
        pre-fix `(EReco>=lo)*(EReco>=hi)` selected everything above the UPPER
        edge; (2) the loop now runs over BINS (ebins.size-1), so ebins[i+1]
        never overruns; (3) the acceptor-bin index (ebins[i+2] up / ebins[i-1]
        down) is bounds-guarded; (4) empty-acceptor divides are guarded. This
        body is DEAD REFERENCE only — the active era wrappers below return
        identity and the histogram-level operator owns the transfer (see the
        block comment below).
        """
        escale = np.ones(experiment.NumberOfEvents)
        for sample in experiment.Samples:
            ebins = experiment.EnergyBins[sample]
            nb = ebins.size - 1                       # number of reco-E bins
            if nb < 1:
                continue
            for i in range(nb):
                if i == 0 and x < 0:
                    continue  # lowest bin, downward shift: nothing below to fill
                if i == nb - 1 and x > 0:
                    continue  # highest bin, upward shift: nothing above to fill
                bin_cut = (experiment.EReco >= ebins[i]) & (experiment.EReco < ebins[i+1])
                events_in_bin = np.sum(bin_cut)
                escale[bin_cut] = 1 + x
                if x > 0 and i + 2 <= ebins.size - 1:   # acceptor bin above exists
                    bin_cut_above = ((experiment.EReco >= ebins[i+1])
                                     & (experiment.EReco < ebins[i+2]))
                    n_above = np.sum(bin_cut_above)
                    if n_above > 0:
                        escale[bin_cut_above] = 1 - x * events_in_bin / n_above
                elif x <= 0 and i - 1 >= 0:             # acceptor bin below exists
                    bin_cut_below = ((experiment.EReco >= ebins[i-1])
                                     & (experiment.EReco < ebins[i]))
                    n_below = np.sum(bin_cut_below)
                    if n_below > 0:
                        escale[bin_cut_below] = 1 - x * events_in_bin / n_below
        return escale

    def diff_energy_scale(self, experiment, x):
        """See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.diff_energy_scale`.

        Same minimal bin-cut bugfix as `energy_scale`; same DEAD-REFERENCE status."""
        escale = np.zeros(experiment.NumberOfEvents)
        for sample in experiment.Samples:
            ebins = experiment.EnergyBins[sample]
            nb = ebins.size - 1
            if nb < 1:
                continue
            for i in range(nb):
                if i == 0 and x < 0:
                    continue
                if i == nb - 1 and x > 0:
                    continue
                bin_cut = (experiment.EReco >= ebins[i]) & (experiment.EReco < ebins[i+1])
                events_in_bin = np.sum(bin_cut)
                escale[bin_cut] = 1
                # compensation weight is 1 - x*n_src/n_acc, so its derivative is
                # -n_src/n_acc (the pre-fix diff was missing this minus sign, so
                # the analytic gradient had the WRONG SIGN on the acceptor bin).
                if x > 0 and i + 2 <= ebins.size - 1:
                    bin_cut_above = ((experiment.EReco >= ebins[i+1])
                                     & (experiment.EReco < ebins[i+2]))
                    n_above = np.sum(bin_cut_above)
                    if n_above > 0:
                        escale[bin_cut_above] = -events_in_bin / n_above
                elif x <= 0 and i - 1 >= 0:
                    bin_cut_below = ((experiment.EReco >= ebins[i-1])
                                     & (experiment.EReco < ebins[i]))
                    n_below = np.sum(bin_cut_below)
                    if n_below > 0:
                        escale[bin_cut_below] = -events_in_bin / n_below
        return escale

    # ---- energy_scale era wrappers: per-event weight-emulation RETIRED --------
    # The event side adopts the SAME histogram-level transfer as the binned
    # engine (pynu/binned/escale_operator.py, transcribed from
    # sk_binned_engine._escale_migrate), applied POST-binning to the binned
    # expectation. The per-event weight-emulation above (energy_scale /
    # diff_energy_scale) can NEVER bit-match the binned histogram transfer on the
    # quantized SK public MC (one EReco value per bin -> no sub-bin structure to
    # reweight), so it is retired: the era wrappers now return IDENTITY (weight 1,
    # diff 0) so they contribute nothing to the per-event weight product, and the
    # histogram operator owns the transfer at SetBinnedExpectedEvents time.
    #
    # The base energy_scale / diff_energy_scale bodies above are kept as DEAD
    # reference (still importable, no longer routed to by the active wrappers).
    def energy_scale_sk1(self, experiment, x):
        return np.ones(experiment.NumberOfEvents)

    def diff_energy_scale_sk1(self, experiment, x):
        return np.zeros(experiment.NumberOfEvents)

    def energy_scale_sk2(self, experiment, x):
        return np.ones(experiment.NumberOfEvents)

    def diff_energy_scale_sk2(self, experiment, x):
        return np.zeros(experiment.NumberOfEvents)

    def energy_scale_sk3(self, experiment, x):
        return np.ones(experiment.NumberOfEvents)

    def diff_energy_scale_sk3(self, experiment, x):
        return np.zeros(experiment.NumberOfEvents)

    def energy_scale_sk45(self, experiment, x):
        return np.ones(experiment.NumberOfEvents)

    def diff_energy_scale_sk45(self, experiment, x):
        return np.zeros(experiment.NumberOfEvents)

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

    def subgev_2ring_pi0_sk1(self, experiment, x):
        w = self.subgev_2ring_pi0(experiment, x)
        w[experiment.SKPhase != 1] = 1
        return w

    def diff_subgev_2ring_pi0_sk1(self, experiment, x):
        w = self.diff_subgev_2ring_pi0(experiment, x)
        w[experiment.SKPhase != 1] = 0
        return w

    def subgev_2ring_pi0_sk2(self, experiment, x):
        w = self.subgev_2ring_pi0(experiment, x)
        w[experiment.SKPhase != 2] = 1
        return w

    def diff_subgev_2ring_pi0_sk2(self, experiment, x):
        w = self.diff_subgev_2ring_pi0(experiment, x)
        w[experiment.SKPhase != 2] = 0
        return w

    def subgev_2ring_pi0_sk3(self, experiment, x):
        w = self.subgev_2ring_pi0(experiment, x)
        w[experiment.SKPhase != 3] = 1
        return w

    def diff_subgev_2ring_pi0_sk3(self, experiment, x):
        w = self.diff_subgev_2ring_pi0(experiment, x)
        w[experiment.SKPhase != 3] = 0
        return w

    def subgev_2ring_pi0_sk45(self, experiment, x):
        w = self.subgev_2ring_pi0(experiment, x)
        w[experiment.SKPhase < 4] = 1
        return w

    def diff_subgev_2ring_pi0_sk45(self, experiment, x):
        w = self.diff_subgev_2ring_pi0(experiment, x)
        w[experiment.SKPhase < 4] = 0
        return w

    def fcpc_separation(self, experiment, x):
        r"""Method changing the efficiency of the fully and partially-contained events in SK.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        # logging.info(f"Entering {__name__}")
        # Rate-conserving FC/PC migration, algebra matching the binned engine
        # (sk_binned_engine.py:1296-1314): FC leg *= x,
        # PC leg *= y = ((wpc+wfc) - x*wfc)/wpc = 1 + (wfc/wpc)(1-x). The ratio
        # in y is wfc/wpc (NOT wpc/wfc).
        fcpc = np.ones(experiment.NumberOfEvents)

        pc = (experiment.Sample == 14) | (experiment.Sample == 15)
        um = (experiment.Sample >= 16) & (experiment.Sample <= 18)
        fc = np.logical_not((pc | um))

        wfc = np.sum(fc)
        wpc = np.sum(pc)
        r = wfc/wpc

        if self._unphysical_value(x):
            fcpc[fc] = 1e-3
            y = (wpc + wfc) / wpc
            fcpc[pc] = y
        else:
            fcpc[fc] = x
            y = ((wpc + wfc) - x * wfc) / wpc   # == 1 + r*(1-x), r = wfc/wpc
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
            # dy/dx = -r with r = wfc/wpc (matches the binned engine, ENG:1314
            # dS_pc = (-wfc/wpc)/y; the fitter divides by the weight, so the
            # raw derivative returned here is -wfc/wpc).
            y = -wfc / wpc
            fcpc[pc] = y

        return fcpc

    def fcpc_separation_sk1(self, experiment, x):
        w = self.fcpc_separation(experiment, x)
        w[experiment.SKPhase != 1] = 1
        return w

    def diff_fcpc_separation_sk1(self, experiment, x):
        w = self.diff_fcpc_separation(experiment, x)
        w[experiment.SKPhase != 1] = 0
        return w

    def fcpc_separation_sk2(self, experiment, x):
        w = self.fcpc_separation(experiment, x)
        w[experiment.SKPhase != 2] = 1
        return w

    def diff_fcpc_separation_sk2(self, experiment, x):
        w = self.diff_fcpc_separation(experiment, x)
        w[experiment.SKPhase != 2] = 0
        return w

    def fcpc_separation_sk3(self, experiment, x):
        w = self.fcpc_separation(experiment, x)
        w[experiment.SKPhase != 3] = 1
        return w

    def diff_fcpc_separation_sk3(self, experiment, x):
        w = self.diff_fcpc_separation(experiment, x)
        w[experiment.SKPhase != 3] = 0
        return w

    def fcpc_separation_sk45(self, experiment, x):
        w = self.fcpc_separation(experiment, x)
        w[experiment.SKPhase < 4] = 1
        return w

    def diff_fcpc_separation_sk45(self, experiment, x):
        w = self.diff_fcpc_separation(experiment, x)
        w[experiment.SKPhase < 4] = 0
        return w

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

    def fc_reduction_sk1(self, experiment, x):
        w = self.fc_reduction(experiment, x)
        w[experiment.SKPhase != 1] = 1
        return w

    def diff_fc_reduction_sk1(self, experiment, x):
        w = self.diff_fc_reduction(experiment, x)
        w[experiment.SKPhase != 1] = 0
        return w

    def fc_reduction_sk2(self, experiment, x):
        w = self.fc_reduction(experiment, x)
        w[experiment.SKPhase != 2] = 1
        return w

    def diff_fc_reduction_sk2(self, experiment, x):
        w = self.diff_fc_reduction(experiment, x)
        w[experiment.SKPhase != 2] = 0
        return w

    def fc_reduction_sk3(self, experiment, x):
        w = self.fc_reduction(experiment, x)
        w[experiment.SKPhase != 3] = 1
        return w

    def diff_fc_reduction_sk3(self, experiment, x):
        w = self.diff_fc_reduction(experiment, x)
        w[experiment.SKPhase != 3] = 0
        return w

    def fc_reduction_sk45(self, experiment, x):
        w = self.fc_reduction(experiment, x)
        w[experiment.SKPhase < 4] = 1
        return w

    def diff_fc_reduction_sk45(self, experiment, x):
        w = self.diff_fc_reduction(experiment, x)
        w[experiment.SKPhase < 4] = 0
        return w

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

    def pc_reduction_sk1(self, experiment, x):
        w = self.pc_reduction(experiment, x)
        w[experiment.SKPhase != 1] = 1
        return w

    def diff_pc_reduction_sk1(self, experiment, x):
        w = self.diff_pc_reduction(experiment, x)
        w[experiment.SKPhase != 1] = 0
        return w

    def pc_reduction_sk2(self, experiment, x):
        w = self.pc_reduction(experiment, x)
        w[experiment.SKPhase != 2] = 1
        return w

    def diff_pc_reduction_sk2(self, experiment, x):
        w = self.diff_pc_reduction(experiment, x)
        w[experiment.SKPhase != 2] = 0
        return w

    def pc_reduction_sk3(self, experiment, x):
        w = self.pc_reduction(experiment, x)
        w[experiment.SKPhase != 3] = 1
        return w

    def diff_pc_reduction_sk3(self, experiment, x):
        w = self.diff_pc_reduction(experiment, x)
        w[experiment.SKPhase != 3] = 0
        return w

    def pc_reduction_sk45(self, experiment, x):
        w = self.pc_reduction(experiment, x)
        w[experiment.SKPhase < 4] = 1
        return w

    def diff_pc_reduction_sk45(self, experiment, x):
        w = self.diff_pc_reduction(experiment, x)
        w[experiment.SKPhase < 4] = 0
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

    def subgev_1ring_pi0_sk1(self, experiment, x):
        w = self.subgev_1ring_pi0(experiment, x)
        w[experiment.SKPhase != 1] = 1
        return w

    def diff_subgev_1ring_pi0_sk1(self, experiment, x):
        w = self.diff_subgev_1ring_pi0(experiment, x)
        w[experiment.SKPhase != 1] = 0
        return w

    def subgev_1ring_pi0_sk2(self, experiment, x):
        w = self.subgev_1ring_pi0(experiment, x)
        w[experiment.SKPhase != 2] = 1
        return w

    def diff_subgev_1ring_pi0_sk2(self, experiment, x):
        w = self.diff_subgev_1ring_pi0(experiment, x)
        w[experiment.SKPhase != 2] = 0
        return w

    def subgev_1ring_pi0_sk3(self, experiment, x):
        w = self.subgev_1ring_pi0(experiment, x)
        w[experiment.SKPhase != 3] = 1
        return w

    def diff_subgev_1ring_pi0_sk3(self, experiment, x):
        w = self.diff_subgev_1ring_pi0(experiment, x)
        w[experiment.SKPhase != 3] = 0
        return w

    def subgev_1ring_pi0_sk45(self, experiment, x):
        w = self.subgev_1ring_pi0(experiment, x)
        w[experiment.SKPhase < 4] = 1
        return w

    def diff_subgev_1ring_pi0_sk45(self, experiment, x):
        w = self.diff_subgev_1ring_pi0(experiment, x)
        w[experiment.SKPhase < 4] = 0
        return w

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

    def mre_nonubkg_sk1(self, experiment, x):
        w = self.mre_nonubkg(experiment, x)
        w[experiment.SKPhase != 1] = 1
        return w

    def diff_mre_nonubkg_sk1(self, experiment, x):
        w = self.diff_mre_nonubkg(experiment, x)
        w[experiment.SKPhase != 1] = 0
        return w

    def mre_nonubkg_sk2(self, experiment, x):
        w = self.mre_nonubkg(experiment, x)
        w[experiment.SKPhase != 2] = 1
        return w

    def diff_mre_nonubkg_sk2(self, experiment, x):
        w = self.diff_mre_nonubkg(experiment, x)
        w[experiment.SKPhase != 2] = 0
        return w

    def mre_nonubkg_sk3(self, experiment, x):
        w = self.mre_nonubkg(experiment, x)
        w[experiment.SKPhase != 3] = 1
        return w

    def diff_mre_nonubkg_sk3(self, experiment, x):
        w = self.diff_mre_nonubkg(experiment, x)
        w[experiment.SKPhase != 3] = 0
        return w

    def mre_nonubkg_sk45(self, experiment, x):
        w = self.mre_nonubkg(experiment, x)
        w[experiment.SKPhase < 4] = 1
        return w

    def diff_mre_nonubkg_sk45(self, experiment, x):
        w = self.diff_mre_nonubkg(experiment, x)
        w[experiment.SKPhase < 4] = 0
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

    def mge_nonubkg_sk1(self, experiment, x):
        w = self.mge_nonubkg(experiment, x)
        w[experiment.SKPhase != 1] = 1
        return w

    def diff_mge_nonubkg_sk1(self, experiment, x):
        w = self.diff_mge_nonubkg(experiment, x)
        w[experiment.SKPhase != 1] = 0
        return w

    def mge_nonubkg_sk2(self, experiment, x):
        w = self.mge_nonubkg(experiment, x)
        w[experiment.SKPhase != 2] = 1
        return w

    def diff_mge_nonubkg_sk2(self, experiment, x):
        w = self.diff_mge_nonubkg(experiment, x)
        w[experiment.SKPhase != 2] = 0
        return w

    def mge_nonubkg_sk3(self, experiment, x):
        w = self.mge_nonubkg(experiment, x)
        w[experiment.SKPhase != 3] = 1
        return w

    def diff_mge_nonubkg_sk3(self, experiment, x):
        w = self.diff_mge_nonubkg(experiment, x)
        w[experiment.SKPhase != 3] = 0
        return w

    def mge_nonubkg_sk45(self, experiment, x):
        w = self.mge_nonubkg(experiment, x)
        w[experiment.SKPhase < 4] = 1
        return w

    def diff_mge_nonubkg_sk45(self, experiment, x):
        w = self.diff_mge_nonubkg(experiment, x)
        w[experiment.SKPhase < 4] = 0
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

    def multiring_nunubar_separation_sk1(self, experiment, x):
        w = self.multiring_nunubar_separation(experiment, x)
        w[experiment.SKPhase != 1] = 1
        return w

    def diff_multiring_nunubar_separation_sk1(self, experiment, x):
        w = self.diff_multiring_nunubar_separation(experiment, x)
        w[experiment.SKPhase != 1] = 0
        return w

    def multiring_nunubar_separation_sk2(self, experiment, x):
        w = self.multiring_nunubar_separation(experiment, x)
        w[experiment.SKPhase != 2] = 1
        return w

    def diff_multiring_nunubar_separation_sk2(self, experiment, x):
        w = self.diff_multiring_nunubar_separation(experiment, x)
        w[experiment.SKPhase != 2] = 0
        return w

    def multiring_nunubar_separation_sk3(self, experiment, x):
        w = self.multiring_nunubar_separation(experiment, x)
        w[experiment.SKPhase != 3] = 1
        return w

    def diff_multiring_nunubar_separation_sk3(self, experiment, x):
        w = self.diff_multiring_nunubar_separation(experiment, x)
        w[experiment.SKPhase != 3] = 0
        return w

    def multiring_nunubar_separation_sk45(self, experiment, x):
        w = self.multiring_nunubar_separation(experiment, x)
        w[experiment.SKPhase < 4] = 1
        return w

    def diff_multiring_nunubar_separation_sk45(self, experiment, x):
        w = self.diff_multiring_nunubar_separation(experiment, x)
        w[experiment.SKPhase < 4] = 0
        return w

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

    def multiring_emu_separation_sk1(self, experiment, x):
        w = self.multiring_emu_separation(experiment, x)
        w[experiment.SKPhase != 1] = 1
        return w

    def diff_multiring_emu_separation_sk1(self, experiment, x):
        w = self.diff_multiring_emu_separation(experiment, x)
        w[experiment.SKPhase != 1] = 0
        return w

    def multiring_emu_separation_sk2(self, experiment, x):
        w = self.multiring_emu_separation(experiment, x)
        w[experiment.SKPhase != 2] = 1
        return w

    def diff_multiring_emu_separation_sk2(self, experiment, x):
        w = self.diff_multiring_emu_separation(experiment, x)
        w[experiment.SKPhase != 2] = 0
        return w

    def multiring_emu_separation_sk3(self, experiment, x):
        w = self.multiring_emu_separation(experiment, x)
        w[experiment.SKPhase != 3] = 1
        return w

    def diff_multiring_emu_separation_sk3(self, experiment, x):
        w = self.diff_multiring_emu_separation(experiment, x)
        w[experiment.SKPhase != 3] = 0
        return w

    def multiring_emu_separation_sk45(self, experiment, x):
        w = self.multiring_emu_separation(experiment, x)
        w[experiment.SKPhase < 4] = 1
        return w

    def diff_multiring_emu_separation_sk45(self, experiment, x):
        w = self.diff_multiring_emu_separation(experiment, x)
        w[experiment.SKPhase < 4] = 0
        return w

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

    def multiring_eother_separation_sk1(self, experiment, x):
        w = self.multiring_eother_separation(experiment, x)
        w[experiment.SKPhase != 1] = 1
        return w

    def diff_multiring_eother_separation_sk1(self, experiment, x):
        w = self.diff_multiring_eother_separation(experiment, x)
        w[experiment.SKPhase != 1] = 0
        return w

    def multiring_eother_separation_sk2(self, experiment, x):
        w = self.multiring_eother_separation(experiment, x)
        w[experiment.SKPhase != 2] = 1
        return w

    def diff_multiring_eother_separation_sk2(self, experiment, x):
        w = self.diff_multiring_eother_separation(experiment, x)
        w[experiment.SKPhase != 2] = 0
        return w

    def multiring_eother_separation_sk3(self, experiment, x):
        w = self.multiring_eother_separation(experiment, x)
        w[experiment.SKPhase != 3] = 1
        return w

    def diff_multiring_eother_separation_sk3(self, experiment, x):
        w = self.diff_multiring_eother_separation(experiment, x)
        w[experiment.SKPhase != 3] = 0
        return w

    def multiring_eother_separation_sk45(self, experiment, x):
        w = self.multiring_eother_separation(experiment, x)
        w[experiment.SKPhase < 4] = 1
        return w

    def diff_multiring_eother_separation_sk45(self, experiment, x):
        w = self.diff_multiring_eother_separation(experiment, x)
        w[experiment.SKPhase < 4] = 0
        return w

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

    def pc_stopthru_separation_sk1(self, experiment, x):
        w = self.pc_stopthru_separation(experiment, x)
        w[experiment.SKPhase != 1] = 1
        return w

    def diff_pc_stopthru_separation_sk1(self, experiment, x):
        w = self.diff_pc_stopthru_separation(experiment, x)
        w[experiment.SKPhase != 1] = 0
        return w

    def pc_stopthru_separation_sk2(self, experiment, x):
        w = self.pc_stopthru_separation(experiment, x)
        w[experiment.SKPhase != 2] = 1
        return w

    def diff_pc_stopthru_separation_sk2(self, experiment, x):
        w = self.diff_pc_stopthru_separation(experiment, x)
        w[experiment.SKPhase != 2] = 0
        return w

    def pc_stopthru_separation_sk3(self, experiment, x):
        w = self.pc_stopthru_separation(experiment, x)
        w[experiment.SKPhase != 3] = 1
        return w

    def diff_pc_stopthru_separation_sk3(self, experiment, x):
        w = self.diff_pc_stopthru_separation(experiment, x)
        w[experiment.SKPhase != 3] = 0
        return w

    def pc_stopthru_separation_sk45(self, experiment, x):
        w = self.pc_stopthru_separation(experiment, x)
        w[experiment.SKPhase < 4] = 1
        return w

    def diff_pc_stopthru_separation_sk45(self, experiment, x):
        w = self.diff_pc_stopthru_separation(experiment, x)
        w[experiment.SKPhase < 4] = 0
        return w

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

    def pi0_ring_separation_sk1(self, experiment, x):
        w = self.pi0_ring_separation(experiment, x)
        w[experiment.SKPhase != 1] = 1
        return w

    def diff_pi0_ring_separation_sk1(self, experiment, x):
        w = self.diff_pi0_ring_separation(experiment, x)
        w[experiment.SKPhase != 1] = 0
        return w

    def pi0_ring_separation_sk2(self, experiment, x):
        w = self.pi0_ring_separation(experiment, x)
        w[experiment.SKPhase != 2] = 1
        return w

    def diff_pi0_ring_separation_sk2(self, experiment, x):
        w = self.diff_pi0_ring_separation(experiment, x)
        w[experiment.SKPhase != 2] = 0
        return w

    def pi0_ring_separation_sk3(self, experiment, x):
        w = self.pi0_ring_separation(experiment, x)
        w[experiment.SKPhase != 3] = 1
        return w

    def diff_pi0_ring_separation_sk3(self, experiment, x):
        w = self.diff_pi0_ring_separation(experiment, x)
        w[experiment.SKPhase != 3] = 0
        return w

    def pi0_ring_separation_sk45(self, experiment, x):
        w = self.pi0_ring_separation(experiment, x)
        w[experiment.SKPhase < 4] = 1
        return w

    def diff_pi0_ring_separation_sk45(self, experiment, x):
        w = self.diff_pi0_ring_separation(experiment, x)
        w[experiment.SKPhase < 4] = 0
        return w

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

    def e_ring_separation_sk1(self, experiment, x):
        w = self.e_ring_separation(experiment, x)
        w[experiment.SKPhase != 1] = 1
        return w

    def diff_e_ring_separation_sk1(self, experiment, x):
        w = self.diff_e_ring_separation(experiment, x)
        w[experiment.SKPhase != 1] = 0
        return w

    def e_ring_separation_sk2(self, experiment, x):
        w = self.e_ring_separation(experiment, x)
        w[experiment.SKPhase != 2] = 1
        return w

    def diff_e_ring_separation_sk2(self, experiment, x):
        w = self.diff_e_ring_separation(experiment, x)
        w[experiment.SKPhase != 2] = 0
        return w

    def e_ring_separation_sk3(self, experiment, x):
        w = self.e_ring_separation(experiment, x)
        w[experiment.SKPhase != 3] = 1
        return w

    def diff_e_ring_separation_sk3(self, experiment, x):
        w = self.diff_e_ring_separation(experiment, x)
        w[experiment.SKPhase != 3] = 0
        return w

    def e_ring_separation_sk45(self, experiment, x):
        w = self.e_ring_separation(experiment, x)
        w[experiment.SKPhase < 4] = 1
        return w

    def diff_e_ring_separation_sk45(self, experiment, x):
        w = self.diff_e_ring_separation(experiment, x)
        w[experiment.SKPhase < 4] = 0
        return w

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

    def mu_ring_separation_sk1(self, experiment, x):
        w = self.mu_ring_separation(experiment, x)
        w[experiment.SKPhase != 1] = 1
        return w

    def diff_mu_ring_separation_sk1(self, experiment, x):
        w = self.diff_mu_ring_separation(experiment, x)
        w[experiment.SKPhase != 1] = 0
        return w

    def mu_ring_separation_sk2(self, experiment, x):
        w = self.mu_ring_separation(experiment, x)
        w[experiment.SKPhase != 2] = 1
        return w

    def diff_mu_ring_separation_sk2(self, experiment, x):
        w = self.diff_mu_ring_separation(experiment, x)
        w[experiment.SKPhase != 2] = 0
        return w

    def mu_ring_separation_sk3(self, experiment, x):
        w = self.mu_ring_separation(experiment, x)
        w[experiment.SKPhase != 3] = 1
        return w

    def diff_mu_ring_separation_sk3(self, experiment, x):
        w = self.diff_mu_ring_separation(experiment, x)
        w[experiment.SKPhase != 3] = 0
        return w

    def mu_ring_separation_sk45(self, experiment, x):
        w = self.mu_ring_separation(experiment, x)
        w[experiment.SKPhase < 4] = 1
        return w

    def diff_mu_ring_separation_sk45(self, experiment, x):
        w = self.diff_mu_ring_separation(experiment, x)
        w[experiment.SKPhase < 4] = 0
        return w

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

    def singlering_pid_sk1(self, experiment, x):
        w = self.singlering_pid(experiment, x)
        w[experiment.SKPhase != 1] = 1
        return w

    def diff_singlering_pid_sk1(self, experiment, x):
        w = self.diff_singlering_pid(experiment, x)
        w[experiment.SKPhase != 1] = 0
        return w

    def singlering_pid_sk2(self, experiment, x):
        w = self.singlering_pid(experiment, x)
        w[experiment.SKPhase != 2] = 1
        return w

    def diff_singlering_pid_sk2(self, experiment, x):
        w = self.diff_singlering_pid(experiment, x)
        w[experiment.SKPhase != 2] = 0
        return w

    def singlering_pid_sk3(self, experiment, x):
        w = self.singlering_pid(experiment, x)
        w[experiment.SKPhase != 3] = 1
        return w

    def diff_singlering_pid_sk3(self, experiment, x):
        w = self.diff_singlering_pid(experiment, x)
        w[experiment.SKPhase != 3] = 0
        return w

    def singlering_pid_sk45(self, experiment, x):
        w = self.singlering_pid(experiment, x)
        w[experiment.SKPhase < 4] = 1
        return w

    def diff_singlering_pid_sk45(self, experiment, x):
        w = self.diff_singlering_pid(experiment, x)
        w[experiment.SKPhase < 4] = 0
        return w

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

    def multiring_pid_sk1(self, experiment, x):
        w = self.multiring_pid(experiment, x)
        w[experiment.SKPhase != 1] = 1
        return w

    def diff_multiring_pid_sk1(self, experiment, x):
        w = self.diff_multiring_pid(experiment, x)
        w[experiment.SKPhase != 1] = 0
        return w

    def multiring_pid_sk2(self, experiment, x):
        w = self.multiring_pid(experiment, x)
        w[experiment.SKPhase != 2] = 1
        return w

    def diff_multiring_pid_sk2(self, experiment, x):
        w = self.diff_multiring_pid(experiment, x)
        w[experiment.SKPhase != 2] = 0
        return w

    def multiring_pid_sk3(self, experiment, x):
        w = self.multiring_pid(experiment, x)
        w[experiment.SKPhase != 3] = 1
        return w

    def diff_multiring_pid_sk3(self, experiment, x):
        w = self.diff_multiring_pid(experiment, x)
        w[experiment.SKPhase != 3] = 0
        return w

    def multiring_pid_sk45(self, experiment, x):
        w = self.multiring_pid(experiment, x)
        w[experiment.SKPhase < 4] = 1
        return w

    def diff_multiring_pid_sk45(self, experiment, x):
        w = self.diff_multiring_pid(experiment, x)
        w[experiment.SKPhase < 4] = 0
        return w

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
        # Rate-conserving n-tag migration, algebra matching the binned engine
        # (sk_binned_engine.py:1319-1332, apply2):
        # DONOR (0-neutron, {20,22,25,27}) *= x; ACCEPTOR (1-neutron,
        # {21,23,26,28}) *= 1 + r(1-x) with r = rate(donor)/rate(acceptor).
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
        # _mask_ratio(m0, m1) = sum W[m1]/sum W[m0]; donor=nn0, acceptor=nn1
        # => r = rate(nn0)/rate(nn1) = _mask_ratio(experiment, nn1, nn0).
        r = self._mask_ratio(experiment, nn1, nn0)
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
        # Matches the binned engine (see `neutron_tagging`): DONOR
        # (nn0) dw/dx = 1; ACCEPTOR (nn1) dw/dx = -r, r = rate(donor)/rate(acc).
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
        r = self._mask_ratio(experiment, nn1, nn0)
        nn[nn0] = 1
        nn[nn1] = -r
        return nn

    # --- Era-split neutron tagging (published-SK granularity) ----------------
    # The published SK analysis constrains the SK IV-V neutron-tagging
    # efficiency with two independent dofs (one sub-GeV, one multi-GeV,
    # both N(1, 0.12)), whereas the shared `neutron_tagging` above ties
    # SubGeV+MGeV (and e-like+mu-like) to one parameter. Same
    # rate-conserving migration form as `neutron_tagging`, restricted per era.
    # Sample ids: SubGeV untagged {20 (ebar 0n), 22 (numu)} <-> tagged
    # {21 (ebar 1n), 23 (numubar)}; MGeV untagged {25, 27} <-> tagged {26, 28}.

    def _neutron_tagging_era(self, experiment, x, s0, s1, diff=False):
        # s0 = DONOR (0-neutron / untagged), s1 = ACCEPTOR (1-neutron / tagged).
        # Matches the binned engine (apply2, ENG:1319-1332):
        # donor *= x (dw/dx=1), acceptor *= 1+r(1-x) (dw/dx=-r), with
        # r = rate(donor)/rate(acceptor) = _mask_ratio(experiment, nn1, nn0).
        nn0 = (experiment.Sample == s0[0]) | (experiment.Sample == s0[1])
        nn1 = (experiment.Sample == s1[0]) | (experiment.Sample == s1[1])
        r = self._mask_ratio(experiment, nn1, nn0)
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

    def decay_e_tagging_sk1(self, experiment, x):
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
        # subgev e-like
        n0 = np.sum(W[(experiment.Sample == 0) & (experiment.SKphase == 1)])
        n1 = np.sum(W[(experiment.Sample == 1) & (experiment.SKphase == 1)])
        r = n1/n0
        mue[(experiment.Sample == 0) & (experiment.SKphase == 1)] = 1 + r+(1-x)
        mue[(experiment.Sample == 1) & (experiment.SKphase == 1)] = x
        # multigev e-like
        n0 = np.sum(W[(experiment.Sample == 8) & (experiment.SKphase == 1)])
        n1 = np.sum(W[(experiment.Sample == 7) & (experiment.SKphase == 1)])
        r = n1/n0
        mue[(experiment.Sample == 8) & (experiment.SKphase == 1)] = 1 + r+(1-x)
        mue[(experiment.Sample == 7) & (experiment.SKphase == 1)] = x
        # subgev mu-like
        n0 = np.sum(W[(experiment.Sample == 3) & (experiment.SKphase == 1)])
        n1 = np.sum(W[(experiment.Sample == 4) & (experiment.SKphase == 1)])
        n2 = np.sum(W[(experiment.Sample == 5) & (experiment.SKphase == 1)])
        N = n0 + n1 + n2
        r0 = n0 / N
        r1 = n1 / N
        r2 = n2 / N
        rx1 = x * r1 + 2 * (1 - x) * r2
        rx2 = x * x * r2
        rx0 = 1 - rx1 - rx2
        mue[(experiment.Sample == 3) & (experiment.SKphase == 1)] = rx0 / r0
        mue[(experiment.Sample == 4) & (experiment.SKphase == 1)] = rx1 / r1
        mue[(experiment.Sample == 5) & (experiment.SKphase == 1)] = rx2 / r2
        return mue

    def diff_decay_e_tagging_sk1(self, experiment, x):
        r"""Method changing the efficiency of decay electron tagging.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if self._unphysical_value(x):
            return 1e-3
        mue = np.zeros(experiment.NumberOfEvents)
        W = self._rate_weight(experiment)
        # subgev e-like
        n0 = np.sum(W[(experiment.Sample == 0) & (experiment.SKphase == 1)])
        n1 = np.sum(W[(experiment.Sample == 1) & (experiment.SKphase == 1)])
        r = n1/n0
        mue[(experiment.Sample == 0) & (experiment.SKphase == 1)] = -r
        mue[(experiment.Sample == 1) & (experiment.SKphase == 1)] = 1
        # multigev e-like
        n0 = np.sum(W[(experiment.Sample == 8) & (experiment.SKphase == 1)])
        n1 = np.sum(W[(experiment.Sample == 7) & (experiment.SKphase == 1)])
        r = n1/n0
        mue[(experiment.Sample == 8) & (experiment.SKphase == 1)] = -r
        mue[(experiment.Sample == 7) & (experiment.SKphase == 1)] = 1
        # subgev mu-like
        n0 = np.sum(W[(experiment.Sample == 3) & (experiment.SKphase == 1)])
        n1 = np.sum(W[(experiment.Sample == 4) & (experiment.SKphase == 1)])
        n2 = np.sum(W[(experiment.Sample == 5) & (experiment.SKphase == 1)])
        N = n0 + n1 + n2
        r0 = n0 / N
        r1 = n1 / N
        r2 = n2 / N
        rx1 = r1 - 2 * r2
        rx2 = 2 * x * r2
        rx0 = 0 - rx1 - rx2
        mue[(experiment.Sample == 3) & (experiment.SKphase == 1)] = rx0 / r0
        mue[(experiment.Sample == 4) & (experiment.SKphase == 1)] = rx1 / r1
        mue[(experiment.Sample == 5) & (experiment.SKphase == 1)] = rx2 / r2
        return mue

    def decay_e_tagging_sk2(self, experiment, x):
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
        # subgev e-like
        n0 = np.sum(W[(experiment.Sample == 0) & (experiment.SKphase == 1)])
        n1 = np.sum(W[(experiment.Sample == 1) & (experiment.SKphase == 1)])
        r = n1/n0
        mue[(experiment.Sample == 0) & (experiment.SKphase == 2)] = 1 + r+(1-x)
        mue[(experiment.Sample == 1) & (experiment.SKphase == 2)] = x
        # multigev e-like
        n0 = np.sum(W[(experiment.Sample == 8) & (experiment.SKphase == 2)])
        n1 = np.sum(W[(experiment.Sample == 7) & (experiment.SKphase == 2)])
        r = n1/n0
        mue[(experiment.Sample == 8) & (experiment.SKphase == 2)] = 1 + r+(1-x)
        mue[(experiment.Sample == 7) & (experiment.SKphase == 2)] = x
        # subgev mu-like
        n0 = np.sum(W[(experiment.Sample == 3) & (experiment.SKphase == 2)])
        n1 = np.sum(W[(experiment.Sample == 4) & (experiment.SKphase == 2)])
        n2 = np.sum(W[(experiment.Sample == 5) & (experiment.SKphase == 2)])
        N = n0 + n1 + n2
        r0 = n0 / N
        r1 = n1 / N
        r2 = n2 / N
        rx1 = x * r1 + 2 * (1 - x) * r2
        rx2 = x * x * r2
        rx0 = 1 - rx1 - rx2
        mue[(experiment.Sample == 3) & (experiment.SKphase == 2)] = rx0 / r0
        mue[(experiment.Sample == 4) & (experiment.SKphase == 2)] = rx1 / r1
        mue[(experiment.Sample == 5) & (experiment.SKphase == 2)] = rx2 / r2
        return mue

    def diff_decay_e_tagging_sk2(self, experiment, x):
        r"""Method changing the efficiency of decay electron tagging.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if self._unphysical_value(x):
            return 1e-3
        mue = np.zeros(experiment.NumberOfEvents)
        W = self._rate_weight(experiment)
        # subgev e-like
        n0 = np.sum(W[(experiment.Sample == 0) & (experiment.SKphase == 2)])
        n1 = np.sum(W[(experiment.Sample == 1) & (experiment.SKphase == 2)])
        r = n1/n0
        mue[(experiment.Sample == 0) & (experiment.SKphase == 2)] = -r
        mue[(experiment.Sample == 1) & (experiment.SKphase == 2)] = 1
        # multigev e-like
        n0 = np.sum(W[(experiment.Sample == 8) & (experiment.SKphase == 2)])
        n1 = np.sum(W[(experiment.Sample == 7) & (experiment.SKphase == 2)])
        r = n1/n0
        mue[(experiment.Sample == 8) & (experiment.SKphase == 2)] = -r
        mue[(experiment.Sample == 7) & (experiment.SKphase == 2)] = 1
        # subgev mu-like
        n0 = np.sum(W[(experiment.Sample == 3) & (experiment.SKphase == 2)])
        n1 = np.sum(W[(experiment.Sample == 4) & (experiment.SKphase == 2)])
        n2 = np.sum(W[(experiment.Sample == 5) & (experiment.SKphase == 2)])
        N = n0 + n1 + n2
        r0 = n0 / N
        r1 = n1 / N
        r2 = n2 / N
        rx1 = r1 - 2 * r2
        rx2 = 2 * x * r2
        rx0 = 0 - rx1 - rx2
        mue[(experiment.Sample == 3) & (experiment.SKphase == 2)] = rx0 / r0
        mue[(experiment.Sample == 4) & (experiment.SKphase == 2)] = rx1 / r1
        mue[(experiment.Sample == 5) & (experiment.SKphase == 2)] = rx2 / r2
        return mue

    def decay_e_tagging_sk3(self, experiment, x):
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
        # subgev e-like
        n0 = np.sum(W[(experiment.Sample == 0) & (experiment.SKphase == 3)])
        n1 = np.sum(W[(experiment.Sample == 1) & (experiment.SKphase == 3)])
        r = n1/n0
        mue[(experiment.Sample == 0) & (experiment.SKphase == 3)] = 1 + r+(1-x)
        mue[(experiment.Sample == 1) & (experiment.SKphase == 3)] = x
        # multigev e-like
        n0 = np.sum(W[(experiment.Sample == 8) & (experiment.SKphase == 3)])
        n1 = np.sum(W[(experiment.Sample == 7) & (experiment.SKphase == 3)])
        r = n1/n0
        mue[(experiment.Sample == 8) & (experiment.SKphase == 3)] = 1 + r+(1-x)
        mue[(experiment.Sample == 7) & (experiment.SKphase == 3)] = x
        # subgev mu-like
        n0 = np.sum(W[(experiment.Sample == 3) & (experiment.SKphase == 3)])
        n1 = np.sum(W[(experiment.Sample == 4) & (experiment.SKphase == 3)])
        n2 = np.sum(W[(experiment.Sample == 5) & (experiment.SKphase == 3)])
        N = n0 + n1 + n2
        r0 = n0 / N
        r1 = n1 / N
        r2 = n2 / N
        rx1 = x * r1 + 2 * (1 - x) * r2
        rx2 = x * x * r2
        rx0 = 1 - rx1 - rx2
        mue[(experiment.Sample == 3) & (experiment.SKphase == 3)] = rx0 / r0
        mue[(experiment.Sample == 4) & (experiment.SKphase == 3)] = rx1 / r1
        mue[(experiment.Sample == 5) & (experiment.SKphase == 3)] = rx2 / r2
        return mue

    def diff_decay_e_tagging_sk3(self, experiment, x):
        r"""Method changing the efficiency of decay electron tagging.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if self._unphysical_value(x):
            return 1e-3
        mue = np.zeros(experiment.NumberOfEvents)
        W = self._rate_weight(experiment)
        # subgev e-like
        n0 = np.sum(W[(experiment.Sample == 0) & (experiment.SKphase == 3)])
        n1 = np.sum(W[(experiment.Sample == 1) & (experiment.SKphase == 3)])
        r = n1/n0
        mue[(experiment.Sample == 0) & (experiment.SKphase == 3)] = -r
        mue[(experiment.Sample == 1) & (experiment.SKphase == 3)] = 1
        # multigev e-like
        n0 = np.sum(W[(experiment.Sample == 8) & (experiment.SKphase == 3)])
        n1 = np.sum(W[(experiment.Sample == 7) & (experiment.SKphase == 3)])
        r = n1/n0
        mue[(experiment.Sample == 8) & (experiment.SKphase == 3)] = -r
        mue[(experiment.Sample == 7) & (experiment.SKphase == 3)] = 1
        # subgev mu-like
        n0 = np.sum(W[(experiment.Sample == 3) & (experiment.SKphase == 3)])
        n1 = np.sum(W[(experiment.Sample == 4) & (experiment.SKphase == 3)])
        n2 = np.sum(W[(experiment.Sample == 5) & (experiment.SKphase == 3)])
        N = n0 + n1 + n2
        r0 = n0 / N
        r1 = n1 / N
        r2 = n2 / N
        rx1 = r1 - 2 * r2
        rx2 = 2 * x * r2
        rx0 = 0 - rx1 - rx2
        mue[(experiment.Sample == 3) & (experiment.SKphase == 3)] = rx0 / r0
        mue[(experiment.Sample == 4) & (experiment.SKphase == 3)] = rx1 / r1
        mue[(experiment.Sample == 5) & (experiment.SKphase == 3)] = rx2 / r2
        return mue

    def decay_e_tagging_sk45(self, experiment, x):
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
        # subgev e-like
        n0 = np.sum(W[(experiment.Sample == 20) | (experiment.Sample == 21)])
        n1 = np.sum(W[(experiment.Sample == 19)])
        r = n1/n0
        mue[(experiment.Sample == 20) | (experiment.Sample == 21)] = 1 + r+(1-x)
        mue[(experiment.Sample == 19)] = x
        # multigev e-like
        n0 = np.sum(W[(experiment.Sample == 25) | (experiment.Sample == 26)])
        n1 = np.sum(W[(experiment.Sample == 24)])
        r = n1/n0
        mue[(experiment.Sample == 25) | (experiment.Sample == 26)] = 1 + r+(1-x)
        mue[(experiment.Sample == 24)] = x
        # subgev mu-like
        n0 = np.sum(W[(experiment.Sample == 28)])
        n1 = np.sum(W[(experiment.Sample == 27)])
        r = n1/n0
        mue[(experiment.Sample == 28)] = 1 + r+(1-x)
        mue[(experiment.Sample == 27)] = x
        return mue

    def diff_decay_e_tagging_sk45(self, experiment, x):
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
        # subgev e-like
        n0 = np.sum(W[(experiment.Sample == 20) | (experiment.Sample == 21)])
        n1 = np.sum(W[(experiment.Sample == 19)])
        r = n1/n0
        mue[(experiment.Sample == 20) | (experiment.Sample == 21)] = -r
        mue[(experiment.Sample == 19)] = 1
        # multigev e-like
        n0 = np.sum(W[(experiment.Sample == 25) | (experiment.Sample == 26)])
        n1 = np.sum(W[(experiment.Sample == 24)])
        r = n1/n0
        mue[(experiment.Sample == 25) | (experiment.Sample == 26)] = -r
        mue[(experiment.Sample == 24)] = 1
        # subgev mu-like
        n0 = np.sum(W[(experiment.Sample == 28)])
        n1 = np.sum(W[(experiment.Sample == 27)])
        r = n1/n0
        mue[(experiment.Sample == 28)] = -r
        mue[(experiment.Sample == 27)] = 1
        return mue


    # def decay_e_tagging(self, experiment, x):
    #     r"""Method changing the efficiency of decay electron tagging.

    #     Args:
    #         x (float): Value of the tuning parameter.
    #         experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

    #     Returns:
    #         Numpy.array or float with the weights from this tune.
    #     """
    #     if self._unphysical_value(x):
    #         return 1e-3
    #     mue = np.ones(experiment.NumberOfEvents)
    #     W = self._rate_weight(experiment)
    #     n0 = np.sum(W[experiment.DecayE < 1])
    #     n1 = np.sum(W[(experiment.DecayE >= 1) & (experiment.DecayE < 2)])
    #     n2 = np.sum(W[experiment.DecayE >= 2])
    #     N = n0 + n1 + n2
    #     r0 = n0 / N
    #     r1 = n1 / N
    #     r2 = n2 / N
    #     rx1 = x * r1 + 2 * (1 - x) * r2
    #     rx2 = x * x * r2 + 2 * (1 - x) * r2
    #     rx0 = 1 - rx1 - rx2
    #     mue[experiment.DecayE == 0] = rx0 / r0
    #     mue[experiment.DecayE == 1] = rx1 / r1
    #     mue[experiment.DecayE > 1] = rx2 / r2
    #     return mue

    # def diff_decay_e_tagging(self, experiment, x):
    #     r"""Method for computing the derivative of the weights w.r.t. the decay electron tagging efficiency tuning
    #     parameter.

    #     Args:
    #         x (float): Value of the tuning parameter.
    #         experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.

    #     Returns:
    #         Numpy.array or float with the derivative of the `decay_e_tagging` weights.
    #     """
    #     if self._unphysical_value(x):
    #         return 0
    #     mue = np.zeros(experiment.NumberOfEvents)
    #     W = self._rate_weight(experiment)
    #     n0 = np.sum(W[experiment.DecayE < 1])
    #     n1 = np.sum(W[(experiment.DecayE >= 1) & (experiment.DecayE < 2)])
    #     n2 = np.sum(W[experiment.DecayE >= 2])
    #     N = n0 + n1 + n2
    #     r0 = n0 / N
    #     r1 = n1 / N
    #     r2 = n2 / N
    #     rx1 = r1 - 2 * r2
    #     rx2 = 2 * x * r2 - 2 * r2
    #     rx0 = -rx1 - rx2
    #     mue[experiment.DecayE == 0] = rx0 / r0
    #     mue[experiment.DecayE == 1] = rx1 / r1
    #     mue[experiment.DecayE > 1] = rx2 / r2
    #     return mue

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

    def upmu_shower_separation_sk1(self, experiment, x):
        w = self.upmu_shower_separation(experiment, x)
        w[experiment.SKPhase != 1] = 1
        return w

    def diff_upmu_shower_separation_sk1(self, experiment, x):
        w = self.diff_upmu_shower_separation(experiment, x)
        w[experiment.SKPhase != 1] = 0
        return w

    def upmu_shower_separation_sk2(self, experiment, x):
        w = self.upmu_shower_separation(experiment, x)
        w[experiment.SKPhase != 2] = 1
        return w

    def diff_upmu_shower_separation_sk2(self, experiment, x):
        w = self.diff_upmu_shower_separation(experiment, x)
        w[experiment.SKPhase != 2] = 0
        return w

    def upmu_shower_separation_sk3(self, experiment, x):
        w = self.upmu_shower_separation(experiment, x)
        w[experiment.SKPhase != 3] = 1
        return w

    def diff_upmu_shower_separation_sk3(self, experiment, x):
        w = self.diff_upmu_shower_separation(experiment, x)
        w[experiment.SKPhase != 3] = 0
        return w

    def upmu_shower_separation_sk45(self, experiment, x):
        w = self.upmu_shower_separation(experiment, x)
        w[experiment.SKPhase < 4] = 1
        return w

    def diff_upmu_shower_separation_sk45(self, experiment, x):
        w = self.diff_upmu_shower_separation(experiment, x)
        w[experiment.SKPhase < 4] = 0
        return w

    def upmu_stop_bkg(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        um = np.ones(experiment.NumberOfEvents)
        um[(experiment.Sample == 16) * (experiment.CosZReco>-0.1)] = x
        return um

    def diff_upmu_stop_bkg(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        um = np.zeros(experiment.NumberOfEvents)
        um[experiment.Sample == 16] = 1
        return um

    def upmu_stop_bkg_sk1(self, experiment, x):
        w = self.upmu_stop_bkg(experiment, x)
        w[experiment.SKPhase != 1] = 1
        return w

    def diff_upmu_stop_bkg_sk1(self, experiment, x):
        w = self.diff_upmu_stop_bkg(experiment, x)
        w[experiment.SKPhase != 1] = 0
        return w

    def upmu_stop_bkg_sk2(self, experiment, x):
        w = self.upmu_stop_bkg(experiment, x)
        w[experiment.SKPhase != 2] = 1
        return w

    def diff_upmu_stop_bkg_sk2(self, experiment, x):
        w = self.diff_upmu_stop_bkg(experiment, x)
        w[experiment.SKPhase != 2] = 0
        return w

    def upmu_stop_bkg_sk3(self, experiment, x):
        w = self.upmu_stop_bkg(experiment, x)
        w[experiment.SKPhase != 3] = 1
        return w

    def diff_upmu_stop_bkg_sk3(self, experiment, x):
        w = self.diff_upmu_stop_bkg(experiment, x)
        w[experiment.SKPhase != 3] = 0
        return w

    def upmu_stop_bkg_sk45(self, experiment, x):
        w = self.upmu_stop_bkg(experiment, x)
        w[experiment.SKPhase < 4] = 1
        return w

    def diff_upmu_stop_bkg_sk45(self, experiment, x):
        w = self.diff_upmu_stop_bkg(experiment, x)
        w[experiment.SKPhase < 4] = 0
        return w

    def upmu_showering_bkg(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        um = np.ones(experiment.NumberOfEvents)
        um[(experiment.Sample == 18) * (experiment.CosZReco>-0.1)] = x
        return um

    def diff_upmu_showering_bkg(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        um = np.zeros(experiment.NumberOfEvents)
        um[(experiment.Sample == 18) * (experiment.CosZReco>-0.1)] = 1
        return um

    def upmu_showering_bkg_sk1(self, experiment, x):
        w = self.upmu_showering_bkg(experiment, x)
        w[experiment.SKPhase != 1] = 1
        return w

    def diff_upmu_showering_bkg_sk1(self, experiment, x):
        w = self.diff_upmu_showering_bkg(experiment, x)
        w[experiment.SKPhase != 1] = 0
        return w

    def upmu_showering_bkg_sk2(self, experiment, x):
        w = self.upmu_showering_bkg(experiment, x)
        w[experiment.SKPhase != 2] = 1
        return w

    def diff_upmu_showering_bkg_sk2(self, experiment, x):
        w = self.diff_upmu_showering_bkg(experiment, x)
        w[experiment.SKPhase != 2] = 0
        return w

    def upmu_showering_bkg_sk3(self, experiment, x):
        w = self.upmu_showering_bkg(experiment, x)
        w[experiment.SKPhase != 3] = 1
        return w

    def diff_upmu_showering_bkg_sk3(self, experiment, x):
        w = self.diff_upmu_showering_bkg(experiment, x)
        w[experiment.SKPhase != 3] = 0
        return w

    def upmu_showering_bkg_sk45(self, experiment, x):
        w = self.upmu_showering_bkg(experiment, x)
        w[experiment.SKPhase < 4] = 1
        return w

    def diff_upmu_showering_bkg_sk45(self, experiment, x):
        w = self.diff_upmu_showering_bkg(experiment, x)
        w[experiment.SKPhase < 4] = 0
        return w

    def upmu_nonshowering_bkg(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        um = np.ones(experiment.NumberOfEvents)
        um[(experiment.Sample == 17) * (experiment.CosZReco>-0.1)] = x
        return um

    def diff_upmu_nonshowering_bkg(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        um = np.zeros(experiment.NumberOfEvents)
        um[(experiment.Sample == 17) * (experiment.CosZReco>-0.1)] = 1
        return um

    def upmu_nonshowering_bkg_sk1(self, experiment, x):
        w = self.upmu_nonshowering_bkg(experiment, x)
        w[experiment.SKPhase != 1] = 1
        return w

    def diff_upmu_nonshowering_bkg_sk1(self, experiment, x):
        w = self.diff_upmu_nonshowering_bkg(experiment, x)
        w[experiment.SKPhase != 1] = 0
        return w

    def upmu_nonshowering_bkg_sk2(self, experiment, x):
        w = self.upmu_nonshowering_bkg(experiment, x)
        w[experiment.SKPhase != 2] = 1
        return w

    def diff_upmu_nonshowering_bkg_sk2(self, experiment, x):
        w = self.diff_upmu_nonshowering_bkg(experiment, x)
        w[experiment.SKPhase != 2] = 0
        return w

    def upmu_nonshowering_bkg_sk3(self, experiment, x):
        w = self.upmu_nonshowering_bkg(experiment, x)
        w[experiment.SKPhase != 3] = 1
        return w

    def diff_upmu_nonshowering_bkg_sk3(self, experiment, x):
        w = self.diff_upmu_nonshowering_bkg(experiment, x)
        w[experiment.SKPhase != 3] = 0
        return w

    def upmu_nonshowering_bkg_sk45(self, experiment, x):
        w = self.upmu_nonshowering_bkg(experiment, x)
        w[experiment.SKPhase < 4] = 1
        return w

    def diff_upmu_nonshowering_bkg_sk45(self, experiment, x):
        w = self.diff_upmu_nonshowering_bkg(experiment, x)
        w[experiment.SKPhase < 4] = 0
        return w

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

    # =====================================================================
    #  r2_fude_ccqe event-engine dials
    #  Sample/bin-keyed detector dials transcribed 1:1 from the binned
    #  engine (ENG = pynu/binned/sk_binned_engine.py). All W-type;
    #  guard -> 1e-3.
    # =====================================================================

    #: FC multi-GeV sample group for rel_norm_fcmg (ENG:318).
    _REL_NORM_FCMG_SAMPLES = frozenset({7, 8, 9, 10, 11, 12, 13, 24, 25, 26, 27, 28})

    def rel_norm_fcmg(self, experiment, x):
        r"""Relative normalization of the FC multi-GeV sample group.

        Flat multiplicative norm on samples {7,8,9,10,11,12,13,24,25,26,27,28}
        (SK Rel.Norm FC-MultiGeV). $w = x$ on those samples, 1 elsewhere.
        nominal x=1 (exact no-op). Mirrors sk_binned_engine.py:1480-1484 (D fold)
        with d ln D/dx = 1/x; REL_NORM_FCMG_SAMPLES at ENG:318.

        Args:
            x (float): Value of the tuning parameter (nominal 1).
            experiment: Experiment class with per-event `Sample`.

        Returns:
            Numpy.array with the per-event weights from this tune.
        """
        if self._unphysical_value(x):
            return 1e-3
        w = np.ones(experiment.NumberOfEvents)
        w[np.isin(experiment.Sample, list(self._REL_NORM_FCMG_SAMPLES))] = x
        return w

    def diff_rel_norm_fcmg(self, experiment, x):
        r"""Derivative of `rel_norm_fcmg` w.r.t. x: 1 on the FC multi-GeV group, 0 else."""
        if self._unphysical_value(x):
            return 0
        w = np.zeros(experiment.NumberOfEvents)
        w[np.isin(experiment.Sample, list(self._REL_NORM_FCMG_SAMPLES))] = 1.0
        return w

    # ---- up-mu background zenith x momentum SHAPE (UBS, ENG:407-420) ----------
    # Per-bin norms on the near-horizon reco-cosZ x reco-momentum cells of the
    # up-mu background samples. The event-side selection reproduces the binned
    # (sample, iz, ie) cell sets by reco-variable membership. Binned reference:
    # UPMU_BKG_SHAPE_SPEC ENG:411-415 (name -> (sample, iz set, ie set)), per-bin
    # mask build ENG:962-975, D fold + d ln D/dx=1/x ENG:1533-1540.
    #
    # Reco-cosZ bins use CTBins[sample] (up-mu samples 16/17/18 use z10bins_up,
    # edges [-1..0] in 10 bins) so iz = digitize(CosZReco, edges)-1. Reco-momentum
    # bins use EnergyBins[sample] so ie = digitize(EReco, edges)-1.

    @staticmethod
    def _reco_bin_index(values, edges):
        r"""Bin index of `values` in ascending `edges` (0..len(edges)-2), matching
        np.histogram's convention (right-open bins, last bin closed)."""
        idx = np.digitize(values, edges) - 1
        idx = np.clip(idx, 0, len(edges) - 2)
        return idx

    #: name -> (sample id, frozenset of reco-cosZ bin indices iz,
    #:          frozenset of reco-momentum bin indices ie or None=all). ENG:411-415
    _UPMU_BKG_SHAPE_SPEC = {
        "upmu_stop_bkg_horiz_lowp":  (16, frozenset({8, 9}), frozenset({0})),
        "upmu_stop_bkg_horiz_highp": (16, frozenset({8, 9}), frozenset({1, 2})),
        "upmu_nonshow_bkg_horiz":    (17, frozenset({9}), None),
    }

    def _upmu_bkg_shape_mask(self, experiment, name):
        r"""Per-event boolean mask for a UBS dial's (sample, iz, ie) cell set."""
        sid, iz_set, ie_set = self._UPMU_BKG_SHAPE_SPEC[name]
        in_sample = experiment.Sample == sid
        mask = in_sample.copy()
        if not np.any(in_sample):
            return mask
        cz_edges = experiment.CTBins[sid]
        e_edges = experiment.EnergyBins[sid]
        iz = self._reco_bin_index(experiment.CosZReco, cz_edges)
        mask &= np.isin(iz, list(iz_set))
        if ie_set is not None:
            ie = self._reco_bin_index(experiment.EReco, e_edges)
            mask &= np.isin(ie, list(ie_set))
        return mask

    def _upmu_bkg_shape_weight(self, experiment, name, x, diff=False):
        if self._unphysical_value(x):
            return 0 if diff else 1e-3
        mask = self._upmu_bkg_shape_mask(experiment, name)
        w = np.zeros(experiment.NumberOfEvents) if diff \
            else np.ones(experiment.NumberOfEvents)
        w[mask] = 1.0 if diff else x
        return w

    def upmu_stop_bkg_horiz_lowp(self, experiment, x):
        r"""Up-mu stopping bkg, horizon-localized, low momentum (sample 16, iz{8,9}, ie{0})."""
        return self._upmu_bkg_shape_weight(experiment, "upmu_stop_bkg_horiz_lowp", x)

    def diff_upmu_stop_bkg_horiz_lowp(self, experiment, x):
        return self._upmu_bkg_shape_weight(experiment, "upmu_stop_bkg_horiz_lowp", x, diff=True)

    def upmu_stop_bkg_horiz_highp(self, experiment, x):
        r"""Up-mu stopping bkg, horizon-localized, high momentum (sample 16, iz{8,9}, ie{1,2})."""
        return self._upmu_bkg_shape_weight(experiment, "upmu_stop_bkg_horiz_highp", x)

    def diff_upmu_stop_bkg_horiz_highp(self, experiment, x):
        return self._upmu_bkg_shape_weight(experiment, "upmu_stop_bkg_horiz_highp", x, diff=True)

    def upmu_nonshow_bkg_horiz(self, experiment, x):
        r"""Up-mu non-showering bkg, horizon-localized (sample 17, iz{9}, all momentum)."""
        return self._upmu_bkg_shape_weight(experiment, "upmu_nonshow_bkg_horiz", x)

    def diff_upmu_nonshow_bkg_horiz(self, experiment, x):
        return self._upmu_bkg_shape_weight(experiment, "upmu_nonshow_bkg_horiz", x, diff=True)

    # ---- up/down energy-scale (UDE, ENG:440-445, 977-1000, 1542-1555) --------
    # Anti-symmetric per-era normalization of up-going (reco-cosZ<0) vs
    # down-going (reco-cosZ>=0) FC+PC events: up *= (1+d), down *= (1-d).
    # W-type signed NORM (NOT a migration; ENG:360-364). nominal d=0 (exact
    # no-op). Excludes up-mu samples {16,17,18} (all up-going) and the single-
    # reco-zenith FC samples whose CTBins is z1bins (nz=1, straddle cz=0). The
    # binned convention is iz<nz//2 up / iz>=nz//2 down on the z10bins grids,
    # whose central edge is cz=0 -> equivalently CosZReco<0 up, >=0 down.
    _UDE_EXCLUDE_SAMPLES = frozenset({16, 17, 18})

    def _ude_sign(self, experiment):
        r"""Signed up/down mask: +1 up-going (cz<0), -1 down-going (cz>=0), 0 on
        excluded samples (up-mu + single-reco-zenith z1bins FC samples)."""
        sign = np.zeros(experiment.NumberOfEvents)
        for sid in experiment.Samples:
            if int(sid) in self._UDE_EXCLUDE_SAMPLES:
                continue
            # z1bins (single reco-cosZ bin) samples straddle cz=0 -> excluded.
            if experiment.CTBins[sid].size - 1 != 10:
                continue
            in_s = experiment.Sample == sid
            sign[in_s & (experiment.CosZReco < 0.0)] = 1.0
            sign[in_s & (experiment.CosZReco >= 0.0)] = -1.0
        return sign

    def _updown_escale_era(self, experiment, d, era, diff=False):
        r"""Per-era up/down energy-scale. w = 1 + d*sign on the era's FC+PC bins,
        1 elsewhere; diff twin returns dw/dd = sign on the era, 0 elsewhere.
        `era` is the SKPhase int (45 selects SKPhase>=4)."""
        if self._unphysical_value(d, unphys_low=-9999999):
            return 0 if diff else 1e-3
        sign = self._ude_sign(experiment)
        if era == 45:
            era_mask = experiment.SKPhase >= 4
        else:
            era_mask = experiment.SKPhase == era
        if diff:
            w = np.zeros(experiment.NumberOfEvents)
            w[era_mask] = sign[era_mask]
            return w
        w = np.ones(experiment.NumberOfEvents)
        w[era_mask] = 1.0 + d * sign[era_mask]
        return w

    def updown_escale_sk1(self, experiment, x):
        r"""Up/down energy-scale asymmetry, SK-I. up*=(1+x), down*=(1-x); nominal 0."""
        return self._updown_escale_era(experiment, x, 1)

    def diff_updown_escale_sk1(self, experiment, x):
        return self._updown_escale_era(experiment, x, 1, diff=True)

    def updown_escale_sk2(self, experiment, x):
        r"""Up/down energy-scale asymmetry, SK-II. up*=(1+x), down*=(1-x); nominal 0."""
        return self._updown_escale_era(experiment, x, 2)

    def diff_updown_escale_sk2(self, experiment, x):
        return self._updown_escale_era(experiment, x, 2, diff=True)

    def updown_escale_sk3(self, experiment, x):
        r"""Up/down energy-scale asymmetry, SK-III. up*=(1+x), down*=(1-x); nominal 0."""
        return self._updown_escale_era(experiment, x, 3)

    def diff_updown_escale_sk3(self, experiment, x):
        return self._updown_escale_era(experiment, x, 3, diff=True)

    def updown_escale_sk45(self, experiment, x):
        r"""Up/down energy-scale asymmetry, SK-IV+V. up*=(1+x), down*=(1-x); nominal 0."""
        return self._updown_escale_era(experiment, x, 45)

    def diff_updown_escale_sk45(self, experiment, x):
        return self._updown_escale_era(experiment, x, 45, diff=True)