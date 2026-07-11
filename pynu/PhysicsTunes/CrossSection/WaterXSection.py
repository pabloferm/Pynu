from ..PhysicsTunes import Tune  # , _unphysical
import numpy as np

import sys

sys.path.append("../")

##########################
#  Water Cross-section   #
##########################


class WaterXSection(Tune):
    r"""Class containing the tunes for the neutrino-water cross section. Note that there are some dependencies on the NEUT interaction mode definition."""

    # @_unphysical(lambda x: x<0)
    def XSecNuTau(self, experiment, x):
        r"""Method for modifying the $\nu_\tau$ cross-section normalization.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if self._unphysical_value(x):
            return 1e-3
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
        if self._unphysical_value(x):
            return 0
        tau = np.zeros(experiment.NumberOfEvents)
        tau[np.abs(experiment.nuPDG) == 16] = 1
        return tau

    def NCoverCC(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        nc = np.ones(experiment.NumberOfEvents)
        nc[experiment.CC == 0] = x
        return nc

    def diff_NCoverCC(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        nc = np.zeros(experiment.NumberOfEvents)
        nc[experiment.CC == 0] = 1
        return nc

    def AxialMass(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        cc = np.ones(experiment.NumberOfEvents)
        cc[experiment.CC == 1] = 1 + 0.042 * (x - 1) * 1.05 * np.log10(
            experiment.ETrue[experiment.CC == 1]
        )
        return cc

    def diff_AxialMass(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        cc = np.zeros(experiment.NumberOfEvents)
        cc[experiment.CC == 1] = (
            0.042 * 1.05 * np.log10(experiment.ETrue[experiment.CC == 1])
        )
        return cc

    def NCHad(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        nc = np.ones(experiment.NumberOfEvents)
        nc[experiment.CC == 0] = x
        return nc

    def diff_NCHad(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        nc = np.zeros(experiment.NumberOfEvents)
        nc[experiment.CC == 0] = 1
        return nc

    def DIS(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        w = np.ones(experiment.NumberOfEvents)
        cond = np.abs(experiment.Mode) > 25 * experiment.CC
        w[cond] = x
        return w

    def diff_DIS(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        w = np.zeros(experiment.NumberOfEvents)
        cond = np.abs(experiment.Mode) > 25 * experiment.CC
        w[cond] = 1
        return w

    def CC_2p2h(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        cc_2p2h = np.ones(experiment.NumberOfEvents)
        cc_2p2h[np.abs(experiment.Mode) == 2] = x
        return cc_2p2h

    def diff_CC_2p2h(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        cc_2p2h = np.zeros(experiment.NumberOfEvents)
        cc_2p2h[np.abs(experiment.Mode) == 2] = 1
        return cc_2p2h

    def CC_2p2hNuBarNu(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        cc_2p2h = np.ones(experiment.NumberOfEvents)
        cc_2p2h[experiment.Mode == -2] = x
        return cc_2p2h

    def diff_CC_2p2hNuBarNu(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        cc_2p2h = np.zeros(experiment.NumberOfEvents)
        cc_2p2h[experiment.Mode == -2] = 1
        return cc_2p2h

    def CC_2p2hMuE(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        cc_2p2h = np.ones(experiment.NumberOfEvents)
        cond = (np.abs(experiment.Mode) == 2) * (np.abs(experiment.nuPDG) == 14)
        cc_2p2h[cond] = x
        return cc_2p2h

    def diff_CC_2p2hMuE(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        cc_2p2h = np.zeros(experiment.NumberOfEvents)
        cond = (np.abs(experiment.Mode) == 2) * (np.abs(experiment.nuPDG) == 14)
        cc_2p2h[cond] = 1
        return cc_2p2h

    def CCQE(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        ccqe = np.ones(experiment.NumberOfEvents)
        ccqe[np.abs(experiment.Mode) == 1] = x
        return ccqe

    def diff_CCQE(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        ccqe = np.zeros(experiment.NumberOfEvents)
        ccqe[np.abs(experiment.Mode) == 1] = 1
        return ccqe

    def CCQENuBarNu(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        ccqe = np.ones(experiment.NumberOfEvents)
        ccqe[experiment.Mode == -1] = x
        return ccqe

    def diff_CCQENuBarNu(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        ccqe = np.zeros(experiment.NumberOfEvents)
        ccqe[experiment.Mode == -1] = 1
        return ccqe

    def CCQEMuE(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        ccqe = np.ones(experiment.NumberOfEvents)
        cond = (np.abs(experiment.Mode) == 1) * (np.abs(experiment.nuPDG) == 14)
        ccqe[cond] = x
        return ccqe

    def diff_CCQEMuE(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        ccqe = np.zeros(experiment.NumberOfEvents)
        cond = (np.abs(experiment.Mode) == 1) * (np.abs(experiment.nuPDG) == 14)
        ccqe[cond] = 1
        return ccqe

    def CC1Pi_Pi0Pi(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        ccpi = np.ones(experiment.NumberOfEvents)
        ccpi[np.abs(experiment.Mode) == 12] = x
        return ccpi

    def diff_CC1Pi_Pi0Pi(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        ccpi = np.zeros(experiment.NumberOfEvents)
        ccpi[np.abs(experiment.Mode) == 12] = 1
        return ccpi

    def CC1Pi_NuBarNuE(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        ccpi = np.ones(experiment.NumberOfEvents)
        cond = (
            (np.abs(experiment.Mode) > 10)
            * (np.abs(experiment.Mode) < 17)
            * (experiment.nuPDG == -12)
        )
        ccpi[cond] = x
        return ccpi

    def diff_CC1Pi_NuBarNuE(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        ccpi = np.zeros(experiment.NumberOfEvents)
        cond = (
            (np.abs(experiment.Mode) > 10)
            * (np.abs(experiment.Mode) < 17)
            * (experiment.nuPDG == -12)
        )
        ccpi[cond] = 1
        return ccpi

    def CC1Pi_NuBarNuMu(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        ccpi = np.ones(experiment.NumberOfEvents)
        cond = (
            (np.abs(experiment.Mode) > 10)
            * (np.abs(experiment.Mode) < 17)
            * (experiment.nuPDG == -14)
        )
        ccpi[cond] = x
        return ccpi

    def diff_CC1Pi_NuBarNuMu(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        ccpi = np.zeros(experiment.NumberOfEvents)
        cond = (
            (np.abs(experiment.Mode) > 10)
            * (np.abs(experiment.Mode) < 17)
            * (experiment.nuPDG == -14)
        )
        ccpi[cond] = 1
        return ccpi

    def CC1PiProduction(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        ccpi = np.ones(experiment.NumberOfEvents)
        cond = (np.abs(experiment.Mode) > 10) * (np.abs(experiment.Mode) < 17)
        ccpi[cond] = x
        return ccpi

    def diff_CC1PiProduction(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        ccpi = np.zeros(experiment.NumberOfEvents)
        cond = (np.abs(experiment.Mode) > 10) * (np.abs(experiment.Mode) < 17)
        ccpi[cond] = 1
        return ccpi

    def CohPiProduction(self, experiment, x):
        if self._unphysical_value(x):
            return 1e-3
        ccpi = np.ones(experiment.NumberOfEvents)
        ccpi[np.abs(experiment.Mode) == 16] = x
        return ccpi

    def diff_CohPiProduction(self, experiment, x):
        if self._unphysical_value(x):
            return 0
        ccpi = np.zeros(experiment.NumberOfEvents)
        ccpi[np.abs(experiment.Mode) == 16] = 1
        return ccpi

    # =====================================================================
    #  r2_fude_ccqe event-engine dials
    #  Transcribed 1:1 from the binned engine (ENG = pynu/binned/
    #  sk_binned_engine.py); each method is line-cited to the ENG block it
    #  mirrors. All W-type. Guard convention: return 1e-3 (the binned
    #  engine's punishment convention).
    # =====================================================================

    #: SK sub-GeV/multi-GeV boundary on true energy (GeV), ENG:271
    #: (CCQE_SHAPE_SUBGEV_E). Sub-GeV CCQE shape confined below; multi-GeV
    #: CCQE flavor norms act above.
    _CCQE_SHAPE_SUBGEV_E = 1.33

    # -- xsec_ccqe_shape_subgev centering constant mu ----------------------------
    # The sub-GeV log-E tilt sh(E) = ln E - mu is mean-zero over the sub-GeV
    # region. mu is part of the DIAL'S DEFINITION and must equal the binned
    # engine's centering EXACTLY: the binned engine subtracts the unweighted mean
    # of ln(e_c) over the true-E CELL centres with e_c < 1.33 GeV, where
    # e_c = sqrt(e_edges[:-1]*e_edges[1:]) are the geometric cell centres of the
    # response's true-E grid (sk_binned_engine.py:848,1067-1070;
    # build_sk_response.make_true_grid). It is a build-time GEOMETRY constant tied
    # to the response's e_edges, NOT a property of the events. The per-event
    # evaluation of ln(E) stays per-event; only the constant mu changes.
    #
    # e_edges are supplied by an overridable hook so callers can inject the
    # response's ACTUAL edges (mu then matches the binned engine bit-for-bit).
    # The default replicates make_true_grid over the production grid so the
    # method is standalone-correct against the response in use. The production
    # r2_fude_ccqe response is n_e=400 over the MC true-E range
    # [0.1000172462187625, 466048.7104907757]; on that grid
    # mu = -1.0164880658631577 (e^mu = 0.36186 GeV, sub-GeV as required).
    _CCQE_SHAPE_SUBGEV_GRID_EMIN = 0.1000172462187625   # MC ETrue min (skb_phased)
    _CCQE_SHAPE_SUBGEV_GRID_EMAX = 466048.7104907757     # MC ETrue max (skb_phased)
    _CCQE_SHAPE_SUBGEV_GRID_NE = 400                      # production n_etrue
    #: overridable e_edges for the sub-GeV centering. None -> the computed default
    #: (make_true_grid over the production grid). Set this (per instance or per
    #: class) to the response's actual e_edges to pin mu to that response.
    ccqe_shape_subgev_e_edges = None

    @classmethod
    def _make_true_grid_e_edges(cls, emin, emax, n_e):
        r"""Replicate build_sk_response.make_true_grid's true-E edges EXACTLY:
        geomspace(emin, emax, n_e+1) with the edge nearest ln(1.0) snapped to
        exactly 1.0 GeV (sk_binned_engine tunes norm_below/above1GeV on it)."""
        e_edges = np.geomspace(emin, emax, n_e + 1)
        i = np.argmin(np.abs(np.log(e_edges) - np.log(1.0)))
        e_edges[i] = 1.0
        return e_edges

    def _ccqe_shape_subgev_mu(self):
        r"""Binned-engine centering constant mu for xsec_ccqe_shape_subgev.

        mu = unweighted mean of ln(e_c) over the true-E CELL centres e_c < 1.33
        GeV, e_c = sqrt(e_edges[:-1]*e_edges[1:]) (sk_binned_engine.py:848,
        1067-1070). Uses the injected `ccqe_shape_subgev_e_edges` if set, else
        the production grid (make_true_grid over the class GRID constants).
        This is the IDENTICAL computation the binned engine performs on the same
        e_edges, so mu matches the binned engine bit-for-bit when both derive
        from the same edges."""
        e_edges = self.ccqe_shape_subgev_e_edges
        if e_edges is None:
            e_edges = self._make_true_grid_e_edges(
                self._CCQE_SHAPE_SUBGEV_GRID_EMIN,
                self._CCQE_SHAPE_SUBGEV_GRID_EMAX,
                self._CCQE_SHAPE_SUBGEV_GRID_NE)
        e_edges = np.asarray(e_edges, float)
        e_c = np.sqrt(e_edges[:-1] * e_edges[1:])            # geometric cell centres
        sub = e_c < self._CCQE_SHAPE_SUBGEV_E
        return float(np.log(e_c[sub]).mean())

    @staticmethod
    def _ccqe_mask(experiment):
        r"""CCQE class == 1p1h on the 2p2h MC: |Mode|==1 (WX.CCQE convention)."""
        return np.abs(experiment.Mode) == 1

    def xsec_ccqe_shape(self, experiment, x):
        r"""GLOBAL CCQE energy-shape freedom (all energies).

        Power-law E-tilt on the CCQE class: $w = E_\nu^{\,x-1}$ on CCQE events,
        1 elsewhere. nominal x=1 (exact no-op: $E^0 = 1$).
        Mirrors sk_binned_engine.py:1145-1148 (apply) / :1807-1812 (grad),
        prior (1.0, 0.20) ENG:275, box (0.3,1.7).

        Args:
            x (float): Value of the tuning parameter (nominal 1).
            experiment: Experiment class with per-event `ETrue`, `Mode`.

        Returns:
            Numpy.array with the per-event weights from this tune.
        """
        if self._unphysical_value(x):
            return 1e-3
        ccqe = self._ccqe_mask(experiment)
        w = np.ones(experiment.NumberOfEvents)
        w[ccqe] = experiment.ETrue[ccqe] ** (x - 1.0)
        return w

    def diff_xsec_ccqe_shape(self, experiment, x):
        r"""Derivative of `xsec_ccqe_shape`: $dw/dx = E^{x-1}\ln E$ on CCQE.

        Matches ENG:1810 d ln W/dr = ln(E); here in absolute (non-log) form
        the full derivative is $E^{x-1}\ln E$ so that diff/w == ln(E)."""
        if self._unphysical_value(x):
            return 0
        ccqe = self._ccqe_mask(experiment)
        w = np.zeros(experiment.NumberOfEvents)
        w[ccqe] = experiment.ETrue[ccqe] ** (x - 1.0) * np.log(experiment.ETrue[ccqe])
        return w

    def _ccqe_shape_subgev_tilt(self, experiment):
        r"""Mean-zero sub-GeV log-E tilt sh(E) for `xsec_ccqe_shape_subgev`.

        ENG:1067-1070: sh = ln(E) - mu for E<1.33 (0 above), a rate-neutral
        (mean-zero) log-E tilt. Per-event ln(E) is evaluated on the event's own
        ETrue; mu is the binned engine's CENTERING CONSTANT — the unweighted
        mean of ln(e_c) over the true-E CELL centres below 1.33 GeV.

        mu is part of the dial's DEFINITION, so the event engine adopts the
        IDENTICAL binned cell-grid mu (`_ccqe_shape_subgev_mu`) rather than an
        event-sampled sub-GeV mean (which would differ by the event-sampled vs
        cell-weighted <ln E> offset). The per-event evaluation of ln(E) stays
        per-event; only the constant is shared. Both engines key on the SAME
        e_edges grid, so on the production response the tilt is bit-for-bit
        identical up to the one controlled binned cell-centering approximation
        (cell-smooth Jensen gap).
        """
        E = experiment.ETrue
        sub = E < self._CCQE_SHAPE_SUBGEV_E
        mu = self._ccqe_shape_subgev_mu()                # binned cell-grid centering
        sh = np.where(sub, np.log(E) - mu, 0.0)
        return sh, sub

    def xsec_ccqe_shape_subgev(self, experiment, x):
        r"""sub-GeV-LOCALIZED CCQE shape freedom (distinct from the global one).

        Mean-zero log-E tilt confined to E_true<1.33 GeV, applied to CCQE:
        $w = 1 + x\,\mathrm{sh}(E)$, sh = ln(E) - <ln E>_subgev below 1.33,
        0 above. nominal x=0 (exact no-op). Mirrors sk_binned_engine.py:1149-1151
        (apply) / :1813-1821 (grad), prior (0.0, 0.40) ENG:277, box (-2,2).

        Args:
            x (float): Value of the tuning parameter (nominal 0).
            experiment: Experiment class with per-event `ETrue`, `Mode`.

        Returns:
            Numpy.array with the per-event weights from this tune.
        """
        if self._unphysical_value(x, unphys_low=-9999999):
            return 1e-3
        ccqe = self._ccqe_mask(experiment)
        sh, _ = self._ccqe_shape_subgev_tilt(experiment)
        w = np.ones(experiment.NumberOfEvents)
        w[ccqe] = 1.0 + x * sh[ccqe]
        return w

    def diff_xsec_ccqe_shape_subgev(self, experiment, x):
        r"""Derivative of `xsec_ccqe_shape_subgev` w.r.t. x: $dw/dx = \mathrm{sh}(E)$
        on CCQE events (ENG:1816-1818 numerator sh)."""
        if self._unphysical_value(x, unphys_low=-9999999):
            return 0
        ccqe = self._ccqe_mask(experiment)
        sh, _ = self._ccqe_shape_subgev_tilt(experiment)
        w = np.zeros(experiment.NumberOfEvents)
        w[ccqe] = sh[ccqe]
        return w

    def xsec_1p1h_subgev_nue(self, experiment, x):
        r"""sub-GeV nu_e 1p1h (=CCQE) cross-section norm.

        Multiplicative norm on sub-GeV (E<1 GeV) CCQE nu_e+nu-bar_e events:
        $w = 1 + \mathbb{1}[E<1]\,(x-1)$ on the CCQE nu_e class, 1 elsewhere.
        nominal x=1 (exact no-op). Mirrors the SUBGEV_NUE_NORM path
        sk_binned_engine.py:1154-1158 (apply) / :1824-1836 (grad), mask
        ccqe_nue_cls = CCQE & flavor==0 (ENG:1054), band e_below1 = E<1
        (ENG:1029), prior (1.0, 0.05), box (0.5,1.5).

        Args:
            x (float): Value of the tuning parameter (nominal 1).
            experiment: Experiment class with per-event `ETrue`, `Mode`, `nuPDG`.

        Returns:
            Numpy.array with the per-event weights from this tune.
        """
        if self._unphysical_value(x):
            return 1e-3
        cls = self._ccqe_mask(experiment) & (np.abs(experiment.nuPDG) == 12)
        w = np.ones(experiment.NumberOfEvents)
        sel = cls & (experiment.ETrue < 1.0)
        w[sel] = x
        return w

    def diff_xsec_1p1h_subgev_nue(self, experiment, x):
        r"""Derivative of `xsec_1p1h_subgev_nue` w.r.t. x: 1 on the masked band, 0 else."""
        if self._unphysical_value(x):
            return 0
        cls = self._ccqe_mask(experiment) & (np.abs(experiment.nuPDG) == 12)
        w = np.zeros(experiment.NumberOfEvents)
        w[cls & (experiment.ETrue < 1.0)] = 1.0
        return w

    def _multigev_ccqe_norm(self, experiment, x, flavor_pdg):
        r"""Multi-GeV CCQE flavor norm on E_true>=1.33 GeV (ENG:1160-1168)."""
        if self._unphysical_value(x):
            return 1e-3
        cls = self._ccqe_mask(experiment) & (np.abs(experiment.nuPDG) == flavor_pdg)
        w = np.ones(experiment.NumberOfEvents)
        sel = cls & (experiment.ETrue >= self._CCQE_SHAPE_SUBGEV_E)
        w[sel] = x
        return w

    def _diff_multigev_ccqe_norm(self, experiment, x, flavor_pdg):
        if self._unphysical_value(x):
            return 0
        cls = self._ccqe_mask(experiment) & (np.abs(experiment.nuPDG) == flavor_pdg)
        w = np.zeros(experiment.NumberOfEvents)
        w[cls & (experiment.ETrue >= self._CCQE_SHAPE_SUBGEV_E)] = 1.0
        return w

    def xsec_ccqe_multigev_nue(self, experiment, x):
        r"""multi-GeV nu_e CCQE flavor norm (E_true>=1.33 GeV, nu_e+nu-bar_e CCQE).

        $w = 1 + \mathbb{1}[E\geq1.33]\,(x-1)$ on the CCQE nu_e class.
        nominal x=1 (exact no-op). Mirrors MULTIGEV_CCQE_NORM
        sk_binned_engine.py:1164-1168 / :1839-1848, prior (1.0, 0.25), box (0,3).
        """
        return self._multigev_ccqe_norm(experiment, x, 12)

    def diff_xsec_ccqe_multigev_nue(self, experiment, x):
        return self._diff_multigev_ccqe_norm(experiment, x, 12)

    def xsec_ccqe_multigev_numu(self, experiment, x):
        r"""multi-GeV nu_mu CCQE flavor norm (E_true>=1.33 GeV, nu_mu+nu-bar_mu CCQE).

        $w = 1 + \mathbb{1}[E\geq1.33]\,(x-1)$ on the CCQE nu_mu class.
        nominal x=1 (exact no-op). Mirrors MULTIGEV_CCQE_NORM
        sk_binned_engine.py:1164-1168 / :1839-1848, prior (1.0, 0.25), box (0,3).
        """
        return self._multigev_ccqe_norm(experiment, x, 14)

    def diff_xsec_ccqe_multigev_numu(self, experiment, x):
        return self._diff_multigev_ccqe_norm(experiment, x, 14)
