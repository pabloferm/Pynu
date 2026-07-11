from ..PhysicsTunes import Tune
import numpy as np

####################
# Atmospheric flux #
####################


class AtmosphericFlux(Tune):
    """Class containing the tunes for the atmospheric neutrino flux."""

    def solar_activity(self, experiment, x):
        w = np.ones(experiment.NumberOfEvents)
        w = 1.0 - x * 0.08 * np.exp(-experiment.ETrue / 3.0)
        return w

    def diff_solar_activity(self, experiment, x):
        w = np.ones(experiment.NumberOfEvents)
        w = - 0.08 * np.exp(-experiment.ETrue / 3.0)
        return w


    def normalization(self, experiment, x):
        r"""Method for modifying the atmospheric flux normalization.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if self._unphysical_value(x):
            return 1e-3
        return x

    def diff_normalization(self, experiment, x):
        r"""Method for computing the derivative of the weights of the atm. flux normalization
        w.r.t. the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.

        Returns:
            Numpy.array or float with the derivative of the `normalization` weights.
        """
        if self._unphysical_value(x):
            return 0
        return 1

    def normalization_below1GeV(self, experiment, x):
        r"""Method for modifying the atmospheric flux normalization
        below 1 GeV.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if self._unphysical_value(x):
            return 1e-3
        nev = np.ones(experiment.NumberOfEvents)
        nev[experiment.ETrue < 1] = x
        return nev

    def diff_normalization_below1GeV(self, experiment, x):
        r"""Method for computing the derivative of the weights of the atm. flux normalization
        below 1 GeV w.r.t. the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.

        Returns:
            Numpy.array or float with the derivative of the `normalization_below1GeV` weights.
        """
        if self._unphysical_value(x):
            return 0
        nev = np.zeros(experiment.NumberOfEvents)
        nev[experiment.ETrue < 1] = 1
        return nev

    def normalization_above1GeV(self, experiment, x):
        r"""Method for modifying the atmospheric flux normalization above 1 GeV.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if self._unphysical_value(x):
            return 1e-3
        nev = np.ones(experiment.NumberOfEvents)
        nev[experiment.ETrue > 1] = x
        return nev

    def diff_normalization_above1GeV(self, experiment, x):
        r"""Method for computing the derivative of the weights of the atm. flux normalization
        above 1 GeV w.r.t. the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.

        Returns:
            Numpy.array or float with the derivative of the `normalization_above1GeV` weights.
        """
        if self._unphysical_value(x):
            return 0
        nev = np.zeros(experiment.NumberOfEvents)
        nev[experiment.ETrue > 1] = 1
        return nev

    def tilt(self, experiment, x):
        r"""Method for modifying the power-law of the atmospheric flux normalization taking as reference
        $E_{\nu}^0 = 10~GeV$. That is $\Phi(E_{\nu}) \sim \big( \frac{E_{\nu}}{E_{\nu}^0} \big)^{x}$.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        E0Gam = 10  # GeV
        nev = (experiment.ETrue / E0Gam) ** x
        return nev

    def diff_tilt(self, experiment, x):
        r"""Method for computing the derivative of the weights of the flux tilt w.r.t. the tuning
        parameter, i.e. $\frac{\partial \Phi(E_{\nu})}{\partial x} \sim \big( \frac{E_{\nu}}{E_{\nu}^0} \big)^{x} \ln \big( \frac{E_{\nu}}{E_{\nu}^0} \big)$.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.

        Returns:
            Numpy.array or float with the derivative of the `tilt` weights.
        """
        E0Gam = 10  # GeV
        nev = (experiment.ETrue / E0Gam) ** x * np.log(experiment.ETrue / E0Gam)
        return nev

    def nunubar_ratio(self, experiment, x):
        r"""Method for modifying the neutrino and anti-neutrino ratio of the atmospheric flux.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if self._unphysical_value(x):
            return 0
        nnbar = np.ones(experiment.NumberOfEvents)
        nnbar[experiment.nuPDG < 0] = x
        return nnbar

    def diff_nunubar_ratio(self, experiment, x):
        r"""Method for computing the derivative of the neutrino anti-neutrino ratio weights w.r.t.
        the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.

        Returns:
            Numpy.array or float with the derivative of the `nunubar_ratio` weights.
        """
        if self._unphysical_value(x):
            return 0
        nnbar = np.zeros(experiment.NumberOfEvents)
        nnbar[experiment.nuPDG < 0] = 1
        return nnbar

    def flavor_ratio(self, experiment, x):
        r"""Method for modifying the neutrino flavor ratio of the atmospheric flux.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        if self._unphysical_value(x):
            return 0
        eovermu = np.ones(experiment.NumberOfEvents)
        eovermu[np.abs(experiment.nuPDG) == 12] = x
        return eovermu

    def diff_flavor_ratio(self, experiment, x):
        r"""Method for computing the derivative of the neutrino falvor ratio weights w.r.t.
        the tuning parameter.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.

        Returns:
            Numpy.array or float with the derivative of the `flavor_ratio` weights.
        """
        if self._unphysical_value(x):
            return 0
        eovermu = np.zeros(experiment.NumberOfEvents)
        eovermu[abs(experiment.nuPDG) == 12] = 1
        return eovermu

    def zenith_up(self, experiment, x):
        r"""Method for modifying the zenith angle dependence of the up-going (negative
        $\cos \theta_{zen}$) fraction of the atmospheric flux assuming the relative
        uncertainty is parametrized as,
        $\eta(\cos \theta_{zen}) = 1 - x * \tanh^2 (\cos \theta_{zen})$.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        zenith = np.ones(experiment.NumberOfEvents)
        zenith[experiment.CosZTrue < 0] = (
            zenith[experiment.CosZTrue < 0]
            - x * np.tanh(experiment.CosZTrue[experiment.CosZTrue < 0]) ** 2
        )
        return zenith

    def diff_zenith_up(self, experiment, x):
        r"""Method for computing the derivative of the weights of zenith-dependence variation of
        up-going neutrinos w.r.t. the tuning parameter, i.e.
        $\frac{d \eta(\cos \theta_{zen})}{d x} = - \tanh^2 (\cos \theta_{zen})$.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.

        Returns:
            Numpy.array or float with the derivative of the `zenith_up` weights.
        """
        zenith = np.zeros(experiment.NumberOfEvents)
        zenith[experiment.CosZTrue < 0] = -(
            np.tanh(experiment.CosZTrue[experiment.CosZTrue < 0]) ** 2
        )
        return zenith

    def zenith_down(self, experiment, x):
        r"""Method for modifying the zenith angle dependence of the down-going (positive
        $\cos \theta_{zen}$) fraction of the atmospheric flux assuming the relative
        uncertainty is parametrized as,
        $\eta(\cos \theta_{zen}) = 1 - x * \tanh^2 (\cos \theta_{zen})$.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
        zenith = np.ones(experiment.NumberOfEvents)
        zenith[experiment.CosZTrue >= 0] = (
            zenith[experiment.CosZTrue >= 0]
            - x * np.tanh(experiment.CosZTrue[experiment.CosZTrue >= 0]) ** 2
        )
        return zenith

    def diff_zenith_down(self, experiment, x):
        r"""Method for computing the derivative of the weights of zenith-dependence variation of
        up-going neutrinos w.r.t. the tuning parameter, i.e.
        $\frac{d \eta(\cos \theta_{zen})}{d x} = - \tanh^2 (\cos \theta_{zen})$.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.

        Returns:
            Numpy.array or float with the derivative of the `zenith_up` weights.
        """
        zenith = np.zeros(experiment.NumberOfEvents)
        zenith[experiment.CosZTrue >= 0] = -(
            np.tanh(experiment.CosZTrue[experiment.CosZTrue >= 0]) ** 2
        )
        return zenith

    @staticmethod
    def _barr_zenith_envelope(etrue):
        r"""Barr up/down flux-ratio uncertainty envelope (fractional, 1-sigma).

        $\mathrm{env}(E) = 0.07 / (1 + (E/0.5\,\mathrm{GeV})^2)$ — fit to Barr &
        Robbins: 3.5% at 0.5 GeV,
        1.4% at 1 GeV, <0.1% above ~5 GeV. The envelope carries the physical
        scale, so the nuisance prior is N(0, 1).
        """
        return 0.07 / (1.0 + (etrue / 0.5) ** 2)

    def barr_zenith(self, experiment, x):
        r"""Barr-style energy-damped up/down flux asymmetry.

        $w(E, \cos\theta_z; x) = (1 + \mathrm{env}(E)\,x)^{\tanh(3\cos\theta_z)}$

        Up-going flux is scaled by ~$(1+\mathrm{env}\,x)$ and down-going by its
        inverse (rate-preserving across the horizon), with a smooth
        $\tanh(3\cos\theta_z)$ transition — NOTE this deliberately also moves the
        down-going side by the reciprocal factor, unlike the one-sided
        `zenith_up`/`zenith_down` dials it is meant to replace. Equivalent
        to an up/down ratio reweight with
        $r = 1 + \mathrm{env}(E)\,x$.

        Args:
            x (float): Value of the tuning parameter (nominal 0, prior N(0,1)).
            experiment: Experiment class with per-event `ETrue`, `CosZTrue`.

        Returns:
            Numpy.array with the per-event weights from this tune.
        """
        env = self._barr_zenith_envelope(experiment.ETrue)
        r = 1.0 + env * x
        if np.any(r <= 0):
            return 1e-3 * np.ones(experiment.NumberOfEvents)
        return r ** np.tanh(3.0 * experiment.CosZTrue)

    def diff_barr_zenith(self, experiment, x):
        r"""Derivative of `barr_zenith` w.r.t. the tuning parameter.

        $\frac{dw}{dx} = \tanh(3\cos\theta_z)\,\mathrm{env}(E)\,
        (1 + \mathrm{env}(E)\,x)^{\tanh(3\cos\theta_z) - 1}$

        Args:
            x (float): Value of the tuning parameter.
            experiment: Experiment class with per-event `ETrue`, `CosZTrue`.

        Returns:
            Numpy.array with the derivative of the `barr_zenith` weights.
        """
        env = self._barr_zenith_envelope(experiment.ETrue)
        r = 1.0 + env * x
        if np.any(r <= 0):
            return np.zeros(experiment.NumberOfEvents)
        t = np.tanh(3.0 * experiment.CosZTrue)
        return t * env * r ** (t - 1.0)

    # =====================================================================
    #  r2_fude_ccqe event-engine dials
    #  Transcribed 1:1 from the binned engine (ENG = pynu/binned/
    #  sk_binned_engine.py); each method is line-cited to the ENG:<line>
    #  block it mirrors. All are W-type (per-event weight) dials. The
    #  unphysical-value guard returns 1e-3 (the binned engine's punishment
    #  convention).
    # =====================================================================

    #: K/pi high-E flux ramp pivot (GeV) — ENG:115 (KPI_E0). Below this the
    #: ramp is 0; above it grows as log10(E/KPI_E0).
    _KPI_E0 = 3.0

    def kpi_ratio(self, experiment, x):
        r"""K/pi high-energy flux ramp (SK's kaon-onset flux norm).

        Flavor-blind multiplicative factor growing logarithmically above the
        kaon onset: $w = 1 + x\,\max(0, \log_{10}(E_\nu / E_0))$, $E_0 = 3$ GeV.
        ~0 below ~3 GeV, rising with energy. nominal x=0 (exact no-op).
        Mirrors sk_binned_engine.py:1095-1096 (apply) / :1941-1943 (grad),
        shape `kpi_shape` at :1038.

        Args:
            x (float): Value of the tuning parameter (nominal 0).
            experiment: Experiment class with per-event `ETrue`.

        Returns:
            Numpy.array with the per-event weights from this tune.
        """
        if self._unphysical_value(x, unphys_low=-9999999):
            return 1e-3
        ramp = np.maximum(0.0, np.log10(experiment.ETrue / self._KPI_E0))
        return 1.0 + x * ramp

    def diff_kpi_ratio(self, experiment, x):
        r"""Derivative of `kpi_ratio` w.r.t. x: $dw/dx = \max(0, \log_{10}(E/E_0))$."""
        if self._unphysical_value(x, unphys_low=-9999999):
            return 0
        return np.maximum(0.0, np.log10(experiment.ETrue / self._KPI_E0))

    def flux_horizvert(self, experiment, x):
        r"""Horizontal/vertical flux-ratio SHAPE (SK thesis 5509-18).

        The zenith-SHAPE flux dial that zenith_up/down structurally cannot make
        (their tanh^2 envelope pivots to zero at the horizon). Energy-flat,
        symmetric in cosZ, mean-zero over cosZ (a shape, not a norm):
        $w = 1 + x\,g(\cos\theta_z)$ with $g = (1 - 3\cos^2\theta_z)/2$
        (+0.5 horizontal, -1.0 vertical). nominal x=0 (exact no-op).
        Mirrors sk_binned_engine.py:1116-1118 (apply) / :1931-1935 (grad),
        `horizvert_shape` at :1035.

        Args:
            x (float): Value of the tuning parameter (nominal 0).
            experiment: Experiment class with per-event `CosZTrue`.

        Returns:
            Numpy.array with the per-event weights from this tune.
        """
        if self._unphysical_value(x, unphys_low=-9999999):
            return 1e-3
        g = 0.5 * (1.0 - 3.0 * experiment.CosZTrue ** 2)
        return 1.0 + x * g

    def diff_flux_horizvert(self, experiment, x):
        r"""Derivative of `flux_horizvert` w.r.t. x: $dw/dx = (1 - 3\cos^2\theta_z)/2$."""
        if self._unphysical_value(x, unphys_low=-9999999):
            return 0
        return 0.5 * (1.0 - 3.0 * experiment.CosZTrue ** 2)

    # ---- energy-banded flux-ratio dials (rate-conserving symmetric pairs) ----
    # Nine dials, each a (band, heavy pdg leg, light pdg leg) triple. The
    # rate-conserving symmetric ("FEATURE") convention transcribed from
    # sk_binned_engine.py:1130-1139 (apply) / :1789-1804 (grad), spec
    # FLUX_RATIO_SPEC at :235-245:
    #     heavy leg *= 1 + band*(2r/(1+r) - 1)
    #     light leg *= 1 + band*(2/(1+r)  - 1)
    # where r == the DIAL VALUE x itself (NOT a BaseWeight rate ratio) and
    # `band` is the true-energy band indicator (1 inside the band, 0 outside).
    # r=1 => both leg factors are 1 (exact no-op). The binned engine uses the
    # dial value directly as r — it does not recompute a per-leg weighted rate
    # ratio — and this implementation matches that convention exactly.
    #
    # Energy bands on ETrue (ENG:1041-1043): sub E<1, mid 1<=E<10, high E>=10.
    # pdg legs (ENG:929-933): nuebar pdg==-12, nue pdg==12, e |pdg|==12,
    # mu |pdg|==14, numubar pdg==-14, numu pdg==14.
    _FLUX_RATIO_SPEC = {
        "flux_nuebar_subgev":  ("sub", "nuebar", "nue"),
        "flux_nuebar_mid":     ("mid", "nuebar", "nue"),
        "flux_nuebar_high":    ("high", "nuebar", "nue"),
        "flux_flavor_subgev":  ("sub", "mu", "e"),
        "flux_flavor_mid":     ("mid", "mu", "e"),
        "flux_flavor_high":    ("high", "mu", "e"),
        "flux_numubar_subgev": ("sub", "numubar", "numu"),
        "flux_numubar_mid":    ("mid", "numubar", "numu"),
        "flux_numubar_high":   ("high", "numubar", "numu"),
    }

    @staticmethod
    def _flux_ratio_band(experiment, band):
        r"""True-energy band indicator (1 inside, 0 outside). ENG:1041-1043."""
        E = experiment.ETrue
        if band == "sub":
            return (E < 1.0)
        if band == "mid":
            return (E >= 1.0) & (E < 10.0)
        if band == "high":
            return (E >= 10.0)
        raise ValueError(f"unknown flux band {band!r}")

    @staticmethod
    def _flux_ratio_leg(experiment, leg):
        r"""Per-event pdg leg mask. ENG:929-933."""
        pdg = experiment.nuPDG
        if leg == "nuebar":
            return pdg == -12
        if leg == "nue":
            return pdg == 12
        if leg == "e":
            return np.abs(pdg) == 12
        if leg == "mu":
            return np.abs(pdg) == 14
        if leg == "numubar":
            return pdg == -14
        if leg == "numu":
            return pdg == 14
        raise ValueError(f"unknown flux leg {leg!r}")

    def _flux_ratio_weight(self, experiment, name, x):
        r"""Rate-conserving symmetric flux-ratio weight for `name` at x.

        heavy *= 1 + band*(2x/(1+x) - 1); light *= 1 + band*(2/(1+x) - 1).
        r == x (the dial value). r=1 -> no-op. Guard: x<=0 makes 1+x<=0
        (division blow-up) => punish with 1e-3.
        """
        if self._unphysical_value(x):
            return 1e-3
        band_name, hv, lt = self._FLUX_RATIO_SPEC[name]
        band = self._flux_ratio_band(experiment, band_name)
        heavy = self._flux_ratio_leg(experiment, hv)
        light = self._flux_ratio_leg(experiment, lt)
        fh = 1.0 + band * (2.0 * x / (1.0 + x) - 1.0)     # heavy leg
        fl = 1.0 + band * (2.0 / (1.0 + x) - 1.0)         # light leg
        w = np.ones(experiment.NumberOfEvents)
        w[heavy] = fh[heavy]
        w[light] = fl[light]
        return w

    def _diff_flux_ratio_weight(self, experiment, name, x):
        r"""Derivative of `_flux_ratio_weight` w.r.t. x (matches ENG:1795-1802).

        heavy: dw/dx = band * d/dx[2x/(1+x)] = band * 2/(1+x)^2
        light: dw/dx = band * d/dx[2/(1+x)]  = band * -2/(1+x)^2
        """
        if self._unphysical_value(x):
            return 0
        band_name, hv, lt = self._FLUX_RATIO_SPEC[name]
        band = self._flux_ratio_band(experiment, band_name)
        heavy = self._flux_ratio_leg(experiment, hv)
        light = self._flux_ratio_leg(experiment, lt)
        dh = 2.0 / (1.0 + x) ** 2
        dl = -2.0 / (1.0 + x) ** 2
        w = np.zeros(experiment.NumberOfEvents)
        w[heavy] = (band * dh)[heavy]
        w[light] = (band * dl)[light]
        return w

    def flux_nuebar_subgev(self, experiment, x):
        r"""nu-bar_e/nu_e flux ratio, sub-GeV band. See `_flux_ratio_weight`."""
        return self._flux_ratio_weight(experiment, "flux_nuebar_subgev", x)

    def diff_flux_nuebar_subgev(self, experiment, x):
        return self._diff_flux_ratio_weight(experiment, "flux_nuebar_subgev", x)

    def flux_nuebar_mid(self, experiment, x):
        r"""nu-bar_e/nu_e flux ratio, mid-energy band. See `_flux_ratio_weight`."""
        return self._flux_ratio_weight(experiment, "flux_nuebar_mid", x)

    def diff_flux_nuebar_mid(self, experiment, x):
        return self._diff_flux_ratio_weight(experiment, "flux_nuebar_mid", x)

    def flux_nuebar_high(self, experiment, x):
        r"""nu-bar_e/nu_e flux ratio, high-energy band. See `_flux_ratio_weight`."""
        return self._flux_ratio_weight(experiment, "flux_nuebar_high", x)

    def diff_flux_nuebar_high(self, experiment, x):
        return self._diff_flux_ratio_weight(experiment, "flux_nuebar_high", x)

    def flux_flavor_subgev(self, experiment, x):
        r"""(nu_mu+nu-bar_mu)/(nu_e+nu-bar_e) flux ratio, sub-GeV band."""
        return self._flux_ratio_weight(experiment, "flux_flavor_subgev", x)

    def diff_flux_flavor_subgev(self, experiment, x):
        return self._diff_flux_ratio_weight(experiment, "flux_flavor_subgev", x)

    def flux_flavor_mid(self, experiment, x):
        r"""(nu_mu+nu-bar_mu)/(nu_e+nu-bar_e) flux ratio, mid-energy band."""
        return self._flux_ratio_weight(experiment, "flux_flavor_mid", x)

    def diff_flux_flavor_mid(self, experiment, x):
        return self._diff_flux_ratio_weight(experiment, "flux_flavor_mid", x)

    def flux_flavor_high(self, experiment, x):
        r"""(nu_mu+nu-bar_mu)/(nu_e+nu-bar_e) flux ratio, high-energy band."""
        return self._flux_ratio_weight(experiment, "flux_flavor_high", x)

    def diff_flux_flavor_high(self, experiment, x):
        return self._diff_flux_ratio_weight(experiment, "flux_flavor_high", x)

    def flux_numubar_subgev(self, experiment, x):
        r"""nu-bar_mu/nu_mu flux ratio, sub-GeV band. See `_flux_ratio_weight`."""
        return self._flux_ratio_weight(experiment, "flux_numubar_subgev", x)

    def diff_flux_numubar_subgev(self, experiment, x):
        return self._diff_flux_ratio_weight(experiment, "flux_numubar_subgev", x)

    def flux_numubar_mid(self, experiment, x):
        r"""nu-bar_mu/nu_mu flux ratio, mid-energy band. See `_flux_ratio_weight`."""
        return self._flux_ratio_weight(experiment, "flux_numubar_mid", x)

    def diff_flux_numubar_mid(self, experiment, x):
        return self._diff_flux_ratio_weight(experiment, "flux_numubar_mid", x)

    def flux_numubar_high(self, experiment, x):
        r"""nu-bar_mu/nu_mu flux ratio, high-energy band. See `_flux_ratio_weight`."""
        return self._flux_ratio_weight(experiment, "flux_numubar_high", x)

    def diff_flux_numubar_high(self, experiment, x):
        return self._diff_flux_ratio_weight(experiment, "flux_numubar_high", x)
