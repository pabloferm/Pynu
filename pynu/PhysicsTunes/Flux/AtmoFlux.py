from ..PhysicsTunes import Tune
import numpy as np

####################
# Atmospheric flux #
####################


class AtmosphericFlux(Tune):
    """Class containing the tunes for the atmospheric neutrino flux."""

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
        Robbins, as used by Newtrinos.jl (`atm_flux.jl:391`): 3.5% at 0.5 GeV,
        1.4% at 1 GeV, <0.1% above ~5 GeV. The envelope carries the physical
        scale, so the nuisance prior is N(0, 1).
        """
        return 0.07 / (1.0 + (etrue / 0.5) ** 2)

    def barr_zenith(self, experiment, x):
        r"""Barr-style energy-damped up/down flux asymmetry (PE/Newtrinos.jl form).

        $w(E, \cos\theta_z; x) = (1 + \mathrm{env}(E)\,x)^{\tanh(3\cos\theta_z)}$

        Up-going flux is scaled by ~$(1+\mathrm{env}\,x)$ and down-going by its
        inverse (rate-preserving across the horizon), with a smooth
        $\tanh(3\cos\theta_z)$ transition — NOTE this deliberately also moves the
        down-going side by the reciprocal factor, unlike the one-sided
        `zenith_up`/`zenith_down` dials it is meant to replace. Mirrors
        Newtrinos.jl `updown()` (`atm_flux.jl:332-338`) with
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
