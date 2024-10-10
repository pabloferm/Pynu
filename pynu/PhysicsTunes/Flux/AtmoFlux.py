from PhysicsTunes import Tune
import numpy as np

import sys

sys.path.append("../")

####################
# Atmospheric flux #
####################


class AtmosphericFlux(Tune):
    r"""Class containing the tunes for the atmospheric neutrino flux."""

    def normalization(self, experiment, x):
        r"""Method for modifying the atmospheric flux normalization.

        Args:
            x (float): Value of the tuning parameter.
            experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.

        Returns:
            Numpy.array or float with the weights from this tune.
        """
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
        if self._unphysical_value(x):
            return 0
        zenith = np.zeros(experiment.NumberOfEvents)
        zenith[experiment.CosZTrue >= 0] = -(
            np.tanh(experiment.CosZTrue[experiment.CosZTrue >= 0]) ** 2
        )
        return zenith
