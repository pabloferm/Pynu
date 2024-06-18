import sys
import numpy as np
from .distributions import (
    diff_log_gaussian_ratio,
    diff_log_beta_ratio,
    log_gaussian_ratio,
    log_beta_ratio,
)


class BinnedLogLikelihoodRatio:
    r"""Class containing all the information needed to perform an analysis and the methods for computing
    the log likelihood ratio ($-2\ln\big(\frac{L(Exp.)}{L(Obs.)}\big)\sim\chi^2$) given a set of binned
    observed data, binned expected events at a given physics point and nuisance parameters, and assuming
    Poisson statistics.
    """

    def __init__(self, observation, nominal_nuisance, sigma_nuisance, dist_nuisance):
        r"""Initiates the class by storing the non-changing items of the $\chi^2$ calculation.

        Args:
            observation (dict): Produced by PyNuFit and follows the structue (Experiment(str): binned events (numpy.array).
            nominal_nuisance (list of float): Produced from the xml analysis file, it contains the nominal values assumed
            for the nuisance parameters.
            sigma_nuisance (list of float): Produced from the xml analysis file, it contains the standard deviation values
            assumed for the nuisance parameters.
            dist_nuisance (list of str): Produced from the xml analysis file, it contains the type of distribution which is
            assumed for each nuisance.

        """
        self.observation = observation
        self.nominal_nuisance = nominal_nuisance
        self.sigma_nuisance = sigma_nuisance
        self.dist_nuisance = dist_nuisance
        self.number_of_nuisance = len(self.nominal_nuisance)

    def stats_only(self, expectation):
        r"""Returns the value of binned $\chi^2 = 2\sum_i \Big(
        E_i-O_i+O_i\ln\big(\frac{O_i}{E_i}\big)\Big)$, given the dictionary of binned expected number of events
        for each experiment of the analysis.

        Args:
            expectation (dict): Produced by `pynu.PyNuFit` and follows the structue (Experiment(str): binned events
            (numpy.array): similarly to observation, but for a given physics and nuisance values.

        Returns:
            Float with the value of $\chi^2$.
        """
        X2: float = 0
        for O, E in zip(self.observation.values(), expectation.values()):
            X2 += 2 * np.sum(E - O + O * np.log(O / E))
        return X2

    def stats_and_systematics(self, expectation, nuisance):
        r"""Returns the value of binned $\chi^2 = 2\sum_i \Big(
        E_i-O_i+O_i\ln\big(\frac{O_i}{E_i}\big)\Big) + 2\sum_j \ln\Big(\frac{P^{nuis}_j(x)}{P^{nuis}_j(x=\mu)}\Big)$,
        given the dictionary of binned expected number of events for each experiment of the analysis and taking
        into account the nuisance penalty terms.

        Args:
            expectation (dict): Produced by PyNuFit and follows the structue (Experiment(str): binned events
            (numpy.array) similarly to observation, but for a given physics and nuisance values.
            nuisance (list of float): Values for the nuisance parameters ordered as provided by ParseXML class.

        Returns:
            Float with the value of $\chi^2$ with nuisance.
        """
        X2: float
        if set(nuisance) == set(self.nominal_nuisance):
            X2 = self.stats_only(expectation)
        else:
            X2 = self.stats_only(expectation) + self.nuisance_penalty(nuisance)
        return X2

    def gradient(self, expectation, diff_expectation, nuisance):
        r"""Returns the gradient of binned $\chi^2$ computed analytically, given the dictionary of binned
        expected number of events for each experiment of the analysis and its derivative with respect to every
        nuisance parameter.

        $\nabla_j \chi^2 = 2~\sum_{i} \Big( 1 - \frac{O_i}{E_i}\Big)\frac{\partial E_i}{\partial x_j} + \frac{2}{P^{nuis}_j(x)} \frac{d~P^{nuis}_j(x)}{dx_j}$

        LIMITATION: Currently, this is only done for nuisance following normal and beta distributions. Other distributions
        will come soon.

        Args:
            expectation (dict): Produced by `pynu.PyNuFit` and follows the structue (Experiment(str): binned events
            (numpy.array) similarly to observation, but for a given physics and nuisance values.
            diff_expectation (dict): Produced by `pynu.PyNuFit` and follows the structue (nuisance parameter (str):
            (Experiment(str): binned events (numpy.array)).
            nuisance (list of float): Values for the nuisance parameters ordered as provided by ParseXML class.

        Returns:
            Numpy array with each component of $\nabla \chi^2$.
        """
        nabla_X2 = np.zeros(len(self.nominal_nuisance))
        for i, (dE, mu, sig, dist, nuis) in enumerate(
            zip(
                diff_expectation.values(),
                self.nominal_nuisance,
                self.sigma_nuisance,
                self.dist_nuisance,
                nuisance,
            )
        ):
            if dist == "normal":
                nabla_X2[i] += diff_log_gaussian_ratio(nuis, mu, sig)
            elif dist == "beta":
                nabla_X2[i] += diff_log_beta_ratio(nuis, mu, sig)
            else:
                sys.exit(
                    f"Not an implemented distribution for nuisance {list(diff_expectation.keys())[i]}."
                )

            for O, E, dEdx in zip(
                self.observation.values(), expectation.values(), dE.values()
            ):
                nabla_X2[i] += 2 * np.sum((1 - O / E) * dEdx)

        return nabla_X2

    def nuisance_penalty(self, nuisance):
        r"""Returns the penalty term associated to nuisance parameters for the $\chi^2$ computation.

        Args:
            nuisance (list of float): Values for the nuisance parameters ordered as provided by
            `pynu.analysis_reader.ParseXML` class.

        Returns:
            Float with $\sum_j \ln\big(\frac{P^{nuis}_j(x)}{P^{nuis}_j(x=\mu)}\big)$.
        """
        X2: float = 0
        for mu, sig, dist, nuis in zip(
            self.nominal_nuisance, self.sigma_nuisance, self.dist_nuisance, nuisance
        ):
            if "normal" in dist:
                X2 += log_gaussian_ratio(nuis, mu, sig)
            elif dist == "beta":
                X2 += log_beta_ratio(nuis, mu, sig)
        return X2

    def analytic_priors_bounds(self, expectation, diff_expectation):
        r"""Returns the first-order values of the nuisance parameters which minimize the $\chi^2$ at a given
        physics points. Here, first-order means we assume that the binned expected number of events is not
        modified by nuisance parameters, i.e. nuisance parameters are assumed to take the default value in this
        approximation.

        $\nabla_j \chi^2  =0$, and at first order, $E'_i \approx E_i + \frac{\partial E_i}{\partial x_j} (x_j-\mu_j)$,
        where $E_i$ is the number of expected events with nuisance at their nomnial values.

        $\widetilde{x_j} = \mu_j + \frac{\sum \Big(\frac{O_i}{E_i} - 1 \Big) \left.\frac{\partial~E_i}{\partial x_j}\right\vert_{x_j=\mu_j} } {\sum \frac{O_i}{{E_i}^2} \Big( \left.\frac{\partial~E_i}{\partial x_j}\right\vert_{x_j=\mu_j}\Big)^2 + \frac{1}{\sigma^2_j}}$

        Further, bounds for the final values of the nuisance parameters as follows.

        $x_j\in[\widetilde{x_j}-\delta_j,\widetilde{x_j}+\delta_j]$, where $\delta_j = \min(2\cdot|\widetilde{x_j} - \mu_j|, \sigma)$

        All this information is very useful for the minimizer to find faster the values of nuisance parameters
        minimizing the $\chi^2$.

        Args:
            expectation (dict): Produced by `pynu.PyNuFit` and follows the structue (Experiment(str): binned events
            (numpy.array) similarly to observation, but for a given physics and nuisance values.
            diff_expectation (dict): Produced by `pynu.PyNuFit` and follows the structue (nuisance parameter (str):
            (Experiment(str): binned events (numpy.array)).

        Returns:
            Numpy array with the estimate for the nuisance parameters.
            Tuple with the lower and upper bounds for the nuisance parameters. Tuple(Tuple(lower,upper)).
        """
        A = np.zeros(self.number_of_nuisance)
        B = np.zeros(self.number_of_nuisance)
        mu = np.array(self.nominal_nuisance)
        sig = np.array(self.sigma_nuisance)

        # Experiments
        for i, dE in enumerate(diff_expectation.values()):
            for O, E, dEdx in zip(
                self.observation.values(), expectation.values(), dE.values()
            ):
                A[i] += np.sum((O / E - 1) * dEdx)
                B[i] += np.sum(O / E**2 * dEdx**2)

        # Missing non-normal distribution cases
        priors = mu + A / (B + 1 / sig**2)

        # delta = np.minimum(2 * np.abs(priors - mu), sig) # previous estimate
        delta = 2 * np.abs(priors - mu)
        delta[delta == 0] = sig[delta == 0]

        bounds = np.c_[priors, priors + delta]
        bounds = tuple(map(tuple, bounds))

        self.diff_expectation_nominal = diff_expectation
        self.expectation_nominal = expectation

        priors = 0.5 * (priors + mu)

        return priors, bounds

    def parabolic_priors(self, expectation_prior, diff_expectation_prior, prior):
        r"""Second order analytic computation of values for parameters to be mariginalized
        assuming we are close enough to the minimum , i.e. a parabola, i.e. linear derivative.

        $\widetilde{x_j} = \frac{\nabla_j \chi^2(x'_j) * \mu_j - \nabla_j \chi^2(\mu_j) * x'_j}{\nabla_j \chi^2(x'_j) - \nabla_j \chi^2(\mu_j)}$

        NOTE: This method does not work well and needs more thought so use it carefully.

        Args:
            expectation (dict): Produced by `pynu.PyNuFit` and follows the structue (Experiment(str): binned events
            (numpy.array) similarly to observation, but for a given physics and nuisance values.
            diff_expectation (dict): Produced by `pynu.PyNuFit` and follows the structue (nuisance parameter (str):
            (Experiment(str): binned events (numpy.array)).
            priors (list of float): Values of nuisance parameter estimates other than the nominal values.

        Returns:
            Numpy array with the estimate for the nuisance parameters.

        """

        if not self.expectation_nominal:
            sys.exit("Expectation for nominal values of nuisance not defined.")

        if not self.diff_expectation_nominal:
            sys.exit(
                "Derivative of the expectation for nominal values of nuisance not defined."
            )

        D_Chi2_1 = self.gradient(expectation_prior, diff_expectation_prior, prior)
        X_1 = prior

        D_Chi2_0 = self.gradient(
            self.expectation_nominal,
            self.diff_expectation_nominal,
            self.nominal_nuisance,
        )
        X_0 = np.array(self.nominal_nuisance)

        priors_2nd = (D_Chi2_1 * X_0 - D_Chi2_0 * X_1) / (D_Chi2_1 - D_Chi2_0)

        return priors_2nd


    def approximate_fisher(self, expectation, diff_expectation):
        I_stats = 0
        for i, dE in enumerate(diff_expectation.values()):
            for O, E, dEdx in zip(self.observation.values(),
                                  expectation.values(), dE.values()):
                I_stats += np.sum(O / E**2 * dEdx**2)

        I_prior = 0
        for sig in self.sigma_nuisance:
            I_prior += 1/sig**2

        return I_stats + I_prior
