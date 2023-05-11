import sys
import numpy as np
import KDEpy
from .distributions import *


class UnbinnedLogLikelihoodRatio:
    r"""Class containing all the information needed to perform an analysis and the methods for computing
    the log likelihood ratio ($-2\ln\big(\frac{L(Exp.)}{L(Obs.)}\big)\sim\chi^2$) given a set of binned
    observed data, binned expected events at a given physics point and nuisance parameters, and assuming
    Poisson statistics.

    Args:
        obervation (dict): Produced by PyNuFit and follows the structue (Experiment(str): binned events (numpy.array).
        nominal_nuisance (list of float): Produced from the xml analysis file, it contains the nominal values assumed for the nuisance parameters.
        sigma_nuisance (list of float): Produced from the xml analysis file, it contains the standard deviation values assumed for the nuisance parameters.
        dist_nuisance (list of str): Produced from the xml analysis file, it contains the type of distribution which is assumed for each nuisance.

   """

    def __init__(self, observed_KDE_dict, NominalNuisance_list,
                 SigmaNuisance_list,
                 DistNuisance_list):

        self.observed_KDE_dict = observed_KDE_dict
        self.NominalNuisance_list = NominalNuisance_list
        self.SigmaNuisance_list = SigmaNuisance_list
        self.DistNuisance_list = DistNuisance_list
        self.number_of_nuisance = len(self.NominalNuisance_list)

    def stats_only(self, Expectation_dict):
        ''' Compute statistics only binned chi-squared '''
        X2 = 0
        for O, E in zip(self.Observation_dict.values(),
                        Expectation_dict.values()):
            X2 += 2 * np.sum(E - O + O * np.log(O / E))
        return X2

    def stats_and_systematics(
            self,
            Expectation_dict,
            nuisance_vector):
        if set(nuisance_vector) == set(self.NominalNuisance_list):
            return self.stats_only(Expectation_dict)
        return self.stats_only(Expectation_dict) + \
            self.nuisance_pleantly(nuisance_vector)

    def gradient(
            self,
            Expectation_dict,
            DiffExpectation_dict,
            nuisance_vector):
        nabla_X2 = np.zeros(len(self.NominalNuisance_list))
        for i, (dE, mu, sig, dist, nuis) in enumerate(zip(DiffExpectation_dict.values(
        ), self.NominalNuisance_list, self.SigmaNuisance_list, self.DistNuisance_list, nuisance_vector)):
            if dist == 'normal':
                nabla_X2[i] += diff_log_gaussian_ratio(nuis, mu, sig)
            elif dist == 'beta':
                nabla_X2[i] += diff_log_beta_ratio(nuis, mu, sig)
            else:
                sys.exit(
                    f'Not an implemented distribution for nuisance {list(DiffExpectation_dict.keys())[i]}.')

            for O, E, dEdx in zip(self.Observation_dict.values(),
                                  Expectation_dict.values(), dE.values()):
                nabla_X2[i] += 2 * np.sum((1 - O / E) * dEdx)

        return nabla_X2

    def nuisance_pleantly(
            self,
            nuisance_vector):
        X2: float = 0.0
        for mu, sig, dist, nuis in zip(
                self.NominalNuisance_list, self.SigmaNuisance_list, self.DistNuisance_list,
                nuisance_vector):
            if 'normal' in dist:
                X2 += log_gaussian_ratio(nuis, mu, sig)
            elif dist == 'beta':
                X2 += log_beta_ratio(nuis, mu, sig)
        return X2

    def analytic_priors_bounds(self,
                               Expectation_dict,
                               DiffExpectation_dict):
        ''' First order analytic computation of values for parameters to be mariginalized '''
        A = np.zeros(self.number_of_nuisance)
        B = np.zeros(self.number_of_nuisance)
        mu = np.array(self.NominalNuisance_list)
        sig = np.array(self.SigmaNuisance_list)

        # Experiments
        for i, dE in enumerate(DiffExpectation_dict.values()):
            for O, E, dEdx in zip(self.Observation_dict.values(),
                                  Expectation_dict.values(), dE.values()):
                A[i] += np.sum((O / E - 1) * dEdx)
                B[i] += np.sum(O / E**2 * dEdx**2)

        # Missing non-normal distribution cases
        priors = mu + 0.5 * A / (B + 1 / sig**2)

        delta = np.minimum(2 * np.abs(priors - mu), sig)
        delta[delta == 0] = sig[delta == 0]

        bounds = np.c_[priors - delta, priors + delta]
        bounds = tuple(map(tuple, bounds))

        self.DiffExpectation_dict_Nominal = DiffExpectation_dict
        self.Expectation_dict_Nominal = Expectation_dict

        return priors, bounds

    def parabolic_priors(self,
                         Expectation_dict_prior,
                         DiffExpectation_dict_prior,
                         prior):
        ''' Second order analytic computation of values for parameters to be mariginalized
        assuming we are close enough to the minimum , i.e. a parabola, i.e. linear derivative'''

        if not self.Expectation_dict_Nominal:
            sys.exit('Expectation for nominal values of nuisance not defined.')

        if not self.DiffExpectation_dict_Nominal:
            sys.exit(
                'Derivative of the expectation for nominal values of nuisance not defined.')

        D_Chi2_1 = self.gradient(
            Expectation_dict_prior,
            DiffExpectation_dict_prior,
            prior)
        X_1 = prior

        D_Chi2_0 = self.gradient(
            self.Expectation_dict_Nominal,
            self.DiffExpectation_dict_Nominal,
            self.NominalNuisance_list)
        X_0 = np.array(self.NominalNuisance_list)

        priors_2nd = (D_Chi2_1 * X_0 - D_Chi2_0 * X_1) / (D_Chi2_1 - D_Chi2_0)

        return priors_2nd
