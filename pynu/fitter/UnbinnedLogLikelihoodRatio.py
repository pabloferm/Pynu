import sys
import numpy as np
import KDEpy
from .distributions import *

import numpy.typing as npt
from typing import List, Tuple, Dict

vector = npt.NDArray[np.float64]


class UnbinnedLogLikelihoodRatio:
    '''Class containing all the information needed to perform an analysis and the methods for computing
    the log likelihood ratio or χ2 given a set of observed data, expected events at a given physics point
    and nuisance parameters.

    Parameters:
        Observation_dict (Dict): Produced by PyNy and follows the structue (Experiment(str):
        events (vector)).
        NominalNuisance_list (vector): Produced from the xml analysis file, it contains the nominal
        values assumed for the nuisance parameters.
        SigmaNuisance_list (vector): Produced from the xml analysis file, it contains the standard
        deviation values assumed for the nuisance parameters.
        DistNuisance_list (List): Produced from the xml analysis file, it contains the type of
        distribution which is assumed for each nuisance.

   '''

    def __init__(self, observed_KDE_dict: Dict, NominalNuisance_list: vector,
                 SigmaNuisance_list: vector,
                 DistNuisance_list: List[str]) -> None:

        self.observed_KDE_dict = observed_KDE_dict
        self.NominalNuisance_list = NominalNuisance_list
        self.SigmaNuisance_list = SigmaNuisance_list
        self.DistNuisance_list = DistNuisance_list
        self.number_of_nuisance = len(self.NominalNuisance_list)

    def stats_only(self, Expectation_dict) -> float:
        ''' Compute statistics only binned chi-squared '''
        X2 = 0
        for O, E in zip(self.Observation_dict.values(),
                        Expectation_dict.values()):
            X2 += 2 * np.sum(E - O + O * np.log(O / E))
        return X2

    def stats_and_systematics(
            self,
            Expectation_dict,
            nuisance_vector: vector) -> float:
        if set(nuisance_vector) == set(self.NominalNuisance_list):
            return self.stats_only(Expectation_dict)
        return self.stats_only(Expectation_dict) + \
            self.nuisance_pleantly(nuisance_vector)

    def gradient(
            self,
            Expectation_dict,
            DiffExpectation_dict,
            nuisance_vector: vector) -> vector:
        nabla_X2: vector = np.zeros(len(self.NominalNuisance_list))
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
            nuisance_vector: vector) -> float:
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
                               DiffExpectation_dict) -> Tuple[vector,
                                                              Tuple[Tuple[float,
                                                                          float]]]:
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
                         prior: vector) -> vector:
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
