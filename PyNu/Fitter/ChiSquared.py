import sys
import numpy as np
from .Distributions import *


def ChiSquaredStatsOnly(Observation_dict, Expectation_dict):
    ''' Compute statistics only binned chi-squared '''
    X2 = 0
    for O, E in zip(Observation_dict.values(), Expectation_dict.values()):
        X2 += 2 * np.sum(E - O + O * np.log(O / E))
    return X2


def ChiSquared(
        Observation_dict,
        Expectation_dict,
        NominalNuisance_list,
        SigmaNuisance_list,
        DistNuisance_list,
        nuisance_vector):
    if set(nuisance_vector) == set(NominalNuisance_list):
        return ChiSquaredStatsOnly(Observation_dict, Expectation_dict)
    return ChiSquaredStatsOnly(Observation_dict,
                               Expectation_dict) + NuisancePenalty(NominalNuisance_list,
                                                                   SigmaNuisance_list,
                                                                   DistNuisance_list,
                                                                   nuisance_vector)


def ChiSquaredGradient(
        Observation_dict,
        Expectation_dict,
        DiffExpectation_dict,
        NominalNuisance_list,
        SigmaNuisance_list,
        DistNuisance_list,
        nuisance_vector):
    nabla_X2 = [0] * len(NominalNuisance_list)
    for i, (dE, mu, sig, dist, nuis) in enumerate(zip(DiffExpectation_dict.values(
    ), NominalNuisance_list, SigmaNuisance_list, DistNuisance_list, nuisance_vector)):
        if dist == 'normal':
            nabla_X2[i] += DifflogGaussianPrior(nuis, mu, sig)
        elif dist == 'beta':
            nabla_X2[i] += DifflogBetaPrior(nuis, mu, sig)
        else:
            sys.exit(
                f'Not an implemented distribution for nuisance {list(DiffExpectation_dict.keys())[i]}.')

        for O, E, dEdx in zip(Observation_dict.values(),
                              Expectation_dict.values(), dE.values()):
            nabla_X2[i] += 2 * np.sum((1 - O / E) * dEdx)

    return nabla_X2


def NuisancePenalty(
        NominalNuisance_list,
        SigmaNuisance_list,
        DistNuisance_list,
        nuisance_vector):
    X2 = 0
    for mu, sig, dist, nuis in zip(
            NominalNuisance_list, SigmaNuisance_list, DistNuisance_list,
            nuisance_vector):
        if 'normal' in dist:
            X2 += logGaussianPrior(nuis, mu, sig)
        elif dist == 'beta':
            X2 += logBetaPrior(nuis, mu, sig)
    return X2


def AnalyticPriorsBounds(
        Observation_dict,
        Expectation_dict,
        DiffExpectation_dict,
        NominalNuisance_list,
        SigmaNuisance_list):
    ''' First order analytic computation of values for parameters to be mariginalized '''
    number_of_nuisance = len(NominalNuisance_list)
    A = np.zeros(number_of_nuisance)
    B = np.zeros(number_of_nuisance)
    mu = np.array(NominalNuisance_list)
    sig = np.array(SigmaNuisance_list)

    # Experiments
    for i, dE in enumerate(DiffExpectation_dict.values()):
        for O, E, dEdx in zip(Observation_dict.values(),
                              Expectation_dict.values(), dE.values()):
            A[i] += np.sum((O / E - 1) * dEdx)
            B[i] += np.sum(O / E**2 * dEdx**2)

    # Missing non-normal distribution cases
    priors = mu + 0.5 * A / (B + 1 / sig**2)

    delta = np.minimum(2 * np.abs(priors - mu), sig)
    delta[delta == 0] = sig[delta == 0]

    bounds = np.c_[priors - delta, priors + delta]
    bounds = tuple(map(tuple, bounds))

    return priors, bounds


def AnalyticPriors_2ndOrder(
        D_Chi2_0,
        D_Chi2_1,
        X_0,
        X_1):
    ''' Second order analytic computation of values for parameters to be mariginalized
    assuming we are close enough to the minimum , i.e. a parabola'''
    priors = (D_Chi2_1 * X_0 - D_Chi2_0 * X_1) / (D_Chi2_1 - D_Chi2_0)
    return priors
