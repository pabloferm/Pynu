'''
====================================================
Chebyshev Series (:mod:`numpy.polynomial.chebyshev`)
====================================================

This module provides a number of objects (mostly functions) useful for
dealing with Chebyshev series, including a `Chebyshev` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Class
-------

.. autosummary::
   :toctree: generated/

   BinnedLogLikelihoodRatio


Constants
---------

.. autosummary::
   :toctree: generated/

   chebdomain
   chebzero
   chebone
   chebx

Arithmetic
----------

.. autosummary::
   :toctree: generated/

   chebadd
   chebsub
   chebmulx
   chebmul
   chebdiv
   chebpow
   chebval
   chebval2d
   chebval3d
   chebgrid2d
   chebgrid3d

Calculus
--------

.. autosummary::
   :toctree: generated/

   chebder
   chebint

Misc Functions
--------------

.. autosummary::
   :toctree: generated/

   chebfromroots
   chebroots
   chebvander

Notes
-----
The implementations of multiplication, division, integration, and
differentiation use the algebraic identities [1]_:

.. math ::
    T_n(x) = \\frac{z^n + z^{-n}}{2} \\\\
    z\\frac{dx}{dz} = \\frac{z - z^{-1}}{2}.

where

.. math :: x = \\frac{z + z^{-1}}{2}.

'''

import sys
import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Dict
from .Distributions import *

vector = npt.NDArray[np.float64]


class BinnedLogLikelihoodRatio:
    def __init__(self, Observation_dict, NominalNuisance_list: vector,
                 SigmaNuisance_list: vector,
                 DistNuisance_list: List[str]) -> None:

        self.Observation_dict = Observation_dict
        self.NominalNuisance_list = NominalNuisance_list
        self.SigmaNuisance_list = SigmaNuisance_list
        self.DistNuisance_list = DistNuisance_list
        self.number_of_nuisance = len(self.NominalNuisance_list)

    def StatsOnly(self, Expectation_dict) -> float:
        ''' Compute statistics only binned chi-squared '''
        X2 = 0
        for O, E in zip(self.Observation_dict.values(),
                        Expectation_dict.values()):
            X2 += 2 * np.sum(E - O + O * np.log(O / E))
        return X2

    def StatsAndSystematics(
            self,
            Expectation_dict,
            nuisance_vector: vector) -> float:
        if set(nuisance_vector) == set(self.NominalNuisance_list):
            return self.StatsOnly(Expectation_dict)
        return self.StatsOnly(Expectation_dict) + \
            self.NuisancePenalty(nuisance_vector)

    def Gradient(
            self,
            Expectation_dict,
            DiffExpectation_dict,
            nuisance_vector: vector) -> vector:
        nabla_X2: vector = np.zeros(len(self.NominalNuisance_list))
        for i, (dE, mu, sig, dist, nuis) in enumerate(zip(DiffExpectation_dict.values(
        ), self.NominalNuisance_list, self.SigmaNuisance_list, self.DistNuisance_list, nuisance_vector)):
            if dist == 'normal':
                nabla_X2[i] += DifflogGaussianPrior(nuis, mu, sig)
            elif dist == 'beta':
                nabla_X2[i] += DifflogBetaPrior(nuis, mu, sig)
            else:
                sys.exit(
                    f'Not an implemented distribution for nuisance {list(DiffExpectation_dict.keys())[i]}.')

            for O, E, dEdx in zip(self.Observation_dict.values(),
                                  Expectation_dict.values(), dE.values()):
                nabla_X2[i] += 2 * np.sum((1 - O / E) * dEdx)

        return nabla_X2

    def NuisancePenalty(
            self,
            nuisance_vector: vector) -> float:
        X2: float = 0.0
        for mu, sig, dist, nuis in zip(
                self.NominalNuisance_list, self.SigmaNuisance_list, self.DistNuisance_list,
                nuisance_vector):
            if 'normal' in dist:
                X2 += logGaussianPrior(nuis, mu, sig)
            elif dist == 'beta':
                X2 += logBetaPrior(nuis, mu, sig)
        return X2

    def AnalyticPriorsBounds(self,
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

    def AnalyticPriors_2ndOrder(self,
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

        D_Chi2_1 = self.Gradient(
            Expectation_dict_prior,
            DiffExpectation_dict_prior,
            prior)
        X_1 = prior

        D_Chi2_0 = self.Gradient(
            self.Expectation_dict_Nominal,
            self.DiffExpectation_dict_Nominal,
            self.NominalNuisance_list)
        X_0 = np.array(self.NominalNuisance_list)

        priors_2nd = (D_Chi2_1 * X_0 - D_Chi2_0 * X_1) / (D_Chi2_1 - D_Chi2_0)

        return priors_2nd
