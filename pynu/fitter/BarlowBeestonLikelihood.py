import sys
from typing import Dict, List, Any, Optional
import numpy as np
from .distributions import (
    diff_log_gaussian_ratio,
    diff_log_beta_ratio,
    log_gaussian_ratio,
    log_beta_ratio,
)


class BarlowBeestonLikelihood:
    """
    Barlow-Beeston likelihood that accounts for MC statistical uncertainty.

    This likelihood treats the MC prediction in each bin as having uncertainty
    from finite MC statistics, introducing auxiliary parameters (beta) that
    scale the prediction.

    Supports:
    - MC statistical uncertainty via beta scaling factors
    - Muon background (optional, for ORCA-like experiments)
    - Backward compatibility with experiments without muons

    The chi-squared is:
        chi2 = sum_i[2*(beta_i*N_mod_i - N_dat_i + N_dat_i*ln(N_dat_i/(beta_i*N_mod_i)))]
             + sum_i[(beta_i - 1)^2 / tau_i]
             + sum_j[penalty_j]

    where tau_i = sigma^2_MC_i / N_mod_i^2 is the relative MC variance.
    """

    def __init__(
        self, observation, nominal_nuisance, sigma_nuisance, dist_nuisance
    ) -> None:
        """
        Initialize the Barlow-Beeston likelihood.

        Args:
            observation (dict): Experiment name -> binned observed events
            nominal_nuisance (list): Nominal values for nuisance parameters
            sigma_nuisance (list): Standard deviations for nuisance parameters
            dist_nuisance (list): Distribution types ('normal', 'beta') for each nuisance
        """
        self.observation = observation
        self.nominal_nuisance = nominal_nuisance
        self.sigma_nuisance = sigma_nuisance
        self.dist_nuisance = dist_nuisance
        self.number_of_nuisance = len(self.nominal_nuisance)
        # MC variance and muon background will be set dynamically
        self.mc_variance = None
        self.muon_background = None  # Dict: exp_name -> (binned_counts, binned_variance)

    def set_mc_variance(self, mc_variance):
        """Set the MC variance dictionary."""
        self.mc_variance = mc_variance

    def set_muon_background(self, muon_background):
        """
        Set the muon background dictionary.

        Args:
            muon_background: Dict of experiment_name -> (binned_counts, binned_variance)
                             or None if experiment has no muons
        """
        self.muon_background = muon_background

    def _compute_beta(self, N_mod, mc_var, N_dat):
        """
        Compute optimal beta values using Barlow-Beeston lite formula.

        Solves: beta = argmin[ 2*(beta*N_mod - N_dat + N_dat*ln(N_dat/(beta*N_mod))) + (beta-1)^2/tau ]

        Using quadratic formula: beta = 0.5 * (-b + sqrt(b^2 - 4c))
        where b = N_mod*tau - 1, c = -N_dat*tau

        Args:
            N_mod: Model prediction array
            mc_var: MC variance array
            N_dat: Observed data array

        Returns:
            beta: Optimal scaling factors
            tau: Relative variance (sigma^2/N_mod^2)
        """
        # Compute tau = sigma^2_MC / N_mod^2
        tau = np.divide(mc_var, N_mod**2, out=np.zeros_like(mc_var), where=N_mod != 0)

        # Quadratic formula coefficients
        b = N_mod * tau - 1.0
        c = -N_dat * tau

        # Solve quadratic: beta = 0.5 * (-b + sqrt(b^2 - 4c))
        discriminant = np.maximum(0, b**2 - 4*c)
        beta = 0.5 * (-b + np.sqrt(discriminant))

        # Safeguard: beta should be positive
        beta = np.maximum(beta, 1e-9)

        return beta, tau

    def _get_total_model_and_variance(self, exp_name, E_nu, mc_var_nu):
        """
        Combine neutrino model with muon background if available.

        Args:
            exp_name: Experiment name
            E_nu: Neutrino expectation (binned)
            mc_var_nu: Neutrino MC variance (binned)

        Returns:
            E_total: Total model (neutrino + muon)
            mc_var_total: Total MC variance
            has_muons: Whether muons were added
        """
        # Check if this experiment has muon background
        if (self.muon_background is not None and
            exp_name in self.muon_background and
            self.muon_background[exp_name] is not None):

            muon_counts, muon_var = self.muon_background[exp_name]

            # Ensure shapes match
            if muon_counts is not None and len(muon_counts) == len(E_nu):
                E_total = E_nu + muon_counts
                mc_var_total = mc_var_nu + muon_var
                return E_total, mc_var_total, True

        # No muons - return neutrino only
        return E_nu, mc_var_nu, False

    def stats_only(self, expectation, mc_variance=None) -> float:
        """
        Compute Barlow-Beeston chi-squared (statistics only, no nuisance penalty).

        Args:
            expectation (dict): Experiment name -> binned expected events (neutrino only)
            mc_variance (dict): Experiment name -> binned MC variance (optional)

        Returns:
            Chi-squared value
        """
        if mc_variance is None:
            mc_variance = self.mc_variance
        if mc_variance is None:
            # Fall back to standard Poisson if no variance provided
            X2 = 0.0
            for (exp_name, O), E_nu in zip(self.observation.items(), expectation.values()):
                # Add muon background if available
                E_total, _, _ = self._get_total_model_and_variance(exp_name, E_nu, np.zeros_like(E_nu))
                if np.any(E_total <= 0):
                    X2 = 9e9
                X2 += np.sum(E_total - O + O * np.log(O / E_total))
            return 2 * X2

        X2 = 0.0
        for (exp_name, O), E_nu in zip(self.observation.items(), expectation.values()):
            mc_var_nu = mc_variance.get(exp_name, np.zeros_like(E_nu))

            # Combine neutrino + muon if available
            E_total, mc_var_total, has_muons = self._get_total_model_and_variance(
                exp_name, E_nu, mc_var_nu
            )

            # Compute optimal beta using total model
            beta, tau = self._compute_beta(E_total, mc_var_total, O)

            # Compute chi-squared terms
            beta_E = np.maximum(beta * E_total, 1e-9)

            # Poisson term: 2*(beta*E - O + O*ln(O/(beta*E)))
            log_term = np.log(np.divide(O, beta_E, out=np.ones_like(O), where=beta_E > 0))
            log_term[O == 0] = 0  # Handle 0*log(0) = 0
            poisson_chi2 = np.sum(2 * (beta_E - O + O * log_term))

            # BB penalty term: (beta - 1)^2 / tau
            bb_penalty = np.sum(np.divide((beta - 1)**2, tau, out=np.zeros_like(tau), where=tau > 0))

            X2 += poisson_chi2 + bb_penalty

        return X2

    def stats_and_systematics(self, expectation, nuisance: List[float], mc_variance=None) -> float:
        """
        Compute Barlow-Beeston chi-squared with nuisance penalties.

        Args:
            expectation (dict): Experiment name -> binned expected events
            nuisance (list): Current nuisance parameter values
            mc_variance (dict): Optional MC variance override

        Returns:
            Chi-squared value including nuisance penalties
        """
        return self.stats_only(expectation, mc_variance) + self.nuisance_penalty(nuisance)

    def gradient(self, expectation, diff_expectation, nuisance: List[float], mc_variance=None) -> np.ndarray:
        """
        Compute gradient of Barlow-Beeston chi-squared w.r.t. nuisance parameters.

        Uses first-order approximation: ignores d(beta)/d(nuisance) terms.

        Note: For experiments with muons, the gradient only accounts for neutrino
        systematic effects. Muon-specific systematics would need separate handling.

        Args:
            expectation (dict): Experiment name -> binned expected events
            diff_expectation (dict): Nuisance name -> (Experiment name -> dE/d(nuisance))
            nuisance (list): Current nuisance parameter values
            mc_variance (dict): Optional MC variance override

        Returns:
            Gradient array
        """
        if mc_variance is None:
            mc_variance = self.mc_variance

        nabla_X2 = np.zeros(self.number_of_nuisance)

        # If no MC variance, fall back to standard gradient
        if mc_variance is None:
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
                    sys.exit(f"Not an implemented distribution for nuisance {list(diff_expectation.keys())[i]}.")

                for (exp_name, O), E_nu, dEdx in zip(
                    self.observation.items(), expectation.values(), dE.values()
                ):
                    # Add muon background for total model
                    E_total, _, _ = self._get_total_model_and_variance(exp_name, E_nu, np.zeros_like(E_nu))
                    nabla_X2[i] += 2 * np.sum((1 - O / E_total) * dEdx)
            return nabla_X2

        for i, (dE, mu, sig, dist, nuis) in enumerate(
            zip(
                diff_expectation.values(),
                self.nominal_nuisance,
                self.sigma_nuisance,
                self.dist_nuisance,
                nuisance,
            )
        ):
            # Nuisance penalty gradient
            if dist == "normal":
                nabla_X2[i] += diff_log_gaussian_ratio(nuis, mu, sig)
            elif dist == "beta":
                nabla_X2[i] += diff_log_beta_ratio(nuis, mu, sig)
            else:
                sys.exit(f"Not an implemented distribution for nuisance {list(diff_expectation.keys())[i]}.")

            # Statistics gradient with BB
            for (exp_name, O), E_nu, dEdx in zip(
                self.observation.items(), expectation.values(), dE.values()
            ):
                mc_var_nu = mc_variance.get(exp_name, np.zeros_like(E_nu))

                # Combine neutrino + muon
                E_total, mc_var_total, has_muons = self._get_total_model_and_variance(
                    exp_name, E_nu, mc_var_nu
                )

                # Compute beta for total model
                beta, tau = self._compute_beta(E_total, mc_var_total, O)

                beta_E = np.maximum(beta * E_total, 1e-9)

                # First-order gradient: 2 * sum[(1 - O/(beta*E)) * beta * dE/d(syst)]
                # Note: dEdx is for neutrino only, muon derivative is 0 for neutrino systematics
                nabla_X2[i] += 2 * np.sum((1 - O / beta_E) * beta * dEdx)

        return nabla_X2

    def nuisance_penalty(self, nuisance: List[float]) -> float:
        """
        Compute penalty term for nuisance parameters.

        Args:
            nuisance (list): Current nuisance parameter values

        Returns:
            Penalty chi-squared contribution
        """
        X2 = 0.0
        for mu, sig, dist, nuis in zip(
            self.nominal_nuisance, self.sigma_nuisance, self.dist_nuisance, nuisance
        ):
            if "normal" in dist:
                X2 += log_gaussian_ratio(nuis, mu, sig)
            elif dist == "beta":
                X2 += log_beta_ratio(nuis, mu, sig)
        return X2

    def analytic_priors_bounds(self, expectation, diff_expectation, mc_variance=None):
        """
        Compute first-order analytic estimates for nuisance parameters.

        Args:
            expectation (dict): Binned expected events
            diff_expectation (dict): Derivatives of expectation
            mc_variance (dict): MC variance per bin

        Returns:
            Tuple of (priors, bounds)
        """
        if mc_variance is None:
            mc_variance = self.mc_variance

        A = np.zeros(self.number_of_nuisance)
        B = np.zeros(self.number_of_nuisance)
        mu = np.array(self.nominal_nuisance)
        sig = np.array(self.sigma_nuisance)

        for i, dE in enumerate(diff_expectation.values()):
            for (exp_name, O), E_nu, dEdx in zip(
                self.observation.items(), expectation.values(), dE.values()
            ):
                # Get total model with muons
                mc_var_nu = mc_variance.get(exp_name, np.zeros_like(E_nu)) if mc_variance else np.zeros_like(E_nu)
                E_total, mc_var_total, _ = self._get_total_model_and_variance(exp_name, E_nu, mc_var_nu)

                # Effective variance includes MC uncertainty
                variance = O + mc_var_total
                safe_variance = np.where(variance > 0, variance, 1)

                A[i] += np.sum((O / E_total - 1) * dEdx)
                B[i] += np.sum(dEdx**2 / safe_variance)

        priors = mu + A / (B + 1 / sig**2)

        delta = np.minimum(2 * np.abs(priors - mu), sig)
        delta[delta == 0] = sig[delta == 0]

        bounds = np.c_[priors - 3 * delta, priors + 3 * delta]
        bounds = tuple(map(tuple, bounds))

        self.diff_expectation_nominal = diff_expectation
        self.expectation_nominal = expectation

        priors = 0.5 * (priors + mu)

        return priors, bounds

    def approximate_fisher(self, expectation, diff_expectation, mc_variance=None):
        """
        Approximate Fisher information including MC variance.

        Args:
            expectation (dict): Binned expected events
            diff_expectation (dict): Derivatives of expectation
            mc_variance (dict): MC variance per bin

        Returns:
            Fisher information array
        """
        if mc_variance is None:
            mc_variance = self.mc_variance

        I_stats = np.zeros(self.number_of_nuisance)
        for i, dE in enumerate(diff_expectation.values()):
            for (exp_name, O), E_nu, dEdx in zip(
                self.observation.items(), expectation.values(), dE.values()
            ):
                # Get total model with muons
                mc_var_nu = mc_variance.get(exp_name, np.zeros_like(E_nu)) if mc_variance else np.zeros_like(E_nu)
                E_total, mc_var_total, _ = self._get_total_model_and_variance(exp_name, E_nu, mc_var_nu)

                # Effective variance includes MC uncertainty
                variance = O + mc_var_total
                safe_variance = np.where(variance > 0, variance, 1)

                I_stats[i] += np.sum(dEdx**2 / safe_variance)

        I_prior = np.zeros(self.number_of_nuisance)
        for i, sig in enumerate(self.sigma_nuisance):
            I_prior[i] += 1 / sig**2

        return I_stats + I_prior
