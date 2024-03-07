import numpy as np
import sys


class MCMC:
    """ Implementation of Metropolis-Hastings MCMC"""

    def __init__(
            self,
            num_samples,
            method_parameters,
            initial_values,
            initial_ranges,
            neg_log_likelihood):

        self.num_samples = num_samples
        self.num_steps = method_parameters["num_steps"]
        self.dim = len(initial_values)
        self.initial_values = initial_values
        self.initial_ranges = initial_ranges

        self.neg_log_likelihood = neg_log_likelihood

    def metropolis_hastings(self, target, proposal, sigma=1):
        samples = [self.initial_values]
        current_state = initial_state

        for _ in range(self.num_samples):
            proposed_state = proposal(current_state, sigma)

            acceptance_ratio = np.exp(
                self.neg_log_likelihood(current_state) -
                self.neg_log_likelihood(proposed_state))

            if np.random.rand() < acceptance_ratio:
                current_state = proposed_state

            samples.append(current_state)

        return np.array(samples)


class HMC(MCMC):
    """Implementation of Hamiltonian MC"""

    def __init__(
            self,
            num_samples,
            method_parameters,
            initial_values,
            initial_ranges,
            neg_log_likelihood,
            grad_neg_log_likelihood):
        super(
            HMC,
            self).__init__(
            num_samples,
            method_parameters,
            initial_values,
            initial_ranges,
            neg_log_likelihood)

        self.lf_epsilon = method_parameters["lf_epsilon"]

        self.grad_neg_log_likelihood = grad_neg_log_likelihood

    def leapfrog_integration(self, current_q, current_p):
        # Perform one leapfrog integration step
        p_half = current_p - self.lf_epsilon * \
            self.grad_neg_log_likelihood(current_q) / 2.0
        q_new = current_q + self.lf_epsilon * p_half
        p_new = p_half - self.lf_epsilon * \
            self.grad_neg_log_likelihood(q_new) / 2.0
        return q_new, p_new

    def kinetic_energy(self, p):
        # Kinetic energy: 0.5 * p^T * p
        # Include mass term
        return 0.5 * np.dot(p, p)

    def hamiltonian(self, q, p):
        # Hamiltonian: potential energy + kinetic energy
        return self.neg_log_likelihood(q) + self.kinetic_energy(p)

    def hamiltonian_monte_carlo(self):
        all_samples = []
        for _ in range(self.num_samples):
            samples = []
            # Random initialization of position
            current_q = self.initial_ranges * \
                np.random.randn(self.dim) + self.initial_values
            # Random initialization of momentum
            current_p = np.random.randn(self.dim)
            for _ in range(self.num_steps):
                samples.append(current_q.copy())
                current_q, current_p = self.leapfrog_integration(
                    current_q, current_p)
            proposed_energy = hamiltonian(
                current_q, current_p, energy_function)
            current_energy = hamiltonian(
                samples[0], np.random.randn(dim), energy_function)
            if np.random.rand() < np.exp(current_energy - proposed_energy):
                accept_count += 1
                all_samples.append(samples)
        return np.array(all_samples)


class tHMC(HMC):
    """Implementation of tempered HMC"""

    def __init__(
            self,
            num_samples,
            method_parameters,
            initial_values,
            initial_ranges,
            neg_log_likelihood,
            grad_neg_log_likelihood):
        super(
            tHMC,
            self).__init__(
            num_samples,
            method_parameters,
            initial_values,
            initial_ranges,
            neg_log_likelihood,
            grad_neg_log_likelihood)

        if method_parameters["t_alpha"] > 1:
            self.t_alpha = method_parameters["t_alpha"]
        elif method_parameters["t_alpha"] < 1:
            self.t_alpha = 1 / method_parameters["t_alpha"]
        elif method_parameters["t_alpha"] == 1:
            print("You are using to default Hamiltonian MC.")
        else:
            sys.exit("Not a valid temperature value.")

    def leapfrog_integration(self, current_q, current_p):
        # Perform one leapfrog integration step
        p_half = current_p - \
            (self.lf_epsilon * self.grad_neg_log_likelihood(current_q) / 2.0) * self.t_alpha
        q_new = current_q + self.lf_epsilon * p_half
        p_new = p_half - \
            (self.lf_epsilon * self.grad_neg_log_likelihood(q_new) / 2.0) / self.t_alpha
        return q_new, p_new
