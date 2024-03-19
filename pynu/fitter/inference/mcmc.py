import numpy as np
import sys


def _proposal(x, sigma):
    return np.random.normal(x, sigma)


class MCMC:
    """ Implementation of Metropolis-Hastings MCMC"""

    def __init__(
            self,
            neg_log_likelihood, initial_values, sigma=0.1, num_samples=100):

        self.num_samples = num_samples
        self.dim = len(initial_values)
        self.initial_values = initial_values
        self.initial_values = np.abs(np.random.randn(self.dim) + 1)

        if sigma:
            self.sigma = sigma
        else:
            self.sigma = np.ones((self.dim))

        self.neg_log_likelihood = neg_log_likelihood

    def metropolis_hastings(self):
        samples = []
        current_state = self.initial_values

        for _ in range(self.num_samples):
            proposed_state = _proposal(current_state, self.sigma)
            print(f"Proposed state is {proposed_state}")

            acceptance_ratio = np.exp(0.5 * (
                self.neg_log_likelihood(current_state) -
                self.neg_log_likelihood(proposed_state)))
            print(
                f"current - 2 ln(L) is {self.neg_log_likelihood(current_state)}")
            print(
                f"proposed - 2 ln(L) is {self.neg_log_likelihood(proposed_state)}")
            print(f"Acceptance ratio is {acceptance_ratio}")

            if np.random.rand() < acceptance_ratio:
                current_state = proposed_state

            samples.append(current_state)
            print(current_state)

        return np.reshape(np.array(samples), (-1, self.dim))


class HMC(MCMC):
    """Implementation of Hamiltonian MC"""

    def __init__(
        self,
        neg_log_likelihood,
        grad_neg_log_likelihood,
        initial_values,
        sigma=0.1,
        num_steps=10,
        num_samples=100,
        lf_epsilon=5e-3,
    ):
        super(
            HMC,
            self).__init__(
            neg_log_likelihood,
            initial_values,
            sigma=sigma,
            num_samples=num_samples)

        self.lf_epsilon = lf_epsilon

        self.num_steps = num_steps

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
        acceptance = 0
        for k in range(self.num_samples):
            print(f"Running sample {k} of {self.num_samples}")
            # Random initialization of position
            # current_q = self.initial_ranges * \
            print(f"Initial values, {self.initial_values}")
            current_q = np.abs(
                0.5 *
                np.random.randn(
                    self.dim) +
                self.initial_values)
            # Random initialization of momentum
            current_p = np.random.randn(self.dim)

            initial_energy = self.hamiltonian(
                current_q, current_p)
            initial_q = current_q
            print(f"initial state is {initial_q}")

            for k in range(self.num_steps):  # leapfrog integrator
                # if k % 10 == 0:
                #     print(f"Step {k} of {self.num_steps} at the leapfrog integrator")
                current_q, current_p = self.leapfrog_integration(
                    current_q, current_p)
                # print(f"current state is {current_q}")

            proposed_energy = self.hamiltonian(
                current_q, current_p)
            print(f"proposed state is {current_q}")

            if 1 < min(1, np.exp(0.5 * (initial_energy - proposed_energy))):
                all_samples.append([current_q])
                print(f"Proposal accepted, saving current state: {current_q}")
                acceptance += 1
            else:
                all_samples.append([initial_q])
                print(f"Proposal rejected, saving original: {initial_q}")
        print(
            f"Accepted {100*float(acceptance)/float(self.num_samples)}% of the proposals")
        return np.reshape(np.array(all_samples), (-1, self.dim))


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
        self.t_alpha = 1 / self.t_alpha
        return q_new, p_new

    def kinetic_energy(self, p):
        # Kinetic energy: 0.5 * p^T * p
        # Include mass term
        return 0.5 * np.dot(p, p)
