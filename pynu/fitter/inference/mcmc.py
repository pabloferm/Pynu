import numpy as np
from numpy.random import Generator, PCG64
from .symplectic_integrators import *
import sys
import multiprocessing


class MCMC:
    """Implementation of Metropolis-Hastings MCMC"""

    def __init__(
        self,
        neg_log_likelihood,
        initial_values,
        sigma=0.1,
        chain_steps=100,
        samples=100,
    ):
        self.samples = samples
        self.dim = len(initial_values)
        self.initial_q = initial_values

        if sigma:
            self.sigma = sigma
        else:
            self.sigma = np.ones((self.dim))

        self.neg_log_likelihood = neg_log_likelihood

        self.rng = Generator(PCG64())

    def metropolis_hastings(self):
        samples = []
        current_state = self.initial_q

        for _ in range(self.num_samples):
            proposed_state = _proposal(current_state, self.sigma)
            print(f"Proposed state is {proposed_state}")

            acceptance_ratio = np.exp(
                0.5
                * (
                    self.neg_log_likelihood(current_state)
                    - self.neg_log_likelihood(proposed_state)
                )
            )
            print(f"current - 2 ln(L) is {self.neg_log_likelihood(current_state)}")
            print(f"proposed - 2 ln(L) is {self.neg_log_likelihood(proposed_state)}")
            print(f"Acceptance ratio is {acceptance_ratio}")

            if np.random.rand() < acceptance_ratio:
                current_state = proposed_state

            samples.append(current_state)
            print(current_state)

        return np.reshape(np.array(samples), (-1, self.dim))


class HMC(MCMC):
    """Implementation of Hamiltonian MC, assumes metric matrix is diagonal"""

    def __init__(
        self,
        neg_log_likelihood,
        grad_neg_log_likelihood,
        initial_values,
        range_of_initial_values=False,
        num_steps=20,
        samples=1,
        epsilon=1e-1,
        riemann_mass=1,
        random_steps=False,
        trajectory=True,  # returns the trajectory, tbd
        multiprocessing=True
    ):
        super(HMC, self).__init__(neg_log_likelihood, initial_values, samples=samples)

        self.epsilon = epsilon

        self.random_steps = random_steps
        self.MAX_STEPS = num_steps

        self.grad_neg_log_likelihood = grad_neg_log_likelihood

        self.riemann_mass = riemann_mass

        self.multiprocessing = multiprocessing

        self.range_of_initial_values = range_of_initial_values

        self.initial_position = self.initial_q

        self.save_trajectory = trajectory
        # if self.save_trajectory:
        #     self.trajectory = {
        #         "q": np.zeros((self.num_steps + 1, self.dim)),
        #         "p": np.zeros((self.num_steps + 1, self.dim)),
        #         "H": np.zeros((self.num_steps + 1, self.dim)),
        #     }

        self.check_parameters()

    def integration(self, current_q, current_p, method="leapfrog"):
        if method == "leapfrog":
            return leapfrog(
                current_q,
                current_p,
                self.grad_kinetic_energy,
                self.grad_neg_log_likelihood,
                self.epsilon,
            )
        else:
            sys.exit(f"{method} integration is not implemented yet, please do!")

    def set_parameters(self):
        pass

    def check_parameters(self):
        q = self.initial_q
        p = self.riemann_mass * self.rng.normal(size=self.dim)
        dK = max(np.abs(self.grad_kinetic_energy(p)))
        dU = max(np.abs(self.grad_neg_log_likelihood(q)))

        if self.epsilon < min(dK, dU) / 10:
            print("Seems like a good value for the step size of the integrator.")
        elif self.epsilon < min(dK, dU):
            print(
                "The step size of the integrator might be a bit too coarse for this analysis."
            )
        else:
            print(
                "The step size of the integrator is larger than either the kinetic term, the potential, or both. Please, make it smaller."
            )

        if self.MAX_STEPS * self.epsilon < 0.75:
            print(
                "Integration time may be too short, please increase the number of steps for the integrator."
            )
        elif self.MAX_STEPS * self.epsilon < 10:
            print("Integration time looks appropriate.")
        else:
            print(
                "Integration time is too long and you may consider shorten it to improve speed."
            )

    def kinetic_energy(self, p):
        # Kinetic energy: 0.5 * p^T * p
        # Include mass term
        return 0.5 * np.dot(p, p / self.riemann_mass) + 0.5 * np.log(
            np.prod(self.riemann_mass)
        )

    def grad_kinetic_energy(self, p):
        # Kinetic energy: 0.5 * p^T * p
        # Include mass term
        return p / self.riemann_mass

    def hamiltonian(self, q, p):
        # Hamiltonian: potential energy + kinetic energy
        return self.neg_log_likelihood(q) + self.kinetic_energy(p)

    def compute_trajectory(self, samples=None):
        if samples is None:
            samples = self.samples
        momenta = np.sqrt(self.riemann_mass) * self.rng.normal(size=(samples, self.dim))

        if self.random_steps=="linear" or self.random_steps is True:
            steps = self.rng.integers(int(self.MAX_STEPS/3), high=self.MAX_STEPS, size=samples)
        elif self.random_steps=="exponential":
            values = np.linspace(1, self.MAX_STEPS, dtype=int)
            p = 1 - np.exp(values/self.MAX_STEPS)
            steps = self.rng.choice(values, size=samples, p=p/np.sum(p))
        else:
            steps = [self.MAX_STEPS] * samples

        if np.any(self.range_of_initial_values):
            d_positions = self.range_of_initial_values * self.rng.normal(size=(samples, self.dim))
        else:
            d_positions = self.zeros((samples, self.dim))

        if self.multiprocessing:
            cores = multiprocessing.cpu_count()
            processes = []
            for i, (p, dq) in enumerate(zip(momenta, d_positions)):
                self.initial_p = p
                self.initial_q = self.initial_position + dq
                self.num_steps = steps[i]
                print(f"inital mometa, {self.initial_p}")
                print(f"inital positions, {self.initial_q}")
                print(
                    f"Processing chain {i} of {samples} HMC trajectories."
                )
                if (i + 1) % cores == 0:
                    for proc in processes:
                        proc.join()
                    processes = []
                proc = multiprocessing.Process(target=self.compute_single_trajectory,)
                proc.start()
                processes.append(proc)
            
        else:
            for i, (p, dq) in enumerate(zip(momenta, d_positions)):
                self.initial_p = p
                self.initial_q = self.initial_position + dq
                self.num_steps = steps[i]
                print(f"inital mometa, {self.initial_p}")
                print(f"inital positions, {self.initial_q}")
                self.compute_single_trajectory()

    def compute_single_trajectory(self):
        current_q = self.initial_q
        current_p = self.initial_p
        initial_energy = self.hamiltonian(current_q, current_p)
        print(f"initial energy: {initial_energy}")

        for k in range(self.num_steps):  # integrator
            if k % 10 == 0:
                print(f"Step {k} of {self.num_steps} at the leapfrog integrator")
            current_q, current_p = self.integration(
                current_q, current_p, method="leapfrog"
            )
            with open("positions_tottraj.txt", "a") as f:
                np.savetxt(f, current_q, fmt="%1.6f", newline=" ", delimiter=",")
                f.write("\n")
            with open("momenta_tottraj.txt", "a") as f:
                np.savetxt(f, current_p, fmt="%1.6f", newline=" ", delimiter=",")
                f.write("\n")
            print(f"positions: {current_q}")
            print(f"momenta: {current_p}")
        current_p = -current_p
        proposed_energy = self.hamiltonian(current_q, current_p)
        print(f"proposed energy: {proposed_energy}")
        mh = min(1, np.exp((initial_energy - proposed_energy)))
        print(f"Metropolis-Hastings correction: {mh}")

        if np.random.uniform() < mh:
            final_q = current_q
            final_p = current_p
            save_ini = 0
            print(f"Proposal accepted, saving current state: {current_q}")
        else:
            final_q = self.initial_q
            final_p = self.initial_p
            save_ini = 1
            print(f"Proposal rejected, saving original: {self.initial_q}")

        with open("positions_endtraj.txt", "a") as f:
            np.savetxt(f, current_q, fmt="%1.6f", newline=" ", delimiter=",")
            f.write("\n")
        with open("momenta_endtraj.txt", "a") as f:
            np.savetxt(f, current_p, fmt="%1.6f", newline=" ", delimiter=",")
            f.write("\n")
        with open("accept.txt", "a") as f:
            f.write(str(save_ini))
            f.write("\n")


class tHMC(HMC):
    """Implementation of tempered HMC, still pending on finishing HMC super class"""

    def __init__(
        self,
        num_samples,
        method_parameters,
        initial_values,
        initial_ranges,
        neg_log_likelihood,
        grad_neg_log_likelihood,
    ):
        super(tHMC, self).__init__(
            num_samples,
            method_parameters,
            initial_values,
            initial_ranges,
            neg_log_likelihood,
            grad_neg_log_likelihood,
        )

        if method_parameters["t_alpha"] > 1:
            self.t_alpha = 1 / method_parameters["t_alpha"]
        elif method_parameters["t_alpha"] < 1:
            self.t_alpha = method_parameters["t_alpha"]
        elif method_parameters["t_alpha"] == 1:
            print("You are using to default Hamiltonian MC.")
        else:
            sys.exit("Not a valid temperature value.")

    def integration(self, current_q, current_p, method):
        if method == "leapfrog":
            self.t_alpha = 1 / self.t_alpha
            return tempered_leapfrog(
                current_q,
                current_q,
                grad_kinetic_energy,
                grad_neg_log_likelihood,
                self.epsilon,
                self.t_alpha,
            )
        else:
            sys.exit(f"{method} integration is not implemented yet, please do!")
