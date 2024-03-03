import numpy as np
import sys


class MCMC:
	"""docstring for MCMC"""
	def __init__(self, num_samples, method_parameters, neg_log_likelihood):

		self.num_samples = num_samples
		self.num_steps = method_parameters["num_steps"]

		self.neg_log_likelihood = neg_log_likelihood

		pass



class HMC(MCMC):
	"""docstring for HMC"""
	def __init__(self, num_samples, method_parameters, neg_log_likelihood, grad_neg_log_likelihood):
		super(HMC, self).__init__(neg_log_likelihood, **kwargs)

		self.lf_epsilon = method_parameters["lf_epsilon"]

		self.grad_neg_log_likelihood = grad_neg_log_likelihood


	def leapfrog_integration(self, current_q, current_p):
	    # Perform one leapfrog integration step
	    p_half = current_p - self.lf_epsilon * self.grad_neg_log_likelihood(current_q) / 2.0
	    q_new = current_q + self.lf_epsilon * p_half
	    p_new = p_half - self.lf_epsilon * self.grad_neg_log_likelihood(q_new) / 2.0
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
	    dim = len(grad_neg_log_likelihood(np.zeros(3)))  # Dimensionality of the grad_neg_log_likelihood
	    for _ in range(self.num_samples):
	        samples = []
	        current_q = np.random.randn(dim)  # Random initialization of position
	        current_p = np.random.randn(dim)  # Random initialization of momentum
	        for _ in range(self.num_steps):
	            samples.append(current_q.copy())
	            current_q, current_p = self.leapfrog_integration(current_q, current_p)
	        all_samples.append(samples)
	    return np.array(all_samples)


class tHMC(HMC):
	"""docstring for tempered HMC"""
	def __init__(self, num_samples, method_parameters, neg_log_likelihood, grad_neg_log_likelihood):
		super(tHMC, self).__init__(num_samples, method_parameters, neg_log_likelihood, grad_neg_log_likelihood)

		if method_parameters["t_alpha"] > 1 :
			self.t_alpha = method_parameters["t_alpha"]
		elif method_parameters["t_alpha"] < 1 :
			self.t_alpha = 1 / method_parameters["t_alpha"]
		elif method_parameters["t_alpha"] == 1:
			print("You are using to default Hamiltonian MC.")
		else:
			sys.exit("Not a valid temperature value.")

	def leapfrog_integration(self, current_q, current_p):
	    # Perform one leapfrog integration step
	    p_half = current_p - (self.lf_epsilon * self.grad_neg_log_likelihood(current_q) / 2.0) * self.t_alpha
	    q_new = current_q + self.lf_epsilon * p_half
	    p_new = p_half - (self.lf_epsilon * self.grad_neg_log_likelihood(q_new) / 2.0) / self.t_alpha
	    return q_new, p_new