import numpy as np
cimport numpy as np
from libc.math cimport exp

cdef double[:, :] metropolis_hastings(int num_samples, object neg_log_likelihood, double[:] initial_values, double[:] sigma):
    cdef double[:, :] samples = np.empty((num_samples, len(initial_values)), dtype=np.float64)
    cdef double[:] current_state = initial_values
    cdef double[:] proposed_state
    cdef double acceptance_ratio
    cdef int i

    for i in range(num_samples):
        proposed_state = _proposal(current_state, sigma)
        acceptance_ratio = exp(0.5 * (
            neg_log_likelihood(current_state) -
            neg_log_likelihood(proposed_state)))

        if np.random.rand() < acceptance_ratio:
            current_state = proposed_state

        samples[i, :] = current_state

    return samples

cdef double[:] _proposal(double[:] x, double[:] sigma):
    cdef int dim = len(x)
    cdef double[:] result = np.empty(dim, dtype=np.float64)

    for i in range(dim):
        result[i] = np.random.normal(x[i], sigma[i])

    return result

def run_metropolis_hastings(int num_samples, object neg_log_likelihood, double[:] initial_values, double[:] sigma):
    return metropolis_hastings(num_samples, neg_log_likelihood, initial_values, sigma)
