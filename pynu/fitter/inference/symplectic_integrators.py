def leapfrog(q0, p0, dK, dU, eps):
    """Same as Störmer-Verlet method as follows:
            $p_{n+1/2} = p_{n} - 0.5 \epsilon \cdot \nabla_{q} U(q_{n})$
            $q_{n+1} = q_{n} + \epsilon \cdot \nabla_{q} K(p_{n+1/2})$
            $p_{n+1} = p_{n+1/2} - 0.5 \epsilon \cdot \nabla_{q} U(q_{n+1})$

    Args:
        q0 (numpy.array): initial values of parameters (positions)
        p0 (numpy.array): initial values of parameters conjugates (momenta)
        dK (method): function computing the gradient of the kinetic energy w.r.t. momenta
        dU (method): function computing the gradient of the potential energy (negative log likelihood) w.r.t. parameters
        eps (float): step size of the integrator

    Returns:
        Numpy array with the next values of positions and momenta
    """
    p_05 = p0 - 0.5 * eps * dU(q0)
    q = q0 + eps * dK(p_05)
    p = p_05 - 0.5 * eps * dU(q)
    return q, p


def tempered_leapfrog(q0, p0, dK, dU, eps, T):
    """Same as leapfrog method but with an additional temperature term:
            $p_{n+1/2} = p_{n} - 0.5T \epsilon \cdot \nabla_{q} U(q_{n})$
            $q_{n+1} = q_{n} + \epsilon \cdot \nabla_{q} K(p_{n+1/2})$
            $p_{n+1} = p_{n+1/2} - 0.5/T \epsilon \cdot \nabla_{q} U(q_{n+1})$

    Args:
        q0 (numpy.array): initial values of parameters (positions)
        p0 (numpy.array): initial values of parameters conjugates (momenta)
        dK (method): function computing the gradient of the kinetic energy w.r.t. momenta
        dU (method): function computing the gradient of the potential energy (negative log likelihood) w.r.t. parameters
        eps (float): step size of the integrator
        T (float): temperature parameter

    Returns:
        Numpy array with the next values of positions and momenta
    """
    p_05 = p0 - 0.5 * T * eps * dU(q0)
    q = q0 + eps * dK(p_05)
    p = p_05 - 0.5 / T * eps * dU(q)
    return q, p


# def third_order(q0, p0, dK, dU, eps):
# 	""" Third order symplectic integrator:
# 		$p_{n+1/2} = p_{n} - 0.5 \epsilon \cdot \nabla_{q} U(q_{n})$
# 		$q_{n+1} = q_{n} + \epsilon \cdot \nabla_{q} K(p_{n+1/2})$
# 		$p_{n+1} = p_{n+1/2} - 0.5 \epsilon \cdot \nabla_{q} U(q_{n+1})$

#         Args:
#             q0 (numpy.array): initial values of parameters (positions)
#             p0 (numpy.array): initial values of parameters conjugates (momenta)
#             dK (method): function computing the gradient of the kinetic energy w.r.t. momenta
#             dU (method): function computing the gradient of the potential energy (negative log likelihood) w.r.t. parameters
#             eps (float): step size of the integrator

#         Returns:
#             Numpy array with the next values of positions and momenta
#     """
# 	p_05 = p0 - 0.5 * eps * dU(q0)
# 	q = q0 + eps * dK(p0)
# 	p = p_05 - 0.5 * eps * dU(q)
# 	return q, p
