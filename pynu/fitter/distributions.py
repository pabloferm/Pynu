# Provides several functions for probability distributions
from math import gamma, exp, sqrt, pi, log
import numpy as np


##############################################
############ Gaussian distribution ###########
##############################################

def gaussian(x, m, s):
    r"""Returns the values of Gaussian or Normal probability distribution,
    $\mathcal{N}(x;\mu,\sigma) = \frac{1}{\sigma\sqrt{2\pi}}
    e^{\frac{(x-\mu)^2}{2\sigma^2}}$

    Args:
        x (numpy array): Values for which you want to obtain the probability.
        m (float): Mean of the distribution.
        s (float): Standard deviation of the distribution.

    Returns:
        Numpy array with values of $\mathcal{N}(x;\mu,\sigma)$.
   """
    return np.exp(-0.5 * (x - m)**2 / s**2) / (s * np.sqrt(2 * pi))


def log_gaussian_ratio(x, m, s):
    r"""Returns the values of -2 times the logarithm of the likelihood ratio
    assuming `gaussian` probability distribution.

    Args:
        x (numpy array): Values for which you want to obtain logLLH ratio.
        m (float): Mean of the distribution.
        s (float): Standard deviation of the distribution.

    Returns:
        Numpy array with the values for $-2\cdot\ln{\Big(\frac{\mathcal{N}(x;\mu,\sigma)}
        {\mathcal{N}(\mu;\mu,\sigma)}\Big)}$.
   """
    return (x - m)**2 / s**2


def diff_log_gaussian_ratio(x, m, s):  # Actually, -2 ln(L/L0)
    r"""Returns the values of the derivative with respect to 'x´ of -2 times
    the logarithm of the likelihood ratio assuming `gaussian` probability distribution, 
    $ -2 \ln{\Big(\frac{\mathcal{N}(x;\mu,\sigma)}{\mathcal{N}(\mu;\mu,\sigma)}\Big)}$.

    Args:
        x (numpy array): Values for which you want to obtain the derivative of
        the logLLH ratio.
        m (float): Mean of the distribution.
        s (float): Standard deviation of the distribution.

    Returns:
        Numpy array with the values for $-2\frac{\mathrm{d}}{\mathrm{d}x}
        \Big(\ln{\Big(\frac{\mathcal{N}(x;\mu,\sigma)}{\mathcal{N}
        (\mu;\mu,\sigma)}\Big)}\Big)$
   """
    return 2 * (x - m) / s**2


##############################################
############## Beta distribution #############
##############################################

def beta(x, a, b):
    r"""Returns the values of Beta probability distribution,
    $B(x;\alpha,\beta) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}
    x^{\alpha-1} (1-x)^{\beta-1}$.

    Args:
        x (numpy array): Values for which you want to obtain the probability.
        a (float): $\alpha$ parameter.
        b (float): $\beta$ parameter.

    Returns:
        Numpy array with values of $B(x;\alpha,\beta)$.
    """
    return x**(a - 1) * (1 - x)**(b - 1) * gamma(a + b) / gamma(a) / gamma(b)


def beta_param(m, s):  # Mode and std
    r"""Returns the α and β parameters of the `beta` probability distribution from
    the mode and standrd deviation

    Args:
        m (float): Mode of the distribution.
        s (float): Standard deviation of the distribution.

    Returns:
        Two floats, the α and β values .
    """
    alpha = ((1 - m) / s**2 - 1 / m) * m**2
    beta = alpha * (1 / m - 1)
    return alpha, beta


def beta_mean(a, b):
    r"""Returns the mean of the `beta` probability distribution, provided the
    α and β parameters.

    $mean = \frac{\alpha}{\alpha+\beta}$

    Args:
        a (float): $\alpha$ parameter of the distribution.
        b (float): $\beta$ parameter of the distribution.

    Returns:
        Float, the mean of the distribution.
    """
    return a / (a + b)


def beta_mode(a, b):
    r"""Returns the mode of the `beta` probability distribution, provided the
    α and β parameters.

    $\mu = \frac{\alpha-1}{\alpha+\beta-2}$

    Args:
        a (float): $\alpha$ parameter of the distribution.
        b (float): $\beta$ parameter of the distribution.

    Returns:
        Float, the mode of the distribution.
    """
    return (a - 1) / (a + b - 2)


def beta_std(a, b):
    r"""Returns the standard deviation of the `beta` probability distribution,
    provided the α and β parameters.

    $\sigma = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$

    Args:
        a (float): $\alpha$ parameter of the distribution.
        b (float): $\beta$ parameter of the distribution.

    Returns:
        Float, the Standard deviation of the distribution.
    """
    return np.sqrt((a * b) / ((a + b)**2 * (a + b + 1)))


def log_beta_ratio(x, a, b):
    r"""Returns the values of -2 times the logarithm of the likelihood ratio
    assuming `beta` probability distribution.

    Args:
        x (numpy array): Values for which you want to obtain logLLH ratio.
        a (float): $\alpha$ parameter of the distribution.
        b (float): $\beta$ parameter of the distribution.

    Returns:
        Numpy array with the values for $-2\cdot\ln{\Big(\frac{B(x;\alpha,\beta)}
        {B(\mu;\alpha,\beta)}\Big)}$.
   """
    m = beta_mode(a, b)
    return 2 * ((a - 1) * np.log(m / x) + (b - 1) * np.log((1 - m) / (1 - x)))


def diff_log_beta_ratio(x, m, s):
    r"""Returns the values of of the derivative with respect to 'x´ of -2 times
    the logarithm of the likelihood ratio assuming `beta` probability distribution.

    Args:
        x (numpy array): Values for which you want to obtain the derivative of
        the logLLH ratio.
        m (float): Mode of the distribution.
        s (float): Standard deviation of the distribution.

    Returns:
        Numpy array with the values for $-2\frac{\mathrm{d}}{\mathrm{d}x}
        \Big(\ln{\Big(\frac{B(x;\alpha,\beta)}{B(\mu;\alpha,\beta)}\Big)}\Big)$
    """
    a, b = beta_param(m, s)
    return diff_log_beta_ratio_args(x, a, b)


def diff_log_beta_ratio_args(x, a, b):
    r"""Returns the values of of the derivative with respect to 'x´ of -2
    times the logarithm of the likelihood ratio assuming `beta` probability
    distribution.

    Args:
        x (numpy array): Values for which you want to obtain the derivative of
        the logLLH ratio.
        a (float): $\alpha$ parameter of the distribution.
        b (float): $\beta$ parameter of the distribution.

    Returns:
        Numpy array with the values for $-2\frac{\mathrm{d}}{\mathrm{d}x}
        \Big(\ln{\Big(\frac{B(x;\alpha,\beta)}{B(\mu;\alpha,\beta)}\Big)}\Big)$
   """
    return 2 * (- (a - 1) / x + (b - 1) / (1 - x))
