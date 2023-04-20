# Provides several functions for probability distributions
from math import gamma, exp, sqrt, pi, log
import numpy as np
import numpy.typing as npt
from typing import Tuple
vector = npt.NDArray[np.float64]

##############################################


def GaussianDistribution(x: vector, m: float, s: float) -> vector:
    r"""Returns the values of Gaussian or Normal probability distribution, 
    $\frac{1}{\sigma\sqrt(2\pi)}\exp{\frac{(x-\mu)^2}{2\sigma^2}}$

    Args:
        x (vector): Values for which you want to obtain the probability.
        m (float): Mean of the distribution.
        s (float): Standard deviation of the distribution.

    Returns:
        vector: Probabilities

   """
    return np.exp(-0.5 * (x - m)**2 / s**2) / (s * np.sqrt(2 * pi))


def logGaussianPrior(x: float, m: float, s: float) -> float:
    """Returns the values of -2 times the logarithm of the likelihood ratio assuming Gaussian or Normal probability distribution.

    Args:
        x (vector): Values for which you want to obtain -2·ln(L/L0).
        m (float): Mean of the distribution.
        s (float): Standard deviation of the distribution.

    Returns:
        vector: -2·ln(L/L0)

   """
    return (x - m)**2 / s**2


def DifflogGaussianPrior(x: float, m: float, s: float) -> float:  # Actually, -2 ln(L/L0)
    """Returns the values of the derivative with respect to 'x´ of -2 times the logarithm of the likelihood ratio assuming Gaussian or Normal probability distribution.

    Args:
        x (vector): Values for which you want to obtain d(-2·ln(L/L0))/dx.
        m (float): Mean of the distribution.
        s (float): Standard deviation of the distribution.

    Returns:
        vector: d(-2·ln(L/L0))/dx

   """
    return 2 * (x - m) / s**2


##############################################

def BetaDistribution(x: vector, a: float, b: float) -> vector:
    """Returns the values of Beta probability distribution

    Args:
        x (vector): Values for which you want to obtain the probability.
        m (float): Mean of the distribution.
        s (float): Standard deviation of the distribution.

    Returns:
        vector: Probabilities

    """
    return x**(a - 1) * (1 - x)**(b - 1) * gamma(a + b) / gamma(a) / gamma(b)


def BetaPar(m: float, s: float) -> Tuple[float, float]:  # Mode and std
    """Returns the α and β parameters of the Beta probability distribution from the mode and standrd deviation

    Args:
        m (float): Mode of the distribution.
        s (float): Standard deviation of the distribution.

    Returns:
        Tuple: α and β

    """
    alpha = ((1 - m) / s**2 - 1 / m) * m**2
    beta = alpha * (1 / m - 1)
    return alpha, beta


def BetaMean(a: float, b: float) -> float:
    """Returns the mean provided the α and β parameters of the Beta probability distribution

    Args:
        a (float): Alpha parameter of the distribution.
        b (float): Beta parameter of the distribution.

    Returns:
        float: Mean of the distribution

    """
    return a / (a + b)


def BetaMode(a: float, b: float) -> float:
    """Returns the mode provided the α and β parameters of the Beta probability distribution

    Args:
        a (float): Alpha parameter of the distribution.
        b (float): Beta parameter of the distribution.

    Returns:
        float: Mode of the distribution

    """
    return (a - 1) / (a + b - 2)


def BetaSTD(a: float, b: float) -> float:
    """Returns the standard deviation provided the α and β parameters of the Beta probability distribution

    Args:
        a (float): Alpha parameter of the distribution.
        b (float): Beta parameter of the distribution.

    Returns:
        float: Standard deviation of the distribution

    """
    return np.sqrt((a * b) / ((a + b)**2 * (a + b + 1)))


def logBetaPrior(x: vector, a: float, b: float) -> vector:
    """Returns the values of -2 times the logarithm of the likelihood ratio assuming Beta probability distribution.

    Args:
        x (vector): Values for which you want to obtain -2·ln(L/L0).
        a (float): Alpha parameter of the distribution.
        b (float): Beta parameter of the distribution.

    Returns:
        vector: -2·ln(L/L0)

   """
    m = BetaMode(a, b)
    return 2 * ((a - 1) * np.log(m / x) + (b - 1) * np.log((1 - m) / (1 - x)))


def DifflogBetaPrior(x: vector, m: float, s: float) -> vector:
    """Returns the values of of the derivative with respect to 'x´ of -2 times the logarithm of the likelihood ratio assuming Beta probability distribution.

    Args:
        x (vector): Values for which you want to obtain -2·ln(L/L0).
        m (float): Mean of the distribution.
        s (float): Standard deviation of the distribution.

    Returns:
        vector: d(-2·ln(L/L0))/dx

    """
    a, b = BetaPar(m, s)
    return DifflogBetaPrior_wArgs(x, a, b)


def DifflogBetaPrior_wArgs(x: vector, a: float, b: float) -> vector:
    """Returns the values of of the derivative with respect to 'x´ of -2 times the logarithm of the likelihood ratio assuming Beta probability distribution.

    Args:
        x (vector): Values for which you want to obtain -2·ln(L/L0).
        a (float): Alpha parameter of the distribution.
        b (float): Beta parameter of the distribution.

    Returns:
        vector: d(-2·ln(L/L0))/dx

   """
    return 2 * (- (a - 1) / x + (b - 1) / (1 - x))
