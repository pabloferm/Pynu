# Provides several functions for probability distributions
from math import gamma, exp, sqrt, pi, log
import numpy as np

# Normal distribution


def GaussianDistribution(x, m, s):
    return np.exp(-0.5 * (x - m)**2 / s**2) / (s * np.sqrt(2 * pi))


def logGaussianPrior(x, m, s):  # Actually, -2 ln(L/L0)
    return (x - m)**2 / s**2


def DifflogGaussianPrior(x, m, s):  # Actually, -2 ln(L/L0)
    return 2 * (x - m) / s**2


# Beta distribution

def BetaDistribution(x, a, b):
    return x**(a - 1) * (1 - x)**(b - 1) * gamma(a + b) / gamma(a) / gamma(b)


def BetaPar(m, s):  # Mode and std
    alpha = ((1 - m) / s**2 - 1 / m) * m**2
    beta = alpha * (1 / m - 1)
    return alpha, beta


def BetaMean(a, b):
    return a / (a + b)


def BetaMode(a, b):
    return (a - 1) / (a + b - 2)


def BetaSTD(a, b):
    return np.sqrt((a * b) / ((a + b)**2 * (a + b + 1)))


def logBetaPrior(x, a, b):
    m = BetaMode(a, b)
    return 2 * ((a - 1) * np.log(m / x) + (b - 1) * np.log((1 - m) / (1 - x)))


def DifflogBetaPrior(x, m, s):
    a, b = BetaPar(m, s)
    return 2 * (- (a - 1) / x + (b - 1) / (1 - x))
