# Provides several functions for probability distributions
from math import gamma, exp, sqrt, pi, log
import numpy as np
import numpy.typing as npt
from typing import List, Tuple

vector = npt.NDArray[np.float64]

# Normal distribution


def GaussianDistribution(x: vector, m: float, s: float) -> vector:
    return np.exp(-0.5 * (x - m)**2 / s**2) / (s * np.sqrt(2 * pi))


def logGaussianPrior(x: float, m: float, s: float) -> float:  # Actually, -2 ln(L/L0)
    return (x - m)**2 / s**2


def DifflogGaussianPrior(x: float, m: float, s: float) -> float:  # Actually, -2 ln(L/L0)
    return 2 * (x - m) / s**2


# Beta distribution

def BetaDistribution(x: vector, a: float, b: float) -> vector:
    return x**(a - 1) * (1 - x)**(b - 1) * gamma(a + b) / gamma(a) / gamma(b)


def BetaPar(m: float, s: float) -> Tuple[float, float]:  # Mode and std
    alpha = ((1 - m) / s**2 - 1 / m) * m**2
    beta = alpha * (1 / m - 1)
    return alpha, beta


def BetaMean(a: float, b: float) -> float:
    return a / (a + b)


def BetaMode(a: float, b: float) -> float:
    return (a - 1) / (a + b - 2)


def BetaSTD(a: float, b: float) -> float:
    return np.sqrt((a * b) / ((a + b)**2 * (a + b + 1)))


def logBetaPrior(x: float, a: float, b: float) -> float:
    m = BetaMode(a, b)
    return 2 * ((a - 1) * np.log(m / x) + (b - 1) * np.log((1 - m) / (1 - x)))


def DifflogBetaPrior(x: float, m: float, s: float) -> float:
    a, b = BetaPar(m, s)
    return 2 * (- (a - 1) / x + (b - 1) / (1 - x))


def DifflogBetaPrior_wParameters(x: float, a: float, b: float) -> float:
    return 2 * (- (a - 1) / x + (b - 1) / (1 - x))
