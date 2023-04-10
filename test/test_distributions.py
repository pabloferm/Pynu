from Fitter import Distributions as dt
import matplotlib.pyplot as plt
import numpy as np
import os 

plt.style.use(os.environ['PYNU'] + '/utils/plot.mplstyle')

mu = [0.2, 0.65, 0.97, 0.9]
sig = [0.08, 0.15, 0.025, 0.07]

x = np.linspace(0, 1, 200)
G = np.zeros(200)
B = np.zeros(200)
logG = np.zeros(200)
logB = np.zeros(200)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
fig.tight_layout()
axis = axes.flat

for i, (m, s) in enumerate(zip(mu, sig)):
    a, b = dt.BetaPar(m, s)
    for j in range(200):
        G[j] = dt.Gaussian(x[j], m, s)
        B[j] = dt.Beta(x[j], a, b)
    axis[i].plot(x, G / np.sum(G), label='Gaussian')
    axis[i].plot(x, B / np.sum(B), label='Beta')
    axis[i].set_ylim(0,)
    axis[i].set_title(r'$\mu$=' + str(m) + r', $\sigma$=' + str(s))
    axis[i].set_xlabel('Efficiency')
    axis[i].legend()

# plt.show()

fig2, axes2 = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
fig2.tight_layout()
axis2 = axes2.flat

for i, (m, s) in enumerate(zip(mu, sig)):
    a, b = dt.BetaPar(m, s)
    for j in range(200):
        if x[j] > 0 and x[j] < 1:
            logG[j] = dt.logGaussianPrior(x[j], m, s)
            logB[j] = dt.logBetaPrior(x[j], m, s)
    axis2[i].plot(x, logG, label='log Gaussian')
    axis2[i].plot(x, logB, label='log Beta')
    axis2[i].set_ylim(0,2)
    axis2[i].set_xlim(0.01, 0.99)
    axis2[i].set_title(r'$\mu$=' + str(m) + r', $\sigma$=' + str(s))
    axis2[i].set_xlabel('Efficiency')
    axis2[i].legend()
    axis2[i].axvline(x=m+s)
    axis2[i].axvline(x=m-s)
    axis2[i].axvline(x=m)
    axis2[i].axvline(x=dt.BetaMean(a,b))
    axis2[i].axhline(y=1)

plt.show()
