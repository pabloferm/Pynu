from Fitter import distributions as dt
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use(os.environ["PYNU"] + "/../utils/plot.mplstyle")

mu = [0.2, 0.65, 0.97, 0.9]
sig = [0.08, 0.15, 0.025, 0.07]
alpha = [3, 5, 2, 7]
beta = [2, 4, 3, 7]

x = np.linspace(0.01, 0.99, 200)
G = np.zeros(200)
B = np.zeros(200)
logG = np.zeros(200)
logB = np.zeros(200)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
fig.tight_layout()
axis = axes.flat

# for i, (m, s) in enumerate(zip(mu, sig)):
for i, (a, b) in enumerate(zip(alpha, beta)):
    #    a, b = dt.BetaPar(m, s)
    #    for j in range(200):
    #        G[j] = dt.Gaussian(x[j], m, s)
    B = dt.Beta(x, a, b)
    #    axis[i].plot(x, G / np.sum(G), label='Gaussian')
    axis[i].plot(x, B / np.sum(B), label="Beta")
    axis[i].set_ylim(
        0,
    )
    s = dt.BetaSTD(a, b)
    mode = dt.BetaMode(a, b)
    mean = dt.BetaMean(a, b)
    axis[i].axvline(x=mean + s, label="mean + sigma", color="pink")
    axis[i].axvline(x=mean - s, label="mean - sigma", color="green")
    axis[i].axvline(x=mode, label="mode", color="red")
    axis[i].axvline(x=mean, label="mean", color="violet")
    #    axis[i].set_title(r'$\mu$=' + str(m) + r', $\sigma$=' + str(s))
    axis[i].set_xlabel("Efficiency")
    axis[i].legend()

# plt.show()

fig2, axes2 = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
fig2.tight_layout()
axis2 = axes2.flat

# for i, (m, s) in enumerate(zip(mu, sig)):
for i, (a, b) in enumerate(zip(alpha, beta)):
    #    a, b = dt.BetaPar(m, s)
    logB = dt.logBetaPrior(x, a, b)
    axis2[i].plot(x, logB, label="log Beta")
    axis2[i].set_ylim(0, 2)
    axis2[i].set_xlim(0.0, 1.0)
    axis2[i].set_xlabel("Efficiency")
    s = dt.BetaSTD(a, b)
    mode = dt.BetaMode(a, b)
    mean = dt.BetaMean(a, b)
    axis2[i].axvline(x=mean + s, label="mean + sigma", color="pink")
    axis2[i].axvline(x=mean - s, label="mean - sigma", color="green")
    axis2[i].axvline(x=mode, label="mode", color="red")
    axis2[i].axvline(x=mean, label="mean", color="violet")
    axis2[i].axhline(y=1, color="grey")
    axis2[i].legend()

plt.show()
