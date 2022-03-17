import numpy as np
import pandas as pd
from plotting import cornerPlot, cornerPlotBothO
from matplotlib.colors import LogNorm
import sys
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt

def AnalysisReader(filename):	
	df = pd.read_csv(filename, sep=' ')
	rawOscPar = {}
	oscPar = {}
	for i in range(7):
		rawOscPar[i] = df.values[:,i]
		oscPar[i] = np.unique(rawOscPar[i])
	flux = np.array(df["NuNuBarRatio"])
	flux = np.array(df["ZenithFluxUp"])
	flux = np.array(df["ZenithFluxDown"])
	flux = np.array(df["XSecNuTau"])
	flux = np.array(df["FluxNormalization_Below1GeV"])
	flux = np.array(df["FluxTilt"])
	flux = np.array(df["FluxNormalization_Above1GeV"])
	flux = np.array(df["CCQENuBarNu"])
	flux = np.array(df["FlavorRatio"])
	penalty = ((flux-1)/0.05)**2
	theta_23 = np.array(df["Sin2Theta23"])
	dm_31 = np.array(df["Dm231"])
	# theta_23 = np.unique(theta_23)
	# dm_31 = np.unique(dm_31)
	X2 = np.array(df["X2"])
	# plt.plot(theta_23[dm_31==0.0025],flux[dm_31==0.0025])
	# plt.show()
	# plt.plot(theta_23[dm_31==0.0025],penalty[dm_31==0.0025])
	# plt.show()
	X, Y = np.meshgrid(theta_23, dm_31)
	norm = plt.Normalize(penalty.min(), 0.5*penalty.max())
	plt.hist2d(theta_23, dm_31, bins=15, weights=penalty, norm=norm, cmap='viridis')
	plt.colorbar()
	plt.show()


AnalysisReader(str(sys.argv[1]))



