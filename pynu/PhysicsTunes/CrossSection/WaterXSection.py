import numpy as np

import sys
sys.path.append('../')
from PhysicsTunes import Tune

##########################
#  Water Cross-section   #
##########################

class WaterXSection(Tune):

	def XSecNuTau(self, experiment, x):
		tau = np.ones(experiment.NumberOfEvents)
		tau[np.abs(experiment.nuPDG)==16] = x
		return tau
	def Diff_XSecNuTau(x, experiment):
		tau = np.zeros(experiment.NumberOfEvents)
		tau[np.abs(experiment.nuPDG)==16] = 1
		return tau

	def NCoverCC(self, experiment, x):
		nc = np.ones(experiment.NumberOfEvents)
		nc[experiment.CC==0] = x 
		return nc
	def Diff_NCoverCC(x, experiment):
		nc = np.zeros(experiment.NumberOfEvents)
		nc[experiment.CC==0] = 1
		return nc

	def AxialMass(self, experiment, x):
		cc = np.ones(experiment.NumberOfEvents)
		cc[experiment.CC==1] = 1+0.042*(x-1)*1.05*np.log10(experiment.ETrue[experiment.CC==1]) 
		return cc
	def Diff_AxialMass(x, experiment):
		cc = np.zeros(experiment.NumberOfEvents)
		cc[experiment.CC==1] = 0.042*1.05*np.log10(experiment.ETrue[experiment.CC==1]) 
		return cc

	def NCHad(self, experiment, x):
		nc = np.ones(experiment.NumberOfEvents)
		nc[experiment.CC==0] = x 
		return nc
	def Diff_NCHad(x, experiment):
		nc = np.zeros(experiment.NumberOfEvents)
		nc[experiment.CC==0] = 1 
		return nc

	def DIS(self, experiment, x):
		dis = np.ones(experiment.NumberOfEvents)
		cond = np.abs(experiment.Mode)>25 * experiment.CC
		dis[cond] = x
		return dis
	def Diff_DIS(x,experiment):
		dis = np.zeros(experiment.NumberOfEvents)
		cond = np.abs(experiment.Mode)>25 * experiment.CC
		dis[cond] = 1
		return dis

	def CCQE(self, experiment, x):
		ccqe = np.ones(experiment.NumberOfEvents)
		ccqe[np.abs(experiment.Mode)==1] = x
		return ccqe
	def Diff_CCQE(x,experiment):
		ccqe = np.zeros(experiment.NumberOfEvents)
		ccqe[np.abs(experiment.Mode)==1] = 1
		return ccqe

	def CCQENuBarNu(self, experiment, x):
		ccqe = np.ones(experiment.NumberOfEvents)
		ccqe[experiment.Mode==-1] = x
		return ccqe
	def Diff_CCQENuBarNu(x,experiment):
		ccqe = np.zeros(experiment.NumberOfEvents)
		ccqe[experiment.Mode==-1] = 1
		return ccqe

	def CCQEMuE(self, experiment, x):
		ccqe = np.ones(experiment.NumberOfEvents)
		cond = (np.abs(experiment.Mode)==1) * (np.abs(experiment.nuPDG)==14)
		ccqe[cond] = x
		return ccqe
	def Diff_CCQEMuE(x,experiment):
		ccqe = np.zeros(experiment.NumberOfEvents)
		cond = (np.abs(experiment.Mode)==1) * (np.abs(experiment.nuPDG)==14)
		ccqe[cond] = 1
		return ccqe

	def CC1Pi_Pi0Pi(self, experiment, x):
		ccpi = np.ones(experiment.NumberOfEvents)
		ccpi[np.abs(experiment.Mode)==12] = x
		return ccpi
	def Diff_CC1Pi_Pi0Pi(x,experiment):
		ccpi = np.zeros(experiment.NumberOfEvents)
		ccpi[np.abs(experiment.Mode)==12] = 1
		return ccpi

	def CC1Pi_NuBarNuE(self, experiment, x):
		ccpi = np.ones(experiment.NumberOfEvents)
		cond = (np.abs(experiment.Mode)>10) * (np.abs(experiment.Mode)<17) * (experiment.nuPDG==-12)
		ccpi[cond] = x
		return ccpi
	def Diff_CC1Pi_NuBarNuE(x,experiment):
		ccpi = np.zeros(experiment.NumberOfEvents)
		cond = (np.abs(experiment.Mode)>10) * (np.abs(experiment.Mode)<17) * (experiment.nuPDG==-12)
		ccpi[cond] = 1
		return ccpi

	def CC1Pi_NuBarNuMu(self, experiment, x):
		ccpi = np.ones(experiment.NumberOfEvents)
		cond = (np.abs(experiment.Mode)>10) * (np.abs(experiment.Mode)<17) * (experiment.nuPDG==-14)
		ccpi[cond] = x
		return ccpi
	def Diff_CC1Pi_NuBarNuMu(x,experiment):
		ccpi = np.zeros(experiment.NumberOfEvents)
		cond = (np.abs(experiment.Mode)>10) * (np.abs(experiment.Mode)<17) * (experiment.nuPDG==-14)
		ccpi[cond] = 1
		return ccpi

	def CC1PiProduction(self, experiment, x):
		ccpi = np.ones(experiment.NumberOfEvents)
		cond = (np.abs(experiment.Mode)>10) * (np.abs(experiment.Mode)<17)
		ccpi[cond] = x
		return ccpi
	def Diff_CC1PiProduction(x,experiment):
		ccpi = np.zeros(experiment.NumberOfEvents)
		cond = (np.abs(experiment.Mode)>10) * (np.abs(experiment.Mode)<17)
		ccpi[cond] = 1
		return ccpi

	def CohPiProduction(self, experiment, x):
		ccpi = np.ones(experiment.NumberOfEvents)
		ccpi[np.abs(experiment.Mode)==16] = x
		return ccpi
	def Diff_CohPiProduction(x,experiment):
		ccpi = np.zeros(experiment.NumberOfEvents)
		ccpi[np.abs(experiment.Mode)==16] = 1
		return ccpi