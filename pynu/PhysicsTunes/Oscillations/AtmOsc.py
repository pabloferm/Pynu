import numpy as np
from math import asin, sqrt, pi
from itertools import repeat
from .Oscillations import Oscillator
import nuSQuIDS as nsq


####################
# Atmospheric flux #
####################

class AtmosphericOscillations(Oscillator):
	def __init__(self, scenario, neutrino_flavors, experiment):
		super().__init__(scenario, neutrino_flavors, source='Atmospheric')

		self.Z_nodes = 40
		self.energy_nodes = nsq.logspace(experiment.Etrue_min,experiment.Etrue_max,self.E_nodes)
		self.cth_nodes = nsq.linspace(experiment.Z_edges[0],experiment.Z_edges[1],self.Z_nodes)

		self.CosZTrue = experiment.CosZTrue
		self.ETrue = experiment.ETrue

		self.SetUpOscillator()

		self.NSQneutype = self.NSQNeutrinoType(experiment)
		self.NSQneuflavor = self.NSQNeutrinoFlavor(experiment)

		self.InitialFlux = experiment.SetInitialFlux(self.energy_nodes, self.cth_nodes, neutrino_flavors)


	def GetOscillations(self):
		# print(self.Parameters)
		self.Osc.Set_initial_state(self.InitialFlux,nsq.Basis.flavor)
		self.Osc.EvolveState()
		# print(self.Osc.Get_MixingAngle(1,2))
		w = list(map(self.Osc.EvalFlavor, self.NSQneuflavor, self.CosZTrue, self.ETrue*self.units.GeV, self.NSQneutype, repeat(True)))
		# print(w)
		return np.array(w)
