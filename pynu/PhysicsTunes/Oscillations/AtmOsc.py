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
		super().__init__(scenario, neutrino_flavors)

		self.Z_nodes = 40
		energy_nodes = nsq.logspace(experiment.Etrue_min,experiment.Etrue_max,self.E_nodes)
		cth_nodes = nsq.linspace(experiment.Z_edges[0],experiment.Z_edges[1],self.Z_nodes)

		self.Osc = nsq.nuSQUIDSAtm(cth_nodes,energy_nodes*self.units.GeV,self.NeutrinoFlavors,nsq.NeutrinoType.both,self.interactions)
		self.Osc.Set_rel_error(self.rel_error)
		self.Osc.Set_abs_error(self.abs_error)

		self.NSQneutype = self.NSQNeutrinoType(experiment)
		self.NSQneuflavor = self.NSQNeutrinoFlavor(experiment)


	def Oscillator(self):
		self.Osc.Set_initial_state(self.experiment.InitialFlux,nsq.Basis.flavor)
		self.Osc.EvolveState()
		w = list(map(self.Osc.EvalFlavor, self.NSQneuflavor, self.experiment.CosZTrue, self.experiment.ETrue*self.units.GeV, self.NSQneutype, repeat(True)))
		return np.array(w)
