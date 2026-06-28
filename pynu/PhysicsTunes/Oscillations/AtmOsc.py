import numpy as np
from itertools import repeat
from .Oscillations import Oscillator
import pychic_earth as pe


####################
# Atmospheric flux #
####################

class AtmosphericOscillations(Oscillator):
    def __init__(self, scenario, neutrino_flavors, experiment):
        super().__init__(scenario, neutrino_flavors, source='Atmospheric')

        self.E_nodes = 200
        self.Z_nodes = 40
        self.energy_nodes = nsq.logspace(
            experiment.Etrue_min,
            experiment.Etrue_max,
            self.E_nodes)
        self.cth_nodes = nsq.linspace(
            experiment.Z_edges[0],
            experiment.Z_edges[1],
            self.Z_nodes)

        self.CosZTrue = experiment.CosZTrue
        self.ETrue = experiment.ETrue

        self.SetUpOscillator()

        self.NSQneutype = self.NSQNeutrinoType(experiment)
        self.NSQneuflavor = self.NSQNeutrinoFlavor(experiment)

        self.InitialFlux = experiment.SetInitialFlux(
            self.energy_nodes, self.cth_nodes, neutrino_flavors)

    def GetOscillations(self): # oscillations
        self.Osc.Set_initial_state(self.InitialFlux, nsq.Basis.flavor)
        self.Osc.EvolveState()
        w = list(map(self.Osc.EvalFlavor,
                     self.NSQneuflavor,
                     self.CosZTrue,
                     self.ETrue * self.units.GeV,
                     self.NSQneutype,
                     repeat(True)))
        return np.asarray(w)

    def diff_oscillations(self): # oscillations
        self.Osc.Set_initial_state(self.InitialFlux, nsq.Basis.flavor)
        self.Osc.EvolveState()
        w = list(map(self.Osc.EvalFlavor,
                     self.NSQneuflavor,
                     self.CosZTrue,
                     self.ETrue * self.units.GeV,
                     self.NSQneutype,
                     repeat(True)))
        return np.asarray(w)