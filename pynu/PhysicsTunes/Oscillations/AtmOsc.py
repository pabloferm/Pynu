import numpy as np
from itertools import repeat
from .Oscillations import Oscillator
import nuSQuIDS as nsq


####################
# Atmospheric flux #
####################


class AtmosphericOscillations(Oscillator):
    def __init__(self, scenario, neutrino_flavors, experiment):
        super().__init__(scenario, neutrino_flavors, source="Atmospheric")

        self.E_nodes = 200
        self.Z_nodes = 40

        self.energy_nodes = np.geomspace(
                experiment.Etrue_min, experiment.Etrue_max, self.E_nodes
            )
        self.cth_nodes = np.linspace(
            experiment.Z_edges[0], experiment.Z_edges[1], self.Z_nodes
        )

        self.CosZTrue = experiment.CosZTrue
        self.ETrue = experiment.ETrue
        self.CC = experiment.CC

        self.SetUpOscillator()

        self.NSQneutype = self.NSQNeutrinoType(experiment)
        self.NSQneuflavor = self.NSQNeutrinoFlavor(experiment)

        self.InitialFlux = experiment.SetInitialFlux(
            self.energy_nodes, self.cth_nodes, neutrino_flavors
        )

    def GetOscillations(self):
        self.Osc.Set_initial_state(self.InitialFlux, nsq.Basis.flavor)
        self.Osc.EvolveState()
        w = np.ones(self.ETrue.size)
        
        # Use self.Osc.EvalFlavor as the method to map over events
        dw = list(
            map(
                self.Osc.EvalFlavor,
                self.NSQneuflavor,
                self.CosZTrue.astype(float).tolist(),
                (self.ETrue * self.UNITS.GeV).astype(float).tolist(),
                self.NSQneutype,
                repeat(True),
            )
        )
        dw = np.asarray(dw)
        
        # Apply oscillation weights only to CC events
        w = dw  # Apply oscillation weights to ALL events (CC and NC)

        return np.asarray(w)
