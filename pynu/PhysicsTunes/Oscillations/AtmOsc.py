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

        self.E_nodes = 300
        self.Z_nodes = 40

        self.energy_nodes = nsq.logspace(
                experiment.Etrue_min, experiment.Etrue_max, self.E_nodes
            )
        self.cth_nodes = nsq.linspace(
            experiment.Z_edges[0], experiment.Z_edges[1], self.Z_nodes
        )

        self.CosZTrue = experiment.CosZTrue
        self.ETrue = experiment.ETrue

        self.SetUpOscillator()

        self.NSQneutype = self.NSQNeutrinoType(experiment)
        self.NSQneuflavor = self.NSQNeutrinoFlavor(experiment)

        self.InitialFlux = experiment.SetInitialFlux(
            self.energy_nodes, self.cth_nodes, neutrino_flavors
        )

    def GetOscillations(self):
        print(
            f"At computation time oscillation parameters are, \ns12:{self.Osc.Get_MixingAngle(0, 1)}\ns13:{self.Osc.Get_MixingAngle(0, 2)}\ns23:{self.Osc.Get_MixingAngle(1, 2)}\ndm221:{self.Osc.Get_SquareMassDifference(1)}\ndm231:{self.Osc.Get_SquareMassDifference(2)}"
        )
        self.Osc.Set_initial_state(self.InitialFlux, nsq.Basis.flavor)
        self.Osc.EvolveState()
        w = list(
            map(
                self.Osc.EvalFlavor,
                self.NSQneuflavor,
                self.CosZTrue,
                self.ETrue * self.UNITS.GeV,
                self.NSQneutype,
                repeat(True),
            )
        )
        return np.asarray(w)
