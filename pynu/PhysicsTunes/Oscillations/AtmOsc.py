import numpy as np
from itertools import repeat
from .Oscillations import Oscillator
import nuSQuIDS as nsq


class AtmosphericOscillations(Oscillator):
    """Atmospheric neutrino oscillations: the flux evolved through the Earth by nuSQuIDS,
    evaluated at per-event (or per-cell) flavor, cos(zenith), energy and type."""

    def __init__(self, neutrino_flavors, experiment, avg_scale=None):
        """
        avg_scale: `scale` of nuSQuIDS' fast-oscillation averaging (None = averaging off).
        nuSQuIDS compares it to the eigenvalue phase dm2*L/(2E), so 4*pi averages the
        modes whose sin^2 argument 1.2667*dm2*L/E exceeds 2*pi. Read at evaluation time.
        """
        super().__init__(neutrino_flavors)

        self.E_nodes = 200
        self.Z_nodes = 40

        self.energy_nodes = np.geomspace(
                experiment.Etrue_min, experiment.Etrue_max, self.E_nodes
            )
        self.cth_nodes = np.linspace(
            experiment.Z_edges[0], experiment.Z_edges[1], self.Z_nodes
        )

        self.osc_avg_scale = avg_scale

        self.CosZTrue = experiment.CosZTrue
        self.ETrue = experiment.ETrue

        self.SetUpOscillator()

        self.NSQneutype = self.NSQNeutrinoType(experiment)
        self.NSQneuflavor = self.NSQNeutrinoFlavor(experiment)

        self.InitialFlux = experiment.SetInitialFlux(
            self.energy_nodes, self.cth_nodes, neutrino_flavors
        )

    def _eval_flavor_weights(self):
        """EvalFlavor over all events/cells for the current evolved state.

        Averaging off (osc_avg_scale None): EvalFlavor(flv, cz, E, type,
        randomize_height=True), a fresh production-height draw per call, so the
        result is stochastic. Averaging on: nuSQuIDS' fast-oscillation averaging
        overload EvalFlavor(flv, cz, E, type, scale, avr) (fixed height; avr is an
        ignored per-call output buffer of length n_flavors).
        """
        flav = self.NSQneuflavor
        cosz = self.CosZTrue.astype(float).tolist()
        enu = (self.ETrue * self.UNITS.GeV).astype(float).tolist()
        ntype = self.NSQneutype
        if self.osc_avg_scale is None:
            return np.asarray(list(map(
                self.Osc.EvalFlavor, flav, cosz, enu, ntype, repeat(True))))
        s, nf = self.osc_avg_scale, self.NeutrinoFlavors
        return np.asarray([
            self.Osc.EvalFlavor(f, c, e, t, s, [False] * nf)
            for f, c, e, t in zip(flav, cosz, enu, ntype)
        ])

    def GetOscillations(self):
        """Oscillated flux at every event/cell for the current parameters."""
        self.ApplyParameters()
        self.Osc.Set_initial_state(self.InitialFlux, nsq.Basis.flavor)
        self.Osc.EvolveState()

        return self._eval_flavor_weights()
