import numpy as np
from itertools import repeat
from .Oscillations import Oscillator
import nuSQuIDS as nsq


####################
# Atmospheric flux #
####################


class AtmosphericOscillations(Oscillator):
    """
    Atmospheric neutrino oscillations handler.

    Extended to support CPT invariance testing:
    - When Dm231_bar is not set or equals Dm231: standard single propagation
    - When Dm231_bar differs from Dm231: dual propagation for CPT test
    """

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
        self.nuPDG = experiment.nuPDG  # Store for CPT mode

        self.SetUpOscillator()

        self.NSQneutype = self.NSQNeutrinoType(experiment)
        self.NSQneuflavor = self.NSQNeutrinoFlavor(experiment)

        self.InitialFlux = experiment.SetInitialFlux(
            self.energy_nodes, self.cth_nodes, neutrino_flavors
        )

        # Initialize Dm231_bar to None (will be set from Dm231 when needed)
        # Note: Dm231 is not yet set from XML at this point, so we can't copy it here
        # The GetOscillations() method handles the case where Dm231_bar is None or 0
        if "Dm231_bar" not in self.Parameters or self.Parameters["Dm231_bar"] == 0:
            self.Parameters["Dm231_bar"] = None

        # Cache for optimization
        self._last_params = None
        self._cached_weights = None

    def reset_cache(self):
        """Reset the weight cache when parameters change."""
        self._last_params = None
        self._cached_weights = None

    def _get_param_tuple(self):
        """Get tuple of current parameters for cache comparison."""
        return (
            self.Parameters.get("Sin2Theta12"),
            self.Parameters.get("Sin2Theta13"),
            self.Parameters.get("Sin2Theta23"),
            self.Parameters.get("Dm221"),
            self.Parameters.get("Dm231"),
            self.Parameters.get("Dm231_bar"),
            self.Parameters.get("dCP"),
            self.Parameters.get("Ordering"),
        )

    def _single_propagation(self, dm31_value):
        """
        Run single propagation with given Dm31 value.

        Args:
            dm31_value: The Dm31 value to use

        Returns:
            Array of oscillation weights
        """
        # Set Dm31
        self.Osc.Set_SquareMassDifference(2, dm31_value)

        # Propagate
        self.Osc.Set_initial_state(self.InitialFlux, nsq.Basis.flavor)
        self.Osc.EvolveState()

        # Evaluate weights for each event
        weights = list(
            map(
                self.Osc.EvalFlavor,
                self.NSQneuflavor,
                self.CosZTrue.astype(float).tolist(),
                (self.ETrue * self.UNITS.GeV).astype(float).tolist(),
                self.NSQneutype,
                repeat(True),
            )
        )

        return np.asarray(weights)

    def GetOscillations(self):
        """
        Calculate oscillation weights.

        In CPT mode (Dm231 != Dm231_bar):
        - Propagates neutrinos with Dm231
        - Propagates antineutrinos with Dm231_bar
        - Combines based on PDG code

        In standard mode:
        - Single propagation with Dm231
        """
        # Check cache
        current_params = self._get_param_tuple()
        if self._last_params == current_params and self._cached_weights is not None:
            return self._cached_weights

        dm31_nu = self.Parameters.get("Dm231", 2.511e-3)
        # Handle case where Dm231_bar is 0 (uninitialized) or not set
        dm31_nubar = self.Parameters.get("Dm231_bar")
        if dm31_nubar is None or dm31_nubar == 0:
            dm31_nubar = dm31_nu

        # Check if we need CPT mode (dual propagation)
        cpt_mode = abs(dm31_nu - dm31_nubar) > 1e-10

        if not cpt_mode:
            # Standard mode: single propagation
            self.ApplyParameters()  # Apply all parameters including Dm231
            self.Osc.Set_initial_state(self.InitialFlux, nsq.Basis.flavor)
            self.Osc.EvolveState()

            weights = list(
                map(
                    self.Osc.EvalFlavor,
                    self.NSQneuflavor,
                    self.CosZTrue.astype(float).tolist(),
                    (self.ETrue * self.UNITS.GeV).astype(float).tolist(),
                    self.NSQneutype,
                    repeat(True),
                )
            )
            weights = np.asarray(weights)
        else:
            # CPT mode: dual propagation
            # First, apply all parameters except Dm31
            self.ApplyParameters()

            # Propagate neutrinos with Dm231
            w_nu = self._single_propagation(dm31_nu)

            # Propagate antineutrinos with Dm231_bar
            w_nubar = self._single_propagation(dm31_nubar)

            # Combine based on PDG sign: positive PDG = neutrino, negative = antineutrino
            is_neutrino = self.nuPDG > 0
            weights = np.where(is_neutrino, w_nu, w_nubar)

        # Update cache
        self._last_params = current_params
        self._cached_weights = weights

        return weights
