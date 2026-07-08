import os
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

    def __init__(self, scenario, neutrino_flavors, experiment, avg_scale=None):
        super().__init__(scenario, neutrino_flavors, source="Atmospheric")

        self.E_nodes = 200
        self.Z_nodes = 40

        self.energy_nodes = np.geomspace(
                experiment.Etrue_min, experiment.Etrue_max, self.E_nodes
            )
        self.cth_nodes = np.linspace(
            experiment.Z_edges[0], experiment.Z_edges[1], self.Z_nodes
        )

        # NEW OPTION (default OFF -> prior behavior byte-identical): nuSQuIDS
        # fast-oscillation averaging. Precedence: env PYNU_OSC_AVG_SCALE OVERRIDES
        # (compat with the live branch's build scripts), else the `avg_scale`
        # constructor arg, else OFF. "4pi" reproduces the SK prescription (average
        # modes whose sin^2 argument 1.2667*dm2*L/E exceeds 2*pi, i.e. L/E > ~2000
        # km/GeV); nuSQuIDS compares `scale` to the eigenvalue phase dm2*L/(2E) ==
        # 2x that sin^2 argument, hence 4*pi. OFF -> the prior
        # EvalFlavor(..., randomize_height=True) path.
        _env = os.environ.get("PYNU_OSC_AVG_SCALE")
        self.osc_avg_scale = self._resolve_avg_scale(
            _env if _env is not None else avg_scale)
        if self.osc_avg_scale is not None:
            print(f"[AtmOsc] fast-oscillation averaging ON: EvalFlavor scale="
                  f"{self.osc_avg_scale:.5f} "
                  f"(source={'env' if _env is not None else 'ctor'})")

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

    @staticmethod
    def _resolve_avg_scale(value):
        """Resolve an averaging selector to a nuSQuIDS EvalFlavor scale (float) or
        None (OFF). Accepts the tokens '2pi'/'4pi', a float or float-string, or
        None/''/'off'/'none' -> OFF."""
        if value is None:
            return None
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("", "off", "none"):
                return None
            tok = {"2pi": 2.0 * np.pi, "4pi": 4.0 * np.pi}.get(s)
            return tok if tok is not None else float(s)
        return float(value)

    def _eval_flavor_weights(self):
        """EvalFlavor over all events for the current evolved state.

        Default (osc_avg_scale is None): the prior path, passing
        randomize_height=True. When averaging is enabled, use nuSQuIDS's
        fast-oscillation averaging overload EvalFlavor(flv,cz,E,rho,scale,avr):
        modes whose accumulated eigenvalue phase exceeds `scale` are phase-averaged
        (avr is an ignored per-call output buffer of length n_flavors)."""
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
        return self._eval_flavor_weights()

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

            weights = self._eval_flavor_weights()
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
