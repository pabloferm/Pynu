"""Base class for the oscillation parameters and the nuSQuIDS object."""

from math import asin, sqrt
import numpy as np
import nuSQuIDS as nsq


class Oscillator:
    r""" General class for handling computation of neutrinos oscillations.

    Attributes:
        Parameters (dict): Dictionary of oscillation parameters including:
            - Sin2Theta12: sin²θ₁₂ mixing angle
            - Sin2Theta13: sin²θ₁₃ mixing angle
            - Sin2Theta23: sin²θ₂₃ mixing angle
            - Dm221: Δm²₂₁ mass-squared difference (eV²)
            - Dm231: Δm²₃₁ mass-squared difference (eV²), negative for the inverted ordering
            - dCP: CP-violating phase δ_CP
    """

    def __init__(self, neutrino_flavors: int) -> None:
        r"""
        Args:
            neutrino_flavors (int): Number of neutrino flavors.
        """
        self.NeutrinoFlavors = neutrino_flavors
        self.UNITS = nsq.Const()
        self.interactions = False
        self.REL_ERROR = 1e-8
        self.ABS_ERROR = 1e-8

        self.Parameters = {
            "Sin2Theta12": 0,
            "Sin2Theta13": 0,
            "Sin2Theta23": 0,
            "Dm221": 0,
            "Dm231": 0,
            "dCP": 0,
        }

    def UpdateParameter(self, name: str, value: float) -> None:
        self.Parameters[name] = value
        self.ApplyParameters()

    def ApplyParameters(self):
        for i in range(1, self.NeutrinoFlavors):
            for j in range(i):
                s_theta = f"Sin2Theta{str(j + 1)}{str(i + 1)}"
                theta = self.Parameters[s_theta]
                self.Osc.Set_MixingAngle(j, i, asin(sqrt(theta)))
            s_dm = f"Dm2{str(i + 1)}1"
            dm = self.Parameters[s_dm]
            self.Osc.Set_SquareMassDifference(i, dm)
        self.Osc.Set_CPPhase(0, 2, self.Parameters["dCP"])

    def SetUpOscillator(self):
        self.Osc = nsq.nuSQUIDSAtm(
            self.cth_nodes,
            self.energy_nodes * self.UNITS.GeV,
            self.NeutrinoFlavors,
            nsq.NeutrinoType.both,
            self.interactions,
        )

        self.Osc.Set_rel_error(self.REL_ERROR)
        self.Osc.Set_abs_error(self.ABS_ERROR)

    def NSQNeutrinoType(self, experiment):
        neutype = np.zeros(experiment.NumberOfEvents)
        neutype[experiment.nuPDG < 0] = 1
        return neutype.astype(np.uint32).tolist()

    def NSQNeutrinoFlavor(self, experiment):
        neuflavor = 0.5 * np.abs(experiment.nuPDG) - 6
        # A row with nuPDG = 0 would give -6, which wraps under the uint32 cast and
        # crashes EvalFlavor; clip it to flavor 0. No-op for every physical PDG.
        neuflavor = np.clip(neuflavor, 0, None)
        return neuflavor.astype(np.uint32).tolist()
