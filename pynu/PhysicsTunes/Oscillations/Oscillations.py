"""
Neutrino Oscillations module for Pynu.

This module provides the base Oscillator class that handles
oscillation parameter setup and weight calculation for neutrino experiments.

Extended to support CPT invariance testing with separate mass-squared
differences for neutrinos (Dm231) and antineutrinos (Dm231_bar).

Also supports Dm232 parameterization for profile likelihood marginalization,
where Dm232 = Dm32 affects both neutrino and antineutrino oscillations equally.
"""

from math import asin, sqrt
import numpy as np
import nuSQuIDS as nsq
from ..PhysicsTunes import Tune
import sys


class Oscillator(Tune):
    r""" General class for handling computation of neutrinos oscillations.

    Extended to support CPT invariance testing with separate mass-squared
    differences for neutrinos (Dm231) and antineutrinos (Dm231_bar).

    Attributes:
        Parameters (dict): Dictionary of oscillation parameters including:
            - Sin2Theta12: sin²θ₁₂ mixing angle
            - Sin2Theta13: sin²θ₁₃ mixing angle
            - Sin2Theta23: sin²θ₂₃ mixing angle
            - Dm221: Δm²₂₁ mass-squared difference (eV²)
            - Dm231: Δm²₃₁ mass-squared difference for neutrinos (eV²)
            - Dm231_bar: Δm̄²₃₁ mass-squared difference for antineutrinos (eV²)
            - Dm232: Δm²₃₂ mass-squared difference (eV²), used for marginalization
            - dCP: CP-violating phase δ_CP
            - Ordering: Mass ordering ("normal" or "inverted")
    """

    def __init__(self, scenario: str, neutrino_flavors: int, source=None) -> None:
        r""" Initial method for setting the basic variables.

        Args:
            scenario (str): Description of the physics scenario to be accounted for in oscillations. Options are standard and NSI.
            neutrino_flavors (int): Number of neutrinos considered (active and non-active).

        Returns:
            None.
        """
        super().__init__()

        """ Support for SM and NSI scenarios """
        self.Scenario = scenario
        self.Source = source
        self.NSI = False
        if "NSI" in self.Scenario or "nsi" in self.Scenario:
            self.NSI = True

        """ Support for 3 active neutrinos and any number of sterile neutrinos """
        self.NeutrinoFlavors = neutrino_flavors
        self.UNITS = nsq.Const()
        self.interactions = False
        self.REL_ERROR = 1e-8
        self.ABS_ERROR = 1e-8
        self.E_nodes = 200
        self.EPS = 1e-2

        self.ParameterLabels = None
        self.Parameters = {
            "Sin2Theta12": 0,
            "Sin2Theta13": 0,
            "Sin2Theta23": 0,
            "Dm221": 0,
            "Dm231": 0,
            "Dm231_bar": 0,  # CPT parameter: antineutrino mass-squared difference
            "Dm232": None,   # If set, overrides Dm31 = Dm32 + Dm21 (for marginalization)
            "dCP": 0,
            "Ordering": "normal",
        }

    def SetParameterLabels(self, **kwpars) -> None:
        if self.ParameterLabels is None:
            self.ParameterLabels = kwpars.items()

    def UpdateParameter(self, name: str, value: float) -> None:
        self.Parameters[name] = value
        self.ApplyParameters()

    def SetUpParameters(self, **kwpars) -> None:
        self.Parameters = kwpars
        self.ApplyParameters()

    def ApplyParameters(self):
        for i in range(1, self.NeutrinoFlavors):
            for j in range(i):
                s_theta = f"Sin2Theta{str(j + 1)}{str(i + 1)}"
                if s_theta in self.Parameters:
                    theta = self.Parameters[s_theta]
                    self.Osc.Set_MixingAngle(j, i, asin(sqrt(theta)))
            s_dm = f"Dm2{str(i + 1)}1"
            if s_dm in self.Parameters:
                dm = self.Parameters[s_dm]
                if "inverted" in self.Parameters["Ordering"] and s_dm == "Dm231":
                    dm = self.Parameters["Dm221"] - self.Parameters["Dm231"]
                self.Osc.Set_SquareMassDifference(i, dm)
        if "dCP" in self.Parameters:
            self.Osc.Set_CPPhase(0, 2, self.Parameters["dCP"])
        if self.NeutrinoFlavors > 3 and "dCP2" in self.Parameters:
            self.Osc.Set_CPPhase(0, 3, self.Parameters["dCP2"])

    def SetUpOscillator(self):
        if self.Source == "Atmospheric":
            print("Atmospheric")
            self.Osc = nsq.nuSQUIDSAtm(
                self.cth_nodes,
                self.energy_nodes * self.UNITS.GeV,
                self.NeutrinoFlavors,
                nsq.NeutrinoType.both,
                self.interactions,
            )
        elif self.Source == "Sun":
            pass
        elif self.Source == "Accelerator":
            pass

        self.Osc.Set_rel_error(self.REL_ERROR)
        self.Osc.Set_abs_error(self.ABS_ERROR)

    def GetOscillations(self):
        sys.exit("Oscillator not defined.")

    def is_cpt_asymmetric(self):
        """
        Check if CPT-asymmetric parameters are being used.

        Returns:
            bool: True if Dm231 != Dm231_bar (CPT violation scenario)
        """
        dm31 = self.Parameters.get("Dm231", 0)
        dm31_bar = self.Parameters.get("Dm231_bar", dm31)
        return abs(dm31 - dm31_bar) > 1e-10

    # =========================================================================
    # Parameter setter methods
    # =========================================================================

    def Sin2Theta13(self, experiment, x):
        self.Osc.Set_MixingAngle(0, 2, asin(sqrt(x)))
        self.Parameters["Sin2Theta13"] = x
        return self.GetOscillations()

    def diff_Sin2Theta13(self, experiment, x):  # Numerical derivation
        h0 = x * (1 + self.EPS)
        h1 = x * (1 - self.EPS)
        w0 = self.Sin2Theta13(experiment, h0)
        w1 = self.Sin2Theta13(experiment, h1)
        dw = (w0 - w1) / (h0 - h1)
        self.Parameters["Sin2Theta13"] = x
        return dw

    def Sin2Theta12(self, experiment, x):
        self.Osc.Set_MixingAngle(0, 1, asin(sqrt(x)))
        self.Parameters["Sin2Theta12"] = x
        return self.GetOscillations()

    def diff_Sin2Theta12(self, experiment, x):  # Numerical derivation
        h0 = x * (1 + self.EPS)
        h1 = x * (1 - self.EPS)
        w0 = self.Sin2Theta12(experiment, h0)
        w1 = self.Sin2Theta12(experiment, h1)
        dw = (w0 - w1) / (h0 - h1)
        self.Parameters["Sin2Theta12"] = x
        return dw

    def Sin2Theta23(self, experiment, x):
        self.Osc.Set_MixingAngle(1, 2, asin(sqrt(x)))
        self.Parameters["Sin2Theta23"] = x
        return self.GetOscillations()

    def diff_Sin2Theta23(self, experiment, x):  # Numerical derivation
        h0 = x * (1 + self.EPS)
        h1 = x * (1 - self.EPS)
        w0 = self.Sin2Theta23(experiment, h0)
        w1 = self.Sin2Theta23(experiment, h1)
        dw = (w0 - w1) / (h0 - h1)
        self.Parameters["Sin2Theta23"] = x
        return dw

    def dCP(self, experiment, x):
        self.Osc.Set_CPPhase(0, 2, x)
        self.Parameters["dCP"] = x
        return self.GetOscillations()

    def diff_dCP(self, experiment, x):  # Numerical derivation
        h0 = x * (1 + self.EPS)
        h1 = x * (1 - self.EPS)
        w0 = self.dCP(experiment, h0)
        w1 = self.dCP(experiment, h1)
        dw = (w0 - w1) / (h0 - h1)
        self.Parameters["dCP"] = x
        return dw

    def Dm221(self, experiment, x):
        self.Osc.Set_SquareMassDifference(1, x)
        self.Parameters["Dm221"] = x
        return self.GetOscillations()

    def diff_Dm221(self, experiment, x):  # Numerical derivation
        h0 = x * (1 + self.EPS)
        h1 = x * (1 - self.EPS)
        w0 = self.Dm221(experiment, h0)
        w1 = self.Dm221(experiment, h1)
        dw = (w0 - w1) / (h0 - h1)
        self.Parameters["Dm221"] = x
        return dw

    def Dm231(self, experiment, x):
        """
        Set Δm²₃₁ mass-squared difference for neutrinos (eV²).

        In standard oscillations, this applies to both neutrinos and
        antineutrinos. For CPT studies, set Dm231_bar separately.
        """
        self.Osc.Set_SquareMassDifference(2, x)
        self.Parameters["Dm231"] = x
        return self.GetOscillations()

    def diff_Dm231(self, experiment, x):  # Numerical derivation
        h0 = x * (1 + self.EPS)
        h1 = x * (1 - self.EPS)
        w0 = self.Dm231(experiment, h0)
        w1 = self.Dm231(experiment, h1)
        dw = (w0 - w1) / (h0 - h1)
        self.Parameters["Dm231"] = x
        return dw

    def Dm231_bar(self, experiment, x):
        """
        Set Δm̄²₃₁ mass-squared difference for antineutrinos (eV²).

        This parameter enables CPT invariance testing by allowing different
        mass-squared differences for neutrinos and antineutrinos.

        When Dm231_bar != Dm231, dual propagation is used to compute
        oscillation probabilities separately for ν and ν̄.

        Args:
            experiment: Experiment object
            x: Antineutrino mass-squared difference value (eV²)

        Returns:
            Oscillation weights from GetOscillations()
        """
        self.Parameters["Dm231_bar"] = x
        return self.GetOscillations()

    def diff_Dm231_bar(self, experiment, x):  # Numerical derivation
        h0 = x * (1 + self.EPS)
        h1 = x * (1 - self.EPS)
        w0 = self.Dm231_bar(experiment, h0)
        w1 = self.Dm231_bar(experiment, h1)
        dw = (w0 - w1) / (h0 - h1)
        self.Parameters["Dm231_bar"] = x
        return dw

    def Dm232(self, experiment, x):
        """
        Set Δm²₃₂ mass-squared difference (eV²).

        When Dm232 is set (not None), it overrides the Dm31 calculation:
        effective Dm31 = Dm32 + Dm21 for both neutrinos and antineutrinos.

        This is used for profile likelihood marginalization where Dm32
        affects both ν and ν̄ oscillations equally while Dm231_bar can
        still be scanned independently for CPT studies.

        Args:
            experiment: Experiment object
            x: Mass-squared difference Dm32 value (eV²)

        Returns:
            Oscillation weights from GetOscillations()
        """
        self.Parameters["Dm232"] = x
        return self.GetOscillations()

    def diff_Dm232(self, experiment, x):  # Numerical derivation
        h0 = x * (1 + self.EPS)
        h1 = x * (1 - self.EPS)
        w0 = self.Dm232(experiment, h0)
        w1 = self.Dm232(experiment, h1)
        dw = (w0 - w1) / (h0 - h1)
        self.Parameters["Dm232"] = x
        return dw

    # =========================================================================
    # Helper methods
    # =========================================================================

    def NSQNeutrinoType(self, experiment):
        neutype = np.zeros(experiment.NumberOfEvents)
        neutype[experiment.nuPDG < 0] = 1
        return neutype.astype(np.uint32).tolist()

    def NSQNeutrinoFlavor(self, experiment):
        neuflavor = 0.5 * np.abs(experiment.nuPDG) - 6
        return neuflavor.astype(np.uint32).tolist()


# Alias for backwards compatibility
NeutrinoOscillations = Oscillator
