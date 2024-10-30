from math import asin, sqrt
import numpy as np
import nuSQuIDS as nsq
from PhysicsTunes import Tune
import sys

sys.path.append("../")


# General oscillator


class Oscillator(Tune):
    def __init__(self, scenario, neutrino_flavors, source=None):
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
        self.REL_ERROR = 1e-4
        self.ABS_ERROR = 1e-4
        self.E_nodes = 100
        self.eps = 1e-2

        self.ParameterLabels = None
        self.Parameters = {
            "Sin2Theta12": 0,
            "Sin2Theta13": 0,
            "Sin2Theta23": 0,
            "Dm221": 0,
            "Dm231": 0,
            "dCP": 0,
            "Ordering": "normal",
        }

    def SetParameterLabels(self, **kwpars):
        if self.ParameterLabels is None:
            self.ParameterLabels = kwpars.items()

    def UpdateParameter(self, name, value):
        self.Parameters[name] = value
        self.ApplyParameters()

    def SetUpParameters(self, **kwpars):
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

    def Sin2Theta13(self, experiment, x):
        self.Osc.Set_MixingAngle(0, 2, asin(sqrt(x)))
        self.Parameters["Sin2Theta13"] = x
        return self.GetOscillations()

    def diff_Sin2Theta13(self, experiment, x):  # Numerical derivation
        h0 = x * (1 + self.eps)
        h1 = x * (1 - self.eps)
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
        h0 = x * (1 + self.eps)
        h1 = x * (1 - self.eps)
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
        h0 = x * (1 + self.eps)
        h1 = x * (1 - self.eps)
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
        h0 = x * (1 + self.eps)
        h1 = x * (1 - self.eps)
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
        h0 = x * (1 + self.eps)
        h1 = x * (1 - self.eps)
        w0 = self.Dm221(experiment, h0)
        w1 = self.Dm221(experiment, h1)
        dw = (w0 - w1) / (h0 - h1)
        self.Parameters["Dm221"] = x
        return dw

    def Dm231(self, experiment, x):
        self.Osc.Set_SquareMassDifference(2, x)
        self.Parameters["Dm231"] = x
        return self.GetOscillations()

    def diff_Dm231(self, experiment, x):  # Numerical derivation
        h0 = x * (1 + self.eps)
        h1 = x * (1 - self.eps)
        w0 = self.Dm231(experiment, h0)
        w1 = self.Dm231(experiment, h1)
        dw = (w0 - w1) / (h0 - h1)
        self.Parameters["Dm231"] = x
        return dw

    def NSQNeutrinoType(self, experiment):
        neutype = np.zeros(experiment.NumberOfEvents)
        neutype[experiment.nuPDG < 0] = 1
        return neutype.astype(np.uint32).tolist()

    def NSQNeutrinoFlavor(self, experiment):
        neuflavor = 0.5 * np.abs(experiment.nuPDG) - 6
        return neuflavor.astype(np.uint32).tolist()
