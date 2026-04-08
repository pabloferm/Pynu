# Class for ORCA-Full atmospheric neutrinos (from response matrices)
# Subclass of ORCA_Atm with no muon background and no detector systematics.
#
# Weight convention (same as ICUpgrade):
# - MC weights are in m^2 (Aeff * detector response)
# - Oscillated flux from nuSQuIDS is in (GeV^-1 cm^-2 sr^-1 s^-1)
# - NORM = (exposure/mc_exposure) * seconds_per_year * 1e4 (m^2 -> cm^2)
# - Expected events = Weight * NORM * PhysicsWeight * NuisanceWeight

import numpy as np
from .Orca import ORCA_Atm


class ORCAFull_Atm(ORCA_Atm):
    """
    ORCA-Full experiment class using meta-event parquet from response matrices.

    Differences from ORCA_Atm:
    - Detector name: "ORCAFull" (routes to no-op detector)
    - No atmospheric muon background
    - 3 PID samples: 0=cascade, 1=track, 2=intermediate
    - Weight in m^2 (not pre-multiplied by flux), needs SECONDS_PER_YEAR factor
    - Binning loaded from numpy files alongside the parquet
    """

    SECONDS_PER_YEAR = 3.15576e7  # Julian year

    def __init__(self, dict_of_details, scenario):
        super(ORCAFull_Atm, self).__init__(dict_of_details, scenario)
        # Override detector after parent init (parent sets it to "ORCA")
        self.Detector = "ORCAFull"

    def MCVariables(self):
        """
        Override ORCA_Atm.MCVariables to fix NORM for m^2 weight convention.

        ORCA-6 weights already include flux, so NORM = exposure * 1e4.
        ORCAFull weights are in m^2 (like ICUpgrade), so NORM needs
        the additional SECONDS_PER_YEAR * 1e4 factor.
        """
        # Call parent MCVariables (sets up all variables including NORM)
        super(ORCAFull_Atm, self).MCVariables()

        # Fix NORM: parent sets NORM = FitExposure * 1e4
        # We need NORM = (FitExposure / MCExposure) * SECONDS_PER_YEAR * 1e4
        # Parent already computed NORM = FitExposure * 1e4 (line 180 of Orca.py)
        # but MCExposure division was done in parent Experiment.__init__
        # So we just need to multiply by SECONDS_PER_YEAR
        self.NORM = self.NORM * self.SECONDS_PER_YEAR

        # Recompute BaseWeight with corrected NORM
        self.BaseWeight = self.Weight * self.NORM

    def Reader(self):
        """
        Override to skip muon MC extraction (no muons in ORCAFull).
        """
        import pandas as pd

        self.MC = {}
        for i, f in enumerate(self.MCFiles):
            if f.endswith('.parquet'):
                df = pd.read_parquet(f)
                newdata = {col: df[col].values for col in df.columns}
            else:
                from . import MCReader
                newdata = MCReader.reader(f)

            if i == 0:
                self.MC = newdata
            else:
                for key, value in newdata.items():
                    if key in self.MC:
                        self.MC[key] = np.append(self.MC[key], value)

        # No muon MC
        self.MuonMC = {}

        # Handle data files
        if self.DataFit:
            self.Data = {}
            for i, f in enumerate(self.DataFiles):
                if f.endswith('.parquet'):
                    df = pd.read_parquet(f)
                    newdata = {col: df[col].values for col in df.columns}
                else:
                    from . import MCReader
                    newdata = MCReader.reader(f)

                if i == 0:
                    self.Data = newdata
                else:
                    for key, value in newdata.items():
                        if key in self.Data:
                            self.Data[key] = np.append(self.Data[key], value)


# Alias for MCReader routing
ORCAFull = ORCAFull_Atm
