# Class for ORCA-Full atmospheric neutrinos using event-by-event MC
# (5x_with_interm.csv from MCGenerator.py pipeline).
# Thin subclass of ORCAFull_Atm — only difference is detector name.

from .ORCAFull_Atm import ORCAFull_Atm


class ORCAFullEvtMC_Atm(ORCAFull_Atm):
    """ORCA-Full using event-by-event MC (5x_with_interm.csv)."""

    def __init__(self, dict_of_details, scenario):
        super(ORCAFullEvtMC_Atm, self).__init__(dict_of_details, scenario)
        self.Detector = "ORCAFullEvtMC"


# Alias for MCReader routing
ORCAFullEvtMC = ORCAFullEvtMC_Atm
