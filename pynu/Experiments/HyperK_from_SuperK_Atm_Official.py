# Class for the atmospheric neutrinos in Hyper-Kamiokande based on
# the SuperK_IV official MC simulation

import numpy as np
import nuflux
from .SuperK_Atm_Official import SuperK_IV


class HyperK(SuperK_IV):
    def __init__(self, dict_of_details, scenario):
        super(HyperK, self).__init__(dict_of_details)
        self.Detector = 'HyperK'
        self.SOURCE = 'Atmospheric'
        self.Target = 'Water'
        self.SCENARIO = scenario

        self.SetDefinition()

        self.NORM *= 8.2
        self.BaseWeight = self.NORM * self.Weight
