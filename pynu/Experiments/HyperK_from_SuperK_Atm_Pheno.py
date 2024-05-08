# Class for the atmospheric neutrinos in Hyper-Kamiokande based on
# the SuperK_Htag MC simulation developed for phenomenological studies

from .SuperK_Atm_Pheno import SuperK_Htag


class HyperK(SuperK_Htag):
    def __init__(self, dict_of_details, scenario):
        super(HyperK, self).__init__(dict_of_details)
        self.Detector = "HyperK_Pheno"
        self.SOURCE = "Atmospheric"
        self.Target = "Water"
        self.SCENARIO = scenario

        self.SetDefinition()

        self.NORM *= 8.2
        self.BaseWeight = self.NORM * self.Weight
