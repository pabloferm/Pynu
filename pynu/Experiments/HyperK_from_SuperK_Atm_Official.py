# Class for the atmospheric neutrinos in Hyper-Kamiokande based on
# the SuperK_IV official MC simulation

from .SuperK_Atm_Official import SuperK_IV, SuperK_IV_noNtag


class HyperK(SuperK_IV):
    def __init__(self, dict_of_details, scenario):
        super(HyperK, self).__init__(dict_of_details, scenario)
        self.Detector = 'HyperK'
        self.SOURCE = 'Atmospheric'
        self.Target = 'Water'
        self.SCENARIO = scenario

        self.SetDefinition()

        self.NORM *= 8.2
        self.BaseWeight = self.NORM * self.Weight


class HyperK_NoNeutron(SuperK_IV_noNtag):
    def __init__(self, dict_of_details, scenario):
        super(HyperK_NoNeutron, self).__init__(dict_of_details, scenario)
        self.Detector = 'HyperK_NoNeutron'
        self.SOURCE = 'Atmospheric'
        self.Target = 'Water'
        self.SCENARIO = scenario

        self.SetDefinition()
        print('definition')
        print(self.Definition)

        self.NORM *= 8.2
        self.BaseWeight = self.NORM * self.Weight
