# from .CrossSection import *
# from .Detector import *
# from .Oscillations.Oscillations import Oscillations
import sys


class PhysicsTunes:
    ''' Contains all physics tunes of a given experiment '''

    def __init__(self, experiment, scenario, neutrino_flavors, set_all=False):

        __slots__ = (
            'Detector',
            'Target',
            'SOURCE',
            'SCENARIO',
            'NeutrinoFlavors',
            '_Experiment')
        self.Detector = experiment.Detector
        self.Target = experiment.Target
        self.SOURCE = experiment.SOURCE

        self.SCENARIO = scenario
        self.NeutrinoFlavors = neutrino_flavors

        self._Experiment = experiment

        if set_all:
            ''' Set the flux '''
            self.SetFlux()
            ''' Set the cross-section '''
            self.SetXSection()
            ''' Set the detector '''
            self.SetDetector()
            ''' Set the oscillations '''
            self.SetOscillation()

    @property
    def Experiment(self):
        return self._Experiment

    @Experiment.setter
    def Experiment(self, experiment):
        self._Experiment = experiment

    def GetFlux(self, func_name, x):
        return self.FluxTunes.Get(func_name, self._Experiment, x)

    def GetXSection(self, func_name, x):
        return self.XSectionTunes.Get(func_name, self._Experiment, x)

    def GetDetector(self, func_name, x):
        return self.DetectorTunes.Get(func_name, self._Experiment, x)

    def GetOscillation(self, func_name, x):
        return self.OscillationTunes.Get(func_name, self._Experiment, x)

    def SetFlux(self):
        if self.SOURCE == 'Atmospheric':
            from .Flux.AtmoFlux import AtmosphericFlux
            self.FluxTunes = AtmosphericFlux()
        elif self.SOURCE == 'Solar':
            pass
        elif self.SOURCE == 'Reactors':
            pass
        elif self.SOURCE in ['Accelerator', 'LBL', 'T2K']:
            # from .SuperK.SuperK import SuperK_LBL
            # return SuperK_LBL(experiment)
            pass
        else:
            sys.exit(f'{self._Experiment.SOURCE} source not found.')

    def SetXSection(self):
        if self.Target == 'Water':
            from .CrossSection.WaterXSection import WaterXSection
            self.XSectionTunes = WaterXSection()
        else:
            sys.exit(f'{self._Experiment.Target} target not found.')

    def SetDetector(self):
        if self.Detector == 'IceCube-Upgrade':
            from .Detector.ICUpDetector import ICUpgrade
            self.DetectorTunes = ICUpgrade()
        elif self.Detector == 'SuperK_IV':
            from .Detector.SuperKIV_Detector import SuperK_IV
            self.DetectorTunes = SuperK_IV()
        else:
            sys.exit(f'{self.Detector} detector not found.')

    def SetOscillation(self):
        if self.SOURCE == 'Atmospheric':
            from .Oscillations.AtmOsc import AtmosphericOscillations
            self.OscillationTunes = AtmosphericOscillations(
                self.SCENARIO, self.NeutrinoFlavors, self._Experiment)
        else:
            sys.exit(
                f'{self._Experiment.SCENARIO} oscillaiton scenario not found.')


#############################################################################

class Tune:
    ''' Base class for physics tunes '''

    def __init__(self):
        pass

    def Get(self, tune, exp, x):
        ''' Get specific weights for a given experiment from tune evaluated at x '''
        try:
            # return getattr(self, tune)(exp, x)
            return self.__getattribute__(tune)(exp, x)
        except BaseException:
            print(tune + ' not found!!')
            return 1
