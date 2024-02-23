# from .CrossSection import *
# from .Detector import *
# from .Oscillations.Oscillations import Oscillations
import sys
from LoggingDecorator import logd


class PhysicsTunes:
    ''' Contains all physics tunes of a given experiment '''

    # @logd(file=False, logging_level='debug')
    def __init__(self, experiment, scenario, neutrino_flavors, set_all=False):

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

    # @logd(file=False, logging_level='debug')
    def GetFlux(self, func_name, x):
        return self.FluxTunes.Get(func_name, self._Experiment, x)

    # @logd(file=False, logging_level='debug')
    def GetXSection(self, func_name, x):
        return self.XSectionTunes.Get(func_name, self._Experiment, x)

    # @logd(file=False, logging_level='debug')
    def GetDetector(self, func_name, x):
        return self.DetectorTunes.Get(func_name, self._Experiment, x)

    # @logd(file=False, logging_level='debug')
    def GetOscillation(self, func_name, x):
        return self.OscillationTunes.Get(func_name, self._Experiment, x)

    # @logd(file=False, logging_level='debug')
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

    # @logd(file=False, logging_level='debug')
    def SetXSection(self):
        if self.Target == 'Water':
            from .CrossSection.WaterXSection import WaterXSection
            self.XSectionTunes = WaterXSection()
        else:
            sys.exit(f'{self._Experiment.Target} target not found.')

    # @logd(file=False, logging_level='debug')
    def SetDetector(self):
        if self.Detector == 'IceCube-Upgrade':
            from .Detector.ICUpDetector import ICUpgrade
            self.DetectorTunes = ICUpgrade()
        elif 'SuperK' in self.Detector: # needs more work
            from .Detector.SKIVDetector import SuperK_IV
            self.DetectorTunes = SuperK_IV()
        elif 'HyperK' in self.Detector:  # to be changed
            from .Detector.SKDetector import SuperK
            self.DetectorTunes = SuperK()
        else:
            sys.exit(f'{self.Detector} detector not found.')

    # @logd(file=False, logging_level='debug')
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

    # @logd(file=False, logging_level='debug')
    def __init__(self):
        pass

    # @logd(file=False, logging_level='debug')
    def Get(self, tune, exp, x):
        """ Get specific weights for a given `experiment` from tune evaluated 
        at `x`, given the name of the `tune`. """
        # print("====================================")
        # print(f'tune {tune}')
        # print(f'exp {exp}')
        # print(f'x {x}')
        try:
            return self.__getattribute__(tune)(exp, x)
        except BaseException:
            print(tune + ' not found!!')
            return 1
        print("====================================")
