# from .CrossSection import *
# from .Detector import *
# from .Oscillations.Oscillations import Oscillations
import sys
from functools import wraps
import time
from inspect import signature

class PhysicsTunes:
    """Contains all physics tunes of a given experiment"""

    # @logd(file=False, logging_level='debug')
    def __init__(self, experiment, scenario, neutrino_flavors, set_all=False):
        self.Detector = experiment.Detector
        self.Target = experiment.Target
        self.SOURCE = experiment.SOURCE

        self.SCENARIO = scenario
        self.NeutrinoFlavors = neutrino_flavors

        self._Experiment = experiment

        if set_all:
            """ Set the flux """
            self.SetFlux()
            """ Set the cross-section """
            self.SetXSection()
            """ Set the detector """
            self.SetDetector()
            """ Set the oscillations """
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
        if self.SOURCE == "Atmospheric":
            from .Flux.AtmoFlux import AtmosphericFlux

            self.FluxTunes = AtmosphericFlux()
        elif self.SOURCE == "Solar":
            pass
        elif self.SOURCE == "Reactors":
            pass
        elif self.SOURCE in ["Accelerator", "LBL", "T2K"]:
            # from .SuperK.SuperK import SuperK_LBL
            # return SuperK_LBL(experiment)
            pass
        else:
            sys.exit(f"{self._Experiment.SOURCE} source not found.")

    # @logd(file=False, logging_level='debug')
    def SetXSection(self):
        if self.Target == "Water":
            from .CrossSection.WaterXSection import WaterXSection

            self.XSectionTunes = WaterXSection()
        else:
            sys.exit(f"{self._Experiment.Target} target not found.")

    # @logd(file=False, logging_level='debug')
    def SetDetector(self):
        if self.Detector == "IceCube-Upgrade":
            from .Detector.ICUpDetector import ICUpgrade

            self.DetectorTunes = ICUpgrade()
        elif self.Detector == "DeepCore":
            from .Detector.DeepCoreDetector import DeepCore

            self.DetectorTunes = DeepCore()
        elif "SuperK" in self.Detector:  # needs more work
            if "IV" in self.Detector:
                from .Detector.SKIVDetector import SuperK_IV

                self.DetectorTunes = SuperK_IV()
            elif "Gd" in self.Detector:
                from .Detector.SKGdDetector import SuperK_Gd

                self.DetectorTunes = SuperK_Gd()
            elif "2023" in self.Detector:
                from .Detector.SKCombinedDetector import SuperK_Combined

                self.DetectorTunes = SuperK_Combined()
            else:
                from .Detector.SKDetector import SuperK

                self.DetectorTunes = SuperK()
        elif "HyperK" in self.Detector:  # to be changed
            if "NoNeutron" in self.Detector:
                pass
            else:
                from .Detector.SKDetector import SuperK

                self.DetectorTunes = SuperK()
        else:
            sys.exit(f"{self.Detector} detector not found.")

    # @logd(file=False, logging_level='debug')
    def SetOscillation(self):
        if self.SOURCE == "Atmospheric":
            from .Oscillations.AtmOsc import AtmosphericOscillations

            self.OscillationTunes = AtmosphericOscillations(
                self.SCENARIO, self.NeutrinoFlavors, self._Experiment
            )
        else:
            sys.exit(f"{self._Experiment.SCENARIO} oscillaiton scenario not found.")


#############################################################################


class Tune:
    """Base class for physics tunes"""

    # @logd(file=False, logging_level='debug')
    def __init__(self):
        self.cache = {}
        self.cache_size = 0
        max_cache_size_mb = 100
        self.max_cache_size = max_cache_size_mb * 1024 * 1024

    def cache_method(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Normalize function arguments using the signature
            sig = signature(func)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()

            # Convert any numpy arrays in args or kwargs to tuples
            def make_hashable(o):
                if isinstance(o, (list, tuple)):
                    return tuple(make_hashable(i) for i in o)
                elif isinstance(o, dict):
                    return tuple((k, make_hashable(v)) for k, v in o.items())
                else:
                    return o

            def get_size(obj):
                if isinstance(obj, (list, tuple, set, frozenset)):
                    return sum(get_size(i) for i in obj) + sys.getsizeof(obj)
                if isinstance(obj, dict):
                    return sum(
                        get_size(k) + get_size(v) for k, v in obj.items()
                    ) + sys.getsizeof(obj)
                return sys.getsizeof(obj)

            # Create a cache key from normalized arguments
            cache_key = tuple(
                (k, make_hashable(v)) for k, v in bound_args.arguments.items()
            )

            if cache_key in self.cache:
                print("Using cached result.")
                return self.cache[cache_key]["result"]

            start_time = time.time()
            result = func(*bound_args.args, **bound_args.kwargs)
            end_time = time.time()

            result_size = get_size(result)
            computation_time = end_time - start_time

            # Add the new result to the cache
            self.cache[cache_key] = {
                "result": result,
                "size": result_size,
                "computation_time": computation_time,
            }

            self.cache_size += result_size

            # Enforce cache size limit
            while self.cache_size > self.max_cache_size:
                # Find the entry with the highest computation time
                least_time_key = min(
                    self.cache, key=lambda k: self.cache[k]["computation_time"]
                )
                self.cache_size -= self.cache[least_time_key]["size"]
                del self.cache[least_time_key]

            return result

        return wrapper

    # @logd(file=False, logging_level='debug')
    @cache_method
    def Get(self, tune, exp, x):
        """Get specific weights for a given `experiment` from tune evaluated
        at `x`, given the name of the `tune`."""
        try:
            return self.__getattribute__(tune)(exp, x)
        except BaseException:
            print(tune + " not found!!")
            return 1
        print("====================================")

    def _unphysical_value(self, x, unphys_low=0, unphys_up=9999999):
        if x < unphys_low or x > unphys_up: return True
        return False

