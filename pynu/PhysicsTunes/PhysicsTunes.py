import sys
from functools import wraps
import time
from inspect import signature


class PhysicsTunes:
    """Class for handling the computation and application of all physics tunes relevant for a given experiment."""

    def __init__(
        self, Experiment, scenario: str, neutrino_flavors: int, set_all: bool = True
    ) -> None:
        r"""Initial method which sets the basic information of an experiment, inhereting from the Experiment class.

        Args:
            Experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.
            scenario (str): Name of the neutrino physics scenario used to compute oscillations.
            neutrino_flavors (int): Number of neutrino flavors.

        Kwargs:
            set_all (bool): Option to set all physics tunes for the experiment, that is flux, cross section, detector and physics scenario. Default is set to True as it should be the most common use.

        Returns:
            None
        """
        self.DETECTOR: str = Experiment.Detector
        self.TARGET: str = Experiment.Target
        self.SOURCE: str = Experiment.SOURCE
        self.SCENARIO: str = scenario
        self.NEUTRINO_FLAVORS: int = neutrino_flavors
        self._Experiment = Experiment  # `pynu.Experiments.Experiment` class

        if set_all:
            self.set_flux()
            self.set_xsection()
            self.set_detector()
            self.set_oscillation()

    @property
    def Experiment(self):
        r"""Method for getting the Experiment object associated with the present set of physics tunes.

        Args:
            None.

        Returns:
            Experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment
        """
        return self._Experiment

    @Experiment.setter
    def Experiment(self, Experiment):
        r"""Method for explicitly setting the Experiment object associated with the present set of physics tunes.

        Args:
            Experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment

        Returns:
            None.
        """
        self._Experiment = Experiment

    def set_flux(self) -> None:
        r"""Method for selecting the flux block of physics tunes for this experiment.

        Args:
            None.

        Returns:
            None.
        """
        if self.SOURCE == "Solar":
            pass
        elif self.SOURCE == "Atmospheric":
            from .Flux.AtmoFlux import AtmosphericFlux

            self.FluxTunes = AtmosphericFlux()
        elif self.SOURCE == "Reactors":
            pass
        elif self.SOURCE not in ["Accelerator", "LBL", "T2K"]:
            sys.exit(f"{self._Experiment.SOURCE} source not found.")

    def get_flux(self, func_name: str, x: float):
        r"""Method for getting the weights associated to a tune of flux block for this experiment.

        Args:
            func_name (str): Name of the tune.
            x (float): Input parameter for the tune. TO BE GENERALIZED TO AN ARRAY OF PARAMETERS.

        Returns:
            Numpy.array with the event-by-event values corresponding to tune func_name evaluated at x.
        """
        return self.FluxTunes.Get(func_name, self._Experiment, x)

    def set_xsection(self) -> None:
        r"""Method for selecting the cross-section block of physics tunes for this experiment.

        Args:
            None.

        Returns:
            None.
        """
        if self.TARGET == "Water":
            from .CrossSection.WaterXSection import WaterXSection

            self.XSectionTunes = WaterXSection()
        else:
            sys.exit(f"{self._Experiment.Target} target not found.")

    def get_xsection(self, func_name, x):
        r"""Method for getting the weights associated to a tune of cross section block for this experiment.

        Args:
            func_name (str): Name of the tune.
            x (float): Input parameter for the tune. TO BE GENERALIZED TO AN ARRAY OF PARAMETERS.

        Returns:
            Numpy.array with the event-by-event values corresponding to tune func_name evaluated at x.
        """
        return self.XSectionTunes.Get(func_name, self._Experiment, x)

    def set_detector(self) -> None:
        r"""Method for selecting the detector block of physics tunes for this experiment.

        Args:
            None.

        Returns:
            None.
        """
        if self.DETECTOR == "IceCube-Upgrade":
            from .Detector.ICUpDetector import ICUpgrade

            self.DetectorTunes = ICUpgrade()
        elif self.DETECTOR == "DeepCore":
            from .Detector.DeepCoreDetector import DeepCore

            self.DetectorTunes = DeepCore()

        elif "IceCube-2017" in self.DETECTOR:  # needs more work
            from .Detector.IC2017Detector import IC2017

            self.DetectorTunes = IC2017()

        elif "SuperK" in self.DETECTOR:  # needs more work
            if "IV" in self.DETECTOR:
                from .Detector.SKIVDetector import SuperK_IV

                self.DetectorTunes = SuperK_IV()
            elif "Gd" in self.DETECTOR:
                from .Detector.SKGdDetector import SuperK_Gd

                self.DetectorTunes = SuperK_Gd()
            elif "2023" in self.DETECTOR:
                from .Detector.SKCombinedDetector import SuperK_Combined

                self.DetectorTunes = SuperK_Combined()
            else:
                from .Detector.SKDetector import SuperK

                self.DetectorTunes = SuperK()
        elif "HyperK" in self.DETECTOR:  # to be changed
            if "NoNeutron" not in self.DETECTOR:
                from .Detector.SKDetector import SuperK

                self.DetectorTunes = SuperK()
        elif self.DETECTOR == "ORCA":
            from .Detector.ORCADetector import ORCADetector

            self.DetectorTunes = ORCADetector()
        else:
            sys.exit(f"{self.DETECTOR} detector not found.")

    def get_detector(self, func_name, x):
        r"""Method for getting the weights associated to a tune of detector block for this experiment.

        Args:
            func_name (str): Name of the tune.
            x (float): Input parameter for the tune. TO BE GENERALIZED TO AN ARRAY OF PARAMETERS.

        Returns:
            Numpy.array with the event-by-event values corresponding to tune func_name evaluated at x.
        """
        return self.DetectorTunes.Get(func_name, self._Experiment, x)

    def set_oscillation(self) -> None:
        r"""Method for selecting the oscillations/physics scenario block of physics tunes for this experiment.

        Args:
            None.

        Returns:
            None.
        """
        if self.SOURCE == "Atmospheric":
            from .Oscillations.AtmOsc import AtmosphericOscillations

            self.OscillationTunes = AtmosphericOscillations(
                self.SCENARIO, self.NEUTRINO_FLAVORS, self._Experiment
            )
        else:
            sys.exit(f"{self._Experiment.SCENARIO} oscillaiton scenario not found.")

    def get_oscillation(self, func_name, x):
        r"""Method for getting the weights associated to a tune of oscillations block for this experiment.

        Args:
            func_name (str): Name of the tune.
            x (float): Input parameter for the tune. TO BE GENERALIZED TO AN ARRAY OF PARAMETERS.

        Returns:
            Numpy.array with the event-by-event values corresponding to tune func_name evaluated at x.
        """
        return self.OscillationTunes.Get(func_name, self._Experiment, x)


#############################################################################


class Tune:
    """Base class for the physics tunes of each block. It contains generic methods which are useful for every tune."""

    def __init__(self, MAX_CACHE_SIZE_MB: float = 100, verbosity: bool = False) -> None:
        r"""Sets up the basic variables for caching results.

        Kwargs:
            MAX_CACHE_SIZE_MB (float): Maximum size of cached computations in MB. Default is set to 100 MB.

        Returns:
            None.
        """
        self.cache = {}
        self.cache_size = 0
        self.MAX_CACHE_SIZE = MAX_CACHE_SIZE_MB * 1024 * 1024
        self.VERBOSITY = verbosity

    def cache_method(func: str):
        r"""Sets up the basic variables for caching results.

        Args:
            func (str): Name of tune.

        Returns:
            Numpy.array (float): Computed or cached weights for "func" tune evaluated at the provided parameter for this experiment.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            r"""Stores most time consuming tunes in cache.

            Args:
                Any args for the tune, typically `x` (float) and `experiment` (`pynu.Experiments.Experiment` class).

            Kwargs:
                Any args for the tune, typically none.

            Returns:
                Numpy.array (float): Cached weights for `func` tune evaluated at the provided parameter for this experiment.
            """
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
                print(
                    f"Using previously cached result for {cache_key[1][1]} of {cache_key[2][1].SOURCE} at {cache_key[2][1].Detector}, with x = {cache_key[3][1]}."
                )
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
            while self.cache_size > self.MAX_CACHE_SIZE:
                # Find the entry with the highest computation time
                least_time_key = min(
                    self.cache, key=lambda k: self.cache[k]["computation_time"]
                )
                self.cache_size -= self.cache[least_time_key]["size"]
                del self.cache[least_time_key]

            return result

        return wrapper

    @cache_method
    def Get(self, tune: str, Experiment, x: float):
        """Get specific weights for a given `experiment` from tune evaluated
        at `x`, given the name of the `tune`.

        Args:
            tune (str): Name of tune.
            Experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment.
            x (float): Input parameter for the tune. TO BE GENERALIZED TO AN ARRAY OF PARAMETERS.

        Returns:
            Numpy.array (float) of computed weights for `tune` evaluated at `x` for `Experiment`.

        """
        try:
            if self.VERBOSITY:
                print(
                    f"Computing {tune} weights for {Experiment.SOURCE} at {Experiment.Detector}, with x = {x}"
                )
            return self.__getattribute__(tune.strip())(Experiment, x)
        except BaseException:
            sys.exit(
                f"{tune} not found. Please, check if it is defined, the name is correct and/or the implementation refers to existing variables of {Experiment.SOURCE} at {Experiment.Detector}"
            )

    def _unphysical_value(
        self, x: float, unphys_low: float = 0, unphys_up: float = 9999999
    ) -> bool:
        """Checks if value of parameter is unphysical for a given tune to prevent non-meaningful results.

        Args:
            x (float): Input parameter for the tune. TO BE GENERALIZED TO AN ARRAY OF PARAMETERS.

        Kwargs:
            unphys_low (float): Lower bound for physicallity.
            unphys_up (float): Upper bound for physicallity.

        Returns:
            True if x is within bounds.

        """
        return x < unphys_low or x > unphys_up
