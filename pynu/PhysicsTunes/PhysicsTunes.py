class PhysicsTunes:
    """The cross-section tunes and the oscillation object of one experiment."""

    def __init__(self, Experiment, neutrino_flavors: int) -> None:
        r"""
        Args:
            Experiment (`pynu.Experiments.Experiment` class): Class containing the information of the experiment,
            of special interest are the Monte Carlos simulations.
            neutrino_flavors (int): Number of neutrino flavors.
        """
        self.NEUTRINO_FLAVORS: int = neutrino_flavors
        self._Experiment = Experiment  # `pynu.Experiments.Experiment` class

        self.set_xsection()
        self.set_oscillation()

    def set_xsection(self) -> None:
        from .CrossSection.WaterXSection import WaterXSection

        self.XSectionTunes = WaterXSection()

    def set_oscillation(self) -> None:
        from .Oscillations.AtmOsc import AtmosphericOscillations

        self.OscillationTunes = AtmosphericOscillations(
            self.NEUTRINO_FLAVORS, self._Experiment
        )
