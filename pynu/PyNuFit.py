"""Construct the experiment and the oscillation object of an event XML, and evaluate the
oscillated expectation at the oscillation point the XML states."""

from .analysis_reader import ParseXML
from . import Experiments as Exp

from .PhysicsTunes.PhysicsTunes import PhysicsTunes as PT


class PyNuFit:
    """The experiments of one event XML and their physics tunes."""

    def __init__(self, analysis_file):
        self.Analysis = ParseXML(analysis_file)
        self.Analysis.get_analysis()

        self.physics_tunes = {}

        self.SetUpExperiments()
        self.SetUpPhysicsTunes()

        self.ComputeBinnedObservation()

    def ComputeBinnedObservation(self):
        self.ApplyFixedWeights()
        self.ApplyOscillations()
        self.SetBinnedObservedEvents()

    def SetUpExperiments(self):
        """One entry per experiment of the XML, with key 'detector+source'."""
        experiment = {}
        for det in self.Analysis.Experiments.keys():
            for src in self.Analysis.Experiments[det].keys():
                details = self.Analysis.Experiments[det][src]
                exp = f"{det}+{src}"
                experiment[exp] = Exp.Manager(det, src, details)
        self.Experiments = experiment

    def SetUpPhysicsTunes(self):
        for name, exp in self.Experiments.items():
            self.physics_tunes[name] = PT(exp, self.Analysis.Flavors)

    def SetBinnedObservedEvents(self):
        for name, exp in self.Experiments.items():
            exp.SetObservedBinned()

    def ApplyFixedWeights(self):
        """Set the oscillation parameters of the XML, in XML order."""
        for name in self.Experiments:
            for tune in self.Analysis.Fixed:
                self.physics_tunes[name].OscillationTunes.UpdateParameter(
                    tune, self.Analysis.FixedValue[tune]
                )

    def ApplyOscillations(self):
        for name, exp in self.Experiments.items():
            w = self.physics_tunes[name].OscillationTunes.GetOscillations()
            exp.UpdateNominalWeights(w)
