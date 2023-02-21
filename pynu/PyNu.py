import AnalysisReader as AR # contains parse class to read and setup the analysis
import Experiments as Exp # contains rd class to read and setup each experiment
from PhysicsTunes.PhysicsTunes import PhysicsTunes as PT # contains everything to modify your simulations to help figuring out what you have measured
import Fitter as FT # does all the fitting calculations


class PyNu:
	""" Top class containing everything """
	def __init__(self, analysis_file, verbosity=False):

		""" Set up basic analysis variables and structure to build full analysis """
		self.Analysis = AR.parse(analysis_file, check=verbosity)

		"""  """
		self.PhysicsItems = ['Flux', 'XSec', 'Det', 'Osc']
		self.Treatments = ['Fixed', 'Physics', 'Nuisance']

		""" Define dictionary for PhysicsTunes """
		self.PhysicsTunes = {}

		self.Flux = {}
		self.CrossSection = {}
		self.Detector = {}
		self.Oscillations = {}

		""" Set up all experiment classes of the analysis """
		""" Let experiment be the pair (detector, source), 1-to-1 """

		# self.Experiments = self.SetUpExperiments()


		""" Set up all physics tune classes of the analysis """
		""" Let physics tune be any physics model or phenomenon which can modify each
		experiment's event rate expectation. Therefore, physics tunes are functions of
		the pair (detector, source), but only onto """
		""" There are two different and independent classifications for physics tunes:
		 	 + Depending of their physics nature they may belong to flux, oscillations,
		 	 detector or cross-section
		 	 + Depending on their treatment in the analysis they may belong to fixed, 
		 	 physics (to be fitted) or nuisance (systematics) """

		# self.FixedPhysicsTunes = self.SetUpPhysicsTunes()



	def SetUpExperiments(self):
		""" Loop over experiments specified in analysis file and store each of them
		into a dictionary with keys 'detector_source' (e.g. HyperK+Atmospheric) """
		""" Provides a dict of all experiments """
		experiment = {}
		for det in self.Analysis.Experiments.keys():
			for src in self.Analysis.Experiments[det].keys():
				details = self.Analysis.Experiments[det][src]
				exp = det + '+' + src
				experiment[exp] = Exp.Manager(det, src, details)
		self.Experiments = experiment


	def SetUpPhysicsTunes(self):
		""" Loop over physics tunes specified in analysis file and store each of them
		into a dictionary with keys 'detector+source' (e.g. HyperK+Atmospheric) """
		for name, exp in self.Experiments.items():
			self.PhysicsTunes[name] = PT(exp, self.Analysis.OscScenario, self.Analysis.Flavors, set_all=True)


	def StartExpectation(self):
		for exp in self.Experiments.values():
			exp.StartExpectedWeights()


	def SetObservedEvents(self):
		self.Observation = {}
		for name, exp in self.Experiments.items():
			exp.SetObservedBinned()
			self.Observation[name] = exp.GetObservedBinned()


	def SetExpectedEvents(self):
		self.Expectation = {}
		for name, exp in self.Experiments.items():
			exp.SetExpectedBinned()
			self.Expectation[name] = exp.GetExpectedBinned()


	def ApplyFixedWeights(self): # Nuisance parameters
		for name, exp in self.Experiments.items():
			for source in self.Analysis.Fixed:
				if source in exp.Definition.keys():
					tune_block = exp.Definition[source]
					for tune in self.Analysis.Fixed[source]:
						if tune_block == 'Flux':
							w = self.PhysicsTunes[name].GetFlux(tune, self.Analysis.FixedValue[source][tune])
						elif tune_block == 'XSection':
							w = self.PhysicsTunes[name].GetXSection(tune, self.Analysis.FixedValue[source][tune])
						elif tune_block == 'Detector':
							w = self.PhysicsTunes[name].GetDetector(tune, self.Analysis.FixedValue[source][tune])
						elif tune_block == 'Osc':
							self.PhysicsTunes[name].UpdateParameter(tune, self.Analysis.FixedValue[source][tune])

						exp.UpdateBaseWeights(w)
						exp.UpdateObservedWeights(w)


	def ApplyNominalWeights(self): # Nuisance parameters
		for name, exp in self.Experiments.items():
			for source in self.Analysis.Nuisance:
				if source in exp.Definition.keys():
					tune_block = exp.Definition[source]
					for tune in self.Analysis.Nuisance[source]:
						if tune_block == 'Flux':
							w = self.PhysicsTunes[name].GetFlux(tune, self.Analysis.NuisNominal[source][tune])
						elif tune_block == 'XSection':
							w = self.PhysicsTunes[name].GetXSection(tune, self.Analysis.NuisNominal[source][tune])
						elif tune_block == 'Detector':
							w = self.PhysicsTunes[name].GetDetector(tune, self.Analysis.NuisNominal[source][tune])
						elif tune_block == 'Osc':
							self.PhysicsTunes[name].UpdateParameter(tune, self.Analysis.NuisNominal[source][tune])

						exp.UpdateObservedWeights(w)


	def ApplyTrueWeights(self): # Physics parameters
		for name, exp in self.Experiments.items():
			for source in self.Analysis.Physics:
				if source in exp.Definition.keys():
					tune_block = exp.Definition[source]
					for tune in self.Analysis.Physics[source]:
						if tune_block == 'Flux':
							w = self.PhysicsTunes[name].GetFlux(tune, self.Analysis.PhysTrue[source][tune])
						elif tune_block == 'XSection':
							w = self.PhysicsTunes[name].GetXSection(tune, self.Analysis.PhysTrue[source][tune])
						elif tune_block == 'Detector':
							w = self.PhysicsTunes[name].GetDetector(tune, self.Analysis.PhysTrue[source][tune])
						elif tune_block == 'Osc':
							self.PhysicsTunes[name].UpdateParameter(tune, self.Analysis.PhysTrue[source][tune])

						exp.UpdateObservedWeights(w)


	def ApplyPhysicsWeights(self, point):
		for name, exp in self.Experiments.items():
			for source in self.Analysis.Physics:
				if source in exp.Definition.keys():
					tune_block = exp.Definition[source]
					for tune in self.Analysis.Physics[source]:
						idx = self.Analysis.PhysicsList.index(tune)
						if tune_block == 'Flux':
							w = self.PhysicsTunes[name].GetFlux(tune, self.Analysis.FullPhysicsGrid[point][idx])
						elif tune_block == 'XSection':
							w = self.PhysicsTunes[name].GetXSection(tune, self.Analysis.FullPhysicsGrid[point][idx])
						elif tune_block == 'Detector':
							w = self.PhysicsTunes[name].GetDetector(tune, self.Analysis.FullPhysicsGrid[point][idx])
						elif tune_block == 'Osc':
							self.PhysicsTunes[name].UpdateParameter(tune, self.Analysis.FullPhysicsGrid[point][idx])

						exp.UpdateExpectedWeights(w)


	def ApplySystematicsWeights(self, vector):
		for name, exp in self.Experiments.items():
			for source in self.Analysis.Physics:
				if source in exp.Definition.keys():
					tune_block = exp.Definition[source]
					for tune in self.Analysis.Physics[source]:
						idx = self.Analysis.PhysicsList.index(tune)
						if tune_block == 'Flux':
							w = self.PhysicsTunes[name].GetFlux(tune, vector[idx])
						elif tune_block == 'XSection':
							w = self.PhysicsTunes[name].GetXSection(tune, vector[idx])
						elif tune_block == 'Detector':
							w = self.PhysicsTunes[name].GetDetector(tune, vector[idx])
						elif tune_block == 'Osc':
							self.PhysicsTunes[name].UpdateParameter(tune, vector[idx])

						exp.UpdateExpectedWeights(w)


	def ApplyOscillations(self, Expectation=False): # Tag can be either "Nominal" or "Variable"
		for name, exp in self.Experiments.items():
			w = self.PhysicsTunes[name].OscillationTunes.Oscillator()
			print(w)
			if Expectation:
				exp.UpdateExpectedWeights(w)
			else:
				exp.UpdateObservedWeights(w)
		
