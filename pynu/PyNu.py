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


	def SetUpObservedEvents(self):
		self.Observation = {}
		for name, exp in self.Experiments.items():
			exp.SetObservedBinned()
			self.Observation[name] = exp.GetObservedBinned()


	def ApplyFixedWeights(self):
		for name, exp in self.Experiments.items():
			for source in self.Analysis.Fixed:
				if source in exp.Definition().keys():
					tune_block = exp.Definition()[source]
					for tune in self.Analysis.Fixed[source]:
						if tune_block == 'Flux':
							w = self.PhysicsTunes[name].GetFlux(tune, self.Analysis.FixedValue[source][tune])
						elif tune_block == 'XSection':
							w = self.PhysicsTunes[name].GetXSection(tune, self.Analysis.FixedValue[source][tune])
						elif tune_block == 'Detector':
							w = self.PhysicsTunes[name].GetDetector(tune, self.Analysis.FixedValue[source][tune])

						exp.UpdateNominalWeights(w)
						exp.UpdateBaseWeights(w)
						exp.UpdateExpectedWeights(w)


	def ApplyNominalWeights(self): # Nuisance parameters
		for name, exp in self.Experiments.items():
			for source in self.Analysis.Nuisance:
				if source in exp.Definition().keys():
					tune_block = exp.Definition()[source]
					for tune in self.Analysis.Nuisance[source]:
						if tune_block == 'Flux':
							w = self.PhysicsTunes[name].GetFlux(tune, self.Analysis.NuisNominal[source][tune])
						elif tune_block == 'XSection':
							w = self.PhysicsTunes[name].GetXSection(tune, self.Analysis.NuisNominal[source][tune])
						elif tune_block == 'Detector':
							w = self.PhysicsTunes[name].GetDetector(tune, self.Analysis.NuisNominal[source][tune])

						exp.UpdateNominalWeights(w)


	def ApplyTrueWeights(self): # Physics parameters
		for name, exp in self.Experiments.items():
			for source in self.Analysis.Physics:
				if source in exp.Definition().keys():
					tune_block = exp.Definition()[source]
					for tune in self.Analysis.Physics[source]:
						if tune_block == 'Flux':
							w = self.PhysicsTunes[name].GetFlux(tune, self.Analysis.PhysTrue[source][tune])
						elif tune_block == 'XSection':
							w = self.PhysicsTunes[name].GetXSection(tune, self.Analysis.PhysTrue[source][tune])
						elif tune_block == 'Detector':
							w = self.PhysicsTunes[name].GetDetector(tune, self.Analysis.PhysTrue[source][tune])

						exp.UpdateNominalWeights(w)


		# fluxes = []
		# xsections = []
		# detectors = []
		# for exp in self.Experiments.values:
		# 	fluxes.append(exp.Source)
		# 	xsections.append(exp.Target)
		# 	detectors.append(exp.Detector)		
		# fluxes = [*set(fluxes)]
		# xsections = [*set(xsections)]
		# detectors = [*set(detectors)]

		# for f in fluxes:
		# 	self.Flux[f] = PT.GetFluxTunes(f)
		
		# for xs in xsections:
		# 	self.CrossSection[xs] = PT.GetXSectionTunes(xs)
		
		# for d in detectors:
		# 	self.Detector[d] = PT.GetDetectorTunes(d)

		# self.Oscillations[an.OscScenario] = PT.GetOscTunes(an.OscScenario)

	# def GetFlux(self, exp, name):
	# 	return self.Flux[exp.Source].Get(name)