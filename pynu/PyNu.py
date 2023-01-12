import AnalysisReader as AR # contains parse class to read and setup the analysis
import Experiments as Exp # contains rd class to read and setup each experiment
from PhysicsTunes.PhysicsTunes import PhysicsTunes as PT # contains everything to modify your simulations to help figuring out what you have measured
import Fitter as FT # does all the fitting calculations


class PyNu:
	""" Top class containing everything """
	def __init__(self, analysis_file, verbosity=False):

		""" Set up basic analysis variables and structure to build full analysis """
		self.analysis = AR.parse(analysis_file, check=verbosity)

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
		for det in self.analysis.Experiments.keys():
			for src in self.analysis.Experiments[det].keys():
				details = self.analysis.Experiments[det][src]
				exp = det + '+' + src
				experiment[exp] = Exp.Manager(det, src, details)
		self.Experiments = experiment



	def SetUpPhysicsTunes(self):
		""" Loop over physics tunes specified in analysis file and store each of them
		into a dictionary with keys 'detector+source' (e.g. HyperK+Atmospheric) """

		for name, exp in self.Experiments.items():
			self.PhysicsTunes[name] = PT(exp, set_all=True)
			# print(self.PhysicsTunes[name].GetFlux('Diff_FluxNormalization', 1))




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