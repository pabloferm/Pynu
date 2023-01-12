# from .CrossSection import *
# from .Detector import *
# from .Oscillations.Oscillations import Oscillations


class PhysicsTunes:
	""" Contains all physics tunes of a given experiment """
	def __init__(self, experiment, set_all=False):
		self.Detector = experiment.Detector
		self.Target = experiment.Target
		self.Source = experiment.Source

		self._Experiment = experiment

		if set_all:
			""" Set the flux """
			self.SetFlux()

	@property
	def Experiment(self):
		return self._Experiment

	@Experiment.setter
	def Experiment(self, experiment):
		self._Experiment = experiment

	def GetFlux(self, func_name, x):
		return self.Flux.Get(func_name, self._Experiment, x)

	def SetFlux(self):
		if self.Source == 'Atmospheric':
			from .Flux.AtmoFlux import AtmosphericFlux
			self.Flux = AtmosphericFlux()
		elif self.Source == 'Solar':
			pass
		elif self.Source == 'Reactors':
			pass
		elif self.Source in ['Accelerator','LBL','T2K']:
			# from .SuperK.SuperK import SuperK_LBL
			# return SuperK_LBL(experiment)
			pass
		else:
			sys.exit('{name} not found.')

	def SetXSection(self):
		pass
		self.XSec = Manager(self._Experiment.Target)	

	def SetDetector(self):
		pass
		self.Det = Manager(self._Experiment.Detector)

	def SetOscillations(self):
		pass
		self.Osc = Manager(self._Experiment.OscScenario)





class Tune:
	""" Base class for physics tunes """
	def __init__(self):
		pass

	def Get(self, tune, exp, x):
		""" Get specific weights for a given experiment from tune evaluated at x """
		try:
			return getattr(self,tune)(exp, x)
		except:
			print(tune + ' not found!!')
