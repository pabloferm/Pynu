import h5py
import numpy as np

import AnalysisReader as AR # contains parse class to read and setup the analysis
import Experiments as Exp # contains rd class to read and setup each experiment
from PhysicsTunes.PhysicsTunes import PhysicsTunes as PT # contains everything to modify your simulations to help figuring out what you have measured
import Fitter as FT # does all the fitting calculations


class PyNu:
	""" Top class containing everything """
	def __init__(self, analysis_file, verbosity=False):

		self.verbosity = verbosity

		""" Set up basic analysis variables and structure to build full analysis """
		self.Analysis = AR.parse(analysis_file, check=self.verbosity)

		""" Define dictionary for PhysicsTunes """
		self.PhysicsTunes = {}


	def SetUpExperiments(self):
		""" Loop over experiments specified in analysis file and store each of them
		into a dictionary with keys 'detector_source' (e.g. HyperK+Atmospheric) """
		""" Provides a dict of all experiments """
		experiment = {}
		for det in self.Analysis.Experiments.keys():
			for src in self.Analysis.Experiments[det].keys():
				details = self.Analysis.Experiments[det][src]
				exp = det + '+' + src
				experiment[exp] = Exp.Manager(det, src, details, self.Analysis.Scenario)
		self.Experiments = experiment


	def SetUpPhysicsTunes(self):
		""" Loop over physics tunes specified in analysis file and store each of them
		into a dictionary with keys 'detector+source' (e.g. HyperK+Atmospheric) """
		for name, exp in self.Experiments.items():
			self.PhysicsTunes[name] = PT(exp, self.Analysis.Scenario, self.Analysis.Flavors, set_all=True)


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
		if self.verbosity: print("Applying Fixed Weights")
		self.ApplyWeights('Fixed')


	def ApplyNominalWeights(self): # Nuisance parameters
		if self.verbosity: print("Applying Nominal Nuisance Weights")
		self.ApplyWeights('Nominal')


	def ApplyTrueWeights(self): # Physics parameters
		if self.verbosity: print("Applying Physics True Weights")
		self.ApplyWeights('True')


	def ApplyPhysicsWeights(self, point): # Physics parameters
		if self.verbosity: print("Applying Physics Point Weights")
		self.ApplyWeights('Physics', vector=self.Analysis.FullPhysicsGrid[point])


	def ApplyNuisanceWeights(self, vector): # Physics parameters
		if self.verbosity: print("Applying Nuisance Weights")
		self.ApplyWeights('Nuisance', vector=vector)


	def ApplyOscillations(self, Expectation=False): # Tag can be either "Nominal" or "Variable"
		for name, exp in self.Experiments.items():
			w = self.PhysicsTunes[name].OscillationTunes.GetOscillations()
			if Expectation:
				exp.UpdateExpectedWeights(w)
			else:
				exp.UpdateObservedWeights(w)
		

	def ApplyWeights(self, tag, vector=None):
		if tag == 'Fixed':
			labels = self.Analysis.Fixed
			vec = self.Analysis.FixedValue
		elif tag == 'Nominal':
			labels = self.Analysis.Nuisance
			vec = self.Analysis.NuisNominal
		elif tag == 'True':
			labels = self.Analysis.Physics
			vec = self.Analysis.PhysTrue
		elif tag == 'Physics':
			labels = self.Analysis.Physics
			v_id = self.Analysis.PhysicsList
		elif tag == 'Nuisance':
			labels = self.Analysis.Nuisance
			v_id = self.Analysis.NuisanceList
		else:
			sys.exit('Not a valid tag for applying weights.')

		for name, exp in self.Experiments.items():
			for source in labels:
				if source in exp.Definition.keys():
					tune_block = exp.Definition[source]
					for tune in labels[source]:
						if vector is not None:
							idx = v_id.index(tune)
							value = vector[idx]
						else:
							value = vec[source][tune]

						if tune_block == 'Flux':
							w = self.PhysicsTunes[name].GetFlux(tune, value)
						elif tune_block == 'XSection':
							w = self.PhysicsTunes[name].GetXSection(tune, value)
						elif tune_block == 'Detector':
							w = self.PhysicsTunes[name].GetDetector(tune, value)
						elif tune_block == 'Osc':
							self.PhysicsTunes[name].OscillationTunes.UpdateParameter(tune, value)
		
						if tune_block != 'Osc':
							exp.UpdateObservedWeights(w)
							if tag == 'Fixed':
								exp.UpdateBaseWeights(w)


	def CreateOutFile(self, fname):
		self.outfile = fname
		with h5py.File(fname, 'w') as hf:
			grp = hf.create_group('Fixed Parameters')
			for key in self.Analysis.Fixed.keys():
				this = grp.create_group(key)
				for par, val in self.Analysis.FixedValue[key].items():
					this.create_dataset(par, data=[val], compression='gzip')
			if self.Analysis.wSyst:
				grp = hf.create_group('Nuisance Parameters')
				for key in self.Analysis.Nuisance.keys():
					this = grp.create_group(key)
					for par in self.Analysis.Nuisance[key]:
						this.create_dataset(par, data=[0.0]*self.Analysis.NumberOfPhysPoints, compression='gzip')
			grp = hf.create_group('Physics Parameters')
			for key in self.Analysis.Physics.keys():
				this = grp.create_group(key)
				for par in self.Analysis.Physics[key]:
					this.create_dataset(par, data=[0.0]*self.Analysis.NumberOfPhysPoints, compression='gzip')
			grp = hf.create_group('Analysis')
			grp.create_dataset('Chi2 Stats. Only', data=[0.0]*self.Analysis.NumberOfPhysPoints, compression='gzip')
			if self.Analysis.wSyst: grp.create_dataset('Chi2 Systs.', data=[0.0]*self.Analysis.NumberOfPhysPoints, compression='gzip')


	def WriteToOutFile(self, point):
		with h5py.File(self.outfile, 'r+') as hf:
			hf['Analysis/Chi2 Stats. Only'][point] = self.Sensitivity()
			i = 0
			for key in self.Analysis.Physics.keys():
				for par in self.Analysis.Physics[key]:
					hf['Physics Parameters/'+key+'/'+par][point] = self.Analysis.FullPhysicsGrid[point][i]
					i =+ 1





	def Sensitivity(self):
		X2 = 0
		for exp in self.Observation.keys():
			# Binned statistics
			E = self. Expectation[exp]
			O = self. Observation[exp]
			# print(E, O)
			X2 += FT.ChiSquared.Chi2StatsCombined(O, E)
		return X2
