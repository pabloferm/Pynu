import numpy as np
import pandas as pd
import h5py
import nuflux
import nuSQuIDS as nsq
import nuSQUIDSTools
from itertools import repeat

class Experiment:
	def __init__(self, source, experiment, exposure, filename, physics):
		self.Experiment = experiment
		self.Source = source
		self.Exposure = exposure
		self.FewEntries = None
		self.Physics = 0 # list of physics parameters grid

	def Detector(self):
		if self.Experiment == 'Super-Kamiokande' or self.Experiment == 'SK' or self.Experiment == 'SuperK':
			self.Detector = 'Water'
			print(f'Processing simulation of {self.Experiment} experiment with a exposure of {self.Exposure} years.')
			# call SK reader
			# call SK binning

		elif self.Experiment == 'Hyper-Kamiokande' or self.Experiment == 'HK' or self.Experiment == 'HyperK':
			self.Detector = 'Water'
			print(f'Processing simulation of {self.Experiment} experiment with a exposure of {self.Exposure} years.')
			# call HK reader
			# call HK binning

	def Source(self):
		if self.Source == 'Atmospheric':
			'''Set up atmospheric neutrino oscillations'''
			# call atmospheric stuff, oscillations

	def Physics(self):
		# return ist of physics tunes functions

	pass

