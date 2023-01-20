import sys
sys.path.append('../')
from PhysicsTunes import Tune
import nuSQuIDS as nsq


# General oscillator

class Oscillator(Tune):
	def __init__(self, scenario, neutrino_flavors):
		super().__init__()
		self.Scenario = scenario
		self.NeutrinoFlavors = neutrino_flavors
		self.units = nsq.Const()
		self.interactions = False
		self.rel_error = 1e-4
		self.abs_error = 1e-4

	def Parameters(self, **kwpars):
		if self.NeutrinoFlavors <= 3:
			parameters = {'Sin2Theta12':0, 'Sin2Theta13':0, 'Sin2Theta23':0, 'Dm221':0, 'Dm231':0, 'dCP':0, 'Ordering':'normal'}
		else:
			# At least...
			parameters = {'Sin2Theta12':0, 'Sin2Theta13':0, 'Sin2Theta23':0, 'Dm221':0, 'Dm231':0, 'dCP':0, 'Ordering':'normal', 'Sin2Theta14':0, 'Sin2Theta24':0, 'Sin2Theta34':0, 'Dm241':0, 'dCP2':0}
		for par, value in kwpars.items():
			parameters[par] = value
		return parameters

	def UpdateParameters(self, **kwpars):
		for par, value in kwpars.items():
			self.parameters[par] = value
		self.SetParameters(**self.parameters)

	def SetParameters(self, **kwpars):
		parameters = Parameters(self.neutrino_flavors,**kwpars)
		for i in range(1,self.neutrino_flavors):
			for j in range(i):
				s_theta = 't'+str(j+1)+str(i+1)
				if s_theta in parameters:
					theta = parameters[s_theta]
					self.Osc.Set_MixingAngle(j,i,asin(sqrt(theta)))
			s_dm = 'Dm2'+str(i+1)+'1'
		if s_dm in parameters:
			dm = parameters[s_dm]
			if 'inverted' in parameters['Ordering'] and s_dm == 'Dm231':
				dm = parameters['Dm221'] - parameters['Dm231']
			self.Osc.Set_SquareMassDifference(i,dm)
		if 'dCP' in parameters:
			self.Osc.Set_CPPhase(0,2,parameters['dCP'])
		if self.neutrino_flavors > 3 and 'dCP2' in parameters:
			self.Osc.Set_CPPhase(0,3,parameters['dCP2'])

		self.Parameters = parameters


# def Oscillations(scenario, source, neutrino_flavors, experiment):
# 	if source == 'Atmospheric':
# 		from .AtmOsc import AtmosphericOscillations
# 		return AtmosphericOscillations(scenario, neutrino_flavors, experiment)	