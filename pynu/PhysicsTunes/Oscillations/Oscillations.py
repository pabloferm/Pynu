# Manages physics tunes related to oscillations

def Oscillations(scenario, source, neutrino_flavors, experiment):
	if source == 'Atmospheric':
		from .AtmOsc import AtmosphericOscillations
		return AtmosphericOscillations(scenario, neutrino_flavors, experiment)


def Parameters(neutrino_flavors, **kwpars):
	steriles = neutrino_flavors -3
	if steriles == 0:
		parameters = {'Sin2Theta12':0, 'Sin2Theta13':0, 'Sin2Theta23':0, 'Dm221':0, 'Dm231':0, 'dCP':0, 'Ordering':'normal'}
	else:
		# At least...
		parameters = {'Sin2Theta12':0, 'Sin2Theta13':0, 'Sin2Theta23':0, 'Dm221':0, 'Dm231':0, 'dCP':0, 'Ordering':'normal', 'Sin2Theta14':0, 'Sin2Theta24':0, 'Sin2Theta34':0, 'Dm241':0, 'dCP2':0}
	for par, value in kwpars.items():
		parameters[par] = value
	return parameters