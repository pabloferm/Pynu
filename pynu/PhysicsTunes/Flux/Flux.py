# Manages physics tunes related to the flux

def FluxManager(source, experiment):
	if source == 'Atmospheric':
		from .AtmoFlux import AtmosphericFlux
		return AtmosphericFlux(experiment)

