# Manages physics tunes related to the flux

def Flux(source, experiment):
	if source == 'Atmospheric':
		print(source)
		from .AtmoFlux import AtmosphericFlux
		return AtmosphericFlux(experiment)

