import xml.etree.ElementTree as ET
import sys
import numpy as np

class parseXML:
	def __init__(self, xmlfile='AnalysisTemplate.xml'):
		# create element tree object
		self.xmlfile = xmlfile
		self.tree = ET.parse(xmlfile)
		# get root element of XML file
		self.root = self.tree.getroot()
		self.disabledsyst = []
		self.SystematicsList = np.array([])
		self.Systematics = {}
		self.SystNominal = {}
		self.SystNominalList = []
		self.SystPrior = []
		self.SystSigma = {}
		self.SystSigmaList = []

	def reader(self, item, atrib='name'):
		itemList = []
		if item == 'NeutrinoExperiment':
			self.mcFiles = []
			self.Exposure = []
		for source in self.root.iter(item):
			if atrib=='name':
				if int(source.find('status').text):
					itemList.append(source.attrib[atrib])
					if item == 'NeutrinoExperiment':
						self.mcFiles.append(source.find('simulation').attrib['filename'])
						self.Exposure.append(float(source.find('exposure').text))
					sname = source.attrib['name']
					self.Systematics[sname] = []
					self.SystSigma[sname] = []
					self.SystNominal[sname] = []
					for syst in source.findall('systematic'):
						if int(syst.find('status').text):
							s = syst.attrib['name']
							self.SystSigma[sname].append(float(syst.find('sigma').text))
							self.SystSigmaList.append(float(syst.find('sigma').text))
							self.SystNominal[sname].append(float(syst.find('nominal').text))
							self.SystNominalList.append(float(syst.find('nominal').text))
							self.SystPrior.append(float(syst.find('nominal').text))
							self.Systematics[sname].append(s)
							self.SystematicsList = np.append(self.SystematicsList,s)
			else:
				itemList.append(source.attrib[atrib])
		return itemList

	def readSources(self):
		self.sources = self.reader('NeutrinoSource')
		print('------------------------------------')
		print('Neutrino sources considered:')
		for s in self.sources:
			print(' + ',s)
		print('====================================')

	def readDetectors(self):
		self.detectors = self.reader('Detector')
		print('------------------------------------')
		print('Neutrino detectors considered:')
		for s in self.detectors:
			print(' + ',s)
		print('====================================')

	def readExperiments(self):
		self.experiments = self.reader('NeutrinoExperiment')
		print('------------------------------------')
		print('Experiments considered:')
		for s,m in zip(self.experiments,self.mcFiles):
			print(' + ',s, ', at ',m)
		print('====================================')
		print(f' + {self.Systematics}')


	def readPhysics(self):
		self.physics = self.reader('NeutrinoPhysics')
		print('------------------------------------')
		print('Neutrino physics considered:')
		for s in self.physics:
			print(' + ',s)
		print('====================================')
	
	def readOscPar(self):
		keys = ['Sin2Theta12','Sin2Theta13','Sin2Theta23','Dm221','Dm231','dCP','Ordering']
		self.parameters = keys
		self.OscParametersGrid  = {}
		self.OscParametersEdges = {}
		self.OscParametersBest  = {}
		for node in self.root.iter('NeutrinoPhysics'):
			for phys in self.physics:
				if node.attrib['name'] == phys:
					self.neutrinos = int(node.find('flavours').text)
					for key in keys:
						par = node.find('parameters/'+key)
						self.OscParametersGrid[key] = self.readOscParValues(par, key)
						self.OscParametersEdges[key] = self.readOscParPrior(par, key)
						self.OscParametersBest[key] = self.readOscParBestValue(par, key)

	def readOscParValues(self, pars, key):
		if key=='Ordering':
			if int(pars.find('normal').text):
				x = np.array(['normal'])
				if int(pars.find('inverted').text):
					x = np.append(x,'inverted')
			else:
				if int(pars.find('inverted').text):
					x = np.array(['inverted'])
			return x
		else:
			n = int(pars.find('points').text)
			mini = float(pars.find('min').text)
			maxi = float(pars.find('max').text)
			return np.linspace(mini, maxi, n, endpoint = True)

	def readOscParBestValue(self, pars, key):
		if key=='Ordering':
			return str(pars.find('best').text).replace(" ", "")
		else:
			return float(pars.find('best').text)

	def readOscParPrior(self, pars, key):
		if key=='Ordering':
			if int(pars.find('normal').text):
				x = np.array(['normal'])
				if int(pars.find('inverted').text):
					x = np.append(x,'inverted')
			else:
				if int(pars.find('inverted').text):
					x = np.array(['inverted'])
			return x
		else:
			n = int(pars.find('points').text)
			mini = float(pars.find('min').text)
			maxi = float(pars.find('max').text)
			return np.array([mini, maxi])

	def CheckSystematics(self):
		nosyst = 0
		print('List of Systematics')
		for source in self.Systematics:
			print(f' + From {source}: {self.Systematics[source]}')
			if len(self.SystSigma[source]) > 0:
				nosyst = nosyst + 1
		print('====================================')
		if nosyst==0:
			self.NoSyst = 1
		else:
			self.NoSyst = 0

	