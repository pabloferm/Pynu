import xml.etree.ElementTree as ET
import sys
import collections
import numpy as np

class parseXML:
	def __init__(self, xmlfile='AnalysisFiles/test.xml'):
		
		# create element tree object
		self.xmlfile = xmlfile
		self.tree = ET.parse(xmlfile)
		
		# get root element of XML file
		self.root = self.tree.getroot()
		
		# Nuisance / Nuisematics
		self.disabledNuis = []
		self.NuisanceList = np.array([])
		self.Nuisance = {}
		self.NuisNominal = {}
		self.NuisNominalList = []
		self.NuisSigma = {}
		self.NuisSigmaList = []

		# Physics
		self.disabledPhys = []
		self.PhysicsList = np.array([])
		self.Physics = {}
		self.PhysTrue = {}
		self.PhysTrueList = []
		self.PhysGrid= {}
		self.PhysGridList= []
		self.PhysEdges = {}

		# Fixed
		self.disabledFixed = []
		self.FixedList = np.array([])
		self.Fixed = {}
		self.FixedValue = {}
		self.FixedValueList = []

		self.MCFiles = {}
		self.MCyears = {}
		self.DataFiles = {}
		self.Exposure = {}

		self.Experiments = {}

		self.readSources()
		self.readDetectors()
		self.readExperiments()
		self.readOscillations()
		self.CheckSources()


	def reader(self, item, atrib='name'):
		itemList = []
		# if item == 'NeutrinoExperiment':
		# 	self.MCFiles = {}
		# 	self.MCyears = {}
		# 	self.DataFiles = {}
		# 	self.Exposure = {}
		for source in self.root.iter(item):
			if atrib=='name':
				if int(source.find('status').text):
					sname = source.attrib['name']
					itemList.append(source.attrib[atrib])
					if item == 'NeutrinoExperiment':
						self.Experiments[sname] = {}
						for src in source.findall('source'):
							if int(src.find('status').text):
								self.Experiments[sname][src.attrib['name']] = {}
								MCFiles = []
								MCyears = []
								Exposure = []
								DataFiles = []
								# self.MCFiles[src.attrib['name']] = []
								# self.MCyears[src.attrib['name']] = []
								# self.Exposure[src.attrib['name']] = []
								# self.DataFiles[src.attrib['name']] = []
								for fi in src.findall('MCFiles'):
									if int(fi.find('status').text):
										MCFiles.append(fi.attrib['name'])
										MCyears.append(float(fi.find('mcyears').text))
								for fi in src.findall('DataFiles'):
									if int(fi.find('status').text):
										DataFiles.append(fi.attrib['name'])										
								Exposure.append(float(src.find('exposure').text))
						self.Experiments[sname][src.attrib['name']] = {'MCFiles':MCFiles, 'MCyears': MCyears, 'DataFiles': DataFiles, 'Exposure': Exposure}


					self.Nuisance[sname] = []
					self.NuisSigma[sname] = []
					self.NuisNominal[sname] = []
					for nuis in source.findall('nuisance'):
						if int(nuis.find('status').text):
							s = nuis.attrib['name']
							if s == 'Ordering':
								sys.exit('Neutrino mass ordering cannot be a nuisance parameter.')
							else:
								self.NuisSigma[sname].append(float(nuis.find('sigma').text))
								self.NuisSigmaList.append(float(nuis.find('sigma').text))
								self.NuisNominal[sname].append(float(nuis.find('nominal').text))
								self.NuisNominalList.append(float(nuis.find('nominal').text))
								self.Nuisance[sname].append(s)
								self.NuisanceList = np.append(self.NuisanceList,s)

					self.Fixed[sname] = []
					self.FixedValue[sname] = []
					for fix in source.findall('fixed'):
						if int(fix.find('status').text):
							s = fix.attrib['name']
							if s == 'Ordering':
								pass
							else:
								self.FixedValue[sname].append(float(fix.find('value').text))
								self.FixedValueList.append(float(fix.find('value').text))
								self.Fixed[sname].append(s)
								self.FixedList = np.append(self.FixedList,s)

					self.Physics[sname] = []
					self.PhysTrue[sname] = []
					self.PhysGrid[sname] = []
					self.PhysEdges[sname] = []
					for phys in source.findall('physics'):
						s = phys.attrib['name']
						if s == 'Ordering':
							pass
						else:
							self.PhysTrue[sname].append(float(phys.find('true').text))
							self.PhysTrueList.append(float(phys.find('true').text))
							self.PhysGrid[sname].append(float(phys.find('points').text))
							self.PhysGridList.append(float(phys.find('points').text))
							self.PhysEdges[sname].append([float(phys.find('min').text),float(phys.find('max').text)])
							self.Physics[sname].append(s)
							self.PhysicsList = np.append(self.PhysicsList,s)

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
		self.detectors = self.reader('NeutrinoTarget')
		print('------------------------------------')
		print('Neutrino targets considered:')
		for s in self.detectors:
			print(' + ',s)
		print('====================================')

	def readExperiments(self):
		self.experiments = self.reader('NeutrinoExperiment')
		print('------------------------------------')
		print('Experiments considered:')
		for s in self.experiments:
			print(' + ',s)
		print('====================================')
		# print(f' + {self.Nuisance}')
	
	def readOscillations(self):
		self.oscillations = self.reader('NeutrinoOscillations')
		print('------------------------------------')
		print('Oscillation scenario:')
		for i,s in enumerate(self.oscillations):
			if i>0: sys.exit('*********************************************************\n** You have selected multiple oscillation scenarios. ****\n** Please restric to a SINGLE scenario which contains ***\n** all the parameters. **********************************\n*********************************************************')
			print(' + ',s)
		self.oscillations = self.oscillations[0]
		print('====================================')

	def CheckSources(self):
		sources2 = []
		for i in self.Experiments.keys():
			for j in self.Experiments[i].keys():
				sources2.append(j)

		if collections.Counter(self.sources) == collections.Counter(sources2):
			print('You have specified the following files for each experiment and source:')
			for i in self.Experiments.keys():
				for j in self.Experiments[i].keys():
					print(' - MC files for '+ j + ' in ' + i)	
					print('   + ' + str(self.Experiments[i][j]['MCFiles']))
			for i in self.Experiments.keys():
				for j in self.Experiments[i].keys():
					print(' - Data files for '+ j + ' in ' + i)	
					print('   + ' + str(self.Experiments[i][j]['DataFiles']))
		else: 
			sys.exit('You are missing some files for some sources or experiments. Please, check your xml file.')

	def readOscPar(self):
		# 3-Osc, 3+N-Osc, LV, NSI
		params_3f = ['Sin2Theta12','Sin2Theta13','Sin2Theta23','Dm221','Dm231','dCP','Ordering']
		self.parameters = keys
		self.OscParametersGrid  = {}
		self.OscParametersEdges = {}
		self.OscParametersTrue  = {}
		for node in self.root.iter('NeutrinoPhysics'):
			for phys in self.physics:
				if node.attrib['name'] == phys:
					self.neutrinos = int(node.find('flavours').text)
					for key in keys:
						par = node.find('parameters/'+key)
						self.OscParametersGrid[key] = self.readOscParValues(par, key)
						self.OscParametersEdges[key] = self.readOscParNominal(par, key)
						self.OscParametersTrue[key] = self.readOscParTrueValue(par, key)

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

	def readOscParTrueValue(self, pars, key):
		if key=='Ordering':
			if 'normal' in str(pars.find('best').text).lower():
				return 'normal'
			else:
				return 'inverted'
		else:
			return float(pars.find('best').text)

	def readOscParNominal(self, pars, key):
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

	def CheckNuisance(self):
		nonuis = 0
		print('List of Nuisance')
		for source in self.Nuisance:
			print(f' + From {source}: {self.Nuisance[source]}')
			if len(self.NuisSigma[source]) > 0:
				nonuis = nonuis + 1
		print('====================================')
		if nonuis==0:
			self.NoNuis = 1
		else:
			self.NoNuis = 0

	