import xml.etree.ElementTree as ET
import sys
import collections
import numpy as np
from .Distributions import Beta

class parseXML:
	def __init__(self, xmlfile='AnalysisFiles/test.xml', check=False):
		
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
		self.PhysPoints= {}
		self.PhysPointsList= []
		self.PhysEdges = {}

		# Fixed
		self.disabledFixed = []
		self.FixedList = np.array([])
		self.Fixed = {}
		self.FixedValue = {}
		self.FixedValueList = []

		# Experiments
		self.MCFiles = {}
		self.MCyears = {}
		self.DataFiles = {}
		self.Exposure = {}
		self.Experiments = {}
		self.ExpTarget = {}

		# Oscillation parameters
		self.OscParameters = []
		self.FluxParameters = []
		self.XSectionParameters = []
		self.DetectorParameters = []

		# Reading
		self.readSources()
		self.readDetectors()
		self.readExperiments()
		self.readOscillations()
		self.CheckSources()
		self.makePhysicsGrid()
		
		self.OscNominalParameters = self.GetNominalValues(self.OscScenario)

		self.wSyst = False
		if len(self.NuisanceList) > 0 : self.wSyst = True

		self.NumberOfPhys = len(self.PhysicsList)
		self.NumberOfPhysPoints = np.prod(self.PhysPointsList)
		self.NumberOfNuis = len(self.NuisanceList)

		# Optional checks
		if check:
			self.CheckNuisance()
			self.CheckPhysics()
			self.CheckFixed()

	def reader(self, item, atrib='name'):
		itemList = []
		osc = item=='NeutrinoOscillations'
		for source in self.root.iter(item):
			if atrib=='name':
				if int(source.find('status').text):
					sname = source.attrib['name']
					itemList.append(source.attrib[atrib])
					if item == 'NeutrinoOscillations':
						self.Flavors = int(source.find('flavors').text)
					elif item == 'NeutrinoExperiment':
						self.Experiments[sname] = {}
						self.ExpTarget[sname] = source.find('target').attrib['name']
						for src in source.findall('source'):
							if int(src.find('status').text):
								self.Experiments[sname][src.attrib['name']] = {}
								MCFiles = []
								DataFiles = []
								for fi in src.findall('MCFiles'):
									if int(fi.find('status').text):
										MCFiles.append(fi.attrib['name'])
								for fi in src.findall('DataFiles'):
									if int(fi.find('status').text):
										DataFiles.append(fi.attrib['name'])										
								Exposure = float(src.find('exposure').text)
								MCyears = float(src.find('MCexposure').text)
						self.Experiments[sname][src.attrib['name']] = {'MCFiles':MCFiles, 'TotalMCexposure': MCyears, 'DataFiles': DataFiles, 'Exposure': Exposure}
					self.Nuisance[sname] = []
					self.NuisSigma[sname] = {}
					self.NuisNominal[sname] = {}
					for nuis in source.findall('nuisance'):
						if int(nuis.find('status').text):
							s = nuis.attrib['name']
							if osc: self.OscParameters.append(s)
							if s == 'Ordering':
								sys.exit('Neutrino mass ordering cannot be a nuisance parameter. Please, test both ordering hypotheses.')
							else:
								self.NuisSigma[sname][s] = float(nuis.find('sigma').text)
								self.NuisSigmaList.append(float(nuis.find('sigma').text))
								self.NuisNominal[sname][s] = float(nuis.find('nominal').text)
								self.NuisNominalList.append(float(nuis.find('nominal').text))
								self.Nuisance[sname].append(s)
								self.NuisanceList = np.append(self.NuisanceList,s)
					self.Fixed[sname] = []
					self.FixedValue[sname] = {}
					for fix in source.findall('fixed'):
						if int(fix.find('status').text):
							s = fix.attrib['name']
							if osc: self.OscParameters.append(s)
							if s == 'Ordering':
								self.FixedValue[sname][s] = fix.find('value').text
								self.FixedValueList.append(fix.find('value').text)
								self.Fixed[sname].append(s)
								self.FixedList = np.append(self.FixedList,s)
							else:
								self.FixedValue[sname][s] = float(fix.find('value').text)
								self.FixedValueList.append(float(fix.find('value').text))
								self.Fixed[sname].append(s)
								self.FixedList = np.append(self.FixedList,s)
					self.Physics[sname] = []
					self.PhysTrue[sname] = {}
					self.PhysPoints[sname] = []
					self.PhysEdges[sname] = []
					for phys in source.findall('physics'):
						s = phys.attrib['name']
						if osc: self.OscParameters.append(s)
						if s == 'Ordering':
							points = int(phys.find('points').text)
							no = 0
							io = 0
							if 'norm' in phys.find('min').text or 'norm' in phys.find('max').text :
								no = 1
							if 'inv' in phys.find('min').text or 'inv' in phys.find('max').text :
								io = 1
							if io+no==2 and points==2:
								self.PhysTrue[sname][s] = phys.find('true').text
								self.PhysTrueList.append(phys.find('true').text)
								self.PhysPoints[sname].append(2)
								self.PhysPointsList.append(2)
								self.PhysEdges[sname].append(['normal','inverted'])
								self.Physics[sname].append(s)
								self.PhysicsList = np.append(self.PhysicsList,s)
							elif io+no==1 and points==1:
								if io:
									self.FixedValue[sname][s] = 'inverted'
									self.FixedValueList.append('inverted')
									self.Fixed[sname].append(s)
									self.FixedList = np.append(self.FixedList,s)
								elif no:
									self.FixedValue[sname][s] = 'normal'
									self.FixedValueList.append('normal')
									self.Fixed[sname].append(s)
									self.FixedList = np.append(self.FixedList,s)
								print('Notice: Parameter '+str(s)+' has been moved to fixed.')
							elif io+no==0 or points==0:
								sys.exit('Please, specify a neutrino mass ordering')
							else:
								sys.exit('Please, take a look to the Ordering, something is not well defined.')
						else:
							if float(phys.find('min').text)==float(phys.find('max').text) or int(phys.find('points').text)<=1: # this parameter should be fixed
								self.FixedValue[sname][s] = float(phys.find('true').text)
								self.FixedValueList.append(float(phys.find('true').text))
								self.Fixed[sname].append(s)
								self.FixedList = np.append(self.FixedList,s)
								print('Notice: Parameter '+str(s)+' has been moved to fixed.')
							elif float(phys.find('min').text) ==float(phys.find('max').text):
								sys.exit('Please, check parameter '+ str(s))
							else:
								self.PhysTrue[sname][s] = float(phys.find('true').text)
								self.PhysTrueList.append(float(phys.find('true').text))
								self.PhysPoints[sname].append(int(phys.find('points').text))
								self.PhysPointsList.append(int(phys.find('points').text))
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
		self.targets = self.reader('NeutrinoTarget')
		print('------------------------------------')
		print('Neutrino targets considered:')
		for s in self.targets:
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
		self.OscScenario = self.oscillations[0]
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
		print('====================================')

	def makePhysicsGrid(self):
		for par in self.Physics:
			self.PhysGrid[par] = {}
			for j,item in enumerate(self.Physics[par]):
				if item == 'Ordering':
					self.PhysGrid[par][item] = np.array(self.PhysEdges[par][j])
				else:
					self.PhysGrid[par][item] = np.linspace(self.PhysEdges[par][j][0],self.PhysEdges[par][j][-1],self.PhysPoints[par][j])

	def CheckNuisance(self):
		print('List of Nuisance')
		for source in self.Nuisance:
			print(f' + From {source}: {self.Nuisance[source]}')
		if len(self.NuisanceList) == 0:
			self.StatsOnly = True
			print('Stats. only analysis?!')
		else:
			self.StatsOnly = False
		print('====================================')

	def CheckPhysics(self):
		print('List of Physics/Fit')
		for source in self.Physics:
			print(f' + From {source}: {self.Physics[source]}')
		if len(self.PhysicsList) == 0:
			sys.exit('I am done, you requested nothing to fit.')
		print('====================================')

	def CheckFixed(self):
		print('List of Fixed')
		for source in self.Fixed:
			print(f' + From {source}: {self.Fixed[source]}')
		print('====================================')

	def GetNominalValues(self, keyw):
		values = self.FixedValue[keyw] | self.NuisNominal[keyw] | self.PhysTrue[keyw]
		return values
