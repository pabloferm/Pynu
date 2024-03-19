import xml.etree.ElementTree as ET
import sys
import collections
import numpy as np
import itertools


class ParseXML:
    """Class handling the xml input analysis file, reading all the items for the analysis and
    storing them to be used elsewhere.
    """

    def __init__(self, xmlfile='AnalysisFiles/test.xml', check=False):
        """Initiates class with input analysis file, declares the necessary lists and dicts.

        Args:
            xmlfile (str): Name of the xml analysis input file.
            check (bool): Optional. Checks consistency of the analysis.
        """

        self.check = check
        self.tree = ET.parse(xmlfile)  # create element tree object
        self.root = self.tree.getroot()  # get root element of XML file

        # Declare lists and dicts for Nuisance parameters
        self.NuisanceList = []
        self.Nuisance = {}
        self.NuisNominal = {}
        self.NuisNominalList = []
        self.NuisSigma = {}
        self.NuisSigmaList = []
        self.NuisDistribution = {}
        self.NuisDistributionList = []

        # Declare lists and dicts for Physics parameters
        self.PhysicsList = []
        self.Physics = {}
        self.PhysTrue = {}
        self.PhysTrueList = []
        self.PhysGrid = {}
        self.PhysGridList = []
        self.PhysPoints = {}
        self.PhysPointsList = []
        self.PhysEdges = {}

        # Declare lists and dicts for Fixed parameters
        self.FixedList = []
        self.Fixed = {}
        self.FixedValue = {}
        self.FixedValueList = []

        # Declare lists and dicts for Experiments details and files
        self.MCFiles = {}
        self.MCyears = {}
        self.DataFiles = {}
        self.Exposure = {}
        self.Experiments = {}
        self.ExpTarget = {}

        self.n_sphere = False
        self.n_sphere_cut = None

    def set_spherical_grid(self, radius=1):
        """Calls the n-dimensional sphere cut over the analysis grid of points.

        Args:
            float (optional): relative radius of the elipse in each dimension

        Returns:
            None
        """
        self.n_sphere = True
        self.n_sphere_cut = self.apply_n_sphere(radius=radius)
        print(f"****************************************************************")
        print(
            f"**** Your analysis grid has gone from {self.NumberOfPhysPoints} to {np.sum(self.n_sphere_cut)} points. ****")
        print(f"****************************************************************")

    def do_point(self, point):
        """Checks whether a point has to be analized or not.

        Args:
            int: index of point in the analysis

        Returns:
            bool
        """
        if self.n_sphere and self.n_sphere_cut is not None:
            return self.n_sphere_cut[point]
        return True

    def get_analysis(self):
        """Sets all analysis variables, that is all the sources, targets, detectors and oscillation
        parameters of the given analysis. It also computes the number of nuisance and physics parameters,
        in case it is useful at some point.
        Further, if the option 'check' is `True`, it performs consistency checks to the analysis file.

        Args:
            None

        Returns:
            None
        """
        self.read_sources()
        self.read_detectors()
        self.read_experiments()
        self.read_oscillations()
        # self.read_method_parameters()
        # dict of arrays with all physics points for each parameter
        self.PhysGrid = self.physics_grid()
        # cartesian product of the previous arrays
        self.FullPhysicsGrid = self.cartesian_physics_grid()
        self.n_sphere = True

        self.wSyst = self.with_nuisance()  # True if analysis with nuisance

        # number of physics parameters.
        self.NumberOfPhys = len(self.PhysicsList)
        # number of physics points.
        self.NumberOfPhysPoints = np.prod(self.PhysPointsList)
        # number of nuisance parameters.
        self.NumberOfNuis = len(self.NuisanceList)

        # Optional checks
        if self.check:
            self.check_nuisance()
            self.check_physics()
            self.check_fixed()
            self.check_sources()

        del self.root

    def apply_n_sphere(self, radius=1):
        """Removes the corners of the n-dimensional grid of physics point to be probed,
        accepting only those points inside a n-dimensional elipse.
        This is useful to reduce the number of points to be evaluated during grid search
        as the expected best fit values are suposed to be towards the center of the grid.
        NOTICE: Please use it wisely.

        Args:
            float (optional): relative radius of the elipse in each dimension

        Returns:
            Numpy array of bools
        """
        if "Ordering" in self.PhysicsList:
            _means = np.zeros(self.NumberOfPhys - 1)
            _radii = np.zeros(self.NumberOfPhys - 1)
        else:
            _means = np.zeros(self.NumberOfPhys)
            _radii = np.zeros(self.NumberOfPhys)
        kk = 0
        for par in self.Physics:
            for j, item in enumerate(self.Physics[par]):
                print(par, item)
                if item == 'Ordering':
                    pass
                else:
                    _means[kk] = 0.5 * (self.PhysEdges[par]
                                        [j][0] + self.PhysEdges[par][j][-1])
                    _radii[kk] = - _means[kk] + self.PhysEdges[par][j][-1]
                    kk += 1
        norm_grid = (self.FullPhysicsGrid - _means) / _radii
        return np.sum(norm_grid**2, axis=1) <= radius

    def with_nuisance(self):
        """Checks if the analysis contains nuisance parameters or it's stats. only.

        Args:
            None

        Returns:
            Bool
        """
        if self.NuisanceList:
            return True
        return False

    def read_sources(self):
        """Reads the neutrino source to be included in the analysis

        Args:
            None

        Returns:
            Bool
        """
        self.sources = self.reader('NeutrinoSource')
        print('------------------------------------')
        print('Neutrino sources considered:')
        for s in self.sources:
            print(' + ', s)
        print('====================================')

    def read_detectors(self):
        """Reads the neutrino targets from each detector to be included in the analysis

        Args:
            None

        Returns:
            Bool
        """
        self.targets = self.reader('NeutrinoTarget')
        print('------------------------------------')
        print('Neutrino targets considered:')
        for s in self.targets:
            print(' + ', s)
        print('====================================')

    def read_experiments(self):
        """Reads the neutrino experiments to be included in the analysis associating detectors
        with neutrino sources.

        Args:
            None

        Returns:
            Bool
        """
        self.detectors = self.reader('NeutrinoExperiment')
        print('------------------------------------')
        print('Detectors considered:')
        for s in self.detectors:
            print(' + ', s)
        print('====================================')
        # print(f' + {self.Nuisance}')

    def read_oscillations(self):
        """Reads the neutrino oscillation scenario and parameters of the analysis.

        Args:
            None

        Returns:
            Bool
        """
        self.oscillations = self.reader('NeutrinoOscillations')
        print('------------------------------------')
        print('Oscillation scenario:')
        for i, s in enumerate(self.oscillations):
            if i > 0:
                sys.exit('*********************************************************\n** You have selected multiple oscillation scenarios. ****\n** Please restric to a SINGLE scenario which contains ***\n** all the parameters. **********************************\n*********************************************************')
            print(' + ', s)
        self.SCENARIO = self.oscillations[0]
        print('====================================')

    def check_sources(self):
        """Checks that the neutrino sources declared are the same of the experiments.

        Args:
            None

        Returns:
            None. Raises error if finds a mismatch.
        """
        sources2 = []
        for i in self.Experiments.keys():
            for j in self.Experiments[i].keys():
                sources2.append(j)
        if collections.Counter(self.sources) == collections.Counter(sources2):
            print(
                'You have specified the following files for each experiment and source:')
            for i in self.Experiments.keys():
                for j in self.Experiments[i].keys():
                    print(' - MC files for ' + j + ' in ' + i)
                    print('   + ' + str(self.Experiments[i][j]['MCFiles']))
            for i in self.Experiments.keys():
                for j in self.Experiments[i].keys():
                    print(' - Data files for ' + j + ' in ' + i)
                    print('   + ' + str(self.Experiments[i][j]['DataFiles']))
        else:
            sys.exit(
                'You are missing some files for some sources or experiments. Please, check your xml file.')
        print('====================================')

    def physics_grid(self):
        """Builds the arrays with all the physics points to be sampled from in the analysis.

        Args:
            None

        Returns:
            Dict with the structure {parameter (str): numpy.array}
        """
        phys_grid = {}
        for par in self.Physics:
            phys_grid[par] = {}
            for j, item in enumerate(self.Physics[par]):
                if item == 'Ordering':
                    phys_grid[par][item] = np.array(self.PhysEdges[par][j])
                elif item[-1].isdigit():
                    if int(item[-1]) > 3:
                        phys_grid[par][item] = np.geomspace(
                            self.PhysEdges[par][j][0],
                            self.PhysEdges[par][j][-1],
                            self.PhysPoints[par][j])
                    else:
                        phys_grid[par][item] = np.linspace(
                            self.PhysEdges[par][j][0],
                            self.PhysEdges[par][j][-1],
                            self.PhysPoints[par][j])
                else:
                    phys_grid[par][item] = np.linspace(
                        self.PhysEdges[par][j][0],
                        self.PhysEdges[par][j][-1],
                        self.PhysPoints[par][j])
        return phys_grid

    def cartesian_physics_grid(self):
        """Builds the grid of physics points to be sampled from in the analysis. It is computed as the cartesian
        product of the 'self.PhysGrid' arrays and ordered in the same way.
        the

        Args:
            None

        Returns:
            List of lists with all the pysics points.
        """
        v = []
        for par in self.PhysGrid.keys():
            for item, values in self.PhysGrid[par].items():
                v.append(values)
        return [*itertools.product(*v)]

    def spherical_physics_grid(self):
        pass

    def check_nuisance(self):
        """Prints each nuisance parameter to be checked by user.

        Args:
            None

        Returns:
            None
        """
        print('List of Nuisance')
        for source in self.Nuisance:
            print(f' + From {source}: {self.Nuisance[source]}')
        print('====================================')

    def check_physics(self):
        """Prints each physics parameter to be checked by user.

        Args:
            None

        Returns:
            None. Exits if there are no physics parameters.
        """
        print('List of Physics/Fit')
        for source in self.Physics:
            print(f' + From {source}: {self.Physics[source]}')
        if len(self.PhysicsList) == 0:
            sys.exit('I am done, you requested nothing to fit.')
        print('====================================')

    def check_fixed(self):
        """Prints each fixec parameter to be checked by user.

        Args:
            None

        Returns:
            None
        """
        print('List of Fixed')
        for source in self.Fixed:
            print(f' + From {source}: {self.Fixed[source]}')
        print('====================================')

    def get_nominal_values(self, keyw):
        """Returns the values (if fixed), nominal values (if nuisance) or true values (if physics)
        of the input parameter.

        Args:
            keyw (str): Name of the parameter.

        Returns:
            Float: Value of the parameter.
        """
        values = self.FixedValue[keyw] | self.NuisNominal[keyw] | self.PhysTrue[keyw]
        return values

    def get_tune(self, tune):
        """Returns the source or Physics Tunes block of a given parameter/tune.

        Args:
            tune (str): Name  of the parameter or tune.

        Returns:
            Str of the block of the given tune.
        """
        for all_parameters in [self.Nuisance, self.Physics, self.Fixed]:
            for source, pars in all_parameters.items():
                if tune in pars:
                    return source

    def reader(self, item, atrib='name'):
        """Main general method for reading a given item or block from the analysis xml file. It adds the
        read information into the class variables classifying each of the tunes as fixed, nuisance or
        physics as stated in the xml file.

        Args:
            item (str): It provides the block to be read. 'NeutrinoSource', 'NeutrinoTarget', etc.
            atrib (str, optional): By default this variable is set to 'name' as it is the most common use
            to read a whole block. However, further functionality is provided to read a given tune or
            parameter in a block.

        Returns:
            None if atrib == 'name' and a list of items in the rest of the cases.
        """
        itemList = []
        for source in self.root.iter(item):
            if atrib == 'name':
                if int(source.find('status').text):
                    sname = source.attrib['name']
                    itemList.append(source.attrib[atrib])
                    if item == 'NeutrinoOscillations':
                        self.Flavors = int(source.find('flavors').text)
                        print(self.Flavors)
                    elif item == 'NeutrinoExperiment':
                        self.Experiments[sname] = {}
                        self.ExpTarget[sname] = source.find(
                            'target').attrib['name']
                        for src in source.findall('source'):
                            if int(src.find('status').text):
                                self.Experiments[sname][src.attrib['name']] = {
                                }
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
                        self.Experiments[sname][src.attrib['name']] = {
                            'MCFiles': MCFiles, 'TotalMCexposure': MCyears,
                            'DataFiles': DataFiles, 'Exposure': Exposure}
                    self.Nuisance[sname] = []
                    self.NuisSigma[sname] = {}
                    self.NuisNominal[sname] = {}
                    self.NuisDistribution[sname] = {}
                    for nuis in source.findall('nuisance'):
                        if int(nuis.find('status').text):
                            s = nuis.attrib['name']
                            if s == 'Ordering':
                                sys.exit(
                                    'Neutrino mass ordering cannot be a nuisance parameter. Please, test both ordering hypotheses.')
                            else:
                                self.NuisSigma[sname][s] = float(
                                    nuis.find('sigma').text)
                                self.NuisSigmaList.append(
                                    float(nuis.find('sigma').text))
                                self.NuisNominal[sname][s] = float(
                                    nuis.find('nominal').text)
                                self.NuisNominalList.append(
                                    float(nuis.find('nominal').text))
                                self.NuisDistribution[sname][s] = str(
                                    nuis.find('distribution').text).strip()
                                self.NuisDistributionList.append(
                                    str(nuis.find('distribution').text).strip())
                                self.Nuisance[sname].append(s)
                                self.NuisanceList.append(s)
                    self.Fixed[sname] = []
                    self.FixedValue[sname] = {}
                    for fix in source.findall('fixed'):
                        if int(fix.find('status').text):
                            s = fix.attrib['name']
                            if s == 'Ordering':
                                self.FixedValue[sname][s] = fix.find(
                                    'value').text
                                self.FixedValueList.append(
                                    fix.find('value').text)
                                self.Fixed[sname].append(s)
                                self.FixedList = np.append(self.FixedList, s)
                            else:
                                self.FixedValue[sname][s] = float(
                                    fix.find('value').text)
                                self.FixedValueList.append(
                                    float(fix.find('value').text))
                                self.Fixed[sname].append(s)
                                self.FixedList.append(s)
                    self.Physics[sname] = []
                    self.PhysTrue[sname] = {}
                    self.PhysPoints[sname] = []
                    self.PhysEdges[sname] = []
                    for phys in source.findall('physics'):
                        s = phys.attrib['name']
                        if s == 'Ordering':
                            points = int(phys.find('points').text)
                            no = 0
                            io = 0
                            if 'norm' in phys.find(
                                    'min').text or 'norm' in phys.find('max').text:
                                no = 1
                            if 'inv' in phys.find(
                                    'min').text or 'inv' in phys.find('max').text:
                                io = 1
                            if io + no == 2 and points == 2:
                                self.PhysTrue[sname][s] = phys.find(
                                    'true').text
                                self.PhysTrueList.append(
                                    phys.find('true').text)
                                self.PhysPoints[sname].append(2)
                                self.PhysPointsList.append(2)
                                self.PhysEdges[sname].append(
                                    ['normal', 'inverted'])
                                self.Physics[sname].append(s)
                                self.PhysicsList.append(s)
                            elif io + no == 1 and points == 1:
                                if io:
                                    self.FixedValue[sname][s] = 'inverted'
                                    self.FixedValueList.append('inverted')
                                    self.Fixed[sname].append(s)
                                    self.FixedList.append(s)
                                elif no:
                                    self.FixedValue[sname][s] = 'normal'
                                    self.FixedValueList.append('normal')
                                    self.Fixed[sname].append(s)
                                    self.FixedList.append(s)
                                print(
                                    'Notice: Parameter ' +
                                    str(s) +
                                    ' has been moved to fixed.')
                            elif io + no == 0 or points == 0:
                                sys.exit(
                                    'Please, specify a neutrino mass ordering')
                            else:
                                sys.exit(
                                    'Please, take a look to the Ordering, something is not well defined.')
                        else:
                            if float(
                                    phys.find('min').text) == float(
                                    phys.find('max').text) or int(
                                    phys.find('points').text) <= 1:  # this parameter should be fixed
                                self.FixedValue[sname][s] = float(
                                    phys.find('true').text)
                                self.FixedValueList.append(
                                    float(phys.find('true').text))
                                self.Fixed[sname].append(s)
                                self.FixedList.append(s)
                                print(
                                    'Notice: Parameter ' +
                                    str(s) +
                                    ' has been moved to fixed.')
                            elif float(phys.find('min').text) == float(phys.find('max').text):
                                sys.exit('Please, check parameter ' + str(s))
                            else:
                                self.PhysTrue[sname][s] = float(
                                    phys.find('true').text)
                                self.PhysTrueList.append(
                                    float(phys.find('true').text))
                                self.PhysPoints[sname].append(
                                    int(phys.find('points').text))
                                self.PhysPointsList.append(
                                    int(phys.find('points').text))
                                self.PhysEdges[sname].append([
                                    float(phys.find('min').text),
                                    float(phys.find('max').text)])
                                self.Physics[sname].append(s)
                                self.PhysicsList.append(s)
            else:
                itemList.append(source.attrib[atrib])
        return itemList
