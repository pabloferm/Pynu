import xml.etree.ElementTree as ET
import sys
import collections
import numpy as np
import itertools


class ParseXML:
    """Class handling the xml input analysis file, reading all the items for the analysis and
    storing them to be used elsewhere.
    """

    def __init__(self, xmlfile='AnalysisFiles/test.xml', check=False) -> None:
        """Initiates class with input analysis file, declares the necessary lists and dicts.

        Args:
            xmlfile (str): Name of the xml analysis input file.
            check (bool): Optional. Checks consistency of the analysis.
        """

        self.check = check
        self.tree = ET.parse(xmlfile)  # create element tree object
        self.root = self.tree.getroot()  # get root element of XML file

        """Profiling or marginalization, important to know how to treat analysis parameters:
        - Profiling: there are physics parameters which form a grid to sample and systematics 
        which are minimize over for each grid point.
        - Marginalization: No grids, no difference between physics and nuisance in the analysis.
        Fixed parameters are treated equally.
        """
        # Maybe we don't need the previous, it can be handled when calling the fitter.
        self.profiling = False
        # Read description of the analysis method from the XML file
        self.analysis_method()

        if self.profiling:
            print("We're profiling")
        else:
            print("We're marginalizing")

        # Declare lists and dicts for Physics parameters
        self.PhysicsList = []
        self.Physics = {}
        self.PhysTrue = {}
        self.PhysTrueList = []
        
        self.PhysPoints = {}
        self.PhysPointsList = []
        self.PhysEdges = {}

        # INCLUDE PRIORS TO PHYSICS PARAMETERS
        self.PhysNominal = {}
        self.PhysNominalList = []
        self.PhysSigma = {}
        self.PhysSigmaList = []
        self.PhysDistribution = {}
        self.PhysDistributionList = []

        # Declare lists and dicts for Fixed parameters
        self.FixedList = []
        self.Fixed = {}
        self.FixedValue = {}
        self.FixedValueList = []

        # Declare lists and dicts for Nuisance parameters
        self.NuisanceList = []
        self.Nuisance = {}
        self.NuisNominal = {}
        self.NuisNominalList = []
        self.NuisSigma = {}
        self.NuisSigmaList = []
        self.NuisDistribution = {}
        self.NuisDistributionList = []

       # Declare lists and dicts for Experiments details and files
        self.MCFiles = {}
        self.MCyears = {}
        self.DataFiles = {}
        self.Exposure = {}
        self.Experiments = {}
        self.ExpTarget = {}


        if self.profiling:
            # If profiling, build the physics grid
            self.PhysGrid = {}
            self.PhysGridList = []

            # Check if we want the spherical grid
            self.n_sphere = False
            self.n_sphere_cut = None

    @staticmethod
    def _cast(value: str):
        """Convert an XML text value to int/float where possible, else leave as string."""
        value = (value or "").strip()
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    def _get_key(self, parent, tag):
        elem = parent.find(tag)
        return self._cast(elem.text) if elem is not None else None

    def analysis_method(self) -> None:
        """Reads the statistical method to be used in the analysis. Expects exactly one
        active <Method> block in the XML file; exits with an error if zero or more than
        one are found.
        """
        active_methods = []
        for method_elem in self.root.findall('.//Method'):
            status_elem = method_elem.find('status')
            if status_elem is not None and int(status_elem.text or "0"):
                active_methods.append(method_elem)

        if len(active_methods) == 0:
            sys.exit('No active Method found in the XML file. ')
        if len(active_methods) > 1:
            sys.exit('More than one active sMethods found in the XML file. ')

        method_elem = active_methods[0]
        name = method_elem.attrib.get('name')

        sampler = None
        minimizer = None
        params = {}

        for child in method_elem:
            if child.tag == 'sampler':
                sampler = child.attrib.get('name')
            elif child.tag == 'minimizer':
                minimizer = child.attrib.get('name')
            else:
                params[child.tag] = self._cast(child.text)

        if sampler is None: # Profiling
            self.method = {
                'name': name,
                'minimizer': minimizer,
                'params': params,
            }
            self.profiling =True
        elif minimizer is None: # Marginalization
            self.method = {
                'name': name,
                'sampler': sampler,
                'params': params,
            }
        else:
            sys.exit('No minimizer or sampler has been defined.')


        print('------------------------------------')
        print('Analysis method:')
        print(' + ', self.method)
        print('====================================')


    def set_spherical_grid(self, radius=1) -> None:
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

    def do_point(self, point) -> bool:
        """Checks whether a point has to be analized or not.

        Args:
            int: index of point in the analysis

        Returns:
            bool
        """
        if self.n_sphere and self.n_sphere_cut is not None:
            return self.n_sphere_cut[point]
        return True

    def get_analysis(self) -> None:
        """ Sets structure and analysis variables from XML file depending on 
        the type of method, profiling or marginalization.

        Args:
            None

        Returns:
            None
        """
        
        if self.profiling:
            self.get_profiling_analysis()
        else:
            self.get_marginalization_analysis()

    def get_marginalization_analysis(self) -> None:
        """

        Args:
            None

        Returns:
            None
        """
        self.read_sources()
        self.read_detectors()
        self.read_experiments()
        self.read_oscillations()

        # Remove remmanents from profiling grid
        del self.PhysPoints
        del self.PhysPointsList
        del self.PhysEdges

        # Merge Nuisance into Physics to ensure equal treatment in the analysis (marginalization)
        self.PhysicsList = []
        self.PhysTrueList = []
        self.PhysNominalList = []
        self.PhysSigmaList = []
        self.PhysDistributionList = []

        for key in self.Physics.keys():
            self.Physics[key] += self.Nuisance[key]
            self.PhysTrue[key].update(self.NuisNominal[key])
            self.PhysNominal[key].update(self.NuisNominal[key])
            self.PhysSigma[key].update(self.NuisSigma[key])
            self.PhysDistribution[key].update(self.NuisDistribution[key])

            self.PhysicsList += self.Physics[key]
            self.PhysTrueList += self.PhysTrue[key].values()
            self.PhysNominalList += self.PhysNominal[key].values()
            self.PhysSigmaList += self.PhysSigma[key].values()
            self.PhysDistributionList += self.PhysDistribution[key].values()

        # Remove nuisances to avoid double counting
        self.Nuisance = None
        del self.NuisanceList
        del self.NuisNominal
        del self.NuisNominalList
        del self.NuisSigma
        del self.NuisSigmaList
        del self.NuisDistribution
        del self.NuisDistributionList

    def get_profiling_analysis(self) -> None:
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

    def apply_n_sphere(self, radius=1) -> np.dtype[np.bool]:
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
                # if int(source.find('status').text):
                if self._get_key(source, "status"):
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
                                self.NuisSigma[sname][s] = self._get_key(
                                    nuis, 'sigma')
                                self.NuisSigmaList.append(
                                    self._get_key(nuis, 'sigma'))
                                self.NuisNominal[sname][s] = self._get_key(
                                    nuis, 'nominal')
                                self.NuisNominalList.append(
                                    self._get_key(nuis, 'nominal'))
                                self.NuisDistribution[sname][s] = self._get_key(
                                    nuis, 'distribution') # .strip()
                                self.NuisDistributionList.append(
                                    self._get_key(nuis, 'distribution'))
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
                                self.FixedValue[sname][s] = self._get_key(
                                    fix, 'value')
                                self.FixedValueList.append(
                                    self._get_key(fix, 'value'))
                                self.Fixed[sname].append(s)
                                self.FixedList.append(s)
                    self.Physics[sname] = []
                    self.PhysTrue[sname] = {}
                    self.PhysNominal[sname] = {}
                    self.PhysSigma[sname] = {}
                    self.PhysDistribution[sname] = {}
                    self.PhysPoints[sname] = []
                    self.PhysEdges[sname] = []
                    for phys in source.findall('physics'):
                        s = phys.attrib['name']
                        if s == 'Ordering':
                            points = self._get_key(phys, 'points')
                            no = 0
                            io = 0
                            if 'norm' in self._get_key(phys, 
                                    'min') or 'norm' in self._get_key(phys, 'max'):
                                no = 1
                            if 'inv' in self._get_key(phys, 
                                    'min') or 'inv' in self._get_key(phys, 'max'):
                                io = 1
                            if io + no == 2 and points == 2 or (io + no == 0 or points == 0): # If no MO is specified, assume both
                                self.PhysTrue[sname][s] = phys.find(
                                    'true').text
                                self.PhysTrueList.append(
                                    self._get_key(phys, 'true'))
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
                            else:
                                sys.exit(
                                    'Please, take a look to the Ordering, something is not well defined.')
                        else:
                            if self.profiling and (self._get_key(phys, 'min') == self._get_key(phys, 'max') or
                                    self._get_key(phys, 'points') <= 1):  # this parameter should be fixed
                                self.FixedValue[sname][s] = self._get_key(phys, 'true')
                                self.FixedValueList.append(
                                    self._get_key(phys, 'true'))
                                self.Fixed[sname].append(s)
                                self.FixedList.append(s)
                                print(
                                    'Notice: Parameter ' +
                                    str(s) +
                                    ' has been moved to fixed.')
                            else:
                                self.PhysTrue[sname][s] = self._get_key(phys, 'true')
                                self.PhysTrueList.append(self._get_key(phys, 'true'))
                                self.PhysPoints[sname].append(self._get_key(phys, 'points'))
                                self.PhysPointsList.append(self._get_key(phys, 'points'))
                                self.PhysEdges[sname].append([
                                    self._get_key(phys, 'min'),
                                    self._get_key(phys, 'max')])
                                self.PhysSigma[sname][s] = self._get_key(
                                    phys, 'sigma')
                                self.PhysSigmaList.append(
                                    self._get_key(phys, 'sigma'))
                                self.PhysNominal[sname][s] = self._get_key(
                                    phys, 'nominal')
                                self.PhysNominalList.append(
                                    self._get_key(phys, 'nominal'))
                                self.PhysDistribution[sname][s] = self._get_key(
                                    phys, 'distribution') # .strip()
                                self.PhysDistributionList.append(
                                    self._get_key(phys, 'distribution'))
                                self.Physics[sname].append(s)
                                self.PhysicsList.append(s)
            else:
                itemList.append(source.attrib[atrib])
        return itemList