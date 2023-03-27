import AnalysisReader as AR  # contains parse class to read and setup the analysis
import Experiments as Exp  # contains rd class to read and setup each experiment
# contains everything to modify your simulations to help figuring out what
# you have measured
from PhysicsTunes.PhysicsTunes import PhysicsTunes as PT
import Fitter as FT  # does all the fitting calculations

import h5py
import numpy as np
from scipy.optimize import minimize


class PyNu:
    ''' Top class containing everything '''

    def __init__(self, analysis_file, verbosity=False):

        __slots__ = (
            'verbosity',
            'Analysis',
            'PhysicsTunes',
            'Experiments',
            'Observation')

        self.verbosity = verbosity

        ''' Set up basic analysis variables and structure to build full analysis '''
        self.Analysis = AR.parse(analysis_file, check=self.verbosity)

        ''' Define dictionary for PhysicsTunes '''
        self.PhysicsTunes = {}

        ''' Start the analysis '''
        self.SetUpExperiments()
        self.SetUpPhysicsTunes()

        ''' Compute Observation '''
        self.ComputeBinnedObservation()

    def ComputeBinnedObservation(self):
        self.ApplyFixedWeights()
        self.ApplyNominalWeights()
        self.ApplyTrueWeights()
        self.ApplyOscillations('Nominal')
        self.SetBinnedObservedEvents()
        # print(self.Observation)

    def ComputeBinnedExpectation(self, point, nuisance_vector=None, physics=False):
        if physics:
            self.StartPhysics()
            self.ApplyPhysicsWeights(point)
            if not self.Analysis.Nuisance[self.Analysis.Scenario]:
                self.ApplyOscillations('Physics')

        self.StartNuisance()
        if nuisance_vector is None:
            self.ApplyNuisanceWeights(self.Analysis.NuisNominalList)
        else:
            self.ApplyNuisanceWeights(nuisance_vector)
        if not self.Analysis.Nuisance[self.Analysis.Scenario]:
            self.ApplyOscillations('Nuisance')         

        self.SetBinnedExpectedEvents()

    def ComputeBinnedDiffExpectation(self, nuisance_vector=None):
        if nuisance_vector is None:
            nuisance_vector = self.Analysis.NuisNominalList
        dW_W = self.GetDiffLogWeights(nuisance_vector)
        self.DiffExpectation = self.SetBinnedDiffExpectedEvents(dW_W)

    def SetUpExperiments(self):
        ''' Loop over experiments specified in analysis file and store each of them
        into a dictionary with keys 'detector_source' (e.g. HyperK+Atmospheric) '''
        ''' Provides a dict of all experiments '''
        experiment = {}
        for det in self.Analysis.Experiments.keys():
            for src in self.Analysis.Experiments[det].keys():
                details = self.Analysis.Experiments[det][src]
                exp = det + '+' + src
                experiment[exp] = Exp.Manager(
                    det, src, details, self.Analysis.Scenario)
        self.Experiments = experiment

    def SetUpPhysicsTunes(self):
        ''' Loop over physics tunes specified in analysis file and store each of them
        into a dictionary with keys 'detector+source' (e.g. HyperK+Atmospheric) '''
        for name, exp in self.Experiments.items():
            self.PhysicsTunes[name] = PT(
                exp,
                self.Analysis.Scenario,
                self.Analysis.Flavors,
                set_all=True)

    def StartPhysics(self):
        for exp in self.Experiments.values():
            exp.StartPhysicsWeights()
    def StartNuisance(self):
        for exp in self.Experiments.values():
            exp.StartNuisanceWeights()

    def SetBinnedObservedEvents(self):
        self.Observation = {}
        for name, exp in self.Experiments.items():
            exp.SetObservedBinned()
            self.Observation[name] = exp.GetObservedBinned()

    def SetBinnedExpectedEvents(self):
        self.Expectation = {}
        for name, exp in self.Experiments.items():
            exp.SetExpectedBinned()
            self.Expectation[name] = exp.GetExpectedBinned()

    def SetBinnedDiffExpectedEvents(self, dW_W):
        dEdx = {}
        for nuis, experiments in dW_W.items():
            for exp, weights in experiments.items():
                # Make it easier !!!
                dEdx[nuis] = {
                    exp: self.Experiments[exp].BinMC(
                        weights *
                        self.Experiments[exp].ExpectedWeight)[
                        self.Experiments[exp].FewEntries]}

        return dEdx

    def ApplyFixedWeights(self):  # Nuisance parameters
        if self.verbosity:
            print('Applying Fixed Weights')
        self.ApplyWeights('Fixed')

    def ApplyNominalWeights(self):  # Nuisance parameters
        if self.verbosity:
            print('Applying Nominal Nuisance Weights')
        self.ApplyWeights('Nominal')

    def ApplyTrueWeights(self):  # Physics parameters
        if self.verbosity:
            print('Applying Physics True Weights')
        self.ApplyWeights('True')

    def ApplyPhysicsWeights(self, point):  # Physics parameters
        if self.verbosity:
            print('Applying Physics Point Weights')
        self.ApplyWeights(
            'Physics',
            vector=self.Analysis.FullPhysicsGrid[point])

    def ApplyNuisanceWeights(self, vector):  # Physics parameters
        if self.verbosity:
            print('Applying Nuisance Weights')
        self.ApplyWeights('Nuisance', vector=vector)

    # Tag can be either 'Nominal' or 'Variable'
    def ApplyOscillations(self, tag=None):
        for name, exp in self.Experiments.items():
            w = self.PhysicsTunes[name].OscillationTunes.GetOscillations()
            if tag == 'Physics':
                exp.UpdatePhysicsWeights(w)
            elif tag == 'Nuisance':
                exp.UpdateNuisanceWeights(w)
            elif tag == 'Nominal':
                if not self.Analysis.Nuisance[self.Analysis.Scenario] and not self.Analysis.Physics[self.Analysis.Scenario]:
                    exp.UpdateBaseWeights(w)
                else:
                    exp.UpdateNominalWeights(w)


    def ApplyWeights(self, tag, vector=None):
        if tag == 'Fixed':
            labels = self.Analysis.Fixed
            vec = self.Analysis.FixedValue
        elif tag == 'Nominal':
            labels = self.Analysis.Nuisance
            vec = self.Analysis.NuisNominal
        elif tag == 'True':
            labels = self.Analysis.Physics
            vec = self.Analysis.PhysTrue
        elif tag == 'Physics':
            labels = self.Analysis.Physics
            v_id = self.Analysis.PhysicsList
        elif tag == 'Nuisance':
            labels = self.Analysis.Nuisance
            v_id = self.Analysis.NuisanceList
        else:
            sys.exit('Not a valid tag for applying weights.')

        for name, exp in self.Experiments.items():
            for source in labels:
                if source in exp.Definition.keys():
                    tune_block = exp.Definition[source]
                    for tune in labels[source]:
                        if vector is not None:
                            idx = v_id.index(tune)
                            value = vector[idx]
                        else:
                            value = vec[source][tune]

                        if tune_block == 'Flux':
                            w = self.PhysicsTunes[name].GetFlux(tune, value)
                        elif tune_block == 'XSection':
                            w = self.PhysicsTunes[name].GetXSection(
                                tune, value)
                        elif tune_block == 'Detector':
                            w = self.PhysicsTunes[name].GetDetector(
                                tune, value)
                        elif tune_block == 'Osc':
                            self.PhysicsTunes[name].OscillationTunes.UpdateParameter(
                                tune, value)

                        if tune_block != 'Osc':
                            if tag == 'Fixed':
                                exp.UpdateBaseWeights(w)
                            elif tag in ['True', 'Nominal']:
                                exp.UpdateNominalWeights(w)
                            elif tag == 'Physics':
                                exp.UpdatePhysicsWeights(w)
                            elif tag == 'Nuisance':
                                exp.UpdateNuisanceWeights(w)


    def GetDiffLogWeights(self, vector):
        ''' Computes the derivative with respect the nuisance parameter nuis '''
        ''' Returns a dict of nuis : experiment : partial of weight with respect to nuis over weight '''
        dWoverW = {}

        for source, nuisance_list in self.Analysis.Nuisance.items():
            for name, exp in self.Experiments.items():
                if source in exp.Definition.keys():
                    tune_block = exp.Definition[source]
                    for tune in self.Analysis.Nuisance[source]:
                        dWoverW[tune] = {name: 0}
                        idx = self.Analysis.NuisanceList.index(tune)
                        if tune_block == 'Flux':
                            dWoverW[tune][name] = self.PhysicsTunes[name].GetFlux(
                                'Diff_' + tune, vector[idx]) / self.PhysicsTunes[name].GetFlux(tune, vector[idx])
                        elif tune_block == 'XSection':
                            dWoverW[tune][name] = self.PhysicsTunes[name].GetXSection(
                                'Diff_' + tune, vector[idx]) / self.PhysicsTunes[name].GetXSection(tune, vector[idx])
                        elif tune_block == 'Detector':
                            dWoverW[tune][name] = self.PhysicsTunes[name].GetDetector(
                                'Diff_' + tune, vector[idx]) / self.PhysicsTunes[name].GetDetector(tune, vector[idx])
                        elif tune_block == 'Osc':
                            dWoverW[tune][name] = self.PhysicsTunes[name].GetOscillation(
                                'Diff_' + tune, vector[idx]) / self.PhysicsTunes[name].OscillationTunes.GetOscillations()
        return dWoverW

    def CreateOutFile(self, fname):
        self.outfile = fname
        with h5py.File(fname, 'w') as hf:
            grp = hf.create_group('Fixed Parameters')
            for key in self.Analysis.Fixed.keys():
                this = grp.create_group(key)
                for par, val in self.Analysis.FixedValue[key].items():
                    this.create_dataset(par, data=[val], compression='gzip')

            if self.Analysis.wSyst:
                grp = hf.create_group('Nuisance Parameters')
                for key in self.Analysis.Nuisance.keys():
                    this = grp.create_group(key)
                    for par in self.Analysis.Nuisance[key]:
                        this.create_dataset(
                            par,
                            data=[0.0] *
                            self.Analysis.NumberOfPhysPoints,
                            compression='gzip')
            grp = hf.create_group('Physics Parameters')

            i = 0
            for key in self.Analysis.Physics.keys():
                this = grp.create_group(key)
                for par in self.Analysis.Physics[key]:
                    # this.create_dataset(par, data=[0.0]*self.Analysis.NumberOfPhysPoints, compression='gzip')
                    this.create_dataset(
                        par, data=self.Analysis.FullPhysicsGrid[:][i],
                        compression='gzip')
                    i = + 1

            grp = hf.create_group('Analysis')
            grp.create_dataset(
                'Chi2 Stats. Only',
                data=[0.0] *
                self.Analysis.NumberOfPhysPoints,
                compression='gzip')
            if self.Analysis.wSyst:
                grp.create_dataset(
                    'Chi2 Systs.',
                    data=[0.0] *
                    self.Analysis.NumberOfPhysPoints,
                    compression='gzip')

    def WriteToOutFile(self, point, block, item, value):
        with h5py.File(self.outfile, 'r+') as hf:
            try:
                for par, val in zip(item, value):
                    source = self.Analysis.GetSourceOfTune(par)
                    hf[block + '/' + source + '/' + par][point] = val
            except BaseException:
                hf[block + '/' + item][point] = value

    def FitBinnedLLH(self, point):
        ''' Binned log-Likelihood fit assuming data is Poisson-distributed '''
        self.ComputeBinnedExpectation(point, physics=True)  # Nominal expectation
        # Statistics only computation to start guiding the minimization
        X2_stats = FT.ChiSquaredStatsOnly(self.Observation, self.Expectation)
        self.WriteToOutFile(point, 'Analysis', 'Chi2 Stats. Only', X2_stats)

        '''Get Jacobian of expected events w.r.t. nuisance parameters'''
        self.ComputeBinnedDiffExpectation()

        '''Analytic estimate for priors and bounds'''
        AnalyticPrior, AnalyticBounds = FT.AnalyticPriorsBounds(
            self.Observation, self.Expectation, self.DiffExpectation, self.
            Analysis.NuisNominalList, self.Analysis.NuisSigmaList)

        '''Combined chi^2 minimization'''
        tol = max(1e-4, np.sqrt(X2_stats) * 1e-5)
        res = minimize(
            self.ModelTester,
            AnalyticPrior,
            args=(point),
            # method='L-BFGS-B',
            jac=True,
            bounds=AnalyticBounds,
            options={
                'disp': False})

        nuisance_postfit = res.x.tolist()
        self.WriteToOutFile(
            point,
            'Nuisance Parameters',
            self.Analysis.NuisanceList,
            nuisance_postfit)

        X2_systs = res.fun
        self.WriteToOutFile(point, 'Analysis', 'Chi2 Systs.', X2_systs)

        return - 0.5 * X2_systs

    def ModelTester(self, nuisance_vector, point):
        ''' Compute expected and its derivatives '''
        self.ComputeBinnedExpectation(
            point, nuisance_vector=nuisance_vector)  # Nominal expectation
        self.ComputeBinnedDiffExpectation(nuisance_vector=nuisance_vector)

        ''' Get -2 ln(H/H0) ~ χ2 '''
        Chi2 = FT.ChiSquared(
            self.Observation,
            self.Expectation,
            self.Analysis.NuisNominalList,
            self.Analysis.NuisSigmaList,
            self.Analysis.NuisDistributionList,
            nuisance_vector)

        ''' The gradient of the above '''
        D_Chi2 = FT.ChiSquaredGradient(
            self.Observation,
            self.Expectation,
            self.DiffExpectation,
            self.Analysis.NuisNominalList,
            self.Analysis.NuisSigmaList,
            self.Analysis.NuisDistributionList,
            nuisance_vector)

        return (Chi2, D_Chi2)
