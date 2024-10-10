import sys
import time
import fcntl

import numpy as np
from scipy.optimize import minimize

import h5py

import analysis_reader as ar  # contains parse class to read and setup the analysis
import Experiments as Exp  # contains rd class to read and setup each experiment
from PhysicsTunes.PhysicsTunes import (
    PhysicsTunes as PT,
)  # contains everything to modify your simulations
import fitter as ft  # does all the fitting calculations


class PyNuFit:
    """Top class containing everything"""

    def __init__(self, analysis_file, path=None, verbosity=False):
        self.verbosity = verbosity
        self.path = path

        """ Set up basic analysis variables and structure to build full analysis """
        self.Analysis = ar.ParseXML(analysis_file, check=self.verbosity)
        self.Analysis.get_analysis()

        """ Define dictionary for PhysicsTunes """
        self.physics_tunes = {}

        """ Start the analysis """
        self.SetUpExperiments()
        self.SetUpPhysicsTunes()

        """ Compute Observation """
        self.ComputeBinnedObservation()

    def ComputeBinnedObservation(self):
        self.ApplyFixedWeights()
        self.ApplyNominalWeights()
        self.ApplyTrueWeights()
        self.ApplyOscillations("Nominal")
        self.SetBinnedObservedEvents()

    def ComputeBinnedExpectation(self, point, nuisance_vector=None, physics=False):
        if physics:
            self.StartPhysics()
            self.ApplyPhysicsWeights(point)
            if (
                not self.Analysis.Nuisance[self.Analysis.SCENARIO]
                and self.Analysis.Physics[self.Analysis.SCENARIO]
            ):
                self.ApplyOscillations("Physics")

        self.StartNuisance()
        if nuisance_vector is None:
            self.ApplyNuisanceWeights(self.Analysis.NuisNominalList)
        else:
            self.ApplyNuisanceWeights(nuisance_vector)
        if self.Analysis.Nuisance[self.Analysis.SCENARIO]:
            self.ApplyOscillations("Nuisance")

        self.SetExpectedWeights()
        self.SetBinnedExpectedEvents()

    def ComputeBinnedDiffExpectation(self, nuisance_vector=None):
        if nuisance_vector is None:
            nuisance_vector = self.Analysis.NuisNominalList
        dW_W = self.GetDiffLogWeights(nuisance_vector)
        self.DiffExpectation = self.SetBinnedDiffExpectedEvents(dW_W)

    def SetUpExperiments(self):
        """Loop over experiments specified in analysis file and store each of them
        into a dictionary with keys 'detector_source' (e.g. HyperK+Atmospheric)"""
        """ Provides a dict of all experiments """
        experiment = {}
        for det in self.Analysis.Experiments.keys():
            for src in self.Analysis.Experiments[det].keys():
                details = self.Analysis.Experiments[det][src]
                exp = f"{det}+{src}"
                experiment[exp] = Exp.Manager(det, src, details, self.Analysis.SCENARIO)
        self.Experiments = experiment

    def SetUpPhysicsTunes(self):
        """Loop over physics tunes specified in analysis file and store each of them
        into a dictionary with keys 'detector+source' (e.g. HyperK+Atmospheric)"""
        for name, exp in self.Experiments.items():
            self.physics_tunes[name] = PT(
                exp, self.Analysis.SCENARIO, self.Analysis.Flavors, set_all=True
            )

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

    def SetExpectedWeights(self):
        for name, exp in self.Experiments.items():
            exp.SetExpectedWeight()

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
                        weights * self.Experiments[exp].ExpectedWeight
                    )[self.Experiments[exp].FewEntries]
                }
        return dEdx

    def ApplyFixedWeights(self):  # Nuisance parameters
        if self.verbosity:
            print("Applying Fixed Weights")
        self.ApplyWeights("Fixed")

    def ApplyNominalWeights(self):  # Nuisance parameters
        if self.verbosity:
            print("Applying Nominal Nuisance Weights")
        self.ApplyWeights("Nominal")

    def ApplyTrueWeights(self):  # Physics parameters
        if self.verbosity:
            print("Applying Physics True Weights")
        self.ApplyWeights("True")

    def ApplyPhysicsWeights(self, point):  # Physics parameters
        if self.verbosity:
            print("Applying Physics Point Weights")
        self.ApplyWeights("Physics", vector=self.Analysis.FullPhysicsGrid[point])

    def ApplyNuisanceWeights(self, vector):  # Physics parameters
        if self.verbosity:
            print("Applying Nuisance Weights")
        self.ApplyWeights("Nuisance", vector=vector)

    # Tag can be either 'Nominal' or 'Variable'
    def ApplyOscillations(self, tag=None):
        for name, exp in self.Experiments.items():
            w = self.physics_tunes[name].OscillationTunes.GetOscillations()
            if tag == "Physics":
                exp.UpdatePhysicsWeights(w)
            elif tag == "Nuisance":
                exp.UpdateNuisanceWeights(w)
            elif tag == "Nominal":
                if (
                    not self.Analysis.Nuisance[self.Analysis.SCENARIO]
                    and not self.Analysis.Physics[self.Analysis.SCENARIO]
                ):
                    exp.UpdateBaseWeights(w)
                else:
                    exp.UpdateNominalWeights(w)

    def ApplyWeights(self, tag, vector=None):
        if tag == "Fixed":
            labels = self.Analysis.Fixed
            vec = self.Analysis.FixedValue
        elif tag == "Nominal":
            labels = self.Analysis.Nuisance
            vec = self.Analysis.NuisNominal
        elif tag == "True":
            labels = self.Analysis.Physics
            vec = self.Analysis.PhysTrue
        elif tag == "Physics":
            labels = self.Analysis.Physics
            v_id = self.Analysis.PhysicsList
        elif tag == "Nuisance":
            labels = self.Analysis.Nuisance
            v_id = self.Analysis.NuisanceList
        else:
            sys.exit("Not a valid tag for applying weights.")

        w = 1  # Solve and understand why

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
                        if tune_block == "Flux":
                            w = self.physics_tunes[name].GetFlux(tune, value)
                        elif tune_block == "XSection":
                            w = self.physics_tunes[name].GetXSection(tune, value)
                        elif tune_block == "Detector":
                            w = self.physics_tunes[name].GetDetector(tune, value)
                        elif tune_block == "Osc":
                            self.physics_tunes[name].OscillationTunes.UpdateParameter(
                                tune, value
                            )

                        # if self.verbosity:
                        #     print(f"{tune} -- {w}")

                        if tune_block != "Osc":
                            if tag == "Fixed":
                                exp.UpdateBaseWeights(w)
                            elif tag in ["True", "Nominal"]:
                                exp.UpdateNominalWeights(w)
                            elif tag == "Physics":
                                exp.UpdatePhysicsWeights(w)
                            elif tag == "Nuisance":
                                exp.UpdateNuisanceWeights(w)

    def GetDiffLogWeights(self, vector):
        """Computes the derivative with respect the nuisance parameter nuis"""
        """ Returns a dict of nuis : experiment : partial of weight with respect to nuis over weight """
        dWoverW = {}
        for source, nuisance_list in self.Analysis.Nuisance.items():
            for name, exp in self.Experiments.items():
                if source in exp.Definition.keys():
                    tune_block = exp.Definition[source]
                    for tune in self.Analysis.Nuisance[source]:
                        dWoverW[tune] = {name: 0}
                        idx = self.Analysis.NuisanceList.index(tune)
                        if tune_block == "Detector":
                            dWoverW[tune][name] = self.physics_tunes[name].GetDetector(
                                f"diff_{tune}", vector[idx]
                            ) / self.physics_tunes[name].GetDetector(tune, vector[idx])
                        elif tune_block == "Flux":
                            dWoverW[tune][name] = self.physics_tunes[name].GetFlux(
                                f"diff_{tune}", vector[idx]
                            ) / self.physics_tunes[name].GetFlux(tune, vector[idx])
                        elif tune_block == "Osc":
                            dWoverW[tune][name] = (
                                self.physics_tunes[name].GetOscillation(
                                    f"diff_{tune}", vector[idx]
                                )
                                / self.physics_tunes[
                                    name
                                ].OscillationTunes.GetOscillations()
                            )
                        elif tune_block == "XSection":
                            dWoverW[tune][name] = self.physics_tunes[name].GetXSection(
                                f"diff_{tune}", vector[idx]
                            ) / self.physics_tunes[name].GetXSection(tune, vector[idx])
        return dWoverW

    def set_likelihood(self, mode):
        if mode == "BinnedLogLikelihoodRatio":
            self.LLH = ft.BinnedLogLikelihoodRatio(
                self.Observation,
                self.Analysis.NuisNominalList,
                self.Analysis.NuisSigmaList,
                self.Analysis.NuisDistributionList,
            )
        else:
            sys.exit("Mode not yet implemented")

    # 'SLSQP' 'GD' 'ADAM' 'MINUIT'
    def FitModel(self, point, mode="BinnedLogLikelihoodRatio", method="L-BFGS-B", eps=1e-5):
        if not self.Analysis.do_point(point):
            print(f"Skipping point {point}.")
            return False

        """ Binned log-Likelihood fit assuming data is Poisson-distributed """
        self.set_likelihood(mode)
        self.point = point

        """ Binned log-Likelihood fit assuming data is Poisson-distributed """
        self.ComputeBinnedExpectation(self.point, physics=True)  # Nominal expectation

        """ Statistics only computation to start guiding the minimization """
        X2_stats = self.LLH.stats_only(self.Expectation)
        print(f"Stats only, chi2 = {X2_stats}")

        self.WriteToOutFile("Analysis", "Chi2 Stats. Only", X2_stats)

        if self.Analysis.wSyst:
            """Get Jacobian of expected events w.r.t. nuisance parameters"""
            self.ComputeBinnedDiffExpectation()

            """Analytic estimate for priors and bounds at first order"""
            AnalyticPrior, AnalyticBounds = self.LLH.analytic_priors_bounds(
                self.Expectation, self.DiffExpectation
            )

            """Combined chi^2 minimization"""
            if method == "GD":
                from .gradient_descent_minimizer import gradient_descent_minimizer

                gradient_descent_minimizer(
                    self.model_tester_and_gradient,
                    AnalyticPrior,
                    # epsilon = eps,
                    bounds=AnalyticBounds,
                )

            elif method == "ADAM":
                from .adam_minimizer import adam_minimizer

                adam_minimizer(
                    self.model_tester_and_gradient,
                    AnalyticPrior,
                    # precission = eps,
                    bounds=AnalyticBounds,
                )

            elif method == "MINUIT":
                import iminuit

                res = iminuit.minimize(
                    self.model_tester,
                    AnalyticPrior,
                    method="migrad",
                    jac=self.model_tester_gradient,
                    bounds=AnalyticBounds,
                    tol=eps,
                    options={"disp": self.verbosity},
                )

            elif method == "TEST":
                for i in range(2 * self.Analysis.NumberOfNuis):
                    x = np.asarray(AnalyticPrior) - (
                        i - self.Analysis.NumberOfNuis
                    ) * np.asarray(self.Analysis.NuisSigmaList)
                    x2, dx2 = self.model_tester_and_gradient(x)

            else:
                res = minimize(
                    self.model_tester_and_gradient,
                    AnalyticPrior,
                    # method="Newton-CG", # 5min 45s
                    # method="BFGS",  # 2min 38s
                    method="L-BFGS-B",  # 3min 11s
                    jac=True,
                    bounds=AnalyticBounds,
                    tol=eps,
                    options={"disp": self.verbosity},
                )

            self.WriteToOutFile(
                "Nuisance Parameters", self.Analysis.NuisanceList, res.x.tolist()
            )
            self.WriteToOutFile("Analysis", "Chi2 Systs.", res.fun)

            return -0.5 * res.fun
        return -X2_stats

    def model_tester_and_gradient(self, nuisance_vector):
        """Compute expected and its derivatives"""
        self.ComputeBinnedExpectation(
            self.point, nuisance_vector=nuisance_vector
        )  # Nominal expectation
        self.ComputeBinnedDiffExpectation(nuisance_vector=nuisance_vector)

        """ Get -2 ln(H/H0) ~ χ2 """
        Chi2 = self.LLH.stats_and_systematics(self.Expectation, nuisance_vector)

        """ The gradient of the above """
        D_Chi2 = self.LLH.gradient(
            self.Expectation, self.DiffExpectation, nuisance_vector
        )

        return (Chi2, D_Chi2)

    def model_tester(self, nuisance_vector):
        """Compute expected and its derivatives"""
        self.ComputeBinnedExpectation(
            self.point, nuisance_vector=nuisance_vector
        )  # Nominal expectation
        self.ComputeBinnedDiffExpectation(nuisance_vector=nuisance_vector)

        """ Get -2 ln(H/H0) ~ χ2 """
        return self.LLH.stats_and_systematics(self.Expectation, nuisance_vector)

    def model_tester_gradient(self, nuisance_vector):
        """Compute expected and its derivatives"""
        self.ComputeBinnedExpectation(
            self.point, nuisance_vector=nuisance_vector
        )  # Nominal expectation
        self.ComputeBinnedDiffExpectation(nuisance_vector=nuisance_vector)

        """ The gradient of the above """
        return self.LLH.gradient(
            self.Expectation, self.DiffExpectation, nuisance_vector
        )

    def SetOutFile(self, fname):
        self.outfile = fname

    def CreateOutFile(self, fname):
        self.outfile = fname
        with h5py.File(fname, "w") as hf:
            grp = hf.create_group("Fixed Parameters")
            for key in self.Analysis.Fixed.keys():
                this = grp.create_group(key)
                for par, val in self.Analysis.FixedValue[key].items():
                    this.create_dataset(par, data=[val], compression="gzip")

            if self.Analysis.wSyst:
                grp = hf.create_group("Nuisance Parameters")
                for key in self.Analysis.Nuisance.keys():
                    this = grp.create_group(key)
                    for par in self.Analysis.Nuisance[key]:
                        this.create_dataset(
                            par,
                            data=[0.0] * self.Analysis.NumberOfPhysPoints,
                            compression="gzip",
                        )
            grp = hf.create_group("Physics Parameters")

            physics_lists = list(zip(*self.Analysis.FullPhysicsGrid))
            for key in self.Analysis.Physics.keys():
                this = grp.create_group(key)
                for par in self.Analysis.Physics[key]:
                    idx = self.Analysis.PhysicsList.index(par)
                    this.create_dataset(
                        par, data=physics_lists[idx], compression="gzip"
                    )

            grp = hf.create_group("Analysis")
            grp.create_dataset(
                "Chi2 Stats. Only",
                data=[0.0] * self.Analysis.NumberOfPhysPoints,
                compression="gzip",
            )
            if self.Analysis.wSyst:
                grp.create_dataset(
                    "Chi2 Systs.",
                    data=[0.0] * self.Analysis.NumberOfPhysPoints,
                    compression="gzip",
                )

    def WriteToOutFile(self, block, item, value):
        while True:
            try:
                with open(self.outfile, "a") as f:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(f, fcntl.LOCK_UN)
                    break
            except IOError:
                time.sleep(1)

        with h5py.File(self.outfile, "r+") as hf:
            print("Writing to output file.")
            try:
                for par, val in zip(item, value):
                    source = self.Analysis.get_tune(par)
                    hf[f"{block}/{source}/{par}"][self.point] = val
            except BaseException:
                hf[f"{block}/{item}"][self.point] = value


# project to improve ApplyWeights
"""
def ApplyWeights(self, tag, vector=None):
    # Define tag-based configuration
    tag_config = {
        "Fixed": (self.Analysis.Fixed, self.Analysis.FixedValue, None),
        "Nominal": (self.Analysis.Nuisance, self.Analysis.NuisNominal, None),
        "True": (self.Analysis.Physics, self.Analysis.PhysTrue, None),
        "Physics": (self.Analysis.Physics, None, self.Analysis.PhysicsList),
        "Nuisance": (self.Analysis.Nuisance, None, self.Analysis.NuisanceList),
    }
    
    if tag not in tag_config:
        sys.exit("Not a valid tag for applying weights.")

    labels, vec, v_id = tag_config[tag]
    w = 1  # Placeholder for weight calculation

    # Helper function to get weight based on tune_block type
    def get_weight(tune_block, name, tune, value):
        if tune_block == "Flux":
            return self.physics_tunes[name].GetFlux(tune, value)
        elif tune_block == "XSection":
            return self.physics_tunes[name].GetXSection(tune, value)
        elif tune_block == "Detector":
            return self.physics_tunes[name].GetDetector(tune, value)
        elif tune_block == "Osc":
            self.physics_tunes[name].OscillationTunes.UpdateParameter(tune, value)
            return None
        return 1

    # Helper function to update weights
    def update_weights(exp, tune_block, w):
        if tune_block != "Osc":
            if tag == "Fixed":
                exp.UpdateBaseWeights(w)
            elif tag in ["True", "Nominal"]:
                exp.UpdateNominalWeights(w)
            elif tag == "Physics":
                exp.UpdatePhysicsWeights(w)
            elif tag == "Nuisance":
                exp.UpdateNuisanceWeights(w)

    # Iterate over experiments and sources
    for name, exp in self.Experiments.items():
        for source in labels:
            if source not in exp.Definition:
                continue

            tune_block = exp.Definition[source]
            for tune in labels[source]:
                value = vector[v_id.index(tune)] if vector else vec[source][tune]
                w = get_weight(tune_block, name, tune, value)

                if w is not None:
                    update_weights(exp, tune_block, w)
"""
