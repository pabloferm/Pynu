"""
PyNuFit - Python Neutrino Fitting Framework

Main fitting module that handles XML configuration parsing and
orchestrates the oscillation parameter grid scan.

Extended to support CPT invariance testing with separate Dm231 and Dm231_bar parameters.
Supports 1D and 2D profile likelihood scans with marginalization over nuisance parameters.
"""

import sys
import time
import fcntl
import os
from datetime import datetime

import numpy as np
from scipy.optimize import minimize

import h5py

from .analysis_reader import ParseXML
from . import Experiments as Exp

from .PhysicsTunes.PhysicsTunes import PhysicsTunes as PT
from . import fitter as ft
from .fitter.inference import mcmc


class PyNuFit:
    """
    Top class containing everything for neutrino oscillation fitting.

    Extended with CPT analysis capabilities including profile likelihood scans.
    """

    # Standard oscillation parameters
    STANDARD_PARAMS = [
        "Sin2Theta12", "Sin2Theta13", "Sin2Theta23",
        "Dm221", "Dm231", "Dm232", "dCP", "Ordering"
    ]

    # CPT-extended parameters
    CPT_PARAMS = ["Dm231_bar"]

    # All recognized physics parameters
    ALL_PARAMS = STANDARD_PARAMS + CPT_PARAMS

    def __init__(self, analysis_file, path=None, verbosity=False):
        self.verbosity = verbosity
        self.path = path

        """ Set up basic analysis variables and structure to build full analysis """
        self.Analysis = ParseXML(analysis_file, check=self.verbosity)
        self.Analysis.get_analysis()

        """ Define dictionary for PhysicsTunes """
        self.physics_tunes = {}

        """ CPT-specific: marginalization parameters """
        self.marginalize_params = {}
        self._parse_marginalization_config(analysis_file)

        """ Start the analysis """
        self.SetUpExperiments()
        self.SetUpPhysicsTunes()

        """ Compute Observation """
        self.ComputeBinnedObservation()

    def _parse_marginalization_config(self, analysis_file):
        """Parse marginalization parameters from XML config for profile scans."""
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse(analysis_file)
            root = tree.getroot()

            # Look for marginalize elements in NeutrinoOscillations section
            osc_config = root.find(".//NeutrinoOscillations")
            if osc_config is not None:
                for marginalize in osc_config.findall("marginalize"):
                    param_name = marginalize.get("name")
                    if param_name in self.ALL_PARAMS:
                        self.marginalize_params[param_name] = {
                            "min": float(marginalize.find("min").text),
                            "max": float(marginalize.find("max").text),
                            "true": float(marginalize.find("true").text)
                        }
        except Exception as e:
            if self.verbosity:
                print(f"Note: Could not parse marginalization config: {e}")

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
        # Compute MC variance for BB likelihood
        self.SetBinnedMCVariance()
        # Get muon background for experiments that have it
        self.SetMuonBackground()

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

    def SetBinnedMCVariance(self):
        """Compute binned MC variance for Barlow-Beeston likelihood."""
        self.MCVariance = {}
        for name, exp in self.Experiments.items():
            # Check if experiment supports MC variance (e.g., ORCA)
            if hasattr(exp, 'GetMCVariance'):
                mc_var = exp.GetMCVariance(exp.ExpectedWeight)
                # Apply FewEntries filter to match expectation shape
                if hasattr(exp, 'FewEntries') and exp.FewEntries is not None:
                    self.MCVariance[name] = mc_var[exp.FewEntries]
                else:
                    self.MCVariance[name] = mc_var
            else:
                # Default: use Poisson variance (weights squared)
                # This is a fallback for experiments without explicit MC variance
                binned_w2 = exp.BinMC(exp.ExpectedWeight**2)
                if hasattr(exp, 'FewEntries') and exp.FewEntries is not None:
                    self.MCVariance[name] = binned_w2[exp.FewEntries]
                else:
                    self.MCVariance[name] = binned_w2


    def SetMuonBackground(self):
        """Get muon background for experiments that have it (e.g., ORCA)."""
        self.MuonBackground = {}
        for name, exp in self.Experiments.items():
            # Check if experiment has muon background (e.g., ORCA)
            if hasattr(exp, 'GetMuonBackground'):
                muon_counts, muon_var = exp.GetMuonBackground()
                if muon_counts is not None:
                    # Apply FewEntries filter to match expectation shape
                    if hasattr(exp, 'FewEntries') and exp.FewEntries is not None:
                        self.MuonBackground[name] = (
                            muon_counts[exp.FewEntries],
                            muon_var[exp.FewEntries]
                        )
                    else:
                        self.MuonBackground[name] = (muon_counts, muon_var)
                else:
                    self.MuonBackground[name] = None
            else:
                # Experiment doesn't have muon background
                self.MuonBackground[name] = None

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
                            w = self.physics_tunes[name].get_flux(tune, value)
                        elif tune_block == "XSection":
                            w = self.physics_tunes[name].get_xsection(tune, value)
                        elif tune_block == "Detector":
                            w = self.physics_tunes[name].get_detector(tune, value)
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
                            dWoverW[tune][name] = self.physics_tunes[name].get_detector(
                                f"diff_{tune}", vector[idx]
                            ) / self.physics_tunes[name].get_detector(tune, vector[idx])
                        elif tune_block == "Flux":
                            dWoverW[tune][name] = self.physics_tunes[name].get_flux(
                                f"diff_{tune}", vector[idx]
                            ) / self.physics_tunes[name].get_flux(tune, vector[idx])
                        elif tune_block == "Osc":
                            dWoverW[tune][name] = (
                                self.physics_tunes[name].get_oscillation(
                                    f"diff_{tune}", vector[idx]
                                )
                                / self.physics_tunes[
                                    name
                                ].OscillationTunes.get_oscillation()
                            )
                        elif tune_block == "XSection":
                            dWoverW[tune][name] = self.physics_tunes[name].get_xsection(
                                f"diff_{tune}", vector[idx]
                            ) / self.physics_tunes[name].get_xsection(tune, vector[idx])
        return dWoverW

    def set_likelihood(self, mode):
        if mode == "BinnedLogLikelihoodRatio":
            self.LLH = ft.BinnedLogLikelihoodRatio(
                self.Observation,
                self.Analysis.NuisNominalList,
                self.Analysis.NuisSigmaList,
                self.Analysis.NuisDistributionList,
            )
        elif mode == "BarlowBeestonLikelihood":
            self.LLH = ft.BarlowBeestonLikelihood(
                self.Observation,
                self.Analysis.NuisNominalList,
                self.Analysis.NuisSigmaList,
                self.Analysis.NuisDistributionList,
            )
            # Auto-detect muon_norm in nuisance list and wire it up
            if 'muon_norm' in self.Analysis.NuisanceList:
                idx = self.Analysis.NuisanceList.index('muon_norm')
                self.LLH.set_muon_norm_index(idx)
        else:
            sys.exit("Mode not yet implemented")

    # 'SLSQP' 'GD' 'ADAM' 'MINUIT'
    def FitModel(
        self, point, mode="BinnedLogLikelihoodRatio", method="BFGS", eps=None
    ):

        if not self.Analysis.do_point(point):
            print(f"Skipping point {point}.")
            return False

        """ Binned log-Likelihood fit assuming data is Poisson-distributed """
        self.set_likelihood(mode)
        self.point = point

        """ Binned log-Likelihood fit assuming data is Poisson-distributed """
        self.ComputeBinnedExpectation(self.point, physics=True)  # Nominal expectation

        """ Statistics only computation to start guiding the minimization """
        X2_stats = self.LLH.stats_and_systematics(
            self.Expectation, self.Analysis.NuisNominalList
        )
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

            elif method == "BFGS":
                res = minimize(
                    self.model_tester_and_gradient,
                    self.Analysis.NuisNominalList,
                    # AnalyticPrior,
                    # method="Newton-CG", # 5min 45s
                    method="BFGS",  # 2min 38s
                    # method="L-BFGS-B",  # 3min 11s
                    jac=True,
                    # bounds=AnalyticBounds,
                    tol=eps,
                    options={
                        "disp": self.verbosity,
                        "hess_inv0": self.fisher_information(
                            self.Analysis.NuisNominalList
                        ),
                        "gtol": 1e-4,
                    },
                )

            elif method == "HMC":
                """Hamiltonian MCMC"""
                import numpy as np
                riemann_mass = 1 / np.array(self.Analysis.NuisSigmaList) ** 2
                print(riemann_mass)
                riemann_mass = self.fisher_information(AnalyticPrior)
                print(riemann_mass)
                ranges = (np.array(list(zip(*AnalyticBounds)))[1] - np.array(list(zip(*AnalyticBounds)))[0])/2
                sampler = mcmc.HMC(
                    self.model_tester,
                    self.model_tester_gradient,
                    AnalyticPrior,
                    range_of_initial_values=ranges,
                    num_steps=20,
                    random_steps="linear",
                    riemann_mass=riemann_mass,
                    epsilon=5e-2,
                )
                sampler.compute_trajectory(samples=200)

            else:
                sys.exit(f"{method} is not a valid fitting method, please check PyNuFit.py")

                # sampler = mcmc.MCMC(
                #     self.model_tester, AnalyticPrior)
                # all_samples = sampler.metropolis_hastings()

                """ Cython version of MCMC Metropolis-Hastings"""
                # initial_values = np.abs(np.random.randn(len(AnalyticPrior)) + 1, dtype=np.float64)
                # sigma = np.zeros(len(AnalyticPrior), dtype=np.float64) + 0.5
                # num_samples = 500
                # all_samples = np.asarray(run_metropolis_hastings(num_samples, self.model_tester, initial_values, sigma))

                """SVGD"""
                # x0 = np.random.uniform(0.5, 1.5, (50, len(AnalyticPrior)))
                # all_samples = variational.SVGD().update(
                #     x0, self.model_tester_gradient, n_iter=50)

                # import pandas as pd
                # df = pd.DataFrame(all_samples, columns=self.Analysis.NuisanceList)
                # import seaborn as sns
                # import matplotlib.pyplot as plt
                # g = sns.PairGrid(df, corner=True, aspect=1.5)
                # g.map_diag(sns.histplot, bins=20)
                # g.map_offdiag(sns.kdeplot, levels=[0.68, 0.95, 0.997])
                # # g.map_offdiag(sns.scatterplot)
                # plt.show()

            if res: # quick and dirty workaround until inference is fully supported
                self.WriteToOutFile(
                    "Nuisance Parameters", self.Analysis.NuisanceList, res.x.tolist()
                )
                self.WriteToOutFile("Analysis", "Chi2 Systs.", res.fun)
                return res.fun

        return X2_stats

    def fisher_information(self, nuisance_vector):
        """Compute expected and its derivatives"""
        self.ComputeBinnedExpectation(
            self.point, nuisance_vector=nuisance_vector
        )  # Nominal expectation
        self.ComputeBinnedDiffExpectation(nuisance_vector=nuisance_vector)

        """ The gradient of the above """
        I = self.LLH.approximate_fisher(self.Expectation, self.DiffExpectation)

        return np.diag(I)

    def _llh_chi2(self, expectation, nuisance, mc_var=None):
        """Call LLH.stats_and_systematics, passing mc_var only if the LLH supports it.

        Barlow-Beeston accepts an optional ``mc_variance`` argument; the standard
        binned/unbinned LLR classes do not. This wrapper hides that asymmetry.
        """
        if mc_var is not None and hasattr(self.LLH, 'set_mc_variance'):
            return self.LLH.stats_and_systematics(expectation, nuisance, mc_var)
        return self.LLH.stats_and_systematics(expectation, nuisance)

    def _llh_grad(self, expectation, diff_expectation, nuisance, mc_var=None):
        """Call LLH.gradient, passing mc_var only if the LLH supports it."""
        if mc_var is not None and hasattr(self.LLH, 'set_mc_variance'):
            return self.LLH.gradient(expectation, diff_expectation, nuisance, mc_var)
        return self.LLH.gradient(expectation, diff_expectation, nuisance)

    def model_tester_and_gradient(self, nuisance_vector):
        if self.verbosity:
            print(
                f"Values of varying parameters:\n{self.Analysis.NuisanceList}\n{nuisance_vector}"
            )
            print(
                "--------------------------------------------------------------------------"
            )
        """Compute expected and its derivatives"""
        self.ComputeBinnedExpectation(
            self.point, nuisance_vector=nuisance_vector
        )  # Nominal expectation
        self.ComputeBinnedDiffExpectation(nuisance_vector=nuisance_vector)

        """ Get -2 ln(H/H0) ~ χ2 """
        # Pass MC variance and muon background if using Barlow-Beeston likelihood
        mc_var = getattr(self, 'MCVariance', None)
        muon_bkg = getattr(self, 'MuonBackground', None)
        if hasattr(self.LLH, 'set_mc_variance') and mc_var is not None:
            self.LLH.set_mc_variance(mc_var)
        if hasattr(self.LLH, 'set_muon_background') and muon_bkg is not None:
            self.LLH.set_muon_background(muon_bkg)
        Chi2 = self._llh_chi2(self.Expectation, nuisance_vector, mc_var)

        """ The gradient of the above """
        D_Chi2 = self._llh_grad(
            self.Expectation, self.DiffExpectation, nuisance_vector, mc_var
        )

        return (Chi2, D_Chi2)

    def model_tester(self, nuisance_vector):
        """Compute expected and its derivatives"""
        self.ComputeBinnedExpectation(
            self.point, nuisance_vector=nuisance_vector
        )  # Nominal expectation
        self.ComputeBinnedDiffExpectation(nuisance_vector=nuisance_vector)

        """ Get -2 ln(H/H0) ~ χ2 """
        mc_var = getattr(self, 'MCVariance', None)
        return self._llh_chi2(self.Expectation, nuisance_vector, mc_var)

    def model_tester_gradient(self, nuisance_vector):
        """Compute expected and its derivatives"""
        self.ComputeBinnedExpectation(
            self.point, nuisance_vector=nuisance_vector
        )  # Nominal expectation
        self.ComputeBinnedDiffExpectation(nuisance_vector=nuisance_vector)

        """ The gradient of the above """
        mc_var = getattr(self, 'MCVariance', None)
        return self._llh_grad(
            self.Expectation, self.DiffExpectation, nuisance_vector, mc_var
        )

    # =========================================================================
    # CPT Profile Likelihood Methods
    # =========================================================================

    def run_profile_scan(self, scan_param, scan_values, marginalize_over=None,
                         mode="BarlowBeestonLikelihood", verbose=True):
        """
        Run 1D profile likelihood scan with marginalization over nuisance parameters.

        At each scan point, minimizes chi² over the specified marginalization parameters.

        Args:
            scan_param: Name of oscillation parameter to scan (e.g., "Dm231")
            scan_values: Array of values to scan over
            marginalize_over: Dict of {param_name: (min, max)} for marginalization.
                            If None, uses self.marginalize_params from config.
            mode: Likelihood mode ("BarlowBeestonLikelihood" or "BinnedLogLikelihoodRatio")
            verbose: If True, print progress

        Returns:
            dict: Results including chi2_profile, scan_values, best_fit_nuisance, etc.
        """
        # Setup likelihood
        self.set_likelihood(mode)

        # Determine marginalization parameters
        if marginalize_over is None:
            marginalize_over = {
                name: (cfg["min"], cfg["max"])
                for name, cfg in self.marginalize_params.items()
            }

        n_scan = len(scan_values)
        chi2_profile = np.zeros(n_scan)
        best_fit_nuisance = []

        # Get the first experiment's oscillation tunes for direct parameter access
        exp_name = list(self.physics_tunes.keys())[0]
        osc_tunes = self.physics_tunes[exp_name].OscillationTunes

        if verbose:
            print(f"Running profile scan over {scan_param}")
            print(f"  Scan range: [{scan_values[0]:.4e}, {scan_values[-1]:.4e}]")
            print(f"  Scan points: {n_scan}")
            print(f"  Marginalization parameters: {list(marginalize_over.keys())}")

        for i, scan_val in enumerate(scan_values):
            # Set scan parameter for all experiments
            for name, pt in self.physics_tunes.items():
                pt.OscillationTunes.UpdateParameter(scan_param, scan_val)

            if len(marginalize_over) == 0:
                # No marginalization - just evaluate with nominal nuisance
                self.ComputeBinnedExpectation(0, physics=False)
                mc_var = getattr(self, 'MCVariance', None)
                chi2_profile[i] = self._llh_chi2(
                    self.Expectation, self.Analysis.NuisNominalList, mc_var
                )
                best_fit_nuisance.append({})
            else:
                # Define objective for marginalization
                margin_names = list(marginalize_over.keys())
                margin_bounds = list(marginalize_over.values())

                def objective(margin_vals):
                    # Set marginalization parameters
                    for j, param_name in enumerate(margin_names):
                        for name, pt in self.physics_tunes.items():
                            pt.OscillationTunes.UpdateParameter(param_name, margin_vals[j])

                    # Recompute expectation
                    self.ComputeBinnedExpectation(0, physics=False)
                    mc_var = getattr(self, 'MCVariance', None)
                    return self._llh_chi2(
                        self.Expectation, self.Analysis.NuisNominalList, mc_var
                    )

                # Initial guess
                x0 = []
                for param_name, bounds in marginalize_over.items():
                    if param_name in self.marginalize_params:
                        x0.append(self.marginalize_params[param_name]["true"])
                    else:
                        x0.append((bounds[0] + bounds[1]) / 2)
                x0 = np.array(x0)

                # Minimize
                result = minimize(
                    objective, x0, method='L-BFGS-B', bounds=margin_bounds,
                    options={'ftol': 1e-6, 'gtol': 1e-5, 'maxiter': 100}
                )

                chi2_profile[i] = result.fun
                best_fit_nuisance.append(dict(zip(margin_names, result.x)))

            if verbose and ((i + 1) % 10 == 0 or i == n_scan - 1):
                print(f"  Progress: {i+1}/{n_scan} ({100*(i+1)/n_scan:.1f}%)")

        # Calculate Delta chi2
        min_chi2 = np.min(chi2_profile)
        delta_chi2 = chi2_profile - min_chi2
        best_idx = np.argmin(chi2_profile)

        return {
            "scan_param": scan_param,
            "scan_values": scan_values,
            "chi2_profile": chi2_profile,
            "delta_chi2": delta_chi2,
            "min_chi2": min_chi2,
            "best_fit_scan": scan_values[best_idx],
            "best_fit_nuisance": best_fit_nuisance,
            "best_fit_nuisance_at_min": best_fit_nuisance[best_idx],
            "marginalize_params": marginalize_over
        }

    def run_2d_profile_scan(self, scan_params_2d, grid1, grid2, marginalize_over=None,
                              mode="BarlowBeestonLikelihood", verbose=True):
        """
        Run 2D profile likelihood scan with marginalization.

        Scans a 2D grid over two parameters while minimizing chi² over remaining
        marginalization parameters at each grid point.

        Args:
            scan_params_2d: Tuple of (param1_name, param2_name) to scan
            grid1: Array of values for first parameter
            grid2: Array of values for second parameter
            marginalize_over: Dict of {param_name: (min, max)} for marginalization
            mode: Likelihood mode
            verbose: If True, print progress

        Returns:
            dict: Results including chi2_grid, param_values, best_fit, etc.
        """
        # Setup likelihood
        self.set_likelihood(mode)

        # Determine marginalization parameters
        if marginalize_over is None:
            marginalize_over = {
                name: (cfg["min"], cfg["max"])
                for name, cfg in self.marginalize_params.items()
                if name not in scan_params_2d
            }

        n1, n2 = len(grid1), len(grid2)
        total_points = n1 * n2
        chi2_grid = np.zeros((n1, n2))
        best_fit_nuisance = [[{} for _ in range(n2)] for _ in range(n1)]

        margin_names = list(marginalize_over.keys())
        margin_bounds = list(marginalize_over.values())

        # Initial guess
        x0 = []
        for param_name, bounds in marginalize_over.items():
            if param_name in self.marginalize_params:
                x0.append(self.marginalize_params[param_name]["true"])
            else:
                x0.append((bounds[0] + bounds[1]) / 2)
        x0 = np.array(x0) if len(x0) > 0 else None

        if verbose:
            print(f"Running 2D profile scan over ({scan_params_2d[0]}, {scan_params_2d[1]})")
            print(f"  Grid shape: ({n1}, {n2}) = {total_points} points")
            print(f"  Marginalization parameters: {margin_names}")

        done = 0
        for i, val1 in enumerate(grid1):
            for j, val2 in enumerate(grid2):
                # Set 2D scan parameters
                for name, pt in self.physics_tunes.items():
                    pt.OscillationTunes.UpdateParameter(scan_params_2d[0], val1)
                    pt.OscillationTunes.UpdateParameter(scan_params_2d[1], val2)

                if len(margin_names) == 0:
                    # No marginalization
                    self.ComputeBinnedExpectation(0, physics=False)
                    mc_var = getattr(self, 'MCVariance', None)
                    chi2_grid[i, j] = self._llh_chi2(
                        self.Expectation, self.Analysis.NuisNominalList, mc_var
                    )
                else:
                    def objective(margin_vals):
                        for k, param_name in enumerate(margin_names):
                            for name, pt in self.physics_tunes.items():
                                pt.OscillationTunes.UpdateParameter(param_name, margin_vals[k])
                        self.ComputeBinnedExpectation(0, physics=False)
                        mc_var = getattr(self, 'MCVariance', None)
                        return self._llh_chi2(
                            self.Expectation, self.Analysis.NuisNominalList, mc_var
                        )

                    result = minimize(
                        objective, x0, method='L-BFGS-B', bounds=margin_bounds,
                        options={'ftol': 1e-6, 'gtol': 1e-5, 'maxiter': 100}
                    )

                    chi2_grid[i, j] = result.fun
                    best_fit_nuisance[i][j] = dict(zip(margin_names, result.x))
                    x0 = result.x.copy()  # Warm start

                done += 1
                if verbose and (done % max(1, total_points // 20) == 0 or done == total_points):
                    print(f"  Progress: {done}/{total_points} ({100*done/total_points:.1f}%)")

        # Find best fit
        min_chi2 = np.nanmin(chi2_grid)
        delta_chi2 = chi2_grid - min_chi2
        best_idx = np.unravel_index(np.nanargmin(chi2_grid), chi2_grid.shape)

        return {
            "scan_params": scan_params_2d,
            "param_values": {scan_params_2d[0]: grid1, scan_params_2d[1]: grid2},
            "chi2_grid": chi2_grid,
            "delta_chi2": delta_chi2,
            "min_chi2": float(min_chi2),
            "best_fit": {
                scan_params_2d[0]: float(grid1[best_idx[0]]),
                scan_params_2d[1]: float(grid2[best_idx[1]])
            },
            "best_fit_nuisance": best_fit_nuisance,
            "best_fit_nuisance_at_min": best_fit_nuisance[best_idx[0]][best_idx[1]],
            "marginalize_params": marginalize_over
        }

    def find_confidence_intervals(self, results, levels=None):
        """
        Find confidence intervals from profile likelihood results.

        Args:
            results: Results dict from run_profile_scan
            levels: Dict of {name: delta_chi2_threshold} (default: 1σ, 2σ, 3σ for 1 DOF)

        Returns:
            dict: Confidence intervals for each level
        """
        if levels is None:
            levels = {"1sigma": 1.0, "2sigma": 4.0, "3sigma": 9.0}

        scan_values = results["scan_values"]
        delta_chi2 = results["delta_chi2"]
        best_fit = results["best_fit_scan"]

        intervals = {}
        for name, threshold in levels.items():
            within = delta_chi2 <= threshold
            if np.any(within):
                vals_within = scan_values[within]
                intervals[name] = {
                    "lower": float(np.min(vals_within)),
                    "upper": float(np.max(vals_within)),
                    "best_fit": float(best_fit),
                    "delta_chi2_threshold": threshold
                }
            else:
                intervals[name] = None

        return intervals

    def save_profile_results(self, results, output_dir, prefix="cpt_profile"):
        """Save profile likelihood scan results to files."""
        import json
        os.makedirs(output_dir, exist_ok=True)

        np.save(os.path.join(output_dir, f"{prefix}_chi2.npy"), results["chi2_profile"])
        np.save(os.path.join(output_dir, f"{prefix}_delta_chi2.npy"), results["delta_chi2"])
        np.save(os.path.join(output_dir, f"{prefix}_{results['scan_param']}.npy"),
                results["scan_values"])

        intervals = self.find_confidence_intervals(results)

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "scan_param": results["scan_param"],
            "scan_range": [float(results["scan_values"][0]),
                          float(results["scan_values"][-1])],
            "n_scan_points": len(results["scan_values"]),
            "marginalize_params": {k: list(v) for k, v in results["marginalize_params"].items()},
            "min_chi2": float(results["min_chi2"]),
            "best_fit_scan": float(results["best_fit_scan"]),
            "best_fit_nuisance": {k: float(v) for k, v in
                                  results["best_fit_nuisance_at_min"].items()},
            "confidence_intervals": intervals
        }

        with open(os.path.join(output_dir, f"{prefix}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Profile results saved to {output_dir}")

    def save_2d_profile_results(self, results, output_dir, prefix="cpt_2d_profile"):
        """Save 2D profile likelihood scan results to files."""
        import json
        os.makedirs(output_dir, exist_ok=True)

        scan_params = results["scan_params"]

        np.save(os.path.join(output_dir, f"{prefix}_chi2.npy"), results["chi2_grid"])
        np.save(os.path.join(output_dir, f"{prefix}_delta_chi2.npy"), results["delta_chi2"])

        for param in scan_params:
            np.save(os.path.join(output_dir, f"{prefix}_{param}.npy"),
                    results["param_values"][param])

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "scan_params": scan_params,
            "grid_shape": list(results["chi2_grid"].shape),
            "marginalize_params": {k: list(v) for k, v in results["marginalize_params"].items()},
            "min_chi2": results["min_chi2"],
            "best_fit": results["best_fit"],
            "best_fit_nuisance": {k: float(v) for k, v in
                                 results["best_fit_nuisance_at_min"].items()}
        }

        with open(os.path.join(output_dir, f"{prefix}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"2D profile results saved to {output_dir}")

    # =========================================================================
    # Output File Management
    # =========================================================================

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
