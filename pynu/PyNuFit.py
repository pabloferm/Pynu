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

        """ Per-dial tune-block owner map, built ONCE from the tune classes'
        own method inventories. Zero
        per-iteration cost; ApplyWeights/GetDiffLogWeights dispatch by this map
        so a flat-order XML (all dials under one <NeutrinoSource>) routes each
        dial to the class that actually owns its method, and existing 3-block
        XMLs route identically to the legacy block scheme. """
        self.DialOwnerMap = self._build_dial_owner_map()

        """ Compute Observation """
        self.ComputeBinnedObservation()

        """ Optional native binned-tensor engine (default OFF: {} unless the XML
        declares a <BinnedEngine> block; then one BinnedBinding per opted-in
        experiment). Staged (phi, theta) for the modular path live on self below. """
        self.BinnedEngines = self._setup_binned_engines(analysis_file)
        # modular-path staged state (Track S / E6: was the adapter's; now here).
        self._binned_staged_phi = None      # dCP slice selected by ApplyPhysicsWeights
        self._binned_staged_theta = None    # nuisance vector staged by ApplyNuisanceWeights

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

    # infrastructure methods on the tune classes that are NOT dials (shared
    # base-class helpers); excluded when inventorying method names.
    _NON_DIAL_TUNE_METHODS = frozenset({
        "Get", "cache_method", "Tune",
    })

    def _tune_dial_names(self, tune_obj):
        """The set of dial method names a tune class exposes: public, callable,
        not a ``diff_`` gradient twin, not a base-class helper. This is the same
        lookup ``PhysicsTunes.Get`` performs (``__getattribute__(name)``), so a
        name in this set is exactly a name ``get_flux/get_xsection/get_detector``
        can evaluate."""
        if tune_obj is None:
            return set()
        names = set()
        for n in dir(tune_obj):
            if n.startswith("_") or n.startswith("diff_"):
                continue
            if n in self._NON_DIAL_TUNE_METHODS:
                continue
            if callable(getattr(tune_obj, n, None)):
                names.add(n)
        return names

    def _build_dial_owner_map(self):
        """Return ``{dial_name: block}`` where block is one of
        ``'Flux'/'XSection'/'Detector'`` (``'Osc'`` dials are handled by the
        oscillation path and are not in this map). Built from the tune classes'
        method inventories so a new dial auto-registers to whichever class owns
        its method — no hardcoded name list.

        Regression contract: for a legacy 3-block XML every ACTIVE nuisance is
        declared under the block whose tune class owns it, so dispatching by
        this map yields the identical (block -> get_* method) outcome as the old
        ``exp.Definition[source]`` scheme. A dial that no tune class owns is
        omitted (dispatch then falls back to the XML block, preserving the exact
        legacy behaviour for e.g. ``muon_norm``/``Osc`` entries).

        A dial owned by two different blocks (across the loaded experiments)
        is a genuine ambiguity and raises — it cannot happen for the SK tune
        set (verified: the only Flux/XSec/Det method-name overlaps are the
        excluded base helpers)."""
        block_of = {}
        for name, pt in self.physics_tunes.items():
            for block, attr in (("Flux", "FluxTunes"),
                                 ("XSection", "XSectionTunes"),
                                 ("Detector", "DetectorTunes")):
                tune_obj = getattr(pt, attr, None)
                for dial in self._tune_dial_names(tune_obj):
                    prev = block_of.get(dial)
                    if prev is not None and prev != block:
                        raise ValueError(
                            f"dial {dial!r} is owned by both {prev!r} and "
                            f"{block!r} across the loaded experiments; the "
                            "owner-lookup routing needs a unique owner")
                    block_of[dial] = block
        return block_of

    def _dial_block(self, tune, source, exp):
        """Resolve the tune-block for one dial. Prefers the owner-lookup map
        (built from tune-class method inventories); falls back to the legacy
        ``exp.Definition[source]`` for anything the map does not own (Osc dials,
        muon_norm, or any dial without a matching tune method). For a legacy
        3-block XML both agree by construction, so this is a behaviour-preserving
        generalisation."""
        blk = self.DialOwnerMap.get(tune) if getattr(self, "DialOwnerMap", None) \
            else None
        if blk is not None:
            return blk
        return exp.Definition[source]

    def _binned_active(self):
        """True when a <BinnedEngine> block is active (mirrors FitModel:395)."""
        return bool(getattr(self, "BinnedEngines", None))

    def _the_binned_engine(self):
        """The single active binned binding (phase-1: exactly one binned
        experiment). Also returns its experiment name so the modular methods
        key ``self.Expectation``/``self.MCVariance``/``self.Observation`` under
        the same name the event path uses."""
        if len(self.BinnedEngines) != 1:
            raise NotImplementedError(
                "binned engine modular path supports exactly one binned "
                f"experiment (got {len(self.BinnedEngines)})")
        name, binding = next(iter(self.BinnedEngines.items()))
        return name, binding

    def _require_binned_staged(self):
        """Guard: ApplyPhysicsWeights + ApplyNuisanceWeights must have run before
        the expectation is contracted (mirrors the former adapter guard)."""
        if self._binned_staged_phi is None:
            raise RuntimeError("binned modular path: ApplyPhysicsWeights() must "
                               "run before the expectation is contracted")
        if self._binned_staged_theta is None:
            raise RuntimeError("binned modular path: ApplyNuisanceWeights() must "
                               "run before the expectation is contracted")

    # dCP node the modular ApplyPhysicsWeights/contraction currently targets;
    # the worker profiles dCP by setting this before each objective (default 0).
    _binned_dcp_node = 0

    def SetBinnedDcpNode(self, dcp_index):
        """Select the dCP node the modular binned path contracts at (worker-level
        dCP profile scan, same structure as the event engine's node loop)."""
        self._binned_dcp_node = int(dcp_index)

    def StageBinnedPhysics(self, dm231, s23, dcp_index):
        """Stage phi[dcp_index] at (dm231, s23) directly, bypassing the analysis
        physics grid. The exact numerical work ApplyPhysicsWeights does on the
        binned path, but keyed on explicit (dm231, s23) — used by a scan worker
        whose grid the analysis XML does not declare (former adapter.apply_physics
        delegation target; Track S / E6)."""
        _, binding = self._the_binned_engine()
        self._binned_dcp_node = int(dcp_index)
        phi = binding.store.phi(dm231, s23)
        self._binned_staged_phi = phi[int(dcp_index)].astype(float)
        return self._binned_staged_phi

    def StartPhysics(self):
        for exp in self.Experiments.values():
            exp.StartPhysicsWeights()

    def StartNuisance(self):
        # Binned modular path: reset the staged nuisance vector (cheap; mirrors
        # the event engine's per-event weight-array reset).
        if self._binned_active():
            self._binned_staged_theta = None
            return
        for exp in self.Experiments.values():
            exp.StartNuisanceWeights()

    def SetBinnedObservedEvents(self):
        self.Observation = {}
        for name, exp in self.Experiments.items():
            exp.SetObservedBinned()
            self.Observation[name] = exp.GetObservedBinned()

    def SetExpectedWeights(self):
        # Binned modular path: cell_weights x detector_factors are assembled
        # inside the engine contraction (SetBinnedExpectedEvents), so this stage
        # is a no-op — the staged (phi, theta) already fully determine the
        # expectation. Kept in the method vocabulary for call-sequence parity.
        if self._binned_active():
            return
        for name, exp in self.Experiments.items():
            exp.SetExpectedWeight()

    # active energy_scale era dials (event-side histogram-transfer operator).
    _ESCALE_ERA_DIALS = ("energy_scale_sk1", "energy_scale_sk2",
                         "energy_scale_sk3", "energy_scale_sk45")

    def _event_escale_active(self):
        """The 4 energy_scale era dials active in this analysis (event path)."""
        nl = getattr(self.Analysis, "NuisanceList", [])
        return [d for d in self._ESCALE_ERA_DIALS if d in nl]

    def SetBinnedExpectedEvents(self):
        self.Expectation = {}
        # Binned modular path: contract the response for the staged point into
        # the FewEntries-filtered expectation, keyed by the binned experiment.
        if self._binned_active():
            name, binding = self._the_binned_engine()
            self._require_binned_staged()
            # exactly ``n_nu[few]`` inside engine.chi2 for the staged (phi, theta)
            n_nu, _ = binding.engine.expectation(self._binned_staged_phi,
                                                 self._binned_staged_theta)
            self.Expectation[name] = n_nu[binding.engine.few]
            return
        escale = self._event_escale_active()
        for name, exp in self.Experiments.items():
            if escale and self._exp_supports_escale(exp):
                # Event-side energy_scale = the binned histogram-transfer operator
                # (escale_operator.py), applied POST-binning, PRE-FewEntries — the
                # weight-emulation is RETIRED (era wrappers return identity). This
                # replaces exp.SetExpectedBinned's escale-less bin/FewEntries pair
                # with: bin -> migrate -> RemoveFewEntries, matching the binned
                # engine's ordering (rate migrated before the FewEntries mask).
                exp.SetExpectedWeight()  # already called by SetExpectedWeights;
                                         # harmless idempotent re-eval
                hist = exp.BinMC(exp.ExpectedWeight)
                hist = self._apply_event_escale(exp, hist, var=False)
                exp.ExpectedBinned = hist[exp.FewEntries] \
                    if getattr(exp, "FewEntries", None) is not None else hist
                self.Expectation[name] = exp.ExpectedBinned
            else:
                exp.SetExpectedBinned()
                self.Expectation[name] = exp.GetExpectedBinned()

    def _exp_supports_escale(self, exp):
        """The event experiment exposes the geometry the histogram operator needs
        (per-sample E/cz bin edges + a per-event bin/era map). SK does; other
        experiments do not (and never carry energy_scale_sk* dials)."""
        return (hasattr(exp, "EnergyBins") and hasattr(exp, "CTBins")
                and hasattr(exp, "Samples") and hasattr(exp, "Bin")
                and hasattr(exp, "SKPhase"))

    def _escale_operator_for(self, exp):
        """Build (once, cached on exp) the EScaleHistogramOperator for an event
        experiment from its flat 930-bin geometry: sample_table from the per-
        sample (E, cz) bin-edge sizes in Samples order, and per-bin era from the
        MC SKPhase reduced to the sk45-lumped era index. Byte-matches the binned
        engine's geometry because both key the same 930-bin `bin_number` scheme."""
        cached = getattr(exp, "_escale_op", None)
        if cached is not None:
            return cached
        from .binned.escale_operator import (EScaleHistogramOperator,
                                             ERA_TAGS)
        import numpy as _np
        # sample_table: offset accumulates over Samples order; ne=#E bins,
        # nz=#cz bins per sample (edges-1). Mirrors BinIt_MC_2D's concatenation.
        sample_table = {}
        off = 0
        for s in exp.Samples:
            ne_ = int(exp.EnergyBins[s].size - 1)
            nz = int(exp.CTBins[s].size - 1)
            sample_table[int(s)] = (off, ne_, nz)
            off += ne_ * nz
        n_bins = off
        # per-bin era: reduce SKPhase to the era index (sk1..sk3 -> 0..2, >=4 ->3),
        # taking the modal era of the events in each bin (bins are single-era in
        # the SK MC by construction; mode is a safe reducer).
        era_of_phase = {1: 0, 2: 1, 3: 2}
        ev_era = _np.array([era_of_phase.get(int(p), 3) for p in exp.SKPhase])
        bin_era = _np.zeros(n_bins, dtype=_np.int64)
        binidx = _np.asarray(exp.Bin)
        for b in range(n_bins):
            m = binidx == b
            if _np.any(m):
                vals = ev_era[m]
                bin_era[b] = _np.bincount(vals).argmax()
        op = EScaleHistogramOperator(sample_table, n_bins, bin_era)
        op._era_tags = ERA_TAGS
        exp._escale_op = op
        return op

    def _apply_event_escale(self, exp, hist, var=False):
        """Apply the energy_scale histogram transfer to a full (pre-FewEntries)
        binned expectation ``hist`` for one SK event experiment, one delta per
        era from the current nuisance vector. No-op when x==1 for every era."""
        import numpy as _np
        op = self._escale_operator_for(exp)
        nl = self.Analysis.NuisanceList
        theta = self._current_nuisance_vector()
        deltas = _np.array([
            theta[nl.index(f"energy_scale_{op._era_tags[e]}")] - 1.0
            if f"energy_scale_{op._era_tags[e]}" in nl else 0.0
            for e in range(op.n_era)])
        if not _np.any(deltas):
            return hist
        return op.migrate(hist, deltas, var=var)

    def _current_nuisance_vector(self):
        """The nuisance vector currently applied. During a fit the minimizer
        supplies theta to ApplyNuisanceWeights; we cache it there. Falls back to
        the nominal vector (used for the pre-fit nominal expectation)."""
        v = getattr(self, "_last_nuisance_vector", None)
        if v is not None:
            return v
        return self.Analysis.NuisNominalList

    def SetBinnedMCVariance(self):
        """Compute binned MC variance for Barlow-Beeston likelihood."""
        self.MCVariance = {}
        # Binned modular path: var[few] from the engine contraction (BB only;
        # pure Poisson ignores it but the vector is provided for parity).
        if self._binned_active():
            name, binding = self._the_binned_engine()
            self._require_binned_staged()
            # var[few] from the engine contraction (BB only; pure Poisson ignores
            # it but the vector is provided for parity).
            _, var = binding.engine.expectation(self._binned_staged_phi,
                                                self._binned_staged_theta)
            self.MCVariance[name] = var[binding.engine.few]
            return
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
        # Event-side energy_scale gradient: the per-event weight-emulation is
        # retired (dW/W == 0 above), so the histogram operator supplies dE/ddelta
        # = migrate_derivative(N, era) on the current pre-migration binned
        # expectation N (the same linear operator applied to gradient histograms,
        # migration held fixed at first order — binned-engine convention).
        escale = self._event_escale_active()
        if escale:
            for name, exp in self.Experiments.items():
                if not self._exp_supports_escale(exp):
                    continue
                op = self._escale_operator_for(exp)
                N = exp.BinMC(exp.ExpectedWeight)     # pre-migration binned rate
                few = getattr(exp, "FewEntries", None)
                for e in range(op.n_era):
                    dial = f"energy_scale_{op._era_tags[e]}"
                    if dial not in self.Analysis.NuisanceList:
                        continue
                    dE = op.migrate_derivative(N, e)
                    dEdx[dial] = {name: dE[few] if few is not None else dE}
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
        # Binned modular path: select the oscillation tensor slice for this grid
        # point's (Dm231, Sin2Theta23) at the currently-targeted dCP node. The
        # worker profiles dCP by SetBinnedDcpNode() before each objective.
        if self._binned_active():
            _, binding = self._the_binned_engine()
            dm231 = self._binned_phys_value(point, "Dm231")
            s23 = self._binned_phys_value(point, "Sin2Theta23")
            # select the oscillation tensor slice at the targeted dCP node; the
            # worker profiles dCP by SetBinnedDcpNode() before each objective.
            phi = binding.store.phi(dm231, s23)
            self._binned_staged_phi = phi[self._binned_dcp_node].astype(float)
            return
        self.ApplyWeights("Physics", vector=self.Analysis.FullPhysicsGrid[point])

    def ApplyNuisanceWeights(self, vector):  # Physics parameters
        if self.verbosity:
            print("Applying Nuisance Weights")
        # Binned modular path: stage theta for the response contraction.
        if self._binned_active():
            self._binned_staged_theta = np.asarray(vector, dtype=float)
            return
        # cache the current theta so the event-side energy_scale histogram
        # operator (SetBinnedExpectedEvents) can read its per-era deltas.
        self._last_nuisance_vector = vector
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
                    for tune in labels[source]:
                        # Per-dial owner lookup (routing decision (b)): resolve
                        # the tune-block from the tune-class method inventory,
                        # falling back to the legacy exp.Definition[source] for
                        # anything the map does not own (Osc dials). Identical to
                        # the old block scheme for legacy 3-block XMLs.
                        tune_block = self._dial_block(tune, source, exp)
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
                    for tune in self.Analysis.Nuisance[source]:
                        # per-dial owner lookup (see ApplyWeights)
                        tune_block = self._dial_block(tune, source, exp)
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

    def _binned_chi2_and_grad(self):
        """(f, g) at the staged (phi, theta) via the engine's analytic kernel —
        the modular gradient path. Bit-identical to a direct
        ``engine.chi2_and_grad(phi_slice, theta)``. (Track S / E6: was the
        adapter's chi2_and_grad_binned; the staged state now lives on self.)"""
        _, binding = self._the_binned_engine()
        self._require_binned_staged()
        return binding.engine.chi2_and_grad(self._binned_staged_phi,
                                            self._binned_staged_theta)

    def set_likelihood(self, mode):
        if mode == "PoissonLikelihood":
            # Binned modular path: pure-Poisson LLH whose statistics kernel is
            # the engine's poisson_chi2 (design §2.4). Observation is the
            # engine's FewEntries-filtered obs_f, keyed by the binned experiment.
            name, binding = self._the_binned_engine()
            self.LLH = ft.PoissonLikelihood(
                {name: binding.observed_binned()},
                self.Analysis.NuisNominalList,
                self.Analysis.NuisSigmaList,
                self.Analysis.NuisDistributionList,
            )
            self.LLH.set_engine(binding.engine)
        elif mode == "BinnedLogLikelihoodRatio":
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

        # Native binned-tensor engine takes over when a <BinnedEngine> block is
        # active (default-OFF: BinnedEngines is {} -> falsy -> this guard is a no-op).
        if getattr(self, "BinnedEngines", None):
            return self.FitModelBinned(point, mode=mode)

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

    # ---------------- native binned-tensor engine (default-OFF toggle) ----------------
    # Additive methods: never reached unless an analysis XML declares an enabled
    # <BinnedEngine> block (parsed independently of ParseXML). See pynu/binned/.

    _MODE_TO_LIKELIHOOD = {
        "BinnedLogLikelihoodRatio": "poisson",
        "BarlowBeestonLikelihood": "bb",
        "PoissonLikelihood": "poisson",
    }

    def _setup_binned_engines(self, analysis_file):
        """Return {experiment_name: loaded BinnedBinding} for the XML's enabled
        <BinnedEngine> blocks, or {} (the toggle-OFF default). Lazy: no
        pynu.binned forward-model code runs when the XML has no such block."""
        from .analysis_reader.binned_config import parse_binned_config
        configs = parse_binned_config(analysis_file)
        if not configs:
            return {}
        from .binned.engine_core import BinnedBinding
        return {
            name: BinnedBinding.load(cfg, analysis_xml=analysis_file)
            for name, cfg in configs.items()
        }

    def set_binned_engine(self, exp_name, config, analysis_xml=None):
        """Programmatic opt-in (overrides XML): attach a loaded BinnedBinding
        for `exp_name` from a BinnedConfig. Enables FitModelBinned for this fit.
        Pass analysis_xml when the config uses nuisance_spec='self'."""
        from .binned.engine_core import BinnedBinding
        if getattr(self, "BinnedEngines", None) is None:
            self.BinnedEngines = {}
        self.BinnedEngines[exp_name] = BinnedBinding.load(
            config, analysis_xml=analysis_xml)
        return self.BinnedEngines[exp_name]

    def _binned_phys_value(self, point, name):
        """(Dm231, Sin2Theta23) etc. for a grid point: from FullPhysicsGrid via
        PhysicsList when scanned, else the fixed value."""
        pl = self.Analysis.PhysicsList
        if name in pl:
            return float(self.Analysis.FullPhysicsGrid[point][pl.index(name)])
        for pars in self.Analysis.FixedValue.values():
            if name in pars:
                return float(pars[name])
        raise KeyError(f"binned engine: {name} is neither a scanned physics nor a "
                       "fixed parameter in this analysis")

    def FitModelBinned(self, point, mode=None):
        """Binned-mode fit for one physics grid point. Resolves (Dm231,
        Sin2Theta23) from the analysis grid, runs the engine's dCP-profiled
        L-BFGS-B fit via the binding, writes the chi2 to the h5 output, and dumps
        the engine dial vector to a per-point sidecar JSON (engine dial names do
        not fit the h5 Nuisance Parameters/<source>/<par> hierarchy)."""
        if not self.Analysis.do_point(point):
            print(f"Skipping point {point}.")
            return False

        if len(self.BinnedEngines) != 1:
            raise NotImplementedError(
                "binned engine phase 1 supports exactly one binned experiment "
                f"(got {len(self.BinnedEngines)}); mixed event+binned is phase 2")
        binding = next(iter(self.BinnedEngines.values()))

        # phase-1 physics scope: normal ordering, no CPT (Dm231_bar)
        if "Ordering" in self.Analysis.PhysicsList:
            raise NotImplementedError("binned engine: scanning Ordering is phase 2")
        for pars in self.Analysis.FixedValue.values():
            if "Ordering" in pars and str(pars["Ordering"]).strip() != "normal":
                raise NotImplementedError(
                    "binned engine phase 1 assumes normal ordering; got "
                    f"{str(pars['Ordering']).strip()!r}")
        if ("Dm231_bar" in self.Analysis.PhysicsList
                or "Dm231_bar" in self.Analysis.FixedList):
            raise NotImplementedError("binned engine: CPT (Dm231_bar) is phase 2")

        # likelihood: engine config authoritative; error on an explicit conflict
        if mode is not None:
            want = self._MODE_TO_LIKELIHOOD.get(mode)
            if want is not None and want != binding.config.likelihood:
                raise ValueError(
                    f"binned engine likelihood {binding.config.likelihood!r} "
                    f"conflicts with FitModel mode {mode!r} (={want!r}); make the "
                    "<BinnedEngine> <likelihood> and the fit mode agree")

        dm231 = self._binned_phys_value(point, "Dm231")
        s23 = self._binned_phys_value(point, "Sin2Theta23")

        self.point = point
        chi2_min, dcp_idx, theta, nit, conv = binding.fit_point(dm231, s23)
        chi2_stats = binding.chi2(dm231, s23, binding.nominal)
        print(f"Binned point {point}: dm231={dm231:.6e} s23={s23:.4f} "
              f"chi2={chi2_min:.6f} (stats-only {chi2_stats:.6f}) "
              f"dcp_idx={dcp_idx} nit={nit} conv={conv}")

        self.WriteToOutFile("Analysis", "Chi2 Stats. Only", chi2_stats)
        if self.Analysis.wSyst:
            self.WriteToOutFile("Analysis", "Chi2 Systs.", chi2_min)
        self._dump_binned_sidecar(point, binding, dm231, s23,
                                  chi2_min, dcp_idx, theta, nit, conv)
        return chi2_min

    def _dump_binned_sidecar(self, point, binding, dm231, s23,
                             chi2, dcp_idx, theta, nit, conv):
        """Per-point JSON next to the h5 output carrying the engine dial vector."""
        import json
        base = os.path.splitext(self.outfile)[0] if getattr(self, "outfile", None) \
            else "binned"
        path = f"{base}.binned_point{point:04d}.json"
        with open(path, "w") as f:
            json.dump({
                "point": int(point),
                "dm231": float(dm231),
                "sin2theta23": float(s23),
                "chi2": float(chi2),
                "chi2_stats_only": float(binding.chi2(dm231, s23, binding.nominal)),
                "best_dcp_index": int(dcp_idx),
                "nit": int(nit),
                "converged": bool(conv),
                "likelihood": binding.config.likelihood,
                "osc_averaging": binding.config.osc_averaging,
                "nuisance_names": list(binding.nuisance_names),
                "nuisance": [float(v) for v in theta],
            }, f)
        return path

    def BuildBinnedResponse(self, exp_name=None, out_path=None,
                            n_etrue=200, n_cztrue=40):
        """Build the SK binned forward-model response from this object's own
        event MC (native port of ``build_sk_response.py`` — the standalone
        script stays byte-untouched for cluster SLURM submissions).

        One MC pass THROUGH the live ``exp_name`` experiment, so every convention
        (NC w_no fix, NORM, WMC, CC-mask encoding, the DIS |Mode|>25*CC quirk) is
        inherited, never re-implemented; class signatures come from evaluating
        the actual ``WaterXSection`` tunes at x=2. Caches the sparse-response dict
        on ``self.BinnedResponse[exp_name]`` and returns it. With ``out_path``,
        also writes an npz byte-compatible with the ``SKBinnedEngine`` loader
        (plus the additive schema_version / dial-manifest-hash keys the engine
        checks only when present).

        SLURM fan-out convention: the response is a single per-experiment
        artifact — one node builds it, the submission script stages the npz for
        the fitter fan-out. This method is that per-node kernel.

        Artifact schema (npz keys): classes / xsec_tune_names; the R / Rp / Rm /
        S2 sparse COO quintuples (k,e,z,b,era,v); n_era; e_edges / z_edges /
        n_bins; observed; sample_table; sample_event_counts; meta; and the
        additive schema_version [+ dial_manifest, dial_manifest_hash].

        Args:
          exp_name: which experiment to build for (default: the single
            experiment, matching the standalone script's ``keys()[0]``).
          out_path: optional npz output path.
          n_etrue, n_cztrue: true-grid density (production 200x40 default; the
            engine's production response uses 400x40 — pass n_etrue=400 to match).
        """
        from .binned.builder import build_response
        if exp_name is None:
            exp_name = next(iter(self.Experiments))
        manifest = list(getattr(self.Analysis, "NuisanceList", []))
        resp = build_response(
            self, exp_name, out_path=out_path,
            n_etrue=n_etrue, n_cztrue=n_cztrue,
            dial_manifest=manifest or None)
        if getattr(self, "BinnedResponse", None) is None:
            self.BinnedResponse = {}
        self.BinnedResponse[exp_name] = resp
        return resp

    def BuildOscTensors(self, dm231, s23, exp_name=None, dcp_nodes=None,
                        s13=None, n_etrue=200, n_cztrue=40, avg_scale=None,
                        out_path=None):
        """Build the oscillated-flux tensor Phi[n_dcp, 2, 3, nE, nZ] at
        (``dm231``, ``s23``) from this object's live oscillation handler (native
        port of ``build_osc_tensors.py`` — the standalone script stays
        byte-untouched for cluster SLURM submissions).

        Runs against the production ``AtmosphericOscillations`` object so
        propagation, units, flux init, and the Dm231_bar->Dm231 convention are
        inherited. The osc object's per-event coordinate arrays, ``Parameters``,
        and cache are snapshotted before the build and restored afterwards (even
        on a mid-build exception), so a subsequent event-engine call on this same
        PyNuFit object is byte-unaffected.

        ``avg_scale`` is consumed from the XML ``<osc_averaging>`` field when
        available (see below); ``PYNU_OSC_AVG_SCALE`` in the environment still
        overrides, matching the AtmOsc ctor's back-compat rule. The default
        pulls the declaration off an active ``<BinnedEngine>`` block if present.

        SLURM fan-out convention: the (dm231, s23) grid is decomposed into one
        task per node; each node calls this once and writes ``osc_tensor_<i>_<j>``
        — this method is that per-node tensor kernel.

        Returns ``(phi, meta)``; ``meta`` carries the grid edges, the dcp node
        array, and the averaging actually applied.

        Args:
          dm231, s23: node oscillation parameters.
          exp_name: which experiment (default: the single experiment).
          dcp_nodes: iterable of dCP node values (radians); None -> single dcp=0.
          s13: optional Sin2Theta13 override.
          n_etrue, n_cztrue: true-grid density (must match the response build).
          avg_scale: fast-oscillation averaging selector; None -> the active
            <BinnedEngine> <osc_averaging> declaration (or the osc object's
            current setting) is used.
          out_path: optional npz output path (schema-compatible tensor loader).
        """
        from .binned.builder import build_tensors
        if exp_name is None:
            exp_name = next(iter(self.Experiments))
        if avg_scale is None:
            # default: honour an active <BinnedEngine> <osc_averaging> declaration
            bindings = getattr(self, "BinnedEngines", None) or {}
            b = bindings.get(exp_name)
            if b is not None:
                decl = getattr(b.config, "osc_averaging", "off")
                if str(decl).strip().lower() not in ("off", "", "none"):
                    avg_scale = decl
        return build_tensors(
            self, exp_name, dm231, s23, dcp_nodes=dcp_nodes, s13=s13,
            n_etrue=n_etrue, n_cztrue=n_cztrue, avg_scale=avg_scale,
            out_path=out_path)
