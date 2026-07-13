#!/usr/bin/env python3
"""BinnedExperiment — an ``Experiment`` whose forward model is the native SK
binned-tensor engine (Track S·F, Phase F2, decision D-2).

Where the event ``Experiment`` (``Experiment.py``) holds per-event MC arrays and
builds an expectation by re-weighting + histogramming, a ``BinnedExperiment``
holds a loaded ``SKBinnedEngine`` + oscillation-tensor ``TensorStore`` (via the
``BinnedBinding`` triple it OWNS) and builds its expectation by contracting the
pre-built response for a staged oscillation point. It implements the SAME base
vocabulary (``StartNuisanceWeights`` / ``UpdatePhysicsWeights`` /
``UpdateNuisanceWeights`` / ``SetExpectedBinned`` / ``SetObservedBinned`` /
``GetObservedBinned`` / ``GetExpectedBinned``) with binned semantics, so the
``if self._binned_active():`` branches that used to live inside every PyNuFit
modular method dissolve into ordinary polymorphism (§4 branch-dissolution table).

Binding constraint F-C1 (assemble-level dispatch): there is NO per-dial mode
dispatch here. ``StartNuisanceWeights`` and the staging methods only STAGE the
(phi, theta) point; the full fused expectation assembly is done once in
``SetExpectedBinned`` by ``engine.expectation`` — the single seam. Per-dial
formula evaluation lives inside the engine's fused kernels (already certified).

Mode is per-experiment (decision D-1): a ``BinnedExperiment`` is constructed
only for an experiment whose analysis XML declares an enabled ``<BinnedEngine>``
block; event-mode experiments keep their event ``Manager`` untouched.

This class deliberately does NOT re-run ``Experiment.__init__`` (no event MC to
read): it constructs directly from a loaded ``BinnedBinding``. It still IS an
``Experiment`` (isinstance holds) so PyNuFit can treat it uniformly in the fit
loop; the event-only helpers (binning, KDE, ``Reader``) are simply never called
on it.
"""
import numpy as np

from .Experiment import Experiment


class BinnedExperiment(Experiment):
    """Experiment subclass backed by a ``BinnedBinding`` (engine + tensor store
    + config). Composition: the binding OWNS what the former ``BinnedBinding``
    held in PyNuFit's ``self.BinnedEngines``; this class adds the staged
    (phi, theta, dcp-node) state that used to be PyNuFit singletons
    (``_binned_staged_phi`` / ``_binned_staged_theta`` / ``_binned_dcp_node``),
    now per-experiment so multiple binned experiments can stage independently.
    """

    # Marker so PyNuFit can partition its experiment dict into event vs binned
    # without importing this module at type-check sites (duck-typed guard).
    is_binned = True

    def __init__(self, binding, name=None):
        # NOTE: intentionally do NOT call Experiment.__init__ — a BinnedExperiment
        # reads no event MC. The binding already carries the loaded engine/store.
        self.binding = binding
        self.name = name
        # staged oscillation point (was PyNuFit._binned_staged_phi / _theta):
        self._staged_phi = None            # phi[dcp] slice at (dm, s23)
        self._staged_theta = None          # nuisance vector staged for the fit
        # dCP node the modular staging currently targets (was
        # PyNuFit._binned_dcp_node); the worker profiles dCP by setting this
        # before each objective (default 0).
        self._dcp_node = 0
        # binned expectation vectors, filled by SetExpectedBinned/MCVariance.
        self.ExpectedBinned = None
        self.MCVarianceBinned = None

    # ---- convenience passthroughs to the engine/store (former binding surface) ----
    @property
    def engine(self):
        return self.binding.engine

    @property
    def store(self):
        return self.binding.store

    @property
    def config(self):
        return self.binding.config

    # ---- guards ----
    def _require_staged(self):
        """ApplyPhysicsWeights + ApplyNuisanceWeights must have staged (phi, theta)
        before the expectation is contracted (mirrors the former PyNuFit guard)."""
        if self._staged_phi is None:
            raise RuntimeError("binned modular path: ApplyPhysicsWeights() must "
                               "run before the expectation is contracted")
        if self._staged_theta is None:
            raise RuntimeError("binned modular path: ApplyNuisanceWeights() must "
                               "run before the expectation is contracted")

    # ---- base vocabulary, binned semantics (branch-dissolution targets) ----
    def StartNuisanceWeights(self):
        """Reset the staged nuisance vector (§4: StartNuisance branch). Cheap;
        mirrors the event engine's per-event weight-array reset."""
        self._staged_theta = None

    def SetExpectedWeight(self):
        """No-op on the binned path (§4: SetExpectedWeights no-op branch): the
        cell_weights x detector_factors assembly happens inside the engine
        contraction (SetExpectedBinned), so the staged (phi, theta) already
        fully determine the expectation."""
        return

    def SetBinnedDcpNode(self, dcp_index):
        """Select the dCP node the modular staging contracts at (worker-level dCP
        profile scan, same structure as the event engine's node loop)."""
        self._dcp_node = int(dcp_index)

    def StagePhysicsPoint(self, dm231, s23, dcp_index=None):
        """Stage phi[dcp_index] at (dm231, s23) directly from the tensor store.
        This is the exact numerical staging ApplyPhysicsWeights does on the
        binned path (§4: ApplyPhysicsWeights branch), keyed on explicit
        (dm231, s23). ``dcp_index=None`` uses the currently-selected node."""
        if dcp_index is not None:
            self._dcp_node = int(dcp_index)
        phi = self.store.phi(dm231, s23)
        self._staged_phi = phi[self._dcp_node].astype(float)
        return self._staged_phi

    def UpdateNuisanceWeights(self, w):
        """Stage the nuisance vector for the response contraction (§4:
        ApplyNuisanceWeights branch). On the event path this multiplies a weight
        array; on the binned path ``w`` IS the full theta vector the fused
        assembly consumes, so it replaces rather than accumulates."""
        self._staged_theta = np.asarray(w, dtype=float)

    def UpdatePhysicsWeights(self, w):
        """Physics staging on the binned path is done by StagePhysicsPoint (phi
        selection from the tensor store), driven by PyNuFit's ApplyPhysicsWeights
        which resolves (dm231, s23) uniformly. The event ``w`` weight vector has
        no binned counterpart (F-C1: no per-dial dispatch), so this is a no-op
        kept for call-sequence parity."""
        return

    def StartPhysicsWeights(self):
        """No-op: physics on the binned path is the staged phi slice, reset when
        a new point is staged. Kept for call-sequence parity with the event
        StartPhysics loop."""
        return

    def SetExpectedBinned(self):
        """Fused binned expectation (§4: SetBinnedExpectedEvents branch; the
        F-C1 seam). Contract the response for the staged (phi, theta) into the
        FewEntries-filtered expectation vector."""
        self._require_staged()
        n_nu, var = self.engine.expectation(self._staged_phi, self._staged_theta)
        self.ExpectedBinned = n_nu[self.engine.few]
        # variance rides along the same contraction (BB path); pure Poisson
        # ignores it but the vector is provided for parity (§4: SetBinnedMCVariance).
        self.MCVarianceBinned = var[self.engine.few]
        return self.ExpectedBinned

    def SetBinnedMCVariance(self):
        """Return the FewEntries-filtered MC variance (§4: SetBinnedMCVariance
        branch). Reuses the last SetExpectedBinned contraction when present, else
        recontracts the staged point (BB only; Poisson ignores it)."""
        if self.MCVarianceBinned is not None:
            return self.MCVarianceBinned
        self._require_staged()
        _, var = self.engine.expectation(self._staged_phi, self._staged_theta)
        self.MCVarianceBinned = var[self.engine.few]
        return self.MCVarianceBinned

    def SetObservedBinned(self):
        """Read the response data vector (§4: SetBinnedObservedEvents branch —
        the base name already fits). The engine's FewEntries-filtered obs_f is
        the observation the fit compares against."""
        self.ObservedBinned = self.binding.observed_binned()
        return self.ObservedBinned

    def GetObservedBinned(self):
        return self.ObservedBinned

    def GetExpectedBinned(self):
        return self.ExpectedBinned

    # ---- binned-specific methods backing PyNuFit ----
    def chi2_and_grad(self):
        """(f, g) at the staged (phi, theta) via the engine's analytic kernel —
        the modular gradient path (§4: SetBinnedDiffExpectedEvents / analytic-
        gradient branch). Bit-identical to a direct
        ``engine.chi2_and_grad(phi_slice, theta)``."""
        self._require_staged()
        return self.engine.chi2_and_grad(self._staged_phi, self._staged_theta)

    # ---- read-only passthroughs (former BinnedBinding / adapter surface) ----
    @property
    def DM(self):
        return self.binding.DM

    @property
    def S23(self):
        return self.binding.S23

    @property
    def n_dcp(self):
        return self.binding.n_dcp

    @property
    def nuisance_names(self):
        return self.binding.nuisance_names

    @property
    def nominal(self):
        return self.binding.nominal

    @property
    def sigma(self):
        return self.binding.sigma

    def phi(self, dm231, s23):
        return self.binding.phi(dm231, s23)

    def observed_binned(self):
        return self.binding.observed_binned()

    def chi2(self, dm231, s23, theta, dcp_index=None):
        return self.binding.chi2(dm231, s23, theta, dcp_index=dcp_index)

    def fit_point(self, dm231, s23, x0=None, free_mask=None):
        return self.binding.fit_point(dm231, s23, x0=x0, free_mask=free_mask)
