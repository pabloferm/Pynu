"""BinnedEngineAdapter — bridges a PyNuFit (Dm231, Sin2Theta23) physics point to
the standalone SK binned forward model + minimizer.

This is the only genuinely new numerical code in ``pynu.binned``; the engine and
interpolator are verbatim snapshots (see ``PROVENANCE.md``). The adapter adds NO
numerical behaviour beyond node-lookup / cubic interpolation of the pre-built osc
tensors — the gate-2 adapter-vs-direct parity test proves the ``chi2`` / fit
values are bit-for-bit those of a direct engine call with identical inputs.

Import note: the ``try/except`` below lets this module load both as a package
submodule (``pynu.binned.adapter``, the production path) and as a bare top-level
module (``sys.path`` -> ``Pynu/pynu/binned``), which is how the local gate tests
exercise it without the heavy ``pynu`` import chain (nuSQuIDS, event MC).
"""
import glob
import os

import numpy as np

try:                                                        # package context
    from .sk_binned_engine import SKBinnedEngine, resolve_nuisance_spec, CANONICAL_DIALS
    from .interp_engine import PhiInterpolator, detect_grid
except ImportError:                                         # bare sys.path context
    from sk_binned_engine import SKBinnedEngine, resolve_nuisance_spec, CANONICAL_DIALS
    from interp_engine import PhiInterpolator, detect_grid


class BinnedEngineAdapter:
    """(Dm231, Sin2Theta23) -> profiled binned chi2 via the SK binned engine.

    Construct with a ``BinnedConfig`` (and the analysis XML path when
    ``nuisance_spec='self'``), call :meth:`load`, then :meth:`fit_point` /
    :meth:`chi2`. The oscillation tensor ``phi[n_dcp, 2, 3, nE, nZ]`` is looked
    up per physics point (exact grid node by default, cubic interpolation opt-in)
    and handed to the engine's dCP-profiled L-BFGS-B minimizer.
    """

    def __init__(self, config, analysis_xml=None):
        self.config = config
        self.analysis_xml = analysis_xml
        self.engine = None
        self.interp = None
        self.DM = None
        self.S23 = None
        self._spec = None
        self._phi_cache = None      # 1-slot ((i, j) -> phi) cache for the node path
        # ---- modular-path staged state. None until the modular method
        # vocabulary is used; the packaged fit_point path never touches
        # these, so the convenience path is unchanged. ----
        self._staged_phi = None     # dCP slice selected by apply_physics()
        self._staged_theta = None   # nuisance vector staged by stage_nuisance()

    # ---- nuisance-spec resolution ----
    def _resolve_spec(self):
        """Map the config selector to what ``resolve_nuisance_spec`` accepts.

        'self' -> the analysis XML path (routed to ``_parse_xml_active``); any
        other string ('barr'/'R2'/'phased'/... or an explicit .xml path) or a
        list passes through unchanged; None -> engine default 41-vector.
        """
        spec = self.config.nuisance_spec
        if spec == "self":
            if not self.analysis_xml:
                raise ValueError(
                    "binned engine nuisance_spec='self' requires the analysis "
                    "XML path (pass analysis_xml=...)")
            return self.analysis_xml
        if spec in ("", "barr", None):
            return None
        return spec

    def load(self):
        """Build the engine (validates the nuisance spec against the response
        build) and read the tensor grid axes. Returns self."""
        cfg = self.config
        self._spec = self._resolve_spec()
        self.engine = SKBinnedEngine(
            cfg.response,
            migration_mode=cfg.migration,
            likelihood=cfg.likelihood,
            nuisance_spec=self._spec,
        )
        self.DM, self.S23 = detect_grid(cfg.tensors)
        if cfg.interp == "cubic":
            self.interp = PhiInterpolator(cfg.tensors, cache_raw=False)
        elif cfg.interp != "nodes":
            raise ValueError(f"unknown interp {cfg.interp!r} (expected 'nodes' or 'cubic')")
        # osc_averaging is a PROVENANCE declaration of the averaging baked into the
        # tensor build; warn-only validation (current tensor sets carry no metadata).
        self.osc_averaging = cfg.osc_averaging
        for w in self._validate_osc_averaging():
            print(f"WARNING [binned osc_averaging]: {w}")
        return self

    def _validate_osc_averaging(self):
        """Warn-only cross-check of the ``<osc_averaging>`` declaration against
        tensor-set metadata. Current builds embed no averaging key, so this only
        records provenance; once a build embeds one (e.g. an 'osc_averaging' array
        in the npz), a mismatch warns. Peeks at a single tensor (keys only)."""
        warns = []
        decl = str(self.config.osc_averaging).strip().lower()
        files = sorted(glob.glob(os.path.join(self.config.tensors,
                                              "osc_tensor_*_*.npz")))
        meta = None
        if files:
            with np.load(files[0], allow_pickle=False) as z:
                for key in ("osc_averaging", "avg_scale", "averaging"):
                    if key in z.files:
                        meta = str(z[key])
                        break
        if meta is None:
            if decl not in ("off", "", "none"):
                warns.append(
                    f"declared '{decl}' but the tensor set carries no averaging "
                    "metadata -> recorded as provenance only, unverified")
        elif meta.strip().lower() != decl:
            warns.append(f"declaration '{decl}' != tensor metadata '{meta}'")
        return warns

    # ---- phi lookup ----
    def _node_index(self, axis, value, label):
        """Exact grid-node index for ``value`` on ``axis`` (rtol 1e-9). Errors
        with the nearest node + neighbours when the request is off-grid."""
        axis = np.asarray(axis)
        idx = int(np.argmin(np.abs(axis - value)))
        if not np.isclose(axis[idx], value, rtol=1e-9, atol=0.0):
            lo = max(0, idx - 1)
            near = ", ".join(f"{a:.6g}" for a in axis[lo:idx + 2])
            raise ValueError(
                f"binned engine interp='nodes': {label}={value:.6g} is not a "
                f"grid node (nearest {axis[idx]:.6g}; neighbours [{near}]). "
                "Use physics-grid edges that match the tensor build, or set "
                "interp='cubic'.")
        return idx

    def phi(self, dm231, s23):
        """Oscillation tensor ``phi[n_dcp, 2, 3, nE, nZ]`` at (dm231, s23).

        The node path caches the last-loaded (i, j) so a per-cell restart-polish
        loop (repeated fit_point at the same node) reads the npz once, not once
        per call. The cached array is returned as-is; the engine copies per dCP
        slice (`phi[di].astype(float)`) and never mutates it."""
        if self.config.interp == "cubic":
            return self.interp(dm231, s23)
        i = self._node_index(self.DM, dm231, "Dm231")
        j = self._node_index(self.S23, s23, "Sin2Theta23")
        if self._phi_cache is not None and self._phi_cache[0] == (i, j):
            return self._phi_cache[1]
        p = os.path.join(self.config.tensors, f"osc_tensor_{i:03d}_{j:03d}.npz")
        arr = np.load(p)["phi"]
        self._phi_cache = ((i, j), arr)
        return arr

    # ---- objectives ----
    def chi2(self, dm231, s23, theta, dcp_index=None):
        """Binned chi2 at fixed nuisance vector ``theta``. ``dcp_index`` selects
        a single dCP slice; None profiles (min) over all dCP nodes."""
        phi = self.phi(dm231, s23)
        if dcp_index is not None:
            return float(self.engine.chi2(phi[dcp_index].astype(float), theta))
        return float(min(self.engine.chi2(phi[d].astype(float), theta)
                         for d in range(phi.shape[0])))

    def fit_point(self, dm231, s23, x0=None, free_mask=None):
        """dCP-profiled nuisance fit. Returns the engine tuple
        ``(chi2, best_dcp_index, theta, nit, converged)``."""
        phi = self.phi(dm231, s23)
        return self.engine.fit_point(phi, x0=x0, free_mask=free_mask)

    # ---- modular method-vocabulary state-holder ----
    # These back the PyNuFit modular methods (StartNuisance / ApplyPhysicsWeights
    # / ApplyNuisanceWeights / SetExpectedWeights / SetBinnedExpectedEvents /
    # SetBinnedMCVariance) so a worker can drive the binned engine through the
    # same call sequence as the event engine. The engine kernels are unchanged;
    # this is purely staging + a call into engine.expectation / poisson_chi2.
    def start_nuisance(self):
        """Reset the staged nuisance vector (cheap; mirrors the event engine's
        per-event weight-array reset)."""
        self._staged_theta = None

    def apply_physics(self, dm231, s23, dcp_index):
        """Select the oscillation tensor slice for (dm231, s23) at a single dCP
        node. Stores the (2,3,nE,nZ) slice the engine expectation consumes.
        The worker profiles dCP by looping ``dcp_index`` (same structure as the
        event engine's worker-level node scan)."""
        phi = self.phi(dm231, s23)
        self._staged_phi = phi[dcp_index].astype(float)
        return self._staged_phi

    def stage_nuisance(self, theta):
        """Stage the nuisance vector for the next contraction."""
        self._staged_theta = np.asarray(theta, dtype=float)
        return self._staged_theta

    @property
    def n_dcp(self):
        """Number of dCP nodes in the staged tensor grid (for the worker loop)."""
        return int(self.phi(self.DM[0], self.S23[0]).shape[0])

    def _require_staged(self):
        if self._staged_phi is None:
            raise RuntimeError("binned modular path: apply_physics() must run "
                               "before the expectation is contracted")
        if self._staged_theta is None:
            raise RuntimeError("binned modular path: stage_nuisance() must run "
                               "before the expectation is contracted")

    def expected_binned(self):
        """Contract the response to the FewEntries-filtered expectation for the
        staged (phi, theta) — exactly ``n_nu[few]`` inside ``engine.chi2``."""
        self._require_staged()
        n_nu, _ = self.engine.expectation(self._staged_phi, self._staged_theta)
        return n_nu[self.engine.few]

    def mc_variance_binned(self):
        """FewEntries-filtered MC variance ``var[few]`` for the staged point
        (used only by the BB likelihood; pure Poisson ignores it)."""
        self._require_staged()
        _, var = self.engine.expectation(self._staged_phi, self._staged_theta)
        return var[self.engine.few]

    def observed_binned(self):
        """The engine's FewEntries-filtered observation vector ``obs_f`` — the
        modular likelihood's observation for this experiment."""
        return self.engine.obs_f

    def chi2_and_grad_binned(self):
        """(f, g) at the staged (phi, theta) via the engine's analytic kernel —
        the modular gradient path. Bit-identical to a direct
        ``engine.chi2_and_grad(phi_slice, theta)``."""
        self._require_staged()
        return self.engine.chi2_and_grad(self._staged_phi, self._staged_theta)

    def chi2_binned(self):
        """chi2 at the staged (phi, theta) via the engine kernel (parity ref)."""
        self._require_staged()
        return float(self.engine.chi2(self._staged_phi, self._staged_theta))

    # ---- XML nuisance cross-check ----
    def crosscheck_xml_nuisances(self, analysis):
        """Compare the XML-active nuisances (``analysis.NuisanceList``) against
        the engine's ``CANONICAL_DIALS`` for the overlap. Returns a list of
        warning strings for (nominal, sigma) mismatches; raises on a non-normal
        distribution (the engine penalty is pure Gaussian). ``muon_norm`` is
        ignored (SK has no muon background; the engine owns no such dial)."""
        warnings = []
        names = list(getattr(analysis, "NuisanceList", []))
        nominals = list(getattr(analysis, "NuisNominalList", []))
        sigmas = list(getattr(analysis, "NuisSigmaList", []))
        dists = list(getattr(analysis, "NuisDistributionList", []))
        for i, name in enumerate(names):
            if name == "muon_norm" or name not in CANONICAL_DIALS:
                continue
            if i < len(dists) and dists[i] != "normal":
                raise ValueError(
                    f"binned engine: nuisance {name!r} has distribution "
                    f"{dists[i]!r}; only 'normal' is supported (engine penalty "
                    "is pure Gaussian)")
            eng_nom, eng_sig = CANONICAL_DIALS[name]
            if i < len(nominals) and not np.isclose(nominals[i], eng_nom, rtol=0.0, atol=1e-9):
                warnings.append(
                    f"{name}: XML nominal {nominals[i]:g} != engine {eng_nom:g}")
            if i < len(sigmas) and not np.isclose(sigmas[i], eng_sig, rtol=0.0, atol=1e-9):
                warnings.append(
                    f"{name}: XML sigma {sigmas[i]:g} != engine {eng_sig:g}")
        return warnings

    # ---- engine nuisance vectors ----
    @property
    def nuisance_names(self):
        return self.engine.nuisance_names

    @property
    def nominal(self):
        return self.engine.nominal

    @property
    def sigma(self):
        return self.engine.sigma
