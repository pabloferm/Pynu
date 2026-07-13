#!/usr/bin/env python3
"""Binned per-point fit protocol + tensor-store / binding (Track S·F, Phase F3).

Re-homed here from ``pynu/binned/engine_core.py`` into ``pynu.fitter.minimizer``
— the functional home of the production per-point minimizer protocol and the
grid-node φ lookup that feeds it. THREE objects move verbatim (ZERO numerical
change; every guard, ordering, epsilon and comment preserved):

  * ``fit_point(eng, ...)`` — the dCP-profiled L-BFGS-B nuisance minimization
    (the production per-point protocol). Takes the engine instance as its first
    argument (explicit state) exactly as before; ``SKBinnedEngine.fit_point`` is
    the one-line delegate. ``engine_core`` re-imports it for its own delegate.
  * ``TensorStore`` — (Δm², s²θ₂₃) -> φ tensor grid lookup / caching.
  * ``BinnedBinding`` — the loaded (engine, store, config) triple for ONE binned
    experiment; ``PyNuFit`` and ``BinnedExperiment`` construct via
    ``BinnedBinding.load``. Re-exported through ``pynu.binned`` for back-compat.

The fit-time box dicts (``FLUX_RATIO_BOX`` … ``DIR_SMEAR_BOX``) are imported from
the engine module exactly as the former in-``engine_core`` code referenced them;
``PhiInterpolator`` / ``detect_grid`` are imported (method-local) from their new
home ``pynu.fitter.inference.interp_engine``; ``SKBinnedEngine`` is imported
(method-local) inside ``BinnedBinding.load`` as before. Method-local heavy imports
keep this module import-light (F-C2).
"""
import glob
import os

import numpy as np
from scipy.optimize import minimize

# fit-time box dicts the moved ``fit_point`` references from the engine module's
# globals (the truncation limits for the optional dials). Unchanged from the
# former ``engine_core`` import.
from ...binned.sk_binned_engine import (
    FLUX_RATIO_BOX,
    FLUX_BAND_NAMES,
    XSEC_EXTRA_BOX,
    MULTIGEV_CCQE_BOX,
    NEUTRON_MIG_BOX_PINNED,
    DIR_SMEAR_NAME,
    DIR_SMEAR_BOX,
)


# ---------------- L-BFGS-B nuisance bounds (production box) ----------------
def build_nuisance_bounds(nominal, sigma, names, free_mask=None):
    """L-BFGS-B bounds for the binned nuisance vector — the single source of the
    production box (Track S·F / F5: this is what worker deviation D2 transcribed).

    ``nominal`` / ``sigma`` are the per-dial ±10σ default box; the named-dial
    ``_box`` overrides are the truncation limits (sub-GeV absorbers, energy-banded
    flux ratios [0.3,1.7], sub-GeV/multi-GeV xsec dials, the H5 pinned
    neutron-migration box, the one-sided dir_smear [0,1]). ``free_mask`` (optional
    boolean, one entry per dial) collapses False dials to their nominal so they
    drop out of the fit exactly. Returns ``list(zip(lower, upper))`` — byte for
    byte what ``fit_point`` built inline and what the scan worker transcribed.
    """
    nominal = np.asarray(nominal, float).copy()
    sigma = np.asarray(sigma, float)
    names = list(names)
    lower = nominal - 10 * sigma
    upper = nominal + 10 * sigma
    lower[(nominal > 0) & (lower < 0.01)] = 0.01
    # box bounds for optional dials (truncation limits): sub-GeV absorbers,
    # energy-banded flux ratios ([0.3,1.7]), and sub-GeV xsec dials.
    _box = dict(FLUX_RATIO_BOX)
    _box.update({n: (0.3, 1.7) for n in FLUX_BAND_NAMES})
    _box.update(XSEC_EXTRA_BOX)
    _box.update(MULTIGEV_CCQE_BOX)           # multi-GeV CCQE flavor norms [0,3]
    _box.update(NEUTRON_MIG_BOX_PINNED)      # H5 pinned: x in [0, 1+1/r] (trial unpinned)
    _box[DIR_SMEAR_NAME] = DIR_SMEAR_BOX     # one-sided [0,1] (nominal-0 dial)
    for name, (lo, hi) in _box.items():
        if name in names:
            k = names.index(name)
            lower[k], upper[k] = lo, hi
    if free_mask is not None:
        fixed = ~np.asarray(free_mask, bool)
        lower[fixed] = nominal[fixed]
        upper[fixed] = nominal[fixed]
    return list(zip(lower, upper))


# ---------------- per-point fit (production minimizer protocol) ----------------
def fit_point(eng, phi_dcp_stack, x0=None, n_dcp=None, free_mask=None,
              jac=None, dcp_warmchain=True):
    """dCP-profiled nuisance minimization (production per-point protocol).

    phi_dcp_stack: phi[n_dcp, 2, 3, nE, nZ]. Returns the tuple
    (chi2, best_dcp_index, nuisance, nit, converged).

    jac: True/None (default) -> analytic gradient (era-aware); False ->
    L-BFGS-B finite differences (kept as a cross-check path).

    free_mask: optional boolean (41,) — parameters with False are FIXED
    at nominal via collapsed bounds (ablation-ladder mechanism; penalty
    contribution at nominal is identically zero, so fixed parameters
    drop out of the chi2 exactly).

    dcp_warmchain: if True (default), warm-start each dCP node from the best
    converged solution so far (node 0 from x0). Adjacent dCP nodes share
    nearly the same likelihood surface, so this reaches the same basin in
    ~1 cold + (n-1) warm L-BFGS solves instead of n cold solves — the source
    of the old scans' speed. False recovers the legacy cold-from-x0-per-node
    path EXACTLY (x_seed never leaves x0); kept as the validation baseline.
    """
    nominal = eng.nominal.copy()
    if x0 is None:
        x0 = nominal
    # box bounds for the optional dials (truncation limits) — the single source
    # (Track S·F / F5: dissolves worker deviation D2's transcription).
    bounds = build_nuisance_bounds(nominal, eng.sigma, eng.nuisance_names,
                                   free_mask=free_mask)
    if free_mask is not None:
        fixed = ~np.asarray(free_mask, bool)
        x0 = np.where(fixed, nominal, x0)

    use_jac = True if jac is None else jac
    if eng.solar_mix_f is not None:
        # solar-mix mode: phi_dcp_stack is the PAIR (stack_solmin, stack_solmax)
        stack_a, stack_b = phi_dcp_stack
        n = stack_a.shape[0] if n_dcp is None else n_dcp
    else:
        n = phi_dcp_stack.shape[0] if n_dcp is None else n_dcp
    best = (np.inf, 0, x0, 0, False)
    x_seed = x0                       # node 0 from x0; warm-chained thereafter
    for di in range(n):
        if eng.solar_mix_f is not None:
            phi = (stack_a[di].astype(float), stack_b[di].astype(float))
        else:
            phi = phi_dcp_stack[di].astype(float)
        # tolerance scaling from stat-only chi2 at the current seed
        n_nu, var = eng.expectation(phi, x_seed)
        if eng.likelihood == "poisson":
            chi2_stat = eng.poisson_chi2(eng.obs_f, n_nu[eng.few])
        else:
            chi2_stat, _, _ = eng.bb_chi2(eng.obs_f, n_nu[eng.few],
                                          var[eng.few])
        tol = max(1e-5, np.sqrt(max(min(chi2_stat, 1e7), 0)) * 1e-5)
        if use_jac:
            res = minimize(lambda th: eng.chi2_and_grad(phi, th), x_seed,
                           method="L-BFGS-B", jac=True, bounds=bounds,
                           options={"ftol": tol, "gtol": 1e-5, "maxiter": 200})
        else:
            res = minimize(lambda th: eng.chi2(phi, th), x_seed,
                           method="L-BFGS-B", bounds=bounds,
                           options={"ftol": tol, "gtol": 1e-5, "maxiter": 200})
        if res.fun < best[0]:
            best = (res.fun, di, res.x.copy(), res.nit, res.success)
        if dcp_warmchain:
            x_seed = best[2]          # next node warm-starts from the best basin
    return best


# --------------------------------------------------------------------------- #
#  TensorStore — oscillation-tensor grid lookup / caching (Track S, Phase E6)
# --------------------------------------------------------------------------- #
# The φ[n_dcp, 2, 3, nE, nZ] oscillation tensors are built per (Δm², s²θ₂₃) grid
# node (BuildOscTensors) and stored one npz per node. TensorStore reads the grid
# axes and serves the tensor for a physics point — exact grid-node lookup by
# default (a 1-slot cache so a per-cell restart-polish loop reads the npz once),
# cubic interpolation opt-in. This is the φ lookup/caching that used to live in
# ``adapter.py`` (BinnedEngineAdapter.load / _node_index / phi / n_dcp +
# osc_averaging validation), moved here verbatim (ZERO behaviour change) so the
# adapter can be deleted; the PyNuFit modular methods now hold their own staged
# (phi, theta) state and drive the engine kernels directly.
class TensorStore:
    """(Δm², s²θ₂₃) -> φ tensor, from a directory of per-node osc_tensor npz files.

    ``interp='nodes'`` (default): exact grid-node lookup (rtol 1e-9), 1-slot
    cache. ``interp='cubic'``: PhiInterpolator over the node grid.
    """

    def __init__(self, tensors_dir, interp="nodes", osc_averaging="off"):
        from ..inference.interp_engine import PhiInterpolator, detect_grid
        self.tensors_dir = tensors_dir
        self.interp = interp
        self.osc_averaging = osc_averaging
        self.DM, self.S23 = detect_grid(tensors_dir)
        self._interp = None
        self._phi_cache = None       # 1-slot ((i, j) -> phi) node cache
        if interp == "cubic":
            self._interp = PhiInterpolator(tensors_dir, cache_raw=False)
        elif interp != "nodes":
            raise ValueError(
                f"unknown interp {interp!r} (expected 'nodes' or 'cubic')")
        for w in self._validate_osc_averaging():
            print(f"WARNING [binned osc_averaging]: {w}")

    def _validate_osc_averaging(self):
        """Warn-only cross-check of the ``osc_averaging`` declaration against
        tensor-set metadata. Current builds embed no averaging key, so this only
        records provenance; once a build embeds one (e.g. an 'osc_averaging'
        array in the npz), a mismatch warns. Peeks at a single tensor."""
        warns = []
        decl = str(self.osc_averaging).strip().lower()
        files = sorted(glob.glob(os.path.join(self.tensors_dir,
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
        loop (repeated fits at the same node) reads the npz once, not once per
        call. The cached array is returned as-is; callers copy per dCP slice
        (``phi[di].astype(float)``) and never mutate it."""
        if self.interp == "cubic":
            return self._interp(dm231, s23)
        i = self._node_index(self.DM, dm231, "Dm231")
        j = self._node_index(self.S23, s23, "Sin2Theta23")
        if self._phi_cache is not None and self._phi_cache[0] == (i, j):
            return self._phi_cache[1]
        p = os.path.join(self.tensors_dir, f"osc_tensor_{i:03d}_{j:03d}.npz")
        arr = np.load(p)["phi"]
        self._phi_cache = ((i, j), arr)
        return arr

    @property
    def n_dcp(self):
        """Number of dCP nodes in the tensor grid (for the worker loop)."""
        return int(self.phi(self.DM[0], self.S23[0]).shape[0])


# --------------------------------------------------------------------------- #
#  BinnedBinding — engine + tensor store + config holder (Track S, Phase E6)
# --------------------------------------------------------------------------- #
# Replaces ``adapter.py``'s construction/holding role WITHOUT the staged-state
# and φ-lookup logic (staged (phi, theta) now lives on PyNuFit's modular methods;
# φ lookup is TensorStore above). A binding is the loaded (engine, store, config)
# triple for ONE binned experiment; PyNuFit keeps ``{exp_name: BinnedBinding}``
# in ``self.BinnedEngines`` and drives ``binding.engine`` / ``binding.store``
# directly. The read-only accessors (nominal/sigma/nuisance_names/n_dcp/DM/S23/
# observed/phi/chi2) preserve the exact surface the packaged fit path + the
# scan worker used on the former adapter (ZERO numerical change).
class BinnedBinding:
    def __init__(self, engine, store, config):
        self.engine = engine
        self.store = store
        self.config = config

    @classmethod
    def load(cls, config, analysis_xml=None):
        """Build the engine (validates the nuisance spec against the response
        build) + the TensorStore for a BinnedConfig. ``nuisance_spec='self'``
        resolves to the analysis XML path."""
        from ...binned.sk_binned_engine import SKBinnedEngine
        spec = cls._resolve_spec(config, analysis_xml)
        engine = SKBinnedEngine(
            config.response,
            migration_mode=config.migration,
            likelihood=config.likelihood,
            nuisance_spec=spec,
        )
        store = TensorStore(config.tensors, interp=config.interp,
                            osc_averaging=config.osc_averaging)
        return cls(engine, store, config)

    @staticmethod
    def _resolve_spec(config, analysis_xml):
        """Map the config selector to what ``resolve_nuisance_spec`` accepts.

        'self' -> the analysis XML path; any other string ('barr'/'R2'/... or an
        explicit .xml path) or a list passes through; None/''/'barr' -> engine
        default 41-vector (None)."""
        spec = config.nuisance_spec
        if spec == "self":
            if not analysis_xml:
                raise ValueError(
                    "binned engine nuisance_spec='self' requires the analysis "
                    "XML path (pass analysis_xml=...)")
            return analysis_xml
        if spec in ("", "barr", None):
            return None
        return spec

    # ---- read-only passthroughs (former adapter surface) ----
    @property
    def DM(self):
        return self.store.DM

    @property
    def S23(self):
        return self.store.S23

    @property
    def n_dcp(self):
        return self.store.n_dcp

    @property
    def nuisance_names(self):
        return self.engine.nuisance_names

    @property
    def nominal(self):
        return self.engine.nominal

    @property
    def sigma(self):
        return self.engine.sigma

    def phi(self, dm231, s23):
        return self.store.phi(dm231, s23)

    def nuisance_bounds(self, free_mask=None):
        """L-BFGS-B bounds for this binding's (post-override) nominal/sigma —
        the production box, from the single source ``build_nuisance_bounds``.
        Track S·F / F5: replaces the scan worker's transcription (deviation D2)."""
        return build_nuisance_bounds(self.nominal, self.sigma,
                                     self.nuisance_names, free_mask=free_mask)

    def observed_binned(self):
        """The engine's FewEntries-filtered observation vector ``obs_f``."""
        return self.engine.obs_f

    def chi2(self, dm231, s23, theta, dcp_index=None):
        """Binned chi2 at fixed nuisance vector ``theta``. ``dcp_index`` selects
        a single dCP slice; None profiles (min) over all dCP nodes."""
        phi = self.store.phi(dm231, s23)
        if dcp_index is not None:
            return float(self.engine.chi2(phi[dcp_index].astype(float), theta))
        return float(min(self.engine.chi2(phi[d].astype(float), theta)
                         for d in range(phi.shape[0])))

    def fit_point(self, dm231, s23, x0=None, free_mask=None):
        """dCP-profiled nuisance fit. Returns the engine tuple
        ``(chi2, best_dcp_index, theta, nit, converged)``."""
        phi = self.store.phi(dm231, s23)
        return self.engine.fit_point(phi, x0=x0, free_mask=free_mask)
