#!/usr/bin/env python3
"""SK binned forward-model engine — the consumer of sk_response.npz + osc tensors.

Replicates the event-by-event PyNuFit SK2023 expectation exactly, modulo the one
controlled approximation: per-event (ETrue, CosZTrue)-dependent weights (osc Phi,
flux tunes, AxialMass) are evaluated at true-cell centers instead of per event.
Everything else — xsec masks, detector per-sample factors with rate-conserving
migration ratios, BB-lite likelihood, FewEntries filter, penalty terms, the
L-BFGS-B protocol — is transcribed 1:1 from the production code paths:

  pynu/PhysicsTunes/Flux/AtmoFlux.py          (flux tunes)
  pynu/PhysicsTunes/CrossSection/WaterXSection.py  (xsec tunes, AxialMass)
  pynu/PhysicsTunes/Detector/SKCombinedDetector.py (detector tunes, migration r)
  pynu/Experiments/SuperK_Atm_Pheno.py:SuperK_2023 (BaseWeight, GetMCVariance, NC fix)
  pynu/Experiments/Experiment.py              (FewEntries: observed > 5)
  pynu/fitter/BarlowBeestonLikelihood.py      (BB-lite beta, penalties, gradient)
  analysis/SuperK-datafit/run_sk_datafit_row_worker.py (minimizer protocol, dCP profiling)

Engine model per (physics point, dCP slice):
  W[k,cE,cZ]  = Phi_or_1 * F_flux(cell, pdg_k) * A_axial(cell, CC_k) * X_k(mask bits)
  n_pre[b]    = sum_nz R_v * W[R_k,R_e,R_z] scattered to R_b
  n_nu[b]     = n_pre[b] * D_b(theta_det; r from physics-only rates)
  var[b]      = D_b^2 * sum_nz S2_v * W^2[...] scattered to S2_b
  chi2        = BB(obs[m], n_nu[m], var[m]) + sum_j (x_j - mu_j)^2 / sigma_j^2
with m = FewEntries mask (observed > 5, strict), NC classes get Phi = 1
(SuperK_2023.UpdatePhysicsWeights override).

No muon background (SuperK_2023 has no GetMuonBackground).
energy_scale is not in the 41-param list, so R_plus/R_minus are unused here.
"""
import json

import numpy as np
from scipy.optimize import minimize

# ---- dial vocabulary + XML value authority (Track T / T1) --------------------
# The entire dial vocabulary (name lists, sigma/box tables, spec registry) and
# the XML dial-value authority (XML_DIAL_VALUES, resolve_nuisance_spec, the
# θ-order assert) live in ``pynu.analysis_reader.binned_dials`` — a leaf module
# (stdlib+numpy only) — so the fitter and PhysicsTunes import the vocabulary
# WITHOUT importing this engine module. Every name is re-imported here so this
# module's namespace (the surface engine_core/masks/grid_experiment, the gates,
# and downstream scripts reference) is unchanged. ZERO value/semantics change.
from ..analysis_reader.binned_dials import (  # noqa: F401
    FLUX_NAMES, XSEC_VECTOR_NAMES, MASK_TUNES, DET_NAMES, NUISANCE_NAMES,
    ERA_TAGS, DET_ERA_STEMS, SIGMA, NOMINAL, MIN_ENTRIES,
    CORE_FLUX_NAMES, ZENITH_DIALS, SOLAR_AMP, SOLAR_SCALE, KPI_E0,
    OPTIONAL_FLUX_NAMES, ALL_FLUX_NAMES, ENERGY_SCALE_NAMES,
    FLUX_RATIO_NAMES, FLUX_RATIO_BOX, NTAG_SPLIT_NAMES, NTAG_PSPLIT,
    FLUX_BAND_NAMES, FLUX_RATIO_SPEC, ALL_FLUX_RATIO_NAMES,
    CCQE_SHAPE_SUBGEV_E, XSEC_EXTRA_NAMES, XSEC_EXTRA_BOX, SUBGEV_NUE_NORM,
    MULTIGEV_CCQE_NAMES, MULTIGEV_CCQE_NORM, MULTIGEV_CCQE_BOX,
    REL_NORM_FCMG_SAMPLES, REL_NORM_NAMES,
    NEUTRON_MIG_PAIRS, NEUTRON_MIG_NAMES, NEUTRON_MIG_TRIAL_NAMES,
    NEUTRON_MIG_PINNED_NAMES, NEUTRON_MIG_RAWCOUNTS, NEUTRON_MIG_BOX_PINNED,
    DECAY_E_SAMPLES, DECAY_E_NAME,
    UPMU_BKG_SHAPE_NAMES, UPMU_BKG_SHAPE_SPEC, UPMU_BKG_SHAPE_SIGMA,
    UPDOWN_ESCALE_NAMES, UPDOWN_ESCALE_SIGMA, UPDOWN_ESCALE_EXCLUDE,
    DIR_SMEAR_NAME, DIR_SMEAR_SIGMA, DIR_SMEAR_BOX,
    DIAL_VALUE_XML, DIAL_VALUE_XML_EXTRA, XML_DIAL_VALUES,
    SPEC_MANIFESTS, SPEC_REQUIRES_WEIGHTED, resolve_nuisance_spec,
    _MANIFEST_DIR, _dial_value, _parse_xml_active, _xml_document_order,
    _assert_production_theta_order, _assert_mirror_agrees,
    _load_xml_dial_values, _unphys,
)


class SKBinnedEngine:
    """migration_mode selects the migration-ratio convention for the
    rate-conserving detector tunes:

      'weighted' (default, current production): r = physics-weighted rates
          (BaseWeight*PhysicsWeight sums per sample), recomputed at every
          oscillation point — the post-bugfix SKCombinedDetector behavior.
      'rawcount' (legacy): r = raw MC event counts per sample, frozen across
          the oscillation grid — the pre-bugfix behavior. Rationale:
          experiments generate migration templates
          once, not per oscillation point, so the physics-independent ratio
          may model the actual systematic more faithfully; expected small.

    fcpc_separation uses raw counts in BOTH modes (production always did).

    likelihood selects the statistical treatment:
      'bb' (default, production): Barlow-Beeston-lite — per-bin beta profiled
          against the S2-based MC variance.
      'poisson': plain Poisson chi2 = 2*sum(E - O + O ln(O/E)) over the
          FewEntries bins, no MC-variance term — the event engine's
          stats_only fallback form (any E<=0 returns 9e9).
    """

    def __init__(self, response_path, migration_mode="weighted", solar_mix_f=None,
                 likelihood="bb", nuisance_spec=None, dirsmear_matrix=None):
        if migration_mode not in ("weighted", "rawcount"):
            raise ValueError(f"unknown migration_mode {migration_mode!r}")
        if likelihood not in ("bb", "poisson"):
            raise ValueError(f"unknown likelihood {likelihood!r}")
        self.migration_mode = migration_mode
        self.likelihood = likelihood

        # active nuisance set (default None -> production 41-vector, unchanged)
        self.nuisance_names, self.nominal, self.sigma = \
            resolve_nuisance_spec(nuisance_spec)
        self.flux_names = [n for n in self.nuisance_names
                           if n in ALL_FLUX_NAMES]
        # OPTIONAL octant absorbers (OFF unless the spec lists them explicitly).
        # active_flux_ratios now spans BOTH the 2 sub-GeV absorbers and the 7
        # energy-banded extensions (all share FLUX_RATIO_SPEC); active_xsec_extra
        # are the optional sub-GeV xsec dials (CCQE shape / sub-GeV CCQE nue).
        self.active_flux_ratios = [n for n in self.nuisance_names
                                   if n in ALL_FLUX_RATIO_NAMES]
        self.active_xsec_extra = [n for n in self.nuisance_names
                                  if n in XSEC_EXTRA_NAMES]
        self.active_multigev_ccqe = [n for n in self.nuisance_names
                                     if n in MULTIGEV_CCQE_NAMES]
        self.active_rel_norm = [n for n in self.nuisance_names
                                if n in REL_NORM_NAMES]
        self.active_upmu_bkg = [n for n in self.nuisance_names
                                if n in UPMU_BKG_SHAPE_NAMES]
        self.active_ude = [n for n in self.nuisance_names
                           if n in UPDOWN_ESCALE_NAMES]
        # neutron-production 0n/1n migration dials + decay-e norm:
        # OFF unless the spec lists them explicitly. Both ride detector_factors/dlnD.
        self.active_neutron_mig = [n for n in self.nuisance_names
                                   if n in NEUTRON_MIG_NAMES]
        self.active_decay_e = DECAY_E_NAME in self.nuisance_names
        # direction-smearing dial (OFF unless the spec lists it AND a matrix is given)
        self.active_dir_smear = DIR_SMEAR_NAME in self.nuisance_names
        self._dirsmear_matrix_path = dirsmear_matrix
        self.ntag_split = any(n in NTAG_SPLIT_NAMES
                              for n in self.nuisance_names)
        if self.ntag_split and migration_mode != "weighted":
            raise ValueError("ntag momentum-split requires migration_mode="
                             "'weighted' (needs per-bin physics rates)")
        self.active_energy_scale = [n for n in self.nuisance_names
                                    if n in ENERGY_SCALE_NAMES]
        active_xsec = [n for n in self.nuisance_names if n in XSEC_VECTOR_NAMES]
        # the response / osc-tensor build bakes in exactly these xsec classes
        # and detector samples; only the flux-zenith block is switchable here.
        if set(active_xsec) != set(XSEC_VECTOR_NAMES):
            raise ValueError("xsec dial set must match the response build "
                             f"(need {XSEC_VECTOR_NAMES}); got {active_xsec}")
        # detector dials: each base stem must be covered either era-INDEPENDENT
        # (its base name in the spec) or era-SPLIT (all 4 <stem>_<era> dials).
        # det_split_stems drives _era_theta; det_names are the base stems that
        # detector_factors / the gradient iterate (era-split or indep alike).
        self.det_split_stems = []
        for stem in DET_ERA_STEMS:
            era_dials = [f"{stem}_{tag}" for tag in ERA_TAGS]
            present = [d in self.nuisance_names for d in era_dials]
            if any(present):
                if not all(present):
                    raise ValueError(f"era-split stem {stem!r} needs all 4 era "
                                     f"dials {era_dials}; got {present}")
                self.det_split_stems.append(stem)
        indep_det = [n for n in DET_NAMES if n in self.nuisance_names]
        covered = set(self.det_split_stems) | set(indep_det)
        # when the ntag split is active, neutron_tagging_subgev is replaced by
        # the two per-band dials (handled per-bin), so it drops from the required.
        required_det = set(DET_NAMES)
        if self.ntag_split:
            required_det = required_det - {"neutron_tagging_subgev"}
        if covered != required_det:
            raise ValueError("detector dial set must cover the response build's "
                             f"stems {sorted(required_det)} (era-split or indep); "
                             f"got split={sorted(self.det_split_stems)} "
                             f"indep={sorted(indep_det)}")
        # base detector stems for detector_factors / dlnD (DET_NAMES order)
        self.det_names = [n for n in DET_NAMES if n in covered]
        if set(CORE_FLUX_NAMES) - set(self.flux_names):
            raise ValueError("all core flux dials are required")
        if not any(n in ZENITH_DIALS for n in self.flux_names):
            raise ValueError("at least one zenith dial must be active")
        if len(set(self.nuisance_names)) != len(self.nuisance_names):
            raise ValueError("duplicate nuisance names in spec")
        z = np.load(response_path, allow_pickle=False)
        self.Rk, self.Re, self.Rz = z["R_k"], z["R_e"], z["R_z"]
        self.Rb, self.Rv = z["R_b"], z["R_v"]
        self.S2k, self.S2e, self.S2z = z["S2_k"], z["S2_e"], z["S2_z"]
        self.S2b, self.S2v = z["S2_b"], z["S2_v"]
        self.classes = z["classes"]                       # (n_cls, 2+15)
        self.xsec_tune_names = [str(s) for s in z["xsec_tune_names"]]
        assert self.xsec_tune_names == MASK_TUNES
        self.e_edges, self.z_edges = z["e_edges"], z["z_edges"]
        self.n_bins = int(z["n_bins"])
        self.observed = z["observed"]
        self.sample_table = json.loads(str(z["sample_table"]))   # s -> (off, ne, nz)
        self.sample_counts = json.loads(str(z["sample_event_counts"]))
        self.meta = json.loads(str(z["meta"]))

        self.n_cls = self.classes.shape[0]
        self.nE = self.e_edges.size - 1
        self.nZ = self.z_edges.size - 1
        # cell centers (geometric in E, arithmetic in cz) — build_osc_tensors.grid_centers
        self.e_c = np.sqrt(self.e_edges[:-1] * self.e_edges[1:])
        self.z_c = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])

        # class decomposition
        self.cls_pdg = self.classes[:, 0].astype(int)
        self.cls_cc = self.classes[:, 1].astype(int)
        self.cls_bits = self.classes[:, 2:].astype(bool)         # (n_cls, 15)
        # flavor / type index into phi[type, flavor, E, Z]
        self.cls_flavor = (np.abs(self.cls_pdg) // 2 - 6)        # 12->0,14->1,16->2
        self.cls_type = (self.cls_pdg < 0).astype(int)           # nu=0, nubar=1

        # flatten (k, e, z) -> single gather index for fast contraction
        self.R_widx = (self.Rk.astype(np.int64) * self.nE + self.Re) * self.nZ + self.Rz
        self.S2_widx = (self.S2k.astype(np.int64) * self.nE + self.S2e) * self.nZ + self.S2z

        # per-bin -> sample id map
        self.bin_sample = np.empty(self.n_bins, dtype=int)
        for s, (off, ne, nz) in self.sample_table.items():
            self.bin_sample[off:off + ne * nz] = int(s)
        self.samples = np.unique(self.bin_sample)

        # ---- SK era partition (phased response). Absent -> single era 0, which
        #      collapses every per-era path to the legacy single-era behaviour.
        self.n_era = int(z["n_era"]) if "n_era" in z.files else 1
        self.R_era = (z["R_era"].astype(np.int64) if "R_era" in z.files
                      else np.zeros(self.Rb.shape, dtype=np.int64))
        self.S2_era = (z["S2_era"].astype(np.int64) if "S2_era" in z.files
                       else np.zeros(self.S2b.shape, dtype=np.int64))
        # flattened (era, bin) gather index for one-pass per-era contraction
        self.R_eb = self.R_era * self.n_bins + self.Rb
        self.S2_eb = self.S2_era * self.n_bins + self.S2b
        # consistency: era-split detector dials (det_split_stems, resolved during
        # validation above) require a 4-era (phased) response.
        if self.det_split_stems and self.n_era != len(ERA_TAGS):
            raise ValueError(f"spec era-splits {self.det_split_stems} but response "
                             f"has n_era={self.n_era} (need {len(ERA_TAGS)})")

        # ---- OPTIONAL per-era solar-flux mixture (None -> single-phi path,
        #      byte-identical prior behaviour). solar_mix_f[e] = solmax fraction
        #      for era e (SK per-phase neutron-monitor livetime weights, Wester
        #      Table 4.1: sk1 0.30, sk2 0.70, sk3 0.00, sk45 0.498 livetime-
        #      weighted IV/V). In this mode every phi argument is the PAIR
        #      (phi_solmin, phi_solmax); nuSQuIDS evolution and the whole
        #      pre-detector chain are linear in phi, so
        #        n_pre_era = n_pre_era(phi_a) + f_era * [n_pre_era(phi_b) -
        #                    n_pre_era(phi_a)]
        #      is the EXACT per-era mixed expectation.
        self.solar_mix_f = None
        if solar_mix_f is not None:
            self.solar_mix_f = np.asarray(solar_mix_f, dtype=float)
            if self.solar_mix_f.shape != (self.n_era,):
                raise ValueError(f"solar_mix_f needs {self.n_era} per-era values;"
                                 f" got shape {self.solar_mix_f.shape}")

        # ---- Track S / Phase E4: per-dial mask & selector assembly lives in the
        # native sk_binned_masks module (descriptor-driven). assemble_masks sets
        # every mask/selector attribute this __init__ used to build inline
        # (energy-scale reco-E adjacency, flux-ratio legs, ntag bands, up-mu masks,
        # the up/down signed mask, dirsmear blocks, static flux fields, xsec class
        # masks, FewEntries) — ZERO numerical change. When the response npz carries
        # baked geometry selectors (`sel_*` keys) they are used AND asserted
        # byte-equal to the descriptor assembly; today's responses have none, so
        # the descriptor path is authoritative (exactly as before). `z` is passed
        # for that baked-selector check.
        _masks.assemble_masks(self, z)

    # ---------------- weight fields ----------------
    def _tune_objects(self):
        """Lazily-constructed real PhysicsTunes instances (AtmoFlux + WaterXSection)
        that source the flux/xsec per-dial factors for cell_weights (Track S,
        Phase E5a). Cached on first use; import is local so the module has no
        hard dependency on the tune classes at import time (keeps the frozen /
        stripped-checkout import path clean)."""
        tp = getattr(self, "_tune_pair", None)
        if tp is None:
            from ..PhysicsTunes.Flux.AtmoFlux import AtmosphericFlux
            from ..PhysicsTunes.CrossSection.WaterXSection import WaterXSection
            tp = (AtmosphericFlux(), WaterXSection())
            self._tune_pair = tp
        return tp

    def cell_weights(self, phi, theta):
        """W[k, cE, cZ] for nuisance vector theta and physics tensor phi[2,3,nE,nZ].

        NC classes get phi = 1 (SuperK_2023.UpdatePhysicsWeights NC override).

        Track S / Phase E5a+E6: the flux/xsec per-dial factors are sourced from
        the REAL AtmoFlux/WaterXSection methods via the GridExperiment shim
        (grid_experiment.cell_weights_via_tunes), reassembled with THIS method's
        exact axis-factored association. The GridExperiment delegate is the SOLE
        path — the former hand-inlined shadow was deleted at E6.
        """
        flux, xsec = self._tune_objects()
        return _grid.cell_weights_via_tunes(self, phi, theta, flux, xsec)

    # ---------------- contractions ----------------
    # Structural kernels below delegate to sk_binned_engine_core (Track S,
    # Phase E1). Each engine_core function takes the engine instance and is a
    # verbatim move of the former in-class body; ZERO numerical change.
    def contract(self, W):
        """n_pre[b] = R contracted with cell weights W."""
        return _core.contract(self, W)

    def contract_var(self, Wsq):
        """sum of BaseWeight^2 * W^2 per bin (pre-detector)."""
        return _core.contract_var(self, Wsq)

    def contract_era(self, W):
        """n_pre[era, b] = R contracted with cell weights W, split by SK era.
        Sums over era to contract(W) exactly (era is a disjoint partition)."""
        return _core.contract_era(self, W)

    def contract_var_era(self, Wsq):
        """Per-era pre-detector BaseWeight^2 * W^2 sum (era, b)."""
        return _core.contract_var_era(self, Wsq)

    def _escale_migrate(self, arr_e, deltas, var=False):
        """Per-era energy-scale reco-E migration of a (n_era, n_bins) array.
        Linear, rate-conserving within each (sample, reco-cz) column:
          N'(ie) = N(ie) + d*( N(ie-1)*[ie>0] - N(ie)*[ie<ne-1] ).
        var=True propagates BB variances (independent-bin squared coefficients)."""
        return _core.escale_migrate(self, arr_e, deltas, var=var)

    def _dir_smear_apply(self, vec, s):
        """Apply the reco-cz migration operator A = I + s*(M - I) to a (n_bins,)
        vector, block-diagonal per sample x reco-momentum row. E'_i = sum_j A[i,j] E_j;
        at s=1, E' = M @ E per zenith row. Identity (z1bins) samples are untouched."""
        return _core.dir_smear_apply(self, vec, s)

    def _dir_smear_apply_T(self, vec, s):
        """Apply A^T = I + s*(M^T - I) block-diagonally (used to pull the likelihood
        residual back through the smearing for the OTHER dials' gradient)."""
        return _core.dir_smear_apply_T(self, vec, s)

    def _era_theta(self, t, e):
        """Per-era view of the nuisance dict: era-split detector stems take their
        era-e dial value; everything else is shared. No-op for legacy specs.

        The up/down energy-scale set is era-split too, but its dials are not in
        DET_ERA_STEMS, so it is routed here into the base key 'updown_escale' that
        detector_factors reads (mirrors the DET_ERA_STEMS remap)."""
        return _core.era_theta(self, t, e)

    # ---------------- detector factors ----------------
    def sample_rates(self, n_phys):
        """Weighted physics rate per sample (BaseWeight*PhysicsWeight sums).
        Track S / Phase E5b: delegates to the native PhysicsTunes detector kernel."""
        return _det.sample_rates(self, n_phys)

    def detector_factors(self, t, rates, n_phys=None):
        """Per-sample detector factor D_s and per-tune d ln D. Track S / Phase E5b:
        delegates to PhysicsTunes.Detector.detector.detector_factors (descriptor-driven
        generic kernels, guards preserved verbatim; ZERO numerical change)."""
        return _det.detector_factors(self, t, rates, n_phys=n_phys)

    # ---------------- expectation + chi2 ----------------
    def expectation(self, phi, theta, return_parts=False):
        """Full binned expectation (930) + variance, replicating the event chain.

        Era-aware: E_b = sum_era D_era[b] * n_pre_era[b]. Detector factors are
        evaluated per era (era-split dials take their era value; migration ratios
        use per-era rates). For a single-era response (n_era=1) this reduces
        exactly to the legacy n_pre * D path.
        """
        return _core.expectation(self, phi, theta, return_parts=return_parts)

    def cell_weights_physics_only(self, phi):
        return _core.cell_weights_physics_only(self, phi)

    @staticmethod
    def bb_chi2(obs, n_mod, var):
        """BarlowBeestonLikelihood.stats_only (BB-lite, no muons)."""
        return _kernels_bb_chi2(obs, n_mod, var)

    @staticmethod
    def poisson_chi2(obs, n_mod):
        """Plain Poisson chi2 (event engine's no-MC-variance fallback form)."""
        return _kernels_poisson_chi2(obs, n_mod)

    def chi2(self, phi, theta):
        return _core.chi2(self, phi, theta)

    # ---------------- analytic gradient ----------------
    def chi2_and_grad(self, phi, theta):
        """f, g with the event engine's first-order convention (beta fixed,
        migration r fixed, Jacobian at the current point).

        Era-aware: physics (flux/xsec) params enter the era-independent cell
        weights W, so dE_b = sum_era D_era[b] * contract_era(W * dlnW/dp)[era][b];
        era-split detector dials route to their per-era dial and era-independent
        detector dials accumulate over eras. Reduces to the single-era gradient
        exactly when n_era==1.
        """
        return _core.chi2_and_grad(self, phi, theta)

    def _flux_dlnw(self, name, t):
        return _core.flux_dlnw(self, name, t)

    # ---------------- per-point fit (production minimizer protocol) ----------------
    def fit_point(self, phi_dcp_stack, x0=None, n_dcp=None, free_mask=None,
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
        return _fit_point(self, phi_dcp_stack, x0=x0, n_dcp=n_dcp,
                               free_mask=free_mask, jac=jac,
                               dcp_warmchain=dcp_warmchain)

    # ---------------- per-bin diagnostics (pull extraction) ----------------
    def per_bin_report(self, phi, theta):
        """Per-bin decomposition of the stat chi2 at (phi, theta), over the
        FewEntries bins (self.few, obs>5). Returns a dict of arrays aligned to
        the masked bins, plus their GLOBAL bin index (0..n_bins-1) so callers
        can join with the SK release BinInfo (sample / momentum / zenith).

        The per-bin BB-lite chi2 is the exact addend-by-addend split of
        bb_chi2's two sums (Poisson term + Barlow-Beeston penalty), so
        sum(chi2_bin) == bb_chi2(...)[0] to floating precision. The primary
        'pull' is the signed sqrt of that per-bin chi2 (its square sums to the
        stat chi2); 'resid_std' is the conventional (obs - model)/sqrt(model +
        var) standardized residual for cross-reference.

        Only valid for likelihood='bb' (the production form used by this
        diagnostic); raises otherwise.
        """
        return _core.per_bin_report(self, phi, theta)


# --- Track S / Phase E2 θ-order assert: now fires at binned_dials import
# (Track T / T1) — every vocabulary consumer hits it, engine included.

# --- Track S / Phase E1 kernels (Track T / T3: co-moved to Experiments/ as
# sk_binned_engine_core). Imported at module bottom so its
# `from .sk_binned_engine import ...` (engine attributes defined above)
# resolves without a circular import; the SKBinnedEngine methods above
# delegate to `_core` at call time only.
from . import sk_binned_engine_core as _core  # noqa: E402
# --- Track S / Phase E4 mask/selector assembly (same bottom-import rationale;
# T3: co-moved as sk_binned_masks).
from . import sk_binned_masks as _masks  # noqa: E402
# --- Track S / Phase E5a cell-weight factor sourcing from the real
# PhysicsTunes methods (T3 / O-1 ruling: re-homed to
# pynu/PhysicsTunes/TuneFactorSource.py, top-level sibling of PhysicsTunes.py
# — it bridges Flux/AtmoFlux AND CrossSection/WaterXSection).
from ..PhysicsTunes import TuneFactorSource as _grid  # noqa: E402
# --- Track S / Phase E5b: descriptor detector-factor kernels.
# (S.F4) re-homed to pynu.PhysicsTunes.Detector.detector (beside SKCombinedDetector).
from pynu.PhysicsTunes.Detector import detector as _det  # noqa: E402
# --- Track T / T6: the chi2 kernels + per-point fit protocol are imported from
# their functional homes directly (engine_core's back-compat re-import block is
# deleted). binned_fit imports only the vocabulary leaf at module top, so this
# bottom import is cycle-free.
from ..fitter.binned_kernels import (  # noqa: E402
    bb_chi2 as _kernels_bb_chi2,
    poisson_chi2 as _kernels_poisson_chi2,
)
from ..fitter.minimizer.binned_fit import fit_point as _fit_point  # noqa: E402

