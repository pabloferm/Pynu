"""Native builder for the SK binned forward-model artifacts (response + osc tensors).

This module relocates the logic of the two standalone build scripts —
``build_sk_response.py`` (response COO bake) and ``build_osc_tensors.py``
(per-node oscillated-flux tensors) — into pynu itself, so the same
``PyNuFit`` object that runs an event fit can also *build* the binned engine's
inputs from its own live ``Experiments`` / ``physics_tunes`` state.

The standalone scripts stay byte-untouched (cluster SLURM submissions call them
unchanged); this is a purely additive second entry point. The two public
functions take a constructed ``PyNuFit`` and an experiment name and return the
artifacts in memory (optionally writing an npz byte-compatible with the schema
the existing ``SKBinnedEngine`` / TensorStore already load):

  * ``build_response(pynufit, exp_name, ...)`` — one MC pass through the live
    ``SuperK_2023`` experiment class, so every convention (NC w_no fix, NORM,
    WMC, CC-mask encoding, the DIS |Mode|>25*CC quirk) is inherited, never
    re-implemented. Class signatures come from EVALUATING the actual
    ``WaterXSection`` tunes at x=2, so the masks can never drift from production
    code.
  * ``build_tensors(pynufit, exp_name, dm231, s23, ...)`` — Phi[type, flavor,
    E_cell, cz_cell] on the true grid via the production
    ``AtmosphericOscillations`` object, evaluated at cell centres by temporarily
    overriding its per-event coordinate arrays (the ``set_grid_coords`` trick).

Three upgrades over the standalone scripts:
  1. ``avg_scale`` — ``build_tensors`` accepts an ``avg_scale`` selector (from
     the XML ``<osc_averaging>`` field via ``PyNuFit.BuildOscTensors``) and
     applies it to the live osc object for the build, restoring the prior value
     afterwards. Back-compat: ``PYNU_OSC_AVG_SCALE`` in the environment still
     overrides, exactly as the ctor does (``AtmOsc._resolve_avg_scale``).
  2. ADDITIVE response keys — the detector sample-ID selectors + bin-mask
     geometry (``sample_table``, ``sample_event_counts``) and, new here, an
     explicit per-sample event-index map are baked into the npz as keys the
     current engine ignores (schema stays loadable).
  3. ``schema_version`` + dial-manifest hash — additive npz keys, checked at
     engine load ONLY when present (TensorStore already peeks for an
     ``osc_averaging`` key the same way).

★ STATE-RESTORE (the one architectural trap): ``build_tensors`` mutates the
shared osc object's per-event arrays (``ETrue``, ``CosZTrue``, ``NSQneutype``,
``NSQneuflavor``) and its ``Parameters`` / cache. It snapshots and restores all
of them in a ``finally`` so a subsequent event-engine call on the same pynufit
object is byte-unaffected, whether the build succeeds or raises mid-way.
"""
import hashlib
import json

import numpy as np

# ---------------------------------------------------------------------------
# Response builder (transcribed from build_sk_response.py)
# ---------------------------------------------------------------------------

# order must match sk_binned_engine.MASK_TUNES and the standalone script's
# XSEC_TUNES (AxialMass handled separately: continuous in log10 ETrue, CC only).
XSEC_TUNES = [
    "XSecNuTau", "NCoverCC", "NCHad", "DIS",
    "CC_2p2h", "CC_2p2hNuBarNu", "CC_2p2hMuE",
    "CCQE", "CCQENuBarNu",
    "CCQEMuE", "CC1Pi_Pi0Pi", "CC1Pi_NuBarNuE", "CC1Pi_NuBarNuMu",
    "CC1PiProduction", "CohPiProduction",
]

# npz schema tag: bumped only when a key's meaning changes; readers check for
# equality ONLY when the key is present (older responses stay loadable).
SCHEMA_VERSION = "sk_binned_response_v1"


def make_true_grid(emin, emax, n_e, n_z):
    """log-E x cosZ true grid with one edge snapped to exactly 1.0 GeV so the
    normalization_below/above1GeV step tunes are exact on the grid.

    Byte-identical to ``build_sk_response.make_true_grid`` (the production build
    machinery); ``build_osc_tensors`` imports it from that module today, so both
    the standalone and native paths share one snapping convention.
    """
    e_edges = np.geomspace(emin, emax, n_e + 1)
    i = np.argmin(np.abs(np.log(e_edges) - np.log(1.0)))
    e_edges[i] = 1.0
    z_edges = np.linspace(-1.0, 1.0, n_z + 1)
    return e_edges, z_edges


def grid_centers(e_edges, z_edges):
    """Geometric E cell centres, arithmetic cz cell centres (build_osc_tensors)."""
    e_c = np.sqrt(e_edges[:-1] * e_edges[1:])
    z_c = 0.5 * (z_edges[:-1] + z_edges[1:])
    return e_c, z_c


def _reco_bin_index(exp, scale=1.0):
    """Per-event flat reco-bin index replicating BinIt_MC_2D ordering (per sample
    in exp.Samples order: C-order (E, cz) flatten). -1 = out of range.

    Verbatim ``build_sk_response.reco_bin_index``."""
    E = exp.EReco * scale
    cz = exp.CosZReco
    idx = np.full(exp.NumberOfEvents, -1, dtype=np.int64)
    offsets = {}
    off = 0
    for s in exp.Samples:
        eb = np.asarray(exp.EnergyBins[s], float)
        zb = np.asarray(exp.CTBins[s], float)
        ne, nz = eb.size - 1, zb.size - 1
        offsets[int(s)] = (off, ne, nz)
        m = exp.Sample == s
        ie = np.digitize(E[m], eb) - 1
        iz = np.digitize(cz[m], zb) - 1
        # histogram fill includes the upper edge in the last bin; digitize doesn't
        ie[E[m] == eb[-1]] = ne - 1
        iz[cz[m] == zb[-1]] = nz - 1
        ok = (ie >= 0) & (ie < ne) & (iz >= 0) & (iz < nz)
        flat = np.where(ok, off + ie * nz + iz, -1)
        idx[m] = flat
        off += ne * nz
    return idx, offsets, off


def _geometry_selectors(offsets, n_bins):
    """Per-bin GEOMETRY selectors bakeable into the response npz (Phase E4).
    Pure function of the reco binning (offsets = {sample: (off, ne, nz)}), so a
    baked copy must equal masks.assemble_masks's descriptor output bit-for-bit
    (the engine asserts this at load). Returns {sel_<attr>: array}.

      sel_fcmg_bin_mask   : FC multi-GeV samples (REL_NORM_FCMG_SAMPLES).
      sel_decay_e_bin_mask: SK I-III sub-GeV decay-e samples (DECAY_E_SAMPLES).
      sel_ude_sign        : +1 up-going / -1 down-going / 0 excluded (up/down
                            energy-scale signed mask; z10bins FC+PC only).
    """
    from .sk_binned_engine import (REL_NORM_FCMG_SAMPLES, DECAY_E_SAMPLES,
                                    UPDOWN_ESCALE_EXCLUDE)
    bin_sample = np.empty(n_bins, dtype=int)
    for s, (off, ne, nz) in offsets.items():
        bin_sample[off:off + ne * nz] = int(s)
    fcmg = np.isin(bin_sample, list(REL_NORM_FCMG_SAMPLES))
    decay_e = np.isin(bin_sample, list(DECAY_E_SAMPLES))
    ude = np.zeros(n_bins)
    for s, (off, ne_, nz) in offsets.items():
        if int(s) in UPDOWN_ESCALE_EXCLUDE or nz != 10:
            continue
        for ie in range(ne_):
            for iz in range(nz):
                ude[off + ie * nz + iz] = 1.0 if iz < nz // 2 else -1.0
    return {"sel_fcmg_bin_mask": fcmg, "sel_decay_e_bin_mask": decay_e,
            "sel_ude_sign": ude}


def build_response(pynufit, exp_name, out_path=None, n_etrue=200, n_cztrue=40,
                   dial_manifest=None, bake_selectors=False):
    """Build the SK binned response from the event MC of ``pynufit``'s
    ``exp_name`` experiment. Returns the sparse-response dict (the exact set of
    arrays the npz schema stores). If ``out_path`` is given, also writes an npz
    byte-compatible with the existing ``SKBinnedEngine`` loader.

    ``dial_manifest`` (optional, iterable of dial names in seed order): hashed
    into the additive ``dial_manifest_hash`` key for the engine's optional
    load-time compatibility check (§2.5 upgrade 3). None -> no hash key.

    The heavy inputs (``exp.BaseWeight``, ``exp.BinData()``, ``exp.BinMC``, the
    ``WaterXSection`` tunes) are read from the live object; nothing is
    re-derived — the same guarantee the standalone script gives.
    """
    exp = pynufit.Experiments[exp_name]
    pt = pynufit.physics_tunes[exp_name]

    N = exp.NumberOfEvents

    # ---- class signatures from the ACTUAL xsec tunes (x=2 -> factor 2 marks mask)
    xsec_obj = pt.XSectionTunes if hasattr(pt, "XSectionTunes") else None
    bits = np.zeros((N, len(XSEC_TUNES)), dtype=np.int8)
    for j, name in enumerate(XSEC_TUNES):
        if xsec_obj is not None and hasattr(xsec_obj, name):
            w = getattr(xsec_obj, name)(exp, 2.0)
        else:  # fall back through the dispatcher
            w = pt.get_xsection(name, 2.0)
        bits[:, j] = (np.asarray(w) != 1.0).astype(np.int8)
    cc = np.asarray(exp.CC, dtype=np.int8)
    pdg = np.asarray(exp.nuPDG, dtype=np.int64)

    sig = np.column_stack([pdg, cc, bits])
    classes, class_inv = np.unique(sig, axis=0, return_inverse=True)
    n_cls = classes.shape[0]

    # ---- true grid
    e_edges, z_edges = make_true_grid(exp.Etrue_min, exp.Etrue_max,
                                      n_etrue, n_cztrue)
    ie = np.clip(np.digitize(exp.ETrue, e_edges) - 1, 0, n_etrue - 1)
    iz = np.clip(np.digitize(exp.CosZTrue, z_edges) - 1, 0, n_cztrue - 1)

    # ---- reco bin indices (nominal and +-2% energy scale)
    b0, offsets, n_bins = _reco_bin_index(exp, 1.0)
    bp, _, _ = _reco_bin_index(exp, 1.02)
    bm, _, _ = _reco_bin_index(exp, 0.98)

    w = np.asarray(exp.BaseWeight, float)
    w2 = w * w

    # ---- SK run-period era group per event (sk_phase 1..5 -> 0=SK-I ... 3=SK-IV+V)
    N_ERA = 4
    skphase = np.asarray(getattr(exp, "SKPhase", np.zeros(N, dtype=np.int64)),
                         dtype=np.int64)
    era = np.minimum(np.maximum(skphase, 1), 4) - 1

    def coo(bidx, weights):
        m = bidx >= 0
        # key digits slowest->fastest: class, ie, iz, recobin, era
        key = (((class_inv[m].astype(np.int64) * n_etrue + ie[m])
                * n_cztrue + iz[m]) * n_bins + bidx[m]) * N_ERA + era[m]
        uniq, inv = np.unique(key, return_inverse=True)
        val = np.bincount(inv, weights=weights[m], minlength=uniq.size)
        er = uniq % N_ERA
        rem = uniq // N_ERA
        b = rem % n_bins
        rem = rem // n_bins
        cz_ = rem % n_cztrue
        rem = rem // n_cztrue
        ce = rem % n_etrue
        k = rem // n_etrue
        return (k.astype(np.int32), ce.astype(np.int32), cz_.astype(np.int32),
                b.astype(np.int32), er.astype(np.int8), val)

    Rk, Re, Rz, Rb, Rera, Rv = coo(b0, w)
    Pk, Pe, Pz, Pb, Pera, Pv = coo(bp, w)
    Mk, Me, Mz, Mb, Mera, Mv = coo(bm, w)
    Sk_, Se, Sz, Sb, Sera, Sv = coo(b0, w2)

    # ---- observed data vector (unfiltered, release order)
    obs = np.asarray(exp.BinData(), float)
    if obs.size != n_bins:
        raise RuntimeError(
            f"observed vector size {obs.size} != n_bins {n_bins}")

    # ---- self-check: R summed over (k, cells) == BinMC(ones)
    binned_base = np.asarray(exp.BinMC(np.ones(N)))
    rsum = np.zeros(n_bins)
    np.add.at(rsum, Rb, Rv)
    resid = np.max(np.abs(rsum - binned_base))
    if resid >= 1e-6 * max(binned_base.max(), 1):
        raise RuntimeError(
            f"R does not reproduce BinMC(ones) (max |diff| {resid:.3e}) "
            "— bin replication bug")

    sample_table = json.dumps({int(s): offsets[int(s)] for s in exp.Samples})
    # raw (unweighted) event counts per sample — some detector tunes (e.g.
    # fcpc_separation) use np.sum(mask) raw counts, which the binned engine
    # replicates verbatim.
    sample_counts = {int(s): int(np.sum(exp.Sample == s)) for s in exp.Samples}

    resp = dict(
        sample_event_counts=json.dumps(sample_counts),
        classes=classes, xsec_tune_names=np.array(XSEC_TUNES),
        R_k=Rk, R_e=Re, R_z=Rz, R_b=Rb, R_era=Rera, R_v=Rv,
        Rp_k=Pk, Rp_e=Pe, Rp_z=Pz, Rp_b=Pb, Rp_era=Pera, Rp_v=Pv,
        Rm_k=Mk, Rm_e=Me, Rm_z=Mz, Rm_b=Mb, Rm_era=Mera, Rm_v=Mv,
        S2_k=Sk_, S2_e=Se, S2_z=Sz, S2_b=Sb, S2_era=Sera, S2_v=Sv,
        n_era=np.int64(N_ERA),
        e_edges=e_edges, z_edges=z_edges, n_bins=np.int64(n_bins),
        observed=obs, sample_table=sample_table,
        meta=json.dumps({
            "config": getattr(pynufit, "_analysis_file", exp_name),
            "n_events": int(N), "n_classes": int(n_cls),
            "etrue_range": [float(exp.Etrue_min), float(exp.Etrue_max)],
            "n_etrue": n_etrue, "n_cztrue": n_cztrue,
            "binned_base_total": float(binned_base.sum()),
        }),
    )

    # ---- ADDITIVE keys (§2.5 upgrade 2 + 3): ignored by the current engine,
    #      checked at load ONLY when present.
    resp["schema_version"] = np.array(SCHEMA_VERSION)
    if dial_manifest is not None:
        manifest = list(dial_manifest)
        h = hashlib.sha256("\n".join(manifest).encode("utf-8")).hexdigest()
        resp["dial_manifest_hash"] = np.array(h)
        resp["dial_manifest"] = np.array(manifest)

    # ---- baked per-bin geometry selectors (Phase E4, additive, OFF by default).
    #      The engine ignores absent keys (today's responses); when present it
    #      uses them AND asserts byte-equality with masks.assemble_masks.
    if bake_selectors:
        resp.update(_geometry_selectors(offsets, n_bins))

    if out_path is not None:
        np.savez_compressed(out_path, **resp)

    return resp


# ---------------------------------------------------------------------------
# Oscillation-tensor builder (transcribed from build_osc_tensors.py)
# ---------------------------------------------------------------------------

def _resolve_avg_scale(value):
    """Resolve an averaging selector to a nuSQuIDS EvalFlavor scale (float) or
    None (OFF). A faithful copy of ``AtmOsc._resolve_avg_scale`` kept here so the
    tensor builder does not trigger the heavy AtmOsc import chain (nuSQuIDS) just
    to interpret a token — the resolution rule is identical, so back-compat with
    the ctor is exact. Accepts '2pi'/'4pi', a float or float-string, or
    None/''/'off'/'none' -> OFF."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("", "off", "none"):
            return None
        tok = {"2pi": 2.0 * np.pi, "4pi": 4.0 * np.pi}.get(s)
        return tok if tok is not None else float(s)
    return float(value)


# the per-event arrays set_grid_coords overrides; snapshotted/restored around a
# build so an event-engine call on the same osc object is byte-unaffected.
_OSC_GRID_ATTRS = ("ETrue", "CosZTrue", "NSQneutype", "NSQneuflavor")
# osc cache attributes reset_cache clears; also snapshotted so the object's cache
# state is restored to exactly what it was pre-call.
_OSC_CACHE_ATTRS = ("_last_params", "_cached_weights")


def _snapshot_osc_state(osc):
    """Deep-copy the osc object's per-event coordinate arrays, its ``Parameters``
    dict, and its cache attributes so ``_restore_osc_state`` can put the object
    back exactly (array_equal, not allclose)."""
    snap = {}
    for a in _OSC_GRID_ATTRS:
        v = getattr(osc, a, None)
        snap[a] = np.array(v, copy=True) if isinstance(v, np.ndarray) \
            else (list(v) if isinstance(v, list) else v)
    snap["Parameters"] = dict(osc.Parameters)
    for a in _OSC_CACHE_ATTRS:
        v = getattr(osc, a, None)
        snap[a] = np.array(v, copy=True) if isinstance(v, np.ndarray) else v
    snap["osc_avg_scale"] = getattr(osc, "osc_avg_scale", None)
    return snap


def _restore_osc_state(osc, snap):
    """Restore every attribute captured by ``_snapshot_osc_state``."""
    for a in _OSC_GRID_ATTRS:
        setattr(osc, a, snap[a])
    osc.Parameters.clear()
    osc.Parameters.update(snap["Parameters"])
    for a in _OSC_CACHE_ATTRS:
        setattr(osc, a, snap[a])
    osc.osc_avg_scale = snap["osc_avg_scale"]


def _set_grid_coords(osc, e_c, z_c):
    """Override the osc object's per-event arrays with the tiled (type, flavor,
    E, cz) grid so GetOscillations evaluates at cell centres.

    Verbatim ``build_osc_tensors.set_grid_coords``."""
    nE, nZ = e_c.size, z_c.size
    E2, Z2 = np.meshgrid(e_c, z_c, indexing="ij")     # (nE, nZ)
    cells_E = E2.ravel()
    cells_Z = Z2.ravel()
    n_cells = cells_E.size
    types, flavors = [], []
    Es, Zs = [], []
    for t in (0, 1):           # 0 = nu, 1 = nubar  (NSQNeutrinoType convention)
        for f in (0, 1, 2):    # e, mu, tau         (NSQNeutrinoFlavor convention)
            types.append(np.full(n_cells, t, dtype=np.uint32))
            flavors.append(np.full(n_cells, f, dtype=np.uint32))
            Es.append(cells_E)
            Zs.append(cells_Z)
    osc.ETrue = np.concatenate(Es)
    osc.CosZTrue = np.concatenate(Zs)
    osc.NSQneutype = np.concatenate(types).tolist()
    osc.NSQneuflavor = np.concatenate(flavors).tolist()
    return nE, nZ


def _eval_point(osc, dm231, s23, dcp, nE, nZ, s13=None, dm221=None):
    """One node evaluation. Verbatim ``build_osc_tensors.eval_point``."""
    P = osc.Parameters
    P["Dm231"] = dm231
    if "Dm231_bar" in P:
        P["Dm231_bar"] = dm231
    P["Sin2Theta23"] = s23
    P["dCP"] = dcp
    if s13 is not None:
        P["Sin2Theta13"] = s13
    if dm221 is not None:
        P["Dm221"] = dm221
    osc.reset_cache()
    w = np.asarray(osc.GetOscillations(), float)
    return w.reshape(2, 3, nE, nZ).astype(np.float32)


def build_tensors(pynufit, exp_name, dm231, s23, dcp_nodes=None, s13=None,
                  n_etrue=200, n_cztrue=40, avg_scale=None, out_path=None):
    """Build the oscillated-flux tensor Phi[n_dcp, 2, 3, nE, nZ] at (dm231, s23)
    for the ``exp_name`` experiment of ``pynufit``, one slice per dCP node.

    Runs against the live ``AtmosphericOscillations`` object (so propagation,
    units, flux init, and the Dm231_bar->Dm231 convention are inherited). The
    object's per-event coordinate arrays, ``Parameters``, and cache are
    snapshotted before the build and restored in a ``finally`` — the object is
    byte-identical afterwards, whether the build finishes or raises.

    Args:
      dm231, s23: node oscillation parameters.
      dcp_nodes: iterable of dCP values (radians). None -> a single dcp=0 slice
        (Phi shape (1, 2, 3, nE, nZ)).
      s13: optional Sin2Theta13 override (None -> config/NOM value).
      avg_scale: fast-oscillation averaging selector for this build (§2.5
        upgrade 1). None -> the osc object's current setting is left as-is
        (which honours PYNU_OSC_AVG_SCALE from the environment). A token
        ('4pi'/'2pi'/'off'/float) sets ``osc.osc_avg_scale`` for the build.
        Env override precedence: if PYNU_OSC_AVG_SCALE is set it wins, matching
        the AtmOsc ctor's back-compat rule.
      out_path: optional npz path (schema-compatible with the tensor loader,
        plus the additive ``osc_averaging`` provenance key).

    Returns:
      (phi, meta) — phi float32 (n_dcp, 2, 3, nE, nZ); meta dict with the grid
      edges, the dcp node array, and the averaging actually applied.
    """
    import os

    exp = pynufit.Experiments[exp_name]
    osc = pynufit.physics_tunes[exp_name].OscillationTunes

    dcp_arr = (np.asarray(list(dcp_nodes), float)
               if dcp_nodes is not None else np.array([0.0]))

    e_edges, z_edges = make_true_grid(exp.Etrue_min, exp.Etrue_max,
                                      n_etrue, n_cztrue)
    e_c, z_c = grid_centers(e_edges, z_edges)

    snap = _snapshot_osc_state(osc)
    try:
        # §2.5 upgrade 1: env override wins (back-compat), else the ctor-style
        # selector passed here, else leave the object as-is.
        env = os.environ.get("PYNU_OSC_AVG_SCALE")
        if env is not None:
            osc.osc_avg_scale = _resolve_avg_scale(env)
            applied_avg = env
        elif avg_scale is not None:
            osc.osc_avg_scale = _resolve_avg_scale(avg_scale)
            applied_avg = avg_scale
        else:
            applied_avg = "off" if osc.osc_avg_scale is None else "on"

        nE, nZ = _set_grid_coords(osc, e_c, z_c)
        phi = np.stack([
            _eval_point(osc, dm231, s23, dcp, nE, nZ, s13=s13)
            for dcp in dcp_arr])
    finally:
        _restore_osc_state(osc, snap)

    meta = {
        "dm231": float(dm231), "s23": float(s23),
        "dcp": dcp_arr, "e_edges": e_edges, "z_edges": z_edges,
        "osc_averaging": str(applied_avg),
    }
    if out_path is not None:
        np.savez_compressed(
            out_path, phi=phi, dm231=float(dm231), s23=float(s23),
            dcp=dcp_arr, e_edges=e_edges, z_edges=z_edges,
            osc_averaging=np.array(str(applied_avg)))

    return phi, meta
