#!/usr/bin/env python3
"""Descriptor-driven mask / selector assembly for the SK binned engine (Phase E4).

Mechanical extraction of the per-dial mask/selector construction out of
``SKBinnedEngine.__init__`` (the geometry masks: FC-multiGeV / decay-e sample
masks, energy-scale reco-E adjacency, flux-ratio class legs, ntag momentum bands,
up-mu background zenith×momentum masks, the up/down signed mask, direction-smear
blocks, the static flux-tune cell fields, and the sub-GeV/multi-GeV xsec class
masks). ``assemble_masks(eng, z)`` sets exactly the same attributes on ``eng`` the
inline code did — ZERO numerical change; every guard, ordering, and comment is
preserved verbatim.

Descriptor-driven: the assembly is parameterized by the declarative descriptor
tables already living in ``sk_binned_engine`` (REL_NORM_FCMG_SAMPLES,
DECAY_E_SAMPLES, NEUTRON_MIG_PAIRS, NTAG_PSPLIT, UPMU_BKG_SHAPE_SPEC,
UPDOWN_ESCALE_EXCLUDE, FLUX_RATIO_SPEC, the band/leg maps) — i.e. sample-ID lists,
band edges and leg selectors, not hand-inlined constants. Mechanism-class grouping
matches src/e45_descriptor_census.json (det-binmask-norm, det-ntag-band-migration,
updown-escale-signed-norm, energy-scale-migration, dir-smear-migration,
flux-band-ratio, xsec-*-norm/shape).

Baked-selector path (Phase E4 forward-compat): ``builder.py`` may write the pure
per-bin GEOMETRY selector arrays into the response npz (keys prefixed ``sel_``).
When a baked array is present it is used AND asserted byte-equal to the
descriptor-assembled array (both paths must agree — the E4-local gate). When
absent (today's responses), the descriptor assembly is authoritative, exactly as
before. Only geometry masks that are a pure function of the response binning are
bakeable; the true-cell flux/xsec fields are functions of the true grid and are
always assembled.
"""
import json

import numpy as np

from .sk_binned_engine import (
    MIN_ENTRIES,
    REL_NORM_FCMG_SAMPLES,
    NEUTRON_MIG_PAIRS,
    DECAY_E_SAMPLES,
    ERA_TAGS,
    NTAG_PSPLIT,
    UPMU_BKG_SHAPE_SPEC,
    UPDOWN_ESCALE_NAMES,
    UPDOWN_ESCALE_EXCLUDE,
    MASK_TUNES,
    KPI_E0,
    CCQE_SHAPE_SUBGEV_E,
    FLUX_RATIO_SPEC,
)

# per-bin GEOMETRY selectors a builder may bake into the response npz. Each is a
# pure function of the reco binning (sample_table + n_bins), so a baked copy must
# equal the descriptor assembly bit-for-bit. Key = engine attribute name.
BAKEABLE_SELECTORS = ("fcmg_bin_mask", "decay_e_bin_mask", "ude_sign")


def _maybe_baked(eng, z, attr, assembled):
    """Return the assembled array, but if the response npz `z` carries a baked
    copy (key ``sel_<attr>``) assert it is byte-equal and prefer it. `z` may be
    None (no npz handle) -> assembly is authoritative."""
    if z is None:
        return assembled
    key = f"sel_{attr}"
    files = getattr(z, "files", [])
    if key not in files:
        return assembled
    baked = np.asarray(z[key])
    if baked.shape != np.asarray(assembled).shape or \
            baked.tobytes() != np.asarray(assembled).tobytes():
        raise ValueError(
            f"baked selector {key!r} in the response npz != descriptor-assembled "
            f"{attr!r} (shapes {baked.shape} vs {np.asarray(assembled).shape}); "
            "the baked response is stale — rebuild it or drop the baked arrays")
    return baked


def assemble_masks(eng, z=None):
    """Populate every mask/selector attribute on ``eng`` (descriptor-driven,
    verbatim). ``z`` is the open response npz (for the optional baked-selector
    path); pass None to force pure descriptor assembly."""
    # FC multi-GeV per-bin mask for the optional rel_norm_fcmg dial
    eng.fcmg_bin_mask = _maybe_baked(
        eng, z, "fcmg_bin_mask",
        np.isin(eng.bin_sample, list(REL_NORM_FCMG_SAMPLES)))
    # neutron-migration donor/acceptor sample pairs (per-dial) and the
    # decay-e per-bin sample mask.
    eng.neutron_mig_pairs = dict(NEUTRON_MIG_PAIRS)
    eng.decay_e_bin_mask = _maybe_baked(
        eng, z, "decay_e_bin_mask",
        np.isin(eng.bin_sample, list(DECAY_E_SAMPLES)))

    # ---- energy-scale bin-level migration geometry (reco-E adjacency) -----
    # Built only when energy_scale dials are active. es_below[b] = the bin one
    # reco-E step lower in the same (sample, reco-cz) column (-1 at ie=0);
    # es_has_above[b] = 1 if b can spill upward (ie < ne-1). Geometry only =>
    # works with the quantized MC; no response rebuild, no Rp/Rm.
    if eng.active_energy_scale:
        if len(eng.active_energy_scale) != eng.n_era:
            raise ValueError("energy_scale needs one dial per era "
                             f"(n_era={eng.n_era}); got {eng.active_energy_scale}")
        eng.es_below = np.full(eng.n_bins, -1, dtype=np.int64)
        eng.es_has_above = np.zeros(eng.n_bins)
        for s, (off, ne_, nz) in eng.sample_table.items():
            for ie in range(ne_):
                base = off + ie * nz
                if ie > 0:
                    eng.es_below[base:base + nz] = np.arange(base, base + nz) - nz
                if ie < ne_ - 1:
                    eng.es_has_above[base:base + nz] = 1.0
        eng.es_has_below = (eng.es_below >= 0).astype(float)
        eng._es_idx = [eng.nuisance_names.index(f"energy_scale_{ERA_TAGS[e]}")
                       for e in range(eng.n_era)]

    # ---- OPTIONAL absorber masks (built only when the dials are active) ----
    # (1) flux-ratio per-class flavor/sign selectors (band E<1 GeV applied
    #     at fit time via self.e_below1, like the AxialMass A_ke factor).
    if eng.active_flux_ratios:
        eng._fr_leg = {
            "nuebar": (eng.cls_pdg == -12), "nue": (eng.cls_pdg == 12),
            "e": (np.abs(eng.cls_pdg) == 12), "mu": (np.abs(eng.cls_pdg) == 14),
            "numubar": (eng.cls_pdg == -14), "numu": (eng.cls_pdg == 14),
        }
        # back-compat aliases used by older code paths
        eng.fr_is_nuebar = eng._fr_leg["nuebar"]
        eng.fr_is_nue = eng._fr_leg["nue"]
        eng.fr_is_e = eng._fr_leg["e"]
        eng.fr_is_mu = eng._fr_leg["mu"]
    # (2) momentum-resolved sub-GeV neutron-tag per-bin band masks. Within a
    #     sample bin = off + ie*nz + iz (build_sk_response.reco_bin_index),
    #     so the reco-momentum index is ie = (bin-off)//nz; ie<NTAG_PSPLIT is
    #     the low band. Donor {20,22}=0-neutron, acceptor {21,23}=1-neutron.
    if eng.ntag_split:
        def _band(sample_ids, low):
            mk = np.zeros(eng.n_bins, dtype=bool)
            for s in sample_ids:
                key = str(s) if str(s) in eng.sample_table else s
                off, ne, nz = eng.sample_table[key]
                idx = np.arange(off, off + ne * nz)
                ie = (idx - off) // nz
                sel = ie < NTAG_PSPLIT if low else ie >= NTAG_PSPLIT
                mk[idx[sel]] = True
            return mk
        # band dial -> (donor per-bin mask, acceptor per-bin mask)
        eng.ntag_bands = {
            "ntag_subgev_lowp": (_band([20, 22], True), _band([21, 23], True)),
            "ntag_subgev_highp": (_band([20, 22], False), _band([21, 23], False)),
        }
    # (3) up-mu background zenith x momentum SHAPE per-bin masks. Bin index =
    #     off + ie*nz + iz (build_sk_response.reco_bin_index); reco-momentum ie,
    #     reco-cosZ iz. Horizon-nearest iz = nz-1 for the up-mu z10bins_up grid.
    if eng.active_upmu_bkg:
        eng.upmu_bkg_masks = {}
        for name in eng.active_upmu_bkg:
            sid, iz_set, ie_set = UPMU_BKG_SHAPE_SPEC[name]
            key = str(sid) if str(sid) in eng.sample_table else sid
            off, ne, nz = eng.sample_table[key]
            mk = np.zeros(eng.n_bins, dtype=bool)
            for ie in range(ne):
                if ie_set is not None and ie not in ie_set:
                    continue
                for iz in iz_set:
                    if 0 <= iz < nz:
                        mk[off + ie * nz + iz] = True
            eng.upmu_bkg_masks[name] = mk

    # (4) up/down energy-scale SIGNED per-bin mask (SK Up/Down Energy Scale,
    #     thesis 6011-6018): +1 on up-going (cz<0) bins, -1 on down-going
    #     (cz>=0), 0 on excluded samples. FC+PC z10bins samples only -- the
    #     z10bins edge index 5 == cz=0 so iz<nz//2 is up-going; up-mu {16,17,18}
    #     (all up-going) + single-reco-zenith FC {1,2,5,6} (nz=1, straddle cz=0)
    #     get factor 1. Era-INDEPENDENT geometry; the per-era dial VALUE is
    #     routed to detector_factors via _era_theta. The signed mask lets one
    #     dial scale up *=(1+d) and down *=(1-d) with the gradient sign built in.
    if eng.active_ude:
        if set(eng.active_ude) != set(UPDOWN_ESCALE_NAMES):
            raise ValueError("up/down energy-scale needs all "
                             f"{len(ERA_TAGS)} era dials {UPDOWN_ESCALE_NAMES}; "
                             f"got {eng.active_ude}")
        if eng.n_era != len(ERA_TAGS):
            raise ValueError("up/down energy-scale dials need the phased "
                             f"{len(ERA_TAGS)}-era response (n_era={eng.n_era})")
        ude_sign = np.zeros(eng.n_bins)
        for s, (off, ne_, nz) in eng.sample_table.items():
            if int(s) in UPDOWN_ESCALE_EXCLUDE or nz != 10:
                continue
            for ie in range(ne_):
                for iz in range(nz):
                    ude_sign[off + ie * nz + iz] = \
                        1.0 if iz < nz // 2 else -1.0
        eng.ude_sign = _maybe_baked(eng, z, "ude_sign", ude_sign)

    # (5) direction-smearing reco-cz confusion matrices. Loaded LAZILY and only
    #     when the dir_smear dial is active -- ALL existing paths are untouched.
    #     Per sample a nz x nz matrix M[sample] (reco-cz confusion, IE-independent);
    #     the migration applies the same matrix to every reco-momentum row of the
    #     sample. Precompute per-sample (off, ne, nz, M) blocks, SKIPPING identity
    #     (z1bins) samples so they are bit-unchanged at any s.
    eng._ds_blocks = []
    if eng.active_dir_smear:
        if eng._dirsmear_matrix_path is None:
            raise ValueError("dir_smear dial active but no dirsmear_matrix given "
                             "(pass dirsmear_matrix=<confusion_*.npz path>)")
        dz = np.load(eng._dirsmear_matrix_path, allow_pickle=True)
        eng.dirsmear_meta = (json.loads(str(dz["manifest"]))
                             if "manifest" in dz.files else {})
        for s, (off, ne_, nz) in eng.sample_table.items():
            M = np.asarray(dz[f"M_{int(s)}"], dtype=float)
            if M.shape != (nz, nz):
                raise ValueError(f"dirsmear M_{s} shape {M.shape} != ({nz},{nz})")
            if np.allclose(M, np.eye(nz), atol=0, rtol=0):
                continue                       # identity (z1bins) -> inert, skip
            eng._ds_blocks.append((int(off), int(ne_), int(nz), M))

    # FewEntries mask from the unfiltered data vector (Experiment.SetObservedBinned)
    eng.few = eng.observed > MIN_ENTRIES
    eng.obs_f = eng.observed[eng.few]

    # static flux-tune cell masks
    eng.e_below1 = eng.e_c < 1.0
    eng.e_above1 = eng.e_c > 1.0
    eng.barr_env = 0.07 / (1.0 + (eng.e_c / 0.5) ** 2)     # _barr_zenith_envelope
    eng.tanh3z = np.tanh(3.0 * eng.z_c)
    eng.tanhz2 = np.tanh(eng.z_c) ** 2          # zenith_up/down envelope
    # H/V flux-ratio shape: mean-zero over cosz, +0.5 horizontal, -1.0 vertical
    eng.horizvert_shape = 0.5 * (1.0 - 3.0 * eng.z_c ** 2)
    eng.log10e = np.log10(eng.e_c)
    # K/pi high-E flux ramp: 0 below KPI_E0, rising as log10(E/KPI_E0) above (nE,)
    eng.kpi_shape = np.maximum(0.0, np.log10(eng.e_c / KPI_E0))

    # energy bands (true E_nu) for the SK 3-band flux ratios
    eng.e_bands = {"sub": (eng.e_c < 1.0).astype(float),
                   "mid": ((eng.e_c >= 1.0) & (eng.e_c < 10.0)).astype(float),
                   "high": (eng.e_c >= 10.0).astype(float)}
    # resolve each active flux-ratio dial -> (band envelope (nE,), heavy (n_cls,),
    # light (n_cls,)) so cell_weights/gradient iterate ONE generic registry.
    eng.fr_resolved = {}
    for nm in eng.active_flux_ratios:
        band, hv, lt = FLUX_RATIO_SPEC[nm]
        eng.fr_resolved[nm] = (eng.e_bands[band],
                               eng._fr_leg[hv], eng._fr_leg[lt])
    # optional sub-GeV xsec masks (CCQE / 2p2h class via the baked mask bits)
    if eng.active_xsec_extra or eng.active_multigev_ccqe:
        eng.ccqe_cls = eng.cls_bits[:, MASK_TUNES.index("CCQE")]   # (n_cls,)
        eng.ccqe_nue_cls = eng.ccqe_cls & (eng.cls_flavor == 0)   # nu_e+nu-bar_e CCQE(=1p1h)
        # multi-GeV CCQE flavor-norm masks: nu_mu+nu-bar_mu CCQE class
        # + the E_true>=1.33 GeV complement of the sub-GeV shape region. Built
        # here (unused, hence output-inert) for any active_xsec_extra spec.
        eng.ccqe_numu_cls = eng.ccqe_cls & (eng.cls_flavor == 1)  # nu_mu+nu-bar_mu CCQE
        eng.e_multigev = eng.e_c >= CCQE_SHAPE_SUBGEV_E            # (nE,) complement of <1.33
        # sub-GeV-localized CCQE shape: mean-zero log-E tilt confined to E_true<1.33 GeV
        # (0 above; centred over the sub-GeV cells => ~rate-neutral SHAPE, pivot-free).
        eng.ccqe_shape_subgev = np.where(eng.e_c < CCQE_SHAPE_SUBGEV_E,
                                         np.log(eng.e_c), 0.0)            # (nE,)
        _sub = eng.e_c < CCQE_SHAPE_SUBGEV_E
        eng.ccqe_shape_subgev[_sub] -= eng.ccqe_shape_subgev[_sub].mean()
        # 2p2h nu_e class (thesis split); zeros if the response predates 2p2h
        if "CC_2p2h" in MASK_TUNES:
            twop2h = eng.cls_bits[:, MASK_TUNES.index("CC_2p2h")]
            eng.twop2h_nue_cls = twop2h & (eng.cls_flavor == 0)
        else:
            eng.twop2h_nue_cls = np.zeros(eng.n_cls, dtype=bool)


def bakeable_selector_arrays(eng):
    """Return {sel_<attr>: array} for the per-bin GEOMETRY selectors a builder can
    bake into the response npz. Computed from a fully-assembled engine. `ude_sign`
    is included only when the up/down set is active (else the attribute is absent).
    """
    out = {}
    for attr in BAKEABLE_SELECTORS:
        if hasattr(eng, attr):
            out[f"sel_{attr}"] = np.asarray(getattr(eng, attr))
    return out
