#!/usr/bin/env python3
"""Descriptor-driven detector-factor kernels for the SK binned engine (Phase E5b).

Native home of ``SKBinnedEngine.detector_factors`` / ``sample_rates`` — the
per-sample detector multiplicative factor D_s and its per-tune d ln D. Extracted
verbatim from the engine (self -> eng); the internal generic kernels
(``wsum`` / ``safe_ratio`` / ``mig`` / ``apply2`` — sample-norm and migration-pair
mechanisms) and the per-bin fold-ins (ntag-split band migration, rel_norm,
neutron-production migration, decay-e, up-mu bkg shape, up/down signed norm) are
preserved in EXACT order with EVERY guard intact:
  - fcpc_separation on RAW counts with the coupled PC leg y = ((wpc+wfc) - x*wfc)/wpc
  - multiring_emu_separation (2-x) guard applied AFTER the build order
  - multiring_pid x->1 snap within 1e-4 then acceptor guard
  - _unphys 1e-3 floor with NO derivative contribution
  - safe_ratio None -> per-era no-op (empty acceptor, e.g. ntag in eras 0-2)
ZERO numerical change: the E5b gate byte-compares (D, dlnD) per era vs the frozen
baseline. Descriptor-driven = the sample-ID lists / migration pairs are the
declarative descriptors (census mechanism classes det-sample-norm /
det-migration-pair / det-ntag-band-migration / det-binmask-norm /
updown-escale-signed-norm / neutron-prod-migration); the algebra is generic.
"""
import numpy as np

from .sk_binned_engine import (
    DET_NAMES,
    DECAY_E_NAME,
    _unphys,
)


def sample_rates(eng, n_phys):
    """Weighted physics rate per sample (BaseWeight*PhysicsWeight sums)."""
    r = {}
    for s in eng.samples:
        r[int(s)] = float(n_phys[eng.bin_sample == s].sum())
    return r


def detector_factors(eng, t, rates, n_phys=None):
    """Per-sample multiplicative factor D_s and its per-tune derivative
    dD_s (with migration ratios r held fixed — matches the event engine's
    analytic-gradient convention). Returns (D[s], {tune: dD[s]}).

    Transcribed tune-by-tune from SKCombinedDetector.SuperK_Combined.

    ``t`` is a {base_detector_name: value} dict. In legacy (single-era) mode
    the caller passes ``dict(zip(nuisance_names, theta))``; in phased mode it
    passes a per-era view that maps each base stem to that era's dial value,
    so this body is reused verbatim per era.

    n_phys (per-bin physics rate) is required only when the sub-GeV
    neutron-tag momentum split is active (per-bin band migration ratios).
    """
    S = {int(s): 1.0 for s in eng.samples}
    dS = {n: {int(s): 0.0 for s in eng.samples} for n in DET_NAMES}

    if eng.migration_mode == "rawcount":
        counts_ = {int(k): float(v) for k, v in eng.sample_counts.items()}

        def wsum(ids):
            return sum(counts_.get(i, 0.0) for i in np.atleast_1d(ids))
    else:
        def wsum(ids):
            return sum(rates.get(i, 0.0) for i in np.atleast_1d(ids))

    # Derivative convention: the event engine uses dW/W = diff_tune/tune per
    # event; for a donor sample with factor x that's 1/x, for an acceptor
    # with factor (1+r(1-x)) it's -r/(1+r(1-x)). We accumulate d(ln D).

    counts = {int(k): v for k, v in eng.sample_counts.items()}

    # fcpc_separation — RAW counts (np.sum(mask)), not weighted rates
    x = t["fcpc_separation"]
    pc_ids, um_ids = [14, 15], [16, 17, 18]
    wfc = sum(c for s_, c in counts.items() if s_ not in pc_ids + um_ids)
    wpc = sum(counts.get(i, 0) for i in pc_ids)
    fc_ids = [int(s) for s in eng.samples if int(s) not in pc_ids + um_ids]
    if _unphys(x):
        y = (wpc + wfc) / wpc
        for s_ in fc_ids:
            S[s_] *= 1e-3
        for s_ in pc_ids:
            S[s_] *= y
    else:
        y = ((wpc + wfc) - x * wfc) / wpc
        for s_ in fc_ids:
            S[s_] *= x
            dS["fcpc_separation"][s_] += 1.0 / x
        for s_ in pc_ids:
            S[s_] *= y
            dS["fcpc_separation"][s_] += (-wfc / wpc) / y

    # pc_reduction: pc *= x
    x = t["pc_reduction"]
    for s_ in pc_ids:
        S[s_] *= (1e-3 if _unphys(x) else x)
        if not _unphys(x):
            dS["pc_reduction"][s_] += 1.0 / x

    # mge_nonubkg: samples {7,8,24,25,26} *= x
    x = t["mge_nonubkg"]
    for s_ in [7, 8, 24, 25, 26]:
        if s_ in S:
            S[s_] *= (1e-3 if _unphys(x) else x)
            if not _unphys(x):
                dS["mge_nonubkg"][s_] += 1.0 / x

    # fc_reduction: all non-pc non-upmu *= x
    x = t["fc_reduction"]
    for s_ in fc_ids:
        S[s_] *= (1e-3 if _unphys(x) else x)
        if not _unphys(x):
            dS["fc_reduction"][s_] += 1.0 / x

    # fiducial_volume: global factor x (all events)
    x = t["fiducial_volume"]
    glob = 1e-3 if _unphys(x) else x
    for s_ in S:
        S[s_] *= glob
        if not _unphys(x):
            dS["fiducial_volume"][s_] += 1.0 / x

    # subgev_2ring_pi0: sample 6 *= x
    x = t["subgev_2ring_pi0"]
    if 6 in S:
        S[6] *= (1e-3 if _unphys(x) else x)
        if not _unphys(x):
            dS["subgev_2ring_pi0"][6] += 1.0 / x

    # subgev_1ring_pi0: sample 2 *= x
    x = t["subgev_1ring_pi0"]
    if 2 in S:
        S[2] *= (1e-3 if _unphys(x) else x)
        if not _unphys(x):
            dS["subgev_1ring_pi0"][2] += 1.0 / x

    # migration pairs (weighted rates). Per-era rates can leave an acceptor
    # sample empty (neutron-tag samples exist only in SK-IV+V), making the
    # ratio undefined -> safe_ratio returns None and the dial is a no-op for
    # that era (correct: no events to migrate, no derivative).
    def safe_ratio(donor, acceptor):
        wa = wsum(acceptor)
        return (wsum(donor) / wa) if wa > 0 else None

    def mig(name, donor, acceptor):
        x_ = t[name]
        if _unphys(x_):
            return
        r_ = safe_ratio(donor, acceptor)
        if r_ is None:
            return
        apply2(name, donor, acceptor, x_, r_)

    def apply2(name, donor, acceptor, x_, r_):
        acc = 1.0 + r_ * (1.0 - x_)
        for s_ in np.atleast_1d(donor):
            if s_ in S:
                S[s_] *= x_
                dS[name][s_] += 1.0 / x_
        for s_ in np.atleast_1d(acceptor):
            if s_ in S:
                S[s_] *= acc
                dS[name][s_] += -r_ / acc

    mig("multiring_nunubar_separation", 10, 11)
    # multiring_emu_separation has guard on (2-x) AFTER build — replicate order
    x = t["multiring_emu_separation"]
    if not _unphys(x):
        r_ = safe_ratio([10, 11, 13], 12)
        if r_ is not None and not _unphys(2 - x):
            apply2("multiring_emu_separation", [10, 11, 13], 12, x, r_)
    mig("multiring_eother_separation", [10, 11], 13)
    mig("pc_stopthru_separation", 14, 15)
    mig("pi0_ring_separation", 2, 6)
    mig("e_ring_separation", [0, 1, 7, 8, 19, 20, 21], [10, 11, 13, 24, 25, 26])
    mig("mu_ring_separation", [3, 4, 5, 9, 22, 23, 27, 28], [12])

    # singlering_pid (guard 1+r(1-x) > 0)
    x = t["singlering_pid"]
    if not _unphys(x):
        e_ids = [0, 1, 7, 8, 19, 20, 21, 24, 25, 26]
        mu_ids = [3, 4, 5, 9, 22, 23, 27, 28]
        r_ = safe_ratio(e_ids, mu_ids)
        if r_ is not None and not _unphys(1 + r_ * (1 - x)):
            apply2("singlering_pid", e_ids, mu_ids, x, r_)

    # multiring_pid (snap x->1 within 1e-4, guard acceptor)
    x = t["multiring_pid"]
    if not _unphys(x):
        if abs(1 - x) < 1e-4:
            x = 1.0
        r_ = safe_ratio([10, 11, 13], 12)
        if r_ is not None and not _unphys(1 + r_ * (1 - x)):
            apply2("multiring_pid", [10, 11, 13], [12], x, r_)

    # neutron tagging by era (mask ratios == sample-rate ratios here).
    # whole-sample sub-GeV form is skipped when the momentum split is active
    # (handled per-bin after D is built, below).
    if not eng.ntag_split:
        x = t["neutron_tagging_subgev"]
        if not _unphys(x):
            r_ = safe_ratio([20, 22], [21, 23])
            if r_ is not None:
                apply2("neutron_tagging_subgev", [20, 22], [21, 23], x, r_)
    x = t["neutron_tagging_multigev"]
    if not _unphys(x):
        r_ = safe_ratio([25, 27], [26, 28])
        if r_ is not None:
            apply2("neutron_tagging_multigev", [25, 27], [26, 28], x, r_)

    # upmu tunes
    x = t["upmu_shower_separation"]
    if not _unphys(x):
        r_ = safe_ratio(18, 17)
        if r_ is not None and not _unphys(1 + r_ * (1 - x)):
            apply2("upmu_shower_separation", 18, 17, x, r_)
    for name, sid in [("upmu_stop_bkg", 16), ("upmu_showering_bkg", 18),
                      ("upmu_nonshowering_bkg", 17)]:
        x = t[name]
        if sid in S:
            S[sid] *= (1e-3 if _unphys(x) else x)
            if not _unphys(x):
                dS[name][sid] += 1.0 / x

    D = np.array([S[int(s)] for s in eng.bin_sample])
    dlnD = {n: np.array([dS[n][int(s)] for s in eng.bin_sample])
            for n in eng.det_names}

    # momentum-resolved sub-GeV neutron tag: per-bin rate-conserving band
    # migrations (donor {20,22} 0-neutron -> acceptor {21,23} 1-neutron),
    # r computed per band from the per-bin physics rate. d ln D convention
    # matches apply2: donor 1/x, acceptor -r/(1+r(1-x)).
    if eng.ntag_split:
        if n_phys is None:
            raise ValueError("ntag split needs per-bin n_phys")
        for name, (donor_m, acc_m) in eng.ntag_bands.items():
            d = np.zeros(eng.n_bins)
            x = t[name]
            if _unphys(x):
                dlnD[name] = d
                continue
            ra = float(n_phys[acc_m].sum())
            r_ = float(n_phys[donor_m].sum()) / ra if ra > 0 else 0.0
            acc = 1.0 + r_ * (1.0 - x)
            fac = np.ones(eng.n_bins)
            fac[donor_m] = x
            d[donor_m] = 1.0 / x
            fac[acc_m] = acc
            d[acc_m] = -r_ / acc
            D = D * fac
            dlnD[name] = d

    # rel_norm_fcmg: flat relative normalization on the FC multi-GeV sample group
    # (SK Rel.Norm FC-MultiGeV, thesis -1.33 sigma). Per-bin scale folded into D so
    # the existing detector gradient (dlnD) handles it; nominal x=1 (no-op),
    # d ln D/dx = 1/x on those bins. var rides D^2 automatically.
    if "rel_norm_fcmg" in t:
        x = t["rel_norm_fcmg"]
        D = D * np.where(eng.fcmg_bin_mask, x, 1.0)
        dlnD["rel_norm_fcmg"] = np.where(eng.fcmg_bin_mask,
                                         (1.0 / x if x != 0 else 0.0), 0.0)

    # neutron-production 0n/1n migration: one PER-PAIR rate-conserving
    # migration between the explicit 0-neutron (donor) and 1-neutron (acceptor)
    # nuebar sample of each pair. Same algebra as apply2 -- donor *= x, acceptor
    # *= 1+r(1-x), r = weighted rate(donor)/rate(acceptor) -- but folded into the
    # already-built per-bin D via the donor/acceptor sample masks (like ntag_split),
    # so total (donor+acceptor) rate is invariant under x. Era-independent: the 0n/1n
    # samples exist only in SK IV+V, so per-era rates leave them empty in eras 0-2
    # (safe_ratio -> None => no-op, no derivative), exactly like the whole-sample ntag.
    if eng.active_neutron_mig:
        for name in eng.active_neutron_mig:
            if name not in t:
                continue
            d = np.zeros(eng.n_bins)
            x = t[name]
            donor_s, acc_s = eng.neutron_mig_pairs[name]
            donor_m = eng.bin_sample == donor_s
            acc_m = eng.bin_sample == acc_s
            ra = float(rates.get(acc_s, 0.0)) if rates is not None else 0.0
            if _unphys(x) or ra <= 0.0:
                dlnD[name] = d          # ratio undefined / unphysical => no-op
                continue
            r_ = float(rates.get(donor_s, 0.0)) / ra
            acc = 1.0 + r_ * (1.0 - x)
            fac = np.ones(eng.n_bins)
            fac[donor_m] = x
            d[donor_m] = 1.0 / x
            fac[acc_m] = acc
            d[acc_m] = (-r_ / acc) if acc != 0 else 0.0
            D = D * fac
            dlnD[name] = d

    # decay-e tagging: flat multiplicative efficiency norm on the SK I-III
    # sub-GeV decay-e-keyed samples (thesis 6022-6024 -- "proportionally changes the
    # normalization"), folded into D like rel_norm_fcmg. nominal x=1 (no-op),
    # d ln D/dx = 1/x on the masked bins. Era-independent (those samples are SK I-III
    # only), so the same whole-sample mask is applied every era.
    if eng.active_decay_e and DECAY_E_NAME in t:
        x = t[DECAY_E_NAME]
        D = D * np.where(eng.decay_e_bin_mask, x, 1.0)
        dlnD[DECAY_E_NAME] = np.where(eng.decay_e_bin_mask,
                                      (1.0 / x if x != 0 else 0.0), 0.0)

    # up-mu background zenith x momentum SHAPE (SK cosmic-mu bkg subtraction,
    # thesis 5975-5984): multiplicative factor on the near-horizon affected model
    # bins, like rel_norm_fcmg (no separate bkg component). nominal x=1 (no-op),
    # d ln D/dx = 1/x on the masked bins -> rides the existing detector gradient.
    # Era-common (mask era-independent; gradient accumulates over eras).
    if eng.active_upmu_bkg:
        for name in eng.active_upmu_bkg:
            if name not in t:
                continue
            x = t[name]
            mk = eng.upmu_bkg_masks[name]
            D = D * np.where(mk, x, 1.0)
            dlnD[name] = np.where(mk, (1.0 / x if x != 0 else 0.0), 0.0)

    # up/down energy-scale (SK Up/Down Energy Scale, thesis 6011-6018):
    # anti-symmetric per-era normalization of up-going (*= 1+d) vs down-going
    # (*= 1-d) FC+PC bins via the signed mask (up +1, down -1, excluded 0).
    # nominal d=0 => factor 1 everywhere (exact no-op). Era-SPLIT: t["updown_escale"]
    # is the current era's dial value (routed by _era_theta); the dlnD sign
    # DIFFERS between up and down: d ln D/dd = sign/(1 + d*sign) -> +1/(1+d) up,
    # -1/(1-d) down. Keyed on the base stem 'updown_escale' (the gradient routes
    # it to updown_escale_<era> via the split set), like the DET_ERA_STEMS.
    if eng.active_ude and "updown_escale" in t:
        d = t["updown_escale"]
        fac = 1.0 + d * eng.ude_sign
        D = D * fac
        dlnD["updown_escale"] = np.divide(eng.ude_sign, fac,
                                          out=np.zeros_like(fac), where=fac != 0)
    return D, dlnD
