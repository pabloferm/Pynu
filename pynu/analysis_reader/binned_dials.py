#!/usr/bin/env python3
"""SK binned-engine dial vocabulary + XML dial-value authority (Track T / T1).

The COMPLETE dial vocabulary (name lists, sigma/box tables, spec registry) and
the XML value authority (``XML_DIAL_VALUES``, ``resolve_nuisance_spec``, the
production θ-order assert) — re-homed here from the top of
``sk_binned_engine.py`` VERBATIM (Track T phase T1; zero value change; the one
edited line is the value-XML directory anchor, which no longer equals
``dirname(__file__)``).

This is a LEAF module: stdlib + numpy only, no pynu imports — so the fitter
(``fitter/minimizer/binned_fit.py``), PhysicsTunes (``Detector/detector.py``),
and the engine itself all import the vocabulary DOWNWARD from here. That breaks
the former ``binned_fit`` ↔ engine ↔ ``engine_core`` import cycle and rights
the dial-table authority direction (the engine consumes the vocabulary; it is
no longer its source). The engine module re-imports every name, so its own
namespace — the surface the kernels, gates, and historical scripts reference —
is unchanged.
"""
import numpy as np

# Nuisance vector order — must match pynufit.Analysis.NuisanceList for the
# xsec_barr_ntag config (verified against point_*.json nuisance_names).
FLUX_NAMES = ["normalization_below1GeV", "normalization_above1GeV", "tilt",
              "nunubar_ratio", "flavor_ratio", "barr_zenith"]
# 15 mask xsec tunes in the order of build_sk_response.XSEC_TUNES (class bits),
# but the VECTOR order below is the Analysis order (AxialMass sits after NCHad).
# The 3 CC_2p2h* dials (real 2p2h, |Mode|==2) sit after DIS, before CCQE — the
# new pfm config order. They require the 2p2h MC + a
# response rebuilt with these mask bits; the engine asserts the response's
# xsec_tune_names == MASK_TUNES, so this is forward-only (old 12-bit responses
# no longer load — by design; their grids are preserved as files).
XSEC_VECTOR_NAMES = ["XSecNuTau", "NCoverCC", "NCHad", "AxialMass", "DIS",
                     "CC_2p2h", "CC_2p2hNuBarNu", "CC_2p2hMuE",
                     "CCQE", "CCQENuBarNu", "CCQEMuE", "CC1Pi_Pi0Pi",
                     "CC1Pi_NuBarNuE", "CC1Pi_NuBarNuMu", "CC1PiProduction",
                     "CohPiProduction"]
MASK_TUNES = ["XSecNuTau", "NCoverCC", "NCHad", "DIS",
              "CC_2p2h", "CC_2p2hNuBarNu", "CC_2p2hMuE",
              "CCQE", "CCQENuBarNu", "CCQEMuE", "CC1Pi_Pi0Pi",
              "CC1Pi_NuBarNuE", "CC1Pi_NuBarNuMu",
              "CC1PiProduction", "CohPiProduction"]
DET_NAMES = ["fcpc_separation", "pc_reduction", "mge_nonubkg", "fc_reduction",
             "fiducial_volume", "subgev_2ring_pi0", "subgev_1ring_pi0",
             "multiring_nunubar_separation", "multiring_emu_separation",
             "multiring_eother_separation", "pc_stopthru_separation",
             "pi0_ring_separation", "e_ring_separation", "mu_ring_separation",
             "singlering_pid", "multiring_pid", "neutron_tagging_subgev",
             "neutron_tagging_multigev", "upmu_shower_separation",
             "upmu_stop_bkg", "upmu_showering_bkg", "upmu_nonshowering_bkg"]
NUISANCE_NAMES = FLUX_NAMES + XSEC_VECTOR_NAMES + DET_NAMES

# ---- SK run-period era split (phased MC / Pablo's datafit-SK release) ----
# Era groups: 0=SK-I, 1=SK-II, 2=SK-III, 3=SK-IV+V (sk_phase 4,5 share _sk45).
ERA_TAGS = ["sk1", "sk2", "sk3", "sk45"]
# Detector stems Pablo era-splits (each -> 4 era dials <stem>_<tag>). These 19
# are implemented as per-sample factors in detector_factors and reused verbatim
# per era; energy_scale (the 20th era-split stem in the release) rides the Rp/Rm
# energy-scale response and is added separately. neutron_tagging / fiducial_volume
# / decay_e_tagging are era-INDEPENDENT in the release.
DET_ERA_STEMS = ["fcpc_separation", "pc_reduction", "mge_nonubkg", "fc_reduction",
                 "subgev_2ring_pi0", "subgev_1ring_pi0",
                 "multiring_nunubar_separation", "multiring_emu_separation",
                 "multiring_eother_separation", "pc_stopthru_separation",
                 "pi0_ring_separation", "e_ring_separation", "mu_ring_separation",
                 "singlering_pid", "multiring_pid", "upmu_shower_separation",
                 "upmu_stop_bkg", "upmu_showering_bkg", "upmu_nonshowering_bkg"]

# sigma / nominal: flux+det from SK2023_Atm_datafit_xsec_barr_ntag.xml (validated
# baseline, era-split ntag); xsec block extended with the 3 CC_2p2h* dials from
# the new pfm config (CC_2p2h sigma 1.0, CC_2p2hNuBarNu 1.0, CC_2p2hMuE 0.1;
# nominal 1.0), inserted after DIS / before CCQE to match XSEC_VECTOR_NAMES.
SIGMA = np.array([0.25, 0.15, 0.10, 0.05, 0.02, 1.0,                       # flux (6)
                  0.2, 0.2, 0.1, 0.1, 0.05,                                # xsec NuTau,NCoverCC,NCHad,AxialMass,DIS
                  1.0, 1.0, 0.1,                                           #   +2p2h CC_2p2h,CC_2p2hNuBarNu,CC_2p2hMuE
                  0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 1.0,                  #   CCQE,QENuBarNu,QEMuE,1Pi_Pi0Pi,1Pi_NuBarNuE,1Pi_NuBarNuMu,1PiProd,CohPi
                  0.06, 0.01, 0.2, 0.003, 0.02, 0.06, 0.25, 0.06, 0.06,    # det (22)
                  0.06, 0.25, 0.02, 0.06, 0.03, 0.0035, 0.04, 0.12, 0.12,
                  0.04, 0.17, 0.17, 0.24])
NOMINAL = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0] + [1.0] * 16 + [1.0] * 22)
assert SIGMA.size == 44 and NOMINAL.size == 44 and len(NUISANCE_NAMES) == 44

MIN_ENTRIES = 5  # Experiment.py: FewEntries = ObservedBinned > MIN_ENTRIES

# ---- config-driven nuisance spec (zenith switch + future expansions) ----
CORE_FLUX_NAMES = ["normalization_below1GeV", "normalization_above1GeV",
                   "tilt", "nunubar_ratio", "flavor_ratio"]
ZENITH_DIALS = ["barr_zenith", "zenith_up", "zenith_down"]
# OPTIONAL flux dials — recognized as flux (so they ride the oscillated tensor at
# fit time and pick up the flux gradient loop) but NOT required, so existing specs
# are unaffected unless they list them explicitly.
#   solar_activity : Pablo's datafit-SK AtmoFlux.solar_activity, a multiplicative
#   true-E reweight w = 1 - x*A*exp(-E_true/L). Pure function of E_true => exact at
#   the 400x80 true-cell centres, NO response rebuild. nominal x=0 (no-op).
SOLAR_AMP = 0.08    # AtmoFlux.solar_activity amplitude
SOLAR_SCALE = 3.0   # GeV, AtmoFlux.solar_activity decay scale
#   kpi_ratio : SK's K/pi production ratio (thesis -1.08 sigma). A high-E flux norm that
#   grows logarithmically above the kaon onset: flux *= 1 + x*max(0, log10(E/KPI_E0)).
#   ~0 below ~3 GeV, rising to ~3% (FC/PC) -> ~10% (Up-mu) at high E. Flavor-blind (the
#   leading effect); the K/pi flavor / nu-nubar structure is left to the banded flux ratios.
KPI_E0 = 3.0        # GeV, kaon-onset pivot for the K/pi high-E flux ramp
#   flux_horizvert : SK's Horizontal/Vertical flux ratio (thesis 5509-18, 1-3%,
#   a high-E pion/mu decay-path effect) — a zenith-SHAPE flux dial the
#   zenith_up/down pair structurally cannot make: they pivot to zero at the
#   horizon (tanh^2) so they cannot reshape horizontal-vs-vertical. Energy-flat,
#   symmetric in cosz: flux *= 1 + x*g(cz) with g = (1-3cz^2)/2 (mean-zero over
#   cosz => a shape, not a norm; +0.5 horizontal, -1.0 vertical). nominal 0 (no-op).
OPTIONAL_FLUX_NAMES = ["solar_activity", "kpi_ratio", "flux_horizvert"]
ALL_FLUX_NAMES = CORE_FLUX_NAMES + ZENITH_DIALS + OPTIONAL_FLUX_NAMES

# Track S / Phase E6: CANONICAL_DIALS (the in-code dial-value table) has been
# DELETED. Dial (nominal, sigma) values are now sourced SOLELY from the package
# value XMLs (XML_DIAL_VALUES; see the "XML dial-value authority" block below).
# The name/spec constants above and below are still the vocabulary the specs and
# native modules build from; only the value table is gone. The former era-split
# detector-sigma table (DET_ERA_SIGMA) fed only CANONICAL_DIALS and is likewise
# retired — the era-split <stem>_<era> (nominal, sigma) values now live in the
# value XMLs (SK2023_Atm_datafit_r2_fude_ccqe_full.xml + the extra-dials XML).

# ---- energy-scale dials (bin-level reco-E migration) ------------------------
# Era-split energy_scale_<tag>. The SK public MC stores reco energy QUANTIZED to one value per
# reco-E bin, so an event-level re-digitization (a +-2% Rp/Rm bake) moves nothing.
# Instead we migrate at the HISTOGRAM level (SK's actual method): a scale x acts
# within each (sample, reco-cz) column along the reco-E index ie as a linear,
# rate-conserving transfer with delta = x - 1:
#   N'(ie) = N(ie) + delta*( N(ie-1)*[ie>0] - N(ie)*[ie<ne-1] )      (delta>0 -> up)
# Conserves sum_ie N exactly (top/bottom bins don't spill out of range). Geometry-
# free (needs only bin membership) => works with the quantized MC; no Rp/Rm, no
# response rebuild. nominal x=1 (no-op). NOTE: standalone-correct; matching Pablo's
# event-engine energy_scale convention is pending his fix (upstream is buggy).
ENERGY_SCALE_NAMES = [f"energy_scale_{tag}" for tag in ERA_TAGS]

# ---- OPTIONAL octant-systematics absorbers (OFF by default) -----------------
# Two SK-thesis-motivated additions, inspired by Wester (BU 2023) §5.2:
#   (1) sub-GeV (E_true<1 GeV) flavor/sign FLUX RATIOS with SK's opposite-sign
#       coupling — the dials SK uses to fit the sub-GeV e-like / nu-bar_e samples
#       down to data; our flat `flavor_ratio`/`nunubar_ratio` lack the sub-GeV
#       band + flavor-split + opposite-sign structure. Rate-conserving symmetric
#       (FEATURE convention): heavy leg *= 2r/(1+r), light leg *= 2/(1+r); r=1 no-op.
#         flux_nuebar_subgev : nu_e/nu-bar_e ratio, E<1 GeV.  r<1 lowers nu-bar_e,
#                              raises nu_e (the Table-5.2 nu-bar_e-over / nu_e-under
#                              signature). nu-bar_e leg=2r/(1+r), nu_e leg=2/(1+r).
#         flux_flavor_subgev : (nu_mu+nu-bar_mu)/(nu_e+nu-bar_e) ratio, E<1 GeV.
#                              r>1 raises mu-flavor, lowers e-flavor. mu leg
#                              =2r/(1+r), e leg=2/(1+r). Spares mu-like => octant-safe.
# These act on the oscillated tensor at fit time => NO response rebuild.
FLUX_RATIO_NAMES = ["flux_nuebar_subgev", "flux_flavor_subgev"]
# (nominal, sigma) values live in the value XMLs (E6). SK's published 1-sigma are
# tighter (nu_e/nu-bar_e ~0.03, flavor ~0.02 sub-GeV) — callers can override the
# sigmas to test absorption within SK's own uncertainty.
FLUX_RATIO_BOX = {"flux_nuebar_subgev": (0.3, 1.7),
                  "flux_flavor_subgev": (0.3, 1.7)}

#   (2) MOMENTUM-RESOLVED sub-GeV neutron-tagging migration. The production
#       `neutron_tagging_subgev` is a single whole-sample efficiency pull (one
#       migration ratio applied uniformly to every momentum bin), so it cannot
#       drain only the lowest-momentum nu-bar_e-tagged bins where the +36%
#       over-prediction sits (thesis Table 5.2). Split it into two independent
#       per-momentum-band efficiency dials (low / high), each a rate-conserving
#       migration WITHIN its band between donor {20,22}(0-neutron) and acceptor
#       {21,23}(1-neutron) sub-GeV samples. NTAG_PSPLIT = first high-band
#       momentum index (ie>=PSPLIT is "high"); ie<PSPLIT ("low") = the lowest
#       reco-momentum slice (ie=0 == logP<=2.4).
NTAG_SPLIT_NAMES = ["ntag_subgev_lowp", "ntag_subgev_highp"]
NTAG_PSPLIT = 2          # low band = ie in {0,1} (logP<=2.6); high = ie>=2
# (nominal, sigma) == neutron_tagging_subgev (1.0, 0.12); values in the value XMLs (E6).

# ---- EXTENDED set: SK systematics the base lacks, screened for octant relevance ----
# (a) ENERGY-BANDED flux ratios — SK's 3-band structure (E<1 / 1-10 / >10 GeV),
#     extending the two sub-GeV absorbers above. Same rate-conserving symmetric
#     form (heavy leg *2r/(1+r), light leg *2/(1+r)); r=1 no-op. Each dial is one
#     (band, heavy-leg, light-leg) triple; FLUX_RATIO_SPEC also covers the 2
#     sub-GeV absorbers so cell_weights/gradient iterate ONE generic registry.
FLUX_BAND_NAMES = ["flux_nuebar_mid", "flux_nuebar_high",
                   "flux_flavor_mid", "flux_flavor_high",
                   "flux_numubar_subgev", "flux_numubar_mid", "flux_numubar_high"]
FLUX_RATIO_SPEC = {
    "flux_nuebar_subgev":  ("sub", "nuebar", "nue"),
    "flux_nuebar_mid":     ("mid", "nuebar", "nue"),
    "flux_nuebar_high":    ("high", "nuebar", "nue"),
    "flux_flavor_subgev":  ("sub", "mu", "e"),
    "flux_flavor_mid":     ("mid", "mu", "e"),
    "flux_flavor_high":    ("high", "mu", "e"),
    "flux_numubar_subgev": ("sub", "numubar", "numu"),
    "flux_numubar_mid":    ("mid", "numubar", "numu"),
    "flux_numubar_high":   ("high", "numubar", "numu"),
}
# FLUX_BAND (nominal, sigma) = (1.0, 0.10) each; values in the value XMLs (E6).
ALL_FLUX_RATIO_NAMES = FLUX_RATIO_NAMES + FLUX_BAND_NAMES   # superset for detection

# (b) sub-GeV XSEC dials we lack (SK's largest sub-GeV levers, thesis Table B.1):
#   xsec_ccqe_shape      — CCQE energy-shape: CCQE *= (E_true)^(r-1). APPROXIMATION
#       of SK's "CCQE Shape" (+2.05sig, RFG-vs-LFG normalized-sigma diff vs E); we
#       lack the RFG/LFG tables, so this is a 1-param power-law E-tilt of CCQE.
#   xsec_ccqe_subgev_nue — sub-GeV (E<1) CCQE nu_e+nu-bar_e norm (SK 5% nu_e); ALSO
#       the surrogate for our MISSING 2p2h (zero in OLD MC -> CCQE inflated).
#   xsec_1p1h_subgev_nue / xsec_2p2h_subgev_nue — the THESIS-FAITHFUL SPLIT of the
#       above. On the new 2p2h MC the CCQE class is pure 1p1h and 2p2h
#       is its own class, so the sub-GeV nu_e norm separates into a 1p1h piece
#       (sigma 0.05, ~SK's sub-GeV nu_e CCQE 5%) and a 2p2h piece (sigma 0.20;
#       box wide since 2p2h is poorly known). Same mask form as
#       xsec_ccqe_subgev_nue but on the 1p1h(=CCQE) and CC_2p2h class bits resp.
#       Diagnostic: with real 2p2h present, does the octant pull load onto 1p1h or 2p2h?
#   xsec_ccqe_shape_subgev — sub-GeV-LOCALIZED CCQE shape: the faithful-ish version of
#       SK's "CCQE Shape" (+2.05 sigma, SK's #1 lever / the octant driver). A log-E tilt
#       confined to E_true < 1.33 GeV (SK sub-GeV/multi-GeV boundary), ~mean-zero over the
#       sub-GeV range (a SHAPE, not a norm) and identically 0 above 1.33 GeV -- so it does
#       NOT reshape multi-GeV like the global power-law xsec_ccqe_shape. nominal x=0 (no-op):
#       CCQE *= 1 + x * 1[E<1.33] * (ln E - <ln E>_subgev) -- a mean-zero (~rate-neutral,
#       pivot-free) log-E tilt over the sub-GeV cells. Not the exact RFG/LFG (tables absent
#       from the public release) but the octant-relevant sub-GeV part; documented approximation.
CCQE_SHAPE_SUBGEV_E = 1.33   # GeV, SK sub-GeV/multi-GeV boundary (E_true proxy for EVis<1330 MeV)
XSEC_EXTRA_NAMES = ["xsec_ccqe_shape", "xsec_ccqe_subgev_nue",
                    "xsec_1p1h_subgev_nue", "xsec_2p2h_subgev_nue",
                    "xsec_ccqe_shape_subgev"]
# XSEC_EXTRA (nominal, sigma) live in the value XMLs (E6): xsec_ccqe_shape (1.0,
# 0.20), xsec_ccqe_subgev_nue (1.0, 0.05), xsec_1p1h_subgev_nue (1.0, 0.05),
# xsec_2p2h_subgev_nue (1.0, 0.20), xsec_ccqe_shape_subgev (0.0, 0.40 shape coeff).
XSEC_EXTRA_BOX = {"xsec_ccqe_shape": (0.3, 1.7), "xsec_ccqe_subgev_nue": (0.5, 1.5),
                  "xsec_1p1h_subgev_nue": (0.5, 1.5), "xsec_2p2h_subgev_nue": (0.0, 3.0),
                  "xsec_ccqe_shape_subgev": (-2.0, 2.0)}
# sub-GeV nu_e norm dials handled by the generic XX_ke / gradient loops (name -> class-mask attr)
SUBGEV_NUE_NORM = {"xsec_ccqe_subgev_nue": "ccqe_nue_cls",
                   "xsec_1p1h_subgev_nue": "ccqe_nue_cls",     # CCQE == 1p1h on the 2p2h MC
                   "xsec_2p2h_subgev_nue": "twop2h_nue_cls"}

# ---- OPTIONAL multi-GeV CCQE flavor-norm dials (OFF by default) ---------------
# SK's "CCQE Norm., Multi-GeV" systematic, applied SEPARATELY to nu_e and nu_mu,
# each ~25% (Wester thesis 5569-5571: "the CCQE normalization ... multi-GeV ...
# nu_e and nu_mu ... 25%"; Table B.1 "Norm., Multi-GeV" 9247). Multi-GeV e-like
# is a primary dCP appearance sample, so a nu_e-localized multi-GeV CCQE norm is
# the largest dCP-relevant absorber the R2 120-set lacks -- only the global 10%
# CCQE norm reaches the multi-GeV region. Mirrors
# the sub-GeV xsec_ccqe_subgev_nue machinery (SUBGEV_NUE_NORM) EXACTLY but on the
# COMPLEMENTARY energy mask E_true >= CCQE_SHAPE_SUBGEV_E (1.33 GeV, the SK sub-/
# multi-GeV boundary the sub-GeV CCQE shape is confined below) and split by flavor:
#   _nue  = CCQE & (cls_flavor==0)  -> nu_e + nu-bar_e
#   _numu = CCQE & (cls_flavor==1)  -> nu_mu + nu-bar_mu
# Multiplicative norm r, nominal 1 = EXACT no-op (fac = 1 + 1[E>=1.33]*(r-1)),
# sigma 0.25. NOT in XSEC_EXTRA_NAMES (that list is spread into octsyst_xsec/_max
# specs) -> own name-list + active flag so pre-existing specs resolve unchanged.
MULTIGEV_CCQE_NAMES = ["xsec_ccqe_multigev_nue", "xsec_ccqe_multigev_numu"]
MULTIGEV_CCQE_NORM = {"xsec_ccqe_multigev_nue": "ccqe_nue_cls",
                      "xsec_ccqe_multigev_numu": "ccqe_numu_cls"}
MULTIGEV_CCQE_BOX = {"xsec_ccqe_multigev_nue": (0.0, 3.0),
                     "xsec_ccqe_multigev_numu": (0.0, 3.0)}   # norm floor 0, +-8 sigma
# MULTIGEV_CCQE (nominal, sigma) = (1.0, 0.25) each; values in the value XMLs (E6).

# ---- OPTIONAL relative-normalization dial (per-sample-group norm) ------------
# SK's "Relative Normalization" between sample groups (thesis Rel.Norm FC-MultiGeV
# -1.33 sigma): a flat multiplicative norm on the FC multi-GeV sample group, applied
# at the binned (per-sample) level inside detector_factors so the existing detector
# gradient machinery (dlnD) handles it. nominal 1 (no-op), sigma 0.05 (SK ~5%).
# FC multi-GeV samples (SuperK_Atm_Pheno sample_names): SK1-3 FC mG 1R {7,8,9} +
# MR {10,11,12,13}, SK4-5 FC mG 1R {24,25,26,27,28}.
REL_NORM_FCMG_SAMPLES = frozenset({7, 8, 9, 10, 11, 12, 13, 24, 25, 26, 27, 28})
REL_NORM_NAMES = ["rel_norm_fcmg"]
# rel_norm_fcmg (nominal, sigma) = (1.0, 0.05); value in the value XMLs (E6).

# ---- OPTIONAL neutron-production 0n/1n migration dials (OFF by default) ------------
# A per-PAIR neutron-tag migration meant to absorb a neutron-PRODUCTION cross-section
# systematic: the true (nu-capture) neutron multiplicity of an interaction shifts events
# between the 0-neutron and 1-neutron reconstructed samples. Unlike the production
# neutron_tagging_subgev/_multigev dials (which lump the numu(bar) single-ring samples
# {22,23}/{27,28} onto the 0n/1n legs by ANALOGY -- those samples carry NO neutron label
# in the release binning), these dials act ONLY on the EXPLICITLY 0n/1n-split pairs:
#   sk4-5_fc_subgev_1ring_nuebarlike_0neutron  (20) -> _1neutron (21)
#   sk4-5_fc_multigev_1ring_nuebarlike_0neutron (25) -> _1neutron (26)
# (SuperK_Atm_Pheno.sample_names: the ONLY samples with 0neutron/1neutron in the name.)
# So this is NOT the same enumeration as neutron_tagging_*: it is nuebar-ONLY and
# per-pair-INDEPENDENT (one dial per pair), whereas ntag is one shared dial per (sub/
# multi)-GeV band spanning nuebar+numu(bar). Same rate-conserving migration algebra as
# apply2 (donor x, acceptor 1+r(1-x), r = rate(0n)/rate(1n) from the weighted rates),
# so total 0n+1n rate per pair is invariant under x -- the systematic reshuffles the
# neutron split, it does not change the sample-pair total. nominal x=1 (exact no-op).
#   TRIAL dials (nmig_*): Gaussian sigma 0.20, default +-10sigma bounds (unpinned).
#     The trial prior is deliberately LOOSE (0.20) to give the
#     maximal-effect / diagnostic estimate of how much depth this migration can absorb.
#   PINNED dials (nmig_*_pinned): sigma 0.10 prior BUT a hard BOX so the fit cannot
#     migrate more events than physically exist. A dial value x drains the 0n leg to
#     x*rate(0n) and adds r(1-x)*rate(1n) = (1-x)*rate(0n) to the 1n leg; the 1n leg stays
#     >=0 for ALL x>=0 and the 0n leg stays >=0 for x>=0, so the physical box is
#     x in [0, 1 + rate(1n)/rate(0n)] = [0, 1 + 1/r] (the upper edge is where the 1n leg
#     is fully drained into 0n: acc-factor 1+r(1-x)=0). BOX = (0.0, 1.0 + 1.0/r_pair)
#     with r_pair from the raw MC event counts (below). Trial and pinned share the SAME
#     migration algebra, donor/acceptor pair, and prior width -- they differ ONLY in the box.
NEUTRON_MIG_PAIRS = {                 # dial name -> (donor 0n sample, acceptor 1n sample)
    "nmig_subgev_nuebar":          (20, 21),
    "nmig_multigev_nuebar":        (25, 26),
    "nmig_subgev_nuebar_pinned":   (20, 21),
    "nmig_multigev_nuebar_pinned": (25, 26),
}
NEUTRON_MIG_NAMES = list(NEUTRON_MIG_PAIRS)
NEUTRON_MIG_TRIAL_NAMES = ["nmig_subgev_nuebar", "nmig_multigev_nuebar"]
NEUTRON_MIG_PINNED_NAMES = ["nmig_subgev_nuebar_pinned", "nmig_multigev_nuebar_pinned"]
# raw MC event counts of the 0n/1n samples in the release (sk_response.npz
# sample_event_counts) -> the physical rate ratio r = counts(0n)/counts(1n) that bounds
# the pinned box. These are FIXED (raw-count) bounds; the runtime migration r used in the
# algebra is the per-oscillation-point WEIGHTED rate (matches every other migration dial).
NEUTRON_MIG_RAWCOUNTS = {20: 294634, 21: 112964, 25: 43858, 26: 38369}
NEUTRON_MIG_BOX_PINNED = {                    # x in [0, 1 + 1/r] (max physical migration)
    n: (0.0, 1.0 + NEUTRON_MIG_RAWCOUNTS[a] / NEUTRON_MIG_RAWCOUNTS[d])
    for n, (d, a) in NEUTRON_MIG_PAIRS.items() if n in NEUTRON_MIG_PINNED_NAMES
}
# nmig (nominal, sigma): trial (1.0, 0.20) / pinned (1.0, 0.10); values in the value XMLs (E6).

# ---- OPTIONAL decay-e tagging dial (OFF by default) --------------------------
# SK's "Decay-e Tagging" detector systematic (Wester thesis 6022-6024, 6061-6062):
# "The decay electron efficiency uncertainty PROPORTIONALLY CHANGES THE NORMALIZATION
# of events in the FC sub-GeV and multi-GeV samples which utilize the number of decay
# electrons as part of their sample selection." So the release mechanism is a NORM
# (efficiency) pull, NOT a migration -- a flat multiplicative factor on the decay-e-keyed
# samples. In the SuperK_2023 release binning the ONLY decay-e-keyed samples are the SK
# I-III sub-GeV single-ring ones (SuperK_Atm_Pheno.sample_names ..._Ndecaye):
#   0 e-like 0decaye, 1 e-like 1decaye, 3 mu-like 0decaye, 4 mu-like 1decaye, 5 mu-like 2decaye
# (verified era-present only in sk1/sk2/sk3, never sk45), so the dial is naturally
# ERA-INDEPENDENT here (the decay-e samples don't exist in SK IV+V). sigma = SK I-III
# published decay-e efficiency uncertainty = 1.5% (thesis 6061; the SK IV-V 0.8% value is
# moot -- no decay-e samples in sk45). Implemented as a whole-sample multiplicative norm
# on those 5 bins folded into D (like rel_norm_fcmg), nominal x=1 = exact no-op,
# d ln D/dx = 1/x. (The pfm.xml carries era-split decay_e_tagging_sk* blocks at sigma 0.13,
# but those are inactive placeholders -- the header states DecayE is unset in SuperK_2023;
# the physical thesis width 0.015 is adopted here.)
DECAY_E_SAMPLES = frozenset({0, 1, 3, 4, 5})   # SK I-III sub-GeV decay-e-keyed samples
DECAY_E_NAME = "decay_e_tagging"
# decay_e_tagging (nominal, sigma) = (1.0, 0.015); value in the value XMLs (E6).

# ---- OPTIONAL up-mu background zenith x momentum SHAPE dials -----------------
# SK's cosmic-mu background subtraction (Wester thesis Sec 5.2, lines 5975-5984)
# acts ONLY on the near-horizon zenith bins: the TWO horizon-nearest reco-cosZ bins
# of the STOPPING sample (16), subtracted AS A FUNCTION OF MOMENTUM, and the SINGLE
# horizon-nearest bin of the THROUGH-GOING NON-showering sample (17). Showering
# through-going (18) is neutrino-induced (minimum-ionizing cosmic mu do not shower)
# so it carries no cosmic-mu background and is excluded. Our engine has the
# whole-sample per-era up-mu bkg NORMS (upmu_*_bkg_<era>, indices 87-98) but no
# zenith x momentum SHAPE; these dials add it as a multiplicative factor on the
# affected model bins (like rel_norm_fcmg -- the engine has no separate bkg
# component), nominal x=1 (no-op), d ln D/dx = 1/x. Era-COMMON (the near-horizon
# cosmic-mu contamination is a fixed geometric feature; era livetime/efficiency is
# already the era-split whole-sample norms these ride on).
# Up-mu reco-cosZ = z10bins_up ([-1..0], 10 bins) so iz=9 (cz in [-0.1,0]) is the
# horizon-nearest bin, iz=8 the second; stopping momentum = upmus_ebins (3 bins):
# low = ie 0 (1.585-2.495 GeV), high = ie {1,2} (>2.495 GeV). sigma = SK's bkg-rate
# uncertainty (~10-20 events, midpoint 15) / affected-bin SK-data content, so every
# dial carries the SAME ~15-event 1sigma background uncertainty.
UPMU_BKG_SHAPE_NAMES = ["upmu_stop_bkg_horiz_lowp", "upmu_stop_bkg_horiz_highp",
                        "upmu_nonshow_bkg_horiz"]
# name -> (sample id, tuple of reco-cosZ bin indices, reco-momentum bin indices or
# None=all momentum bins of the sample)
UPMU_BKG_SHAPE_SPEC = {
    "upmu_stop_bkg_horiz_lowp":  (16, (8, 9), (0,)),
    "upmu_stop_bkg_horiz_highp": (16, (8, 9), (1, 2)),
    "upmu_nonshow_bkg_horiz":    (17, (9,), None),
}
UPMU_BKG_SHAPE_SIGMA = {"upmu_stop_bkg_horiz_lowp": 0.090,   # 15 / 165.8 events
                        "upmu_stop_bkg_horiz_highp": 0.036,  # 15 / 422.0 events
                        "upmu_nonshow_bkg_horiz": 0.013}     # 15 / 1113.9 events
# upmu-bkg-shape (nominal 1.0, sigma UPMU_BKG_SHAPE_SIGMA); values in the value XMLs (E6).

# ---- OPTIONAL up/down energy-scale detector dial set (OFF by default) --------
# SK's "Up/Down Energy Scale" systematic (Wester thesis 6011-6018): decay-e from
# stopped cosmic mu, split by Cherenkov-ring direction, measures the detector's
# energy-scale UNIFORMITY between up-looking and down-looking directions. SK models
# it as a NORMALIZATION variation of upward-going vs downward-going FC and PC events
# "by the largest observed deviation" (Table 5.6 Up/Down Conv. FV: SK I 0.6%,
# II 1.1%, III 0.6%, IV 0.5%, V 0.7%) -- NOT a momentum-bin migration (that is the
# separate absolute energy_scale). Implemented thesis-literally as an ANTI-SYMMETRIC
# per-era multiplicative factor: up-going bins *=(1+d), down-going *=(1-d), so the
# fit can move the up/down ratio in EITHER direction; d nominal 0 = exact no-op.
# Rides the detector-factor + dlnD path (like rel_norm_fcmg / upmu_bkg), but the
# sign DIFFERS between up and down bins so the gradient carries a SIGNED mask.
# Era-SPLIT (4 dials, one per ERA_TAGS -- the non-uniformity is measured per SK
# phase; sk45=SK-IV per the absolute energy_scale sk45 lumping rule). FC+PC only;
# up-mu samples {16,17,18} (all up-going => degenerate with their existing upmu
# norms) and single-reco-zenith FC samples {1,2,5,6} (z1bins straddle cz=0) are
# EXCLUDED (factor 1). Reco-cosZ split at cz=0: z10bins edge index 5 == 0.0, so
# iz < nz//2 is up-going, iz >= nz//2 down-going (all FC/PC samples use nz=10).
UPDOWN_ESCALE_NAMES = [f"updown_escale_{tag}" for tag in ERA_TAGS]
UPDOWN_ESCALE_SIGMA = {"updown_escale_sk1": 0.006, "updown_escale_sk2": 0.011,
                       "updown_escale_sk3": 0.006, "updown_escale_sk45": 0.005}
UPDOWN_ESCALE_EXCLUDE = frozenset({16, 17, 18})   # up-mu samples (all up-going)
# updown-escale (nominal 0.0, sigma UPDOWN_ESCALE_SIGMA); values in the value XMLs (E6).

# ---- OPTIONAL direction-smearing systematic (OFF by default) -----------------
# DIAGNOSTIC beyond SK's published systematics vocabulary: no direction-smearing
# systematic exists in SK's treatment (central-MC direction physics only, which
# the binned response inherits). Norm-type dials can rescale bins but cannot
# redistribute counts BETWEEN zenith bins, so none of them can express a
# direction-RESOLUTION mismatch. This dial adds that one freedom: a zenith-bin
# migration built from a per-sample reco-cz confusion matrix (derived offline by
# re-smearing the per-event MC directions by a few-percent angular resolution
# variation). The dial is FLAVOR-BLIND (it migrates e-like and mu-like bins by
# the SAME reco-cz confusion matrix), so it cannot impose anti-correlated e/mu
# structure; but unlike the norm dials it CAN reshape the zenith DISTRIBUTION
# (redistribute counts between zenith bins), so a measured null is informative.
#
# Single global dial s (nominal 0 = exact no-op), prior sigma 10 (effectively
# unconstrained -- the question is whether the DATA pulls s), custom box [0,1] (s<0 =
# de-smearing is unphysical and can drive negative expectations; s in [0,1] keeps the
# migration operator A=(1-s)I + s*M a convex mix of two non-negative matrices, so
# E'=A@E >= 0). The migration matrices M[sample] (nz x nz, reco-cz confusion, built
# offline from the per-event MC) are chosen at engine construction via the dirsmear_matrix
# ctor arg (default OFF -- no matrix loaded unless the dial is active). Action per
# sample x reco-momentum row: E(s) = E + s*(M - I) @ E; linear in s so dE/ds=(M-I)@E is
# exact and s-independent (mirrors the absolute energy-scale migration precedent).
DIR_SMEAR_NAME = "dir_smear"
DIR_SMEAR_SIGMA = 10.0             # diagnostic: effectively unconstrained prior
DIR_SMEAR_BOX = (0.0, 1.0)         # one-sided s in [0,1] (positivity of A=(1-s)I+sM)
# dir_smear (nominal 0.0, sigma DIR_SMEAR_SIGMA); value in the value XMLs (E6).


# ---- XML dial-value authority (Track S, Phase E2 + E6) ----------------------
# Dial (nominal, sigma) values are read from XML at load — the value XMLs are now
# the SOLE authority (CANONICAL_DIALS was deleted at E6). Two value XMLs are
# merged, together covering every dial across all 26 named specs:
#   1. DIAL_VALUE_XML       — the production 131-dial value XML (kept pristine:
#                             exactly the 131 in the R2FUDECCQE seed order).
#   2. DIAL_VALUE_XML_EXTRA — a supplementary value XML for the 30 non-production
#                             dials some specs resolve (base-name detector/flux
#                             stems used by the non-era specs + the optional
#                             diagnostic dials: barr_zenith, decay_e_tagging,
#                             dir_smear, nmig_*, ntag_subgev_low/high p,
#                             xsec_{1,2}p2h/ccqe_subgev_nue).
import os as _os

# The value XMLs are package data beside the ENGINE (pynu/Experiments/ since
# Track T phase T3) — _ENGINE_DIR anchors there, NOT at dirname(__file__).
_ENGINE_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "Experiments"))
_REPO_ROOT = _os.path.abspath(_os.path.join(_ENGINE_DIR, "..", ".."))

# ---- value XMLs ship as PACKAGE DATA (Track S, Phase E6 / review N-2) --------
# The two value XMLs live beside the engine (pynu/Experiments/) so a
# non-editable wheel install carries them — CANONICAL_DIALS is gone (E6) and is
# no longer a fallback authority, so the package copy is the SOLE source of dial
# values. The `analysis/AnalysisFiles/` copies are the analysis-facing mirror
# used by the event pipeline + XML manifests; when BOTH are present the loader
# HARD-FAILS on any byte-level value divergence (same no-silent-second-authority
# philosophy as the former shadow checks).
_DIAL_VALUE_XML_NAME = "SK2023_Atm_datafit_r2_fude_ccqe_full.xml"
_DIAL_VALUE_XML_EXTRA_NAME = "SK2023_Atm_datafit_binned_extra_dials.xml"

DIAL_VALUE_XML = _os.path.join(_ENGINE_DIR, _DIAL_VALUE_XML_NAME)
DIAL_VALUE_XML_EXTRA = _os.path.join(_ENGINE_DIR, _DIAL_VALUE_XML_EXTRA_NAME)

# analysis-tree mirrors (must byte-match the package copies when present)
_DIAL_VALUE_XML_MIRROR = _os.path.join(
    _REPO_ROOT, "analysis", "AnalysisFiles", _DIAL_VALUE_XML_NAME)
_DIAL_VALUE_XML_EXTRA_MIRROR = _os.path.join(
    _REPO_ROOT, "analysis", "AnalysisFiles", _DIAL_VALUE_XML_EXTRA_NAME)


def _assert_mirror_agrees(pkg_path, mirror_path):
    """When the analysis-tree mirror exists, it MUST byte-match the package copy;
    a divergence means the two authorities have drifted — hard-fail (no silent
    second authority)."""
    if not _os.path.exists(mirror_path):
        return
    with open(pkg_path, "rb") as f:
        pkg = f.read()
    with open(mirror_path, "rb") as f:
        mir = f.read()
    if pkg != mir:
        raise ValueError(
            f"value-XML mirror {mirror_path} diverged from the package copy "
            f"{pkg_path} (byte-level). These must be byte-identical — reconcile "
            "them (the package copy is the authority the engine loads).")


def _load_xml_dial_values(path):
    """Parse {name: (nominal, sigma)} from a Pynu nuisance XML (all <nuisance>
    blocks, document order irrelevant here — this is a value lookup, not an order
    source). The value XMLs are package data (review N-2); an absent package copy
    is a hard error — there is NO fallback authority any more (CANONICAL_DIALS
    was deleted at E6)."""
    import xml.etree.ElementTree as _ET
    if not _os.path.exists(path):
        raise FileNotFoundError(
            f"value XML {path} is missing from the package. The two SK binned "
            "value XMLs ship as package data under pynu/Experiments/; without them "
            "the engine has no dial values (CANONICAL_DIALS was removed at E6). "
            "Reinstall the package or restore the file.")
    root = _ET.parse(path).getroot()
    vals = {}
    for nu in root.iter("nuisance"):
        name = nu.attrib["name"]
        vals[name] = (float(nu.find("nominal").text),
                      float(nu.find("sigma").text))
    return vals


def _merge_dial_value_xmls():
    """Merge the production + supplementary value XMLs. The supplementary file
    MUST NOT redeclare any production-131 dial (that would create a second
    authority for a production value); a name overlap is a hard error."""
    _assert_mirror_agrees(DIAL_VALUE_XML, _DIAL_VALUE_XML_MIRROR)
    _assert_mirror_agrees(DIAL_VALUE_XML_EXTRA, _DIAL_VALUE_XML_EXTRA_MIRROR)
    prod = _load_xml_dial_values(DIAL_VALUE_XML)
    extra = _load_xml_dial_values(DIAL_VALUE_XML_EXTRA)
    overlap = set(prod) & set(extra)
    if overlap:
        raise ValueError(
            f"supplementary value XML {DIAL_VALUE_XML_EXTRA} redeclares "
            f"production dials {sorted(overlap)} — it must cover ONLY the "
            "non-production dials (the full-131 XML is the sole authority for "
            "the production set)")
    merged = dict(prod)
    merged.update(extra)
    return merged


XML_DIAL_VALUES = _merge_dial_value_xmls()


def _xml_document_order(path):
    """Ordered list of <nuisance> names in document order (the semantic θ order)."""
    import xml.etree.ElementTree as _ET
    if not _os.path.exists(path):
        return []
    return [nu.attrib["name"]
            for nu in _ET.parse(path).getroot().iter("nuisance")]


def _assert_production_theta_order():
    """Loud θ-order assert (SCOPE §5 risk 2 / design §2.1): every production seed
    npz encodes the R2FUDECCQE dial order, so the production value XML MUST load
    in exactly that document order. Checked at import against the R2FUDECCQE
    activation manifest; a divergence is a hard error (a silently reordered value
    XML would corrupt every seeded fit). No-op if either file is absent."""
    prod_order = _xml_document_order(DIAL_VALUE_XML)
    if not prod_order:
        return
    # The R2FUDECCQE activation manifest moved to analysis/AnalysisFiles/ at E6
    # (review N-5); read it from _MANIFEST_DIR so the assert follows the manifests.
    manifest = _os.path.join(_MANIFEST_DIR,
                             "SK2023_Atm_datafit_r2_fude_ccqe.xml")
    if not _os.path.exists(manifest):
        return
    seed_order = _parse_xml_active(manifest)
    if prod_order != seed_order:
        i = next((k for k, (a, b) in enumerate(zip(prod_order, seed_order))
                  if a != b), min(len(prod_order), len(seed_order)))
        raise ValueError(
            "production value-XML θ-order diverged from the R2FUDECCQE seed "
            f"manifest at index {i}: value-XML="
            f"{prod_order[i] if i < len(prod_order) else '<end>'} vs seed="
            f"{seed_order[i] if i < len(seed_order) else '<end>'} "
            f"({DIAL_VALUE_XML} vs {manifest}). Production seed npzs encode the "
            "R2FUDECCQE order — a reordered value XML corrupts every seeded fit.")


def _dial_value(name):
    """(nominal, sigma) for a dial. The value XMLs (XML_DIAL_VALUES) are the sole
    authority (Track S / E6 — CANONICAL_DIALS was deleted). A dial with no value
    XML entry is a hard error."""
    xv = XML_DIAL_VALUES.get(name)
    if xv is not None:
        return xv
    raise KeyError(f"no value for dial {name!r} in the value XMLs "
                   f"({DIAL_VALUE_XML} / {DIAL_VALUE_XML_EXTRA})")


def _parse_xml_active(path):
    """Active (status==1) nuisance names from a Pynu XML, in document order."""
    import re
    txt = open(path).read()
    names = []
    for m in re.finditer(r"<nuisance\s+name='([^']+)'>(.*?)</nuisance>",
                         txt, re.S):
        st = re.search(r"<status>\s*(\d+)\s*</status>", m.group(2))
        if st and int(st.group(1)) == 1:
            names.append(m.group(1))
    return names


# ---- Phase E3: XML activation-manifest resolver ------------------------------
# Every named spec is materialized as an activation manifest (an ordered list of
# <nuisance name='..'><status>1</status></nuisance> blocks) in
# analysis/AnalysisFiles/. The resolver reads the manifest for a named spec; the
# legacy string dispatch was deleted at E6, so the manifests are now the sole
# named-spec authority. `.xml` paths and explicit name lists bypass the registry
# (they are already declarative). Aliases map to the same manifest.
#
# <replaces> semantics: the R3/R2NTS manifests already carry the ntag splice
# (neutron_tagging_subgev -> ntag_subgev_lowp/_highp at the original position)
# baked into their document order; the manifest header records the provenance.
_MANIFEST_DIR = _os.path.join(_ENGINE_DIR, "..", "..",
                              "analysis", "AnalysisFiles")
_MANIFEST_DIR = _os.path.abspath(_MANIFEST_DIR)
SPEC_MANIFESTS = {
    "barr": "SK2023_Atm_datafit_barr.xml",
    "updown": "SK2023_Atm_datafit_updown.xml",
    "both": "SK2023_Atm_datafit_both.xml",
    "phased": "SK2023_Atm_datafit_phased.xml",
    "phased_prod": "SK2023_Atm_datafit_phased_prod.xml",
    "phased_full": "SK2023_Atm_datafit_phased_full.xml",
    "R1": "SK2023_Atm_datafit_R1.xml",
    "pfm_base": "SK2023_Atm_datafit_R1.xml",
    "R2": "SK2023_Atm_datafit_R2.xml",
    "ladder_r2": "SK2023_Atm_datafit_R2.xml",
    "R3": "SK2023_Atm_datafit_R3.xml",
    "ladder_r3": "SK2023_Atm_datafit_R3.xml",
    "R2NTS": "SK2023_Atm_datafit_R2NTS.xml",
    "ladder_r2_ntagsplit": "SK2023_Atm_datafit_R2NTS.xml",
    "R2HV": "SK2023_Atm_datafit_R2HV.xml",
    "ladder_r2_horizvert": "SK2023_Atm_datafit_R2HV.xml",
    "R2UBS": "SK2023_Atm_datafit_R2UBS.xml",
    "ladder_r2_upmu_bkg_shape": "SK2023_Atm_datafit_R2UBS.xml",
    "R2UDE": "SK2023_Atm_datafit_R2UDE.xml",
    "ladder_r2_updown_escale": "SK2023_Atm_datafit_R2UDE.xml",
    "R2DS": "SK2023_Atm_datafit_R2DS.xml",
    "ladder_r2_dirsmear": "SK2023_Atm_datafit_R2DS.xml",
    "R2FUDECCQE": "SK2023_Atm_datafit_r2_fude_ccqe.xml",
    "r2_fude_ccqe": "SK2023_Atm_datafit_r2_fude_ccqe.xml",
    "ladder_r2_fude_ccqe": "SK2023_Atm_datafit_r2_fude_ccqe.xml",
    "R2FUDECCQE_NMIG": "SK2023_Atm_datafit_r2_fude_ccqe_nmig.xml",
    "r2_fude_ccqe_nmig": "SK2023_Atm_datafit_r2_fude_ccqe_nmig.xml",
    "R2FUDECCQE_NMIG_PINNED": "SK2023_Atm_datafit_r2_fude_ccqe_nmig_pinned.xml",
    "r2_fude_ccqe_nmig_pinned": "SK2023_Atm_datafit_r2_fude_ccqe_nmig_pinned.xml",
    "R2FUDECCQE_DCYE": "SK2023_Atm_datafit_r2_fude_ccqe_dcye.xml",
    "r2_fude_ccqe_dcye": "SK2023_Atm_datafit_r2_fude_ccqe_dcye.xml",
    "R2FUDECCQE_NMIG_DCYE": "SK2023_Atm_datafit_r2_fude_ccqe_nmig_dcye.xml",
    "r2_fude_ccqe_nmig_dcye": "SK2023_Atm_datafit_r2_fude_ccqe_nmig_dcye.xml",
    "octsyst_base": "SK2023_Atm_datafit_octsyst_base.xml",
    "octsyst_flux": "SK2023_Atm_datafit_octsyst_flux.xml",
    "octsyst_ntag": "SK2023_Atm_datafit_octsyst_ntag.xml",
    "octsyst_both": "SK2023_Atm_datafit_octsyst_both.xml",
    "octsyst_fluxband": "SK2023_Atm_datafit_octsyst_fluxband.xml",
    "octsyst_xsec": "SK2023_Atm_datafit_octsyst_xsec.xml",
    "octsyst_max": "SK2023_Atm_datafit_octsyst_max.xml",
}

# Response-compatibility contract (Phase E3): specs whose mechanisms require a
# particular engine configuration. The ntag momentum-split (R3/R2NTS) needs
# migration_mode='weighted' (per-bin band rates); SKBinnedEngine.__init__ already
# raises today's message when it isn't — this registry lets a caller pre-check.
SPEC_REQUIRES_WEIGHTED = frozenset({
    "R3", "ladder_r3", "R2NTS", "ladder_r2_ntagsplit",
    "octsyst_ntag", "octsyst_both", "octsyst_fluxband", "octsyst_xsec",
    "octsyst_max",
})


def resolve_nuisance_spec(spec):
    """Return (names, nominal, sigma) for a nuisance-set selector.

    Named specs resolve SOLELY through the XML activation-manifest registry
    (SPEC_MANIFESTS) — the legacy string dispatch was deleted at E6. `.xml`
    paths and explicit name lists resolve directly. None/'' == the default
    production 'barr' spec. Values come from the value XMLs (XML_DIAL_VALUES;
    the sole authority after E6).
    """
    if spec is None or spec == "":
        spec = "barr"
    manifest = SPEC_MANIFESTS.get(spec) if isinstance(spec, str) else None
    if manifest is not None:
        names = _parse_xml_active(_os.path.join(_MANIFEST_DIR, manifest))
    elif isinstance(spec, str) and spec.endswith(".xml"):
        names = _parse_xml_active(spec)
    elif isinstance(spec, (list, tuple)):
        names = list(spec)
    else:
        raise ValueError(f"unknown nuisance_spec {spec!r}")

    missing = [n for n in names if n not in XML_DIAL_VALUES]
    if missing:
        raise ValueError(f"nuisance_spec has unsupported dials {missing} "
                         "(only the 41 baseline dials + zenith_up/zenith_down "
                         "are available without a response rebuild)")
    # values come from the value XMLs (the sole authority after E6).
    vals = [_dial_value(n) for n in names]
    nominal = np.array([v[0] for v in vals])
    sigma = np.array([v[1] for v in vals])
    return names, nominal, sigma


def _unphys(x):
    """PhysicsTunes._unphysical_value with default bounds."""
    return x < 0 or x > 9999999


# Loud θ-order assert at import (formerly fired at sk_binned_engine import;
# firing here is equivalent-or-stronger — every vocabulary consumer hits it).
_assert_production_theta_order()
