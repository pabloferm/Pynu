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
  scripts/run_sk_datafit_point_worker.py      (minimizer protocol, dCP profiling)

Engine model per (physics point, dCP slice):
  W[k,cE,cZ]  = Phi_or_1 * F_flux(cell, pdg_k) * A_axial(cell, CC_k) * X_k(mask bits)
  n_pre[b]    = sum_nz R_v * W[R_k,R_e,R_z] scattered to R_b
  n_nu[b]     = n_pre[b] * D_b(theta_det; r from physics-only rates)
  var[b]      = D_b^2 * sum_nz S2_v * W^2[...] scattered to S2_b
  chi2        = BB(obs[m], n_nu[m], var[m]) + sum_j (x_j - mu_j)^2 / sigma_j^2
with m = FewEntries mask (observed > 5, strict), NC classes get Phi = 1
(SuperK_2023.UpdatePhysicsWeights override).

No muon background (SuperK_2023 has no GetMuonBackground; worker printed 0).
energy_scale is not in the 41-param list, so R_plus/R_minus are unused here.
"""
import json

import numpy as np
from scipy.optimize import minimize

# Nuisance vector order — must match pynufit.Analysis.NuisanceList for the
# xsec_barr_ntag config (verified against point_*.json nuisance_names).
FLUX_NAMES = ["normalization_below1GeV", "normalization_above1GeV", "tilt",
              "nunubar_ratio", "flavor_ratio", "barr_zenith"]
# 15 mask xsec tunes in the order of build_sk_response.XSEC_TUNES (class bits),
# but the VECTOR order below is the Analysis order (AxialMass sits after NCHad).
# The 3 CC_2p2h* dials (real 2p2h, |Mode|==2) sit after DIS, before CCQE — the
# new pfm config order (PLAN_2P2H_AUTONOMOUS). They require the 2p2h MC + a
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
#   a high-E pion/mu decay-path effect) — the zenith-SHAPE flux dial the S3.3
#   audit (2026-07-02) found missing: zenith_up/down pivot to zero at the
#   horizon (tanh^2) so they cannot reshape horizontal-vs-vertical. Energy-flat,
#   symmetric in cosz: flux *= 1 + x*g(cz) with g = (1-3cz^2)/2 (mean-zero over
#   cosz => a shape, not a norm; +0.5 horizontal, -1.0 vertical). nominal 0 (no-op).
OPTIONAL_FLUX_NAMES = ["solar_activity", "kpi_ratio", "flux_horizvert"]
ALL_FLUX_NAMES = CORE_FLUX_NAMES + ZENITH_DIALS + OPTIONAL_FLUX_NAMES

# Canonical (nominal, sigma) for every dial the engine can apply at fit time.
# The 41 baseline dials come straight from the production vectors above; the
# one-sided Pynu pair AtmoFlux.zenith_up/zenith_down (nominal 0, sigma 0.2,
# verified against analysis/AnalysisFiles/SK2023_Atm_datafit_xsec_barr.xml) is
# added so it can be switched in instead of / alongside barr_zenith WITHOUT a
# response rebuild — flux dials act on the oscillated tensor at fit time, not in
# the baked response matrices.
CANONICAL_DIALS = {n: (float(NOMINAL[k]), float(SIGMA[k]))
                   for k, n in enumerate(NUISANCE_NAMES)}

# Era-split detector dial sigmas per SK era [sk1, sk2, sk3, sk45], parsed from
# Pablo's datafit-SK pfm config (per-era SK published systematics). Registered
# into CANONICAL_DIALS as <stem>_<era> (nominal 1.0) so the 'phased' spec
# resolves. energy_scale is listed for completeness but needs the Rp/Rm binned
# machinery before it can be activated.
DET_ERA_SIGMA = {
    "energy_scale": [0.033, 0.028, 0.024, 0.021],
    "fcpc_separation": [0.006, 0.005, 0.009, 0.002],
    "pc_reduction": [0.024, 0.048, 0.005, 0.010],
    "mge_nonubkg": [0.13, 0.38, 0.27, 0.18],
    "fc_reduction": [0.002, 0.002, 0.008, 0.013],
    "subgev_2ring_pi0": [0.056, 0.044, 0.059, 0.056],
    "subgev_1ring_pi0": [0.12, 0.12, 0.10, 0.15],
    "multiring_nunubar_separation": [0.072, 0.079, 0.077, 0.068],
    "multiring_emu_separation": [0.06, 0.038, 0.053, 0.033],
    "multiring_eother_separation": [0.057, 0.041, 0.049, 0.034],
    "pc_stopthru_separation": [0.25, 0.14, 0.30, 0.10],
    "pi0_ring_separation": [0.023, 0.023, 0.023, 0.03],
    "e_ring_separation": [0.035, 0.038, 0.013, 0.018],
    "mu_ring_separation": [0.045, 0.08, 0.026, 0.023],
    "singlering_pid": [0.002, 0.007, 0.0026, 0.0035],
    "multiring_pid": [0.065, 0.10, 0.05, 0.035],
    "upmu_shower_separation": [0.034, 0.044, 0.024, 0.03],
    "upmu_stop_bkg": [0.16, 0.21, 0.20, 0.17],
    "upmu_showering_bkg": [0.18, 0.14, 0.24, 0.17],
    "upmu_nonshowering_bkg": [0.18, 0.14, 0.24, 0.24],
}
for _stem, _sigs in DET_ERA_SIGMA.items():
    for _tag, _s in zip(ERA_TAGS, _sigs):
        CANONICAL_DIALS[f"{_stem}_{_tag}"] = (1.0, _s)
CANONICAL_DIALS["zenith_up"] = (0.0, 0.2)
CANONICAL_DIALS["zenith_down"] = (0.0, 0.2)
# solar-activity flux dial (nominal 0 = no-op, sigma 0.15 from the pfm config).
CANONICAL_DIALS["solar_activity"] = (0.0, 0.15)
CANONICAL_DIALS["kpi_ratio"] = (0.0, 0.05)   # K/pi high-E flux ramp (nominal 0 = no-op)
CANONICAL_DIALS["flux_horizvert"] = (0.0, 0.03)  # H/V flux ratio (nominal 0 = no-op)

# ---- energy-scale dials (bin-level reco-E migration) ------------------------
# Era-split energy_scale_<tag> (CANONICAL_DIALS entries set by the DET_ERA_SIGMA
# loop above). The SK public MC stores reco energy QUANTIZED to one value per
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
# generous default priors (diagnostic: "can the SHAPE absorb it?"). SK's published
# 1-sigma are tighter (nu_e/nu-bar_e ~0.03, flavor ~0.02 sub-GeV) — the worker can
# override via --flux-ratio-sigma to test absorption within SK's own uncertainty.
FLUX_RATIO_PRIORS = {"flux_nuebar_subgev": (1.0, 0.10),
                     "flux_flavor_subgev": (1.0, 0.10)}
FLUX_RATIO_BOX = {"flux_nuebar_subgev": (0.3, 1.7),
                  "flux_flavor_subgev": (0.3, 1.7)}
for _rn, _pr in FLUX_RATIO_PRIORS.items():
    CANONICAL_DIALS[_rn] = _pr

#   (2) MOMENTUM-RESOLVED sub-GeV neutron-tagging migration. The production
#       `neutron_tagging_subgev` is a single whole-sample efficiency pull (one
#       migration ratio applied uniformly to every momentum bin), so it cannot
#       drain only the lowest-momentum nu-bar_e-tagged bins where the +36%
#       over-prediction sits (thesis Table 5.2). Split it into two independent
#       per-momentum-band efficiency dials (low / high), each a rate-conserving
#       migration WITHIN its band between donor {20,22}(0-neutron) and acceptor
#       {21,23}(1-neutron) sub-GeV samples. NTAG_PSPLIT = first high-band
#       momentum index (ie>=PSPLIT is "high"); ie<PSPLIT ("low") = the lowest
#       reco-momentum slice (ie=0 == logP<=2.4 per the per-bin study).
NTAG_SPLIT_NAMES = ["ntag_subgev_lowp", "ntag_subgev_highp"]
NTAG_PSPLIT = 2          # low band = ie in {0,1} (logP<=2.6); high = ie>=2
for _rn in NTAG_SPLIT_NAMES:
    CANONICAL_DIALS[_rn] = CANONICAL_DIALS["neutron_tagging_subgev"]  # (1.0, 0.12)

# ---- EXTENDED set: SK systematics we lack, screened for the octant (2026-06-18) ----
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
for _rn in FLUX_BAND_NAMES:
    CANONICAL_DIALS[_rn] = (1.0, 0.10)
ALL_FLUX_RATIO_NAMES = FLUX_RATIO_NAMES + FLUX_BAND_NAMES   # superset for detection

# (b) sub-GeV XSEC dials we lack (SK's largest sub-GeV levers, thesis Table B.1):
#   xsec_ccqe_shape      — CCQE energy-shape: CCQE *= (E_true)^(r-1). APPROXIMATION
#       of SK's "CCQE Shape" (+2.05sig, RFG-vs-LFG normalized-sigma diff vs E); we
#       lack the RFG/LFG tables, so this is a 1-param power-law E-tilt of CCQE.
#   xsec_ccqe_subgev_nue — sub-GeV (E<1) CCQE nu_e+nu-bar_e norm (SK 5% nu_e); ALSO
#       the surrogate for our MISSING 2p2h (zero in OLD MC -> CCQE inflated).
#   xsec_1p1h_subgev_nue / xsec_2p2h_subgev_nue — the THESIS-FAITHFUL SPLIT of the
#       above (2026-06-21). On the new 2p2h MC the CCQE class is pure 1p1h and 2p2h
#       is its own class, so the sub-GeV nu_e norm separates into a 1p1h piece
#       (sigma 0.05, ~SK's sub-GeV nu_e CCQE 5%) and a 2p2h piece (sigma 0.20, ~PE
#       cc2p2h_norm; box wide since 2p2h is poorly known). Same mask form as
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
XSEC_EXTRA_PRIORS = {"xsec_ccqe_shape": (1.0, 0.20), "xsec_ccqe_subgev_nue": (1.0, 0.05),
                     "xsec_1p1h_subgev_nue": (1.0, 0.05), "xsec_2p2h_subgev_nue": (1.0, 0.20),
                     "xsec_ccqe_shape_subgev": (0.0, 0.40)}   # nominal 0, sigma 0.40 (shape coeff)
XSEC_EXTRA_BOX = {"xsec_ccqe_shape": (0.3, 1.7), "xsec_ccqe_subgev_nue": (0.5, 1.5),
                  "xsec_1p1h_subgev_nue": (0.5, 1.5), "xsec_2p2h_subgev_nue": (0.0, 3.0),
                  "xsec_ccqe_shape_subgev": (-2.0, 2.0)}
for _rn, _pr in XSEC_EXTRA_PRIORS.items():
    CANONICAL_DIALS[_rn] = _pr
# sub-GeV nu_e norm dials handled by the generic XX_ke / gradient loops (name -> class-mask attr)
SUBGEV_NUE_NORM = {"xsec_ccqe_subgev_nue": "ccqe_nue_cls",
                   "xsec_1p1h_subgev_nue": "ccqe_nue_cls",     # CCQE == 1p1h on the 2p2h MC
                   "xsec_2p2h_subgev_nue": "twop2h_nue_cls"}

# ---- OPTIONAL multi-GeV CCQE flavor-norm dials (Track H, OFF by default) ------
# SK's "CCQE Norm., Multi-GeV" systematic, applied SEPARATELY to nu_e and nu_mu,
# each ~25% (Wester thesis 5569-5571: "the CCQE normalization ... multi-GeV ...
# nu_e and nu_mu ... 25%"; Table B.1 "Norm., Multi-GeV" 9247). Multi-GeV e-like
# is a primary dCP appearance sample, so a nu_e-localized multi-GeV CCQE norm is
# the largest dCP-relevant absorber our R2 120-set lacks -- only the global 10%
# CCQE norm reaches the multi-GeV region (DCP_DEPTH_AUDIT.md items #1/#3). Mirrors
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
for _rn in MULTIGEV_CCQE_NAMES:
    CANONICAL_DIALS[_rn] = (1.0, 0.25)

# ---- OPTIONAL relative-normalization dial (per-sample-group norm) ------------
# SK's "Relative Normalization" between sample groups (thesis Rel.Norm FC-MultiGeV
# -1.33 sigma): a flat multiplicative norm on the FC multi-GeV sample group, applied
# at the binned (per-sample) level inside detector_factors so the existing detector
# gradient machinery (dlnD) handles it. nominal 1 (no-op), sigma 0.05 (SK ~5%).
# FC multi-GeV samples (SuperK_Atm_Pheno sample_names): SK1-3 FC mG 1R {7,8,9} +
# MR {10,11,12,13}, SK4-5 FC mG 1R {24,25,26,27,28}.
REL_NORM_FCMG_SAMPLES = frozenset({7, 8, 9, 10, 11, 12, 13, 24, 25, 26, 27, 28})
REL_NORM_NAMES = ["rel_norm_fcmg"]
CANONICAL_DIALS["rel_norm_fcmg"] = (1.0, 0.05)

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
# dial carries the SAME ~15-event 1sigma background uncertainty (design doc Sec 3c).
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
for _rn in UPMU_BKG_SHAPE_NAMES:
    CANONICAL_DIALS[_rn] = (1.0, UPMU_BKG_SHAPE_SIGMA[_rn])

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
for _rn in UPDOWN_ESCALE_NAMES:
    CANONICAL_DIALS[_rn] = (0.0, UPDOWN_ESCALE_SIGMA[_rn])   # nominal 0 = no-op

# ---- OPTIONAL direction-smearing systematic (Track D, OFF by default) --------
# BEYOND-SK-VOCABULARY DIAGNOSTIC (HANDOFF_2026-07-04 item 9: NO direction-smearing
# systematic exists anywhere in SK's treatment -- central-MC direction physics only;
# our binned response inherits it; P1.1 bounds zenith-shape mismodeling at <~2%). Every
# norm-type dial tested (R2HV/R2UBS/UDE, r2_faithful) left a coherent, persistent,
# irreducible residual (HANDOFF item 8): the multi-GeV UPGOING oscillation fingerprint
# (e +2.174 / mu +0.901, p 2.5-10 GeV, cz[-0.84,-0.64]), "oscillation-shaped data
# structure no SK-vocabulary reweighting expresses". This dial asks the one question the
# norm dials could not: can a direction-RESOLUTION SHAPE freedom -- zenith-bin migration
# built from a confusion matrix (user directive: "assign a 5-10% directional smearing to
# the per-event MC ... which we in turn use to build the systematic in the binned
# engine") -- reach that residual and/or move Delta m^2, where every norm dial could not?
# The dial is still FLAVOR-BLIND (it migrates e-like and mu-like bins by the SAME reco-cz
# confusion matrix), so it cannot IMPOSE the anti-correlated e/mu fingerprint; but unlike
# the norm dials it CAN reshape the zenith DISTRIBUTION (redistribute counts between
# zenith bins), so a measured null is a genuinely new, informative answer.
#
# Single global dial s (nominal 0 = exact no-op), prior sigma 10 (effectively
# unconstrained -- the question is whether the DATA pulls s), custom box [0,1] (s<0 =
# de-smearing is unphysical and can drive negative expectations; s in [0,1] keeps the
# migration operator A=(1-s)I + s*M a convex mix of two non-negative matrices, so
# E'=A@E >= 0). The migration matrices M[sample] (nz x nz, reco-cz confusion, built by
# build_dirsmear_confusion.py) are chosen at engine construction via the dirsmear_matrix
# ctor arg (default OFF -- no matrix loaded unless the dial is active). Action per
# sample x reco-momentum row: E(s) = E + s*(M - I) @ E; linear in s so dE/ds=(M-I)@E is
# exact and s-independent (mirrors the absolute energy-scale migration precedent).
DIR_SMEAR_NAME = "dir_smear"
DIR_SMEAR_SIGMA = 10.0             # diagnostic: effectively unconstrained prior
DIR_SMEAR_BOX = (0.0, 1.0)         # one-sided s in [0,1] (positivity of A=(1-s)I+sM)
CANONICAL_DIALS[DIR_SMEAR_NAME] = (0.0, DIR_SMEAR_SIGMA)   # nominal 0 = exact no-op


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


def resolve_nuisance_spec(spec):
    """Return (names, nominal, sigma) for a nuisance-set selector.

    spec:
      None / 'barr'  -> production 41-vector (barr_zenith active) — DEFAULT,
                        bit-identical to the pre-switch engine
      'updown'       -> one-sided zenith_up + zenith_down instead of barr (42)
      'both'         -> zenith_up + zenith_down + barr_zenith together (43)
      <path>.xml     -> active dials parsed from a Pynu config (document order)
      [names...]     -> explicit ordered name list
    Only the flux-zenith block is switchable without a response rebuild; the
    xsec (13) and detector (22) blocks are baked into the build artifacts and
    are validated against them in SKBinnedEngine.__init__.
    """
    if spec is None or spec == "barr":
        names = list(NUISANCE_NAMES)
    elif spec == "updown":
        names = CORE_FLUX_NAMES + ["zenith_up", "zenith_down"] \
            + XSEC_VECTOR_NAMES + DET_NAMES
    elif spec == "both":
        names = CORE_FLUX_NAMES + ["zenith_up", "zenith_down", "barr_zenith"] \
            + XSEC_VECTOR_NAMES + DET_NAMES
    elif spec == "phased":
        # Pablo's datafit-SK era-split detector model on the phased (4-era)
        # response. The 19 per-sample detector stems are era-split (x4);
        # fiducial_volume + the two neutron-tag dials stay era-independent.
        # energy_scale (x4), solar_activity, decay_e_tagging need new binned
        # machinery and are staged out (102 of the release's 107 dials).
        det = []
        for stem in DET_ERA_STEMS:
            det += [f"{stem}_{tag}" for tag in ERA_TAGS]
        det += ["fiducial_volume", "neutron_tagging_subgev",
                "neutron_tagging_multigev"]
        names = CORE_FLUX_NAMES + ["zenith_up", "zenith_down"] \
            + XSEC_VECTOR_NAMES + det
    elif spec == "phased_prod":
        # PRODUCTION octant set (2026-06-26): era-split phased (102) + the four
        # octant-relevant systematics added this session + solar_activity:
        #   +9 energy-banded flux ratios (ALL_FLUX_RATIO_NAMES)
        #   +xsec_ccqe_shape_subgev (sub-GeV-localized CCQE shape, the #1 octant lever)
        #   +kpi_ratio (K/pi high-E flux), +rel_norm_fcmg (FC-multiGeV rel. norm)
        #   +solar_activity (flux).
        # NOT included here (CAN be added for a maximal set):
        #   - energy_scale_* : the BIN-LEVEL version is functional (NOT inert -- the
        #     'inert' issue was the abandoned +-2% Rp/Rm re-digitization on the quantized
        #     MC; the histogram-migration version works). Left out only because its thesis
        #     pull is sub-1sigma (weak octant lever) and it can't be cross-checked vs the
        #     buggy upstream event-engine energy_scale.
        #   - xsec_1p1h/2p2h_subgev_nue : phased_max's sub-GeV nu_e norms; scoped out (this
        #     session's flagged 4 only), and partly redundant with the base CCQE/CC_2p2h norms.
        # = 102 + 9 + 4 = 115 dials. All additions act at fit time (no response rebuild).
        det = []
        for stem in DET_ERA_STEMS:
            det += [f"{stem}_{tag}" for tag in ERA_TAGS]
        det += ["fiducial_volume", "neutron_tagging_subgev", "neutron_tagging_multigev"]
        names = (CORE_FLUX_NAMES + ["zenith_up", "zenith_down"] + XSEC_VECTOR_NAMES + det
                 + list(ALL_FLUX_RATIO_NAMES)
                 + ["xsec_ccqe_shape_subgev", "kpi_ratio", "rel_norm_fcmg", "solar_activity"])
    elif spec == "phased_full":
        # MAXIMAL octant set (2026-06-26): phased_prod (115) + the dials that were scoped
        # out of it -- the two phased_max sub-GeV nu_e norms + the 4 era-split energy_scale
        # dials (bin-level, functional). = 115 + 2 + 4 = 121 dials. Tests "does the octant
        # relax further with everything functional turned on?". energy_scale needs the
        # 4-era phased response. All additions act at fit time (no response rebuild).
        names = (list(resolve_nuisance_spec("phased_prod")[0])
                 + ["xsec_1p1h_subgev_nue", "xsec_2p2h_subgev_nue"]
                 + list(ENERGY_SCALE_NAMES))
    elif spec in ("R1", "pfm_base"):
        # PROVENANCE LADDER rung 1 (2026-06-27): the FULL Pablo-FM release (107) =
        # phased (102) + the 4 era-split energy_scale dials + solar_activity. These
        # last 5 are PFM-specified (Pablo's 107-dial release) but were staged out of
        # 'phased' pending the bin-level/binned machinery we since built; per the user
        # they belong in the PFM base. Pure Pablo -- nothing thesis-motivated of ours.
        names = (list(resolve_nuisance_spec("phased")[0])
                 + list(ENERGY_SCALE_NAMES) + ["solar_activity"])
    elif spec in ("R2", "ladder_r2"):
        # rung 2 (120): R1 + the thesis-FAITHFUL SK systematics the base lacks --
        # K/pi + FC-multiGeV rel.norm + the 3-band flux ratios (nu_e/nubar_e, flavor,
        # nu_mu/nubar_mu x sub/mid/high = ALL_FLUX_RATIO_NAMES, 9) + sub-GeV nu_e CCQE
        # norm (xsec_1p1h_subgev_nue, ~SK 5%) + sub-GeV-localized CCQE shape. Each maps
        # 1:1 (form-faithful) to a real Wester-thesis dial. Run loose AND SK-tight (the
        # worker's --tight narrows the flux-ratio sigmas toward SK's published widths).
        names = (list(resolve_nuisance_spec("R1")[0])
                 + ["kpi_ratio", "rel_norm_fcmg"]
                 + list(ALL_FLUX_RATIO_NAMES)
                 + ["xsec_1p1h_subgev_nue", "xsec_ccqe_shape_subgev"])
    elif spec in ("R3", "ladder_r3"):
        # rung 3 (122): R2 + OUR constructions -- the 2p2h sub-GeV nu_e surrogate and
        # the momentum-split neutron tag (low/high p), which REPLACES the single
        # neutron_tagging_subgev (novel beyond SK AND PE). Needs migration_mode=
        # 'weighted' (per-bin band rates); the engine asserts this.
        base = list(resolve_nuisance_spec("R2")[0])
        k = base.index("neutron_tagging_subgev")
        base[k:k + 1] = list(NTAG_SPLIT_NAMES)
        names = base + ["xsec_2p2h_subgev_nue"]
    elif spec in ("R2NTS", "ladder_r2_ntagsplit"):
        # R2 + the momentum-split neutron tag ONLY (121): the R3 ntag
        # replacement (neutron_tagging_subgev -> ntag_subgev_lowp/highp)
        # WITHOUT R3's xsec_2p2h_subgev_nue surrogate. Isolates the octant
        # absorber flagged by the bf-convergence P1.3 audit (2026-07-01) on
        # top of the r2_tight_lump arm. Needs migration_mode='weighted'
        # (the engine asserts this via ntag_split).
        names = list(resolve_nuisance_spec("R2")[0])
        k = names.index("neutron_tagging_subgev")
        names[k:k + 1] = list(NTAG_SPLIT_NAMES)
    elif spec in ("R2HV", "ladder_r2_horizvert"):
        # R2 + the Horizontal/Vertical flux-ratio dial ONLY (121): fills the
        # 2026-07-02 S3.3 audit gap — thesis 5509-18 high-E zenith-shape flux
        # systematic that zenith_up/down structurally cannot make (tanh^2
        # pivots to zero at the horizon). g(cz)=(1-3cz^2)/2, sigma 0.03,
        # nominal 0 = exact no-op.
        names = list(resolve_nuisance_spec("R2")[0]) + ["flux_horizvert"]
    elif spec in ("R2UBS", "ladder_r2_upmu_bkg_shape"):
        # R2 + the up-mu background zenith x momentum SHAPE dials ONLY (123): fills
        # the 2026-07-02 audit gap C -- SK's cosmic-mu bkg subtraction (thesis
        # 5975-5984) is horizon-localized (2 zenith bins stopping / 1 through-going
        # non-showering) + momentum-shaped, while our whole-sample upmu_*_bkg norms
        # are flat per era. 3 era-common dials, nominal 1 = exact no-op. Sibling of
        # R2HV; same r2_tight_lump base so the two smokes compare apples-to-apples.
        names = list(resolve_nuisance_spec("R2")[0]) + list(UPMU_BKG_SHAPE_NAMES)
    elif spec in ("R2UDE", "ladder_r2_updown_escale"):
        # R2 + the up/down energy-scale detector dials ONLY (124): the last
        # direction-coupled thesis detector systematic the base lacks (Wester
        # 6011-6018, Table 5.6). 4 era-split dials, anti-symmetric up/down
        # normalization of FC+PC events (up *= 1+d, down *= 1-d), nominal 0 =
        # exact no-op. Sibling of R2UBS/R2HV on the same r2_tight_lump base so the
        # smokes compare apples-to-apples. Needs the phased 4-era response.
        names = list(resolve_nuisance_spec("R2")[0]) + list(UPDOWN_ESCALE_NAMES)
    elif spec in ("R2DS", "ladder_r2_dirsmear"):
        # R2 + the direction-smearing dial ONLY (121): the beyond-SK-vocabulary
        # direction-RESOLUTION shape freedom (Track D). One global dir_smear dial,
        # nominal 0 = exact no-op, box [0,1], loose prior sigma 10. Needs the
        # dirsmear_matrix ctor arg (reco-cz confusion matrix). Sibling of R2UBS/UDE on
        # the same r2_tight_lump base so the smokes compare apples-to-apples.
        names = list(resolve_nuisance_spec("R2")[0]) + [DIR_SMEAR_NAME]
    elif spec in ("R2FUDECCQE", "r2_fude_ccqe", "ladder_r2_fude_ccqe"):
        # R2 + Track H's multi-GeV CCQE appearance freedom, ON TOP OF the
        # R2HV/R2UBS/R2UDE audit-gap dials (131): the ADOPTED PRODUCTION arm
        # (2026-07-06) -- R2 (120) + flux_horizvert (R2HV) + the up-mu
        # background-shape triplet (R2UBS) + the era-split up/down
        # energy-scale quartet (R2UDE) + xsec_ccqe_shape (global CCQE-shape
        # freedom, distinct from R2's xsec_ccqe_shape_subgev) + the two
        # multi-GeV CCQE flavor norms (xsec_ccqe_multigev_nue/numu, Track H).
        # This is the FIRST Δm²-moving arm found in the campaign; order below
        # is byte-order-verified against the production seed
        # (claude-tmp/dm2/seeds_r2_fude_ccqe/r2_fude_ccqe.json:nuisance_names).
        names = (list(resolve_nuisance_spec("R2")[0]) + ["flux_horizvert"]
                 + list(UPMU_BKG_SHAPE_NAMES) + list(UPDOWN_ESCALE_NAMES)
                 + ["xsec_ccqe_shape"] + list(MULTIGEV_CCQE_NAMES))
    elif isinstance(spec, str) and spec.startswith("octsyst"):
        # octant-absorber ladder arms, all on the updown (42-dial) base:
        #   octsyst_base : the 42-dial reference (== 'updown')
        #   octsyst_flux : + sub-GeV flux ratios (flux_nuebar/flux_flavor)
        #   octsyst_ntag : neutron_tagging_subgev -> momentum-split low/high
        #   octsyst_both : both of the above together
        # extended arms (2026-06-18) build on octsyst_both (ntag split + sub-GeV
        # flux ratios) and add the SK systematics we lack:
        #   octsyst_fluxband : + 7 energy-banded flux ratios (mid/high + numubar)
        #   octsyst_xsec     : + 2 sub-GeV xsec dials (CCQE shape, sub-GeV CCQE nue)
        #   octsyst_max      : + both extension blocks (all 9 new dials)
        ext = {"octsyst_base", "octsyst_flux", "octsyst_ntag", "octsyst_both",
               "octsyst_fluxband", "octsyst_xsec", "octsyst_max"}
        if spec not in ext:
            raise ValueError(f"unknown octsyst spec {spec!r}")
        ntag = spec in ("octsyst_ntag", "octsyst_both", "octsyst_fluxband",
                        "octsyst_xsec", "octsyst_max")
        subgev_ratios = spec in ("octsyst_flux", "octsyst_both", "octsyst_fluxband",
                                 "octsyst_xsec", "octsyst_max")
        det = list(DET_NAMES)
        if ntag:
            i = det.index("neutron_tagging_subgev")
            det[i:i + 1] = NTAG_SPLIT_NAMES
        names = CORE_FLUX_NAMES + ["zenith_up", "zenith_down"] \
            + XSEC_VECTOR_NAMES + det
        if subgev_ratios:
            names = names + FLUX_RATIO_NAMES
        if spec in ("octsyst_fluxband", "octsyst_max"):
            names = names + FLUX_BAND_NAMES
        if spec in ("octsyst_xsec", "octsyst_max"):
            names = names + XSEC_EXTRA_NAMES
    elif isinstance(spec, str) and spec.endswith(".xml"):
        names = _parse_xml_active(spec)
    elif isinstance(spec, (list, tuple)):
        names = list(spec)
    else:
        raise ValueError(f"unknown nuisance_spec {spec!r}")
    missing = [n for n in names if n not in CANONICAL_DIALS]
    if missing:
        raise ValueError(f"nuisance_spec has unsupported dials {missing} "
                         "(only the 41 baseline dials + zenith_up/zenith_down "
                         "are available without a response rebuild)")
    nominal = np.array([CANONICAL_DIALS[n][0] for n in names])
    sigma = np.array([CANONICAL_DIALS[n][1] for n in names])
    return names, nominal, sigma


def _unphys(x):
    """PhysicsTunes._unphysical_value with default bounds."""
    return x < 0 or x > 9999999


class SKBinnedEngine:
    """migration_mode selects the migration-ratio convention for the
    rate-conserving detector tunes:

      'weighted' (default, current production): r = physics-weighted rates
          (BaseWeight*PhysicsWeight sums per sample), recomputed at every
          oscillation point — the post-bugfix SKCombinedDetector behavior.
      'rawcount' (legacy): r = raw MC event counts per sample, frozen across
          the oscillation grid — the pre-bugfix behavior. Collaborator
          rationale (2026-06-11): experiments generate migration templates
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
        # FC multi-GeV per-bin mask for the optional rel_norm_fcmg dial
        self.fcmg_bin_mask = np.isin(self.bin_sample, list(REL_NORM_FCMG_SAMPLES))

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

        # ---- energy-scale bin-level migration geometry (reco-E adjacency) -----
        # Built only when energy_scale dials are active. es_below[b] = the bin one
        # reco-E step lower in the same (sample, reco-cz) column (-1 at ie=0);
        # es_has_above[b] = 1 if b can spill upward (ie < ne-1). Geometry only =>
        # works with the quantized MC; no response rebuild, no Rp/Rm.
        if self.active_energy_scale:
            if len(self.active_energy_scale) != self.n_era:
                raise ValueError("energy_scale needs one dial per era "
                                 f"(n_era={self.n_era}); got {self.active_energy_scale}")
            self.es_below = np.full(self.n_bins, -1, dtype=np.int64)
            self.es_has_above = np.zeros(self.n_bins)
            for s, (off, ne_, nz) in self.sample_table.items():
                for ie in range(ne_):
                    base = off + ie * nz
                    if ie > 0:
                        self.es_below[base:base + nz] = np.arange(base, base + nz) - nz
                    if ie < ne_ - 1:
                        self.es_has_above[base:base + nz] = 1.0
            self.es_has_below = (self.es_below >= 0).astype(float)
            self._es_idx = [self.nuisance_names.index(f"energy_scale_{ERA_TAGS[e]}")
                            for e in range(self.n_era)]

        # ---- OPTIONAL absorber masks (built only when the dials are active) ----
        # (1) flux-ratio per-class flavor/sign selectors (band E<1 GeV applied
        #     at fit time via self.e_below1, like the AxialMass A_ke factor).
        if self.active_flux_ratios:
            self._fr_leg = {
                "nuebar": (self.cls_pdg == -12), "nue": (self.cls_pdg == 12),
                "e": (np.abs(self.cls_pdg) == 12), "mu": (np.abs(self.cls_pdg) == 14),
                "numubar": (self.cls_pdg == -14), "numu": (self.cls_pdg == 14),
            }
            # back-compat aliases used by older code paths
            self.fr_is_nuebar = self._fr_leg["nuebar"]
            self.fr_is_nue = self._fr_leg["nue"]
            self.fr_is_e = self._fr_leg["e"]
            self.fr_is_mu = self._fr_leg["mu"]
        # (2) momentum-resolved sub-GeV neutron-tag per-bin band masks. Within a
        #     sample bin = off + ie*nz + iz (build_sk_response.reco_bin_index),
        #     so the reco-momentum index is ie = (bin-off)//nz; ie<NTAG_PSPLIT is
        #     the low band. Donor {20,22}=0-neutron, acceptor {21,23}=1-neutron.
        if self.ntag_split:
            def _band(sample_ids, low):
                mk = np.zeros(self.n_bins, dtype=bool)
                for s in sample_ids:
                    key = str(s) if str(s) in self.sample_table else s
                    off, ne, nz = self.sample_table[key]
                    idx = np.arange(off, off + ne * nz)
                    ie = (idx - off) // nz
                    sel = ie < NTAG_PSPLIT if low else ie >= NTAG_PSPLIT
                    mk[idx[sel]] = True
                return mk
            # band dial -> (donor per-bin mask, acceptor per-bin mask)
            self.ntag_bands = {
                "ntag_subgev_lowp": (_band([20, 22], True), _band([21, 23], True)),
                "ntag_subgev_highp": (_band([20, 22], False), _band([21, 23], False)),
            }
        # (3) up-mu background zenith x momentum SHAPE per-bin masks. Bin index =
        #     off + ie*nz + iz (build_sk_response.reco_bin_index); reco-momentum ie,
        #     reco-cosZ iz. Horizon-nearest iz = nz-1 for the up-mu z10bins_up grid.
        if self.active_upmu_bkg:
            self.upmu_bkg_masks = {}
            for name in self.active_upmu_bkg:
                sid, iz_set, ie_set = UPMU_BKG_SHAPE_SPEC[name]
                key = str(sid) if str(sid) in self.sample_table else sid
                off, ne, nz = self.sample_table[key]
                mk = np.zeros(self.n_bins, dtype=bool)
                for ie in range(ne):
                    if ie_set is not None and ie not in ie_set:
                        continue
                    for iz in iz_set:
                        if 0 <= iz < nz:
                            mk[off + ie * nz + iz] = True
                self.upmu_bkg_masks[name] = mk

        # (4) up/down energy-scale SIGNED per-bin mask (SK Up/Down Energy Scale,
        #     thesis 6011-6018): +1 on up-going (cz<0) bins, -1 on down-going
        #     (cz>=0), 0 on excluded samples. FC+PC z10bins samples only -- the
        #     z10bins edge index 5 == cz=0 so iz<nz//2 is up-going; up-mu {16,17,18}
        #     (all up-going) + single-reco-zenith FC {1,2,5,6} (nz=1, straddle cz=0)
        #     get factor 1. Era-INDEPENDENT geometry; the per-era dial VALUE is
        #     routed to detector_factors via _era_theta. The signed mask lets one
        #     dial scale up *=(1+d) and down *=(1-d) with the gradient sign built in.
        if self.active_ude:
            if set(self.active_ude) != set(UPDOWN_ESCALE_NAMES):
                raise ValueError("up/down energy-scale needs all "
                                 f"{len(ERA_TAGS)} era dials {UPDOWN_ESCALE_NAMES}; "
                                 f"got {self.active_ude}")
            if self.n_era != len(ERA_TAGS):
                raise ValueError("up/down energy-scale dials need the phased "
                                 f"{len(ERA_TAGS)}-era response (n_era={self.n_era})")
            self.ude_sign = np.zeros(self.n_bins)
            for s, (off, ne_, nz) in self.sample_table.items():
                if int(s) in UPDOWN_ESCALE_EXCLUDE or nz != 10:
                    continue
                for ie in range(ne_):
                    for iz in range(nz):
                        self.ude_sign[off + ie * nz + iz] = \
                            1.0 if iz < nz // 2 else -1.0

        # (5) direction-smearing reco-cz confusion matrices. Loaded LAZILY and only
        #     when the dir_smear dial is active -- ALL existing paths are untouched.
        #     Per sample a nz x nz matrix M[sample] (reco-cz confusion, IE-independent);
        #     the migration applies the same matrix to every reco-momentum row of the
        #     sample. Precompute per-sample (off, ne, nz, M) blocks, SKIPPING identity
        #     (z1bins) samples so they are bit-unchanged at any s.
        self._ds_blocks = []
        if self.active_dir_smear:
            if self._dirsmear_matrix_path is None:
                raise ValueError("dir_smear dial active but no dirsmear_matrix given "
                                 "(pass dirsmear_matrix=<confusion_*.npz path>)")
            dz = np.load(self._dirsmear_matrix_path, allow_pickle=True)
            self.dirsmear_meta = (json.loads(str(dz["manifest"]))
                                  if "manifest" in dz.files else {})
            for s, (off, ne_, nz) in self.sample_table.items():
                M = np.asarray(dz[f"M_{int(s)}"], dtype=float)
                if M.shape != (nz, nz):
                    raise ValueError(f"dirsmear M_{s} shape {M.shape} != ({nz},{nz})")
                if np.allclose(M, np.eye(nz), atol=0, rtol=0):
                    continue                       # identity (z1bins) -> inert, skip
                self._ds_blocks.append((int(off), int(ne_), int(nz), M))

        # FewEntries mask from the unfiltered data vector (Experiment.SetObservedBinned)
        self.few = self.observed > MIN_ENTRIES
        self.obs_f = self.observed[self.few]

        # static flux-tune cell masks
        self.e_below1 = self.e_c < 1.0
        self.e_above1 = self.e_c > 1.0
        self.barr_env = 0.07 / (1.0 + (self.e_c / 0.5) ** 2)     # _barr_zenith_envelope
        self.tanh3z = np.tanh(3.0 * self.z_c)
        self.tanhz2 = np.tanh(self.z_c) ** 2          # zenith_up/down envelope
        # H/V flux-ratio shape: mean-zero over cosz, +0.5 horizontal, -1.0 vertical
        self.horizvert_shape = 0.5 * (1.0 - 3.0 * self.z_c ** 2)
        self.log10e = np.log10(self.e_c)
        # K/pi high-E flux ramp: 0 below KPI_E0, rising as log10(E/KPI_E0) above (nE,)
        self.kpi_shape = np.maximum(0.0, np.log10(self.e_c / KPI_E0))

        # energy bands (true E_nu) for the SK 3-band flux ratios
        self.e_bands = {"sub": (self.e_c < 1.0).astype(float),
                        "mid": ((self.e_c >= 1.0) & (self.e_c < 10.0)).astype(float),
                        "high": (self.e_c >= 10.0).astype(float)}
        # resolve each active flux-ratio dial -> (band envelope (nE,), heavy (n_cls,),
        # light (n_cls,)) so cell_weights/gradient iterate ONE generic registry.
        self.fr_resolved = {}
        for nm in self.active_flux_ratios:
            band, hv, lt = FLUX_RATIO_SPEC[nm]
            self.fr_resolved[nm] = (self.e_bands[band],
                                    self._fr_leg[hv], self._fr_leg[lt])
        # optional sub-GeV xsec masks (CCQE / 2p2h class via the baked mask bits)
        if self.active_xsec_extra or self.active_multigev_ccqe:
            self.ccqe_cls = self.cls_bits[:, MASK_TUNES.index("CCQE")]   # (n_cls,)
            self.ccqe_nue_cls = self.ccqe_cls & (self.cls_flavor == 0)   # nu_e+nu-bar_e CCQE(=1p1h)
            # multi-GeV CCQE flavor-norm masks (Track H): nu_mu+nu-bar_mu CCQE class
            # + the E_true>=1.33 GeV complement of the sub-GeV shape region. Built
            # here (unused, hence output-inert) for any active_xsec_extra spec.
            self.ccqe_numu_cls = self.ccqe_cls & (self.cls_flavor == 1)  # nu_mu+nu-bar_mu CCQE
            self.e_multigev = self.e_c >= CCQE_SHAPE_SUBGEV_E            # (nE,) complement of <1.33
            # sub-GeV-localized CCQE shape: mean-zero log-E tilt confined to E_true<1.33 GeV
            # (0 above; centred over the sub-GeV cells => ~rate-neutral SHAPE, pivot-free).
            self.ccqe_shape_subgev = np.where(self.e_c < CCQE_SHAPE_SUBGEV_E,
                                              np.log(self.e_c), 0.0)            # (nE,)
            _sub = self.e_c < CCQE_SHAPE_SUBGEV_E
            self.ccqe_shape_subgev[_sub] -= self.ccqe_shape_subgev[_sub].mean()
            # 2p2h nu_e class (thesis split); zeros if the response predates 2p2h
            if "CC_2p2h" in MASK_TUNES:
                twop2h = self.cls_bits[:, MASK_TUNES.index("CC_2p2h")]
                self.twop2h_nue_cls = twop2h & (self.cls_flavor == 0)
            else:
                self.twop2h_nue_cls = np.zeros(self.n_cls, dtype=bool)

    # ---------------- weight fields ----------------
    def cell_weights(self, phi, theta):
        """W[k, cE, cZ] for nuisance vector theta and physics tensor phi[2,3,nE,nZ].

        NC classes get phi = 1 (SuperK_2023.UpdatePhysicsWeights NC override).
        """
        t = dict(zip(self.nuisance_names, theta))

        # physics: gather per class, NC -> 1
        P = phi[self.cls_type, self.cls_flavor]                  # (n_cls, nE, nZ)
        P = np.where(self.cls_cc[:, None, None] == 1, P, 1.0)

        # flux field on (nE,) and (nZ,) -> broadcast (per class via pdg)
        f_e = np.ones(self.nE)
        f_e = np.where(self.e_below1, t["normalization_below1GeV"], f_e) \
            * np.where(self.e_above1, t["normalization_above1GeV"], 1.0)
        f_e = f_e * (self.e_c / 10.0) ** t["tilt"]
        # solar-activity (optional): w = 1 - x*A*exp(-E_true/L), energy-only =>
        # folds into the (nE,) flux field exactly like tilt (no response rebuild).
        if "solar_activity" in t:                              # AtmoFlux.solar_activity
            f_e = f_e * (1.0 - t["solar_activity"] * SOLAR_AMP
                         * np.exp(-self.e_c / SOLAR_SCALE))
        if "kpi_ratio" in t:                                   # K/pi high-E flux ramp
            f_e = f_e * (1.0 + t["kpi_ratio"] * self.kpi_shape)
        # zenith flux dials (switchable): barr_zenith (two-sided, energy-damped)
        # and/or the one-sided zenith_up/zenith_down pair. For the default set
        # (barr only) this reproduces the prior F_ez exactly.
        if "barr_zenith" in t:
            barr = 1.0 + self.barr_env * t["barr_zenith"]
            if np.any(barr <= 0):
                F_ez = 1e-3 * np.ones((self.nE, self.nZ))
            else:
                F_ez = f_e[:, None] * barr[:, None] ** self.tanh3z[None, :]
        else:
            F_ez = f_e[:, None] * np.ones((self.nE, self.nZ))
        if "zenith_up" in t:                                   # AtmoFlux.zenith_up
            zu = np.where(self.z_c < 0,
                          1.0 - t["zenith_up"] * self.tanhz2, 1.0)
            F_ez = F_ez * zu[None, :]
        if "zenith_down" in t:                                 # AtmoFlux.zenith_down
            zd = np.where(self.z_c >= 0,
                          1.0 - t["zenith_down"] * self.tanhz2, 1.0)
            F_ez = F_ez * zd[None, :]
        if "flux_horizvert" in t:                    # H/V flux ratio (S3.3 gap #1)
            F_ez = F_ez * (1.0 + t["flux_horizvert"]
                           * self.horizvert_shape)[None, :]
        # per-class flux scalars
        f_cls = np.where(self.cls_pdg < 0, t["nunubar_ratio"], 1.0) \
            * np.where(np.abs(self.cls_pdg) == 12, t["flavor_ratio"], 1.0)

        # xsec: 12 mask tunes -> per-class scalar product
        xs = np.array([t[n] for n in MASK_TUNES])
        X_cls = np.where(self.cls_bits, xs[None, :], 1.0).prod(axis=1)
        # AxialMass: CC only, continuous in log10 ETrue
        ax = 1.0 + 0.042 * (t["AxialMass"] - 1.0) * 1.05 * self.log10e
        A_ke = np.where(self.cls_cc[:, None] == 1, ax[None, :], 1.0)  # (n_cls, nE)

        # OPTIONAL flux ratios (rate-conserving symmetric, per-dial energy band).
        # heavy leg *= 2r/(1+r), light leg *= 2/(1+r); r=1 => no-op. Generic over
        # the resolved registry (sub-GeV absorbers + energy-banded extensions).
        FR_ke = np.ones((self.n_cls, self.nE))
        for name, (band, hv, lt) in self.fr_resolved.items():
            r = t[name]
            fh = 1.0 + band[None, :] * (2.0 * r / (1.0 + r) - 1.0)   # heavy leg
            fl = 1.0 + band[None, :] * (2.0 / (1.0 + r) - 1.0)       # light leg
            FR_ke = FR_ke * np.where(hv[:, None], fh, 1.0) \
                          * np.where(lt[:, None], fl, 1.0)

        # OPTIONAL sub-GeV xsec dials (CCQE shape E-tilt + sub-GeV nu_e norms).
        # r=1 => no-op (factor 1).
        XX_ke = np.ones((self.n_cls, self.nE))
        if self.active_xsec_extra:
            if "xsec_ccqe_shape" in t:
                b = t["xsec_ccqe_shape"] - 1.0
                tilt = self.e_c[None, :] ** b                 # (1,nE), CCQE only
                XX_ke = XX_ke * np.where(self.ccqe_cls[:, None], tilt, 1.0)
            if "xsec_ccqe_shape_subgev" in t:                 # sub-GeV-localized CCQE shape
                fac = 1.0 + t["xsec_ccqe_shape_subgev"] * self.ccqe_shape_subgev  # (nE,)
                XX_ke = XX_ke * np.where(self.ccqe_cls[:, None], fac[None, :], 1.0)
            # sub-GeV nu_e norm dials (lumped ccqe surrogate + the 1p1h/2p2h split),
            # generic over SUBGEV_NUE_NORM (name -> class-mask attribute).
            for _dial, _mattr in SUBGEV_NUE_NORM.items():
                if _dial in t:
                    r = t[_dial]
                    fac = 1.0 + self.e_below1[None, :] * (r - 1.0)   # (1,nE)
                    XX_ke = XX_ke * np.where(getattr(self, _mattr)[:, None], fac, 1.0)

        # OPTIONAL multi-GeV CCQE flavor norms (E_true>=1.33, per-flavor): mirrors
        # the sub-GeV nu_e norm above with the complementary energy mask. r=1 =>
        # no-op. Separate active flag (may be present without any xsec_extra dial).
        if self.active_multigev_ccqe:
            for _dial, _mattr in MULTIGEV_CCQE_NORM.items():
                if _dial in t:
                    r = t[_dial]
                    fac = 1.0 + self.e_multigev[None, :] * (r - 1.0)  # (1,nE)
                    XX_ke = XX_ke * np.where(getattr(self, _mattr)[:, None], fac, 1.0)

        W = P * F_ez[None, :, :] * (f_cls * X_cls)[:, None, None] \
            * A_ke[:, :, None] * FR_ke[:, :, None] * XX_ke[:, :, None]
        return W

    # ---------------- contractions ----------------
    def contract(self, W):
        """n_pre[b] = R contracted with cell weights W."""
        return np.bincount(self.Rb, weights=self.Rv * W.ravel()[self.R_widx],
                           minlength=self.n_bins)

    def contract_var(self, Wsq):
        """sum of BaseWeight^2 * W^2 per bin (pre-detector)."""
        return np.bincount(self.S2b, weights=self.S2v * Wsq.ravel()[self.S2_widx],
                           minlength=self.n_bins)

    def contract_era(self, W):
        """n_pre[era, b] = R contracted with cell weights W, split by SK era.
        Sums over era to contract(W) exactly (era is a disjoint partition)."""
        return np.bincount(self.R_eb, weights=self.Rv * W.ravel()[self.R_widx],
                           minlength=self.n_era * self.n_bins
                           ).reshape(self.n_era, self.n_bins)

    def contract_var_era(self, Wsq):
        """Per-era pre-detector BaseWeight^2 * W^2 sum (era, b)."""
        return np.bincount(self.S2_eb, weights=self.S2v * Wsq.ravel()[self.S2_widx],
                           minlength=self.n_era * self.n_bins
                           ).reshape(self.n_era, self.n_bins)

    def _escale_migrate(self, arr_e, deltas, var=False):
        """Per-era energy-scale reco-E migration of a (n_era, n_bins) array.
        Linear, rate-conserving within each (sample, reco-cz) column:
          N'(ie) = N(ie) + d*( N(ie-1)*[ie>0] - N(ie)*[ie<ne-1] ).
        var=True propagates BB variances (independent-bin squared coefficients)."""
        out = np.empty_like(arr_e)
        for e in range(self.n_era):
            d = deltas[e]
            N = arr_e[e]
            below = np.where(self.es_below >= 0, N[self.es_below], 0.0)   # N(ie-1)
            if not var:
                out[e] = N + d * (below * self.es_has_below - N * self.es_has_above)
            else:
                c_self = 1.0 - d * self.es_has_above
                c_below = d * self.es_has_below
                out[e] = c_self * c_self * N + c_below * c_below * below
        return out

    def _dir_smear_apply(self, vec, s):
        """Apply the reco-cz migration operator A = I + s*(M - I) to a (n_bins,)
        vector, block-diagonal per sample x reco-momentum row. E'_i = sum_j A[i,j] E_j;
        at s=1, E' = M @ E per zenith row. Identity (z1bins) samples are untouched."""
        out = vec.copy()
        for off, ne_, nz, M in self._ds_blocks:
            blk = vec[off:off + ne_ * nz].reshape(ne_, nz)
            sm = blk @ M.T                          # (M @ E_row) for every ie row
            out[off:off + ne_ * nz] = (blk + s * (sm - blk)).ravel()
        return out

    def _dir_smear_apply_T(self, vec, s):
        """Apply A^T = I + s*(M^T - I) block-diagonally (used to pull the likelihood
        residual back through the smearing for the OTHER dials' gradient)."""
        out = vec.copy()
        for off, ne_, nz, M in self._ds_blocks:
            blk = vec[off:off + ne_ * nz].reshape(ne_, nz)
            smT = blk @ M                           # (M^T @ r_row) = r_row @ M
            out[off:off + ne_ * nz] = (blk + s * (smT - blk)).ravel()
        return out

    def _era_theta(self, t, e):
        """Per-era view of the nuisance dict: era-split detector stems take their
        era-e dial value; everything else is shared. No-op for legacy specs.

        The up/down energy-scale set is era-split too, but its dials are not in
        DET_ERA_STEMS, so it is routed here into the base key 'updown_escale' that
        detector_factors reads (mirrors the DET_ERA_STEMS remap)."""
        if not self.det_split_stems and not self.active_ude:
            return t
        te = dict(t)
        tag = ERA_TAGS[e]
        for stem in self.det_split_stems:
            te[stem] = t[f"{stem}_{tag}"]
        if self.active_ude:
            te["updown_escale"] = t[f"updown_escale_{tag}"]
        return te

    # ---------------- detector factors ----------------
    def sample_rates(self, n_phys):
        """Weighted physics rate per sample (BaseWeight*PhysicsWeight sums)."""
        r = {}
        for s in self.samples:
            r[int(s)] = float(n_phys[self.bin_sample == s].sum())
        return r

    def detector_factors(self, t, rates, n_phys=None):
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
        S = {int(s): 1.0 for s in self.samples}
        dS = {n: {int(s): 0.0 for s in self.samples} for n in DET_NAMES}

        if self.migration_mode == "rawcount":
            counts_ = {int(k): float(v) for k, v in self.sample_counts.items()}

            def wsum(ids):
                return sum(counts_.get(i, 0.0) for i in np.atleast_1d(ids))
        else:
            def wsum(ids):
                return sum(rates.get(i, 0.0) for i in np.atleast_1d(ids))

        # Derivative convention: the event engine uses dW/W = diff_tune/tune per
        # event; for a donor sample with factor x that's 1/x, for an acceptor
        # with factor (1+r(1-x)) it's -r/(1+r(1-x)). We accumulate d(ln D).

        counts = {int(k): v for k, v in self.sample_counts.items()}

        # fcpc_separation — RAW counts (np.sum(mask)), not weighted rates
        x = t["fcpc_separation"]
        pc_ids, um_ids = [14, 15], [16, 17, 18]
        wfc = sum(c for s_, c in counts.items() if s_ not in pc_ids + um_ids)
        wpc = sum(counts.get(i, 0) for i in pc_ids)
        fc_ids = [int(s) for s in self.samples if int(s) not in pc_ids + um_ids]
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
        if not self.ntag_split:
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

        D = np.array([S[int(s)] for s in self.bin_sample])
        dlnD = {n: np.array([dS[n][int(s)] for s in self.bin_sample])
                for n in self.det_names}

        # momentum-resolved sub-GeV neutron tag: per-bin rate-conserving band
        # migrations (donor {20,22} 0-neutron -> acceptor {21,23} 1-neutron),
        # r computed per band from the per-bin physics rate. d ln D convention
        # matches apply2: donor 1/x, acceptor -r/(1+r(1-x)).
        if self.ntag_split:
            if n_phys is None:
                raise ValueError("ntag split needs per-bin n_phys")
            for name, (donor_m, acc_m) in self.ntag_bands.items():
                d = np.zeros(self.n_bins)
                x = t[name]
                if _unphys(x):
                    dlnD[name] = d
                    continue
                ra = float(n_phys[acc_m].sum())
                r_ = float(n_phys[donor_m].sum()) / ra if ra > 0 else 0.0
                acc = 1.0 + r_ * (1.0 - x)
                fac = np.ones(self.n_bins)
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
            D = D * np.where(self.fcmg_bin_mask, x, 1.0)
            dlnD["rel_norm_fcmg"] = np.where(self.fcmg_bin_mask,
                                             (1.0 / x if x != 0 else 0.0), 0.0)

        # up-mu background zenith x momentum SHAPE (SK cosmic-mu bkg subtraction,
        # thesis 5975-5984): multiplicative factor on the near-horizon affected model
        # bins, like rel_norm_fcmg (no separate bkg component). nominal x=1 (no-op),
        # d ln D/dx = 1/x on the masked bins -> rides the existing detector gradient.
        # Era-common (mask era-independent; gradient accumulates over eras).
        if self.active_upmu_bkg:
            for name in self.active_upmu_bkg:
                if name not in t:
                    continue
                x = t[name]
                mk = self.upmu_bkg_masks[name]
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
        if self.active_ude and "updown_escale" in t:
            d = t["updown_escale"]
            fac = 1.0 + d * self.ude_sign
            D = D * fac
            dlnD["updown_escale"] = np.divide(self.ude_sign, fac,
                                              out=np.zeros_like(fac), where=fac != 0)
        return D, dlnD

    # ---------------- expectation + chi2 ----------------
    def expectation(self, phi, theta, return_parts=False):
        """Full binned expectation (930) + variance, replicating the event chain.

        Era-aware: E_b = sum_era D_era[b] * n_pre_era[b]. Detector factors are
        evaluated per era (era-split dials take their era value; migration ratios
        use per-era rates). For a single-era response (n_era=1) this reduces
        exactly to the legacy n_pre * D path.
        """
        t = dict(zip(self.nuisance_names, theta))
        if self.solar_mix_f is None:
            W = self.cell_weights(phi, theta)
            Wd = None
            n_pre_e = self.contract_era(W)             # (n_era, n_bins)
            var_e = self.contract_var_era(W * W)       # (n_era, n_bins)
        else:
            # solar-mix pair: phi = (phi_solmin, phi_solmax). W is affine in phi
            # (NC classes constant), so W_era = W_a + f_era*(W_b - W_a) exactly.
            f = self.solar_mix_f[:, None]
            W = self.cell_weights(phi[0], theta)
            Wd = self.cell_weights(phi[1], theta) - W
            n_pre_e = self.contract_era(W) + f * self.contract_era(Wd)
            # var uses W_era^2: exact quadratic expansion in f.
            var_e = (self.contract_var_era(W * W)
                     + 2.0 * f * self.contract_var_era(W * Wd)
                     + f * f * self.contract_var_era(Wd * Wd))
        # energy-scale: bin-level reco-E migration of the pre-detector rates (so
        # detector factors ride on top). Linear + rate-conserving; no-op at x=1.
        n_pre0_es = es_deltas = None
        if self.active_energy_scale:
            es_deltas = np.array([t[f"energy_scale_{ERA_TAGS[e]}"] - 1.0
                                  for e in range(self.n_era)])
            n_pre0_es = n_pre_e                      # unmigrated (for the gradient)
            n_pre_e = self._escale_migrate(n_pre_e, es_deltas)
            var_e = self._escale_migrate(var_e, es_deltas, var=True)
        if self.migration_mode == "rawcount":
            # migration ratios are physics-independent raw counts; no phys rates
            nphys_e = [None] * self.n_era
        elif self.solar_mix_f is None:
            nphys_e = self.contract_era(self.cell_weights_physics_only(phi))
        else:
            Pa = self.cell_weights_physics_only(phi[0])
            Pd = self.cell_weights_physics_only(phi[1]) - Pa
            nphys_e = (self.contract_era(Pa)
                       + self.solar_mix_f[:, None] * self.contract_era(Pd))

        n_nu = np.zeros(self.n_bins)
        var = np.zeros(self.n_bins)
        parts_e = []
        for e in range(self.n_era):
            t_e = self._era_theta(t, e)
            np_e = None if self.migration_mode == "rawcount" else nphys_e[e]
            rates_e = None if np_e is None else self.sample_rates(np_e)
            D_e, dlnD_e = self.detector_factors(t_e, rates_e, n_phys=np_e)
            n_nu = n_nu + n_pre_e[e] * D_e
            var = var + var_e[e] * D_e * D_e
            if return_parts:
                parts_e.append(dict(D=D_e, dlnD=dlnD_e, n_pre=n_pre_e[e],
                                    rates=rates_e, n_phys=np_e))
        # direction-smearing (Track D): reco-cz migration of the FINAL reco expectation
        # E' = E + s*(M - I) @ E, applied POST-detector, POST-era-sum. M is era-common and
        # the migration is linear, so M(sum_e D_e n_pre_e) = sum_e M(D_e n_pre_e); and for
        # R2DS every active detector factor is per-sample-constant in zenith (and the
        # energy-scale migration is on the orthogonal reco-E axis), so M commutes with D
        # and this equals the per-era pre-detector application EXACTLY -- no ordering
        # approximation for R2DS (a zenith-varying detector dial, e.g. UDE/UBS, would add
        # one; R2DS excludes those). var is left unsmeared (production uses poisson; the
        # analytic gradient holds it fixed like the BB beta). s=0 -> exact no-op (guarded).
        ds_raw = None
        if self.active_dir_smear:
            s_ds = t[DIR_SMEAR_NAME]
            ds_raw = n_nu                          # pre-smear reco expectation (for grad)
            if s_ds != 0.0:
                n_nu = self._dir_smear_apply(n_nu, s_ds)
        if return_parts:
            out = dict(W=W, n_pre_e=n_pre_e, var_e=var_e, parts_e=parts_e)
            if ds_raw is not None:
                out.update(dir_smear_raw=ds_raw, dir_smear_s=t[DIR_SMEAR_NAME])
            if Wd is not None:
                out["W_delta"] = Wd
            if self.active_energy_scale:
                out.update(n_pre0_es=n_pre0_es, es_deltas=es_deltas)
            if self.n_era == 1:        # legacy keys for the single-era analytic grad
                p0 = parts_e[0]
                out.update(n_pre=n_pre_e[0], D=p0["D"], dlnD=p0["dlnD"],
                           rates=p0["rates"], n_phys=p0["n_phys"])
            return n_nu, var, out
        return n_nu, var

    def cell_weights_physics_only(self, phi):
        P = phi[self.cls_type, self.cls_flavor]
        return np.where(self.cls_cc[:, None, None] == 1, P, 1.0)

    @staticmethod
    def bb_chi2(obs, n_mod, var):
        """BarlowBeestonLikelihood.stats_only (BB-lite, no muons)."""
        tau = np.divide(var, n_mod ** 2, out=np.zeros_like(var), where=n_mod != 0)
        b = n_mod * tau - 1.0
        c = -obs * tau
        beta = 0.5 * (-b + np.sqrt(np.maximum(0, b * b - 4 * c)))
        beta = np.maximum(beta, 1e-9)
        beta_E = np.maximum(beta * n_mod, 1e-9)
        log_term = np.log(np.divide(obs, beta_E, out=np.ones_like(obs),
                                    where=beta_E > 0))
        log_term[obs == 0] = 0
        poisson = np.sum(2 * (beta_E - obs + obs * log_term))
        bb_pen = np.sum(np.divide((beta - 1) ** 2, tau, out=np.zeros_like(tau),
                                  where=tau > 0))
        return poisson + bb_pen, beta, tau

    @staticmethod
    def poisson_chi2(obs, n_mod):
        """Plain Poisson chi2 (event engine's no-MC-variance fallback form)."""
        if np.any(n_mod <= 0):
            return 9e9
        log_term = np.log(np.divide(obs, n_mod, out=np.ones_like(obs),
                                    where=n_mod > 0))
        log_term[obs == 0] = 0
        return float(2 * np.sum(n_mod - obs + obs * log_term))

    def chi2(self, phi, theta):
        n_nu, var = self.expectation(phi, theta)
        if self.likelihood == "poisson":
            stat = self.poisson_chi2(self.obs_f, n_nu[self.few])
        else:
            stat, _, _ = self.bb_chi2(self.obs_f, n_nu[self.few],
                                      var[self.few])
        pen = np.sum((theta - self.nominal) ** 2 / self.sigma ** 2)
        return stat + pen

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
        n_nu, var, parts = self.expectation(phi, theta, return_parts=True)
        m = self.few
        obs, E, V = self.obs_f, n_nu[m], var[m]
        pen = np.sum((theta - self.nominal) ** 2 / self.sigma ** 2)
        if self.likelihood == "poisson":
            stat = self.poisson_chi2(obs, E)
            if stat >= 9e9:                       # unphysical model region
                return stat + pen, 2 * (theta - self.nominal) / self.sigma ** 2
            f = stat + pen
            resid = 2 * (1 - obs / np.maximum(E, 1e-9))          # dchi2/dE_b
        else:
            stat, beta, tau = self.bb_chi2(obs, E, V)
            f = stat + pen
            beta_E = np.maximum(beta * E, 1e-9)
            resid = 2 * (1 - obs / beta_E) * beta                # dchi2/dE_b

        g = 2 * (theta - self.nominal) / self.sigma ** 2         # penalty grad
        t = dict(zip(self.nuisance_names, theta))
        W = parts["W"]
        parts_e = parts["parts_e"]
        D_stack = np.stack([pe["D"] for pe in parts_e])          # (n_era, n_bins)
        # energy-scale migration, held FIXED in the gradient (like migration r / BB
        # beta): flux & xsec grads ride through it (it is linear in n_pre), and the
        # dial's own grad is added below.
        es_deltas = parts.get("es_deltas")                       # (n_era,) or None

        # direction-smearing residual pullback: chi2 depends on E' = A @ E_raw
        # (A = I + s*(M - I)), so for EVERY other dial dChi2/dp = sum_b resid_b (A dE)_b
        # = sum_c (A^T resid)_c dE_c. When dir_smear is inactive or s=0 (A=I), acc(field)
        # is `sum(resid * field[m])` bit-for-bit (the pre-dir_smear code path). When
        # active, resid is pulled back through A^T over ALL bins (smearing leaks across
        # the FewEntries boundary). The dir_smear dial's OWN gradient is added at the end.
        ds_s = t[DIR_SMEAR_NAME] if self.active_dir_smear else 0.0
        if self.active_dir_smear and ds_s != 0.0:
            _resid_full = np.zeros(self.n_bins)
            _resid_full[m] = resid
            _resid_eff = self._dir_smear_apply_T(_resid_full, ds_s)

            def acc(field):
                return np.sum(_resid_eff * field)
        else:
            def acc(field):
                return np.sum(resid * field[m])

        # physics (flux/xsec) params live in the era-independent cell weights W:
        # dE_b = sum_era D_era[b] * migrate(contract_era(W * dlnW/dp))[era][b].
        def dE_phys(Wg):
            ce = self.contract_era(Wg)
            if es_deltas is not None:
                ce = self._escale_migrate(ce, es_deltas)
            return (D_stack * ce).sum(0)                         # (n_bins,)

        # solar-mix aware dE for a d-ln-W field g: dW_era/dp = (W + f_era*Wd)*g
        # (the dial fields are phi-independent). Wd is None on the single-phi path.
        Wd = parts.get("W_delta")

        def dE_W(gfield):
            if Wd is None:
                return dE_phys(W * gfield)
            ce = self.contract_era(W * gfield) \
                + self.solar_mix_f[:, None] * self.contract_era(Wd * gfield)
            if es_deltas is not None:
                ce = self._escale_migrate(ce, es_deltas)
            return (D_stack * ce).sum(0)                         # (n_bins,)

        # flux tunes: dW/W fields on cells -> dE_b = D * contract(W * g_field)
        # iterate the ACTIVE flux dials (zenith block depends on nuisance_spec)
        for name in self.flux_names:
            gfield = self._flux_dlnw(name, t)                    # (n_cls,nE,nZ) or None
            if gfield is None:
                continue
            dE = dE_W(gfield)
            g[self.nuisance_names.index(name)] += acc(dE)

        # xsec mask tunes: g = bit/x per class
        for name in XSEC_VECTOR_NAMES:
            i = self.nuisance_names.index(name)
            if name == "AxialMass":
                x = t[name]
                num = 0.042 * 1.05 * self.log10e                 # d/dx of (1+0.042(x-1)1.05 log10E)
                den = 1.0 + 0.042 * (x - 1.0) * 1.05 * self.log10e
                gf = np.where(self.cls_cc[:, None] == 1, num / den, 0.0)
                dE = dE_W(gf[:, :, None])
            else:
                x = t[name]
                j = MASK_TUNES.index(name)
                gcls = np.where(self.cls_bits[:, j], 1.0 / x, 0.0)
                dE = dE_W(gcls[:, None, None])
            g[i] += acc(dE)

        # optional flux ratios: per-class d ln W / d r, per-dial energy band.
        # heavy f_h=1+band(2r/(1+r)-1), light f_l=1+band(2/(1+r)-1). Generic over
        # the resolved registry (sub-GeV absorbers + energy-banded extensions).
        for name, (band, hv, lt) in self.fr_resolved.items():
            i = self.nuisance_names.index(name)
            r = t[name]
            dh = 2.0 / (1.0 + r) ** 2           # d/dr [2r/(1+r)]
            dl = -2.0 / (1.0 + r) ** 2          # d/dr [2/(1+r)]
            fh = 1.0 + band * (2.0 * r / (1.0 + r) - 1.0)
            fl = 1.0 + band * (2.0 / (1.0 + r) - 1.0)
            gh = band * dh / fh                 # (nE,)
            gl = band * dl / fl
            gfield = np.where(hv[:, None], gh[None, :], 0.0) \
                + np.where(lt[:, None], gl[None, :], 0.0)       # (n_cls,nE)
            dE = dE_W(gfield[:, :, None])
            g[i] += acc(dE)

        # optional sub-GeV xsec dials: d ln W / d r.
        if "xsec_ccqe_shape" in t and "xsec_ccqe_shape" in self.nuisance_names:
            i = self.nuisance_names.index("xsec_ccqe_shape")
            # W has CCQE *= E^(r-1); d ln/dr = ln(E) on CCQE cells.
            gf = np.where(self.ccqe_cls[:, None], np.log(self.e_c)[None, :], 0.0)
            dE = dE_W(gf[:, :, None])
            g[i] += acc(dE)
        # sub-GeV-localized CCQE shape: W has CCQE *= 1 + x*sh(E); d ln W/dx = sh/(1+x*sh).
        if "xsec_ccqe_shape_subgev" in t and "xsec_ccqe_shape_subgev" in self.nuisance_names:
            i = self.nuisance_names.index("xsec_ccqe_shape_subgev")
            sh = self.ccqe_shape_subgev
            fac = 1.0 + t["xsec_ccqe_shape_subgev"] * sh
            dlnw = np.divide(sh, fac, out=np.zeros_like(sh), where=fac != 0)   # (nE,)
            gf = np.where(self.ccqe_cls[:, None], dlnw[None, :], 0.0)
            dE = dE_W(gf[:, :, None])
            g[i] += acc(dE)
        # sub-GeV nu_e norm dials (lumped surrogate + 1p1h/2p2h split): d ln W/d r
        # = 1[e<1]/fac on the dial's class mask. Generic over SUBGEV_NUE_NORM.
        for _dial, _mattr in SUBGEV_NUE_NORM.items():
            if _dial in t and _dial in self.nuisance_names:
                i = self.nuisance_names.index(_dial)
                r = t[_dial]
                fac = 1.0 + self.e_below1 * (r - 1.0)              # (nE,)
                # d ln W / dr = 1[e<1] / fac; fac->0 as r->0 on sub-GeV cells
                # (where W itself ->0, so those cells contribute 0) — guard the
                # singular divide instead of emitting inf into the gradient.
                dlnw = np.divide(self.e_below1, fac, out=np.zeros_like(fac),
                                 where=fac != 0)
                gf = np.where(getattr(self, _mattr)[:, None], dlnw[None, :], 0.0)
                dE = dE_W(gf[:, :, None])
                g[i] += acc(dE)
        # multi-GeV CCQE flavor norms: d ln W/dr = 1[E>=1.33]/fac on the flavor mask
        # (mirrors the sub-GeV nu_e norm gradient with the complementary energy mask).
        for _dial, _mattr in MULTIGEV_CCQE_NORM.items():
            if _dial in t and _dial in self.nuisance_names:
                i = self.nuisance_names.index(_dial)
                r = t[_dial]
                fac = 1.0 + self.e_multigev * (r - 1.0)            # (nE,)
                dlnw = np.divide(self.e_multigev, fac, out=np.zeros_like(fac),
                                 where=fac != 0)
                gf = np.where(getattr(self, _mattr)[:, None], dlnw[None, :], 0.0)
                dE = dE_W(gf[:, :, None])
                g[i] += acc(dE)

        # detector tunes: dE_b = n_pre_era[b] * D_era[b] * dlnD_era[name][b]
        # (migration r fixed). Era-split stems route to their per-era dial; era-
        # independent detector dials (fiducial_volume, neutron-tag, ntag-split
        # bands) accumulate over eras. Reduces to the single-era loop at n_era==1.
        split_set = set(self.det_split_stems)
        if self.active_ude:            # 'updown_escale' base -> updown_escale_<era>
            split_set = split_set | {"updown_escale"}
        name_set = set(self.nuisance_names)
        for e in range(self.n_era):
            pe = parts_e[e]
            npD = pe["n_pre"] * pe["D"]
            for name, d in pe["dlnD"].items():
                if name in split_set:
                    idx = self.nuisance_names.index(f"{name}_{ERA_TAGS[e]}")
                elif name in name_set:
                    idx = self.nuisance_names.index(name)
                else:
                    continue
                g[idx] += acc(npD * d)

        # energy-scale dials: dN'_e/dx_e = N(ie-1)*[ie>0] - N(ie)*[ie<ne-1] on the
        # UNMIGRATED per-era rates (migration is linear in delta=x-1, so this is exact
        # and delta-independent), propagated through the detector factor D_e.
        if self.active_energy_scale:
            n_pre0_es = parts["n_pre0_es"]
            for e in range(self.n_era):
                N = n_pre0_es[e]
                below = np.where(self.es_below >= 0, N[self.es_below], 0.0)
                dN = below * self.es_has_below - N * self.es_has_above
                dE = parts_e[e]["D"] * dN
                g[self._es_idx[e]] += acc(dE)

        # direction-smearing OWN gradient: n_nu = E_raw + s*((M - I) @ E_raw), so
        # dn_nu/ds = (M - I) @ E_raw (constant in s -> exact, delta-independent), and
        # dChi2/ds = sum_few resid_b * ((M - I) @ E_raw)_b. E_raw is the pre-smear reco
        # expectation captured in expectation(). Added even at s=0 (the gradient is
        # nonzero there); the other dials' grads above are unchanged at s=0 (acc no-op).
        if self.active_dir_smear:
            e_raw = parts["dir_smear_raw"]
            dE_ds = self._dir_smear_apply(e_raw, 1.0) - e_raw    # (M - I) @ E_raw
            g[self.nuisance_names.index(DIR_SMEAR_NAME)] += np.sum(resid * dE_ds[m])

        return f, g

    def _flux_dlnw(self, name, t):
        if name == "normalization_below1GeV":
            gf = np.where(self.e_below1, 1.0 / t[name], 0.0)
            return np.broadcast_to(gf[None, :, None],
                                   (self.n_cls, self.nE, self.nZ))
        if name == "normalization_above1GeV":
            gf = np.where(self.e_above1, 1.0 / t[name], 0.0)
            return np.broadcast_to(gf[None, :, None],
                                   (self.n_cls, self.nE, self.nZ))
        if name == "tilt":
            gf = np.log(self.e_c / 10.0)
            return np.broadcast_to(gf[None, :, None],
                                   (self.n_cls, self.nE, self.nZ))
        if name == "nunubar_ratio":
            gcls = np.where(self.cls_pdg < 0, 1.0 / t[name], 0.0)
            return np.broadcast_to(gcls[:, None, None],
                                   (self.n_cls, self.nE, self.nZ))
        if name == "flavor_ratio":
            gcls = np.where(np.abs(self.cls_pdg) == 12, 1.0 / t[name], 0.0)
            return np.broadcast_to(gcls[:, None, None],
                                   (self.n_cls, self.nE, self.nZ))
        if name == "barr_zenith":
            x = t[name]
            r = 1.0 + self.barr_env * x
            gf = self.tanh3z[None, :] * (self.barr_env / r)[:, None]  # (nE,nZ)
            return np.broadcast_to(gf[None, :, :],
                                   (self.n_cls, self.nE, self.nZ))
        if name == "zenith_up":                          # w = 1 - x*tanh^2, z<0
            w = 1.0 - t[name] * self.tanhz2
            gf = np.where(self.z_c < 0, -self.tanhz2 / w, 0.0)        # (nZ,)
            return np.broadcast_to(gf[None, None, :],
                                   (self.n_cls, self.nE, self.nZ))
        if name == "zenith_down":                        # w = 1 - x*tanh^2, z>=0
            w = 1.0 - t[name] * self.tanhz2
            gf = np.where(self.z_c >= 0, -self.tanhz2 / w, 0.0)       # (nZ,)
            return np.broadcast_to(gf[None, None, :],
                                   (self.n_cls, self.nE, self.nZ))
        if name == "flux_horizvert":                 # w = 1 + x*g(cz), g=(1-3cz^2)/2
            w = 1.0 + t[name] * self.horizvert_shape
            gf = self.horizvert_shape / w                             # (nZ,)
            return np.broadcast_to(gf[None, None, :],
                                   (self.n_cls, self.nE, self.nZ))
        if name == "solar_activity":                     # w = 1 - x*A*exp(-E/L)
            s = SOLAR_AMP * np.exp(-self.e_c / SOLAR_SCALE)           # (nE,)
            gf = -s / (1.0 - t[name] * s)                            # d ln w / dx, (nE,)
            return np.broadcast_to(gf[None, :, None],
                                   (self.n_cls, self.nE, self.nZ))
        if name == "kpi_ratio":                          # w = 1 + x*kpi_shape(E)
            gf = self.kpi_shape / (1.0 + t[name] * self.kpi_shape)   # d ln w / dx, (nE,)
            return np.broadcast_to(gf[None, :, None],
                                   (self.n_cls, self.nE, self.nZ))
        return None

    # ---------------- per-point fit (worker protocol) ----------------
    def fit_point(self, phi_dcp_stack, x0=None, n_dcp=None, free_mask=None,
                  jac=None, dcp_warmchain=True):
        """dCP-profiled nuisance minimization — run_one_point protocol.

        phi_dcp_stack: phi[n_dcp, 2, 3, nE, nZ]. Returns same tuple as the
        event worker: (chi2, best_dcp_index, nuisance, nit, converged).

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
        nominal = self.nominal.copy()
        if x0 is None:
            x0 = nominal
        lower = nominal - 10 * self.sigma
        upper = nominal + 10 * self.sigma
        lower[(nominal > 0) & (lower < 0.01)] = 0.01
        # box bounds for optional dials (truncation limits): sub-GeV absorbers,
        # energy-banded flux ratios ([0.3,1.7]), and sub-GeV xsec dials.
        _box = dict(FLUX_RATIO_BOX)
        _box.update({n: (0.3, 1.7) for n in FLUX_BAND_NAMES})
        _box.update(XSEC_EXTRA_BOX)
        _box.update(MULTIGEV_CCQE_BOX)           # multi-GeV CCQE flavor norms [0,3]
        _box[DIR_SMEAR_NAME] = DIR_SMEAR_BOX     # one-sided [0,1] (nominal-0 dial)
        for name, (lo, hi) in _box.items():
            if name in self.nuisance_names:
                k = self.nuisance_names.index(name)
                lower[k], upper[k] = lo, hi
        if free_mask is not None:
            fixed = ~np.asarray(free_mask, bool)
            lower[fixed] = nominal[fixed]
            upper[fixed] = nominal[fixed]
            x0 = np.where(fixed, nominal, x0)
        bounds = list(zip(lower, upper))

        use_jac = True if jac is None else jac
        if self.solar_mix_f is not None:
            # solar-mix mode: phi_dcp_stack is the PAIR (stack_solmin, stack_solmax)
            stack_a, stack_b = phi_dcp_stack
            n = stack_a.shape[0] if n_dcp is None else n_dcp
        else:
            n = phi_dcp_stack.shape[0] if n_dcp is None else n_dcp
        best = (np.inf, 0, x0, 0, False)
        x_seed = x0                       # node 0 from x0; warm-chained thereafter
        for di in range(n):
            if self.solar_mix_f is not None:
                phi = (stack_a[di].astype(float), stack_b[di].astype(float))
            else:
                phi = phi_dcp_stack[di].astype(float)
            # tolerance scaling from stat-only chi2 at the current seed
            n_nu, var = self.expectation(phi, x_seed)
            if self.likelihood == "poisson":
                chi2_stat = self.poisson_chi2(self.obs_f, n_nu[self.few])
            else:
                chi2_stat, _, _ = self.bb_chi2(self.obs_f, n_nu[self.few],
                                               var[self.few])
            tol = max(1e-5, np.sqrt(max(min(chi2_stat, 1e7), 0)) * 1e-5)
            if use_jac:
                res = minimize(lambda th: self.chi2_and_grad(phi, th), x_seed,
                               method="L-BFGS-B", jac=True, bounds=bounds,
                               options={"ftol": tol, "gtol": 1e-5, "maxiter": 200})
            else:
                res = minimize(lambda th: self.chi2(phi, th), x_seed,
                               method="L-BFGS-B", bounds=bounds,
                               options={"ftol": tol, "gtol": 1e-5, "maxiter": 200})
            if res.fun < best[0]:
                best = (res.fun, di, res.x.copy(), res.nit, res.success)
            if dcp_warmchain:
                x_seed = best[2]          # next node warm-starts from the best basin
        return best

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
        if self.likelihood != "bb":
            raise ValueError("per_bin_report requires likelihood='bb'")
        n_nu, var = self.expectation(phi, theta)
        m = self.few
        obs = self.obs_f
        E = n_nu[m]
        V = var[m]
        stat, beta, tau = self.bb_chi2(obs, E, V)
        beta_E = np.maximum(beta * E, 1e-9)
        log_term = np.log(np.divide(obs, beta_E, out=np.ones_like(obs),
                                    where=beta_E > 0))
        log_term[obs == 0] = 0
        poisson_b = 2.0 * (beta_E - obs + obs * log_term)
        bbpen_b = np.divide((beta - 1.0) ** 2, tau,
                            out=np.zeros_like(tau), where=tau > 0)
        chi2_b = poisson_b + bbpen_b
        pull = np.sign(obs - beta_E) * np.sqrt(np.maximum(chi2_b, 0.0))
        sigma_eff = np.sqrt(np.maximum(E + V, 1e-300))
        resid_std = (obs - E) / sigma_eff
        return dict(
            bin_index=np.nonzero(m)[0].astype(int),
            sample=self.bin_sample[m].astype(int),
            obs=obs.astype(float), model=E.astype(float),
            beta=beta.astype(float), beta_model=beta_E.astype(float),
            var=V.astype(float), sigma_mc=np.sqrt(V).astype(float),
            tau=tau.astype(float),
            chi2_bin=chi2_b.astype(float),
            poisson_bin=poisson_b.astype(float),
            bbpen_bin=bbpen_b.astype(float),
            pull=pull.astype(float),
            resid_std=resid_std.astype(float),
            stat_total=float(stat),
        )
