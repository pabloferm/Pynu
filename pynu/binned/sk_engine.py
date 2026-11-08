"""Super-Kamiokande binned forward model: response matrix x oscillation x systematics.

Consumes sk_response_*.npz and the oscillation tensors. Weights that depend on the
true energy and zenith (oscillation, flux parameters, AxialMass) are evaluated at the
centres of the true-energy and zenith cells.

Model at one lattice node:
  W[k,cE,cZ]    = Phi_or_1 * F_flux(cell, pdg_k) * A_axial(cell, CC_k) * X_k(mask bits)
  n_pre[era,b]  = sum_nz R_v * W[R_k,R_e,R_z], scattered to (R_era, R_b)
  E[b]          = sum_era D_era[b](theta_det; r from physics-only rates) * n_pre'[era,b]
  chi2          = 2 sum_b (E - O + O ln(O/E)) + sum_j (x_j - mu_j)^2 / sigma_j^2
where n_pre' is n_pre after the energy-scale migration. NC classes get Phi = 1. There
is no muon background term.
"""
import json

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

# Cross-section parameters. MASK_TUNES is the order of the response's class bits.
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

# SK run periods: 0=SK-I, 1=SK-II, 2=SK-III, 3=SK-IV+V.
ERA_TAGS = ["sk1", "sk2", "sk3", "sk45"]
# Detector stems with one parameter per era, <stem>_<tag>.
DET_ERA_STEMS = ["pc_reduction", "mge_nonubkg", "fc_reduction",
                 "subgev_2ring_pi0", "subgev_1ring_pi0",
                 "multiring_nunubar_separation", "multiring_emu_separation",
                 "multiring_eother_separation", "pc_stopthru_separation",
                 "pi0_ring_separation", "e_ring_separation", "mu_ring_separation",
                 "singlering_pid", "multiring_pid", "upmu_shower_separation",
                 "upmu_stop_bkg", "upmu_showering_bkg", "upmu_nonshowering_bkg"]

CORE_FLUX_NAMES = ["normalization_below1GeV", "normalization_above1GeV",
                   "tilt", "flavor_ratio"]
ZENITH_DIALS = ["zenith_up", "zenith_down"]
# solar_activity: w = 1 - x*A*exp(-E_true/L).
SOLAR_AMP = 0.08    # amplitude
SOLAR_SCALE = 3.0   # GeV, decay scale
# kpi_ratio: K/pi production ratio, a flavour-blind high-energy flux norm growing above
# the kaon onset, flux *= 1 + x*max(0, log10(E/KPI_E0)).
KPI_E0 = 3.0        # GeV
# flux_horizvert: horizontal/vertical flux ratio, flux *= 1 + x*g(cz), g = (1-3cz^2)/2.
# g is mean-zero over cosz: +0.5 horizontal, -1.0 vertical.
OPTIONAL_FLUX_NAMES = ["solar_activity", "kpi_ratio", "flux_horizvert"]
ALL_FLUX_NAMES = CORE_FLUX_NAMES + ZENITH_DIALS + OPTIONAL_FLUX_NAMES

# FC <-> PC separation: the FC side is one sample, sk1-3_fc_multigev_1ring_mulike.
FCPC_DONOR_SAMPLES = frozenset({9})

# Energy scale per era. The migration acts at the histogram level: within each
# (sample, reco-cz) column along the reco-E index ie, with delta = x - 1,
#   N'(ie) = N(ie) + delta*( N(ie-1)*[ie>0] - N(ie)*[ie<ne-1] )
# which conserves sum_ie N. x = 1 is a no-op.
ENERGY_SCALE_NAMES = [f"energy_scale_{tag}" for tag in ERA_TAGS]

# Energy-banded flux ratios: name -> (band, heavy leg, light leg). Bands in true energy:
# sub E<1 GeV, mid 1-10 GeV, high >10 GeV. The legs scale as heavy *= r(1+q)/(1+qr),
# light *= (1+q)/(1+qr), with q(E) the heavy/light flux ratio, which preserves the band
# total in each true-energy cell. r = 1 is a no-op.
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

# Energy-dependent CCQE parameters.
#   xsec_ccqe_shape          CCQE *= E_true^(r-1)
#   xsec_ccqe_shape_subgev   CCQE *= 1 + x*1[E<1.33]*(ln E - <ln E>_subgev), a log-E tilt
#                            confined to the sub-GeV cells and mean-zero over them
#   xsec_ccqe_shape_multigev the same on E >= 1.33 GeV, centred on the multi-GeV cells
#   xsec_1p1h_subgev_nue     norm of nu_e + nubar_e CCQE below 1 GeV
#   xsec_ccqe_multigev_*     norm of CCQE above 1.33 GeV, per flavour
CCQE_SHAPE_SUBGEV_E = 1.33   # GeV, SK sub-GeV/multi-GeV boundary
XSEC_CCQE_SHAPE_MULTIGEV_NAME = "xsec_ccqe_shape_multigev"
MULTIGEV_CCQE_NAMES = ["xsec_ccqe_multigev_nue", "xsec_ccqe_multigev_numu"]
MULTIGEV_CCQE_NORM = {"xsec_ccqe_multigev_nue": "ccqe_nue_cls",
                      "xsec_ccqe_multigev_numu": "ccqe_numu_cls"}

# rel_norm_fcmg: flat norm on the FC multi-GeV samples: SK1-3 1-ring {7,8,9} and
# multi-ring {10,11,12,13}, SK4-5 1-ring {24,25,26,27,28}.
REL_NORM_FCMG_SAMPLES = frozenset({7, 8, 9, 10, 11, 12, 13, 24, 25, 26, 27, 28})

# upmu_norm: flat norm on all up-mu samples.
UPMU_NORM_NAME = "upmu_norm"
UPMU_NORM_SAMPLES = frozenset({16, 17, 18})

# 0n/1n neutron migration on the two sample pairs with an explicit neutron label:
# name -> (donor 0n sample, acceptor 1n sample). A value x moves the 0n leg to
# x*rate(0n) and adds (1-x)*rate(0n) to the 1n leg. Both legs stay non-negative for
# x in [0, 1 + 1/r], r = counts(0n)/counts(1n) from the raw MC event counts.
NEUTRON_MIG_PAIRS = {
    "nmig_subgev_nuebar_pinned":   (20, 21),
    "nmig_multigev_nuebar_pinned": (25, 26),
}
NEUTRON_MIG_RAWCOUNTS = {20: 294634, 21: 112964, 25: 43858, 26: 38369}
NEUTRON_MIG_BOX_PINNED = {
    n: (0.0, 1.0 + NEUTRON_MIG_RAWCOUNTS[a] / NEUTRON_MIG_RAWCOUNTS[d])
    for n, (d, a) in NEUTRON_MIG_PAIRS.items()
}

# Per-era decay-e tagging: name -> era it acts in, and its migration blocks.
#   ("pair2", acceptor ids, donor ids): acceptor *= x, donor *= 1+r(1-x),
#       r = W(acceptor)/W(donor), from that era's weighted rates.
#   ("cascade3", (s0,s1,s2)): three-state mu-like decay-count cascade.
DECAY_E_TAG_NAMES = ["decay_e_tagging_sk1", "decay_e_tagging_sk2",
                     "decay_e_tagging_sk3", "decay_e_tagging_sk45"]
DECAY_E_TAG_ERA = {"decay_e_tagging_sk1": 0, "decay_e_tagging_sk2": 1,
                   "decay_e_tagging_sk3": 2, "decay_e_tagging_sk45": 3}
DECAY_E_TAG_BLOCKS = {
    "decay_e_tagging_sk1": [("pair2", (1,), (0,)),      # sub-GeV e-like
                            ("pair2", (7,), (8,)),      # multi-GeV e-like
                            ("cascade3", (3, 4, 5))],   # sub-GeV mu-like
    "decay_e_tagging_sk2": [("pair2", (1,), (0,)),
                            ("pair2", (7,), (8,)),
                            ("cascade3", (3, 4, 5))],
    "decay_e_tagging_sk3": [("pair2", (1,), (0,)),
                            ("pair2", (7,), (8,)),
                            ("cascade3", (3, 4, 5))],
    "decay_e_tagging_sk45": [("pair2", (19,), (20, 21)),  # sub-GeV e-like
                             ("pair2", (24,), (25, 26)),  # multi-GeV e-like
                             ("pair2", (22,), (23,)),     # sub-GeV mu pair
                             ("pair2", (27,), (28,))],    # multi-GeV mu pair
}

# Up-mu cosmic-background shape: a multiplicative factor on the near-horizon bins of the
# stopping sample (16), by momentum, and of the non-showering through-going sample (17).
# name -> (sample id, reco-cosZ bin indices, reco-momentum bin indices or None = all).
# Up-mu reco-cosZ covers [-1, 0] in 10 bins, so iz = 9 is the bin nearest the horizon.
UPMU_BKG_SHAPE_NAMES = ["upmu_stop_bkg_horiz_lowp", "upmu_stop_bkg_horiz_highp",
                        "upmu_nonshow_bkg_horiz"]
UPMU_BKG_SHAPE_SPEC = {
    "upmu_stop_bkg_horiz_lowp":  (16, (8, 9), (0,)),
    "upmu_stop_bkg_horiz_highp": (16, (8, 9), (1, 2)),
    "upmu_nonshow_bkg_horiz":    (17, (9,), None),
}

# Up/down energy scale per era: anti-symmetric norm of FC and PC events, up-going bins
# *= 1+d, down-going *= 1-d. Reco-cosZ is split at 0, so in the 10-bin samples iz < 5 is
# up-going. Up-mu samples and samples with a single reco-zenith bin keep factor 1.
UPDOWN_ESCALE_NAMES = [f"updown_escale_{tag}" for tag in ERA_TAGS]
UPDOWN_ESCALE_EXCLUDE = frozenset({16, 17, 18})

# Low-momentum e/mu single-ring migration for SK I-III, in the two lowest reco-momentum
# bins (ie < 2): (donor e-like ids, acceptor mu-like ids).
ELIKE_RINGCOUNT_LOWP_NAME = "elike_ringcount_lowp_sk13"
ELIKE_RINGCOUNT_LOWP_IDS = ((0, 1), (3, 4, 5))
ELIKE_RINGCOUNT_LOWP_PSPLIT = 2

# Nominal value of every parameter the engine can apply.
_ZERO_NOMINAL = (["tilt", "xsec_ccqe_shape_subgev", XSEC_CCQE_SHAPE_MULTIGEV_NAME]
                 + ZENITH_DIALS + OPTIONAL_FLUX_NAMES + UPDOWN_ESCALE_NAMES)
_ONE_NOMINAL = (["normalization_below1GeV", "normalization_above1GeV", "flavor_ratio",
                 "fcpc_separation", "fiducial_volume", "neutron_tagging_subgev",
                 "neutron_tagging_multigev", "xsec_ccqe_shape", "xsec_1p1h_subgev_nue",
                 "rel_norm_fcmg", UPMU_NORM_NAME, ELIKE_RINGCOUNT_LOWP_NAME]
                + XSEC_VECTOR_NAMES + ENERGY_SCALE_NAMES + list(FLUX_RATIO_SPEC)
                + MULTIGEV_CCQE_NAMES + list(NEUTRON_MIG_PAIRS) + DECAY_E_TAG_NAMES
                + UPMU_BKG_SHAPE_NAMES
                + [f"{stem}_{tag}" for stem in DET_ERA_STEMS for tag in ERA_TAGS])
CANONICAL_NOMINAL = {**{n: 0.0 for n in _ZERO_NOMINAL}, **{n: 1.0 for n in _ONE_NOMINAL}}

# Fixed fit boxes; every other parameter gets nominal +- 10 sigma (see fit_box).
FIT_BOX = {n: (0.3, 1.7) for n in FLUX_RATIO_SPEC}
FIT_BOX.update({"xsec_ccqe_shape": (0.3, 1.7), "xsec_1p1h_subgev_nue": (0.5, 1.5),
                "xsec_ccqe_shape_subgev": (-2.0, 2.0),
                XSEC_CCQE_SHAPE_MULTIGEV_NAME: (-2.0, 2.0)})
FIT_BOX.update({n: (0.0, 3.0) for n in MULTIGEV_CCQE_NAMES})
FIT_BOX.update(NEUTRON_MIG_BOX_PINNED)


def resolve_nuisance_spec(spec):
    """Return (names, nominal) for an explicit ordered list of parameter names."""
    names = list(spec)
    missing = [n for n in names if n not in CANONICAL_NOMINAL]
    if missing:
        raise ValueError(f"nuisance_spec has unsupported dials {missing}")
    nominal = np.array([CANONICAL_NOMINAL[n] for n in names])
    return names, nominal


def _unphys(x):
    """Whether a parameter value is outside the physical range [0, 9999999].

    Zero is inside it: parameters with a box lower bound of exactly 0 can reach it, and
    the 1/x derivative folds guard zero at the division.
    """
    return x < 0 or x > 9999999


class SKBinnedEngine:
    """Expected counts, chi^2 and analytic gradient on the SK analysis bins.

    Migration ratios of the rate-conserving detector parameters are physics-weighted
    rates, recomputed at every oscillation point; fcpc_separation uses raw MC counts.
    The likelihood is the saturated Poisson chi^2 over all analysis bins plus a Gaussian
    prior on every parameter.
    """

    def __init__(self, response_path, nuisance_spec, *, fluxq, fit_ftol, fit_maxiter):
        self.fit_ftol, self.fit_maxiter = fit_ftol, fit_maxiter

        self.nuisance_names, self.nominal = resolve_nuisance_spec(nuisance_spec)
        # prior widths: the caller sets them from the configuration (arms.SKArm)
        self.sigma = np.ones(self.nominal.size)
        self.flux_names = [n for n in self.nuisance_names
                           if n in ALL_FLUX_NAMES]
        self.active_flux_ratios = [n for n in self.nuisance_names
                                   if n in FLUX_RATIO_SPEC]
        self.active_upmu_bkg = [n for n in self.nuisance_names
                                if n in UPMU_BKG_SHAPE_NAMES]
        self.active_ude = [n for n in self.nuisance_names
                           if n in UPDOWN_ESCALE_NAMES]
        self.active_neutron_mig = [n for n in self.nuisance_names
                                   if n in NEUTRON_MIG_PAIRS]
        self.active_decay_e_tag = [n for n in self.nuisance_names
                                   if n in DECAY_E_TAG_NAMES]
        self.active_energy_scale = [n for n in self.nuisance_names
                                    if n in ENERGY_SCALE_NAMES]
        active_xsec = [n for n in self.nuisance_names if n in XSEC_VECTOR_NAMES]
        # the response bakes in exactly these cross-section classes and detector samples
        if set(active_xsec) != set(XSEC_VECTOR_NAMES):
            raise ValueError("xsec dial set must match the response build "
                             f"(need {XSEC_VECTOR_NAMES}); got {active_xsec}")
        # each detector stem must be covered era-independently (its base name) or
        # era-split (all 4 <stem>_<era> parameters). det_split_stems drives _era_theta.
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
        required_det = set(DET_NAMES)
        if covered != required_det:
            raise ValueError("detector dial set must cover the response build's "
                             f"stems {sorted(required_det)} (era-split or indep); "
                             f"got split={sorted(self.det_split_stems)} "
                             f"indep={sorted(indep_det)}")
        # base detector stems for detector_factors and the gradient (DET_NAMES order)
        self.det_names = [n for n in DET_NAMES if n in covered]
        _missing_flux = set(ALL_FLUX_NAMES) - set(self.flux_names)
        if _missing_flux:
            raise ValueError(f"flux dials are required; missing {sorted(_missing_flux)}")
        if len(set(self.nuisance_names)) != len(self.nuisance_names):
            raise ValueError("duplicate nuisance names in spec")
        z = np.load(response_path, allow_pickle=False)
        Rk, Re, Rz = z["R_k"], z["R_e"], z["R_z"]
        Rb, Rv = z["R_b"], z["R_v"]
        self.classes = z["classes"]                       # (n_cls, 2+15)
        self.xsec_tune_names = [str(s) for s in z["xsec_tune_names"]]
        assert self.xsec_tune_names == MASK_TUNES
        self.e_edges, self.z_edges = z["e_edges"], z["z_edges"]
        self.n_bins = int(z["n_bins"])
        self.observed = z["observed"]
        self.sample_table = json.loads(str(z["sample_table"]))   # s -> (off, ne, nz)
        self.sample_counts = json.loads(str(z["sample_event_counts"]))

        self.n_cls = self.classes.shape[0]
        self.nE = self.e_edges.size - 1
        self.nZ = self.z_edges.size - 1
        # cell centres: geometric in E, arithmetic in cz
        self.e_c = np.sqrt(self.e_edges[:-1] * self.e_edges[1:])
        self.z_c = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])

        # class decomposition
        self.cls_pdg = self.classes[:, 0].astype(int)
        self.cls_cc = self.classes[:, 1].astype(int)
        self.cls_bits = self.classes[:, 2:].astype(bool)         # (n_cls, 15)
        # flavor / type index into phi[type, flavor, E, Z]. Clipped because NC classes
        # may carry pdg = 0; their gathered value is discarded (NC classes get P = 1).
        self.cls_flavor = np.clip(np.abs(self.cls_pdg) // 2 - 6, 0, 2)  # 12->0,14->1,16->2
        self.cls_type = (self.cls_pdg < 0).astype(int)           # nu=0, nubar=1

        # flatten (k, e, z) -> single gather index
        R_widx = (Rk.astype(np.int64) * self.nE + Re) * self.nZ + Rz

        # per-bin -> sample id map
        self.bin_sample = np.empty(self.n_bins, dtype=int)
        for s, (off, ne, nz) in self.sample_table.items():
            self.bin_sample[off:off + ne * nz] = int(s)
        self.samples = np.unique(self.bin_sample)
        self.fcmg_bin_mask = np.isin(self.bin_sample, list(REL_NORM_FCMG_SAMPLES))
        self.upmu_norm_bin_mask = np.isin(self.bin_sample, list(UPMU_NORM_SAMPLES))
        self.neutron_mig_pairs = dict(NEUTRON_MIG_PAIRS)

        # era partition of the response
        self.n_era = int(z["n_era"])
        R_era = z["R_era"].astype(np.int64)
        # flattened (era, bin) index for one-pass per-era contraction
        R_eb = R_era * self.n_bins + Rb
        if self.det_split_stems and self.n_era != len(ERA_TAGS):
            raise ValueError(f"spec era-splits {self.det_split_stems} but response "
                             f"has n_era={self.n_era} (need {len(ERA_TAGS)})")

        self._R_csr = self._build_csr(Rv, R_widx, R_eb)

        # energy-scale migration geometry (reco-E adjacency). es_below[b] = the bin one
        # reco-E step lower in the same (sample, reco-cz) column (-1 at ie=0);
        # es_has_above[b] = 1 if b can spill upward (ie < ne-1).
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

        # flux-ratio per-class flavor/sign selectors
        self._fr_leg = {
            "nuebar": (self.cls_pdg == -12), "nue": (self.cls_pdg == 12),
            "e": (np.abs(self.cls_pdg) == 12), "mu": (np.abs(self.cls_pdg) == 14),
            "numubar": (self.cls_pdg == -14), "numu": (self.cls_pdg == 14),
        }
        # up-mu background shape per-bin masks. Bin index = off + ie*nz + iz.
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

        # low-momentum e/mu ring-count migration per-bin masks (donor, acceptor)
        def _lowp_band(sample_ids, psplit):
            mk = np.zeros(self.n_bins, dtype=bool)
            for s in sample_ids:
                key = str(s) if str(s) in self.sample_table else s
                if key not in self.sample_table:
                    continue
                off, ne, nz = self.sample_table[key]
                idx = np.arange(off, off + ne * nz)
                ie = (idx - off) // nz
                mk[idx[ie < psplit]] = True
            return mk
        _don_ids, _acc_ids = ELIKE_RINGCOUNT_LOWP_IDS
        self.elike_ringcount_lowp_masks = (
            _lowp_band(_don_ids, ELIKE_RINGCOUNT_LOWP_PSPLIT),
            _lowp_band(_acc_ids, ELIKE_RINGCOUNT_LOWP_PSPLIT))

        # up/down energy-scale signed per-bin mask: +1 on up-going bins, -1 on down-going,
        # 0 on excluded samples. Era-independent geometry; _era_theta routes the per-era
        # value to detector_factors.
        if set(self.active_ude) != set(UPDOWN_ESCALE_NAMES):
            raise ValueError("up/down energy-scale needs all "
                             f"{len(ERA_TAGS)} era dials {UPDOWN_ESCALE_NAMES}; "
                             f"got {self.active_ude}")
        if self.n_era != len(ERA_TAGS):
            raise ValueError("up/down energy-scale dials need the "
                             f"{len(ERA_TAGS)}-era response (n_era={self.n_era})")
        self.ude_sign = np.zeros(self.n_bins)
        for s, (off, ne_, nz) in self.sample_table.items():
            if int(s) in UPDOWN_ESCALE_EXCLUDE or nz != 10:
                continue
            for ie in range(ne_):
                for iz in range(nz):
                    self.ude_sign[off + ie * nz + iz] = \
                        1.0 if iz < nz // 2 else -1.0

        # all analysis bins are fitted
        self.few = np.ones(self.observed.shape, dtype=bool)
        self.obs_f = self.observed[self.few]

        # static flux cell masks
        self.e_below1 = self.e_c < 1.0
        self.e_above1 = self.e_c > 1.0
        self.tanhz2 = np.tanh(self.z_c) ** 2          # zenith_up/down envelope
        # horizontal/vertical shape: mean-zero over cosz, +0.5 horizontal, -1.0 vertical
        self.horizvert_shape = 0.5 * (1.0 - 3.0 * self.z_c ** 2)
        self.log10e = np.log10(self.e_c)
        # K/pi flux ramp: 0 below KPI_E0, rising as log10(E/KPI_E0) above (nE,)
        self.kpi_shape = np.maximum(0.0, np.log10(self.e_c / KPI_E0))

        # true-energy bands of the banded flux ratios
        self.e_bands = {"sub": (self.e_c < 1.0).astype(float),
                        "mid": ((self.e_c >= 1.0) & (self.e_c < 10.0)).astype(float),
                        "high": (self.e_c >= 10.0).astype(float)}
        # each flux ratio -> (band envelope (nE,), heavy (n_cls,), light (n_cls,)), and its
        # q(E) vector from the flux-model table, matched to the family by name prefix
        self.fr_resolved = {}
        self.fr_q = {}
        _fluxq = dict(np.load(fluxq))
        _FAM_KEY = {"flux_flavor": "q_flavor", "flux_nuebar": "q_nuebar",
                    "flux_numubar": "q_numubar"}
        for nm in self.active_flux_ratios:
            band, hv, lt = FLUX_RATIO_SPEC[nm]
            self.fr_resolved[nm] = (self.e_bands[band],
                                    self._fr_leg[hv], self._fr_leg[lt])
            _fam = next(v for k, v in _FAM_KEY.items() if nm.startswith(k))
            q = np.asarray(_fluxq[_fam], float)
            assert q.size == self.nE, f"fluxq {_fam} size {q.size} != nE"
            self.fr_q[nm] = q                       # per-E-cell vector
        # CCQE class masks and energy-dependent shapes
        self.ccqe_cls = self.cls_bits[:, MASK_TUNES.index("CCQE")]   # (n_cls,)
        self.ccqe_nue_cls = self.ccqe_cls & (self.cls_flavor == 0)   # nu_e + nubar_e CCQE
        self.ccqe_numu_cls = self.ccqe_cls & (self.cls_flavor == 1)  # nu_mu + nubar_mu CCQE
        self.e_multigev = self.e_c >= CCQE_SHAPE_SUBGEV_E            # (nE,)
        # sub-GeV CCQE shape: mean-zero log-E tilt on the E_true < 1.33 GeV cells, 0 above
        self.ccqe_shape_subgev = np.where(self.e_c < CCQE_SHAPE_SUBGEV_E,
                                          np.log(self.e_c), 0.0)            # (nE,)
        _sub = self.e_c < CCQE_SHAPE_SUBGEV_E
        self.ccqe_shape_subgev[_sub] -= self.ccqe_shape_subgev[_sub].mean()
        # multi-GeV CCQE shape: mean-zero log-E tilt on the E_true >= 1.33 GeV cells, 0 below
        self.ccqe_shape_multigev = np.where(self.e_c >= CCQE_SHAPE_SUBGEV_E,
                                            np.log(self.e_c), 0.0)          # (nE,)
        _mg = self.e_c >= CCQE_SHAPE_SUBGEV_E
        self.ccqe_shape_multigev[_mg] -= self.ccqe_shape_multigev[_mg].mean()

    # ---------------- weight fields ----------------
    def cell_weights(self, phi, theta):
        """W[k, cE, cZ] for nuisance vector theta and oscillation tensor phi[2,3,nE,nZ].

        NC classes get phi = 1.
        """
        t = dict(zip(self.nuisance_names, theta))

        # oscillation: gather per class, NC -> 1
        P = phi[self.cls_type, self.cls_flavor]                  # (n_cls, nE, nZ)
        P = np.where(self.cls_cc[:, None, None] == 1, P, 1.0)

        # flux field on (nE,) and (nZ,) -> broadcast (per class via pdg)
        f_e = np.ones(self.nE)
        f_e = np.where(self.e_below1, t["normalization_below1GeV"], f_e) \
            * np.where(self.e_above1, t["normalization_above1GeV"], 1.0)
        f_e = f_e * (self.e_c / 10.0) ** t["tilt"]
        # solar activity: w = 1 - x*A*exp(-E_true/L)
        f_e = f_e * (1.0 - t["solar_activity"] * SOLAR_AMP
                     * np.exp(-self.e_c / SOLAR_SCALE))
        # K/pi high-energy flux ramp
        f_e = f_e * (1.0 + t["kpi_ratio"] * self.kpi_shape)
        F_ez = f_e[:, None] * np.ones((self.nE, self.nZ))
        # one-sided zenith parameters, up-going and down-going
        zu = np.where(self.z_c < 0,
                      1.0 - t["zenith_up"] * self.tanhz2, 1.0)
        F_ez = F_ez * zu[None, :]
        zd = np.where(self.z_c >= 0,
                      1.0 - t["zenith_down"] * self.tanhz2, 1.0)
        F_ez = F_ez * zd[None, :]
        # horizontal/vertical flux ratio
        F_ez = F_ez * (1.0 + t["flux_horizvert"]
                       * self.horizvert_shape)[None, :]
        # per-class flux scalar
        f_cls = np.where(np.abs(self.cls_pdg) == 12, t["flavor_ratio"], 1.0)

        # cross section: mask parameters -> per-class scalar product
        xs = np.array([t[n] for n in MASK_TUNES])
        X_cls = np.where(self.cls_bits, xs[None, :], 1.0).prod(axis=1)
        # AxialMass: CC only, continuous in log10 ETrue
        ax = 1.0 + 0.042 * (t["AxialMass"] - 1.0) * 1.05 * self.log10e
        A_ke = np.where(self.cls_cc[:, None] == 1, ax[None, :], 1.0)  # (n_cls, nE)

        # banded flux ratios, per-parameter energy band:
        # heavy leg *= r(1+q)/(1+qr), light leg *= (1+q)/(1+qr); r = 1 => no-op.
        FR_ke = np.ones((self.n_cls, self.nE))
        for name, (band, hv, lt) in self.fr_resolved.items():
            s = t[name]
            q = self.fr_q[name]
            fh = 1.0 + band[None, :] * (s * (1.0 + q) / (1.0 + q * s) - 1.0)
            fl = 1.0 + band[None, :] * ((1.0 + q) / (1.0 + q * s) - 1.0)
            FR_ke = FR_ke * np.where(hv[:, None], fh, 1.0) \
                          * np.where(lt[:, None], fl, 1.0)

        # energy-dependent CCQE parameters
        XX_ke = np.ones((self.n_cls, self.nE))
        b = t["xsec_ccqe_shape"] - 1.0
        tilt = self.e_c[None, :] ** b                     # (1,nE), CCQE only
        XX_ke = XX_ke * np.where(self.ccqe_cls[:, None], tilt, 1.0)
        fac = 1.0 + t["xsec_ccqe_shape_subgev"] * self.ccqe_shape_subgev  # (nE,)
        XX_ke = XX_ke * np.where(self.ccqe_cls[:, None], fac[None, :], 1.0)
        # sub-GeV nu_e 1p1h norm
        r = t["xsec_1p1h_subgev_nue"]
        fac = 1.0 + self.e_below1[None, :] * (r - 1.0)    # (1,nE)
        XX_ke = XX_ke * np.where(self.ccqe_nue_cls[:, None], fac, 1.0)

        # multi-GeV CCQE shape: mean-zero log-E tilt on E_true >= 1.33 GeV CCQE cells
        fac = 1.0 + t[XSEC_CCQE_SHAPE_MULTIGEV_NAME] * self.ccqe_shape_multigev  # (nE,)
        XX_ke = XX_ke * np.where(self.ccqe_cls[:, None], fac[None, :], 1.0)

        # multi-GeV CCQE flavour norms (E_true >= 1.33 GeV)
        for _dial, _mattr in MULTIGEV_CCQE_NORM.items():
            r = t[_dial]
            fac = 1.0 + self.e_multigev[None, :] * (r - 1.0)  # (1,nE)
            XX_ke = XX_ke * np.where(getattr(self, _mattr)[:, None], fac, 1.0)

        W = P * F_ez[None, :, :] * (f_cls * X_cls)[:, None, None] \
            * A_ke[:, :, None] * FR_ke[:, :, None] * XX_ke[:, :, None]
        return W

    # ---------------- contractions ----------------
    def _build_csr(self, v, widx, eb):
        """CSR matrix of the response, rows (era, bin), columns the flattened (k, e, z)
        cell, so that contract_era is one mat-vec. The entries are sorted into row order
        by a stable sort, which keeps the original order inside each row."""
        n_rows = self.n_era * self.n_bins
        n_cols = int(self.n_cls) * int(self.nE) * int(self.nZ)
        if eb.size:
            if int(eb.min()) < 0 or int(eb.max()) >= n_rows:
                raise ValueError(f"R_eb out of range for n_era={self.n_era} "
                                 f"n_bins={self.n_bins}")
            if not np.all(np.diff(eb) >= 0):
                order = np.argsort(eb, kind="stable")
                v, widx, eb = (np.ascontiguousarray(a[order]) for a in (v, widx, eb))
        # int32 indptr/indices keep scipy from upcasting
        i32 = np.iinfo(np.int32).max
        idt = np.int32 if (eb.size <= i32 and n_cols <= i32) else np.int64
        indptr = np.zeros(n_rows + 1, dtype=idt)
        np.cumsum(np.bincount(eb, minlength=n_rows), out=indptr[1:])
        indices = np.ascontiguousarray(widx, dtype=idt)
        data = np.ascontiguousarray(v, dtype=np.float64)
        return sp.csr_matrix((data, indices, indptr),
                             shape=(n_rows, n_cols), copy=False)

    def contract_era(self, W):
        """n_pre[era, b] = R contracted with cell weights W, split by SK era."""
        return self._R_csr.dot(np.ascontiguousarray(W, dtype=np.float64).ravel()
                               ).reshape(self.n_era, self.n_bins)

    def _escale_migrate(self, arr_e, deltas):
        """Per-era energy-scale reco-E migration of a (n_era, n_bins) array.
        Linear, rate-conserving within each (sample, reco-cz) column:
          N'(ie) = N(ie) + d*( N(ie-1)*[ie>0] - N(ie)*[ie<ne-1] )."""
        out = np.empty_like(arr_e)
        for e in range(self.n_era):
            d = deltas[e]
            N = arr_e[e]
            below = np.where(self.es_below >= 0, N[self.es_below], 0.0)   # N(ie-1)
            out[e] = N + d * (below * self.es_has_below - N * self.es_has_above)
        return out

    def _era_theta(self, t, e):
        """Per-era view of the nuisance dict: era-split detector stems and the up/down
        energy scale take their era-e value under the base name that detector_factors
        reads; everything else is shared."""
        te = dict(t)
        tag = ERA_TAGS[e]
        for stem in self.det_split_stems:
            te[stem] = t[f"{stem}_{tag}"]
        te["updown_escale"] = t[f"updown_escale_{tag}"]
        return te

    # ---------------- detector factors ----------------
    def sample_rates(self, n_phys):
        """Weighted physics rate per sample (BaseWeight*PhysicsWeight sums)."""
        r = {}
        for s in self.samples:
            r[int(s)] = float(n_phys[self.bin_sample == s].sum())
        return r

    def detector_factors(self, t, rates, n_phys, era):
        """Per-bin multiplicative factor D and its per-parameter d ln D, with migration
        ratios r held fixed. Returns (D[b], {parameter: dlnD[b]}).

        ``t`` is the per-era view from _era_theta. ``rates`` and ``n_phys`` are this
        era's weighted rates per sample and per bin. ``era`` is the SK era index; it
        gates the per-era decay-e tagging parameters to their own era.
        """
        S = {int(s): 1.0 for s in self.samples}
        dS = {n: {int(s): 0.0 for s in self.samples} for n in DET_NAMES}

        def wsum(ids):
            return sum(rates.get(i, 0.0) for i in np.atleast_1d(ids))

        # We accumulate d ln D: a donor sample with factor x contributes 1/x, an
        # acceptor with factor (1+r(1-x)) contributes -r/(1+r(1-x)).

        counts = {int(k): v for k, v in self.sample_counts.items()}

        # fcpc_separation: migration between the FC side and the PC samples, on raw MC
        # counts, not weighted rates. The FC side is FCPC_DONOR_SAMPLES.
        x = t["fcpc_separation"]
        pc_ids = [14, 15]
        fc_ids = [int(s) for s in self.samples
                  if int(s) in FCPC_DONOR_SAMPLES]
        wfc = sum(c for s_, c in counts.items()
                  if s_ in FCPC_DONOR_SAMPLES)
        wpc = sum(counts.get(i, 0) for i in pc_ids)
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

        # fc_reduction: the samples in fc_ids (fcpc_separation's FC side) *= x
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
        # multiring_emu_separation: guard on (2-x) after the ratio
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

        # neutron tagging (ratios from the per-sample rates)
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

        # up-mu parameters
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

        # low-momentum e/mu single-ring migration for SK I-III: donor {0,1} *= x, acceptor
        # {3,4,5} *= 1+r(1-x), r = W(donor_lowp)/W(acc_lowp) from the per-bin rate, in the
        # two lowest momentum bins. In an era where the samples are empty ra <= 0, so the
        # migration is a no-op with zero derivative.
        name = ELIKE_RINGCOUNT_LOWP_NAME
        d = np.zeros(self.n_bins)
        donor_m, acc_m = self.elike_ringcount_lowp_masks
        x = t[name]
        ra = float(n_phys[acc_m].sum())
        r_ = float(n_phys[donor_m].sum()) / ra if ra > 0 else 0.0
        acc = 1.0 + r_ * (1.0 - x)
        if _unphys(x) or ra <= 0.0 or _unphys(acc):
            dlnD[name] = d          # unphysical or empty acceptor: no-op
        else:
            fac = np.ones(self.n_bins)
            fac[donor_m] = x
            d[donor_m] = (1.0 / x) if x != 0 else 0.0
            fac[acc_m] = acc
            d[acc_m] = -r_ / acc
            D = D * fac
            dlnD[name] = d

        # rel_norm_fcmg: flat normalization on the FC multi-GeV sample group
        x = t["rel_norm_fcmg"]
        D = D * np.where(self.fcmg_bin_mask, x, 1.0)
        dlnD["rel_norm_fcmg"] = np.where(self.fcmg_bin_mask,
                                         (1.0 / x if x != 0 else 0.0), 0.0)

        # upmu_norm: flat normalization on the up-mu samples {16,17,18}
        x = t[UPMU_NORM_NAME]
        D = D * np.where(self.upmu_norm_bin_mask, x, 1.0)
        dlnD[UPMU_NORM_NAME] = np.where(self.upmu_norm_bin_mask,
                                        (1.0 / x if x != 0 else 0.0), 0.0)

        # 0n/1n neutron migration, one parameter per nuebar sample pair: donor *= x,
        # acceptor *= 1+r(1-x), r = weighted rate(donor)/rate(acceptor), so the pair's
        # total rate is invariant under x. The 0n/1n samples exist only in SK IV+V, so in
        # the other eras the acceptor rate is 0 and this is a no-op.
        for name in self.active_neutron_mig:
            d = np.zeros(self.n_bins)
            x = t[name]
            donor_s, acc_s = self.neutron_mig_pairs[name]
            donor_m = self.bin_sample == donor_s
            acc_m = self.bin_sample == acc_s
            ra = float(rates.get(acc_s, 0.0))
            if _unphys(x) or ra <= 0.0:
                dlnD[name] = d          # ratio undefined or unphysical: no-op
                continue
            r_ = float(rates.get(donor_s, 0.0)) / ra
            acc = 1.0 + r_ * (1.0 - x)
            fac = np.ones(self.n_bins)
            fac[donor_m] = x
            d[donor_m] = (1.0 / x) if x != 0 else 0.0
            fac[acc_m] = acc
            d[acc_m] = (-r_ / acc) if acc != 0 else 0.0
            D = D * fac
            dlnD[name] = d

        # per-era decay-e tagging: rate-conserving sample-boundary migration. Each
        # parameter acts only in its own era, so ``rates`` is that era's rate. Two-state
        # block: acceptor *= x, donor *= 1+r(1-x), r = W(acceptor)/W(donor). Three-state
        # mu-like cascade: rx1 = x*r1 + 2(1-x)*r2, rx2 = x^2*r2, rx0 = 1-rx1-rx2. x = 1 is
        # the identity.
        for name in self.active_decay_e_tag:
            d = np.zeros(self.n_bins)
            if era != DECAY_E_TAG_ERA[name]:
                dlnD[name] = d           # wrong era: no-op, no derivative
                continue
            x = t[name]
            if _unphys(x):
                dlnD[name] = d           # unphysical: no-op
                continue
            fac = np.ones(self.n_bins)
            for block in DECAY_E_TAG_BLOCKS[name]:
                if block[0] == "pair2":
                    _, acc_ids, don_ids = block
                    wa = sum(float(rates.get(s, 0.0)) for s in acc_ids)
                    wd = sum(float(rates.get(s, 0.0)) for s in don_ids)
                    if wd <= 0.0 or wa <= 0.0:
                        continue          # empty leg: no migration
                    r_ = wa / wd
                    acc_ml = np.isin(self.bin_sample, acc_ids)
                    don_ml = np.isin(self.bin_sample, don_ids)
                    don_f = 1.0 + r_ * (1.0 - x)
                    fac[acc_ml] = x
                    d[acc_ml] = (1.0 / x) if x != 0 else 0.0
                    fac[don_ml] = don_f
                    d[don_ml] = (-r_ / don_f) if don_f != 0 else 0.0
                else:                     # "cascade3"
                    _, (s0, s1, s2) = block
                    N = sum(float(rates.get(s, 0.0)) for s in (s0, s1, s2))
                    if N <= 0.0:
                        continue          # empty cascade: no migration
                    r0 = float(rates.get(s0, 0.0)) / N
                    r1 = float(rates.get(s1, 0.0)) / N
                    r2 = float(rates.get(s2, 0.0)) / N
                    rx1 = x * r1 + 2.0 * (1.0 - x) * r2
                    rx2 = x * x * r2
                    rx0 = 1.0 - rx1 - rx2
                    drx1 = r1 - 2.0 * r2
                    drx2 = 2.0 * x * r2
                    drx0 = -drx1 - drx2
                    for s_, rr, rx, drx in ((s0, r0, rx0, drx0),
                                            (s1, r1, rx1, drx1),
                                            (s2, r2, rx2, drx2)):
                        if rr <= 0.0:
                            continue
                        m = self.bin_sample == s_
                        fac[m] = rx / rr
                        d[m] = (drx / rx) if rx != 0 else 0.0
            D = D * fac
            dlnD[name] = d

        # up-mu background shape: multiplicative factor on the near-horizon bins
        for name in self.active_upmu_bkg:
            x = t[name]
            mk = self.upmu_bkg_masks[name]
            D = D * np.where(mk, x, 1.0)
            dlnD[name] = np.where(mk, (1.0 / x if x != 0 else 0.0), 0.0)

        # up/down energy scale: anti-symmetric normalization of up-going (*= 1+d) and
        # down-going (*= 1-d) FC and PC bins through the signed mask. d = 0 is a no-op.
        # t["updown_escale"] is this era's value; d ln D/dd = sign/(1 + d*sign).
        d = t["updown_escale"]
        fac = 1.0 + d * self.ude_sign
        D = D * fac
        dlnD["updown_escale"] = np.divide(self.ude_sign, fac,
                                          out=np.zeros_like(fac), where=fac != 0)
        return D, dlnD

    # ---------------- expectation + chi2 ----------------
    def expectation(self, phi, theta, return_parts=False):
        """Expected counts in the analysis bins: E_b = sum_era D_era[b] * n_pre_era[b].

        Detector factors are evaluated per era: era-split parameters take their era
        value and migration ratios use that era's rates.
        """
        t = dict(zip(self.nuisance_names, theta))
        W = self.cell_weights(phi, theta)
        n_pre_e = self.contract_era(W)                 # (n_era, n_bins)

        # energy scale: reco-E migration of the pre-detector rates, so the detector
        # factors ride on top. Linear and rate-conserving; no-op at x = 1.
        es_deltas = np.array([t[f"energy_scale_{ERA_TAGS[e]}"] - 1.0
                              for e in range(self.n_era)])
        n_pre0_es = n_pre_e                            # unmigrated (for the gradient)
        n_pre_e = self._escale_migrate(n_pre_e, es_deltas)
        nphys_e = self.contract_era(self.cell_weights_physics_only(phi))

        n_nu = np.zeros(self.n_bins)
        parts_e = []
        for e in range(self.n_era):
            t_e = self._era_theta(t, e)
            np_e = nphys_e[e]
            rates_e = self.sample_rates(np_e)
            D_e, dlnD_e = self.detector_factors(t_e, rates_e, n_phys=np_e, era=e)
            n_nu = n_nu + n_pre_e[e] * D_e
            if return_parts:
                parts_e.append(dict(D=D_e, dlnD=dlnD_e, n_pre=n_pre_e[e]))
        if return_parts:
            return n_nu, dict(W=W, parts_e=parts_e, n_pre0_es=n_pre0_es,
                              es_deltas=es_deltas)
        return n_nu

    def cell_weights_physics_only(self, phi):
        P = phi[self.cls_type, self.cls_flavor]
        return np.where(self.cls_cc[:, None, None] == 1, P, 1.0)

    @staticmethod
    def poisson_chi2(obs, n_mod):
        """Saturated Poisson chi^2; 9e9 when any model bin is <= 0."""
        if np.any(n_mod <= 0):
            return 9e9
        log_term = np.log(np.divide(obs, n_mod, out=np.ones_like(obs),
                                    where=n_mod > 0))
        log_term[obs == 0] = 0
        return float(2 * np.sum(n_mod - obs + obs * log_term))

    def chi2(self, phi, theta):
        n_nu = self.expectation(phi, theta)
        stat = self.poisson_chi2(self.obs_f, n_nu[self.few])
        pen = np.sum((theta - self.nominal) ** 2 / self.sigma ** 2)
        return stat + pen

    # ---------------- analytic gradient ----------------
    def chi2_and_grad(self, phi, theta):
        """chi^2 and its gradient, first order: migration ratios are held fixed.

        Flux and cross-section parameters enter the era-independent cell weights W,
        so dE_b = sum_era D_era[b] * migrate(contract_era(W * dlnW/dp))[era][b].
        Era-split detector parameters route to their per-era slot; era-independent
        ones accumulate over eras.
        """
        n_nu, parts = self.expectation(phi, theta, return_parts=True)
        m = self.few
        obs, E = self.obs_f, n_nu[m]
        pen = np.sum((theta - self.nominal) ** 2 / self.sigma ** 2)
        stat = self.poisson_chi2(obs, E)
        if stat >= 9e9:                           # a model bin is <= 0
            return stat + pen, 2 * (theta - self.nominal) / self.sigma ** 2
        f = stat + pen
        resid = 2 * (1 - obs / np.maximum(E, 1e-9))              # dchi2/dE_b

        g = 2 * (theta - self.nominal) / self.sigma ** 2         # penalty gradient
        t = dict(zip(self.nuisance_names, theta))
        W = parts["W"]
        parts_e = parts["parts_e"]
        D_stack = np.stack([pe["D"] for pe in parts_e])          # (n_era, n_bins)
        # the energy-scale migration is held fixed here: flux and cross-section
        # gradients ride through it (it is linear in n_pre); the energy-scale
        # parameters' own gradient is added below.
        es_deltas = parts["es_deltas"]                           # (n_era,)

        def acc(field):
            return np.sum(resid * field[m])

        def dE_W(gfield):
            ce = self.contract_era(W * gfield)
            ce = self._escale_migrate(ce, es_deltas)
            return (D_stack * ce).sum(0)                         # (n_bins,)

        # flux: dW/W fields on the true cells
        for name in self.flux_names:
            gfield = self._flux_dlnw(name, t)                    # (n_cls,nE,nZ)
            dE = dE_W(gfield)
            g[self.nuisance_names.index(name)] += acc(dE)

        # cross-section mask parameters: g = bit/x per class
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

        # banded flux ratios: per-class d ln W / d r, per-parameter energy band
        for name, (band, hv, lt) in self.fr_resolved.items():
            i = self.nuisance_names.index(name)
            s = t[name]
            q = self.fr_q[name]
            dh = (1.0 + q) / (1.0 + q * s) ** 2       # d/dr [s(1+q)/(1+qs)]
            dl = -q * (1.0 + q) / (1.0 + q * s) ** 2  # d/dr [(1+q)/(1+qs)]
            fh = 1.0 + band * (s * (1.0 + q) / (1.0 + q * s) - 1.0)
            fl = 1.0 + band * ((1.0 + q) / (1.0 + q * s) - 1.0)
            gh = band * dh / fh                 # (nE,)
            gl = band * dl / fl
            gfield = np.where(hv[:, None], gh[None, :], 0.0) \
                + np.where(lt[:, None], gl[None, :], 0.0)       # (n_cls,nE)
            dE = dE_W(gfield[:, :, None])
            g[i] += acc(dE)

        # global CCQE shape: W has CCQE *= E^(r-1); d ln W/dr = ln(E) on CCQE cells
        i = self.nuisance_names.index("xsec_ccqe_shape")
        gf = np.where(self.ccqe_cls[:, None], np.log(self.e_c)[None, :], 0.0)
        dE = dE_W(gf[:, :, None])
        g[i] += acc(dE)
        # sub-GeV CCQE shape: W has CCQE *= 1 + x*sh(E); d ln W/dx = sh/(1+x*sh)
        i = self.nuisance_names.index("xsec_ccqe_shape_subgev")
        sh = self.ccqe_shape_subgev
        fac = 1.0 + t["xsec_ccqe_shape_subgev"] * sh
        dlnw = np.divide(sh, fac, out=np.zeros_like(sh), where=fac != 0)   # (nE,)
        gf = np.where(self.ccqe_cls[:, None], dlnw[None, :], 0.0)
        dE = dE_W(gf[:, :, None])
        g[i] += acc(dE)
        # multi-GeV CCQE shape: the same form on the complementary energy support
        i = self.nuisance_names.index(XSEC_CCQE_SHAPE_MULTIGEV_NAME)
        sh = self.ccqe_shape_multigev
        fac = 1.0 + t[XSEC_CCQE_SHAPE_MULTIGEV_NAME] * sh
        dlnw = np.divide(sh, fac, out=np.zeros_like(sh), where=fac != 0)   # (nE,)
        gf = np.where(self.ccqe_cls[:, None], dlnw[None, :], 0.0)
        dE = dE_W(gf[:, :, None])
        g[i] += acc(dE)
        # sub-GeV nu_e 1p1h norm: d ln W/dr = 1[E<1]/fac on the class mask. fac -> 0 as
        # r -> 0 on sub-GeV cells, where W itself -> 0, so those cells contribute 0.
        i = self.nuisance_names.index("xsec_1p1h_subgev_nue")
        r = t["xsec_1p1h_subgev_nue"]
        fac = 1.0 + self.e_below1 * (r - 1.0)                    # (nE,)
        dlnw = np.divide(self.e_below1, fac, out=np.zeros_like(fac),
                         where=fac != 0)
        gf = np.where(self.ccqe_nue_cls[:, None], dlnw[None, :], 0.0)
        dE = dE_W(gf[:, :, None])
        g[i] += acc(dE)
        # multi-GeV CCQE flavour norms: d ln W/dr = 1[E>=1.33]/fac on the flavour mask
        for _dial, _mattr in MULTIGEV_CCQE_NORM.items():
            i = self.nuisance_names.index(_dial)
            r = t[_dial]
            fac = 1.0 + self.e_multigev * (r - 1.0)              # (nE,)
            dlnw = np.divide(self.e_multigev, fac, out=np.zeros_like(fac),
                             where=fac != 0)
            gf = np.where(getattr(self, _mattr)[:, None], dlnw[None, :], 0.0)
            dE = dE_W(gf[:, :, None])
            g[i] += acc(dE)

        # detector: dE_b = n_pre_era[b] * D_era[b] * dlnD_era[name][b], ratios fixed.
        # Era-split stems route to their per-era parameter; the rest accumulate over eras.
        split_set = set(self.det_split_stems) | {"updown_escale"}
        for e in range(self.n_era):
            pe = parts_e[e]
            npD = pe["n_pre"] * pe["D"]
            for name, d in pe["dlnD"].items():
                if name in split_set:
                    idx = self.nuisance_names.index(f"{name}_{ERA_TAGS[e]}")
                else:
                    idx = self.nuisance_names.index(name)
                g[idx] += acc(npD * d)

        # energy scale: dN'_e/dx_e = N(ie-1)*[ie>0] - N(ie)*[ie<ne-1] on the unmigrated
        # per-era rates (the migration is linear in x-1, so this is exact and independent
        # of x), propagated through the detector factor D_e.
        n_pre0_es = parts["n_pre0_es"]
        for e in range(self.n_era):
            N = n_pre0_es[e]
            below = np.where(self.es_below >= 0, N[self.es_below], 0.0)
            dN = below * self.es_has_below - N * self.es_has_above
            dE = parts_e[e]["D"] * dN
            g[self._es_idx[e]] += acc(dE)

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
        if name == "flavor_ratio":
            gcls = np.where(np.abs(self.cls_pdg) == 12, 1.0 / t[name], 0.0)
            return np.broadcast_to(gcls[:, None, None],
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
        if name == "flux_horizvert":                     # w = 1 + x*g(cz), g=(1-3cz^2)/2
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

    # ---------------- fit ----------------
    def fit_box(self):
        """(lower, upper) bounds of every parameter in ``fit_point``: nominal +- 10
        sigma, lower bound raised to 0.01 for positive-nominal parameters, and FIT_BOX
        for the flux-ratio, cross-section-shape and migration parameters."""
        nominal = self.nominal.copy()
        lower = nominal - 10 * self.sigma
        upper = nominal + 10 * self.sigma
        lower[(nominal > 0) & (lower < 0.01)] = 0.01
        for name, (lo, hi) in FIT_BOX.items():
            k = self.nuisance_names.index(name)
            lower[k], upper[k] = lo, hi
        return lower, upper

    def fit_point(self, phi_node, x0, free_mask):
        """One L-BFGS-B fit of the nuisance parameters at one lattice node.

        phi_node: phi[2, 3, nE, nZ]. x0 None starts from nominal. Parameters with
        free_mask False stay at nominal through collapsed bounds. Returns
        (chi2, nuisance, nit, converged); chi2 is inf and nuisance is the start point
        if the minimizer returns no value below inf.
        """
        nominal = self.nominal.copy()
        if x0 is None:
            x0 = nominal
        lower, upper = self.fit_box()
        fixed = ~np.asarray(free_mask, bool)
        lower[fixed] = nominal[fixed]
        upper[fixed] = nominal[fixed]
        x0 = np.where(fixed, nominal, x0)
        bounds = list(zip(lower, upper))

        best = (np.inf, x0, 0, False)
        phi = phi_node.astype(float)
        res = minimize(lambda th: self.chi2_and_grad(phi, th), x0,
                       method="L-BFGS-B", jac=True, bounds=bounds,
                       options={"ftol": float(self.fit_ftol), "gtol": 1e-5,
                                "maxiter": int(self.fit_maxiter)})
        if res.fun < best[0]:
            best = (res.fun, res.x.copy(), res.nit, res.success)
        return best
