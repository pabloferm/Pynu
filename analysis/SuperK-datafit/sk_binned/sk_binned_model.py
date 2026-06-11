#!/usr/bin/env python3
"""SK binned forward model: response-matrix expectation with systematics on
bins / the true grid. Pure numpy+scipy; no Pynu import needed at fit time.

Faithfully replicates the event-by-event pipeline (SuperK_2023 + AtmoFlux +
WaterXSection + SKCombinedDetector) at true-grid resolution:

  N_b = [ sum_k  R_k[c,b]^T  ( Phi_k(c) * flux(c,k) * xsec_k * axial(c_E)^CC_k ) ]
        * prod_detector d_b   ( * (1 + Fij_b * (es-1)) if energy scale enabled )

with migration ratios r computed from the physics-only binned rates
N_phys_b = sum_k R_k^T Phi_k (the binned analog of _rate_weight =
BaseWeight*PhysicsWeight — exactly equal, since bins partition events), and
fcpc_separation using RAW per-sample event counts as in production.

Source-of-truth forms ported verbatim from:
  pynu/PhysicsTunes/Flux/AtmoFlux.py (tilt pivot 10 GeV; norm steps at
  1 GeV; nunubar pdg<0; flavor |pdg|==12; barr_zenith; zenith_up/down),
  pynu/PhysicsTunes/CrossSection/WaterXSection.py (12 flat masks captured
  in the response class table; AxialMass = 1+0.042(x-1)*1.05*log10(E), CC),
  pynu/PhysicsTunes/Detector/SKCombinedDetector.py (norm/migration tables
  below; fcpc raw-count form).
"""
import json
import re

import numpy as np
from scipy import sparse

# ---- detector tune tables (sample sets verbatim from SKCombinedDetector.py)
PC_SAMPLES = [14, 15]
UPMU_SAMPLES = [16, 17, 18]
NORM_TUNES = {
    # name: samples ('ALL' = every sample, 'FC' = not PC/UpMu)
    "fiducial_volume": "ALL",
    "subgev_2ring_pi0": [6],
    "subgev_1ring_pi0": [2],
    "fc_reduction": "FC",
    "pc_reduction": PC_SAMPLES,
    "mre_nonubkg": [10, 11, 12, 13],
    "mge_nonubkg": [7, 8, 24, 25, 26],
    "upmu_stop_bkg": [16],
    "upmu_showering_bkg": [18],
    "upmu_nonshowering_bkg": [17],
    "subgev_numulike_sk45_mc": [27],
}
MIGRATION_TUNES = {
    # name: (donor samples scaled x, acceptor samples scaled 1 + r(1-x))
    "multiring_nunubar_separation": ([10], [11]),
    "multiring_emu_separation": ([10, 11, 13], [12]),
    "multiring_eother_separation": ([10, 11], [13]),
    "pc_stopthru_separation": ([14], [15]),
    "pi0_ring_separation": ([2], [6]),
    "e_ring_separation": ([0, 1, 7, 8, 19, 20, 21], [10, 11, 13, 24, 25, 26]),
    "mu_ring_separation": ([3, 4, 5, 9, 22, 23, 27, 28], [12]),
    "singlering_pid": ([0, 1, 7, 8, 19, 20, 21, 24, 25, 26],
                       [3, 4, 5, 9, 22, 23, 27, 28]),
    "multiring_pid": ([10, 11, 13], [12]),
    "upmu_shower_separation": ([18], [17]),
    "neutron_tagging": ([20, 22, 25, 27], [21, 23, 26, 28]),
    "neutron_tagging_subgev": ([20, 22], [21, 23]),
    "neutron_tagging_multigev": ([25, 27], [26, 28]),
}
FLUX_TUNES = ["normalization_below1GeV", "normalization_above1GeV", "tilt",
              "nunubar_ratio", "flavor_ratio", "zenith_up", "zenith_down",
              "barr_zenith"]
XSEC_FLAT = ["XSecNuTau", "NCoverCC", "NCHad", "DIS", "CCQE", "CCQENuBarNu",
             "CCQEMuE", "CC1Pi_Pi0Pi", "CC1Pi_NuBarNuE", "CC1Pi_NuBarNuMu",
             "CC1PiProduction", "CohPiProduction"]


def parse_nuisances(xml_path):
    """Enabled nuisances from the analysis XML: [(name, nominal, sigma)]."""
    text = open(xml_path).read()
    out = []
    for m in re.finditer(r"<nuisance name='([^']+)'>(.*?)</nuisance>", text, re.S):
        name, block = m.group(1), m.group(2)
        status = int(re.search(r"<status>\s*(\S+)\s*</status>", block).group(1))
        if status != 1:
            continue
        sigma = float(re.search(r"<sigma>\s*(\S+)\s*</sigma>", block).group(1))
        nominal = float(re.search(r"<nominal>\s*(\S+)\s*</nominal>", block).group(1))
        out.append((name, nominal, sigma))
    return out


class SKBinnedModel:
    def __init__(self, response_npz, xml_path, energy_scale=True):
        d = np.load(response_npz, allow_pickle=True)
        self.meta = json.loads(str(d["meta"]))
        self.n_bins = int(d["n_bins"])
        self.e_edges = d["e_edges"]; self.z_edges = d["z_edges"]
        self.nE = self.e_edges.size - 1; self.nZ = self.z_edges.size - 1
        self.e_c = np.sqrt(self.e_edges[:-1] * self.e_edges[1:])
        self.z_c = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])
        self.observed = d["observed"]
        self.classes = d["classes"]          # [n_cls, 2+12]: pdg, cc, bits
        self.n_cls = self.classes.shape[0]
        self.xsec_names = [str(s) for s in d["xsec_tune_names"]]
        assert self.xsec_names == XSEC_FLAT
        self.sample_table = {int(k): v for k, v in
                             json.loads(str(d["sample_table"])).items()}
        if "sample_event_counts" in d.files:
            self.sample_counts = {int(k): v for k, v in
                                  json.loads(str(d["sample_event_counts"])).items()}
        else:  # older response file; only fcpc_separation at x != 1 needs counts
            self.sample_counts = None

        n_cells = self.nE * self.nZ

        def csr(prefix):
            k, ce, cz = d[f"{prefix}_k"], d[f"{prefix}_e"], d[f"{prefix}_z"]
            b, v = d[f"{prefix}_b"], d[f"{prefix}_v"]
            rows = k.astype(np.int64) * n_cells + ce.astype(np.int64) * self.nZ + cz
            return sparse.csr_matrix(
                (v, (rows, b)), shape=(self.n_cls * n_cells, self.n_bins))

        self.R = csr("R")
        self.Rp = csr("Rp") if energy_scale else None
        self.Rm = csr("Rm") if energy_scale else None
        self.energy_scale_enabled = energy_scale

        # per-sample bin slices and masks
        self.bin_sample = np.empty(self.n_bins, dtype=np.int64)
        for s, (off, ne, nz) in self.sample_table.items():
            self.bin_sample[off:off + ne * nz] = s
        self.samples = sorted(self.sample_table)
        self.fc_samples = [s for s in self.samples
                           if s not in PC_SAMPLES + UPMU_SAMPLES]

        # nuisance bookkeeping
        self.nuis = parse_nuisances(xml_path)
        if energy_scale:
            self.nuis.append(("energy_scale", 1.0, 0.025))
        self.nuis_names = [n for n, _, _ in self.nuis]
        self.nominal = np.array([nom for _, nom, _ in self.nuis])
        self.sigma = np.array([s for _, _, s in self.nuis])
        known = (set(FLUX_TUNES) | set(XSEC_FLAT) | set(NORM_TUNES)
                 | set(MIGRATION_TUNES)
                 | {"AxialMass", "fcpc_separation", "energy_scale",
                    "decay_e_tagging"})
        unknown = [n for n in self.nuis_names if n not in known]
        if unknown:
            raise ValueError(f"enabled nuisances with no binned implementation: {unknown}")
        if "decay_e_tagging" in self.nuis_names:
            raise NotImplementedError("decay_e_tagging not ported (disabled in config)")

        # per-class index helpers
        pdg = self.classes[:, 0].astype(int)
        cc = self.classes[:, 1].astype(int)
        self.cls_pdg = pdg
        self.cls_cc = cc
        self.cls_type = (pdg < 0).astype(int)          # NSQ type
        self.cls_flavor = (np.abs(pdg) // 2 - 6)       # 0=e,1=mu,2=tau
        self.cls_bits = self.classes[:, 2:].astype(bool)  # [n_cls, 12]

        # static per-class/grid factor pieces
        self.log10_ec = np.log10(self.e_c)

    # ---- physics + true-level nuisances -> stacked cell-weight vector
    def _cell_weights(self, phi, x):
        """phi: [2,3,nE,nZ]; x: dict name->value. Returns (w_full, w_phys)
        stacked over classes (n_cls*nE*nZ,)."""
        gx = {n: x[n] for n in self.nuis_names if n in x}
        # flux factors on the (E, Z) grid, flavor-dependent pieces per class
        E = self.e_c[:, None]          # (nE,1)
        Z = self.z_c[None, :]          # (1,nZ)
        grid_flux = np.ones((self.nE, self.nZ))
        if "normalization_below1GeV" in gx:
            grid_flux = grid_flux * np.where(E < 1.0, gx["normalization_below1GeV"], 1.0)
        if "normalization_above1GeV" in gx:
            grid_flux = grid_flux * np.where(E > 1.0, gx["normalization_above1GeV"], 1.0)
        if "tilt" in gx:
            grid_flux = grid_flux * (E / 10.0) ** gx["tilt"]
        if "zenith_up" in gx:
            grid_flux = grid_flux * np.where(Z < 0, 1.0 - gx["zenith_up"] * np.tanh(Z) ** 2, 1.0)
        if "zenith_down" in gx:
            grid_flux = grid_flux * np.where(Z >= 0, 1.0 - gx["zenith_down"] * np.tanh(Z) ** 2, 1.0)
        if "barr_zenith" in gx:
            env = 0.07 / (1.0 + (self.e_c / 0.5) ** 2)
            grid_flux = grid_flux * (1.0 + env[:, None] * gx["barr_zenith"]) ** np.tanh(3.0 * Z)

        axial = np.ones(self.nE)
        if "AxialMass" in gx:
            axial = 1.0 + 0.042 * (gx["AxialMass"] - 1.0) * 1.05 * self.log10_ec

        w_full = np.empty((self.n_cls, self.nE, self.nZ))
        w_phys = np.empty((self.n_cls, self.nE, self.nZ))
        for k in range(self.n_cls):
            if self.cls_cc[k]:
                p = phi[self.cls_type[k], self.cls_flavor[k]]
            else:
                p = 1.0  # NC: PhysicsWeight forced to 1 (SuperK_2023 convention)
            w_phys[k] = p
            f = grid_flux.copy()
            if "nunubar_ratio" in gx and self.cls_pdg[k] < 0:
                f = f * gx["nunubar_ratio"]
            if "flavor_ratio" in gx and abs(self.cls_pdg[k]) == 12:
                f = f * gx["flavor_ratio"]
            flat = 1.0
            for j, name in enumerate(XSEC_FLAT):
                if name in gx and self.cls_bits[k, j]:
                    flat = flat * gx[name]
            w = p * f * flat
            if self.cls_cc[k]:
                w = w * axial[:, None]
            w_full[k] = w
        return w_full.reshape(-1), w_phys.reshape(-1)

    # ---- detector factors on bins
    def _detector_factors(self, n_phys_b, x):
        d = np.ones(self.n_bins)
        for name, samp in NORM_TUNES.items():
            if name not in self.nuis_names or name not in x:
                continue
            if samp == "ALL":
                d *= x[name]
            else:
                sl = self.fc_samples if samp == "FC" else samp
                d *= np.where(np.isin(self.bin_sample, sl), x[name], 1.0)
        for name, (don, acc) in MIGRATION_TUNES.items():
            if name not in self.nuis_names or name not in x:
                continue
            xv = x[name]
            if name == "multiring_pid" and abs(1.0 - xv) < 1e-4:
                xv = 1.0
            nd = n_phys_b[np.isin(self.bin_sample, don)].sum()
            na = n_phys_b[np.isin(self.bin_sample, acc)].sum()
            r = nd / na
            d *= np.where(np.isin(self.bin_sample, don), xv,
                          np.where(np.isin(self.bin_sample, acc),
                                   1.0 + r * (1.0 - xv), 1.0))
        if "fcpc_separation" in self.nuis_names and "fcpc_separation" in x:
            if self.sample_counts is None:
                if abs(x["fcpc_separation"] - 1.0) > 1e-12:
                    raise RuntimeError("fcpc_separation != 1 needs sample_event_counts "
                                       "(rebuild sk_response.npz with the patched builder)")
                return d
            # RAW event counts, verbatim production form
            wfc = sum(self.sample_counts[s] for s in self.fc_samples)
            wpc = sum(self.sample_counts[s] for s in PC_SAMPLES)
            xv = x["fcpc_separation"]
            y = ((wpc + wfc) - xv * wfc) / wpc
            d *= np.where(np.isin(self.bin_sample, self.fc_samples), xv,
                          np.where(np.isin(self.bin_sample, PC_SAMPLES), y, 1.0))
        return d

    def expectation(self, phi, x):
        """phi: [2,3,nE,nZ] oscillated-flux tensor; x: dict name->value.
        Returns N_b (n_bins,)."""
        w_full, w_phys = self._cell_weights(phi, x)
        n_raw = self.R.T.dot(w_full)
        n_phys = self.R.T.dot(w_phys)
        d = self._detector_factors(n_phys, x)
        n = n_raw * d
        if self.energy_scale_enabled and "energy_scale" in x:
            delta = x["energy_scale"] - 1.0
            if delta != 0.0:
                np_b = self.Rp.T.dot(w_full) * d
                nm_b = self.Rm.T.dot(w_full) * d
                with np.errstate(divide="ignore", invalid="ignore"):
                    fij = np.where(n > 0, (np_b - nm_b) / (2 * 0.02 * np.maximum(n, 1e-300)), 0.0)
                n = n * (1.0 + fij * delta)
        return n

    def vec_to_dict(self, vec):
        return dict(zip(self.nuis_names, vec))

    def chi2(self, phi, vec, min_entries=-1.0):
        """SK-official Eq. 10: 2*sum[E - O + O ln(O/E)] + sum((x-nom)/sigma)^2.

        min_entries: STRICT lower cut on observed counts per bin, matching the
        production filter `ObservedBinned > MIN_ENTRIES` (Experiment.py:241).
        Default -1 keeps all bins (the SK-official full-930 configuration);
        pass 5 to reproduce the event-engine filtered set (n_data=65872.3).
        """
        n = self.expectation(phi, self.vec_to_dict(vec))
        o = self.observed
        m = o > min_entries
        E, O = n[m], o[m]
        if np.any(E <= 0):
            return 9e9
        nz = O > 0
        stats = np.sum(E - O) + np.sum(O[nz] * np.log(O[nz] / E[nz]))
        penalty = np.sum(((vec - self.nominal) / self.sigma) ** 2)
        return 2.0 * stats + penalty
