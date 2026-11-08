"""KM3NeT/ORCA6 binned response-matrix engine with an analytic dial gradient.

The dial algebra is evaluated on the populated true cells, the response is contracted
once, and the gradient comes from a single adjoint pass. The ORCA MC true side is exactly
quantized (one ETrue and one cos ZTrue per true bin), so a dial evaluated at the cell's
true coordinate equals the per-event value.

The engine returns the statistical chi^2 and its gradient, with no prior: the caller
owns the Gaussian prior.

E_shift is pinned. The precomputed response encodes energy scale 1, so a moved E_shift
would be ignored; the engine refuses it, and its gradient slot is a literal 0.0.
"""
import numpy as np

from .contract import contract, adjoint_cells
from . import dial_fields as bdf
from .dial_fields import DialField, build_cell_geometry

# ORCA6 analysis binning (pid x reco-E x reco-cosZ), from the KM3NeT release.
N_ERECO, N_CZRECO, N_PID = 15, 20, 3
N_BINS_ORCA = N_PID * N_ERECO * N_CZRECO                 # 900


def poisson_chi2(obs, n_mod):
    """Saturated Poisson deviance; 9e9 barrier when any model bin is <= 0."""
    if np.any(n_mod <= 0):
        return 9e9
    lt = np.log(np.divide(obs, n_mod, out=np.ones_like(obs), where=n_mod > 0))
    lt[obs == 0] = 0
    return float(2 * np.sum(n_mod - obs + obs * lt))


# The 30 parameters of the arm.
ORCA_MANIFEST = (
    list(bdf.SHARED_FLUX_19)                                  # 19 flux
    + list(bdf.SHARED_XSEC_4)                                 # 4 cross section
    + ["f_all", "f_HPT", "f_Shower", "f_tauCC", "f_NC", "f_HE", "E_shift"]   # 7 detector
)
assert len(ORCA_MANIFEST) == 30

BIN_AXIS_DIALS = ("f_HPT", "f_Shower")     # keyed on b // 300, not on the class
MIGRATION_DIALS = ("E_shift",)             # reco migration; cannot be cell-constant

BARRIER_CHI2 = 9e9                         # chi^2 when a fitted bin's model rate is <= 0
BLOCK = N_ERECO * N_CZRECO                 # 300; b // BLOCK == pid
SCHEMA_FLAT900 = "orca_binned_response_v2_flat900"


# ---------------------------------------------------------------------------
# ORCA detector dial fields on the cell axis. f_HPT and f_Shower act on the bin axis and
# are handled in D_b.
# ---------------------------------------------------------------------------
def _orca_detector_fields(geom):
    """Build the 4 cell-axis ORCA detector fields as closures over cached masks."""
    he = ((geom.E > 500.0) & geom.m_cc) | ((geom.E > 100.0) & geom.m_nc)
    tau_cc = geom.m_abs16 & geom.m_cc
    nc = geom.m_nc
    ones = np.ones(geom.n_cell)

    def f_all(g, x):
        return x * ones

    def d_all(g, x, w):
        return ones / x

    def _mask_field(mask):
        def f(g, x):
            w = np.ones(g.n_cell)
            w[mask] = x
            return w

        def d(g, x, w):
            return mask / x
        return f, d

    f_tau, d_tau = _mask_field(tau_cc)
    f_nc, d_nc = _mask_field(nc)
    f_he, d_he = _mask_field(he)

    return {
        "f_all": DialField("f_all", "N", f_all, d_all,
                           mask_fn=lambda g: np.ones(g.n_cell, bool)),
        "f_tauCC": DialField("f_tauCC", "N", f_tau, d_tau,
                             mask_fn=lambda g: tau_cc),
        "f_NC": DialField("f_NC", "N", f_nc, d_nc,
                          mask_fn=lambda g: nc),
        # f_HE's mask keys on ETrue and current, so it is not class-constant; the
        # engine measures that at load and routes it through the general dot.
        "f_HE": DialField("f_HE", "E", f_he, d_he,
                          mask_fn=lambda g: he),
    }


class ORCABinnedEngine:
    """Binned ORCA arm: statistical pure-Poisson chi^2 and analytic 30-long gradient.

    theta and grad are both in the order of ``nuisance_names``.
    """

    def __init__(self, response_npz, obs900, mu900, few, nuisance_names, norm, fluxq):
        r = np.load(response_npz, allow_pickle=True)

        # the response must have the flat 900-bin layout
        schema = str(np.asarray(r["schema_version"]).item()
                     if np.asarray(r["schema_version"]).ndim == 0
                     else np.asarray(r["schema_version"])[()])
        class_axis = [str(x) for x in np.asarray(r["class_axis"])]
        self.n_bins = int(r["n_bins"])
        if schema != SCHEMA_FLAT900:
            raise AssertionError(
                f"response schema {schema!r} != {SCHEMA_FLAT900!r}: this engine "
                "requires the flat900 layout")
        if self.n_bins != N_BINS_ORCA:
            raise AssertionError(f"n_bins {self.n_bins} != {N_BINS_ORCA}")
        if class_axis != ["pdg", "current"]:
            raise AssertionError(f"class_axis {class_axis} != ['pdg', 'current']")

        Rk = np.asarray(r["R_k"], np.int64)
        Re = np.asarray(r["R_e"], np.int64)
        Rz = np.asarray(r["R_z"], np.int64)
        self.Rb = np.asarray(r["R_b"], np.int64)
        # NORM is applied once, at load: the response stores the raw release weight.
        self.norm = float(norm)
        self.Rvn = np.asarray(r["R_v"], float) * self.norm

        self.classes = np.asarray(r["classes"], np.int64)      # (n_cls, 2) pdg,current
        self.n_cls = int(self.classes.shape[0])
        self.n_etrue = int(r["n_etrue"])
        self.n_cztrue = int(r["n_cztrue"])
        e_c = np.asarray(r["e_true_centers"], float)
        z_c = np.asarray(r["cz_true_centers"], float)

        # populated cell list: unique (class, ie, iz)
        key = (Rk * self.n_etrue + Re) * self.n_cztrue + Rz
        uniq, self.cell_of_nnz = np.unique(key, return_inverse=True)
        self.cell_of_nnz = np.asarray(self.cell_of_nnz, np.int64).ravel()
        self.n_cell = int(uniq.size)
        c_k = (uniq // (self.n_etrue * self.n_cztrue)).astype(np.int64)
        rem = uniq % (self.n_etrue * self.n_cztrue)
        c_ie = (rem // self.n_cztrue).astype(np.int64)
        c_iz = (rem % self.n_cztrue).astype(np.int64)
        self.c_k, self.c_ie, self.c_iz = c_k, c_ie, c_iz
        c_pdg = self.classes[c_k, 0]
        c_cc = self.classes[c_k, 1]
        # phi[ntype, flavor, ie, iz] indices
        self.c_ntype = (c_pdg < 0).astype(np.int64)
        self.c_flavor = (np.abs(c_pdg) // 2 - 6).astype(np.int64)

        self.geom = build_cell_geometry(e_c[c_ie], z_c[c_iz], c_pdg, c_cc, fluxq)

        # dial registry: the shared fields + the 4 ORCA cell-axis detector fields
        self.registry = dict(bdf.FIELDS)
        self.registry.update(_orca_detector_fields(self.geom))

        self.names = list(nuisance_names)
        if set(self.names) != set(ORCA_MANIFEST):
            missing = sorted(set(ORCA_MANIFEST) - set(self.names))
            extra = sorted(set(self.names) - set(ORCA_MANIFEST))
            raise AssertionError(
                f"nuisance_names is not the ORCA manifest: missing {missing}, "
                f"extra {extra}")
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.n_dials = len(self.names)
        self.i_eshift = self.idx["E_shift"]
        self.i_hpt = self.idx["f_HPT"]
        self.i_shower = self.idx["f_Shower"]
        self.cell_dials = [n for n in self.names
                           if n not in BIN_AXIS_DIALS and n not in MIGRATION_DIALS]
        assert len(self.cell_dials) == 27, len(self.cell_dials)

        # observation / muon / few
        self.obs = np.asarray(obs900, float)
        self.mu = np.asarray(mu900, float)
        self.few = np.asarray(few, bool)
        for nm, a in (("obs900", self.obs), ("mu900", self.mu), ("few", self.few)):
            if a.shape != (self.n_bins,):
                raise AssertionError(f"{nm} shape {a.shape} != ({self.n_bins},)")

        # bin-axis blocks: b // 300 == pid
        b_block = np.arange(self.n_bins) // BLOCK
        assert int(b_block.max()) == N_PID - 1
        self.m_hpt_bins = b_block == 1                 # f_HPT, pid 1
        self.m_shower_bins = b_block == 0              # f_Shower, pid 0
        # pid == 2 receives neither dial: D_b == 1 there.

        # class-N fast path: a dial is fast-pathed only if its cell mask is measured
        # constant within every class on this response; otherwise it goes through the
        # general n_cell-length dot.
        self.class_mask = {}
        first_of_class = np.zeros(self.n_cls, np.int64)
        for k in range(self.n_cls):
            first_of_class[k] = int(np.flatnonzero(c_k == k)[0])
        for name in self.cell_dials:
            fld = self.registry[name]
            if fld.algebra != "N" or fld.mask_fn is None:
                continue
            m = np.asarray(fld.mask_fn(self.geom), bool)
            km = m[first_of_class]
            if np.array_equal(m, km[c_k]):
                self.class_mask[name] = km

    # ------------------------------------------------------------------ utils
    def _theta_checked(self, theta):
        th = np.asarray(theta, float).ravel()
        if th.size != self.n_dials:
            raise AssertionError(f"theta length {th.size} != {self.n_dials}")
        # the precomputed response encodes energy scale 1
        if th[self.i_eshift] != 1.0:
            raise AssertionError(
                f"E_shift = {th[self.i_eshift]!r} but the binned response encodes "
                "energy scale 1 exactly; a moved E_shift would be silently ignored.")
        return th

    def _phi_cells(self, phi):
        p = np.asarray(phi, float)
        want = (2, 3, self.n_etrue, self.n_cztrue)
        if p.shape != want:
            raise AssertionError(f"phi shape {p.shape} != {want}")
        return p[self.c_ntype, self.c_flavor, self.c_ie, self.c_iz]

    # ------------------------------------------------------------------- model
    def cell_weights(self, phi, theta):
        """W_c = phi(cell) * prod over the 27 cell-axis dials of f_d(c).

        Here phi is the oscillated flux (flux x P), so NC cells keep their phi.
        """
        th = self._theta_checked(theta)
        W = self._phi_cells(phi).astype(float, copy=True)
        factors = {}
        for name in self.cell_dials:
            f = self.registry[name].factor_fn(self.geom, float(th[self.idx[name]]))
            factors[name] = f
            W = W * f
        return W, factors

    def _detector_bin_factor(self, th):
        """D_b, the per-bin morphology factor, keyed on b // 300 == pid."""
        D = np.ones(self.n_bins)
        D[self.m_hpt_bins] = th[self.i_hpt]
        D[self.m_shower_bins] = th[self.i_shower]
        return D

    def expectation(self, phi, theta, return_parts=False):
        """E_b = D_b * S_b + mu_b, the 900-long model rate."""
        th = self._theta_checked(theta)
        W, factors = self.cell_weights(phi, th)
        Wflat = W[self.cell_of_nnz]
        S = contract(self.Rb, self.Rvn, Wflat, self.n_bins)
        D = self._detector_bin_factor(th)
        E = D * S + self.mu
        if not return_parts:
            return E
        parts = {"S": S, "D": D, "W": W, "factors": factors}
        return E, parts

    # --------------------------------------------------------------------- chi2
    def chi2(self, phi, theta):
        """Statistical pure-Poisson chi^2 on the `few` mask, no prior."""
        E = self.expectation(phi, theta)
        n_mod = E[self.few]
        if np.any(n_mod <= 0):                 # barrier before any log
            return BARRIER_CHI2
        return float(poisson_chi2(self.obs[self.few], n_mod))

    def chi2_and_grad(self, phi, theta):
        """(chi2, grad), both statistical; grad is 30-long.

        The gradient is first-order: `resid` and `D` are held fixed while the cell-axis
        dials are differentiated.
        """
        th = self._theta_checked(theta)
        E, parts = self.expectation(phi, th, return_parts=True)
        n_mod = E[self.few]
        if np.any(n_mod <= 0):
            # barrier value with a zero statistical gradient; the caller adds the prior
            # gradient
            return BARRIER_CHI2, np.zeros(self.n_dials)

        chi2 = float(poisson_chi2(self.obs[self.few], n_mod))

        resid = np.zeros(self.n_bins)
        resid[self.few] = 2.0 * (1.0 - self.obs[self.few] / E[self.few])
        grad = self._adjoint(th, parts, resid)
        return chi2, grad

    # ----------------------------------------------------------------- adjoint
    def _adjoint(self, th, parts, v):
        W, D, S = parts["W"], parts["D"], parts["S"]
        factors = parts["factors"]

        # one bincount, O(nnz); each cell dial is then a dot
        u = adjoint_cells(self.cell_of_nnz, self.Rb, self.Rvn, v * D, self.n_cell)
        uW = u * W

        grad = np.zeros(self.n_dials)

        # class-N dials: one scalar per class, computed once
        s_k = np.bincount(self.c_k, weights=uW, minlength=self.n_cls)

        for name in self.cell_dials:
            x = float(th[self.idx[name]])
            km = self.class_mask.get(name)
            if km is not None:
                # a sum over an empty class set is exactly 0.0
                grad[self.idx[name]] = float(s_k[km].sum()) / x
            else:
                g = self.registry[name].dlnw_fn(self.geom, x, factors[name])
                grad[self.idx[name]] = float(np.dot(uW, g)) if np.ndim(g) else \
                    float(g) * float(uW.sum())

        # bin-axis dials: dE_b/df = S_b on the dial's own 300-bin block, 0 elsewhere
        vS = v * S
        grad[self.i_hpt] = float(vS[self.m_hpt_bins].sum())
        grad[self.i_shower] = float(vS[self.m_shower_bins].sum())

        # mu enters E_b additively with no theta dependence: zero gradient.
        # E_shift is pinned and its slot is a literal 0.0.
        grad[self.i_eshift] = 0.0
        return grad
