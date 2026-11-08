"""Combined Super-Kamiokande + IceCube-DeepCore + KM3NeT/ORCA6 fit.

The fit vector has two blocks, in the order of the configuration's ``<dials>``:

  shared   one slot per flux, cross-section or detector parameter of IC/ORCA and
           every flux/cross-section parameter SK shares with them. Each slot gets
           ONE Gaussian prior (the ``sigma`` of the dial) and bounds nominal +- 5
           sigma (lower bound >= 0.01 for positive-nominal parameters).
  sk       the SK-only parameters (detector systematics, SK-specific shapes). Their
           prior is the SK engine's own; bounds are the SK engine's box.

    chi^2 = chi^2_IC(shared) + chi^2_ORCA(shared) + chi^2_SK(shared, sk) + prior(shared)

SK's engine also adds a prior over every parameter it sees; for the shared ones
its width is set to 1e12 so that each shared parameter is penalised exactly once.
Pinned parameters keep their nominal value (bounds lo = hi = nominal). The fit is
one L-BFGS-B minimization at a fixed (Delta m^2_31, sin^2 theta_23, delta_CP)
lattice node with analytic gradients from all three engines, plus ``npolish``
restarts kept only if lower.
"""
import numpy as np
from scipy.optimize import minimize

from .arms import SKArm, ICArm, ORCAArm, driver_bounds, check_bounds, prior_penalty

SK_SHARED_SIGMA = 1e12      # SK engine width of the shared parameters (prior counted once)
BARRIER_CHI2 = 9e9          # an engine's chi^2 when a fitted bin's model rate is <= 0
BARRIER_MAX_RUN = 40        # consecutive barrier evaluations tolerated in one fit


class CombinedFit:
    def __init__(self, cfg):
        self.cfg = cfg
        dials = cfg.dials
        self.flat_names = [d.name for d in dials]
        self.union_names = [d.name for d in dials if d.block == "shared"]
        self.local_names = [d.name for d in dials if d.block == "sk"]
        n_union = len(self.union_names)
        if self.flat_names != self.union_names + self.local_names:
            raise ValueError("combined <dials> must list every block='shared' dial "
                             "before every block='sk' dial")
        self.n_union = n_union
        upos = {n: i for i, n in enumerate(self.union_names)}
        self.nominal, self.sigma, _, _, pinned_u = cfg.dial_table(self.union_names,
                                                                 ("shared",))
        self.pinned_union = [n for n, p in zip(self.union_names, pinned_u) if p]

        # ---- the three engines
        self.sk = SKArm(cfg, check_names=self.local_names)
        self.ic = ICArm(cfg, pinned=self.pinned_union)
        self.orca = ORCAArm(cfg)
        eng = self.sk.eng
        sk_names = self.sk.names
        # an SK parameter uses its block='sk' slot if it has one, else the shared slot
        local_set = set(self.local_names)
        shared_sk = [n for n in sk_names if n not in local_set]
        if not set(shared_sk) <= set(upos) or not local_set <= set(sk_names):
            raise ValueError("every SK parameter needs a block='sk' or a shared slot")
        for n in shared_sk:
            eng.sigma[sk_names.index(n)] = SK_SHARED_SIGMA

        # ---- scatter maps: flat slot feeding each engine parameter
        lpos = {n: n_union + k for k, n in enumerate(self.local_names)}
        self.sk_from_flat = np.array([lpos[n] if n in lpos else upos[n] for n in sk_names],
                                     np.int64)
        for arm in (self.ic, self.orca):
            missing = [n for n in arm.names if n not in upos]
            if missing:
                raise ValueError(f"{arm.__class__.__name__} parameters without a shared "
                                 f"slot: {missing}")
        self.ic_proj = np.array([upos[n] for n in self.ic.names], np.int64)
        self.orca_proj = np.array([upos[n] for n in self.orca.names], np.int64)
        ic_mov = [n for n in self.ic.names if n not in self.pinned_union]
        orca_mov = [n for n in self.orca.names if n not in self.pinned_union]
        self.ic_slots = np.array([upos[n] for n in ic_mov], np.int64)
        self.ic_rows = np.array([self.ic.names.index(n) for n in ic_mov], np.int64)
        self.orca_slots = np.array([upos[n] for n in orca_mov], np.int64)
        self.orca_rows = np.array([self.orca.names.index(n) for n in orca_mov], np.int64)

        # ---- bounds and start values
        lo_u, hi_u = driver_bounds(self.nominal.copy(), self.sigma)
        check_bounds(cfg, self.union_names, lo_u, hi_u, "shared", ("shared",))
        lo_u[pinned_u] = self.nominal[pinned_u]
        hi_u[pinned_u] = self.nominal[pinned_u]
        lo_sk, hi_sk = eng.fit_box()
        rows = np.array([sk_names.index(n) for n in self.local_names], np.int64)
        lo_l, hi_l = lo_sk[rows].copy(), hi_sk[rows].copy()
        pin_l = ~self.sk.free[rows]
        nom_l = np.asarray(eng.nominal, float)[rows]
        lo_l[pin_l] = nom_l[pin_l]
        hi_l[pin_l] = nom_l[pin_l]
        self.bounds = list(zip(np.concatenate([lo_u, lo_l]), np.concatenate([hi_u, hi_l])))
        f = cfg.fit
        self.options = {"ftol": f["ftol"], "gtol": f["gtol"], "maxiter": f["maxiter"]}
        self.npolish = f["npolish"]

    def start(self, ordering):
        x = np.array([self.cfg.dial(n, ("shared",)).start_value(ordering)
                      for n in self.union_names]
                     + [self.cfg.dial(n, ("sk",)).start_value(ordering)
                        for n in self.local_names], float)
        for k, (a, b) in enumerate(self.bounds):
            if a == b:
                x[k] = a
        return x

    # ------------------------------------------------------------------ per cell
    def bind(self, dm31, sk_phi, ic_phi, orca_phi):
        self.ic.bind(dm31)
        self.sk_phi, self.ic_phi, self.orca_phi = sk_phi, ic_phi, orca_phi

    def chi2_and_grad(self, x):
        """(f, gradient, (chi2_IC, chi2_ORCA, chi2_SK, prior)) at the flat vector x."""
        xu = x[:self.n_union]
        c_orca, g_orca = self.orca.eng.chi2_and_grad(self.orca_phi, xu[self.orca_proj])
        c_ic, g_ic = self.ic.eng.chi2_and_grad(self.ic_phi, xu[self.ic_proj])
        c_sk, g_sk = self.sk.eng.chi2_and_grad(self.sk_phi, x[self.sk_from_flat])
        pen = prior_penalty(xu, self.nominal, self.sigma)
        f = float(c_ic) + float(c_orca) + float(c_sk) + float(pen)
        g = np.zeros(x.size)
        np.add.at(g, self.sk_from_flat, np.asarray(g_sk, float))
        g[:self.n_union] += 2.0 * (xu - self.nominal) / self.sigma ** 2
        np.add.at(g, self.orca_slots, np.asarray(g_orca, float)[self.orca_rows])
        np.add.at(g, self.ic_slots, np.asarray(g_ic, float)[self.ic_rows])
        return f, g, (float(c_ic), float(c_orca), float(c_sk), float(pen))

    def fit_cell(self, x0):
        """One lattice node: L-BFGS-B from x0 plus ``npolish`` restarts kept if lower."""
        run = {"barrier": 0}

        def objective(x):
            f, g, _ = self.chi2_and_grad(x)
            if not np.isfinite(f) or not np.all(np.isfinite(g)):
                raise FloatingPointError(f"non-finite objective f={f!r}")
            if f >= BARRIER_CHI2:
                run["barrier"] += 1
                if run["barrier"] > BARRIER_MAX_RUN:
                    raise RuntimeError("fit stalled on the chi^2 barrier: a fitted bin's "
                                       "model rate is <= 0 and the line search cannot "
                                       "back off")
            else:
                run["barrier"] = 0
            return f, g

        r = minimize(objective, np.asarray(x0, float).copy(), method="L-BFGS-B",
                     jac=True, bounds=self.bounds, options=self.options)
        for _ in range(self.npolish):
            p = minimize(objective, r.x.copy(), method="L-BFGS-B", jac=True,
                         bounds=self.bounds, options=self.options)
            if p.fun < r.fun:
                r = p
        x = r.x.copy()
        _f, _g, (c_ic, c_orca, c_sk, pen) = self.chi2_and_grad(x)
        chi2 = float(c_ic + c_orca + c_sk + pen)
        if not np.isfinite(chi2) or chi2 < 0:
            raise RuntimeError(f"invalid combined chi2 {chi2!r}")
        return dict(chi2=chi2, parts=dict(ic=c_ic, orca=c_orca, sk=c_sk, prior=pen),
                    converged=bool(r.success), n_iter=int(r.nit)), x
