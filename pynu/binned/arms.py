"""The three experiments as fit arms, built from a scan configuration.

Each arm owns its engine, its parameter order, its per-cell state and the inputs
it reads. Engines are pure functions of (oscillation tensor, nuisance vector):
they return the statistical chi^2 and its analytic gradient.

    SKArm    Super-Kamiokande I-V, 930 analysis bins, 4 detector eras.
             The engine adds its own Gaussian prior over its parameters.
    ICArm    IceCube-DeepCore 8-year release, 200 bins, detector hypersurfaces.
             Statistical chi^2 only; the driver adds the prior.
    ORCAArm  KM3NeT/ORCA6, 900 bins (430 fitted). Statistical chi^2 only.

Single-experiment fits (``fit_cell``) follow the protocol in the configuration's
``<fit>`` element.
The combined fit (``combined.py``) uses the arms' engines directly.
"""
import numpy as np
from scipy.optimize import minimize

from .sk_engine import SKBinnedEngine
from .ic_engine import ICBinnedEngine
from .orca_engine import ORCABinnedEngine

BOUND_NSIGMA = 5.0          # IC / ORCA / combined shared parameters: nominal +- 5 sigma
POS_FLOOR = 0.01            # lower bound of every positive-nominal parameter
SENTINEL_CHI2 = 1e8         # a fit at or above this is not used as the next seed


# =============================================================================
# shared rules
# =============================================================================
def driver_bounds(nominal, sigma):
    """nominal +- 5 sigma, lower bound raised to 0.01 for positive-nominal dials."""
    lo = nominal - BOUND_NSIGMA * sigma
    hi = nominal + BOUND_NSIGMA * sigma
    for k in range(lo.size):
        if nominal[k] > 0 and lo[k] < POS_FLOOR:
            lo[k] = POS_FLOOR
    return lo, hi


def prior_penalty(theta, nominal, sigma):
    return float(np.sum(((theta - nominal) / sigma) ** 2))


SK_FIRST = ("", "sk", "shared")          # SK engine parameters: SK's own slot first


def check_bounds(cfg, names, lo, hi, what, prefer=("", "shared", "sk")):
    """The derived bounds must equal the lo/hi written in the configuration."""
    _, _, xlo, xhi, pinned = cfg.dial_table(names, prefer)
    bad = [(n, a, b, c, d) for n, a, b, c, d, p in zip(names, lo, hi, xlo, xhi, pinned)
           if not p and (a != c or b != d)]
    if bad:
        rows = "\n".join(f"  {n}: derived [{a!r}, {b!r}]  xml [{c!r}, {d!r}]"
                         for n, a, b, c, d in bad[:20])
        raise ValueError(f"{cfg.path}: {what} bounds differ from the configuration "
                         f"for {len(bad)} dial(s):\n{rows}")


# =============================================================================
# Super-Kamiokande
# =============================================================================
class SKArm:
    """SK engine on the kernel "Ninv", all 930 bins fitted."""

    def __init__(self, cfg, check_names=None):
        """``check_names``: the parameters whose configured bounds must equal the SK
        engine box (default all; the combined fit passes its SK-only block)."""
        arm = cfg.arms["SK"]
        f = cfg.fit
        self.names = list(arm.dials)
        nominal, sigma, _, _, pinned = cfg.dial_table(self.names, SK_FIRST)
        eng = SKBinnedEngine(cfg.input("sk_response"), list(self.names),
                             fluxq=cfg.input("sk_fluxq"), fit_ftol=f["ftol"],
                             fit_maxiter=f["maxiter"])
        if list(eng.nuisance_names) != self.names:
            raise ValueError("SK engine parameter order differs from the configuration")
        names = self.names if check_names is None else list(check_names)
        rows = [self.names.index(n) for n in names]
        bad = [n for n, k in zip(names, rows) if eng.nominal[k] != nominal[k]]
        if bad:
            raise ValueError(f"SK engine nominal differs from the configuration for {bad}")
        eng.sigma[rows] = sigma[rows]            # the prior widths of record
        self.eng = eng
        self.nominal = np.asarray(eng.nominal, float).copy()
        self.free = ~pinned
        lo, hi = eng.fit_box()
        check_bounds(cfg, names, lo[rows], hi[rows], "SK engine", SK_FIRST)

    def fit_cell(self, phi, i_dcp, x0, npolish):
        """One delta_CP node: ``fit_point`` from x0, then ``npolish`` restarts from the
        running best, each kept only if strictly lower."""
        phi_node = phi[i_dcp]
        c, nu, nit, conv = np.inf, None, 0, False
        cc, nn, _n, _cv = self.eng.fit_point(phi_node, x0, self.free)
        if cc < c:
            c, nu, nit, conv = cc, nn, _n, _cv
        for _ in range(npolish):
            pc, pn, _n2, _c2 = self.eng.fit_point(phi_node, nu, self.free)
            if pc < c:
                c, nu, nit, conv = pc, pn, _n2, _c2
        return dict(chi2=float(c), parts=dict(sk=float(c)), converged=bool(conv),
                    n_iter=int(nit)), np.asarray(nu, float)


# =============================================================================
# IceCube-DeepCore
# =============================================================================
class ICArm:
    """IC engine + the detector hypersurfaces, which depend on |Delta m^2_31| only."""

    def __init__(self, cfg, pinned=None):
        arm = cfg.arms["IC"]
        self.names = list(arm.dials)
        aux = np.load(cfg.input("ic_aux"), allow_pickle=False)
        self.obs = np.asarray(aux["obs"], float)
        self.mu = np.asarray(aux["mu"], float)
        self.norm = float(aux["norm"])
        self.few = self.obs > float(aux["min_entries"])
        self.hs_grid = np.asarray(aux["hs_dm31_grid"], float)
        self.hs_keys = ["intercept"] + [str(s) for s in aux["hs_slope_names"]]
        self.hs = {str(c): {k: np.asarray(aux[f"hs_{c}_{k}"], float) for k in self.hs_keys}
                   for c in aux["hs_categories"]}
        if pinned is None:
            pinned = [n for n in self.names if cfg.dial(n).pinned]
        self.eng = ICBinnedEngine(
            cfg.input("ic_response"), self.obs, self.mu, self.names, norm=self.norm,
            few=self.few, pinned=tuple(pinned), fluxq=cfg.input("ic_fluxq"))

    def interpolate_hs(self, dm31):
        """Piecewise-linear interpolation of the hypersurface tables in Delta m^2_31,
        clamped at the ends of the tabulated range."""
        grid = self.hs_grid
        if dm31 <= grid[0]:
            lo, hi, frac = 0, 0, 0.0
        elif dm31 >= grid[-1]:
            lo, hi, frac = len(grid) - 1, len(grid) - 1, 0.0
        else:
            hi = np.searchsorted(grid, dm31)
            lo = hi - 1
            frac = (dm31 - grid[lo]) / (grid[hi] - grid[lo])
        return {c: {k: t[k][lo] + frac * (t[k][hi] - t[k][lo]) for k in self.hs_keys}
                for c, t in self.hs.items()}

    def bind(self, dm31):
        """Per-cell state: the hypersurfaces at |Delta m^2_31| (tabulated in |dm| only)."""
        h = abs(float(dm31))
        if self.eng.hs_dm31 != h:
            self.eng.set_hs_slopes(self.interpolate_hs(h), h)


# =============================================================================
# KM3NeT/ORCA6
# =============================================================================
class ORCAArm:
    def __init__(self, cfg):
        arm = cfg.arms["ORCA"]
        self.names = list(arm.dials)
        aux = np.load(cfg.input("orca_aux"), allow_pickle=False)
        self.obs = np.asarray(aux["obs"], float)
        self.mu = np.asarray(aux["mu"], float)
        self.few = np.asarray(aux["few"], bool)
        self.norm = float(aux["norm"])
        self.eng = ORCABinnedEngine(
            cfg.input("orca_response"), self.obs, self.mu, self.few, self.names,
            norm=self.norm, fluxq=cfg.input("orca_fluxq"))

    def bind(self, dm31):
        pass


# =============================================================================
# single-experiment fit for IC and ORCA (the driver owns the prior)
# =============================================================================
class DriverFit:
    """Minimize statistical chi^2 + Gaussian prior over the arm's free dials."""

    def __init__(self, cfg, arm):
        self.arm = arm
        self.names = arm.names
        self.nominal, self.sigma, _, _, pinned = cfg.dial_table(self.names)
        self.swept = np.array([k for k, p in enumerate(pinned) if not p], np.int64)
        lo, hi = driver_bounds(self.nominal[self.swept], self.sigma[self.swept])
        check_bounds(cfg, [self.names[k] for k in self.swept], lo, hi, "driver")
        self.bounds = list(zip(lo, hi))
        f = cfg.fit
        self.options = {"ftol": f["ftol"], "gtol": f["gtol"], "maxiter": f["maxiter"]}
        self.npolish = f["npolish"]
        self.min_gain = f.get("polish_min_gain", 1e-3)

    def _theta(self, x):
        th = self.nominal.copy()
        th[self.swept] = x
        return th

    def _minimize(self, phi, x0):
        eng, nominal, sigma = self.arm.eng, self.nominal, self.sigma
        n = {"n": 0}

        def fg(x):
            th = self._theta(x)
            c, g = eng.chi2_and_grad(phi, th)
            n["n"] += 1
            return (float(c) + prior_penalty(th, nominal, sigma),
                    (np.asarray(g, float) + 2.0 * (th - nominal) / sigma ** 2)[self.swept])
        res = minimize(fg, np.asarray(x0, float), method="L-BFGS-B", jac=True,
                       bounds=self.bounds, options=self.options)
        th = self._theta(res.x)
        return (float(res.fun), th, float(eng.chi2(phi, th)),
                prior_penalty(th, nominal, sigma), bool(res.success), n["n"])

    def fit_cell(self, phi, x0=None):
        """Cold start at nominal unless x0 is given; then up to ``npolish`` restarts,
        stopping at the first that does not gain at least ``polish_min_gain``."""
        seed = self.nominal[self.swept] if x0 is None else np.asarray(x0, float)
        f, th, cs, pen, ok, nev = self._minimize(phi, seed)
        for _ in range(max(0, self.npolish)):
            pf, pth, pc, pp, pok, pnev = self._minimize(phi, th[self.swept])
            nev += pnev
            if pf < f - self.min_gain:
                f, th, cs, pen, ok = pf, pth, pc, pp, pok
            else:
                break
        if not np.isfinite(f) or cs < 0.0 or pen < 0.0:
            raise RuntimeError(f"invalid fit result chi2={f!r} stat={cs!r} prior={pen!r}")
        return dict(chi2=f, parts=dict(stat=cs, prior=pen), converged=ok,
                    n_eval=nev), th
