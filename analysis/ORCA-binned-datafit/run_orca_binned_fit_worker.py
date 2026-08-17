#!/usr/bin/env python3
"""Single-experiment ORCA binned fit — SLURM row worker.

One array task = one Delta-m^2 row: a nuisance fit at every sin^2(theta23) point
along that row, driven by `ORCABinnedEngine`'s analytic 30-long dial gradient, and
written as one row json.

============================== DIVISION OF LABOUR ==============================
★ THE ENGINE IS STAT-ONLY. `ORCABinnedEngine.chi2_and_grad` returns a pure-Poisson
chi2 and its stat gradient and NOTHING ELSE — no Gaussian prior. THIS DRIVER owns
the prior, exactly as the combined workers do:

    prior       = sum_d ((theta_d - nominal_d) / sigma_d)^2
    d(prior)/dx = 2 (theta - nominal) / sigma^2

mirrored from `combined_ic_orca_fit_worker.py:242-246` (`prior_penalty`, the value)
and `combined_3exp_fit_worker.py:1331` (`g[:n_union] += 2.0*(theta-nominal)/sigma**2`,
the gradient). Bounds mirror `combined_ic_orca_fit_worker.py:231-239`
(`make_bounds`: nominal +- 5 sigma, positive-nominal dials floored at 0.01), and the
minimizer settings mirror `:149-153`. Adding a prior inside the engine instead would
double-count silently AND still pass every identity gate — see the engine docstring's
risk R2.

(nominal, sigma) come from the LIVE PyNuFit analysis
(`Analysis.NuisNominalList` / `NuisSigmaList`), which is how the combined worker
builds its union prior table (`combined_ic_orca_fit_worker.py:405-419`). They are
never retyped here.

================================ TWO PINNED ANCHORS ============================
Both are production conventions this driver PRESERVES, not choices it makes.

(1) ★ E_shift IS PINNED AT 1.0. The precomputed response encodes ENERGY_SCALE = 1,
    so a moved E_shift would be silently IGNORED rather than error. The engine
    hard-asserts `theta[E_shift] == 1.0` on every evaluation and writes its gradient
    slot a literal 0.0; production pins it at `combined_ic_orca_fit_worker.py:120`.
    This driver therefore drops E_shift from the swept set (its bound is never
    built, it stays at nominal, and it contributes exactly 0 to the prior).

(2) ★ THE FIT MASK IS MC-SUPPORT, NOT `obs > MIN_ENTRIES`. Production takes
    `few_orca = orca_exp.FewEntries` (`combined_ic_orca_fit_worker.py:377`), and the
    ORCA XML's <DataFiles> block carries `<status> 0 </status>`, so `DataFit` is
    False and `Experiment.SetObservedBinned` takes its ELSE branch: the
    "observation" defining the mask is the MC rate, giving "bins the ORCA MC can
    populate" (430 of 900 on the production MC) rather than the 427 bins with data.
    The DATA still enters chi2 — this driver supplies `obs` itself from the data
    parquet via `orca_binned_support.observed_900`, exactly as production does, and
    never reads `exp.ObservedBinned`. Fitting on `obs > MIN_ENTRIES` instead would
    compare against production on a DIFFERENT bin set. This driver reads the mask
    off the live experiment so it cannot drift from that definition.

=================================== PHI SOURCE =================================
`--phi-source tensors` (default) reads one prebuilt tensor npz per grid point
(`build_orca_osc_tensors.py`), which is what makes a cold clone reproducible: the
nuSQuIDS dependency is confined to the build step. The tensor's stored `dm231`/`s23`
are cross-checked against the grid point being fitted, so a mis-indexed tensor
directory is an error rather than a smooth, plausible, wrong surface.

`--phi-source live` instead re-derives the cell flux per grid point from the live
event path — `ApplyOscillations("Physics")` then the MC-weight-weighted cell mean
(`orca_cell_phi.extract_cell_phi`). This is the PRODUCTION convention and the two
routes are NOT identical: nuSQuIDS randomizes each event's production height, so
the live per-cell flux is a weighted mean over a stochastic sample while the tensor
is a single deterministic evaluation at the cell centre. Use `live` to reproduce
production numbers; use `tensors` for a self-contained cluster scan.

Oscillation averaging must be OFF on both routes (the engine asserts
`osc.osc_avg_scale is None`): with averaging on, the per-cell flux stops being an
exact reduction of the event path.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

# ---- pin the Pynu tree this worker ships inside (analysis/ORCA-binned-datafit
# -> ../../ = the Pynu repo root), so the worker is runnable standalone; the
# submission script may additionally export PYTHONPATH explicitly. ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYNU_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PYNU_DIR not in sys.path:
    sys.path.insert(0, PYNU_DIR)

# --- minimizer settings: combined_ic_orca_fit_worker.py:149-153, VERBATIM ---
MINIMIZER_METHOD = "L-BFGS-B"
MINIMIZER_OPTIONS = {"ftol": 1e-5, "gtol": 1e-5, "maxiter": 200}
MINIMIZER_BOUND_NSIGMA = 5.0
MINIMIZER_POS_FLOOR = 0.01

# --- default scan box: the shared IC/ORCA grid (combined_ic_orca_fit_worker.py:139-141)
DM_MIN, DM_MAX = 2.3e-3, 2.7e-3
S23_MIN, S23_MAX = 0.40, 0.65

DEFAULT_PARQUET = os.path.join(PYNU_DIR, "data", "ORCA",
                               "ORCA_MC_dataverse_with_muons.parquet")
DEFAULT_DATA = os.path.join(PYNU_DIR, "data", "ORCA",
                            "ORCA_data_dataverse.parquet")


# ==============================================================================
# prior + bounds — OWNED BY THE DRIVER (see module docstring)
# ==============================================================================

def prior_penalty(theta, nominal, sigma):
    """Sum((theta-nominal)/sigma)^2 over ALL manifest dials.

    combined_ic_orca_fit_worker.py:242-246 verbatim. Pinned dials sit at nominal,
    so they contribute exactly 0 and need no special case.
    """
    return float(np.sum(((theta - nominal) / sigma) ** 2))


def prior_grad(theta, nominal, sigma):
    """d/dtheta of prior_penalty. combined_3exp_fit_worker.py:1331 verbatim
    (`2.0 * (theta_union - nominal) / sigma ** 2`)."""
    return 2.0 * (theta - nominal) / sigma ** 2


def make_bounds(nominal, sigma, swept_idx):
    """nominal +- 5 sigma, positive-nominal dials floored at 0.01, restricted to
    the swept slots. combined_ic_orca_fit_worker.py:231-239 verbatim."""
    lo = nominal[swept_idx] - MINIMIZER_BOUND_NSIGMA * sigma[swept_idx]
    hi = nominal[swept_idx] + MINIMIZER_BOUND_NSIGMA * sigma[swept_idx]
    nom = nominal[swept_idx]
    for k in range(lo.size):
        if nom[k] > 0 and lo[k] < MINIMIZER_POS_FLOOR:
            lo[k] = MINIMIZER_POS_FLOOR
    return list(zip(lo, hi))


def build_theta(x_swept, nominal, swept_idx):
    """Insert the swept sub-vector into a nominal-seeded full manifest vector.
    Pinned dials keep nominal (and so add 0 to the prior).
    combined_ic_orca_fit_worker.py:204-210 verbatim."""
    th = nominal.copy()
    th[swept_idx] = x_swept
    return th


# ==============================================================================
# phi
# ==============================================================================

def load_phi_tensor(path, dm, s23, dcp_node, rtol=1e-9):
    """Read one prebuilt tensor npz and return the (2, 3, nE, nZ) slice.

    Cross-checks the stored (dm231, s23) against the grid point being fitted. A
    tensor directory indexed one step off produces a perfectly smooth, perfectly
    plausible, WRONG surface — nothing downstream looks anomalous — so this is a
    hard error, not a warning.
    """
    r = np.load(path, allow_pickle=True)
    phi = np.asarray(r["phi"], float)
    if phi.ndim != 5:
        raise SystemExit(f"{path}: phi ndim {phi.ndim} != 5 "
                         "(expect [n_dcp, 2, 3, nE, nZ])")
    got_dm = float(np.asarray(r["dm231"]).reshape(-1)[0])
    got_s23 = float(np.asarray(r["s23"]).reshape(-1)[0])
    if not (np.isclose(got_dm, dm, rtol=rtol, atol=0.0)
            and np.isclose(got_s23, s23, rtol=rtol, atol=0.0)):
        raise SystemExit(
            f"{path}: tensor was built at (dm231={got_dm:.8e}, s23={got_s23:.8f}) "
            f"but this cell is (dm231={dm:.8e}, s23={s23:.8f}) — the phi pattern "
            "is indexing the wrong grid point")
    if not (0 <= dcp_node < phi.shape[0]):
        raise SystemExit(f"{path}: --dcp-node {dcp_node} outside the tensor's "
                         f"{phi.shape[0]} dCP slice(s)")
    return phi[dcp_node]


def live_phi(ctx, dm, s23, s13, dcp):
    """PRODUCTION route: stage the oscillation on the live experiment and reduce
    exp.PhysicsWeight to the (2, 3, nE, nZ) cell flux by the MC-weight-weighted
    cell mean — the unique reduction conserving each cell's total contribution
    (orca_cell_phi.extract_cell_phi). Mirrors set_cell_physics
    (combined_ic_orca_fit_worker.py:306-321) + refresh_binned_cell_state
    (combined_3exp_fit_worker.py:1244-1248)."""
    from pynu.Experiments.orca_cell_phi import extract_cell_phi
    osc = ctx["osc"]
    osc.Parameters["Dm231"] = dm
    if "Dm231_bar" in osc.Parameters:
        osc.Parameters["Dm231_bar"] = dm
    osc.Parameters["Sin2Theta23"] = s23
    if s13 is not None and "Sin2Theta13" in osc.Parameters:
        osc.Parameters["Sin2Theta13"] = s13
    if dcp is not None and "dCP" in osc.Parameters:
        osc.Parameters["dCP"] = dcp
    osc.reset_cache()
    ctx["pynufit"].StartPhysics()
    ctx["pynufit"].ApplyOscillations("Physics")
    phi, _info = extract_cell_phi(ctx["exp"], ctx["cell_index"], osc=osc)
    return np.asarray(phi, float)


# ==============================================================================
# the fit
# ==============================================================================

def fit_point(eng, phi, nominal, sigma, swept_idx, bounds, x0):
    """One L-BFGS-B minimization of stat chi2 (engine, analytic grad) + prior
    (this driver) over the swept slots. Returns (chi2_total, theta_full,
    chi2_stat, chi2_prior, converged, nfev, t_wall)."""
    n_eval = {"n": 0}

    def fg(x_sw):
        th = build_theta(x_sw, nominal, swept_idx)
        c_stat, g_stat = eng.chi2_and_grad(phi, th)       # STAT-ONLY, by contract
        f = float(c_stat) + prior_penalty(th, nominal, sigma)
        g = np.asarray(g_stat, float) + prior_grad(th, nominal, sigma)
        n_eval["n"] += 1
        return f, g[swept_idx]

    t0 = time.time()
    res = minimize(fg, np.asarray(x0, float), method=MINIMIZER_METHOD, jac=True,
                   bounds=bounds, options=MINIMIZER_OPTIONS)
    t_wall = time.time() - t0
    th = build_theta(res.x, nominal, swept_idx)
    c_stat = float(eng.chi2(phi, th))                     # re-eval to split terms
    pen = prior_penalty(th, nominal, sigma)
    return (float(res.fun), th, c_stat, pen, bool(res.success), n_eval["n"], t_wall)


# ==============================================================================
# setup
# ==============================================================================

def orca_manifest():
    """The 30-name ORCA manifest, in the XML order the engine returns its
    gradient in. It is a certified constant of the ported package: it lives with
    the engine, with `binned_dial_fields` as the documented alternative home
    (port plan section 3.3) — resolved here so either placement works."""
    from pynu.Experiments import orca_binned_engine as OE
    m = getattr(OE, "ORCA_MANIFEST", None)
    if m is None:
        from pynu.Experiments import binned_dial_fields as bdf
        m = bdf.ORCA_MANIFEST
    return list(m)


def build_ctx(args):
    """One PyNuFit init + one engine construction for the whole row."""
    from pynu import PyNuFit
    from pynu.Experiments.orca_binned_engine import ORCABinnedEngine, ORCA_PINNED
    from pynu.Experiments import orca_binned_support as S

    pynufit = PyNuFit(args.config, verbosity=False)
    exp_name = list(pynufit.Experiments.keys())[0]
    exp = pynufit.Experiments[exp_name]
    osc = pynufit.physics_tunes[exp_name].OscillationTunes
    # Averaging OFF: the engine asserts this (risk R1) and production sets it at
    # combined_ic_orca_fit_worker.py:378.
    osc.osc_avg_scale = None

    names = list(pynufit.Analysis.NuisanceList)
    # Guard mirrored from combined_3exp_fit_worker.py:1156-1160: the engine returns
    # its gradient in MANIFEST order, so a reordered XML would transpose it
    # silently. Order equality, not set equality (the engine checks the set).
    manifest = orca_manifest()
    if names != manifest:
        sys.exit("ORCA NuisanceList != engine manifest — the gradient would be "
                 f"transposed\n live: {names}\n eng : {manifest}")
    nominal = np.array(pynufit.Analysis.NuisNominalList, float)
    sigma = np.array(pynufit.Analysis.NuisSigmaList, float)
    if not (len(names) == nominal.size == sigma.size):
        sys.exit(f"manifest/nominal/sigma length mismatch: {len(names)} / "
                 f"{nominal.size} / {sigma.size}")
    if np.any(sigma <= 0):
        bad = [names[i] for i in np.flatnonzero(sigma <= 0)]
        sys.exit(f"non-positive prior sigma on {bad} — the Gaussian prior this "
                 "driver owns would divide by zero")

    # ---- observation and muon: the production definitions, from the parquets.
    # `exp.ObservedBinned` is NOT usable — Experiment.SetObservedBinned:250 masks
    # it down to FewEntries, and on this XML it is Asimov MC (DataFiles status 0).
    obs = S.observed_900(args.data)
    mu = S.muon_900(args.mc)

    # ---- the mask: MC-support, read off the live experiment (anchor 2) ----
    few = getattr(exp, "FewEntries", None)
    if few is None:
        sys.exit("exp.FewEntries is None — SetObservedBinned never ran, so the "
                 "production fit mask is unavailable")
    few = np.asarray(few, bool)
    if few.shape != obs.shape:
        sys.exit(f"exp.FewEntries shape {few.shape} != 900-bin obs {obs.shape}")
    print(f"[setup] fit mask: {int(few.sum())} of {few.size} bins "
          f"(MC-support; DataFit={getattr(exp, 'DataFit', None)}). "
          f"obs>MIN_ENTRIES would be {int((obs > getattr(exp, 'MIN_ENTRIES', 0.01)).sum())}")

    eng = ORCABinnedEngine(args.response, obs, mu, few, names,
                           norm=float(exp.NORM), osc=osc)

    # ---- pinned set: the engine's own hard pin (E_shift) + any --pin ----
    pinned = list(ORCA_PINNED) + [p for p in (args.pin or []) if p not in ORCA_PINNED]
    unknown = [p for p in pinned if p not in names]
    if unknown:
        sys.exit(f"--pin names not in the ORCA manifest: {unknown}")
    i_eshift = names.index("E_shift")
    if float(nominal[i_eshift]) != 1.0:
        sys.exit(f"E_shift nominal is {nominal[i_eshift]!r}, not 1.0 — the response "
                 "encodes ENERGY_SCALE = 1 exactly and the engine will refuse every "
                 "evaluation. Fix the XML rather than this worker.")
    swept = [n for n in names if n not in pinned]
    swept_idx = np.array([names.index(n) for n in swept], dtype=np.int64)

    ctx = dict(pynufit=pynufit, exp=exp, osc=osc, eng=eng, names=names,
               nominal=nominal, sigma=sigma, swept_idx=swept_idx, pinned=pinned,
               obs=obs, mu=mu, few=few)
    if args.phi_source == "live":
        from pynu.Experiments.orca_cell_phi import build_cell_index
        ctx["cell_index"] = build_cell_index(args.mc)
    print(f"[setup] {eng.summary()}")
    print(f"[setup] dials {len(names)} | swept {len(swept)} | pinned {pinned}")
    return ctx


def main():
    ap = argparse.ArgumentParser(
        description="Single-experiment ORCA binned fit: one Delta-m^2 row of "
                    "sin^2(theta23) points. task/array index = the row.")
    ap.add_argument("--row", type=int, required=True,
                    help="Delta-m^2 row index into the scan grid (= SLURM array index)")
    ap.add_argument("--config", required=True,
                    help="ORCA analysis XML carrying the 30-dial manifest "
                         "(the campaign's ORCA_Atm_r2_fude_ccqe.xml)")
    ap.add_argument("--response", required=True,
                    help="flat900 response npz from build_orca_binned_response.py")
    ap.add_argument("--mc", default=DEFAULT_PARQUET,
                    help="ORCA MC parquet (muon background + cell index)")
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="ORCA data parquet (the 900-bin observation)")
    # ---- phi ----
    ap.add_argument("--phi-source", choices=["tensors", "live"], default="tensors",
                    help="tensors = prebuilt per-point npz (default, cold-clone "
                         "reproducible); live = production weight-weighted cell "
                         "mean off exp.PhysicsWeight (see the module docstring — "
                         "the two are NOT identical)")
    ap.add_argument("--phi-dir", default=None,
                    help="directory of per-point tensors (--phi-source tensors)")
    ap.add_argument("--phi-pattern", default="orca_phi_{i:03d}_{j:03d}.npz",
                    help="tensor filename pattern, formatted with i=dm row, "
                         "j=s23 column")
    ap.add_argument("--dcp-node", type=int, default=0,
                    help="dCP slice index within each tensor npz")
    ap.add_argument("--dcp", type=float, default=None,
                    help="dCP in radians for --phi-source live (default: leave "
                         "the XML value)")
    ap.add_argument("--s13", type=float, default=None,
                    help="Sin2Theta13 for --phi-source live (default: XML value)")
    # ---- grid ----
    ap.add_argument("--ndm", type=int, default=20)
    ap.add_argument("--dm-min", type=float, default=DM_MIN)
    ap.add_argument("--dm-max", type=float, default=DM_MAX)
    ap.add_argument("--ns23", type=int, default=20)
    ap.add_argument("--s23-min", type=float, default=S23_MIN)
    ap.add_argument("--s23-max", type=float, default=S23_MAX)
    # ---- fit ----
    ap.add_argument("--npolish", type=int, default=2,
                    help="restart-polish passes per cell (resets the L-BFGS-B "
                         "Hessian approximation); 0 disables")
    ap.add_argument("--pin", nargs="+", default=None,
                    help="additional dials held at nominal. E_shift is ALWAYS "
                         "pinned (the response encodes ENERGY_SCALE=1). The "
                         "combined U1/3-exp fits additionally pin nunubar_ratio "
                         "(PINNED_UNION, ruling R3) — pass it here to match them.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default="orca_binned")
    ap.add_argument("--pynu-root", default=None,
                    help="prepend this path to sys.path before importing pynu")
    a = ap.parse_args()

    if a.pynu_root:
        root = os.path.abspath(a.pynu_root)
        if root not in sys.path:
            sys.path.insert(0, root)
    if a.phi_source == "tensors" and not a.phi_dir:
        ap.error("--phi-source tensors requires --phi-dir")

    dm_grid = np.linspace(a.dm_min, a.dm_max, a.ndm)
    s23_grid = np.linspace(a.s23_min, a.s23_max, a.ns23)
    if not (0 <= a.row < a.ndm):
        ap.error(f"--row {a.row} outside [0, {a.ndm})")
    dm = float(dm_grid[a.row])

    print(f"[row {a.row}] dm231={dm*1e3:.5f}e-3 over {a.ns23} s23 points "
          f"| PyNuFit({os.path.basename(a.config)})")
    ctx = build_ctx(a)
    eng = ctx["eng"]
    nominal, sigma, swept_idx = ctx["nominal"], ctx["sigma"], ctx["swept_idx"]
    bounds = make_bounds(nominal, sigma, swept_idx)

    seed = nominal[swept_idx].copy()
    pts = []
    for j, s23 in enumerate(s23_grid):
        s23 = float(s23)
        if a.phi_source == "tensors":
            path = os.path.join(a.phi_dir, a.phi_pattern.format(i=a.row, j=j))
            if not os.path.isfile(path):
                sys.exit(f"missing phi tensor {path} — build it with "
                         "build_orca_osc_tensors.py at this grid point")
            phi = load_phi_tensor(path, dm, s23, a.dcp_node)
        else:
            phi = live_phi(ctx, dm, s23, a.s13, a.dcp)

        f, th, c_stat, pen, ok, nev, tw = fit_point(
            eng, phi, nominal, sigma, swept_idx, bounds, seed)
        for _ in range(max(0, a.npolish)):
            pf, pth, pc, pp, pok, pnev, ptw = fit_point(
                eng, phi, nominal, sigma, swept_idx, bounds, th[swept_idx])
            nev += pnev
            tw += ptw
            if pf < f - 1e-3:
                f, th, c_stat, pen, ok = pf, pth, pc, pp, pok
            else:
                break
        seed = th[swept_idx]                       # warm-chain along s23

        # sanity floors: no invented reference value, just the physical ones.
        if not (np.isfinite(f) and c_stat >= 0.0 and pen >= 0.0):
            raise RuntimeError(f"sanity floor: chi2={f!r} stat={c_stat!r} "
                               f"prior={pen!r} at (row={a.row}, j={j})")
        if abs((c_stat + pen) - f) > 1e-6 * max(1.0, abs(f)):
            raise RuntimeError(
                f"term split does not reconstruct the objective at (row={a.row}, "
                f"j={j}): stat {c_stat} + prior {pen} != {f}")
        pts.append(dict(dm231=dm, sin2theta23=s23, chi2=float(f),
                        chi2_stat=float(c_stat), chi2_prior=float(pen),
                        converged=bool(ok), n_eval=int(nev), t_wall=float(tw),
                        nuisance=[float(v) for v in th]))
        print(f"  [{a.row},{j:2d}] s23={s23:.4f}: chi2={f:11.4f} "
              f"(stat {c_stat:10.4f} + prior {pen:7.4f}) "
              f"conv={ok} nev={nev} {tw:.1f}s")

    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, f"{a.tag}_row{a.row:03d}.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(experiment="ORCA", tag=a.tag, row=a.row, dm231=dm,
                       phi_source=a.phi_source, dcp_node=int(a.dcp_node),
                       response=os.path.abspath(a.response),
                       config=os.path.abspath(a.config),
                       n_dials=len(ctx["names"]), nuisance_names=ctx["names"],
                       pinned=ctx["pinned"],
                       swept=[ctx["names"][k] for k in swept_idx],
                       n_few=int(ctx["few"].sum()), points=pts), fh)
    os.replace(tmp, out)
    print(f"[row {a.row} dm={dm*1e3:.5f}e-3] min chi2="
          f"{min(p['chi2'] for p in pts):.4f} over {len(pts)} s23 -> {out}")


if __name__ == "__main__":
    main()
