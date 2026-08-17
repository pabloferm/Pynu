#!/usr/bin/env python3
"""Single-experiment IC DeepCore binned fit — SLURM row worker.

One array task = one Delta-m^2 row: a nuisance fit at every sin^2(theta23) point
along that row, driven by `ICBinnedEngine`'s analytic 39-long dial gradient, and
written as one row json.

============================== DIVISION OF LABOUR ==============================
★ THE ENGINE IS STAT-ONLY. `ICBinnedEngine.chi2_and_grad` returns a pure-Poisson
chi2 and its stat gradient and NOTHING ELSE — no Gaussian prior. THIS DRIVER owns
the prior, exactly as the combined workers do:

    prior       = sum_d ((theta_d - nominal_d) / sigma_d)^2
    d(prior)/dx = 2 (theta - nominal) / sigma^2

mirrored from `combined_ic_orca_fit_worker.py:242-246` (`prior_penalty`, the value)
and `combined_3exp_fit_worker.py:1331` (`g[:n_union] += 2.0*(theta-nominal)/sigma**2`,
the gradient). Bounds mirror `combined_ic_orca_fit_worker.py:231-239`
(`make_bounds`: nominal +- 5 sigma, positive-nominal dials floored at 0.01), and the
minimizer settings mirror `:149-153`. An arm-internal Gaussian would double-count
silently AND still pass an identity gate — the engine docstring's item (4).

(nominal, sigma) come from the LIVE PyNuFit analysis
(`Analysis.NuisNominalList` / `NuisSigmaList`), which is how the combined worker
builds its prior table (`combined_ic_orca_fit_worker.py:405-419`).

======================= THE TWO IC-SPECIFIC OBLIGATIONS ========================
Neither exists on the ORCA arm. Both are silent-wrong-answer failure modes, so
both are asserted rather than trusted.

(a) ★ HYPERSURFACE SLOPES ARE Delta-m^2 DEPENDENT AND MUST BE REFRESHED PER GRID
    CELL, from THAT cell's grid Dm2:

        eng.set_hs_slopes(exp.interpolate_hs(dm_cell), dm_cell)

    The live `ICDeepCore` experiment is instantiated ONCE (it is the only thing
    that can serve `interpolate_hs`) and re-queried per cell. Reusing one cell's
    slopes across a patch, or interpolating at a Dm2 that came from anywhere other
    than the grid, silently fits against the WRONG hypersurface while converging
    cleanly — that is the defect that invalidated the first G-IC-4 postfit run,
    with a measured per-bin |dC|/C up to 6.2e-3 for a ONE-GRID-STEP Dm2 error.
    `set_hs_slopes` (not the plain attribute) records the Dm2 the slopes came
    from, and this worker asserts that record against the cell before every fit —
    the engine never sees the grid and cannot detect staleness itself. Leaving the
    slopes unset entirely trips the engine's own guard: C_cat would fall back to 1
    and every HS dial would get a zero gradient.

(b) ★ THE PHI ROW AND dm31_cell DERIVE FROM THE SAME (i_dm, i_s23). The tensor npz
    is one file of many rows, read at `ipt = i_dm * ns23 + i_s23` — the identical
    arithmetic that produced the Dm2 handed to `interpolate_hs`. Independent
    centre+step arithmetic on the two sides is exactly how the one-grid-step error
    happened. Belt and braces: `build_ic_osc_tensors.py --grid` stores each row's
    `(dm231, s23)` and this worker hard-checks them against the cell.

================================ OTHER CONVENTIONS =============================
  * Response MUST be the mode-axis, UNSNAPPED build (`ic_response_modeaxis_L*`).
    The engine refuses a 12-class response; `phi_cells` hard-fails on a ladder
    shape mismatch.
  * NORM (`FitExposure * SECONDS_PER_YEAR`) is applied at scan time — the response
    stores RAW weight. It is read off the LIVE experiment so a config change
    cannot desync the two sides.
  * The muon background is a 200-bin constant added AFTER the HS correction, with
    zero gradient.
  * `few` mask = `obs > MIN_ENTRIES` (0.01), the production definition.
  * phi is oscillated FLUX (flux x P), not bare probability, and there is no SK NC
    override.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

# ---- pin the Pynu tree this worker ships inside (analysis/IC-binned-datafit
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

N_BINS = 200
DEFAULT_HS_DIR = os.path.join(PYNU_DIR, "data", "IceCube")


# ==============================================================================
# prior + bounds — OWNED BY THE DRIVER (see module docstring)
# ==============================================================================

def prior_penalty(theta, nominal, sigma):
    """Sum((theta-nominal)/sigma)^2 over ALL manifest dials.
    combined_ic_orca_fit_worker.py:242-246 verbatim. Pinned dials sit at nominal,
    so they contribute exactly 0 and need no special case."""
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
    combined_ic_orca_fit_worker.py:204-210 verbatim."""
    th = nominal.copy()
    th[swept_idx] = x_swept
    return th


# ==============================================================================
# per-cell state — obligation (a) and (b)
# ==============================================================================

def refresh_cell_state(ctx, i_dm, i_s23, dm, s23):
    """Set BOTH per-cell states from the SAME (i_dm, i_s23), then verify.

    Mirrors `refresh_binned_cell_state` (combined_3exp_fit_worker.py:1223-1261)
    and the symmetric stale-state guard at :1303-1309. Call once per grid cell,
    NEVER inside the objective.
    """
    ipt = i_dm * ctx["ns23"] + i_s23                    # ★ the one index (b)

    phi_dm = ctx["phi_dm231"]
    if phi_dm is not None:
        if not (0 <= ipt < phi_dm.size):
            sys.exit(f"phi row {ipt} outside the tensor's {phi_dm.size} rows — "
                     "the tensor was not built on this scan grid")
        got_dm, got_s23 = float(phi_dm[ipt]), float(ctx["phi_s23"][ipt])
        # rtol: loose enough for the tensor's stored-coordinate float noise
        # (job 39860001: rows matched to 8 decimals yet failed at 1e-9), tight
        # enough that a one-grid-step error (~1e-2 relative) can never pass.
        if not (np.isclose(got_dm, dm, rtol=1e-6, atol=0.0)
                and np.isclose(got_s23, s23, rtol=1e-6, atol=0.0)):
            sys.exit(
                f"phi row {ipt} was built at (dm231={got_dm:.8e}, s23={got_s23:.8f}) "
                f"but this cell is (dm231={dm:.8e}, s23={s23:.8f}) — the tensor row "
                "order is not the worker's row-major convention "
                "(ipt = i_dm*ns23 + i_s23)")
    ctx["phi_point"] = np.asarray(ctx["phi"][ipt], float)

    # ★ (a) hypersurface slopes at THIS cell's grid Dm2 — the same `dm` that
    # picked the phi row. set_hs_slopes records the Dm2 so staleness is
    # assertable instead of silent.
    ctx["eng"].set_hs_slopes(ctx["exp"].interpolate_hs(dm), dm)
    ctx["cell"] = (i_dm, i_s23, float(dm))

    if ctx["eng"].hs_dm31 != float(dm):
        raise AssertionError(
            f"IC hs_slopes were built for Dm2 {ctx['eng'].hs_dm31} but this cell "
            f"is {dm} — wrong hypersurface")


# ==============================================================================
# the fit
# ==============================================================================

def fit_point(ctx, nominal, sigma, swept_idx, bounds, x0):
    """One L-BFGS-B minimization of stat chi2 (engine, analytic grad) + prior
    (this driver) over the swept slots."""
    eng, phi = ctx["eng"], ctx["phi_point"]
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

def load_observation(exp, args):
    """The 200-bin observation, full length.

    `exp.ObservedBinned` is NOT directly usable: `Experiment.SetObservedBinned`
    masks it down to `FewEntries` in place. It is scattered back here, which is
    exact for every quantity the engine computes — chi2 and the residual both
    read `obs[few]` only, and bins outside the mask never contribute.

    A DataFit=False experiment would make that "observation" Asimov MC rather
    than data, so it is refused loudly; `--observed-npz` is the explicit override.
    """
    if args.observed_npz:
        obs = np.asarray(np.load(args.observed_npz)["obs"], float)
        if obs.shape != (N_BINS,):
            sys.exit(f"--observed-npz 'obs' shape {obs.shape} != ({N_BINS},)")
        print(f"[setup] observation from {args.observed_npz} "
              f"(total {obs.sum():.1f})")
        return obs
    if not getattr(exp, "DataFit", False):
        sys.exit(
            "exp.DataFit is False — this XML's <DataFiles> block is disabled, so "
            "exp.ObservedBinned is ASIMOV MC, not data, and fitting it would "
            "silently produce a self-fit. Enable the data files in the XML, or "
            "pass --observed-npz with the real 200-bin observation.")
    few = np.asarray(exp.FewEntries, bool)
    masked = np.asarray(exp.ObservedBinned, float)
    if few.shape != (N_BINS,):
        sys.exit(f"exp.FewEntries shape {few.shape} != ({N_BINS},)")
    if masked.shape != (int(few.sum()),):
        sys.exit(f"exp.ObservedBinned shape {masked.shape} is neither the masked "
                 f"({int(few.sum())},) nor anything this worker can interpret")
    obs = np.zeros(N_BINS)
    obs[few] = masked
    return obs


def build_ctx(args):
    """One PyNuFit init + one engine construction for the whole row."""
    from pynu import PyNuFit
    from pynu.Experiments.ICDeepCore import ICDeepCore_Atm
    from pynu.Experiments.ic_binned_engine import ICBinnedEngine, MIN_ENTRIES, PINNED

    pynufit = PyNuFit(args.config, verbosity=False)
    exp_name = list(pynufit.Experiments.keys())[0]
    exp = pynufit.Experiments[exp_name]
    osc = pynufit.physics_tunes[exp_name].OscillationTunes
    osc.osc_avg_scale = None            # combined_ic_orca_fit_worker.py:396

    # ★ the LIVE experiment, instantiated ONCE — the only source of interpolate_hs
    if not isinstance(exp, ICDeepCore_Atm):
        sys.exit(f"experiment {exp_name!r} is {type(exp).__name__}, not "
                 "ICDeepCore_Atm — it cannot serve interpolate_hs, and the HS "
                 "block is not optional on this arm")
    exp.load_hypersurfaces(args.hs_dir)

    names = list(pynufit.Analysis.NuisanceList)
    nominal = np.array(pynufit.Analysis.NuisNominalList, float)
    sigma = np.array(pynufit.Analysis.NuisSigmaList, float)
    if not (len(names) == nominal.size == sigma.size):
        sys.exit(f"manifest/nominal/sigma length mismatch: {len(names)} / "
                 f"{nominal.size} / {sigma.size}")
    if np.any(sigma <= 0):
        bad = [names[i] for i in np.flatnonzero(sigma <= 0)]
        sys.exit(f"non-positive prior sigma on {bad} — the Gaussian prior this "
                 "driver owns would divide by zero")

    obs = load_observation(exp, args)
    mu = exp.GetMuonBackground()[0]
    if mu is None:
        sys.exit("exp.GetMuonBackground() returned no muon histogram — the IC "
                 "expectation includes a 200-bin static muon background")
    mu = np.asarray(mu, float)
    if mu.shape != (N_BINS,):
        sys.exit(f"muon background shape {mu.shape} != ({N_BINS},)")

    # production mask (combined_ic_orca_fit_worker.py:394-395): obs > MIN_ENTRIES.
    min_entries = float(getattr(exp, "MIN_ENTRIES", MIN_ENTRIES))
    few = obs > min_entries
    print(f"[setup] fit mask: {int(few.sum())} of {N_BINS} bins "
          f"(obs > MIN_ENTRIES = {min_entries}); obs total {obs.sum():.1f}, "
          f"muon total {mu.sum():.2f}")

    pinned = tuple(args.pin) if args.pin is not None else tuple(PINNED)
    unknown = [p for p in pinned if p not in names]
    if unknown:
        sys.exit(f"--pin names not in the IC manifest: {unknown}")

    eng = ICBinnedEngine(args.response, obs, mu, names,
                         norm=float(exp.NORM),      # response stores RAW weight
                         hs_slopes=None,            # ★ set PER CELL
                         few=few, pinned=pinned)

    # ONE bind-time ladder check. The engine's shape check catches a
    # snapped-vs-unsnapped mismatch; this catches a same-shape-different-ladder
    # mix, which shape alone cannot see. Cheap once, pointless per cell.
    eng.cells.assert_phi_grid(args.phi)
    print(f"[setup] phi-grid check PASS vs {os.path.basename(args.phi)}")

    # ---- phi tensor: rows are the scan grid in row-major order ----
    r = np.load(args.phi, allow_pickle=True)
    phi = np.asarray(r["phi"], float)
    phi_dm = np.asarray(r["dm231"], float).reshape(-1) if "dm231" in r else None
    phi_s23 = np.asarray(r["s23"], float).reshape(-1) if "s23" in r else None
    if phi_dm is not None and phi_dm.size != phi.shape[0]:
        phi_dm = phi_s23 = None            # not a per-row record; skip the check
    if phi_dm is None:
        print("[setup] WARNING: the phi npz carries no per-row (dm231, s23); the "
              "row-order check is DISABLED. Rebuild with build_ic_osc_tensors.py "
              "--grid to enable it.")

    # engine's pinned slots must be exactly the ones held out of the fit, or the
    # minimizer would move a dial whose gradient the engine writes as 0.0
    swept = [n for n in names if n not in eng.pinned]
    swept_idx = np.array([names.index(n) for n in swept], dtype=np.int64)

    print(f"[setup] {eng.summary()}")
    print(f"[setup] dials {len(names)} | swept {len(swept)} | "
          f"pinned {list(eng.pinned)} | HS {eng.hs_names}")
    return dict(pynufit=pynufit, exp=exp, osc=osc, eng=eng, names=names,
                nominal=nominal, sigma=sigma, swept_idx=swept_idx,
                obs=obs, mu=mu, few=few, phi=phi, phi_dm231=phi_dm,
                phi_s23=phi_s23, ns23=args.ns23, phi_point=None, cell=None)


def main():
    ap = argparse.ArgumentParser(
        description="Single-experiment IC DeepCore binned fit: one Delta-m^2 row "
                    "of sin^2(theta23) points. array index = the row.")
    ap.add_argument("--row", type=int, required=True,
                    help="Delta-m^2 row index into the scan grid (= SLURM array index)")
    ap.add_argument("--config", required=True,
                    help="IC DeepCore analysis XML carrying the 39-dial manifest "
                         "(the campaign's IC_DeepCore_r2_fude_ccqe.xml)")
    ap.add_argument("--response", required=True,
                    help="ic_response_modeaxis_L*.npz — MUST carry the |Mode| "
                         "class axis and stay unsnapped")
    ap.add_argument("--phi", required=True,
                    help="osc tensor npz from build_ic_osc_tensors.py --grid "
                         "(rows in row-major order: ipt = i_dm*ns23 + i_s23)")
    ap.add_argument("--hs-dir", default=DEFAULT_HS_DIR,
                    help="dir with the hs_*.csv hypersurfaces "
                         "(default: data/IceCube)")
    ap.add_argument("--observed-npz", default=None,
                    help="explicit 200-bin observation (key 'obs'). Only needed "
                         "when the XML's data files are disabled — see "
                         "load_observation.")
    # ---- grid (MUST match the tensor build) ----
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
                    help="dials held at nominal, overriding the engine default "
                         "(nunubar_ratio, PINNED_UNION ruling R3). The engine "
                         "writes each pinned gradient slot 0.0, so the swept set "
                         "is derived from this — they cannot disagree.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default="ic_binned")
    ap.add_argument("--pynu-root", default=None,
                    help="prepend this path to sys.path before importing pynu")
    a = ap.parse_args()

    if a.pynu_root:
        root = os.path.abspath(a.pynu_root)
        if root not in sys.path:
            sys.path.insert(0, root)

    dm_grid = np.linspace(a.dm_min, a.dm_max, a.ndm)
    s23_grid = np.linspace(a.s23_min, a.s23_max, a.ns23)
    if not (0 <= a.row < a.ndm):
        ap.error(f"--row {a.row} outside [0, {a.ndm})")
    dm = float(dm_grid[a.row])

    print(f"[row {a.row}] dm231={dm*1e3:.5f}e-3 over {a.ns23} s23 points "
          f"| PyNuFit({os.path.basename(a.config)})")
    ctx = build_ctx(a)
    nominal, sigma, swept_idx = ctx["nominal"], ctx["sigma"], ctx["swept_idx"]
    bounds = make_bounds(nominal, sigma, swept_idx)

    seed = nominal[swept_idx].copy()
    pts = []
    for j, s23 in enumerate(s23_grid):
        s23 = float(s23)
        # ★ BOTH per-cell states, from the SAME (i_dm, i_s23), BEFORE the fit
        refresh_cell_state(ctx, a.row, j, dm, s23)

        f, th, c_stat, pen, ok, nev, tw = fit_point(
            ctx, nominal, sigma, swept_idx, bounds, seed)
        for _ in range(max(0, a.npolish)):
            pf, pth, pc, pp, pok, pnev, ptw = fit_point(
                ctx, nominal, sigma, swept_idx, bounds, th[swept_idx])
            nev += pnev
            tw += ptw
            if pf < f - 1e-3:
                f, th, c_stat, pen, ok = pf, pth, pc, pp, pok
            else:
                break
        seed = th[swept_idx]                       # warm-chain along s23

        # the stale-state guard, re-checked AFTER the fit: nothing in the loop may
        # have moved the engine off this cell.
        if ctx["eng"].hs_dm31 != float(dm) or ctx["cell"][:2] != (a.row, j):
            raise AssertionError(
                f"engine drifted off cell ({a.row},{j}): hs_dm31="
                f"{ctx['eng'].hs_dm31}, cell={ctx['cell']}")
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
                        hs_dm31=float(ctx["eng"].hs_dm31),
                        converged=bool(ok), n_eval=int(nev), t_wall=float(tw),
                        nuisance=[float(v) for v in th]))
        print(f"  [{a.row},{j:2d}] s23={s23:.4f}: chi2={f:11.4f} "
              f"(stat {c_stat:10.4f} + prior {pen:7.4f}) "
              f"conv={ok} nev={nev} {tw:.1f}s")

    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, f"{a.tag}_row{a.row:03d}.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(experiment="IC-DeepCore", tag=a.tag, row=a.row, dm231=dm,
                       response=os.path.abspath(a.response),
                       phi=os.path.abspath(a.phi),
                       config=os.path.abspath(a.config),
                       n_dials=len(ctx["names"]), nuisance_names=ctx["names"],
                       pinned=list(ctx["eng"].pinned),
                       swept=[ctx["names"][k] for k in swept_idx],
                       hs_dials=list(ctx["eng"].hs_names),
                       n_few=int(ctx["few"].sum()), points=pts), fh)
    os.replace(tmp, out)
    print(f"[row {a.row} dm={dm*1e3:.5f}e-3] min chi2="
          f"{min(p['chi2'] for p in pts):.4f} over {len(pts)} s23 -> {out}")


if __name__ == "__main__":
    main()
