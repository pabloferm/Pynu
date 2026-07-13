#!/usr/bin/env python3
"""SK binned 20x20 (Dm231 x Sin2Theta23) scan — row worker through the
integrated (PyNuFit-modular) engine.

One SLURM array task = one (arm, Dm231-row): 20 dCP-profiled nuisance fits along
the Sin2Theta23 axis, written as one row json.

The worker reproduces, fit-for-fit, the standalone SKBinnedEngine.fit_point scan
protocol (warm-chain along s23, --npolish restart-polish, dCP profiled per cell,
atomic .tmp json writes) while driving every fit through ONE PyNuFit object and
the modular method vocabulary:

  PyNuFit(analysis_xml)                          # single object, from XML
  exp = pf.set_binned_engine(exp, BinnedConfig(...))  # programmatic opt-in
  pf.BuildBinnedResponse()/pf.BuildOscTensors()  # native builders (--build-missing;
                                                 #   prebuilt artifacts = fast path)
  exp.override_prior_sigma(names, sigma)         # seed flux-ratio σ override
  exp.nuisance_bounds()                          # L-BFGS-B production box
  pf.set_likelihood("PoissonLikelihood", binned_priors=True)  # LLH from priors
  pf.StagePhysics(dm, s23, dcp)                  # grid staging + fallback
  pf.StartNuisance / pf.ApplyNuisanceWeights /
  pf.SetExpectedWeights / pf.SetBinnedExpectedEvents
  pf.LLH.stats_and_systematics                   # objective certification per cell

The per-evaluation (f, g) inside L-BFGS-B is `pf._binned_chi2_and_grad()` —
the engine's own analytic kernel (bit-identical to a direct
`engine.chi2_and_grad` call; the staged (phi, theta) live on the PyNuFit object,
Track S / E6). Every converged cell is additionally re-evaluated through the full
method vocabulary + PoissonLikelihood and must agree EXACTLY (diff == 0.0) or the
task aborts — each row json is therefore self-certifying.

Bit-parity contract vs a standalone engine scan: identical seed (--gbest-list),
identical bounds (BinnedExperiment.nuisance_bounds — the single-source production
box shared with SKBinnedEngine.fit_point), identical ftol scaling, identical dCP
warm-chain + s23 warm-chain + restart-polish ordering => identical L-BFGS-B
trajectory => identical chi2/nuisance to the bit.

CLI arm selection is EXPLICIT about named-spec vs file-path: --arm-specs takes
engine NAMED specs (e.g. r2_fude_ccqe, r2_fude_ccqe_nmig —
resolve_nuisance_spec names); --arm-xmls takes explicit manifest .xml FILE
PATHS. Mutually exclusive; a named spec that looks like a path (contains '/' or
ends '.xml') is rejected.

Every step of the row scan goes through first-class PyNuFit / BinnedExperiment
methods — the four former worker deviations (seed prior override, L-BFGS-B
bounds transcription, direct PoissonLikelihood construction, grid-fallback
staging) were dissolved into the method vocabulary in Track S·F / F5:
``BinnedExperiment.override_prior_sigma`` (seed σ override),
``pf.set_binned_engine(...).nuisance_bounds()`` (production bounds box),
``pf.set_likelihood('PoissonLikelihood', binned_priors=True)`` (Poisson LLH from
the binding's post-override priors), and ``pf.StagePhysics(dm, s23, dcp)`` (grid
staging with automatic StageBinnedPhysics fallback). No worker-side deviations
remain.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

# ---- pin the Pynu tree this worker ships inside (analysis/SK-binned-datafit
# -> ../../ = the Pynu repo root), so the worker is runnable standalone; the
# submission script may additionally export PYTHONPATH explicitly. ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYNU_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PYNU_DIR not in sys.path:
    sys.path.insert(0, PYNU_DIR)

from pynu import PyNuFit                                    # noqa: E402
from pynu.binned import BinnedConfig                        # noqa: E402  (S.F1: re-export from analysis_reader.binned_config)
# The flux-ratio dial set the seed's flux_ratio_sigma applies to (a data list,
# not a bounds/box transcription — the bounds box now lives behind
# BinnedExperiment.nuisance_bounds, S.F5 dissolving deviation D2).
from pynu.binned.sk_binned_engine import ALL_FLUX_RATIO_NAMES  # noqa: E402


# --------------------------------------------------------------------------
# the modular fit — SKBinnedEngine.fit_point protocol (:1806-1886, dcp_warmchain
# default) driven through the method vocabulary. (f, g) per evaluation =
# pf._binned_chi2_and_grad() (the engine's analytic kernel).
# --------------------------------------------------------------------------
def modular_fit_point(pf, llh, dm, s23, x0, bounds, n_dcp):
    best = (np.inf, 0, np.asarray(x0, float), 0, False)
    x_seed = np.asarray(x0, float)
    for di in range(n_dcp):
        pf.StagePhysics(dm, s23, di)    # grid staging + StageBinnedPhysics fallback
        # ftol scaling from the stats-only chi2 at the current seed — through the
        # full vocabulary + PoissonLikelihood (engine fit_point:1866-1873 parity;
        # llh.stats_only delegates to the same engine poisson_chi2 kernel).
        pf.StartNuisance()
        pf.ApplyNuisanceWeights(x_seed)
        pf.SetExpectedWeights()
        pf.SetBinnedExpectedEvents()
        chi2_stat = llh.stats_only(pf.Expectation)
        tol = max(1e-5, np.sqrt(max(min(chi2_stat, 1e7), 0)) * 1e-5)

        def fg(theta):
            pf.StartNuisance()                     # per-evaluation reset
            pf.ApplyNuisanceWeights(theta)         # stage theta on the engine
            return pf._binned_chi2_and_grad()      # (f, g), engine analytic kernel

        res = minimize(fg, x_seed, method="L-BFGS-B", jac=True, bounds=bounds,
                       options={"ftol": tol, "gtol": 1e-5, "maxiter": 200})
        if res.fun < best[0]:
            best = (res.fun, di, res.x.copy(), res.nit, res.success)
        x_seed = best[2]              # dCP warm-chain (engine fit_point:1884-1885)
    return best


def modular_objective(pf, llh, dm, s23, dcp_index, theta):
    """The full modular objective (complete method-vocabulary sequence) — used
    to certify every converged cell."""
    pf.StagePhysics(dm, s23, dcp_index)    # grid staging + StageBinnedPhysics fallback
    pf.StartNuisance()
    pf.ApplyNuisanceWeights(theta)
    pf.SetExpectedWeights()
    pf.SetBinnedExpectedEvents()
    return llh.stats_and_systematics(pf.Expectation, theta)


# --------------------------------------------------------------------------
def ensure_artifacts(pf, exp_name, args, dm, row, s23_grid):
    """Load-from-path fast path for the prebuilt response/tensors; with
    --build-missing, absent artifacts are built through the native methods
    (pf.BuildBinnedResponse / pf.BuildOscTensors — byte-compatible with the
    standalone build scripts)."""
    if not os.path.isfile(args.response):
        if not args.build_missing:
            sys.exit(f"response not found: {args.response} (production fast path;"
                     " pass --build-missing to build it natively on this node)")
        print(f"[build] response absent -> pf.BuildBinnedResponse"
              f"(n_etrue={args.build_n_etrue}, n_cztrue={args.build_n_cztrue})")
        pf.BuildBinnedResponse(exp_name=exp_name, out_path=args.response,
                               n_etrue=args.build_n_etrue,
                               n_cztrue=args.build_n_cztrue)
    # dCP node convention: build_osc_tensors.py:44 DCP_GRID =
    # linspace(0, 2pi, N, endpoint=False); production 20-node axis = N=20.
    dcp_nodes = np.linspace(0.0, 2.0 * np.pi, args.build_n_dcp, endpoint=False)
    avg = None if str(args.osc_averaging).lower() in ("off", "", "none") \
        else args.osc_averaging
    for j, s23 in enumerate(s23_grid):
        p = os.path.join(args.tensors, f"osc_tensor_{row:03d}_{j:03d}.npz")
        if os.path.isfile(p):
            continue
        if not args.build_missing:
            sys.exit(f"osc tensor not found: {p} (production fast path; pass "
                     "--build-missing to build this row's nodes natively)")
        print(f"[build] tensor node ({row},{j}) -> pf.BuildOscTensors"
              f"(dm={dm:.6e}, s23={s23:.6f}, {args.build_n_dcp} dCP)")
        os.makedirs(args.tensors, exist_ok=True)
        pf.BuildOscTensors(dm, s23, exp_name=exp_name, dcp_nodes=dcp_nodes,
                           s13=args.build_s13, n_etrue=args.build_n_etrue,
                           n_cztrue=args.build_n_cztrue, avg_scale=avg,
                           out_path=p)


def main():
    ap = argparse.ArgumentParser(
        description="SK binned 20x20 scan row worker (INTEGRATED pynufit-modular "
                    "engine). task = arm_idx * ndm + dm_row.")
    ap.add_argument("--task", type=int, required=True,
                    help="combined array index: arm_idx * --ndm + dm_row")
    ap.add_argument("--analysis-xml", required=True,
                    help="FULL PyNuFit analysis config (event MC + experiment "
                         "blocks). On SK this MUST be the *_gateb.xml-style "
                         "config — the plain phased.xml carries a bare "
                         "decay_e_tagging (enabled, no method) and crashes "
                         "PyNuFit at init.")
    # ---- arm selection: EXPLICIT named-spec vs file-path (never guess which
    # one a string is) ----
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--arm-specs", nargs="+",
                     help="engine NAMED specs, one per arm in arm_idx order "
                          "(e.g. r2_fude_ccqe r2_fude_ccqe_nmig). NOT file paths.")
    grp.add_argument("--arm-xmls", nargs="+",
                     help="nuisance-manifest .xml FILE PATHS, one per arm in "
                          "arm_idx order (resolve_nuisance_spec(<path>) route).")
    ap.add_argument("--gbest-list", required=True, nargs="+",
                    help="per-arm converged gbest jsons (warm seed + output "
                         "naming), arm_idx order — use the SAME seed files as "
                         "the reference scan when replaying one")
    ap.add_argument("--response", required=True, help="sk_response.npz path")
    ap.add_argument("--tensors", required=True, help="osc tensor dir "
                    "(osc_tensor_<i>_<j>.npz on the scan grid nodes)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--npolish", type=int, default=3)
    ap.add_argument("--ndm", type=int, default=20)
    ap.add_argument("--dm-min", type=float, default=2.0e-3)
    ap.add_argument("--dm-max", type=float, default=3.2e-3)
    ap.add_argument("--ns23", type=int, default=20)
    ap.add_argument("--s23-min", type=float, default=0.30)
    ap.add_argument("--s23-max", type=float, default=0.70)
    ap.add_argument("--osc-averaging", default="4pi",
                    help="<BinnedEngine> osc_averaging provenance declaration "
                         "(production tensors: '4pi')")
    ap.add_argument("--build-missing", action="store_true",
                    help="build absent response/tensor nodes via the NATIVE "
                         "pf.BuildBinnedResponse/BuildOscTensors (default: hard "
                         "error — production uses prebuilt artifacts)")
    ap.add_argument("--build-n-etrue", type=int, default=400,
                    help="true-E grid density for native builds (production "
                         "response = 400; MUST match the response build)")
    ap.add_argument("--build-n-cztrue", type=int, default=40)
    ap.add_argument("--build-n-dcp", type=int, default=20,
                    help="dCP nodes for native tensor builds (production = 20)")
    ap.add_argument("--build-s13", type=float, default=0.022,
                    help="Sin2Theta13 for native tensor builds (production 0.0220)")
    a = ap.parse_args()

    # ---- grids (the production scan axes are the defaults) ----
    dm_fine = np.linspace(a.dm_min, a.dm_max, a.ndm)
    s23_fine = np.linspace(a.s23_min, a.s23_max, a.ns23)
    arm_idx, row = a.task // a.ndm, a.task % a.ndm
    dm = dm_fine[row]

    # ---- arm resolution (explicit named-spec / file-path split) ----
    if a.arm_specs is not None:
        specs = a.arm_specs
        for s in specs:
            if "/" in s or s.endswith(".xml"):
                ap.error(f"--arm-specs takes NAMED specs, got path-like {s!r}; "
                         "use --arm-xmls for manifest files")
        nuis_spec = specs[arm_idx]
    else:
        for s in a.arm_xmls:
            if not os.path.isfile(s):
                ap.error(f"--arm-xmls entry is not a file: {s!r}")
        nuis_spec = a.arm_xmls[arm_idx]
    if len(a.gbest_list) != len(a.arm_specs or a.arm_xmls):
        ap.error("--gbest-list length must equal the number of arms")
    if not (0 <= arm_idx < len(a.gbest_list)):
        ap.error(f"--task {a.task} -> arm_idx {arm_idx} outside the "
                 f"{len(a.gbest_list)}-arm list (ndm={a.ndm})")

    # ---- seed json (warm start + output naming) ----
    gbpath = a.gbest_list[arm_idx]
    gb = json.load(open(gbpath))
    spec_names = list(gb["nuisance_names"])
    theta0 = np.array(gb["nuisance"], float)
    arm = os.path.splitext(os.path.basename(gbpath))[0]   # unique per variant
    arm_spec = gb.get("arm", f"arm{arm_idx}")
    # arm prior-knobs: only flux_ratio_sigma is in the r2_fude_ccqe protocol;
    # other legacy seed knobs are out of scope here — hard error rather than
    # silently diverge from the seed's intended fit.
    if gb.get("lump"):
        sys.exit("seed requests lump free_mask — unsupported by the modular "
                 "worker (legacy knob of the standalone engine path)")
    if gb.get("xsec_tight_sigma") is not None:
        sys.exit("seed requests xsec_tight_sigma — unsupported by the modular "
                 "worker (legacy knob of the standalone engine path)")
    if gb.get("dirsmear_matrix"):
        sys.exit("seed requests dirsmear_matrix — the BinnedConfig binding route "
                 "has no dir-smear passthrough. Production r2_fude_ccqe* seeds "
                 "carry no such key.")

    # ---- ONE PyNuFit object from the XML (heavy: event MC + nuSQuIDS init) ----
    print(f"[task {a.task}] arm={arm} (spec={nuis_spec!r}) row={row} "
          f"dm={dm*1e3:.4f}e-3 | PyNuFit({os.path.basename(a.analysis_xml)})")
    pf = PyNuFit(a.analysis_xml, verbosity=False)
    exp_name = next(iter(pf.Experiments))     # single-experiment convention

    # ---- native builders / load-from-path fast paths ----
    ensure_artifacts(pf, exp_name, a, dm, row, s23_fine)

    # ---- programmatic <BinnedEngine> opt-in (PyNuFit.set_binned_engine) ----
    # Track S·F / F2: set_binned_engine now returns a BinnedExperiment (an
    # Experiment subclass wrapping the loaded binding). Its read-only surface
    # (nuisance_names / DM / S23 / nominal / sigma / observed_binned / engine /
    # n_dcp) is byte-identical to the former BinnedBinding — including the
    # in-place binding.sigma prior override below (D1), since .sigma is the same
    # engine.sigma array — so no call site changed.
    cfg = BinnedConfig(response=a.response, tensors=a.tensors,
                       likelihood="poisson", migration="weighted",
                       nuisance_spec=nuis_spec, interp="nodes",
                       osc_averaging=a.osc_averaging)
    binding = pf.set_binned_engine(exp_name, cfg, analysis_xml=a.analysis_xml)

    # ---- validations (sanity floor) ----
    if list(binding.nuisance_names) != spec_names:
        sys.exit(f"arm spec {nuis_spec!r} resolves to {len(binding.nuisance_names)} "
                 f"dials != seed {gbpath} ({len(spec_names)}) or order differs — "
                 "seed and spec must be the SAME production arm")
    if not (np.allclose(binding.DM, dm_fine, rtol=1e-12, atol=0.0)
            and np.allclose(binding.S23, s23_fine, rtol=1e-12, atol=0.0)):
        sys.exit("tensor grid axes != requested scan grid: the modular scan is "
                 "raw-nodes (interp='nodes'); build/point the tensors at the "
                 "exact production axes (dm 2.0e-3..3.2e-3 x20, s23 0.30..0.70 x20)")

    # ---- seed flux-ratio prior override — before bounds/LLH. The seed's
    # flux_ratio_sigma (0.03 in the production seeds) replaces the engine sigma
    # on all flux-ratio dials, via the first-class
    # BinnedExperiment.override_prior_sigma (S.F5; binding.sigma IS engine.sigma).
    if gb.get("flux_ratio_sigma") is not None:
        binding.override_prior_sigma(ALL_FLUX_RATIO_NAMES, gb["flux_ratio_sigma"])
        print(f"[task {a.task}] flux_ratio_sigma := {gb['flux_ratio_sigma']}")

    # ---- PoissonLikelihood from the binding's post-override priors, via the
    # first-class pf.set_likelihood(..., binned_priors=True) (S.F5). ----
    n_dials = len(spec_names)
    pf.set_likelihood("PoissonLikelihood", binned_priors=True)
    llh = pf.LLH

    bounds = binding.nuisance_bounds()        # production box (S.F5, first-class)
    n_dcp = int(binding.n_dcp)
    grid_covers = pf.resolve_binned_grid_index(dm, s23_fine[0]) is not None
    print(f"[task {a.task}] n_dials={n_dials} n_dcp={n_dcp} physics staging: "
          + ("PyNuFit.ApplyPhysicsWeights (XML grid)" if grid_covers
             else "pf.StageBinnedPhysics (XML grid does not cover the scan; "
                  "explicit-(dm,s23) staging accessor — pf.StagePhysics fallback)"))

    # ---- row scan: fit_point protocol per cell, modular drive ----
    seed = theta0.copy()
    pts = []
    for j, s23 in enumerate(s23_fine):
        c, dcp, nu, _, _ = modular_fit_point(
            pf, llh, dm, s23, seed, bounds, n_dcp)
        for _ in range(a.npolish):            # restart-polish resets the Hessian
            pc, pd, pn, _, _ = modular_fit_point(
                pf, llh, dm, s23, nu, bounds, n_dcp)
            if pc < c - 1e-3:
                c, dcp, nu = pc, pd, pn
            else:
                break
        seed = nu                             # warm-chain along s23
        # per-cell certification: full vocabulary + PoissonLikelihood must equal
        # the kernel objective EXACTLY (self-certifying row).
        f_mod = modular_objective(pf, llh, dm, s23, int(dcp), nu)
        if abs(float(f_mod) - float(c)) != 0.0:
            raise RuntimeError(
                f"modular-vocabulary certification FAILED at (row={row}, j={j}): "
                f"llh={f_mod!r} vs kernel={c!r} — bit-parity broken, aborting")
        if not (np.isfinite(c) and c >= 0.0):
            raise RuntimeError(f"sanity floor: chi2={c!r} at (row={row}, j={j})")
        pts.append(dict(dm231=float(dm), sin2theta23=float(s23), chi2=float(c),
                        best_dcp_idx=int(dcp), nuisance=[float(v) for v in nu]))
        print(f"  [{row},{j:2d}] s23={s23:.4f}: chi2={c:10.4f} dcp={int(dcp):2d} "
              f"cert=0.0")

    # ---- row-json output schema + atomic write ----
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, f"{arm}_row{row:03d}.json")
    tmp = out + ".tmp"
    json.dump(dict(arm=arm, arm_spec=arm_spec, row=row, dm231=float(dm),
                   n_dials=len(spec_names), nuisance_names=spec_names, points=pts),
              open(tmp, "w"))
    os.replace(tmp, out)
    print(f"[{arm} row{row} dm={dm*1e3:.4f}e-3] "
          f"min chi2={min(p['chi2'] for p in pts):.3f} over {len(pts)} s23")


if __name__ == "__main__":
    main()
