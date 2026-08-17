#!/usr/bin/env python3
"""IC DeepCore osc-tensor builder. FASRC/HPC-only (needs nuSQuIDS).

Builds the oscillated-flux tensor Phi[n_pt, 2, 3, nE, nZ] for IC DeepCore on a
ladder grid's CELL CENTRES (the response npz's `e_true_centers`/`cz_true_centers`),
one slice per oscillation test point.

Reuses the CERTIFIED SK binned-builder osc helpers VERBATIM (`_snapshot_osc_state`
/ `_set_grid_coords` / `_eval_point` / `_restore_osc_state`) so propagation, units,
the Dm231_bar->Dm231 convention, the averaging knob, and the byte-exact try/finally
state-restore are all inherited from the certified SK/ORCA machinery. The ONLY IC
difference vs the ORCA tensor builder:
  * the true grid = the LADDER cell centres. IC's true side is event-level (100%
    unique), so these are genuine bin centres and the tensor at them is
    APPROXIMATE per event — the cell-centering (Jensen) residue.
  * osc points are parameterized via CLI, MANY POINTS PER FILE.

Phi convention (mirrors ORCA/SK): Phi[i_pt, type, flavor, iE, iZ] — type 0=nu/
1=nubar, flavor 0=e/1=mu/2=tau, iE/iZ ladder cells. float32.

★ ROW ORDER IS THE FIT WORKER'S CONTRACT. `run_ic_binned_fit_worker.py` reads row
`ipt = i_dm * n_s23 + i_s23`, so build the points in that ROW-MAJOR order (Delta-m^2
outer, sin^2(theta23) inner) — `--grid` does exactly that. The npz also stores the
per-row `dm231`/`s23`, and the worker hard-checks them against the cell it is
fitting, so a mis-ordered point list is an error rather than a one-grid-step
silent offset.

Usage on a cluster (from the repo root, env sourced):
  python3 analysis/IC-binned-datafit/build_ic_osc_tensors.py \
      --config analysis/AnalysisFiles/IC_DeepCore_r2_fude_ccqe.xml \
      --response ic_response_modeaxis_L3.npz \
      --grid 20 2.3e-3 2.7e-3 20 0.40 0.65 \
      --out ic_phi_L3_s13_0p0220.npz
"""
import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Default osc test points: truth + the 4 corners of the IC-datafit box.
DEFAULT_POINTS = [
    (2.511e-3, 0.572),   # truth
    (2.3e-3, 0.45),
    (2.7e-3, 0.45),
    (2.3e-3, 0.65),
    (2.7e-3, 0.65),
]


def add_pynu_root(root):
    root = os.path.abspath(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _parse_points(s):
    """'dm31,s23 dm31,s23 ...' -> [(dm31,s23), ...]."""
    pts = []
    for tok in s.split():
        a, b = tok.split(",")
        pts.append((float(a), float(b)))
    return pts


def _grid_points(ndm, dm_min, dm_max, ns23, s23_min, s23_max):
    """The scan grid in ROW-MAJOR order: ipt = i_dm * ns23 + i_s23. This is the
    order run_ic_binned_fit_worker.py indexes, and the ONLY order that keeps the
    hypersurface (interpolated at the grid Dm2) on the same cell as the flux."""
    dm = np.linspace(dm_min, dm_max, ndm)
    s23 = np.linspace(s23_min, s23_max, ns23)
    return [(float(d), float(s)) for d in dm for s in s23]


def main():
    ap = argparse.ArgumentParser(
        description="Build the IC oscillated-flux tensor npz on a response's "
                    "ladder cell centres, one slice per osc point.")
    ap.add_argument("--config", required=True, help="IC manifest XML")
    ap.add_argument("--response", required=True,
                    help="ic_response_*_L{k}.npz (source of the ladder centres)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--points", default=None,
                     help="'dm31,s23 dm31,s23 ...' (default = truth + 4 box corners)")
    grp.add_argument("--grid", nargs=6, type=float, default=None,
                     metavar=("NDM", "DM_MIN", "DM_MAX", "NS23", "S23_MIN", "S23_MAX"),
                     help="build the full scan grid in the worker's row-major "
                          "order (ipt = i_dm*NS23 + i_s23)")
    ap.add_argument("--dcp", type=float, default=1.36,
                    help="fixed dCP (rad); IC osc block default 1.36 (XML)")
    ap.add_argument("--s13", type=float, default=None)
    ap.add_argument("--dm221", type=float, default=None,
                    help="override Dm221 (set 0 together with a 0,s23 point for "
                         "the no-oscillation identity tensor, mirroring SK/ORCA)")
    ap.add_argument("--avg-scale", default=None,
                    help="fast-osc averaging selector ('4pi'/'2pi'/'off'/float); "
                         "None -> honour PYNU_OSC_AVG_SCALE env / object default")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pynu-root", default=REPO_ROOT,
                    help="Pynu repo root to prepend to sys.path (default: the "
                         "repo this script ships in)")
    args = ap.parse_args()

    if args.grid is not None:
        ndm, dm_min, dm_max, ns23, s23_min, s23_max = args.grid
        points = _grid_points(int(ndm), dm_min, dm_max, int(ns23), s23_min, s23_max)
        grid_meta = np.array([int(ndm), int(ns23)])
    else:
        points = _parse_points(args.points) if args.points else DEFAULT_POINTS
        grid_meta = np.array([-1, -1])       # not a scan grid

    add_pynu_root(args.pynu_root)
    from pynu import PyNuFit
    from pynu.Experiments import sk_binned_builder as B   # certified osc helpers

    # ---- ladder cell centres from the response ----
    resp = np.load(args.response, allow_pickle=True)
    e_c = np.asarray(resp["e_true_centers"], float)
    z_c = np.asarray(resp["cz_true_centers"], float)
    grid_label = str(resp["grid_label"]) if "grid_label" in resp else "?"
    if not (np.all(np.isfinite(e_c)) and np.all(np.isfinite(z_c))):
        raise RuntimeError("response has NaN ladder centres — a true bin was "
                           "unpopulated in the edge build (should not happen).")

    # ---- live osc object from PyNuFit(config) ----
    pynufit = PyNuFit(args.config, verbosity=False)
    exp_name = list(pynufit.Experiments.keys())[0]
    osc = pynufit.physics_tunes[exp_name].OscillationTunes

    dm_arr = np.array([p[0] for p in points], float)
    s23_arr = np.array([p[1] for p in points], float)

    snap = B._snapshot_osc_state(osc)
    try:
        if args.avg_scale is not None:
            osc.osc_avg_scale = B._resolve_avg_scale(args.avg_scale)
            applied = args.avg_scale
        else:
            env = os.environ.get("PYNU_OSC_AVG_SCALE")
            applied = env if env is not None else (
                "off" if getattr(osc, "osc_avg_scale", None) is None else "on")
            if env is not None:
                osc.osc_avg_scale = B._resolve_avg_scale(env)
        nE, nZ = B._set_grid_coords(osc, e_c, z_c)
        phi = np.stack([
            B._eval_point(osc, dm, s23, args.dcp, nE, nZ, s13=args.s13,
                          dm221=args.dm221)
            for dm, s23 in zip(dm_arr, s23_arr)])
    finally:
        B._restore_osc_state(osc, snap)

    np.savez_compressed(
        args.out, phi=phi.astype(np.float32),
        dm231=dm_arr, s23=s23_arr, dcp=float(args.dcp),
        grid_label=np.array(grid_label),
        scan_grid=grid_meta,
        e_true_centers=e_c, cz_true_centers=z_c,
        osc_averaging=np.array(str(applied)),
        grid=np.array([nE, nZ]))

    # ---- sanity floors ----
    print("=== IC osc tensor built ===")
    print(f"  out: {args.out}  (ladder {grid_label})")
    print(f"  n_points: {len(points)}  row order: "
          + ("row-major scan grid (i_dm*ns23 + i_s23)" if args.grid is not None
             else "as given"))
    print(f"  phi shape: {phi.shape}  (expect ({len(points)}, 2, 3, {nE}, {nZ}))")
    shape_ok = phi.shape == (len(points), 2, 3, nE, nZ)
    print(f"  shape gate: {'PASS' if shape_ok else 'FAIL'}")
    print(f"  grid nE,nZ = {nE},{nZ}")
    finite = bool(np.all(np.isfinite(phi)))
    nonneg = bool((phi >= -1e-9).all())
    print(f"  phi finite: {finite}  nonneg: {nonneg}  "
          f"min={phi.min():.4e} max={phi.max():.4e}")
    print(f"  osc averaging applied: {applied}")
    # phi is oscillated FLUX weight (flux x P), NOT bare probability — so it is NOT
    # bounded to [0,1] and unitarity is NOT asserted.
    ok = shape_ok and finite and nonneg
    print(f"  overall floors: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
