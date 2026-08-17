#!/usr/bin/env python3
"""ORCA osc-tensor builder. FASRC/HPC-only (needs nuSQuIDS).

Builds the oscillated-flux tensor Phi[n_dcp, 2, 3, nE=40, nZ=80] for ORCA on the
MC's NATIVE quantized true grid (40 true-E x 80 true-cz), one slice per dCP node,
at a (dm231, s23) point. Reuses the SK binned builder's osc helpers VERBATIM
(_snapshot_osc_state / _set_grid_coords / _eval_point / _restore_osc_state) so
propagation, units, the Dm231_bar->Dm231 convention, the averaging knob, and the
byte-exact try/finally state-restore are all inherited from the certified SK
machinery. The ONLY ORCA change vs the SK tensor builder: the true grid comes
from the ORCA response npz's `e_true_centers`/`cz_true_centers` (the quantized MC
values), NOT make_true_grid — because ORCA's true side is intrinsically binned and
each event carries exactly its cell-centre true coordinate, so the tensor is EXACT
per event (no cell-centering residue; only float-associativity).

Phi convention (mirrors SK):
  Phi[i_dcp, type, flavor, iE, iZ]  — type 0=nu/1=nubar, flavor 0=e/1=mu/2=tau,
  iE in 0..39 (true-E), iZ in 0..79 (true-cz). float32.

★ ONE FILE PER GRID POINT. `run_orca_binned_fit_worker.py` reads these back
through `--phi-dir` + `--phi-pattern` (default `orca_phi_{i:03d}_{j:03d}.npz`),
and cross-checks the stored `dm231`/`s23` scalars against the grid point it is
fitting — so a mis-named tensor is an error, not a silently wrong surface.

Usage on a cluster (from the repo root, env sourced):
  python3 analysis/ORCA-binned-datafit/build_orca_osc_tensors.py \
      --config analysis/AnalysisFiles/ORCA_Atm_r2_fude_ccqe.xml \
      --response orca_binned_response_flat900.npz \
      --dm231 2.511e-3 --s23 0.572 --dcp-nodes 1.36 \
      --out orca_phi_000_000.npz
"""
import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


def add_pynu_root(root):
    root = os.path.abspath(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def main():
    ap = argparse.ArgumentParser(
        description="Build one ORCA oscillated-flux tensor npz at a (dm231, s23) "
                    "point on the response's native true-cell centres.")
    ap.add_argument("--config", required=True, help="ORCA manifest XML")
    ap.add_argument("--response", required=True,
                    help="ORCA response npz (source of the native true centres)")
    ap.add_argument("--dm231", type=float, required=True)
    ap.add_argument("--s23", type=float, required=True)
    ap.add_argument("--s13", type=float, default=None)
    ap.add_argument("--dm221", type=float, default=None,
                    help="override Dm221 (set 0 together with --dm231 0 for the "
                         "no-oscillation identity tensor, mirroring the SK noosc)")
    ap.add_argument("--dcp-nodes", type=float, nargs="+", default=[0.0],
                    help="dCP node values in radians (default single 0.0 slice)")
    ap.add_argument("--avg-scale", default=None,
                    help="fast-osc averaging selector ('4pi'/'2pi'/'off'/float); "
                         "None -> honour PYNU_OSC_AVG_SCALE env / object default. "
                         "★ The binned ORCA arm requires averaging OFF: with it "
                         "on the per-cell flux is no longer exact.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pynu-root", default=REPO_ROOT,
                    help="Pynu repo root to prepend to sys.path (default: the "
                         "repo this script ships in)")
    args = ap.parse_args()

    add_pynu_root(args.pynu_root)
    from pynu import PyNuFit
    from pynu.Experiments import sk_binned_builder as B   # certified osc helpers

    # ---- native ORCA true-cell centres (quantized MC values) ----
    resp = np.load(args.response, allow_pickle=True)
    e_c = np.asarray(resp["e_true_centers"], float)     # (40,)
    z_c = np.asarray(resp["cz_true_centers"], float)    # (80,)
    if not (np.all(np.isfinite(e_c)) and np.all(np.isfinite(z_c))):
        raise RuntimeError("response has NaN true-centres — a true bin was unpopulated; "
                           "the tensor grid must match the response's populated cells")

    # ---- live osc object from PyNuFit(config) ----
    pynufit = PyNuFit(args.config, verbosity=False)
    exp_name = list(pynufit.Experiments.keys())[0]
    osc = pynufit.physics_tunes[exp_name].OscillationTunes

    dcp_arr = np.asarray(args.dcp_nodes, float)

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
            B._eval_point(osc, args.dm231, args.s23, dcp, nE, nZ, s13=args.s13,
                          dm221=args.dm221)
            for dcp in dcp_arr])
    finally:
        B._restore_osc_state(osc, snap)

    np.savez_compressed(
        args.out, phi=phi.astype(np.float32),
        dm231=float(args.dm231), s23=float(args.s23), dcp=dcp_arr,
        e_true_centers=e_c, cz_true_centers=z_c,
        osc_averaging=np.array(str(applied)),
        grid=np.array([nE, nZ]))

    # ---- sanity floors ----
    print("=== ORCA osc tensor built ===")
    print(f"  out: {args.out}")
    print(f"  phi shape: {phi.shape}  (expect ({len(dcp_arr)}, 2, 3, {nE}, {nZ}))")
    shape_ok = phi.shape == (len(dcp_arr), 2, 3, nE, nZ)
    print(f"  shape gate: {'PASS' if shape_ok else 'FAIL'}")
    print(f"  grid nE,nZ = {nE},{nZ}  (expect 40,80)")
    finite = bool(np.all(np.isfinite(phi)))
    nonneg = bool((phi >= -1e-9).all())
    print(f"  phi finite: {finite}  nonneg: {nonneg}  "
          f"min={phi.min():.4e} max={phi.max():.4e}")
    print(f"  osc averaging applied: {applied}")
    # phi is oscillated FLUX weight (flux x P), NOT bare probability — so it is NOT
    # bounded to [0,1] and unitarity is NOT asserted. We assert finiteness +
    # nonnegativity + shape.
    ok = shape_ok and finite and nonneg
    print(f"  overall floors: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
