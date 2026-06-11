#!/usr/bin/env python3
"""Precompute oscillated-flux tensors Phi[type, flavor, E_cell, cz_cell] on the
true grid for the SK binned forward model.

Phi is exactly what PhysicsWeight is in the event engine — the nuSQuIDS-evolved
atmospheric flux evaluated by EvalFlavor at (flavor, cosZ, E, type) — but at
true-cell centers instead of per-event coordinates (the controlled Step-3
approximation, gated by the binned-vs-event convergence check). The evaluation
reuses the production AtmosphericOscillations object by overriding its
per-event coordinate arrays with the tiled grid, so propagation, units, flux
initialization, and the (Dm231_bar -> Dm231) convention are inherited verbatim.

Grid task t in [0, 224]: (i_dm, i_s23) = (t // 15, t % 15) on the SAME grid as
run_sk_datafit_point_worker (linspace [2.0e-3, 3.5e-3] x [0.40, 0.80], 15 pts;
13 dCP values linspace(0, 2pi, endpoint=False)) -> osc_tensor_<i>_<j>.npz with
phi[13, 2, 3, nE, nZ] float32.

Task 225 ("specials"): no-osc identity (Dm231=Dm221=0), the SK 2023 release NO
best fit in Dm231 convention (2.4741e-3, 0.45, dcp=4.3982, s13=0.020), and the
two comparison points A/B with the full 13-dCP set -> osc_tensor_specials.npz.

Usage (cluster):
    python build_osc_tensors.py --config <xsec_barr_ntag.xml> --task <0-225> \
        --output-dir <dir> [--n-etrue 200 --n-cztrue 40]
"""
import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)
from run_sk_datafit_row_worker import setup_pynufit_datafit  # noqa: E402
from build_sk_response import make_true_grid  # noqa: E402

DM_GRID = np.linspace(2.0e-3, 3.5e-3, 15)
S23_GRID = np.linspace(0.40, 0.80, 15)
DCP_GRID = np.linspace(0.0, 2.0 * np.pi, 13, endpoint=False)

SK_BF = dict(dm231=2.4741e-3, s23=0.45, dcp=4.3982, s13=0.020)  # q13-free, Dm231 conv.
POINT_A = dict(dm231=DM_GRID[5], s23=S23_GRID[6])   # point_005_006 (near-truth)
POINT_B = dict(dm231=DM_GRID[8], s23=S23_GRID[9])   # point_008_009 (displaced)


def grid_centers(e_edges, z_edges):
    e_c = np.sqrt(e_edges[:-1] * e_edges[1:])
    z_c = 0.5 * (z_edges[:-1] + z_edges[1:])
    return e_c, z_c


def set_grid_coords(osc, e_c, z_c):
    """Override the osc object's per-event arrays with the tiled (type, flavor,
    E, cz) grid so GetOscillations evaluates at cell centers."""
    nE, nZ = e_c.size, z_c.size
    E2, Z2 = np.meshgrid(e_c, z_c, indexing="ij")     # (nE, nZ)
    cells_E = E2.ravel()
    cells_Z = Z2.ravel()
    n_cells = cells_E.size
    types, flavors = [], []
    Es, Zs = [], []
    for t in (0, 1):           # 0 = nu, 1 = nubar  (NSQNeutrinoType convention)
        for f in (0, 1, 2):    # e, mu, tau         (NSQNeutrinoFlavor convention)
            types.append(np.full(n_cells, t, dtype=np.uint32))
            flavors.append(np.full(n_cells, f, dtype=np.uint32))
            Es.append(cells_E)
            Zs.append(cells_Z)
    osc.ETrue = np.concatenate(Es)
    osc.CosZTrue = np.concatenate(Zs)
    osc.NSQneutype = np.concatenate(types).tolist()
    osc.NSQneuflavor = np.concatenate(flavors).tolist()
    return nE, nZ


def eval_point(osc, dm231, s23, dcp, nE, nZ, s13=None, dm221=None):
    P = osc.Parameters
    P["Dm231"] = dm231
    if "Dm231_bar" in P:
        P["Dm231_bar"] = dm231
    P["Sin2Theta23"] = s23
    P["dCP"] = dcp
    if s13 is not None:
        P["Sin2Theta13"] = s13
    if dm221 is not None:
        P["Dm221"] = dm221
    osc.reset_cache()
    w = np.asarray(osc.GetOscillations(), float)
    return w.reshape(2, 3, nE, nZ).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-etrue", type=int, default=200)
    ap.add_argument("--n-cztrue", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pynufit, _, _ = setup_pynufit_datafit(args.config)
    exp_key = list(pynufit.Experiments.keys())[0]
    exp = pynufit.Experiments[exp_key]
    osc = pynufit.physics_tunes[exp_key].OscillationTunes

    e_edges, z_edges = make_true_grid(exp.Etrue_min, exp.Etrue_max,
                                      args.n_etrue, args.n_cztrue)
    e_c, z_c = grid_centers(e_edges, z_edges)
    nE, nZ = set_grid_coords(osc, e_c, z_c)
    nom_dm221 = osc.Parameters.get("Dm221", 7.41e-5)
    nom_s13 = osc.Parameters.get("Sin2Theta13", 0.022)

    if args.task <= 224:
        i, j = args.task // 15, args.task % 15
        dm, s23 = DM_GRID[i], S23_GRID[j]
        print(f"[osc] grid point ({i},{j}): dm231={dm:.4e} s23={s23:.4f}, 13 dCP")
        phi = np.stack([eval_point(osc, dm, s23, dcp, nE, nZ) for dcp in DCP_GRID])
        out = os.path.join(args.output_dir, f"osc_tensor_{i:03d}_{j:03d}.npz")
        np.savez_compressed(out, phi=phi, dm231=dm, s23=s23, dcp=DCP_GRID,
                            e_edges=e_edges, z_edges=z_edges)
    else:
        print("[osc] specials: no-osc, SK best fit, points A/B")
        specials = {}
        # no-osc identity: both splittings zero (validated mechanism, Phase A)
        specials["noosc"] = eval_point(osc, 0.0, 0.5, 0.0, nE, nZ, dm221=0.0)
        # restore nominal dm221 for everything else
        specials["skbf"] = eval_point(osc, SK_BF["dm231"], SK_BF["s23"],
                                      SK_BF["dcp"], nE, nZ, s13=SK_BF["s13"],
                                      dm221=nom_dm221)
        # restore nominal s13 for A/B (they use config nominal s13)
        specials["pointA"] = np.stack([
            eval_point(osc, POINT_A["dm231"], POINT_A["s23"], dcp, nE, nZ,
                       s13=nom_s13, dm221=nom_dm221) for dcp in DCP_GRID])
        specials["pointB"] = np.stack([
            eval_point(osc, POINT_B["dm231"], POINT_B["s23"], dcp, nE, nZ,
                       s13=nom_s13, dm221=nom_dm221) for dcp in DCP_GRID])
        out = os.path.join(args.output_dir, "osc_tensor_specials.npz")
        np.savez_compressed(out, e_edges=e_edges, z_edges=z_edges, dcp=DCP_GRID,
                            meta=json.dumps({"sk_bf": SK_BF,
                                             "pointA": POINT_A, "pointB": POINT_B,
                                             "nom_s13": float(nom_s13),
                                             "nom_dm221": float(nom_dm221)}),
                            **specials)
    print(f"[osc] wrote {out}")


if __name__ == "__main__":
    main()
