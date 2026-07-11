#!/usr/bin/env python3
"""Cell-by-cell comparator: reference (standalone-engine) scan jsons vs
modular-path jsons (run_sk_binned_scan_row_worker.py outputs). Both sides use
the identical row-json schema ({arm, arm_spec, row, dm231, n_dials,
nuisance_names, points[{dm231, sin2theta23, chi2, best_dcp_idx, nuisance}]}).

Acceptance (two documented regimes — the report states WHICH applies):

  EXACT     — target. The modular kernel is the engine's own analytic kernel,
              and the worker replays the identical seed / bounds / ftol /
              dCP-warm-chain / s23-warm-chain / restart-polish sequencing
              => every cell must satisfy delta chi2 == 0.0, identical
              best_dcp_idx, identical nuisance.

  SCATTER   — documented fallback, admissible ONLY when the warm-chain
              trajectory demonstrably differs (different seed file, different
              polish count, or a scipy version change on the node — flagged
              here via best_dcp_idx / nuisance divergence preceding the first
              chi2 divergence along the row). Then the acceptance bound is the
              heavy-arm convergence scatter: per-cell |delta chi2| <= 1.0
              (max 3.0 on isolated cells).

Usage:
  compare_scan_outputs.py --canonical DIR --modular DIR [--arms a b ...]
                          [--scatter-tol 1.0] [--out report.txt]

Exit 0 = EXACT acceptance. Exit 2 = not exact but within SCATTER (review).
Exit 1 = FAIL (beyond scatter, schema/grid mismatch, or missing rows).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

POINT_KEYS = ["dm231", "sin2theta23", "chi2", "best_dcp_idx", "nuisance"]
TOP_KEYS = ["arm", "arm_spec", "row", "dm231", "n_dials", "nuisance_names",
            "points"]


def load_rows(d, arm):
    rows = {}
    for p in sorted(glob.glob(os.path.join(d, f"{arm}_row*.json"))):
        j = json.load(open(p))
        rows[int(j["row"])] = (p, j)
    return rows


def check_schema(j, path, problems):
    if list(j.keys()) != TOP_KEYS:
        problems.append(f"{path}: top-level keys {list(j.keys())} != {TOP_KEYS}")
    for k, pt in enumerate(j.get("points", [])):
        if list(pt.keys()) != POINT_KEYS:
            problems.append(f"{path}: points[{k}] keys {list(pt.keys())} "
                            f"!= {POINT_KEYS}")
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", required=True,
                    help="reference (standalone-engine) scan output dir")
    ap.add_argument("--modular", required=True,
                    help="modular-path scan output dir")
    ap.add_argument("--arms", nargs="+",
                    default=["r2_fude_ccqe", "r2_fude_ccqe_nmig"])
    ap.add_argument("--scatter-tol", type=float, default=1.0,
                    help="heavy-arm convergence scatter bound (fallback regime)")
    ap.add_argument("--scatter-max", type=float, default=3.0,
                    help="isolated-cell scatter ceiling (fallback regime)")
    ap.add_argument("--out", default=None, help="also write the report here")
    a = ap.parse_args()

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    overall_exact = True
    overall_scatter_ok = True
    problems = []

    for arm in a.arms:
        can = load_rows(a.canonical, arm)
        mod = load_rows(a.modular, arm)
        missing_c = sorted(set(mod) - set(can))
        missing_m = sorted(set(can) - set(mod))
        if missing_m:
            problems.append(f"{arm}: rows missing on modular side: {missing_m}")
        if missing_c:
            problems.append(f"{arm}: rows missing on canonical side: {missing_c}")
        common = sorted(set(can) & set(mod))
        emit(f"\n=== arm {arm}: {len(common)} common rows "
             f"(canonical {len(can)}, modular {len(mod)}) ===")
        if not common:
            overall_exact = overall_scatter_ok = False
            continue

        emit(f"{'row':>3s} {'dm231':>10s} {'max|dX2|':>12s} {'#dX2!=0':>8s} "
             f"{'#dcp!=':>7s} {'max|dnuis|':>12s} {'first-diverge':>13s}")
        arm_max = 0.0
        arm_trajectory_differs = False
        for r in common:
            pc, jc = can[r]
            pm, jm = mod[r]
            check_schema(jm, pm, problems)
            if jc["nuisance_names"] != jm["nuisance_names"]:
                problems.append(f"{arm} row{r}: nuisance_names differ")
                continue
            P, Q = jc["points"], jm["points"]
            if len(P) != len(Q):
                problems.append(f"{arm} row{r}: {len(P)} vs {len(Q)} points")
                continue
            # grid axes must match EXACTLY (sanity floor)
            for k, (p, q) in enumerate(zip(P, Q)):
                if p["dm231"] != q["dm231"] or p["sin2theta23"] != q["sin2theta23"]:
                    problems.append(
                        f"{arm} row{r} pt{k}: grid mismatch "
                        f"({p['dm231']},{p['sin2theta23']}) vs "
                        f"({q['dm231']},{q['sin2theta23']})")
            dchi = np.array([q["chi2"] - p["chi2"] for p, q in zip(P, Q)])
            ddcp = np.array([q["best_dcp_idx"] != p["best_dcp_idx"]
                             for p, q in zip(P, Q)])
            dnu = np.array([np.max(np.abs(np.array(q["nuisance"])
                                          - np.array(p["nuisance"])))
                            for p, q in zip(P, Q)])
            # first index where the warm-chain state (dcp choice or nuisance)
            # diverges — used to attribute nonzero dchi2 to a trajectory split
            div = np.nonzero(ddcp | (dnu > 0.0))[0]
            first_div = int(div[0]) if div.size else -1
            if first_div >= 0:
                arm_trajectory_differs = True
            arm_max = max(arm_max, float(np.max(np.abs(dchi))))
            emit(f"{r:3d} {jc['dm231']:.4e} {np.max(np.abs(dchi)):12.6e} "
                 f"{int(np.sum(dchi != 0.0)):8d} {int(np.sum(ddcp)):7d} "
                 f"{np.max(dnu):12.6e} "
                 f"{('j=%d' % first_div) if first_div >= 0 else '—':>13s}")
            if np.any(np.abs(dchi) > a.scatter_max) or \
                    (np.sum(np.abs(dchi) > a.scatter_tol) > 2):
                overall_scatter_ok = False
            if np.any(dchi != 0.0) or np.any(ddcp) or np.any(dnu > 0.0):
                overall_exact = False

        emit(f"--- arm {arm}: max|dchi2| = {arm_max:.6e}; trajectory "
             + ("DIFFERS (scatter regime admissible)" if arm_trajectory_differs
                else "identical (EXACT regime applies)"))

    emit("\n=== ACCEPTANCE ===")
    for p in problems:
        emit(f"PROBLEM: {p}")
    if problems:
        verdict, code = "FAIL (schema/grid/coverage problems above)", 1
    elif overall_exact:
        verdict, code = ("PASS — EXACT (bit-identical cells, as expected for "
                         "identical trajectories)"), 0
    elif overall_scatter_ok:
        verdict, code = ("REVIEW — not exact, but within the heavy-arm "
                         "convergence scatter (fallback regime). Admissible "
                         "ONLY if the trajectory demonstrably differs — see "
                         "per-row 'first-diverge' column; otherwise treat as "
                         "FAIL and investigate."), 2
    else:
        verdict, code = "FAIL — differences beyond the scatter fallback", 1
    emit(verdict)

    if a.out:
        with open(a.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nreport written: {a.out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
