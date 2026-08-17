#!/usr/bin/env python3
"""ORCA binned forward-model response builder — thin CLI wrapper.

Thin wrapper over ``pynu.Experiments.orca_binned_builder.build_orca_response``,
mirroring the SK precedent (``analysis/SK-binned-datafit/build_sk_binned_response.py``):
heavy imports are deferred into ``main`` so ``--help`` works without pyarrow or
the ORCA MC.

PRODUCTION LAYOUT is ``flat900`` — the 900-bin (pid x reco-E x reco-cz) layout the
``ORCABinnedEngine`` requires. The engine refuses any other schema at load
(``SCHEMA_FLAT900`` guard), so ``--layout v1_300`` exists only to keep the original
Track-R artifact reproducible; it will NOT drive a fit.

Deterministic and local: no nuSQuIDS, no oscillation code. This is one pass over
the MC parquet, reshaping it into a COO response matrix.

Usage (from the repo root):
  python3 analysis/ORCA-binned-datafit/build_orca_binned_response.py \
      --out orca_binned_response_flat900.npz
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_PARQUET = os.path.join(REPO_ROOT, "data", "ORCA",
                               "ORCA_MC_dataverse_with_muons.parquet")
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data", "ORCA")


def main():
    ap = argparse.ArgumentParser(
        description="Build the ORCA binned response npz (production layout "
                    "flat900 — the only layout ORCABinnedEngine accepts).")
    ap.add_argument("--parquet", default=DEFAULT_PARQUET,
                    help="ORCA MC parquet (default: data/ORCA/"
                         "ORCA_MC_dataverse_with_muons.parquet)")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                    help="dir with the _*_bins.npy edge files (default: data/ORCA)")
    ap.add_argument("--out", required=True, help="response npz output path")
    ap.add_argument("--layout", default="flat900", choices=["flat900", "v1_300"],
                    help="flat900 = production 900-bin (default). v1_300 "
                         "reproduces the original Track-R artifact and is NOT "
                         "loadable by the engine.")
    args = ap.parse_args()

    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from pynu.Experiments import orca_binned_builder as B   # deferred: needs pyarrow

    for p in (args.parquet, args.data_dir):
        if not os.path.exists(p):
            sys.exit(f"PREFLIGHT FAIL: missing {p}")

    print(f"[build] ORCA response -> layout={args.layout} out={args.out}")
    resp = B.build_orca_response(args.parquet, args.data_dir, out_path=args.out,
                                 layout=args.layout)
    print("=== ORCA binned response built ===")
    print(f"  out    : {args.out}")
    print(f"  n_bins : {int(resp['n_bins'])}  nnz: {int(len(resp['R_v']))}")
    ok = os.path.isfile(args.out)
    print(f"  file written: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
