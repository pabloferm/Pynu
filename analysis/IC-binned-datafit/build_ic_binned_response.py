#!/usr/bin/env python3
"""IC DeepCore binned response builder — thin CLI wrapper.

Thin wrapper over ``pynu.Experiments.ic_binned_builder.build_ic_response``,
mirroring the SK precedent (``analysis/SK-binned-datafit/build_sk_binned_response.py``):
heavy imports are deferred into ``main`` so ``--help`` works without pyarrow or
the IC MC.

PRODUCTION BUILD = ``--mode-axis --grids L3``, both defaults here.

★ ``--mode-axis`` IS NOT OPTIONAL FOR FITTING. It adds |NEUT mode| to the class
signature (47 classes), which is what makes the 11 Mode-keyed cross-section dials
representable in the response at all. ``ICBinnedEngine`` REFUSES a 12-class
response rather than producing quiet nonsense, so a build without this flag will
not drive a fit.

★ THE LADDER MUST STAY UNSNAPPED. ``--snap-e-edges`` inserts band thresholds into
the true-E edge set, which raises nE above the ladder's own value — and the
oscillation tensors index the response by integer cell, so a snapped response
cannot be indexed by tensors built on the unsnapped grid. Omit it (the default)
unless you are rebuilding tensors to match.

Deterministic and local: no nuSQuIDS.

Usage (from the repo root):
  python3 analysis/IC-binned-datafit/build_ic_binned_response.py \
      --out-prefix ic_response_modeaxis
  # -> ic_response_modeaxis_L3.npz
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_PARQUET = os.path.join(REPO_ROOT, "data", "IceCube", "IC_MC.parquet")
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data", "IceCube")


def main():
    ap = argparse.ArgumentParser(
        description="Build the IC binned response npz (production: mode-axis, "
                    "L3, unsnapped — the only build ICBinnedEngine accepts).")
    ap.add_argument("--parquet", default=DEFAULT_PARQUET,
                    help="IC MC parquet (default: data/IceCube/IC_MC.parquet)")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                    help="dir with _E_reco_bins.npy / _cosT_reco_bins.npy "
                         "(default: data/IceCube)")
    ap.add_argument("--out-prefix", required=True,
                    help="output prefix; writes <prefix>_<grid>.npz per grid")
    ap.add_argument("--grids", nargs="+", default=["L3"],
                    help="ladder levels to build (default: L3, the production grid)")
    ap.add_argument("--no-mode-axis", action="store_true",
                    help="build WITHOUT the |Mode| class axis. The engine will "
                         "refuse the result — diagnostic use only.")
    ap.add_argument("--snap-e-edges", nargs="+", type=float, default=(),
                    metavar="GEV",
                    help="band thresholds forced onto the true-E edge set. Omit "
                         "for the shipped ladder; a snapped response cannot be "
                         "indexed by tensors built on the unsnapped grid.")
    args = ap.parse_args()

    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from pynu.Experiments import ic_binned_builder as B   # deferred: needs pyarrow

    for p in (args.parquet, args.data_dir):
        if not os.path.exists(p):
            sys.exit(f"PREFLIGHT FAIL: missing {p}")

    mode_axis = not args.no_mode_axis
    print(f"[build] IC response -> grids={args.grids} mode_axis={mode_axis} "
          f"snap_e={tuple(args.snap_e_edges)}")
    written = []
    for gl in args.grids:
        out = f"{args.out_prefix}_{gl}.npz"
        B.build_ic_response(args.parquet, args.data_dir, gl, out_path=out,
                            mode_axis=mode_axis,
                            snap_e=tuple(args.snap_e_edges))
        written.append(out)
        print(f"  {gl} -> {out}")

    print("=== IC binned response built ===")
    ok = all(os.path.isfile(p) for p in written)
    for p in written:
        print(f"  {'PASS' if os.path.isfile(p) else 'FAIL'}  {p}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
