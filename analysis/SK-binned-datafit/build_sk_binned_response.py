"""SK binned forward-model response builder — thin CLI wrapper (Track Q / C1).

Thin wrapper over the NATIVE, Gate-D-certified PyNuFit method
``PyNuFit(analysis_xml).BuildBinnedResponse`` (``pynu/PyNuFit.py:1424`` ->
``pynu.binned.builder.build_response``). One MC pass through the live
experiment, so every convention (NC w_no fix, NORM, WMC, CC-mask encoding,
the DIS |Mode|>25*CC quirk) is inherited from the event engine and never
re-implemented. Follows the ``ic_build_tensors.py`` pattern: heavy imports
(``from pynu import PyNuFit``) are deferred into ``main`` so ``--help`` works
without the SK MC + nuSQuIDS environment.

FROZEN-ORIGINAL CONVENTION: the standalone builder at
``analysis/SuperK-datafit/sk_binned/build_sk_response.py`` remains the
byte-parity reference for Gate D and stays the frozen SLURM entry point until
Track S E7. It MUST NOT be modified. This wrapper is the native-method
equivalent for the SK-binned-datafit pipeline; Gate-D certified the two produce
byte-identical artifacts.

PRODUCTION DEFAULTS: the production SK response is 400x80 (400 true-E cells x
80 true-cz cells). The method's own defaults are n_etrue=200 / n_cztrue=40 —
this wrapper defaults both to production (400/80). The response npz is a single
per-experiment artifact (external data), passed downstream by path to the scan
worker.

ENVIRONMENT: a full run needs the SK MC + nuSQuIDS env (FASRC). Local smoke =
``python3 build_sk_binned_response.py --help`` only (import is deferred).

Usage on FASRC (from the staged Pynu root, env sourced):
  source env_fasrc_r2.sh
  export PYNU=$ROOT/backup_pynu/Pynu; export PYTHONPATH=$PYNU:$PYTHONPATH
  python build_sk_binned_response.py \
      --config SK2023_Atm_datafit_r2_fude_ccqe_full.xml \
      --output sk_response.npz --n-etrue 400 --n-cztrue 80
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser(
        description="Thin CLI over PyNuFit.BuildBinnedResponse (native "
                    "Gate-D-certified response builder). Production = 400x80.")
    ap.add_argument("--config", required=True,
                    help="FULL PyNuFit analysis XML (experiment blocks + MC)")
    ap.add_argument("--output", required=True,
                    help="response npz output path (SKBinnedEngine-loadable)")
    ap.add_argument("--n-etrue", type=int, default=400,
                    help="true-E grid density (production 400; method default "
                         "200 — this wrapper defaults to production 400)")
    ap.add_argument("--n-cztrue", type=int, default=80,
                    help="true-cz grid density (production 80; method default "
                         "40 — this wrapper defaults to production 80)")
    ap.add_argument("--exp-name", default=None,
                    help="experiment to build for (default: the single "
                         "experiment, matching the standalone keys()[0])")
    ap.add_argument("--pynu-root", default=None,
                    help="prepend this path to sys.path before importing pynu")
    args = ap.parse_args()

    # Heavy imports deferred so --help works without the SK MC + nuSQuIDS env.
    import os
    if args.pynu_root:
        root = os.path.abspath(args.pynu_root)
        if root not in sys.path:
            sys.path.insert(0, root)
    from pynu import PyNuFit

    pf = PyNuFit(args.config, verbosity=False)
    exp_name = args.exp_name or list(pf.Experiments.keys())[0]
    print(f"[build] response -> pf.BuildBinnedResponse(exp={exp_name}, "
          f"n_etrue={args.n_etrue}, n_cztrue={args.n_cztrue})")
    resp = pf.BuildBinnedResponse(exp_name=exp_name, out_path=args.output,
                                  n_etrue=args.n_etrue,
                                  n_cztrue=args.n_cztrue)

    print("=== SK binned response built ===")
    print(f"  out: {args.output}")
    print(f"  exp: {exp_name}  grid: {args.n_etrue}x{args.n_cztrue}")
    ok = os.path.isfile(args.output)
    print(f"  file written: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
