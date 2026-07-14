"""SK binned oscillated-flux tensor builder — thin CLI wrapper (Track Q / C1).

Thin wrapper over the NATIVE, Gate-D-certified PyNuFit method
``PyNuFit(analysis_xml).BuildOscTensors`` (``pynu/PyNuFit.py:1468`` ->
``pynu.binned.builder.build_tensors``). Runs against the production
``AtmosphericOscillations`` object, so propagation, units, flux init, and the
Dm231_bar->Dm231 convention are inherited from the event engine; the osc
object's per-event state is snapshotted and restored around the build (even on
a mid-build exception). Follows the ``ic_build_tensors.py`` pattern: heavy
imports (``from pynu import PyNuFit``) are deferred into ``main`` so ``--help``
works without the SK MC + nuSQuIDS environment.

FROZEN-ORIGINAL CONVENTION: the standalone builder at
``analysis/SuperK-datafit/sk_binned/build_osc_tensors.py`` remains the
byte-parity reference for Gate D and stays the frozen SLURM entry point until
Track S E7. It MUST NOT be modified. This wrapper is the native-method
equivalent for the SK-binned-datafit pipeline; Gate-D certified the two produce
byte-identical artifacts.

PRODUCTION CONVENTIONS:
  * dCP nodes = ``np.linspace(0, 2*pi, 20, endpoint=False)`` (the 20-node
    production axis; the standalone DCP_GRID convention is
    ``build_osc_tensors.py:44``, there N=13 — production SK-binned uses N=20).
    Expose via --n-dcp (default 20).
  * s13 default 0.0220 (Sin2Theta13).
  * osc_averaging '4pi' in production — the method pulls this from the active
    <BinnedEngine> <osc_averaging> declaration in the XML when present;
    --avg-scale overrides it ('4pi'/'2pi'/'off'/float).
  * ONE (dm231, s23) node per invocation. Output naming
    ``osc_tensor_<row>_<col>.npz`` (zero-padded to 3, matching what
    run_sk_binned_scan_row_worker.py:210 loads:
    ``osc_tensor_{row:03d}_{j:03d}.npz``). Pass --row/--col.

The tensor dir (one npz per (dm,s23) node) is an external data artifact passed
downstream to the scan worker by path (--tensors).

ENVIRONMENT: a full run needs the SK MC + nuSQuIDS env (FASRC). Local smoke =
``python3 build_sk_osc_tensors.py --help`` only (import is deferred).

Usage on FASRC (from the staged Pynu root, env sourced):
  source env_fasrc_r2.sh
  export PYNU=$ROOT/backup_pynu/Pynu; export PYTHONPATH=$PYNU:$PYTHONPATH
  python build_sk_osc_tensors.py \
      --config SK2023_Atm_datafit_r2_fude_ccqe_full.xml \
      --dm231 2.5e-3 --s23 0.5 --row 10 --col 5 \
      --outdir /path/to/tensors --n-etrue 400 --n-cztrue 80
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(
        description="Thin CLI over PyNuFit.BuildOscTensors (native "
                    "Gate-D-certified osc-tensor builder). One (dm,s23) node.")
    ap.add_argument("--config", required=True,
                    help="FULL PyNuFit analysis XML (experiment blocks + MC)")
    ap.add_argument("--dm231", type=float, required=True,
                    help="Dm231 (eV^2) for this node")
    ap.add_argument("--s23", type=float, required=True,
                    help="sin^2(theta23) for this node")
    ap.add_argument("--row", type=int, required=True,
                    help="dm grid row index i -> osc_tensor_<i>_<j>.npz")
    ap.add_argument("--col", type=int, required=True,
                    help="s23 grid col index j -> osc_tensor_<i>_<j>.npz")
    ap.add_argument("--outdir", required=True,
                    help="tensor output dir; file = osc_tensor_<row>_<col>.npz")
    ap.add_argument("--n-dcp", type=int, default=20,
                    help="dCP node count; nodes = linspace(0, 2pi, N, "
                         "endpoint=False). Production = 20 "
                         "(build_osc_tensors.py:44 DCP_GRID convention)")
    ap.add_argument("--s13", type=float, default=0.0220,
                    help="Sin2Theta13 (default production 0.0220)")
    ap.add_argument("--n-etrue", type=int, default=400,
                    help="true-E grid density (MUST match the response build; "
                         "production 400)")
    ap.add_argument("--n-cztrue", type=int, default=80,
                    help="true-cz grid density (MUST match the response; "
                         "production 80; method default 40 — this wrapper "
                         "defaults to production 80)")
    ap.add_argument("--avg-scale", default=None,
                    help="fast-osc averaging override ('4pi'/'2pi'/'off'/"
                         "float); None -> the active <BinnedEngine> "
                         "<osc_averaging> declaration (production '4pi')")
    ap.add_argument("--exp-name", default=None,
                    help="experiment (default: the single experiment)")
    ap.add_argument("--pynu-root", default=None,
                    help="prepend this path to sys.path before importing pynu")
    args = ap.parse_args()

    # Heavy imports deferred so --help works without the SK MC + nuSQuIDS env.
    import numpy as np
    if args.pynu_root:
        root = os.path.abspath(args.pynu_root)
        if root not in sys.path:
            sys.path.insert(0, root)
    from pynu import PyNuFit

    # Production dCP axis: linspace(0, 2pi, N, endpoint=False)
    # (matches run_sk_binned_scan_row_worker.py:206 / build_osc_tensors.py:44).
    dcp_nodes = np.linspace(0.0, 2.0 * np.pi, args.n_dcp, endpoint=False)
    avg = None if (args.avg_scale is not None
                   and str(args.avg_scale).lower() in ("off", "", "none")) \
        else args.avg_scale

    pf = PyNuFit(args.config, verbosity=False)
    exp_name = args.exp_name or list(pf.Experiments.keys())[0]

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir,
                            f"osc_tensor_{args.row:03d}_{args.col:03d}.npz")
    print(f"[build] tensor node ({args.row},{args.col}) -> pf.BuildOscTensors"
          f"(dm={args.dm231:.6e}, s23={args.s23:.6f}, {args.n_dcp} dCP)")
    phi, meta = pf.BuildOscTensors(
        args.dm231, args.s23, exp_name=exp_name, dcp_nodes=dcp_nodes,
        s13=args.s13, n_etrue=args.n_etrue, n_cztrue=args.n_cztrue,
        avg_scale=avg, out_path=out_path)

    print("=== SK osc tensor built ===")
    print(f"  out: {out_path}")
    print(f"  exp: {exp_name}  (dm231={args.dm231:.6e}, s23={args.s23:.6f})")
    print(f"  phi shape: {phi.shape}  n_dcp: {args.n_dcp}")
    finite = bool(np.all(np.isfinite(phi)))
    print(f"  phi finite: {finite}")
    ok = os.path.isfile(out_path) and finite
    print(f"  overall: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
