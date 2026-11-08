#!/usr/bin/env python3
"""Build every input of the binned fit. One subcommand per product, each a thin
wrapper over a ``pynu.build`` function. Input paths default to the release tree
(``data/``, ``configs/events/``) where the input ships with it. The only environment
variable read (by nuSQuIDS itself) is NUSQUIDS_DATA_PATH.

  sk-response    SK response from the event MC (SuperK_fullMC_7_2_x100.h5)
  sk-pin         Ninv = sk-response pinned to the SK release NO tables at the anchor point
  fluxq          flux-ratio q table for one response (sk | ic | orca)
  ic-response    IC DeepCore response (160 x 320 true grid, |mode| axis)
  orca-response  ORCA response (flat900)
  aux            ic_aux.npz / orca_aux.npz
  tensors        one oscillated-flux cell: --experiment {sk,ic,orca} --ordering
                 {NO,IO} --task N (N = i_dm*41 + i_s23); skipped if the output exists.
                 --single writes the SK anchor tensor instead.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

DATA_IC = os.path.join(REPO, "data", "IceCube")
DATA_ORCA = os.path.join(REPO, "data", "ORCA")
EV = os.path.join(REPO, "configs", "events")


def _sk_response(a):
    from pynu.build.sk_response import build_sk_response
    build_sk_response(os.path.join(EV, "SK2023.xml"), a.sk_mc, a.sk_data, a.out)


def _sk_pin(a):
    from pynu.build.sk_pin import build_ninv
    build_ninv(a.response, a.anchor, a.release_dir, a.out)


def _fluxq(a):
    from pynu.build.fluxq import build_fluxq
    site = {"sk": "sk", "ic": "spl", "orca": "spl"}[a.experiment]
    center = {"sk": "geometric", "ic": "stored", "orca": "stored"}[a.experiment]
    build_fluxq(a.response, a.out, site=site, center=center)


def _ic_response(a):
    from pynu.build.ic_response import build_ic_response
    build_ic_response(a.parquet, a.data_dir, a.out)


def _orca_response(a):
    from pynu.build.orca_response import build_orca_response
    build_orca_response(a.parquet, a.data_dir, a.out)


def _aux(a):
    from pynu.build import aux
    if a.experiment == "ic":
        aux.build_ic_aux(os.path.join(EV, "IC_DeepCore.xml"),
                         a.mc or os.path.join(DATA_IC, "IC_MC.parquet"),
                         a.data or os.path.join(DATA_IC, "IC_data.parquet"),
                         a.data_dir or DATA_IC, a.response, a.out)
    else:
        aux.build_orca_aux(os.path.join(EV, "ORCA.xml"),
                           a.mc or os.path.join(DATA_ORCA, "ORCA_MC_dataverse_with_muons.parquet"),
                           a.data or os.path.join(DATA_ORCA, "ORCA_data_dataverse.parquet"),
                           a.response, a.out)


def _tensors(a):
    from pynu.build import osc
    if a.single:
        if a.experiment != "sk":
            sys.exit("--single is the SK anchor path only")
        out = a.out or os.path.join(a.out_dir, "osc_tensor_pe_nominal.npz")
        osc.build_sk_single(out)
        return
    if a.task is None or a.ordering is None:
        sys.exit("tensors needs --ordering and --task (or --single)")
    i, j = osc.cell_of_task(a.task)
    dst = osc.cell_path(a.out_dir, i, j)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"SKIP {dst} already present")
        return
    os.makedirs(a.out_dir, exist_ok=True)
    if a.experiment == "sk":
        out = osc.build_sk_cell(a.task, a.ordering, a.out_dir)
    elif a.experiment == "ic":
        out = osc.build_ic_cell(a.task, a.ordering, a.out_dir,
                                os.path.join(EV, "IC_DeepCore.xml"),
                                a.mc or os.path.join(DATA_IC, "IC_MC.parquet"),
                                a.response)
    else:
        out = osc.build_orca_cell(a.task, a.ordering, a.out_dir,
                                  os.path.join(EV, "ORCA.xml"),
                                  a.mc or os.path.join(DATA_ORCA, "ORCA_MC_dataverse_with_muons.parquet"),
                                  a.response)
    print(f"[tensors] wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sk-response", help="SK response from the event MC")
    p.add_argument("--sk-mc", required=True, help="SuperK_fullMC_7_2_x100.h5")
    p.add_argument("--sk-data", default=os.path.join(REPO, "data", "SuperK",
                                                     "sk_2023_data_bins.csv"),
                   help="SK observed counts per analysis bin (default: the release copy)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_sk_response)

    p = sub.add_parser("sk-pin", help="Ninv: pin the SK response to the release NO tables")
    p.add_argument("--response", required=True, help="the sk-response output")
    p.add_argument("--anchor", required=True,
                   help="osc_tensor_pe_nominal.npz (tensors --experiment sk --single)")
    p.add_argument("--release-dir", required=True, help="sk_atm_2023/bins/normal")
    p.add_argument("--out", required=True, help="Ninv.npz")
    p.set_defaults(fn=_sk_pin)

    p = sub.add_parser("fluxq", help="flux-ratio q table for one response")
    p.add_argument("--experiment", choices=["sk", "ic", "orca"], required=True)
    p.add_argument("--response", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_fluxq)

    p = sub.add_parser("ic-response", help="IC DeepCore response (160 x 320, |mode| axis)")
    p.add_argument("--parquet", default=os.path.join(DATA_IC, "IC_MC.parquet"))
    p.add_argument("--data-dir", default=DATA_IC)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_ic_response)

    p = sub.add_parser("orca-response", help="ORCA response (flat900)")
    p.add_argument("--parquet", default=os.path.join(DATA_ORCA,
                                                     "ORCA_MC_dataverse_with_muons.parquet"))
    p.add_argument("--data-dir", default=DATA_ORCA)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_orca_response)

    p = sub.add_parser("aux", help="ic_aux.npz / orca_aux.npz")
    p.add_argument("--experiment", choices=["ic", "orca"], required=True)
    p.add_argument("--response", required=True, help="the response the aux pairs with")
    p.add_argument("--out", required=True)
    p.add_argument("--mc", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--data-dir", default=None, help="IC: reco edges + hs_*.csv")
    p.set_defaults(fn=_aux)

    p = sub.add_parser("tensors", help="one oscillated-flux lattice cell")
    p.add_argument("--experiment", choices=["sk", "ic", "orca"], required=True)
    p.add_argument("--ordering", choices=["NO", "IO"], default=None)
    p.add_argument("--task", type=int, default=None, help="i_dm*41 + i_s23")
    p.add_argument("--out-dir", default=None, help="directory of the cell tensors")
    p.add_argument("--response", default=None, help="IC/ORCA: the response npz")
    p.add_argument("--mc", default=None)
    p.add_argument("--single", action="store_true",
                   help="SK only: the anchor tensor of the Ninv pin")
    p.add_argument("--out", default=None, help="--single output path")
    p.set_defaults(fn=_tensors)

    a = ap.parse_args(argv)
    if a.cmd == "tensors" and not a.out_dir and not (a.single and a.out):
        ap.error("tensors needs --out-dir (with --single: --out or --out-dir)")
    if a.cmd == "tensors" and a.experiment in ("ic", "orca") and not a.single \
            and a.response is None:
        ap.error("--response is required for ic/orca tensors")
    a.fn(a)


if __name__ == "__main__":
    main()
