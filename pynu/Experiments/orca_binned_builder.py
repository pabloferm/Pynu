"""ORCA binned-response builder — Track R (2026-07-11).

The ORCA MC parquet (`ORCA_MC_dataverse_with_muons.parquet`) is INTRINSICALLY a
binned response matrix: each neutrino row is one populated response cell
`(true_E_bin, true_cz_bin, reco_E_bin, reco_cz_bin, pdg, pid, current)` with an
explicit weight. The true and reco sides are BOTH fully quantized (one distinct
true_energy / cos(true_zenith) value per true bin, one reco value per reco bin),
so this builder is a RESHAPE / REINDEX of the parquet into the pynu.binned COO
response schema — no MC pass, no digitize (the baked `*_bin_num` columns are
authoritative; a naive `digitize-1` does NOT reproduce them).

Mirrors the npz-key conventions of `pynu/binned/builder.py:build_response`
(the SK builder / Phase D `BuildBinnedResponse`) so the binned engine can load
it with minimal adaptation. ORCA-specific DEVIATIONS from the SK response are
documented inline and in SPEC_trackR_orca_matrices.md §"Deviations".

Read-only on all inputs; writes only its output npz. No pynu import needed for
the response bake (pure parquet reshape) — nuSQuIDS is only for the SEPARATE
osc-tensor build (orca_build_tensors.py).
"""
import json
import hashlib
import os
import numpy as np
import pandas as pd

# `_flat900` is the PRODUCTION reco-bin index (orca_exact_scan.py:70-71) — the one
# `observed_900`, `muon_900` and `nu_cell_index` all key on, and therefore the one
# the flat900 layout must reproduce EXACTLY. Imported, never retyped (SCOPE §1.2).
from .orca_binned_support import (                              # noqa: E402
    _flat900, N_ERECO as XS_N_ERECO, N_CZRECO as XS_N_CZRECO,
    N_PID as XS_N_PID, N_BINS as XS_N_BINS,
)

# Layouts. v1 = the original Track-R 300-bin / pid-in-class response. flat900 =
# the production 900-bin layout (design §4.2 option (a), USER-ADOPTED 2026-08-17):
# pid moves OUT of the class axis and INTO the bin index, so `obs`, `mu900` and
# `few` are reused verbatim from production and the class axis collapses 24 -> 8.
SCHEMA_VERSION_V1 = "orca_binned_response_v1"
SCHEMA_VERSION_FLAT900 = "orca_binned_response_v2_flat900"
SCHEMA_VERSION = SCHEMA_VERSION_FLAT900         # default layout's version
LAYOUTS = ("flat900", "v1_300")

# Channel factorization (verified against the parquet): under v1 the "class" axis of
# the SK response is the physical (pdg, pid, current) triple; under flat900 it is
# (pdg, current) only. pdg in {-16,-14,-12,12,14,16}, pid in {0,1,2} (ORCA
# morphology: shower/HPT/track), current in {0,1} (NC/CC).
# 6*3*2 = 36 possible v1 channels (24 populated); 6*2 = 12 flat900 (8 populated).
PDG_VALUES = np.array([-16, -14, -12, 12, 14, 16], dtype=np.int64)
N_PID = 3
N_CUR = 2


def _load_true_reco_edges(data_dir):
    """Native ORCA bin edges from the .npy sidecar files (n+1 edges each).
    Returns (e_true_edges[41], cz_true_edges[81], e_reco_edges[16], cz_reco_edges[21])."""
    import os
    e_t = np.load(os.path.join(data_dir, "_E_true_bins.npy"))
    cz_t = np.load(os.path.join(data_dir, "_cosT_true_bins.npy"))
    e_r = np.load(os.path.join(data_dir, "_E_reco_bins.npy"))
    cz_r = np.load(os.path.join(data_dir, "_cosT_reco_bins.npy"))
    return e_t, cz_t, e_r, cz_r


def _true_cell_centers(e_true_edges, cz_true_edges, df_nu):
    """The QUANTIZED true value carried by each true bin, read straight from the
    MC (not recomputed from edges), so the osc tensor is evaluated at exactly the
    per-event true coordinate. Returns e_c[n_etrue], z_c[n_cztrue] indexed by the
    1-based bin_num - 1. Verified one distinct value per bin."""
    n_et = len(e_true_edges) - 1     # 40
    n_cz = len(cz_true_edges) - 1     # 80
    e_c = np.full(n_et, np.nan)
    z_c = np.full(n_cz, np.nan)
    ie = df_nu["true_energy_bin_num"].values.astype(int) - 1     # 1-indexed -> 0-based
    iz = df_nu["true_cos_zenith_bin_num"].values.astype(int) - 1
    E = df_nu["true_energy"].values
    cz = np.cos(df_nu["true_zenith"].values)
    # take the unique value present in each bin
    for b in np.unique(ie):
        e_c[b] = E[ie == b][0]
    for b in np.unique(iz):
        z_c[b] = cz[iz == b][0]
    return e_c, z_c


def build_orca_response(parquet_path, data_dir, out_path=None, dial_manifest=None,
                        layout="flat900"):
    """Reshape the ORCA parquet into a pynu.binned COO response npz.

    Args:
      parquet_path: ORCA_MC_dataverse_with_muons.parquet.
      data_dir: dir with the _E_true_bins / _cosT_true_bins / _*_reco_bins .npy.
      out_path: optional npz path.
      dial_manifest: optional iterable of dial names (hashed into an additive key
        for the engine's load-time compatibility check, mirroring the SK builder).
      layout: "flat900" (default, production) or "v1_300" (the original Track-R
        layout, kept so the shipped v1 npz stays reproducible).

    Returns the response dict (the npz payload).

    Response layout — COO keyed slowest->fastest (class, ie_true, iz_true, recobin):
      R_k   int32  class index into `classes`
      R_e   int32  true-E bin (0..n_etrue-1)
      R_z   int32  true-cz bin (0..n_cztrue-1)
      R_b   int32  flat reco bin, 0-based
      R_v   float  summed RAW parquet weight in the cell (NO NORM — the engine
                   applies NORM = FitExposure*1e4 at load, Orca.py:180,183)
      S2_*  the same COO for weight^2 (MC variance).

    The two layouts differ ONLY in where `pid` (ORCA morphology) lives:
      v1_300   classes = (pdg, pid, current), 24 populated; R_b = ie*20 + iz,
               300 bins.
      flat900  classes = (pdg, current), 8 populated; R_b = _flat900(pid, ie, iz)
               = (pid*15 + ie)*20 + iz, 900 bins — the PRODUCTION index, so obs900
               / mu900 / few are reused verbatim. nnz is UNCHANGED (592,099) because
               pid is recoverable as R_b // 300, so (pdg,current,ie,iz,b900) is as
               fine a key as (pdg,pid,current,ie,iz,recobin).
    ORCA deviations from SK build_response:
      * NO era axis (N_ERA=1; no SKPhase). No R_era key.
      * NO energy-scale ±2% Rp/Rm (ORCA reco is fixed-quantized; ORCA uses its
        native E_shift detector dial, not the SK histogram-migration escale).
      * true grid = the parquet's native quantized centres, not make_true_grid.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    df = pd.read_parquet(parquet_path)
    df_nu = df[df["MC_type"] == 1].copy()          # neutrinos only (drop 900 muons)
    N = len(df_nu)

    e_t, cz_t, e_r, cz_r = _load_true_reco_edges(data_dir)
    n_etrue = len(e_t) - 1          # 40
    n_cztrue = len(cz_t) - 1        # 80
    n_ereco = len(e_r) - 1          # 15
    n_czreco = len(cz_r) - 1        # 20

    # ---- indices (bin_num is 1-indexed among nu rows -> subtract 1) ----
    ie = df_nu["true_energy_bin_num"].values.astype(np.int64) - 1
    iz = df_nu["true_cos_zenith_bin_num"].values.astype(np.int64) - 1
    ire = df_nu["reco_energy_bin_num"].values.astype(np.int64) - 1
    irz = df_nu["reco_cos_zenith_bin_num"].values.astype(np.int64) - 1
    pdg = df_nu["pdg"].values.astype(np.int64)
    pid = df_nu["pid"].values.astype(np.int64)
    cur = df_nu["current_type"].values.astype(np.int64)

    # ---- reco bin index + class table (the ONLY layout-dependent step) ----
    # pid must move into the BIN index BEFORE it is dropped from the class column,
    # else the COO key collides and nnz would fall below 592,099.
    if layout == "flat900":
        assert (n_ereco, n_czreco) == (XS_N_ERECO, XS_N_CZRECO), (
            f"reco grid {(n_ereco, n_czreco)} != orca_exact_scan "
            f"{(XS_N_ERECO, XS_N_CZRECO)} — _flat900 would not be the production index")
        n_bins = XS_N_PID * n_ereco * n_czreco           # 900
        assert n_bins == XS_N_BINS, (n_bins, XS_N_BINS)
        recobin = _flat900(pid, ire, irz)                # (pid*15 + ie)*20 + iz
        sig = np.column_stack([pdg, cur])
        class_axis = ["pdg", "current"]
        schema_version = SCHEMA_VERSION_FLAT900
    else:                                                 # "v1_300"
        n_bins = n_ereco * n_czreco                       # 300 reco bins
        recobin = ire * n_czreco + irz                    # flat reco bin, 0-based
        sig = np.column_stack([pdg, pid, cur])
        class_axis = ["pdg", "pid", "current"]
        schema_version = SCHEMA_VERSION_V1

    classes, class_inv = np.unique(sig, axis=0, return_inverse=True)
    n_cls = classes.shape[0]

    w = df_nu["weight"].values.astype(float)
    w2 = w * w

    # ---- COO reduce (each row is already a unique cell, but reduce defensively
    #      so the builder is correct even if a future MC has cell duplicates) ----
    def coo(weights):
        key = ((class_inv.astype(np.int64) * n_etrue + ie)
               * n_cztrue + iz) * n_bins + recobin
        uniq, inv = np.unique(key, return_inverse=True)
        val = np.bincount(inv, weights=weights, minlength=uniq.size)
        b = uniq % n_bins
        rem = uniq // n_bins
        cz_ = rem % n_cztrue
        rem = rem // n_cztrue
        ce = rem % n_etrue
        k = rem // n_etrue
        return (k.astype(np.int32), ce.astype(np.int32), cz_.astype(np.int32),
                b.astype(np.int32), val)

    Rk, Re, Rz, Rb, Rv = coo(w)
    Sk, Se, Sz, Sb, Sv = coo(w2)

    # ---- true cell centres (quantized values carried by the MC) ----
    e_c, z_c = _true_cell_centers(e_t, cz_t, df_nu)

    # ---- self-checks (sanity floors) ----
    nnz = Rv.size
    total_w = float(Rv.sum())
    per_cur_w = {int(c): float(df_nu.loc[df_nu["current_type"] == c, "weight"].sum())
                 for c in np.unique(cur)}
    per_pdgcur_w = {f"{int(p)}_{int(c)}":
                    float(df_nu.loc[(df_nu["pdg"] == p) & (df_nu["current_type"] == c),
                                    "weight"].sum())
                    for p in np.unique(pdg) for c in np.unique(cur)
                    if ((df_nu["pdg"] == p) & (df_nu["current_type"] == c)).any()}

    resp = dict(
        schema_version=np.array(schema_version),
        layout=np.array(layout),
        classes=classes,                      # (n_cls, len(class_axis))
        class_axis=np.array(class_axis),
        R_k=Rk, R_e=Re, R_z=Rz, R_b=Rb, R_v=Rv,
        S2_k=Sk, S2_e=Se, S2_z=Sz, S2_b=Sb, S2_v=Sv,
        n_etrue=np.int64(n_etrue), n_cztrue=np.int64(n_cztrue),
        n_ereco=np.int64(n_ereco), n_czreco=np.int64(n_czreco),
        n_bins=np.int64(n_bins), n_era=np.int64(1),        # ORCA: no era axis
        e_true_edges=e_t, cz_true_edges=cz_t,
        e_reco_edges=e_r, cz_reco_edges=cz_r,
        e_true_centers=e_c, cz_true_centers=z_c,           # quantized MC true values
        meta=json.dumps({
            "source": "ORCA_MC_dataverse_with_muons.parquet",
            "layout": layout, "schema_version": schema_version,
            "n_events_nu": int(N), "n_classes": int(n_cls),
            "nnz_response": int(nnz), "total_weight": total_w, "n_bins": int(n_bins),
            "per_current_weight": per_cur_w,
            "per_pdgcurrent_weight": per_pdgcur_w,
            "grids": {"true": [n_etrue, n_cztrue], "reco": [n_ereco, n_czreco]},
            "deviations_from_sk": ["no_era", "no_escale_RpRm", "native_true_grid"],
        }),
    )
    if dial_manifest is not None:
        manifest = list(dial_manifest)
        h = hashlib.sha256("\n".join(manifest).encode("utf-8")).hexdigest()
        resp["dial_manifest_hash"] = np.array(h)
        resp["dial_manifest"] = np.array(manifest)

    if out_path is not None:
        np.savez_compressed(out_path, **resp)

    return resp


if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser(description="Build the ORCA binned response npz.")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--data-dir", required=True, help="dir with _*_bins.npy edge files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout", default="flat900", choices=list(LAYOUTS),
                    help="flat900 = production 900-bin (default); v1_300 = original")
    args = ap.parse_args()

    resp = build_orca_response(args.parquet, args.data_dir, out_path=args.out,
                               layout=args.layout)
    meta = json.loads(resp["meta"].item() if hasattr(resp["meta"], "item") else resp["meta"])
    # POPULATED-class expectation, measured, per layout. (The pre-2026-08-17 print
    # said "expect 36" — the count of POSSIBLE v1 channels, never the populated
    # count, which has been 24 since the response was first built. SCOPE §1.5 R6.)
    expect_cls = {"flat900": 8, "v1_300": 24}[args.layout]
    print("=== ORCA binned response built ===")
    print(f"  out: {args.out}")
    print(f"  layout: {meta['layout']}  schema: {meta['schema_version']}")
    print(f"  n_events_nu (input rows): {meta['n_events_nu']}")
    print(f"  nnz_response (nonzero cells): {meta['nnz_response']}")
    print(f"  cell-count conservation (nnz == n_events_nu): "
          f"{'PASS' if meta['nnz_response'] == meta['n_events_nu'] else 'FAIL'}")
    print(f"  n_classes (populated): {meta['n_classes']} (expect {expect_cls}) "
          f"{'PASS' if meta['n_classes'] == expect_cls else 'FAIL'}")
    print(f"  n_bins: {meta['n_bins']}")
    print(f"  grids true {meta['grids']['true']} reco {meta['grids']['reco']}")
    print(f"  total weight: {meta['total_weight']:.10e}")
    # weight conservation vs raw parquet
    raw = pd.read_parquet(args.parquet)
    raw_w = float(raw.loc[raw["MC_type"] == 1, "weight"].sum())
    dw = abs(meta["total_weight"] - raw_w)
    print(f"  raw parquet nu weight: {raw_w:.10e}  |Δ|={dw:.3e}  "
          f"{'PASS' if dw <= 1e-3 * max(raw_w, 1) else 'FAIL'}")
