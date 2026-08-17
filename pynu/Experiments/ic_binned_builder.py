"""IC DeepCore binned-response builder — Track R·IC (2026-07-11).

Adapted from `orca_binned_builder.py` (the certified ORCA pattern). UNLIKE ORCA,
`IC_MC.parquet` is a HYBRID MC (verified by the local schema read in
SPEC_trackR_ic_binned_matrices.md §1):
  * TRUE side is EVENT-LEVEL — 396843/396843 unique (ETrue,CosZTrue) pairs. There
    are NO baked `*_bin_num` columns (ORCA had them). So the true side must be
    BINNED onto a resolution ladder → a genuine cell-centering (Jensen) residue
    exists. This is the whole reason for the ladder + the divergence scan.
  * RECO side IS binned but by HISTOGRAM at fit time (no reco bin column). The
    builder reproduces the loader's `histogram2d` on the native
    `_E_reco_bins.npy`/`_cosT_reco_bins.npy` edges (ICDeepCore.py:261-275).

Mirrors the npz-key conventions of `orca_binned_builder.py` (which mirror
`pynu/binned/builder.py:build_response`) so the tensor build / divergence scan
load with minimal adaptation.

IC deviations from the ORCA response (SPEC §3):
  * true side BINNED not reshaped (cell-centering residue exists);
  * 12 classes = (pdg,current) — IC NC carries all 6 pdg, not flavor-blind;
  * reco = 200 bins histogrammed on native edges, pid folded into the reco-bin
    axis (b = (pid*n_ereco + ie_reco)*n_czreco + iz_reco);
  * no era axis (n_era=1); no energy-scale Rp/Rm.
  * muon block (MC_type==-1, 200 rows, Σw=512.166) EXCLUDED from the nu response
    and saved separately (reco histogram + total) for fit-time reinjection.

Read-only on all inputs; writes only its output npz. No pynu import (pure parquet
+ numpy). nuSQuIDS is only for the SEPARATE osc-tensor build (ic_build_tensors.py).
"""
import json
import hashlib
import os
import numpy as np
import pandas as pd

SCHEMA_VERSION = "ic_binned_response_v1"

# Resolution ladder: geomspace ETrue in [data-min,data-max] x linear CosZTrue in
# [-1,1]. (nE, nZ) per level. L1 = ORCA-matched baseline.
LADDER = {
    "L0": (20, 40),
    "L1": (40, 80),
    "L2": (80, 160),
    "L3": (160, 320),
}


def _load_reco_edges(data_dir):
    """Native IC reco bin edges (n+1 each). Returns (e_reco[11], cz_reco[11]).
    These are the loader's edges (ICDeepCore.Binning:359-374)."""
    e_r = np.load(os.path.join(data_dir, "_E_reco_bins.npy"))
    cz_r = np.load(os.path.join(data_dir, "_cosT_reco_bins.npy"))
    return e_r, cz_r


def _true_ladder_grid(e_true, cz_true, nE, nZ, snap_e=()):
    """Build the ladder true grid EDGES + CELL CENTRES.

    ETrue: geomspace between the data min/max (so the top edge is the data max
    9292 GeV, not the loader cut 1e5 — no empty high-E cells). Geomspace centre
    = sqrt(edge_lo*edge_hi) (the geometric mean, the natural centre for a
    log-uniform bin — matches the SK/pynu true-grid convention).
    CosZTrue: linear in [-1,1]; centre = arithmetic mean.

    snap_e: band-threshold energies (GeV) to force onto the E-edge set (B3 fix,
    DESIGN §5.3). A threshold interior to a cell makes the cell centre assign ALL
    its events to one side of the band, an O(1) per-event dial error that grid
    refinement does not remove. Thresholds are unioned in only when STRICTLY
    inside (e_min, e_max) — IC's min true E is 1.02344806 GeV, so 1.0 GeV would
    otherwise prepend an empty leading cell (design ADDENDUM item 2) — and are
    dropped when already within 1e-9 relative of a geomspace edge, so the snap
    never manufactures a degenerate sliver cell.

    Returns (e_edges, cz_edges, e_centers, cz_centers, snap_info).
    """
    e_edges = np.geomspace(e_true.min(), e_true.max(), nE + 1)
    snap_info = {"requested": [float(t) for t in snap_e],
                 "inserted": [], "outside": [], "already_edge": []}
    for t in snap_e:
        t = float(t)
        if not (e_edges[0] < t < e_edges[-1]):
            snap_info["outside"].append(t)
            continue
        j = int(np.argmin(np.abs(e_edges - t)))
        if abs(e_edges[j] - t) <= 1e-9 * max(abs(t), 1.0):
            snap_info["already_edge"].append(t)
            continue
        snap_info["inserted"].append(t)
    if snap_info["inserted"]:
        e_edges = np.unique(np.concatenate([e_edges,
                                            np.array(snap_info["inserted"], float)]))
    cz_edges = np.linspace(-1.0, 1.0, nZ + 1)
    e_centers = np.sqrt(e_edges[:-1] * e_edges[1:])          # geometric-mean centre
    cz_centers = 0.5 * (cz_edges[:-1] + cz_edges[1:])        # arithmetic centre
    return e_edges, cz_edges, e_centers, cz_centers, snap_info


def _digitize_clamp(x, edges):
    """0-based bin index of x on `edges` (n+1), clamped to [0, n-1]. Matches
    np.histogram semantics (right-open bins, last bin closed) for interior points;
    the clamp keeps boundary/roundoff events in the edge bins (histogram2d would
    also place min/max on the closed edges). Verified: 0 IC events out of range."""
    n = len(edges) - 1
    idx = np.searchsorted(edges, x, side="right") - 1
    return np.clip(idx, 0, n - 1).astype(np.int64)


def build_ic_response(parquet_path, data_dir, grid_label, out_path=None,
                      dial_manifest=None, mode_axis=False, snap_e=()):
    """Bin the IC parquet into a pynu.binned COO response npz at ladder `grid_label`.

    Args:
      parquet_path: IC_MC.parquet.
      data_dir: dir with _E_reco_bins.npy / _cosT_reco_bins.npy.
      grid_label: one of LADDER keys ("L0".."L3").
      out_path: optional npz path.
      dial_manifest: optional dial-name iterable (hashed for engine load-check,
        mirrors the ORCA/SK builder).
      mode_axis: add |NEUT mode| to the class signature (schema v2).
      snap_e: band thresholds (GeV) to force onto the true-E edge set; empty
        (the default) reproduces the shipped ladder byte-for-byte. Snapping
        raises nE above LADDER[grid_label][0] by the number of edges inserted.

    Returns the response dict (the npz payload).

    Response layout (COO, keyed class -> ie_true -> iz_true -> recobin):
      R_k int32  class index into `classes` (pdg,current)
      R_e int32  true-E ladder bin (0..nE-1)
      R_z int32  true-cz ladder bin (0..nZ-1)
      R_b int32  flat reco bin = (pid*n_ereco + ie_reco)*n_czreco + iz_reco, 0-based
      R_v float  summed RAW `weight` in the cell (NORM applied at scan time)
      S2_* same COO for weight^2 (== weight_variance, MC variance).
    Muon block: mu_bkg(200,), mu_total, mu_S2(200,) — EXCLUDED from R.
    """
    nE, nZ = LADDER[grid_label]
    df = pd.read_parquet(parquet_path)

    # ---- nu block + loader quality cut (ICDeepCore.py:138-139) ----
    df_nu = df[df["MC_type"] == 1].copy()
    et = df_nu["true_energy"].values
    cond = (et >= 0) & (et < 1e5)
    df_nu = df_nu[cond]
    N = len(df_nu)

    e_r, cz_r = _load_reco_edges(data_dir)
    n_ereco = len(e_r) - 1          # 10
    n_czreco = len(cz_r) - 1        # 10
    n_bins = n_ereco * n_czreco * 2  # 200 (pid in {0,1})

    # ---- true ladder grid (edges + cell centres the tensor consumes) ----
    E = df_nu["true_energy"].values
    czt = np.cos(df_nu["true_zenith"].values)
    e_edges, cz_edges, e_c, z_c, snap_info = _true_ladder_grid(E, czt, nE, nZ,
                                                               snap_e=snap_e)
    nE = len(e_edges) - 1          # snapped edges raise nE above LADDER's value
    ie = _digitize_clamp(E, e_edges)
    iz = _digitize_clamp(czt, cz_edges)

    # ---- reco bins: histogram on native edges (reproduce the loader) ----
    er = df_nu["reco_energy"].values
    czr = np.cos(df_nu["reco_zenith"].values)
    ire = _digitize_clamp(er, e_r)
    irz = _digitize_clamp(czr, cz_r)
    pid = df_nu["pid"].values.astype(np.int64)
    recobin = (pid * n_ereco + ire) * n_czreco + irz     # flat 200-bin, pid-slow

    # ---- class index (pdg, current[, |mode|]) -> classes table ----
    # mode_axis=True adds |NEUT mode| to the class signature so the Mode-keyed
    # xsec dials (DIS/CCQE*/CC1Pi*, WaterXSection masks — all functions of
    # (|Mode|, pdg, CC)) become response-representable. Mode replicates
    # ICDeepCore.MCVariables:159-165 + _NEUTMode:196-207 exactly (unsigned:
    # the sign is sign(pdg), already a class column).
    pdg = df_nu["pdg"].values.astype(np.int64)
    cur = df_nu["current_type"].values.astype(np.int64)
    if mode_axis:
        neut_map = {0: 31, 1: 1, 2: 11, 3: 26, 4: 16}
        if "interaction_type" in df_nu:
            itype = df_nu["interaction_type"].values
        elif "type" in df_nu:
            itype = df_nu["type"].values
        else:
            itype = None
        if itype is None:
            absmode = np.zeros(len(df_nu), dtype=np.int64)
        else:
            absmode = np.array([neut_map.get(int(v), 0) for v in itype],
                               dtype=np.int64)
        sig = np.column_stack([pdg, cur, absmode])
    else:
        sig = np.column_stack([pdg, cur])
    classes, class_inv = np.unique(sig, axis=0, return_inverse=True)
    class_inv = np.asarray(class_inv).ravel()
    n_cls = classes.shape[0]

    w = df_nu["weight"].values.astype(float)
    w2 = df_nu["weight_variance"].values.astype(float)   # == weight^2 (verified)

    # ---- COO reduce over (class, ie, iz, recobin) ----
    def coo(weights):
        key = ((class_inv.astype(np.int64) * nE + ie) * nZ + iz) * n_bins + recobin
        uniq, inv = np.unique(key, return_inverse=True)
        val = np.bincount(inv, weights=weights, minlength=uniq.size)
        b = uniq % n_bins
        rem = uniq // n_bins
        cz_i = rem % nZ
        rem = rem // nZ
        ce = rem % nE
        k = rem // nE
        return (k.astype(np.int32), ce.astype(np.int32), cz_i.astype(np.int32),
                b.astype(np.int32), val)

    Rk, Re, Rz, Rb, Rv = coo(w)
    Sk, Se, Sz, Sb, Sv = coo(w2)

    # ---- muon block (MC_type==-1): reco histogram + total, EXCLUDED from R ----
    df_mu = df[df["MC_type"] == -1]
    mu_total = float(df_mu["weight"].sum())
    if len(df_mu) > 0:
        mer = df_mu["reco_energy"].values
        mczr = np.cos(df_mu["reco_zenith"].values)
        mpid = df_mu["pid"].values.astype(np.int64)
        mire = _digitize_clamp(mer, e_r)
        mirz = _digitize_clamp(mczr, cz_r)
        mb = (mpid * n_ereco + mire) * n_czreco + mirz
        mu_bkg = np.bincount(mb, weights=df_mu["weight"].values.astype(float),
                             minlength=n_bins)
        mu_S2 = np.bincount(mb, weights=df_mu["weight_variance"].values.astype(float)
                            if "weight_variance" in df_mu
                            else df_mu["weight"].values.astype(float) ** 2,
                            minlength=n_bins)
    else:
        mu_bkg = np.zeros(n_bins)
        mu_S2 = np.zeros(n_bins)

    # ---- self-checks (sanity floors, measured here) ----
    nnz = Rv.size
    total_w = float(Rv.sum())
    raw_w = float(w.sum())
    rel = abs(total_w - raw_w) / max(raw_w, 1.0)
    populated_classes = int(np.unique(Rk).size)
    min_entry = float(Rv.min()) if Rv.size else 0.0
    per_class_w = {"_".join(str(int(v)) for v in classes[k]):
                   float(Rv[Rk == k].sum()) for k in range(n_cls)}

    meta_extra = {}
    if snap_info["requested"]:
        # only emitted when snapping is requested, so an unsnapped rebuild stays
        # metadata-identical to the shipped ic_response_L*.npz
        meta_extra["snap_e_edges"] = snap_info

    resp = dict(
        schema_version=np.array("ic_binned_response_v2_mode" if mode_axis
                                else SCHEMA_VERSION),
        grid_label=np.array(grid_label),
        classes=classes,               # (n_cls, 2 or 3): pdg, current[, absmode]
        class_axis=np.array(["pdg", "current", "absmode"] if mode_axis
                            else ["pdg", "current"]),
        R_k=Rk, R_e=Re, R_z=Rz, R_b=Rb, R_v=Rv,
        S2_k=Sk, S2_e=Se, S2_z=Sz, S2_b=Sb, S2_v=Sv,
        n_etrue=np.int64(nE), n_cztrue=np.int64(nZ),
        n_ereco=np.int64(n_ereco), n_czreco=np.int64(n_czreco),
        n_pid=np.int64(2),
        n_bins=np.int64(n_bins), n_era=np.int64(1),
        e_true_edges=e_edges, cz_true_edges=cz_edges,
        e_reco_edges=e_r, cz_reco_edges=cz_r,
        e_true_centers=e_c, cz_true_centers=z_c,     # ladder cell centres (tensor)
        mu_bkg=mu_bkg, mu_S2=mu_S2, mu_total=np.float64(mu_total),
        meta=json.dumps({
            "source": "IC_MC.parquet",
            "n_events_nu": int(N), "n_classes": int(n_cls),
            "nnz_response": int(nnz), "total_weight": total_w,
            "raw_nu_weight": raw_w, "conservation_rel": rel,
            "populated_classes": populated_classes,
            "min_response_entry": min_entry,
            "muon_total_weight": mu_total, "muon_n_rows": int(len(df_mu)),
            "per_class_weight": per_class_w,
            "grid": {"true": [nE, nZ], "reco": [n_ereco, n_czreco, 2]},
            "deviations_from_orca": ["true_side_binned_jensen_residue",
                                     "12_classes_pdg_current_nc_all_flavors",
                                     "reco_histogrammed_no_bin_column",
                                     "no_era", "no_escale_RpRm"],
            **meta_extra,
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


def _gate_print(resp):
    """Print the named gate lines for one grid; return (all_pass, meta)."""
    meta = json.loads(resp["meta"].item() if hasattr(resp["meta"], "item")
                      else resp["meta"])
    k = meta_grid = str(resp["grid_label"])
    Rv = resp["R_v"]
    # conservation
    rel = meta["conservation_rel"]
    cons_ok = rel <= 1e-9
    print(f"CONSERVATION {k}: {'PASS' if cons_ok else 'FAIL'} rel={rel:.3e}")
    # channels
    npop = meta["populated_classes"]
    chan_ok = npop == meta["n_classes"] and npop > 0
    print(f"CHANNELS {k}: {'PASS' if chan_ok else 'FAIL'} ({npop} populated)")
    # nonneg
    nn_ok = bool((Rv >= 0).all()) and bool((resp["S2_v"] >= 0).all())
    print(f"NONNEG {k}: {'PASS' if nn_ok else 'FAIL'}")
    return (cons_ok and chan_ok and nn_ok), meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the IC binned response npz (ladder).")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--data-dir", required=True, help="dir with _*_reco_bins.npy")
    ap.add_argument("--out-prefix", required=True,
                    help="output prefix; writes <prefix>_L0.npz .. _L3.npz")
    ap.add_argument("--grids", nargs="+", default=list(LADDER.keys()),
                    help="ladder levels to build (default all four)")
    ap.add_argument("--mode-axis", action="store_true",
                    help="add |NEUT mode| to the class axis (schema v2) so "
                         "Mode-keyed xsec dials are response-representable")
    ap.add_argument("--snap-e-edges", nargs="+", type=float, default=(),
                    metavar="GEV",
                    help="band thresholds forced onto the true-E edge set (B3 "
                         "fix); thresholds outside (e_min,e_max) are skipped. "
                         "Omit for the shipped ladder.")
    args = ap.parse_args()

    print("=== IC binned response builder (Track R·IC) ===")
    all_pass = True
    for gl in args.grids:
        out = f"{args.out_prefix}_{gl}.npz"
        resp = build_ic_response(args.parquet, args.data_dir, gl, out_path=out,
                                 mode_axis=args.mode_axis,
                                 snap_e=tuple(args.snap_e_edges))
        meta = json.loads(resp["meta"].item() if hasattr(resp["meta"], "item")
                          else resp["meta"])
        print(f"\n--- {gl}: true {meta['grid']['true']} reco {meta['grid']['reco']} "
              f"-> {out}")
        print(f"    n_events_nu={meta['n_events_nu']} nnz={meta['nnz_response']} "
              f"Σw={meta['total_weight']:.6e} raw={meta['raw_nu_weight']:.6e} "
              f"muon_total={meta['muon_total_weight']:.4f} ({meta['muon_n_rows']} rows)")
        if "snap_e_edges" in meta:
            s = meta["snap_e_edges"]
            print(f"    snap_e_edges: inserted={s['inserted']} "
                  f"outside={s['outside']} already_edge={s['already_edge']}")
        g_ok, _ = _gate_print(resp)
        all_pass &= g_ok
    print(f"\nIC BUILDER GATES: {'ALL PASS' if all_pass else 'FAIL'}")
    import sys
    sys.exit(0 if all_pass else 1)
