"""IC DeepCore binned response (COO) + the 200-bin data / muon vectors.

IC_MC.parquet is a hybrid MC: event-level true side (binned here onto a 160 x 320
geomspace x linear grid, cell centres = geometric / arithmetic means), reco side
histogrammed on the native edges with pid folded into the bin index
b = (pid*n_ereco + ie_reco)*n_czreco + iz_reco. 47 classes (pdg, current, |NEUT
mode|). Muon block (MC_type == -1) excluded from R and stored separately.
"""
import json
import os

import numpy as np
import pandas as pd

N_ETRUE, N_CZTRUE = 160, 320
N_ERECO, N_CZRECO, N_PID = 10, 10, 2
N_BINS = N_PID * N_ERECO * N_CZRECO                 # 200


def _load_reco_edges(data_dir):
    e_r = np.load(os.path.join(data_dir, "_E_reco_bins.npy"))
    cz_r = np.load(os.path.join(data_dir, "_cosT_reco_bins.npy"))
    return e_r, cz_r


def _true_grid(e_true, nE, nZ):
    """geomspace ETrue between the MC min/max x linspace CosZTrue in [-1, 1]."""
    e_edges = np.geomspace(e_true.min(), e_true.max(), nE + 1)
    cz_edges = np.linspace(-1.0, 1.0, nZ + 1)
    e_centers = np.sqrt(e_edges[:-1] * e_edges[1:])          # geometric-mean centre
    cz_centers = 0.5 * (cz_edges[:-1] + cz_edges[1:])        # arithmetic centre
    return e_edges, cz_edges, e_centers, cz_centers


def _digitize_clamp(x, edges):
    """0-based bin index on `edges` (n+1), clamped to [0, n-1] (np.histogram
    semantics for interior points)."""
    n = len(edges) - 1
    idx = np.searchsorted(edges, x, side="right") - 1
    return np.clip(idx, 0, n - 1).astype(np.int64)


def _flat200(pid, ire0, irz0):
    return (pid * N_ERECO + ire0) * N_CZRECO + irz0


def build_ic_response(parquet_path, data_dir, out_path):
    nE, nZ = N_ETRUE, N_CZTRUE
    df = pd.read_parquet(parquet_path)

    # ---- nu block + the quality cut of ICDeepCore.MCVariables ----
    df_nu = df[df["MC_type"] == 1].copy()
    et = df_nu["true_energy"].values
    cond = (et >= 0) & (et < 1e5)
    df_nu = df_nu[cond]
    N = len(df_nu)

    e_r, cz_r = _load_reco_edges(data_dir)
    n_ereco = len(e_r) - 1          # 10
    n_czreco = len(cz_r) - 1        # 10
    n_bins = n_ereco * n_czreco * 2  # 200 (pid in {0,1})

    E = df_nu["true_energy"].values
    czt = np.cos(df_nu["true_zenith"].values)
    e_edges, cz_edges, e_c, z_c = _true_grid(E, nE, nZ)
    ie = _digitize_clamp(E, e_edges)
    iz = _digitize_clamp(czt, cz_edges)

    er = df_nu["reco_energy"].values
    czr = np.cos(df_nu["reco_zenith"].values)
    ire = _digitize_clamp(er, e_r)
    irz = _digitize_clamp(czr, cz_r)
    pid = df_nu["pid"].values.astype(np.int64)
    recobin = (pid * n_ereco + ire) * n_czreco + irz     # flat 200-bin, pid-slow

    # ---- class index (pdg, current, |NEUT mode|) ----
    pdg = df_nu["pdg"].values.astype(np.int64)
    cur = df_nu["current_type"].values.astype(np.int64)
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
    classes, class_inv = np.unique(sig, axis=0, return_inverse=True)
    class_inv = np.asarray(class_inv).ravel()
    n_cls = classes.shape[0]

    w = df_nu["weight"].values.astype(float)
    w2 = df_nu["weight_variance"].values.astype(float)   # == weight^2 (verified)

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
        mire = _digitize_clamp(df_mu["reco_energy"].values, e_r)
        mirz = _digitize_clamp(np.cos(df_mu["reco_zenith"].values), cz_r)
        mpid = df_mu["pid"].values.astype(np.int64)
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

    nnz = Rv.size
    total_w = float(Rv.sum())
    raw_w = float(w.sum())
    rel = abs(total_w - raw_w) / max(raw_w, 1.0)
    populated_classes = int(np.unique(Rk).size)
    min_entry = float(Rv.min()) if Rv.size else 0.0
    per_class_w = {"_".join(str(int(v)) for v in classes[k]):
                   float(Rv[Rk == k].sum()) for k in range(n_cls)}

    resp = dict(
        schema_version=np.array("ic_binned_response_v2_mode"),
        grid_label=np.array("L3"),
        classes=classes,
        class_axis=np.array(["pdg", "current", "absmode"]),
        R_k=Rk, R_e=Re, R_z=Rz, R_b=Rb, R_v=Rv,
        S2_k=Sk, S2_e=Se, S2_z=Sz, S2_b=Sb, S2_v=Sv,
        n_etrue=np.int64(nE), n_cztrue=np.int64(nZ),
        n_ereco=np.int64(n_ereco), n_czreco=np.int64(n_czreco),
        n_pid=np.int64(2),
        n_bins=np.int64(n_bins), n_era=np.int64(1),
        e_true_edges=e_edges, cz_true_edges=cz_edges,
        e_reco_edges=e_r, cz_reco_edges=cz_r,
        e_true_centers=e_c, cz_true_centers=z_c,
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
        }),
    )
    np.savez_compressed(out_path, **resp)
    print(f"[ic-response] {N} nu events, "
          f"{n_cls} classes, nnz {nnz}, conservation rel {rel:.2e}, "
          f"muon total {mu_total:.4f} ({len(df_mu)} rows)")
    return resp


def observed_200(data_parquet, data_dir):
    """Bin real IC data onto the 200 reco bins (histogram, no bin column)."""
    e_r, cz_r = _load_reco_edges(data_dir)
    d = pd.read_parquet(data_parquet)
    ire = _digitize_clamp(d["reco_energy"].values, e_r)
    irz = _digitize_clamp(np.cos(d["reco_zenith"].values), cz_r)
    pid = d["pid"].values.astype(np.int64)
    w = d["weight"].values.astype(float) if "weight" in d else np.ones(len(d))
    return np.bincount(_flat200(pid, ire, irz), weights=w, minlength=N_BINS)


def muon_200(mc_parquet, data_dir):
    """Static muon background (MC_type==-1) on the 200 reco bins, RAW weight."""
    e_r, cz_r = _load_reco_edges(data_dir)
    df = pd.read_parquet(mc_parquet)
    mu = df[df["MC_type"] == -1]
    ire = _digitize_clamp(mu["reco_energy"].values, e_r)
    irz = _digitize_clamp(np.cos(mu["reco_zenith"].values), cz_r)
    pid = mu["pid"].values.astype(np.int64)
    return np.bincount(_flat200(pid, ire, irz),
                       weights=mu["weight"].values.astype(float), minlength=N_BINS)
