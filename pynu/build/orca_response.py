"""ORCA binned response (COO, flat900 layout) + the 900-bin data / muon vectors.

The ORCA MC parquet is intrinsically a binned response: each nu row is one
populated cell (true E bin, true cz bin, reco E bin, reco cz bin, pdg, pid,
current) with an explicit weight, so this is a reshape (the baked ``*_bin_num``
columns are authoritative). flat900: classes = (pdg, current), 8 populated;
R_b = (pid*15 + ie)*20 + iz, 900 bins.
"""
import json
import os

import numpy as np
import pandas as pd

SCHEMA_VERSION_FLAT900 = "orca_binned_response_v2_flat900"
N_ERECO, N_CZRECO, N_PID = 15, 20, 3
N_BINS = N_PID * N_ERECO * N_CZRECO                 # 900


def _flat900(pid, ire0, irz0):
    return (pid * N_ERECO + ire0) * N_CZRECO + irz0


def _load_true_reco_edges(data_dir):
    e_t = np.load(os.path.join(data_dir, "_E_true_bins.npy"))
    cz_t = np.load(os.path.join(data_dir, "_cosT_true_bins.npy"))
    e_r = np.load(os.path.join(data_dir, "_E_reco_bins.npy"))
    cz_r = np.load(os.path.join(data_dir, "_cosT_reco_bins.npy"))
    return e_t, cz_t, e_r, cz_r


def _true_cell_centers(e_true_edges, cz_true_edges, df_nu):
    """The quantized true value carried by each true bin, read from the MC."""
    e_c = np.full(len(e_true_edges) - 1, np.nan)
    z_c = np.full(len(cz_true_edges) - 1, np.nan)
    ie = df_nu["true_energy_bin_num"].values.astype(int) - 1
    iz = df_nu["true_cos_zenith_bin_num"].values.astype(int) - 1
    E = df_nu["true_energy"].values
    cz = np.cos(df_nu["true_zenith"].values)
    for b in np.unique(ie):
        e_c[b] = E[ie == b][0]
    for b in np.unique(iz):
        z_c[b] = cz[iz == b][0]
    return e_c, z_c


def build_orca_response(parquet_path, data_dir, out_path):
    layout = "flat900"
    df = pd.read_parquet(parquet_path)
    df_nu = df[df["MC_type"] == 1].copy()          # neutrinos only (drop 900 muons)
    N = len(df_nu)

    e_t, cz_t, e_r, cz_r = _load_true_reco_edges(data_dir)
    n_etrue = len(e_t) - 1          # 40
    n_cztrue = len(cz_t) - 1        # 80
    n_ereco = len(e_r) - 1          # 15
    n_czreco = len(cz_r) - 1        # 20

    ie = df_nu["true_energy_bin_num"].values.astype(np.int64) - 1
    iz = df_nu["true_cos_zenith_bin_num"].values.astype(np.int64) - 1
    ire = df_nu["reco_energy_bin_num"].values.astype(np.int64) - 1
    irz = df_nu["reco_cos_zenith_bin_num"].values.astype(np.int64) - 1
    pdg = df_nu["pdg"].values.astype(np.int64)
    pid = df_nu["pid"].values.astype(np.int64)
    cur = df_nu["current_type"].values.astype(np.int64)

    assert (n_ereco, n_czreco) == (N_ERECO, N_CZRECO), (n_ereco, n_czreco)
    n_bins = N_PID * n_ereco * n_czreco           # 900
    recobin = _flat900(pid, ire, irz)
    sig = np.column_stack([pdg, cur])
    class_axis = ["pdg", "current"]
    schema_version = SCHEMA_VERSION_FLAT900

    classes, class_inv = np.unique(sig, axis=0, return_inverse=True)
    n_cls = classes.shape[0]
    w = df_nu["weight"].values.astype(float)
    w2 = w * w

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
    e_c, z_c = _true_cell_centers(e_t, cz_t, df_nu)

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
        classes=classes,
        class_axis=np.array(class_axis),
        R_k=Rk, R_e=Re, R_z=Rz, R_b=Rb, R_v=Rv,
        S2_k=Sk, S2_e=Se, S2_z=Sz, S2_b=Sb, S2_v=Sv,
        n_etrue=np.int64(n_etrue), n_cztrue=np.int64(n_cztrue),
        n_ereco=np.int64(n_ereco), n_czreco=np.int64(n_czreco),
        n_bins=np.int64(n_bins), n_era=np.int64(1),
        e_true_edges=e_t, cz_true_edges=cz_t,
        e_reco_edges=e_r, cz_reco_edges=cz_r,
        e_true_centers=e_c, cz_true_centers=z_c,
        meta=json.dumps({
            "source": "ORCA_MC_dataverse_with_muons.parquet",
            "layout": layout, "schema_version": schema_version,
            "n_events_nu": int(N), "n_classes": int(n_cls),
            "nnz_response": int(nnz), "total_weight": total_w, "n_bins": int(n_bins),
            "per_current_weight": per_cur_w,
            "per_pdgcurrent_weight": per_pdgcur_w,
            "grids": {"true": [n_etrue, n_cztrue], "reco": [n_ereco, n_czreco]},
        }),
    )
    np.savez_compressed(out_path, **resp)
    print(f"[orca-response] flat900: {N} nu rows, nnz {nnz} "
          f"({'PASS' if nnz == N else 'FAIL'} cell conservation), {n_cls} classes "
          f"(expect 8), total weight {total_w:.10e}")
    return resp


def observed_900(data_parquet):
    d = pd.read_parquet(data_parquet)
    b = _flat900(d["pid"].values.astype(np.int64),
                 d["reco_energy_bin_num"].values.astype(np.int64) - 1,
                 d["reco_cos_zenith_bin_num"].values.astype(np.int64) - 1)
    return np.bincount(b, weights=d["weight"].values.astype(float), minlength=N_BINS)


def muon_900(mc_parquet):
    """Static muon background on the 900-bin grid: raw weight, no NORM."""
    df = pd.read_parquet(mc_parquet)
    mu = df[df["MC_type"] == -1]
    b = _flat900(mu["pid"].values.astype(np.int64),
                 mu["reco_energy_bin_num"].values.astype(np.int64) - 1,
                 mu["reco_cos_zenith_bin_num"].values.astype(np.int64) - 1)
    return np.bincount(b, weights=mu["weight"].values.astype(float), minlength=N_BINS)
