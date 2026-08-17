"""IC DeepCore binned-response support — the 200-bin reco grid constants, the
parquet/kernel helpers, and the production reference model the IC binned engine
is certified against.

Extracted VERBATIM from the dev-tree scan driver `ic_divergence_scan.py` (the
AtmNuDataFit multi-experiment-systematics campaign at commit 8f6f098; exact path
in `PROVENANCE_icorca_binned.md`). The exact IC mirror of `orca_binned_support`,
and the same split: the reference model ships, the drivers do not.

`_hs_params_from_theta`, `_hs_correction_factor` and `_corrected_expectation`
are that reference model. Gate G-IC-4 compares `ICBinnedEngine` against
`_corrected_expectation` at the probe points in `POINTS`, so the gate cannot run
without them. They take the live experiment as an ARGUMENT, so this module still
imports no pynu. (The port plan had no slot for this module at all; the addition
is recorded in `PROVENANCE_icorca_binned.md`.)

Note `_hs_correction_factor` calls `exp.interpolate_hs(dm31)` per invocation —
the Delta-m^2-dependent hypersurface refresh that `README_ic_binned.md` flags as
the one sharp usage edge of the IC arm.

The scan-driver parts of `ic_divergence_scan.py` (`selftest`, `run_fasrc`,
`main`, `add_pynu_root`, and the draw / dial-class machinery) do NOT ship; their
upstream equivalents belong under `analysis/IC-binned-datafit/`.

Pure pandas + numpy at module level: no pynu import, no nuSQuIDS.
"""
import os

import numpy as np
import pandas as pd

N_ERECO, N_CZRECO, N_PID = 10, 10, 2
N_BINS = N_PID * N_ERECO * N_CZRECO                 # 200

# osc test points (must match ic_build_tensors.py DEFAULT_POINTS order)
POINTS = [(2.511e-3, 0.572), (2.3e-3, 0.45), (2.7e-3, 0.45),
          (2.3e-3, 0.65), (2.7e-3, 0.65)]


def _load_reco_edges(data_dir):
    e_r = np.load(os.path.join(data_dir, "_E_reco_bins.npy"))
    cz_r = np.load(os.path.join(data_dir, "_cosT_reco_bins.npy"))
    return e_r, cz_r


def _digitize_clamp(x, edges):
    n = len(edges) - 1
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n - 1).astype(np.int64)


def _flat200(pid, ire0, irz0):
    return (pid * N_ERECO + ire0) * N_CZRECO + irz0


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


def nu_index(mc_parquet, data_dir, response_npz):
    """Per-nu-event arrays in the ICDeepCore loader row order (MC_type==1,
    0<=ETrue<1e5): the 200-bin reco index, the ladder true-cell (ie,iz) for THIS
    response grid, and the osc (ntype, flavor) index into phi. Uses the SAME
    ladder edges the response was built on (read from the response npz)."""
    e_r, cz_r = _load_reco_edges(data_dir)
    resp = np.load(response_npz, allow_pickle=True)
    e_edges = np.asarray(resp["e_true_edges"], float)
    cz_edges = np.asarray(resp["cz_true_edges"], float)

    df = pd.read_parquet(mc_parquet)
    nu = df[df["MC_type"] == 1]
    et = nu["true_energy"].values
    cond = (et >= 0) & (et < 1e5)
    nu = nu[cond]

    # reco 200-bin index (matches loader histogram2d, verified byte-equal)
    ire = _digitize_clamp(nu["reco_energy"].values, e_r)
    irz = _digitize_clamp(np.cos(nu["reco_zenith"].values), cz_r)
    pid = nu["pid"].values.astype(np.int64)
    bin_idx = _flat200(pid, ire, irz)

    # ladder true cell index (matches ic_binned_builder.py digitize_clamp)
    ie = _digitize_clamp(nu["true_energy"].values, e_edges)
    iz = _digitize_clamp(np.cos(nu["true_zenith"].values), cz_edges)

    # osc (ntype, flavor) index into phi[point, ntype, flavor, ie, iz]
    pdg = nu["pdg"].values.astype(np.int64)
    ntype = (pdg < 0).astype(np.int64)                 # 0 nu / 1 nubar
    flavor = (np.abs(pdg) // 2 - 6).astype(np.int64)   # 12->0 e, 14->1 mu, 16->2 tau
    return bin_idx, ie, iz, ntype, flavor


def poisson_chi2(obs, n_mod):
    if np.any(n_mod <= 0):
        return 9e9
    lt = np.log(np.divide(obs, n_mod, out=np.ones_like(obs), where=n_mod > 0))
    lt[obs == 0] = 0
    return float(2 * np.sum(n_mod - obs + obs * lt))


# ---------------------------------------------------------------------------
# Reference model — these three take a LIVE experiment (nuSQuIDS at the call
# site, not here). G-IC-4 compares the engine against them.
# ---------------------------------------------------------------------------

def _hs_params_from_theta(theta, names, hs_names):
    return {h: theta[names.index(h)] for h in hs_names if h in names}


def _hs_correction_factor(exp, dm31, hs_params):
    """The per-category HS correction FACTOR (200,) each:
    intercept + Σ slope·(param − nominal), interpolated at dm31. This is exactly
    the multiplier apply_hs_correction (ICDeepCore.py:646-653) applies to each
    category histogram. Computed once per ARM so the two can be byte-compared."""
    hs_slopes = exp.interpolate_hs(dm31)
    corr = {}
    for cat in exp.HS_CATEGORIES:
        s = hs_slopes[cat]
        c = s["intercept"].copy()
        for sname in exp.HS_SLOPE_NAMES:
            c += s[sname] * (hs_params[sname] - exp.HS_NOMINALS[sname])
        corr[cat] = c
    return corr


def _corrected_expectation(exp, per_event_weight, corr, mu200):
    """Bin the per-event weight by HS flavor category, multiply each by its
    correction factor, sum, add muon background. bin_by_flavor multiplies by
    BaseWeight internally (ICDeepCore.py:615), so per_event_weight must be the
    PhysicsWeight·NuisanceWeight product WITHOUT BaseWeight (matching how
    apply_hs_correction is fed exp.ExpectedWeight)."""
    fh = exp.bin_by_flavor(per_event_weight)          # {cat: (200,)}
    tot = np.zeros(N_BINS)
    for cat in exp.HS_CATEGORIES:
        tot += fh[cat] * corr[cat]
    return tot + mu200
