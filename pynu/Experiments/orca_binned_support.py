"""ORCA binned-response support — the flat900 grid constants, the parquet/kernel
helpers the ORCA binned engine and builder depend on, and the production
reference model the engine is certified against.

Extracted VERBATIM from the dev-tree scan driver `orca_exact_scan.py` (the
AtmNuDataFit combined-fit campaign at commit 8f6f098; exact path in
`PROVENANCE_icorca_binned.md`), which is the certifying reference for the ORCA
binned arm: gate G-ORCA-1 (55/55)
compares `ORCABinnedEngine.expectation` against that driver's event path, and the
flat900 bin index `_flat900` is what `observed_900`, `muon_900` and
`nu_cell_index` all key on. The function bodies below are byte-identical to their
source; nothing here is a re-typing.

`binned_expectation` is the PRODUCTION REFERENCE MODEL, not a driver
convenience: it is the expectation that G-ORCA-1 and G-ORCA-2 compare
`ORCABinnedEngine` against, so the gates cannot run without it. (The port plan's
section 3.1 originally grouped it with the non-shipping scan-driver parts; that
exclusion is amended here and the amendment is recorded in
`PROVENANCE_icorca_binned.md`.) It takes the live `PyNuFit` and experiment as
ARGUMENTS, so this module still imports no pynu.

The remaining scan-driver parts of `orca_exact_scan.py` (`event_expectation`,
`selftest`, `main`, and the `add_pynu_root` path helper that none of the
extracted functions call) do NOT ship — their upstream equivalents belong under
`analysis/ORCA-binned-datafit/`.

Pure pandas + numpy at module level: no pynu import, no nuSQuIDS. See
`README_orca_binned.md` for the engine contract and
`PROVENANCE_icorca_binned.md` for the port map.
"""
import numpy as np
import pandas as pd

N_ERECO, N_CZRECO, N_PID = 15, 20, 3
N_BINS = N_PID * N_ERECO * N_CZRECO                 # 900


def _flat900(pid, ire0, irz0):
    return (pid * N_ERECO + ire0) * N_CZRECO + irz0


def observed_900(data_parquet):
    d = pd.read_parquet(data_parquet)
    b = _flat900(d["pid"].values.astype(np.int64),
                 d["reco_energy_bin_num"].values.astype(np.int64) - 1,
                 d["reco_cos_zenith_bin_num"].values.astype(np.int64) - 1)
    return np.bincount(b, weights=d["weight"].values.astype(float), minlength=N_BINS)


def muon_900(mc_parquet):
    """Static muon background on the 900-bin grid. RAW MuonWeight, NO NORM —
    matches Orca._compute_muon_background exactly (w_sample = self.MuonWeight[mask],
    no ×NORM; Orca.py:246). The event path adds this identical vector, so for the
    EXACTNESS gate the muon term is a common additive constant that cancels in
    Δχ² regardless of which side it sits on. (The PHYSICS convention — event worker
    puts muons in the OBSERVATION, run_pynu_grid_worker.py:47-51 — is a separate
    fit-phase question; documented in the SPEC. Here both paths add it to the model
    expectation identically, which is exactness-neutral.)"""
    df = pd.read_parquet(mc_parquet)
    mu = df[df["MC_type"] == -1]
    b = _flat900(mu["pid"].values.astype(np.int64),
                 mu["reco_energy_bin_num"].values.astype(np.int64) - 1,
                 mu["reco_cos_zenith_bin_num"].values.astype(np.int64) - 1)
    return np.bincount(b, weights=mu["weight"].values.astype(float), minlength=N_BINS)


def nu_cell_index(mc_parquet):
    """Per-nu-event 900-bin index + true-cell (ie,iz) index + true centres, in the
    Orca-loader row order (MC_type==1, 0<=Etrue<1e5). Returns arrays aligned to the
    live exp's per-event arrays (same filter as Orca.MCVariables)."""
    df = pd.read_parquet(mc_parquet)
    nu = df[df["MC_type"] == 1]
    et = nu["true_energy"].values
    cond = (et >= 0) & (et < 1e5)
    nu = nu[cond]
    b = _flat900(nu["pid"].values.astype(np.int64),
                 nu["reco_energy_bin_num"].values.astype(np.int64) - 1,
                 nu["reco_cos_zenith_bin_num"].values.astype(np.int64) - 1)
    ie = nu["true_energy_bin_num"].values.astype(np.int64) - 1
    iz = nu["true_cos_zenith_bin_num"].values.astype(np.int64) - 1
    # native quantized true centres (one value per bin)
    n_et = int(ie.max()) + 1
    n_cz = int(iz.max()) + 1
    e_c = np.full(n_et, np.nan); z_c = np.full(n_cz, np.nan)
    E = nu["true_energy"].values; cz = np.cos(nu["true_zenith"].values)
    for bb in np.unique(ie):
        e_c[bb] = E[ie == bb][0]
    for bb in np.unique(iz):
        z_c[bb] = cz[iz == bb][0]
    return b, ie, iz, e_c, z_c


def poisson_chi2(obs, n_mod):
    if np.any(n_mod <= 0):
        return 9e9
    lt = np.log(np.divide(obs, n_mod, out=np.ones_like(obs), where=n_mod > 0))
    lt[obs == 0] = 0
    return float(2 * np.sum(n_mod - obs + obs * lt))


def binned_expectation(pynufit, exp, theta_vec, bin_idx, mu900):
    """REUSE binned path — the FIX (2026-07-11): scatter the EVENT path's OWN
    per-event weight to the 900 bins, so binned == event up to summation order.

    The event path bins `BinMC(ExpectedWeight)` = ExpectedWeight * BaseWeight
    (Experiment.BinIt_MC_2D:142-160: weight_sample = array * self.BaseWeight), where
    ExpectedWeight = PhysicsWeight * NuisanceWeight (Experiment.SetExpectedWeight:229)
    and PhysicsWeight = GetOscillations() = the PROPAGATED FLUX (InitialFlux × P_osc,
    AtmOsc.GetOscillations:178 Set_initial_state(InitialFlux)+EvolveState+EvalFlavor)
    — NOT bare probability. The prior bug used a separately-built tensor phi that was
    bare O(1) probability WITHOUT the nuflux InitialFlux, inflating the rate by ~1e10.
    Reusing exp.PhysicsWeight (set by ApplyOscillations('Physics') just before this
    call) makes the per-event weight byte-identical to the event path; the ONLY
    difference is np.bincount vs histogram2d (both fill the same 900 bins) → exact up
    to float summation order. No tensor phi needed."""
    pynufit.StartNuisance()
    pynufit.ApplyNuisanceWeights(theta_vec)              # -> exp.NuisanceWeight
    expw = np.asarray(exp.PhysicsWeight, float) * np.asarray(exp.NuisanceWeight, float)
    w = expw * np.asarray(exp.BaseWeight, float)         # == BinMC's array*BaseWeight
    return np.bincount(bin_idx, weights=w, minlength=N_BINS) + mu900
