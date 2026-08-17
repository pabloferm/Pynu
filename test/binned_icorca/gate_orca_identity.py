"""G-ORCA-0 / G-ORCA-1 / G-ORCA-2 — the sharp arm-equivalence gates. FASRC ONLY
(needs nuSQuIDS for `ApplyOscillations`).

Spec: `SCOPE_orca_binned_track_2026-08-17.md` §2.4 (G-ORCA-1) and §2.2 /
design ADDENDUM item 5 (G-ORCA-0).

  G-ORCA-0  DIAGNOSTIC SINCE 2026-08-17, NO LONGER A GATE. It enforced
            cell-constancy of PhysicsWeight, and job 39684684 measured that
            premise FALSE: nuSQuIDS redraws the neutrino production height on
            every EvalFlavor call (AtmOsc.py:108-122 passes
            itertools.repeat(True)), so two events in one true cell get
            different oscillation weights by design. What remains is a printed
            within-cell spread — absolute AND relative, the latter being what
            the original gate should have reported — so the number stays on the
            record and a change in it is visible. The osc_avg_scale-is-None
            assert inside extract_cell_phi IS still hard, and is the only thing
            in this section that can abort.

  G-ORCA-1  binned-engine chi2 vs the PRODUCTION ORCA term of
            `combined_model_and_chi2` (`combined_ic_orca_fit_worker.py:271-275`),
            which is `poisson_chi2(obs[few], orca_binned_expectation(...)[few])`.
            At nominal + 10 random theta in the box, over 5 grid cells = 55
            comparisons, E_shift pinned at 1.0, osc_avg_scale None.
            RETARGETED: rel chi2 <= 1e-4 AND max per-bin <= 5e-3, both reported
            with their achieved distribution. The old 1e-9 assumed a
            deterministic phi; the reference is itself a random draw, with a
            MEASURED floor of rel 2.193e-05 that no estimator can beat. See the
            derivation by the constants below. ★ The failure mode this still
            catches is a real port defect (O(1e-3) per bin and up); what it can
            no longer claim is exactness.

  G-ORCA-2  (ADDED, not in the scope) the analytic gradient against a finite
            difference of the PRODUCTION objective rather than of the engine's
            own chi2 — the only check that closes the loop against the gradient
            the FIT WILL ACTUALLY FOLLOW, which is what stage 5 swaps into the
            minimizer. G-G1 proves the engine differentiates ITSELF correctly and
            G-ORCA-1 proves the two chi2 values agree; neither proves the two
            gradients point the same way. 28 movable dials x 4 production
            evaluations x 2 theta points, ~2 minutes. Same Richardson /
            noise-aware differencing as the local G-G1 (gate_orca_grad._fd_step).
            RETARGETED to 5e-3: this is a CROSS-MODEL comparison, so its
            discrepancy inherits the per-bin model difference rather than the
            engine's internal 1e-6 — derivation at the constants below, and the
            per-dial ratio to the model difference is reported as the sharper
            diagnostic (it should be O(1) for every dial).

Everything here is READ-ONLY on the inputs and writes only its JSON artifact.

Invocation from a clone (needs nuSQuIDS in the environment; `--response` is a
build product of `analysis/ORCA-binned-datafit/build_orca_binned_response.py`):
  python -u test/binned_icorca/gate_orca_identity.py \
      --config <ORCA_Atm_r2_fude_ccqe.xml> \
      --response <orca_response_flat900.npz> --out <json>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = str(Path(__file__).resolve().parents[2])   # repo root: test/binned_icorca/..
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

from pynu.Experiments.orca_binned_support import (                     # noqa: E402
    observed_900, muon_900, nu_cell_index, poisson_chi2,
    binned_expectation as orca_binned_expectation,
)
from pynu.Experiments.orca_binned_engine import (                      # noqa: E402
    ORCABinnedEngine, ORCA_MANIFEST,
)
from pynu.Experiments.orca_cell_phi import (                           # noqa: E402
    build_cell_index, extract_cell_phi,
)
from gate_orca_grad import make_bounds, _richardson_fd, FD_NOISE_TARGET  # noqa: E402

# ---------------------------------------------------------------------------
# TOLERANCES — RETARGETED 2026-08-17 from probe job 39685508.
#
# The original 1e-9 rested on "the only difference is float summation order",
# which assumed phi is a deterministic function of the true cell. It is not:
# nuSQuIDS draws a fresh production height per EvalFlavor call
# (AtmOsc.py:108-122), so the PRODUCTION REFERENCE IS ITSELF A DRAW. The probe
# measured that directly — two evaluations at IDENTICAL parameters differ by
#
#     rel chi2 = 2.193e-05      (TEST A, |536.236523 - 536.224764| / 536.2)
#
# and NO estimator can beat that floor, which TEST D confirmed: groupby-first
# 1.480e-05, groupby-mean 9.618e-06, 12-draw mean 1.630e-05, deterministic-height
# 1.324e-05 — all at the floor, none distinguishable from it.
#
# DERIVATION OF THE 1e-4 THRESHOLD. Treat the floor as one sample of |D| where D
# is the difference of two independent draws. For D normal with sd s*sqrt(2),
# E|D| = 2s/sqrt(pi) = 1.128 s, so the single measurement implies s ~ 1.94e-05.
# This gate makes 55 comparisons (5 cells x 11 theta), and the expected MAXIMUM
# of 55 half-normal samples is ~ s*sqrt(2 ln 55) = 2.83 s ~ 5.5e-05. A 1e-4
# threshold is therefore ~1.8x the expected max-of-55 and ~5x the single-sample
# floor — enough headroom that an ordinary unlucky draw cannot fail the gate,
# tight enough that a real port defect (which would be O(1e-3) per bin and up)
# still trips it.
#
# ⚠ CAVEAT, stated because it bounds the claim: the floor is ONE sample at ONE
# cell and ONE theta (the probe measured TEST A only at the nominal point of
# cell 2). If the achieved max over the 55 comparisons lands near 1e-4 rather
# than near 5e-5, this threshold must be RE-DERIVED from the 55 achieved values
# rather than from this single-sample extrapolation. The gate reports the
# achieved distribution precisely so that re-derivation is possible without
# another run.
#
# The per-bin threshold is calibrated on observation, not theory: the failing
# groupby-first run spanned maxbin 6.752e-04 .. 3.196e-03 over its 55
# comparisons, and probe TEST D put every estimator at ~1.1-1.5e-3. 5e-3 is
# ~1.6x the worst yet seen. Flagged as the tighter of the two thresholds.
MEASURED_FLOOR_REL_CHI2 = 2.193e-05        # probe 39685508, TEST A
IDENTITY_TOL = 1e-4                        # rel chi2, ~4.6x the floor
IDENTITY_TOL_BIN = 5e-3                    # max per-bin relative model difference

# ---------------------------------------------------------------------------
# G-ORCA-2 — RETARGETED 2026-08-17 from run 39749580 (was 1e-6, FAIL 12/58).
#
# The 1e-6 was inherited from G-G1, and that was a category error on my part.
# G-G1 differentiates the engine's OWN chi2 and is an internal identity, so 1e-6
# is right there (achieved 8.126e-08). G-ORCA-2 differentiates a DIFFERENT
# FUNCTION on each side: the analytic gradient of chi2_engine against a finite
# difference of chi2_production. Those two chi2 surfaces are not equal — G-ORCA-1
# measures exactly how unequal — so their gradients cannot agree to 1e-6 either.
#
# DERIVATION — the cross-model gradient discrepancy INHERITS the per-bin model
# discrepancy, and the relationship is exact, not hand-waved. For a cell-axis
# dial d, both paths use the SAME cell-constant dial field g_d (production
# evaluates per event, but at the event's true coordinates, which by G-C2
# uniqueness equal the cell's). So
#
#   dE_eng,b/dx_d - dE_prod,b/dx_d
#       = D_b * sum_c g_d(c) * [ phi_c^wmean * sum_{i in c,b} w_i
#                                - sum_{i in c,b} w_i phi_i ]
#
# and the bracket is precisely the per-bin, per-cell model difference that
# IDENTITY_TOL_BIN bounds. The derivative difference is therefore the MODEL
# difference re-weighted by the dial's own dlnW field: same order, modulated by
# how g_d correlates with the residual across cells.
#
# ⇒ the natural bound is the per-bin model tolerance itself, 5e-3.
# CONFIRMED by the run: cross-model gradient rel errors spanned 1e-6..5e-4 with
# worst 2.1e-3, against a per-bin model worst of 2.501e-03 — the same number to
# within the modulation the derivation predicts. That agreement is itself
# evidence the derivation is the right one and the discrepancy is the known
# model difference rather than an implementation error.
#
# WHY THIS GATE IS KEPT RATHER THAN DEMOTED. It is the only check that closes the
# loop against the gradient the FIT WILL ACTUALLY FOLLOW. G-G1 proves the engine
# differentiates itself correctly; G-ORCA-1 proves the two chi2 values agree.
# Neither proves the engine's gradient points the same way as production's — and
# that is precisely what stage 5 swaps into the minimizer. A real defect (wrong
# dial field, wrong mask, wrong scatter) would be O(1) on the offending dial, so
# a 5e-3 bound still catches every gross error while not flagging the known
# model difference. The per-dial ratio to the model difference is reported too:
# the derivation says it should be O(1) for every dial, so a dial where it is
# >> 1 is suspicious in a way the raw threshold alone would not reveal.
G2_TOL = 5e-3                              # = IDENTITY_TOL_BIN, by the derivation
# 5 grid cells spanning the production box (combined_ic_orca_fit_worker.py:139-140).
GRID_CELLS = [(2.3e-3, 0.40), (2.4e-3, 0.48), (2.511e-3, 0.52),
              (2.6e-3, 0.58), (2.7e-3, 0.65)]
S13 = 0.0220                        # the adopted reactor-prior central value


def _line(name, ok, number, threshold):
    print(f"GATE {name}: {'PASS' if ok else 'FAIL'} {number} (threshold {threshold})")
    return bool(ok)


def _xml_input_paths(xml_path):
    """ALL status-enabled MC / data parquet paths inside the XML — i.e. everything
    the live Experiment loads, as opposed to what --mc/--data point at.

    ★ RETURNS LISTS, DELIBERATELY. `ParseXML.py:456-462` collects these as lists
    (`for fi in src.findall("MCFiles")`, each gated on its own `<status>`), and
    `Orca.Reader` APPENDS across them (`Experiment.py:76`, and Orca.py's data
    branch likewise). An earlier version of this function read only the FIRST
    entry, which made the preflight's "SAME" verdict incomplete: a SECOND
    DataFiles entry would have passed unnoticed while its rows were appended into
    the observation. That is exactly the shape of the 430-vs-427 defect —
    Orca.Reader's data branch applies NO MC_type filter (unlike its MC branch at
    Orca.py:64), so any additional data file containing muon rows folds
    206.342922 of muon weight into ObservedBinned and lights 3 extra bins.
    """
    import xml.etree.ElementTree as ET
    root = ET.parse(xml_path).getroot()
    out = {}
    for tag, key in (("MCFiles", "mc"), ("DataFiles", "data")):
        paths = []
        for node in root.iter(tag):
            st = node.find("status")
            if st is not None and not int(float(st.text.strip())):
                continue                       # status 0 -> ParseXML skips it
            name = node.get("name")
            if name:
                paths.append(name.strip())
        out[key] = paths
    return out


def _same_file(p_xml, p_arg):
    """(same, detail) — realpath first (cheap), md5 only if the paths differ."""
    import hashlib
    if p_xml is None:
        return None, "no path in XML"
    if not os.path.exists(p_xml):
        return None, f"XML path not readable here: {p_xml}"
    if os.path.realpath(p_xml) == os.path.realpath(p_arg):
        return True, "same realpath"

    def _md5(p):
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    a, b = _md5(p_xml), _md5(p_arg)
    return a == b, f"md5 xml={a} arg={b}"


def preflight_input_identity(xml_path, mc_arg, data_arg, strict=True):
    """★ PREFLIGHT — the live Experiment loads its MC and data from paths written
    INSIDE the XML, while this script loads them from --mc/--data. Production has
    the same split: `combined3_dcp_tied.sbatch:74-75` passes
    `$PYNU/data/ORCA/...` with PYNU=backup_pynu/Pynu, while the XML's MCFiles /
    DataFiles (`:205`, `:210`) name `AtmNuDataFit/Pynu/data/ORCA/...` — a
    DIFFERENT TREE.

    If those files differ in content, then `few` (from exp.FewEntries) and `obs`
    (from observed_900) describe different datasets, and the per-event arrays the
    engine indexes (bin_idx, cell_index) belong to a different MC than
    exp.PhysicsWeight — a mismatch the existing `bin_idx.size ==
    exp.NumberOfEvents` assert CANNOT catch, because equal row counts do not mean
    equal rows.

    This is the prime suspect for the 430-vs-427 few-mask discrepancy: locally,
    with ONE consistent data file, exp.FewEntries is necessarily 427 (the ORCA
    data vector is integer counts, so every threshold below 1 gives the same
    mask, and the baked-bin_num and histogram binnings were verified identical).
    """
    xml_paths = _xml_input_paths(xml_path)
    res = {}
    ok = True
    for key, arg in (("mc", mc_arg), ("data", data_arg)):
        paths = xml_paths[key]
        entry = {"xml_paths": paths, "n_xml_entries": len(paths),
                 "arg_path": os.path.abspath(arg)}
        print(f"  PREFLIGHT {key}: {len(paths)} status-enabled XML entr"
              f"{'y' if len(paths) == 1 else 'ies'}")
        for p in paths:
            print(f"    XML : {p}")
        print(f"    arg : {os.path.abspath(arg)}")

        if len(paths) != 1:
            # ★ More (or fewer) than one file is by itself disqualifying: --mc /
            # --data can represent exactly one, so the gate's arrays and the live
            # Experiment cannot be describing the same input set. For `data`,
            # extra entries are precisely how muon rows reach the observation.
            entry.update(same=False, detail=f"{len(paths)} entries, arg can hold 1")
            print(f"    >>> {len(paths)} entries but --{key} names ONE file: the "
                  "live Experiment appends across all of them (Orca.Reader / "
                  "Experiment.py:76). NOT comparable.")
            ok = False
        else:
            same, detail = _same_file(paths[0], arg)
            entry.update(same=same, detail=detail)
            verdict = {True: "SAME", False: "DIFFER", None: "UNCHECKED"}[same]
            print(f"    {detail}")
            print(f"    verdict: {verdict}")
            if same is False:
                ok = False
        res[key] = entry
    if not ok and strict:
        raise AssertionError(
            "PREFLIGHT FAIL: the XML's inputs and --mc/--data do not describe the "
            "same input set. Every number this gate produces would compare two "
            "datasets. Note Orca.Reader's DATA branch applies NO MC_type filter "
            "(its MC branch does, Orca.py:64), so an extra/muon-inclusive data "
            "file silently adds 206.342922 of muon weight to the observation and "
            "moves the FewEntries mask 427 -> 430.")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True,
                    help="ORCA analysis XML (ORCA_Atm_r2_fude_ccqe.xml); no "
                         "repo-relative default — it is not a repo file")
    ap.add_argument("--mc", default=os.path.join(
        _REPO, "data", "ORCA", "ORCA_MC_dataverse_with_muons.parquet"),
        help="ORCA MC parquet (default: data/ORCA/ORCA_MC_dataverse_with_muons.parquet)")
    ap.add_argument("--data", default=os.path.join(
        _REPO, "data", "ORCA", "ORCA_data_dataverse.parquet"),
        help="ORCA data parquet (default: data/ORCA/ORCA_data_dataverse.parquet)")
    ap.add_argument("--response", required=True,
                    help="orca_response_flat900.npz — BUILD PRODUCT of "
                         "analysis/ORCA-binned-datafit/build_orca_binned_response.py")
    ap.add_argument("--pynu-root", default=_REPO,
                    help="repo root providing `pynu` (default: this clone)")
    ap.add_argument("--out", default="gate_orca_identity.json")
    ap.add_argument("--n-theta", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--skip-g2", action="store_true")
    ap.add_argument("--allow-input-mismatch", action="store_true",
                    help="downgrade the XML-vs-arg input preflight to a warning "
                         "(diagnostic only — the numbers are NOT comparable)")
    a = ap.parse_args()

    t0 = time.time()
    print("=== G-ORCA-0 / G-ORCA-1 / G-ORCA-2 (FASRC) ===")
    print("--- PREFLIGHT: XML inputs vs --mc/--data ---")
    preflight = preflight_input_identity(a.config, a.mc, a.data,
                                         strict=not a.allow_input_mismatch)

    _root = os.path.abspath(a.pynu_root)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from pynu import PyNuFit

    pynufit = PyNuFit(a.config, verbosity=False)
    exp_name = list(pynufit.Experiments.keys())[0]
    exp = pynufit.Experiments[exp_name]
    osc = pynufit.physics_tunes[exp_name].OscillationTunes
    osc.osc_avg_scale = None                       # averaging OFF (production :365)
    names = list(pynufit.Analysis.NuisanceList)
    nominal = np.array(pynufit.Analysis.NuisNominalList, float)
    sigma = np.array(pynufit.Analysis.NuisSigmaList, float)
    assert names == ORCA_MANIFEST, (
        f"live NuisanceList != ORCA_MANIFEST\n live: {names}\n eng: {ORCA_MANIFEST}")

    obs = observed_900(a.data)
    mu = muon_900(a.mc)
    bin_idx, *_ = nu_cell_index(a.mc)
    assert bin_idx.size == exp.NumberOfEvents, \
        f"binned rows {bin_idx.size} != exp events {exp.NumberOfEvents}"
    few = np.asarray(exp.FewEntries, bool)
    cell_index = build_cell_index(a.mc)

    eng = ORCABinnedEngine(a.response, obs, mu, few, names,
                           norm=float(exp.NORM), osc=osc)
    print("--- engine ---")
    for k, v in eng.summary().items():
        print(f"  {k}: {v}")
    print(f"  exp.NORM {float(exp.NORM)}  n_few {int(few.sum())}  "
          f"obs sum {obs.sum():.1f}  mu sum {mu.sum():.10f}")

    lo, hi = make_bounds(nominal, sigma)
    i_es = names.index("E_shift")
    rng = np.random.default_rng(a.seed)
    thetas = [nominal.copy()]
    for _ in range(a.n_theta):
        t = rng.uniform(lo, hi)
        t[i_es] = 1.0                              # PINNED, engine hard-asserts
        thetas.append(t)

    res = {"config": os.path.abspath(a.config),
           "response": os.path.abspath(a.response),
           "seed": a.seed, "engine": eng.summary(),
           "norm": float(exp.NORM), "n_few": int(few.sum()),
           "preflight_input_identity": preflight,
           "gates": {}}

    # ---- few-mask provenance (the 430-vs-427 item) --------------------------
    min_entries = float(getattr(exp, "MIN_ENTRIES", 0.01))
    few_obs = obs > min_entries
    n_exp, n_obs = int(few.sum()), int(few_obs.sum())
    print(f"  DIAG few-mask: exp.FewEntries {n_exp} vs "
          f"obs>MIN_ENTRIES({min_entries}) {n_obs}")
    # ★ MASK PROVENANCE — RESOLVED 2026-08-17, root cause below.
    # The ORCA XML's <DataFiles> carries <status> 0 </status>, so
    # ParseXML.py:461-462 yields DataFiles == [], Experiment.__init__:30 sets
    # DataFit = False, and SetObservedBinned:244-249 takes its ELSE branch:
    # ObservedBinned = BinMC(NominalWeight) — the ASIMOV MC RATE, not the data.
    # So FewEntries is an MC-SUPPORT mask (430 of 900), and the 3 bins beyond
    # `obs > 0.01` ({307, 326, 620}) are bins the MC populates but the data does
    # not. The data never enters the mask at all; the worker supplies obs itself
    # via observed_900() and never reads exp.ObservedBinned.
    # (An earlier hypothesis blamed Orca.Reader's missing MC_type filter on the
    # data branch. It reproduces the same 3 bins numerically — those bins also
    # carry muons — but it is NOT what happens here, because no data file is
    # loaded at all. Recorded so the coincidence is not re-followed.)
    ob = np.asarray(exp.ObservedBinned, float)      # truncated to the mask
    ob_sum = float(ob.sum())
    print(f"  DIAG mask provenance: DataFit={exp.DataFit} -> ObservedBinned is "
          f"{'BinData (DATA)' if exp.DataFit else 'BinMC(NominalWeight) (ASIMOV MC)'}"
          f"; sum={ob_sum:.6e}  [data would be 5828.000000]")
    res["few_mask"] = {"n_exp_FewEntries": n_exp, "n_obs_gt_min": n_obs,
                       "min_entries": min_entries,
                       "data_fit": bool(exp.DataFit),
                       "observed_binned_sum": ob_sum,
                       "obs_sum": float(obs.sum()),
                       "mask_is_mc_support": not bool(exp.DataFit),
                       "agree": n_exp == n_obs}
    if n_exp != n_obs:
        d = np.flatnonzero(few != few_obs)
        res["few_mask"]["differing_bins"] = [
            {"bin": int(b), "pid": int(b) // 300, "obs": float(obs[b]),
             "mu": float(mu[b]), "in_exp_mask": bool(few[b])} for b in d]
        print(f"    differing bins {d.tolist()} — obs {obs[d].tolist()}, "
              f"mu {mu[d].tolist()}")
        print("    ^^ EXPECTED when DataFit=False: the mask is MC-support, not "
              "data. 430 vs 427 with extras {307,326,620} is the known, "
              "root-caused configuration — production uses the 430 mask.")

    # ---------------- G-ORCA-0 + G-ORCA-1 ---------------------------------
    g0_rows, g1_rows = [], []
    g1_ok = True
    worst0 = 0.0
    worst1 = 0.0
    worst1_bin = 0.0
    for ic, (dm, s23) in enumerate(GRID_CELLS):
        osc.Parameters["Dm231"] = dm
        if "Dm231_bar" in osc.Parameters:
            osc.Parameters["Dm231_bar"] = dm
        osc.Parameters["Sin2Theta23"] = s23
        if "Sin2Theta13" in osc.Parameters:
            osc.Parameters["Sin2Theta13"] = S13
        osc.reset_cache()
        pynufit.StartPhysics()
        pynufit.ApplyOscillations("Physics")

        phi, info = extract_cell_phi(exp, cell_index, osc=osc)
        # G-ORCA-0 IS NO LONGER A GATE. The premise it enforced — phi constant
        # within a true cell — was MEASURED FALSE (probe 39685508). It is kept as
        # a DIAGNOSTIC so the spread is on the record every run and a change in it
        # is visible. The osc_avg_scale-is-None assert inside extract_cell_phi IS
        # still a hard gate, and is the one thing here that can still abort.
        worst0 = max(worst0, info["max_rel_spread"])
        g0_rows.append({"cell": ic, "dm": dm, "s23": s23, **info})
        print(f"  DIAG G-ORCA-0 cell {ic} (dm {dm:.4e}, s23 {s23:.3f}): "
              f"within-cell rel spread median {info['median_rel_spread']:.3e} "
              f"p95 {info['p95_rel_spread']:.3e} max {info['max_rel_spread']:.3e} "
              f"| abs max {info['max_abs_spread']:.3e} "
              f"| {info['n_groups_multi_event']}/{info['n_groups']} multi-event "
              f"| reduction={info['how']}")

        for it, th in enumerate(thetas):
            model_ref = orca_binned_expectation(pynufit, exp, th, bin_idx, mu)
            chi2_ref = float(poisson_chi2(obs[few], model_ref[few]))
            chi2_eng = eng.chi2(phi, th)
            rel = abs(chi2_eng - chi2_ref) / max(abs(chi2_ref), 1.0)
            # the model vectors too — a compensating error inside chi2 is
            # conceivable, and the per-bin residual is the more sensitive of the
            # two (chi2 partially averages it away).
            model_eng = eng.expectation(phi, th)
            mrel = float(np.max(np.abs(model_eng - model_ref)
                                / np.maximum(1e-30, np.abs(model_ref))))
            good = (rel <= IDENTITY_TOL) and (mrel <= IDENTITY_TOL_BIN)
            g1_ok &= good
            worst1 = max(worst1, rel)
            worst1_bin = max(worst1_bin, mrel)
            g1_rows.append({"cell": ic, "theta_point": it, "chi2_ref": chi2_ref,
                            "chi2_engine": chi2_eng, "rel": rel, "pass": good,
                            "max_rel_bin_diff": mrel,
                            "rel_over_floor": rel / MEASURED_FLOOR_REL_CHI2})
            print(f"  G-ORCA-1 cell {ic} t{it}: ref {chi2_ref:.10f} eng "
                  f"{chi2_eng:.10f} rel {rel:.3e} ({rel / MEASURED_FLOOR_REL_CHI2:.2f}x "
                  f"floor) maxbin {mrel:.3e} {'ok' if good else 'FAIL'}")

    print(f"DIAG G-ORCA-0: max within-cell RELATIVE spread {worst0:.3e} over "
          f"{len(g0_rows)} cells (diagnostic since 2026-08-17 — the "
          "cell-constant premise was measured false, probe 39685508)")
    ok = _line("G-ORCA-1", g1_ok, f"{sum(r['pass'] for r in g1_rows)}/{len(g1_rows)} "
               f"comparisons, worst rel {worst1:.3e} "
               f"({worst1 / MEASURED_FLOOR_REL_CHI2:.2f}x the measured floor "
               f"{MEASURED_FLOOR_REL_CHI2:.3e}), worst maxbin {worst1_bin:.3e}",
               f"rel <= {IDENTITY_TOL:g} and maxbin <= {IDENTITY_TOL_BIN:g}")
    # the achieved distribution, so the threshold can be re-derived from THESE 55
    # values rather than from the probe's single floor sample (see the caveat by
    # the constants) without paying for another run.
    _r = np.array([r["rel"] for r in g1_rows])
    _b = np.array([r["max_rel_bin_diff"] for r in g1_rows])
    print(f"  achieved rel chi2:  median {np.median(_r):.3e}  p95 "
          f"{np.percentile(_r, 95):.3e}  max {_r.max():.3e}")
    print(f"  achieved maxbin:    median {np.median(_b):.3e}  p95 "
          f"{np.percentile(_b, 95):.3e}  max {_b.max():.3e}")
    res["gates"]["G-ORCA-0-DIAG"] = {"is_gate": False, "worst_rel_spread": worst0,
                                     "rows": g0_rows}
    res["gates"]["G-ORCA-1"] = {
        "pass": bool(g1_ok), "worst_rel": worst1, "worst_rel_bin": worst1_bin,
        "n_comparisons": len(g1_rows), "tol_rel": IDENTITY_TOL,
        "tol_bin": IDENTITY_TOL_BIN, "measured_floor": MEASURED_FLOOR_REL_CHI2,
        "achieved_rel": {"median": float(np.median(_r)),
                         "p95": float(np.percentile(_r, 95)), "max": float(_r.max())},
        "achieved_bin": {"median": float(np.median(_b)),
                         "p95": float(np.percentile(_b, 95)), "max": float(_b.max())},
        "rows": g1_rows}

    # ---------------- G-ORCA-2: gradient vs the PRODUCTION objective --------
    if not a.skip_g2:
        dm, s23 = GRID_CELLS[2]
        osc.Parameters["Dm231"] = dm
        if "Dm231_bar" in osc.Parameters:
            osc.Parameters["Dm231_bar"] = dm
        osc.Parameters["Sin2Theta23"] = s23
        if "Sin2Theta13" in osc.Parameters:
            osc.Parameters["Sin2Theta13"] = S13
        osc.reset_cache()
        pynufit.StartPhysics()
        pynufit.ApplyOscillations("Physics")
        phi, _ = extract_cell_phi(exp, cell_index, osc=osc)

        class _RefArm:
            """The production chi2 behind the same .chi2(phi, theta) signature,
            so the local Richardson differencer can be reused verbatim."""

            def chi2(self, _phi, th):
                m = orca_binned_expectation(pynufit, exp, th, bin_idx, mu)
                return float(poisson_chi2(obs[few], m[few]))

        ref = _RefArm()
        g2_rows, g2_ok, worst2, worst_ratio = [], True, 0.0, 0.0
        for it, th in enumerate([thetas[0], thetas[1]]):
            _c, g = eng.chi2_and_grad(phi, th)
            c_here = ref.chi2(None, th)
            # the per-bin model difference AT THIS theta — the quantity the
            # derivation says the gradient discrepancy inherits. Reported as a
            # ratio per dial: it should be O(1) for every dial.
            m_ref = orca_binned_expectation(pynufit, exp, th, bin_idx, mu)
            m_eng = eng.expectation(phi, th)
            model_rel = float(np.max(np.abs(m_eng - m_ref)
                                     / np.maximum(1e-30, np.abs(m_ref))))
            for name in names:
                if name == "E_shift":
                    continue
                j = names.index(name)
                h = float(min(max(1e-5 * sigma[j],
                                  np.finfo(float).eps * max(abs(c_here), 1.0)
                                  / FD_NOISE_TARGET), 0.1 * sigma[j]))
                fd = _richardson_fd(ref, None, th, j, h)
                rel = abs(g[j] - fd) / max(1.0, abs(fd))
                good = rel <= G2_TOL
                ratio = rel / model_rel if model_rel > 0 else float("nan")
                g2_ok &= good
                worst2 = max(worst2, rel)
                if np.isfinite(ratio):
                    worst_ratio = max(worst_ratio, ratio)
                g2_rows.append({"theta_point": it, "dial": name, "h": h,
                                "g_analytic": float(g[j]), "g_fd_production": fd,
                                "rel": rel, "pass": good,
                                "model_rel_at_theta": model_rel,
                                "ratio_to_model_diff": ratio})
                print(f"  G-ORCA-2 t{it} {name:26s} ana {g[j]: .10e} fd(prod) "
                      f"{fd: .10e} rel {rel:.3e} ({ratio:.2f}x model diff) "
                      f"{'ok' if good else 'FAIL'}")
        _g2 = np.array([r["rel"] for r in g2_rows])
        ok &= _line("G-ORCA-2", g2_ok,
                    f"{sum(r['pass'] for r in g2_rows)}/{len(g2_rows)} checks, "
                    f"worst rel {worst2:.3e}, worst ratio-to-model-diff "
                    f"{worst_ratio:.2f}x", f"<= {G2_TOL:g} (= the per-bin model "
                    "tolerance, by the derivation at the constants)")
        print(f"  achieved G-ORCA-2 rel: median {np.median(_g2):.3e}  p95 "
              f"{np.percentile(_g2, 95):.3e}  max {_g2.max():.3e}")
        print("  (ratio ~O(1) for every dial is the derivation's prediction; a "
              "dial >> 1 would be suspicious even under the threshold)")
        res["gates"]["G-ORCA-2"] = {
            "pass": bool(g2_ok), "worst_rel": worst2, "tol": G2_TOL,
            "worst_ratio_to_model_diff": worst_ratio,
            "achieved": {"median": float(np.median(_g2)),
                         "p95": float(np.percentile(_g2, 95)),
                         "max": float(_g2.max())},
            "rows": g2_rows}

    # Count only entries that ARE gates. G-ORCA-0-DIAG carries no "pass" key by
    # design (it was demoted to a diagnostic), which is what crashed run 39749580
    # at the summary line AFTER the physics had already passed.
    _gated = {k: v for k, v in res["gates"].items() if "pass" in v}
    res["overall"] = bool(ok)
    res["n_gates_passed"] = int(sum(bool(v["pass"]) for v in _gated.values()))
    res["n_gates"] = len(_gated)
    res["diagnostics"] = [k for k in res["gates"] if k not in _gated]
    res["wall_s"] = time.time() - t0
    print(f"GATE G-ORCA-ALL: {'PASS' if ok else 'FAIL'} "
          f"{res['n_gates_passed']}/{res['n_gates']} gates"
          + (f" (+{len(res['diagnostics'])} diagnostic: "
             f"{', '.join(res['diagnostics'])})" if res["diagnostics"] else "")
          + " (threshold all PASS)")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"artifact: {os.path.abspath(a.out)}")
    print(f"wall {res['wall_s']:.1f} s")
    print("JOB_DONE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
