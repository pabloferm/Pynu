"""G-G1 / G-G2 (+ the dial-field unit tests and the stat-only test) — LOCAL gates
for the ORCA binned engine. No nuSQuIDS, no PyNuFit, no cluster.

Spec: `SCOPE_orca_binned_track_2026-08-17.md` §2.4, with the two ADDENDUM
corrections (item 6 stat-only, item 7 FOUR inert ORCA dials).

  DIALS  dial-field unit tests. Every one of the 23 shared forms plus the 6 ORCA
         detector forms is checked EVENT BY EVENT against the LIVE tune objects
         (`pynu.PhysicsTunes.Flux.AtmoFlux`, `.CrossSection.WaterXSection`,
         `.Detector.ORCADetector`) on 10,000 random events at random theta draws
         in the +/-5 sigma box, at 1e-12. Both the forward factor and the
         derivative are checked: `dlnw * factor` must equal the tune's own
         `diff_*` output. The tunes import cleanly without nuflux (nuflux is
         pulled in by the EXPERIMENT constructors, not by PhysicsTunes), so this
         is a comparison against live code, not against source text.

  G-G1   per-dial analytic gradient vs CENTRAL finite difference, on the real
         flat900 response with a SEEDED SYNTHETIC phi. The gradient identity is
         phi-independent — phi enters only as a positive per-cell prefactor — so
         a synthetic phi makes this a real gate, not a mock. One line per dial
         per theta point; an aggregate norm would hide the one wrong dial.
         Criterion |g_ana - g_FD| / max(1, |g_FD|) <= 1e-6, as specified. The
         DIFFERENCING SCHEME deviates from the scope — Richardson extrapolation
         at a noise-aware step, because the scope's plain difference at
         h = max(1e-5 * sigma_d, 1e-7) is roundoff-limited well ABOVE 1e-6 on a
         chi2 of order 5.7e4. `_fd_step` carries the measurements; the scope's
         own step is reported alongside as `rel_err_spec_step`.

  G-G1J  the same adjoint core against a better-conditioned finite difference —
         FD of the model (O(1e2)) rather than of chi2 (O(5.7e4)). Independent
         corroboration of G-G1 that does not have to fight differencing noise.

  G-G2   zero-gradient invariants, BITWISE (not < tol):
           a) the FOUR inert ORCA dials — the three *_subgev bands plus
              `normalization_below1GeV`, whose E < 1 mask is EMPTY on ORCA
              (min true E 1.1220184543 GeV). The design names only three.
           b) the pinned `E_shift` slot.
           c) muon independence: dE_b/dx carries no muon term, so the
              vector-Jacobian product at a fixed v is BITWISE identical between
              an engine built with mu900 and one built with mu900 = 0. (The chi2
              and its residual DO change with mu900 — that is what makes the
              Jacobian, not the gradient, the honest statement of the invariant.)

  G-ORCA-1L  the LOCAL surrogate of the FASRC identity gate: production's own
         model formula, `bincount(bin_idx, PhysicsWeight * NuisanceWeight *
         BaseWeight) + mu900`, rebuilt per event over all 592,099 rows with the
         dials composed through the LIVE tunes and a synthetic CELL-CONSTANT
         PhysicsWeight standing in for the nuSQuIDS one. Everything G-ORCA-1
         tests except the oscillation call itself, at the same 1e-9 criterion.

  STAT   stat-only contract (risk R2): chi2(nominal) must equal
         poisson_chi2(obs[few], E[few]) exactly, with NO penalty term. An
         arm-internal Gaussian would silently double-count the worker's union
         prior and would still PASS G-ORCA-1.

Usage (from a clone; `--response` is a build product, `--xml` the analysis XML):
  python test/binned_icorca/gate_orca_grad.py \
      --response <orca_response_flat900.npz> \
      --xml      <ORCA_Atm_r2_fude_ccqe.xml> \
      --out      <gate_orca_grad.json>
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = str(Path(__file__).resolve().parents[2])   # repo root: test/binned_icorca/..
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

from pynu.Experiments.orca_binned_support import (                    # noqa: E402
    observed_900, muon_900, poisson_chi2,
)
from pynu.Experiments import binned_dial_fields as bdf                # noqa: E402
from pynu.Experiments.orca_binned_engine import (                     # noqa: E402
    ORCABinnedEngine, ORCA_MANIFEST, MOVABLE, INERT_DIALS,
    _orca_detector_fields,
)
from pynu.Experiments.orca_cell_phi import build_cell_index           # noqa: E402

MIN_ENTRIES = 0.01            # Orca.py:27; FewEntries = Observed > MIN_ENTRIES
NORM = 1.0e4                  # FitExposure (XML <exposure> 1.0) * 1e4, Orca.py:180
BOUND_NSIGMA = 5.0            # combined_ic_orca_fit_worker.py:231-239
POS_FLOOR = 0.01
G1_TOL = 1e-6
UNIT_TOL = 1e-12
# Target roundoff floor for the finite differences, a decade under G1_TOL.
# See `_fd_step` for why the scope's fixed step cannot reach G1_TOL here.
FD_NOISE_TARGET = 1e-7
LOCAL_IDENTITY_TOL = 1e-9   # G-ORCA-1L, same criterion as the FASRC G-ORCA-1


# ---------------------------------------------------------------------------
def read_manifest(xml_path):
    """(names, nominal, sigma) for the ACTIVE dials, in XML order."""
    root = ET.parse(xml_path).getroot()
    names, nom, sig = [], [], []
    for nu in root.iter("nuisance"):
        st = nu.find("status")
        if st is not None and int(float(st.text.strip())) != 1:
            continue
        names.append((nu.get("name") or nu.find("name").text).strip())
        nom.append(float(nu.find("nominal").text))
        sig.append(float(nu.find("sigma").text))
    return names, np.array(nom), np.array(sig)


def make_bounds(nominal, sigma):
    lo = nominal - BOUND_NSIGMA * sigma
    hi = nominal + BOUND_NSIGMA * sigma
    lo = np.where((nominal > 0) & (lo < POS_FLOOR), POS_FLOOR, lo)
    return lo, hi


def _line(name, ok, number, threshold):
    print(f"GATE {name}: {'PASS' if ok else 'FAIL'} {number} (threshold {threshold})")
    return bool(ok)


# ---------------------------------------------------------------------------
# DIALS — the dial-field unit tests against the LIVE tunes
# ---------------------------------------------------------------------------
class MockExperiment:
    """The COMPLETE attribute set the 23 shared + 6 ORCA detector tunes touch.

    Verified by reading every method body: ETrue, CosZTrue, nuPDG, CC,
    NumberOfEvents (all of AtmoFlux + WaterXSection's shared four + ORCADetector),
    Mode (WaterXSection's IC-only block, present so an accidental IC-dial call
    fails loudly rather than silently), Sample (ORCADetector f_HPT/f_Shower).
    """

    def __init__(self, E, cz, pdg, cc, sample=None, mode=None):
        self.ETrue = np.asarray(E, float)
        self.CosZTrue = np.asarray(cz, float)
        self.nuPDG = np.asarray(pdg, np.int64)
        self.CC = np.asarray(cc, np.int64)
        self.NumberOfEvents = int(self.ETrue.size)
        self.Mode = np.zeros(self.NumberOfEvents, np.int64) if mode is None \
            else np.asarray(mode, np.int64)
        self.Sample = np.zeros(self.NumberOfEvents, np.int64) if sample is None \
            else np.asarray(sample, np.int64)
        self.ENERGY_SCALE = 1.0

    def set_energy_scale(self, x):
        self.ENERGY_SCALE = float(x)


def gate_dial_fields(pynu_root, xml_path, n_events=10000, n_draws=8, seed=20260817):
    """Check all 23 shared + 6 ORCA detector forms against the live tunes."""
    sys.path.insert(0, os.path.abspath(pynu_root))
    from pynu.PhysicsTunes.Flux.AtmoFlux import AtmosphericFlux
    from pynu.PhysicsTunes.CrossSection.WaterXSection import WaterXSection
    from pynu.PhysicsTunes.Detector.ORCADetector import ORCADetector

    names, nominal, sigma = read_manifest(xml_path)
    nom = dict(zip(names, nominal))
    sg = dict(zip(names, sigma))
    lo, hi = make_bounds(nominal, sigma)
    box = {n: (lo[i], hi[i]) for i, n in enumerate(names)}

    rng = np.random.default_rng(seed)
    # A deliberately hostile event sample: log-uniform in E across all three flux
    # bands AND both f_HE thresholds, uniform in cz across the horizon, all six
    # pdg, both currents, all three morphologies.
    E = 10 ** rng.uniform(-0.5, 3.2, n_events)
    cz = rng.uniform(-1.0, 1.0, n_events)
    pdg = rng.choice([-16, -14, -12, 12, 14, 16], n_events)
    cc = rng.integers(0, 2, n_events)
    sample = rng.integers(0, 3, n_events)
    exp = MockExperiment(E, cz, pdg, cc, sample=sample)
    geom = bdf.build_cell_geometry(E, cz, pdg, cc)
    det_fields = _orca_detector_fields(geom)

    tunes = {}
    for obj in (AtmosphericFlux(), WaterXSection(), ORCADetector()):
        for n in list(bdf.FIELDS) + list(det_fields):
            if hasattr(obj, n) and n not in tunes:
                tunes[n] = obj

    registry = dict(bdf.FIELDS)
    registry.update(det_fields)

    rows, worst_f, worst_d, ok = [], 0.0, 0.0, True
    for name in list(bdf.SHARED_FLUX_19) + list(bdf.SHARED_XSEC_4) + \
            ["f_all", "f_tauCC", "f_NC", "f_HE"]:
        if name not in tunes:
            raise AssertionError(f"no live tune exposes {name!r}")
        obj = tunes[name]
        fwd = getattr(obj, name)
        dfn = getattr(obj, "diff_" + name)
        lo_n, hi_n = box[name]
        xs = [nom[name]] + list(rng.uniform(lo_n, hi_n, n_draws))
        mf = md = 0.0
        for x in xs:
            want_f = np.asarray(fwd(exp, x), float)
            want_d = np.asarray(dfn(exp, x), float)
            got_f = np.asarray(registry[name].factor_fn(geom, x), float)
            got_d = np.asarray(
                registry[name].dlnw_fn(geom, x, registry[name].factor_fn(geom, x)),
                float) * got_f                       # dlnW * W == the tune's diff
            want_f, got_f = np.broadcast_arrays(want_f, got_f)
            want_d, got_d = np.broadcast_arrays(want_d, got_d)
            mf = max(mf, float(np.max(np.abs(want_f - got_f)
                                      / np.maximum(1.0, np.abs(want_f)))))
            md = max(md, float(np.max(np.abs(want_d - got_d)
                                      / np.maximum(1.0, np.abs(want_d)))))
        good = (mf <= UNIT_TOL) and (md <= UNIT_TOL)
        ok &= good
        worst_f, worst_d = max(worst_f, mf), max(worst_d, md)
        rows.append({"dial": name, "max_rel_factor": mf, "max_rel_diff": md,
                     "pass": good, "n_theta": len(xs)})
        print(f"  DIAL {name:26s} factor {mf:.3e}  diff {md:.3e}  "
              f"{'ok' if good else 'FAIL'}")

    _line("G-DIALS", ok, f"{sum(r['pass'] for r in rows)}/{len(rows)} forms, worst "
          f"factor {worst_f:.3e}, worst diff {worst_d:.3e}", f"<= {UNIT_TOL:g}")
    return {"pass": bool(ok), "worst_rel_factor": worst_f, "worst_rel_diff": worst_d,
            "n_events": n_events, "rows": rows}


# ---------------------------------------------------------------------------
# Engine construction (local): real response + real obs/mu/few + synthetic phi
# ---------------------------------------------------------------------------
def production_few_mask(response, norm=NORM, min_entries=MIN_ENTRIES):
    """The mask PRODUCTION uses — reproduced exactly, and it is NOT `obs > 0.01`.

    ★ ROOT-CAUSED 2026-08-17, correcting this file's earlier assumption.
    `combined_ic_orca_fit_worker.py:364` takes `few_orca = orca_exp.FewEntries`,
    and the ORCA XML's <DataFiles> block carries `<status> 0 </status>`.
    `ParseXML.py:461-462` appends only status-enabled entries, so
    `DataFiles == []`, so `Experiment.__init__:30` sets `DataFit = False`, so
    `Experiment.SetObservedBinned:244-249` takes its ELSE branch:

        self.ObservedBinned = self.BinMC(self.NominalWeight)    # ASIMOV, not data
        self.FewEntries     = self.ObservedBinned > self.MIN_ENTRIES

    So the "observation" defining the mask is the MC rate. The data never enters
    it — the worker supplies `obs` itself via `observed_900(args.orca_data)` and
    never reads `exp.ObservedBinned`. The mask is therefore MC-support, i.e.
    "bins the ORCA MC can populate": 430 of 900.

    Every ORCA tune is exactly 1 elementwise at nominal, so NominalWeight == 1
    and BinMC(NominalWeight) is the summed BaseWeight per bin — which the
    response already stores. Verified three ways: against a direct parquet
    computation (max|diff| 1.1e-01 on values of order 1e12), against
    `obs > 0.01` (430 vs 427, the extra bins exactly {307, 326, 620}), and
    against the cluster's own exp.FewEntries (430).

    Using `obs > MIN_ENTRIES` here instead compares the engine against
    production on a DIFFERENT bin set, worth +2.89 in chi2 at nominal.
    """
    r = np.load(response, allow_pickle=True) \
        if isinstance(response, (str, os.PathLike)) else response
    Rb = np.asarray(r["R_b"], np.int64)
    Rvn = np.asarray(r["R_v"], float) * norm
    return np.bincount(Rb, weights=Rvn,
                       minlength=int(r["n_bins"])) > min_entries


def build_local_engine(response, mc, data, names, mu_zero=False, few=None):
    obs = observed_900(data)
    mu = muon_900(mc)
    if few is None:
        few = production_few_mask(response)   # == exp.FewEntries (430), not obs>0.01
    return ORCABinnedEngine(response, obs, np.zeros_like(mu) if mu_zero else mu,
                            few, names, norm=NORM), obs, mu, few


def synthetic_phi(eng, seed=20260817):
    """Seeded positive (2, 3, 40, 80) phi, scaled so the nominal model total is
    O(obs total). The gradient identity is phi-independent; the scaling only
    keeps the finite differences well conditioned."""
    rng = np.random.default_rng(seed)
    phi = rng.lognormal(mean=0.0, sigma=0.8, size=(2, 3, eng.n_etrue, eng.n_cztrue))
    th = None
    return phi, th


def scale_phi(eng, phi, theta):
    tot = float(eng.expectation(phi, theta).sum() - eng.mu.sum())
    target = float(eng.obs.sum())
    return phi * (target / tot)


# ---------------------------------------------------------------------------
def _central_fd(eng, phi, th, j, h):
    tp, tm = th.copy(), th.copy()
    tp[j] += h
    tm[j] -= h
    return (eng.chi2(phi, tp) - eng.chi2(phi, tm)) / (2.0 * h)


def _richardson_fd(eng, phi, th, j, h):
    """Richardson-extrapolated central difference: (4 D(h) - D(2h)) / 3.

    Cancels the O(h^2) truncation term, leaving O(h^4), at the cost of one extra
    pair of evaluations and a ~1.5x noise amplification. That is what makes a
    single a-priori step rule work for all 29 dials at once (see `_fd_step`).
    """
    d1 = _central_fd(eng, phi, th, j, h)
    d2 = _central_fd(eng, phi, th, j, 2.0 * h)
    return (4.0 * d1 - d2) / 3.0


def _fd_step(sigma_d, chi2_here):
    """The FD step, chosen A PRIORI (never from the analytic answer).

    ★ DEVIATION FROM SCOPE §2.4, with its justification measured. The scope
    prescribes a plain central difference at h = max(1e-5 * sigma_d, 1e-7). On
    this problem chi2 is O(5.7e4), so that step has a ROUNDOFF FLOOR of
    eps * chi2 / h ~ 2.2e-16 * 5.7e4 / 1e-6 = 1.3e-5 ABSOLUTE — an order of
    magnitude ABOVE the 1e-6 criterion, which for a near-zero-gradient dial IS an
    absolute test, because of the max(1, |g_FD|) denominator. The scope's step is
    therefore unreachable for those dials no matter how correct the analytic
    gradient is: 11 of 174 checks failed on it, every one of them a dial whose
    gradient is smaller than the differencing noise.

    Measured proof that the FD, not the analytic gradient, is the inaccurate side
    (flux_numubar_mid at theta point 5; `step_ladder` in the artifact records the
    same ladder for every check that misses at the scope's step):

        h = 1e-6   |g_ana - g_FD| = 6.75e-06     roundoff floor 1.27e-05
        h = 1e-5                    2.04e-07                    1.27e-06
        h = 1e-4                    2.18e-08                    1.27e-07
        h = 1e-3                    7.71e-11                    1.27e-08

    The discrepancy tracks the floor as 1/h across four decades and then settles
    at 1e-10, the SK precedent. But simply raising h trades roundoff for
    truncation — at h ~ 1.3e-3 a plain central difference then misses on 16
    checks, worst 1.26e-3, led by the strongly curved f_HPT. Neither fixed step
    works for all 29 dials.

    Resolution: raise the step until the roundoff floor is a decade under the
    criterion, cap it at 0.1 * sigma_d, and kill the resulting truncation with
    Richardson extrapolation. Measured over the 174 checks:

        plain, floor target 1e-8    16 fail, worst 1.26e-03
        Richardson, target 1e-8      1 fail, worst 1.58e-06
        Richardson, target 1e-9      3 fail, worst 1.90e-02
        Richardson, target 1e-7      0 fail, worst 8.13e-08   <- adopted

    The plain-central-difference result at the scope's own step is reported
    alongside as `rel_err_spec_step`, so nothing is hidden by the change.
    """
    h_spec = max(1e-5 * sigma_d, 1e-7)
    h_noise = np.finfo(float).eps * max(abs(chi2_here), 1.0) / FD_NOISE_TARGET
    return float(min(max(h_spec, h_noise), 0.1 * sigma_d)), float(h_spec)


def gate_g1(eng, phi, names, nominal, sigma, n_theta=5, seed=20260817):
    lo, hi = make_bounds(nominal, sigma)
    i_es = names.index("E_shift")
    rng = np.random.default_rng(seed)
    thetas = [np.array(nominal, float)]
    for _ in range(n_theta):
        t = rng.uniform(lo, hi)
        t[i_es] = 1.0                                # pinned; engine hard-asserts
        thetas.append(t)

    sig = dict(zip(names, sigma))
    rows, worst, worst_spec = [], 0.0, 0.0
    per_dial_worst, ok, ok_spec = {}, True, True
    for it, th in enumerate(thetas):
        c_here, g = eng.chi2_and_grad(phi, th)
        for name in names:
            if name == "E_shift":
                continue
            j = names.index(name)
            h, h_spec = _fd_step(sig[name], c_here)
            g_fd = _richardson_fd(eng, phi, th, j, h)
            g_fd_spec = _central_fd(eng, phi, th, j, h_spec)
            err = abs(g[j] - g_fd) / max(1.0, abs(g_fd))
            err_spec = abs(g[j] - g_fd_spec) / max(1.0, abs(g_fd_spec))
            good, good_spec = err <= G1_TOL, err_spec <= G1_TOL
            ok &= good
            ok_spec &= good_spec
            worst, worst_spec = max(worst, err), max(worst_spec, err_spec)
            per_dial_worst[name] = max(per_dial_worst.get(name, 0.0), err)
            row = {"theta_point": it, "dial": name, "g_analytic": float(g[j]),
                   "g_fd": float(g_fd), "rel_err": err, "pass": good,
                   "h": h, "h_spec": h_spec, "g_fd_spec_step": float(g_fd_spec),
                   "rel_err_spec_step": err_spec, "pass_spec_step": good_spec,
                   "fd_noise_floor": float(np.finfo(float).eps * abs(c_here) / h),
                   "movable": name in MOVABLE}
            if not good_spec:
                # keep the 1/h convergence evidence in the artifact
                row["step_ladder"] = [
                    {"h": float(max(m * sig[name], 1e-7)),
                     "g_fd": float(_central_fd(eng, phi, th, j,
                                               max(m * sig[name], 1e-7))),
                     "abs_err": float(abs(g[j] - _central_fd(
                         eng, phi, th, j, max(m * sig[name], 1e-7)))),
                     "noise_floor": float(np.finfo(float).eps * abs(c_here)
                                          / max(m * sig[name], 1e-7))}
                    for m in (1e-5, 1e-4, 1e-3, 1e-2)]
            rows.append(row)
            note = "" if good_spec else \
                f"   [spec-step rel {err_spec:.2e}, roundoff-limited]"
            print(f"  G-G1 t{it} {name:26s} ana {g[j]: .10e} fd {g_fd: .10e} "
                  f"rel {err:.3e} {'ok' if good else 'FAIL'}{note}")
    _line("G-G1", ok, f"{sum(r['pass'] for r in rows)}/{len(rows)} (dial, theta) "
          f"checks, worst rel {worst:.3e}", f"<= {G1_TOL:g}")
    print(f"  [informational] at the SCOPE's step h = max(1e-5*sigma, 1e-7): "
          f"{sum(r['pass_spec_step'] for r in rows)}/{len(rows)}, worst rel "
          f"{worst_spec:.3e} — the misses are FD roundoff, see `step_ladder`")
    return {"pass": bool(ok), "worst_rel": worst, "n_checks": len(rows),
            "n_theta_points": len(thetas), "per_dial_worst": per_dial_worst,
            "spec_step_pass": bool(ok_spec), "spec_step_worst_rel": worst_spec,
            "spec_step_n_pass": int(sum(r["pass_spec_step"] for r in rows)),
            "rows": rows}


def gate_g1j(eng, phi, names, nominal, sigma, seed=20260819):
    """G-G1J — the same adjoint core against a BETTER-CONDITIONED finite difference.

    `model_jacobian_dot(phi, theta, v)` returns [sum_b v_b dE_b/dx_d]_d, and
    `chi2_and_grad` is exactly that with v = resid. FD-ing the scalar
    `sum_b v_b E_b` instead of chi2 drops the magnitude being differenced from
    O(5e4) to O(1e2), so the roundoff floor drops with it and the analytic
    gradient can be checked without fighting the differencing noise. It also
    exercises the API that G-G2c's muon invariant is stated on.
    """
    rng = np.random.default_rng(seed)
    lo, hi = make_bounds(nominal, sigma)
    th = rng.uniform(lo, hi)
    th[names.index("E_shift")] = 1.0
    v = rng.normal(size=eng.n_bins)
    j_ana = eng.model_jacobian_dot(phi, th, v)
    scale = float(np.abs(v @ eng.expectation(phi, th)))

    rows, worst, ok = [], 0.0, True
    for name in names:
        if name == "E_shift":
            continue
        k = names.index(name)
        s = sigma[k]
        h = float(min(max(1e-5 * s, np.finfo(float).eps * max(scale, 1.0)
                          / FD_NOISE_TARGET), 0.1 * s))
        tp, tm = th.copy(), th.copy()
        tp[k] += h
        tm[k] -= h
        fd = float((v @ eng.expectation(phi, tp) - v @ eng.expectation(phi, tm))
                   / (2.0 * h))
        err = abs(j_ana[k] - fd) / max(1.0, abs(fd))
        good = err <= G1_TOL
        ok &= good
        worst = max(worst, err)
        rows.append({"dial": name, "j_analytic": float(j_ana[k]), "j_fd": fd,
                     "rel_err": err, "pass": good, "h": h})
        print(f"  G-G1J {name:26s} ana {j_ana[k]: .10e} fd {fd: .10e} "
              f"rel {err:.3e} {'ok' if good else 'FAIL'}")
    _line("G-G1J", ok, f"{sum(r['pass'] for r in rows)}/{len(rows)} dials, worst "
          f"rel {worst:.3e} (dE/dx, |sum v.E| ~ {scale:.3e})", f"<= {G1_TOL:g}")
    return {"pass": bool(ok), "worst_rel": worst, "n_checks": len(rows),
            "vE_scale": scale, "rows": rows}


def gate_g2(eng, eng_mu0, phi, names, nominal, sigma, seed=20260818):
    rng = np.random.default_rng(seed)
    lo, hi = make_bounds(nominal, sigma)
    i_es = names.index("E_shift")
    th = rng.uniform(lo, hi)
    th[i_es] = 1.0
    _c, g = eng.chi2_and_grad(phi, th)

    out = {"inert": {}, "eshift": None, "muon": None}
    ok = True
    for name in INERT_DIALS:
        v = float(g[names.index(name)])
        good = (v == 0.0)                       # BITWISE, not < tol
        ok &= good
        out["inert"][name] = {"grad": v, "pass": good}
        print(f"  G-G2 inert {name:26s} grad {v!r} {'ok' if good else 'FAIL'}")
    v_es = float(g[i_es])
    good_es = (v_es == 0.0)
    ok &= good_es
    out["eshift"] = {"grad": v_es, "pass": good_es}
    print(f"  G-G2 pinned {'E_shift':26s} grad {v_es!r} {'ok' if good_es else 'FAIL'}")

    # (c) muon independence, stated on the model Jacobian (see module docstring).
    v = rng.normal(size=eng.n_bins)
    j_mu = eng.model_jacobian_dot(phi, th, v)
    j_0 = eng_mu0.model_jacobian_dot(phi, th, v)
    good_mu = bool(np.array_equal(j_mu, j_0))
    ok &= good_mu
    out["muon"] = {"bitwise_equal": good_mu,
                   "max_abs_diff": float(np.max(np.abs(j_mu - j_0)))}
    print(f"  G-G2 muon  dE/dx bitwise mu900-independent: {good_mu} "
          f"(max |diff| {np.max(np.abs(j_mu - j_0)):.1e})")

    n_ok = sum(d["pass"] for d in out["inert"].values()) + int(good_es) + int(good_mu)
    _line("G-G2", ok, f"{n_ok}/{len(INERT_DIALS) + 2} bitwise invariants "
          f"({len(INERT_DIALS)} inert dials + E_shift + muon)", "all == 0.0 / bitwise")
    out["pass"] = bool(ok)
    return out


def gate_local_identity(response, mc, data, xml, pynu_root, names, nominal, sigma,
                        n_theta=6, seed=20260817):
    """G-ORCA-1L — the LOCAL surrogate of the FASRC identity gate.

    G-ORCA-1 compares the engine against production's
    `poisson_chi2(obs[few], orca_binned_expectation(...)[few])`, whose model is

        bincount(bin_idx, w = PhysicsWeight * NuisanceWeight * BaseWeight) + mu900

    (`orca_exact_scan.binned_expectation`). Exactly one term of that needs
    nuSQuIDS: PhysicsWeight. Everything else — the row order, `bin_idx`,
    BaseWeight = weight x NORM, and the dial composition through the LIVE tunes —
    is available locally off the parquet.

    So this gate rebuilds the production formula verbatim, per event, over all
    592,099 rows, substituting a synthetic PhysicsWeight that is cell-constant BY
    CONSTRUCTION — which is precisely the property G-ORCA-0 asserts of the real
    one. It therefore exercises, locally, every part of the port that G-ORCA-1
    exercises except the oscillation call itself: NORM, the (ntype, flavor, ie,
    iz) phi convention, the class/cell indexing, the COO contraction, the
    per-bin D_b morphology factor, the muon block, and the full 30-dial product.

    NuisanceWeight is composed as `PyNuFit.ApplyWeights` composes it — start at
    1, multiply by each dial's tune output (`Experiment.UpdateNuisanceWeights`,
    `Experiment.py:228-229`; `PyNuFit.py:626-648`). ApplyWeights walks the XML
    source blocks while this walks manifest order; multiplication reorders only
    the last bits, which is well inside the 1e-9 criterion.
    """
    import pandas as pd
    sys.path.insert(0, os.path.abspath(pynu_root))
    from pynu.PhysicsTunes.Flux.AtmoFlux import AtmosphericFlux
    from pynu.PhysicsTunes.CrossSection.WaterXSection import WaterXSection
    from pynu.PhysicsTunes.Detector.ORCADetector import ORCADetector

    eng, obs, mu, few = build_local_engine(response, mc, data, names)
    ci = build_cell_index(mc)

    df = pd.read_parquet(mc)
    nu = df[df["MC_type"] == 1]
    et = nu["true_energy"].values
    nu = nu[(et >= 0) & (et < 1e5)]
    E = nu["true_energy"].values.astype(float)
    cz = np.cos(nu["true_zenith"].values.astype(float))
    pdg = nu["pdg"].values.astype(np.int64)
    cc = nu["current_type"].values.astype(np.int64)
    pid = nu["pid"].values.astype(np.int64)
    base = nu["weight"].values.astype(float) * NORM      # BaseWeight, Orca.py:183
    assert E.size == ci.n_events == eng.nnz, (E.size, ci.n_events, eng.nnz)

    exp = MockExperiment(E, cz, pdg, cc, sample=pid)
    tunes = {}
    for obj in (AtmosphericFlux(), WaterXSection(), ORCADetector()):
        for n in names:
            if hasattr(obj, n) and n not in tunes:
                tunes[n] = obj
    missing = [n for n in names if n not in tunes]
    assert not missing, f"no live tune for {missing}"

    rng = np.random.default_rng(seed)
    phi = rng.lognormal(0.0, 0.8, size=(2, 3, eng.n_etrue, eng.n_cztrue))
    phi *= float(eng.obs.sum()) / float(
        eng.expectation(phi, nominal).sum() - eng.mu.sum())
    pw = phi[ci.ntype, ci.flavor, ci.ie, ci.iz]          # cell-constant by design

    lo, hi = make_bounds(nominal, sigma)
    i_es = names.index("E_shift")
    thetas = [np.array(nominal, float)]
    for _ in range(n_theta - 1):
        t = rng.uniform(lo, hi)
        t[i_es] = 1.0
        thetas.append(t)

    rows, worst_c, worst_m, ok = [], 0.0, 0.0, True
    for it, th in enumerate(thetas):
        nuis = np.ones(exp.NumberOfEvents)               # StartNuisanceWeights
        for k, n in enumerate(names):
            nuis = nuis * np.asarray(getattr(tunes[n], n)(exp, float(th[k])),
                                     float)
        ref_model = np.bincount(ci.bin900, weights=pw * nuis * base,
                                minlength=eng.n_bins) + mu
        ref_chi2 = float(poisson_chi2(obs[few], ref_model[few]))
        eng_model = eng.expectation(phi, th)
        eng_chi2 = eng.chi2(phi, th)
        rc = abs(eng_chi2 - ref_chi2) / max(abs(ref_chi2), 1.0)
        rm = float(np.max(np.abs(eng_model - ref_model)
                          / np.maximum(1e-30, np.abs(ref_model))))
        good = (rc <= LOCAL_IDENTITY_TOL) and (rm <= LOCAL_IDENTITY_TOL)
        ok &= good
        worst_c, worst_m = max(worst_c, rc), max(worst_m, rm)
        rows.append({"theta_point": it, "chi2_ref": ref_chi2, "chi2_engine": eng_chi2,
                     "rel_chi2": rc, "max_rel_bin": rm, "pass": good})
        print(f"  G-ORCA-1L t{it}: ref {ref_chi2:.10f} eng {eng_chi2:.10f} "
              f"rel {rc:.3e} maxbin {rm:.3e} {'ok' if good else 'FAIL'}")
    _line("G-ORCA-1L", ok, f"{sum(r['pass'] for r in rows)}/{len(rows)} theta points, "
          f"worst rel chi2 {worst_c:.3e}, worst per-bin {worst_m:.3e}",
          f"<= {LOCAL_IDENTITY_TOL:g}")
    return {"pass": bool(ok), "worst_rel_chi2": worst_c, "worst_rel_bin": worst_m,
            "n_events": int(E.size), "rows": rows}


def gate_guards(response, response_v1, mc, data, names, nominal):
    """G-GUARDS — the load/eval guards for risks R1, R3, R4 must actually FIRE.

    A guard that is never exercised is a comment. Each of these is the sole
    defence against a failure mode that would otherwise be SILENT:
      R4  a stale v1 300-bin response loading into the flat900 engine
      R3  a moved E_shift being ignored (the response encodes ENERGY_SCALE = 1)
      R1  oscillation averaging on, which breaks cell-constant PhysicsWeight
      --  a Barlow-Beeston request (the arm is pure Poisson by convention)
    """
    checks = {}

    if response_v1 and os.path.exists(response_v1):
        try:
            build_local_engine(response_v1, mc, data, names)
            checks["R4_stale_v1_rejected"] = False
        except AssertionError as e:
            checks["R4_stale_v1_rejected"] = "schema" in str(e) or "n_bins" in str(e)
    else:
        checks["R4_stale_v1_rejected"] = None

    eng, *_ = build_local_engine(response, mc, data, names)
    th = np.array(nominal, float)
    th[names.index("E_shift")] = 1.05
    try:
        eng.chi2(np.ones((2, 3, eng.n_etrue, eng.n_cztrue)), th)
        checks["R3_moved_Eshift_rejected"] = False
    except AssertionError:
        checks["R3_moved_Eshift_rejected"] = True

    class _Osc:
        osc_avg_scale = 4.0
    try:
        build_local_engine(response, mc, data, names)[0].__class__(
            response, np.zeros(900), np.zeros(900), np.ones(900, bool), names,
            norm=NORM, osc=_Osc())
        checks["R1_osc_averaging_rejected"] = False
    except AssertionError:
        checks["R1_osc_averaging_rejected"] = True

    try:
        eng.__class__(response, np.zeros(900), np.zeros(900), np.ones(900, bool),
                      names, norm=NORM, likelihood="barlow_beeston")
        checks["BB_likelihood_rejected"] = False
    except ValueError:
        checks["BB_likelihood_rejected"] = True

    live = {k: v for k, v in checks.items() if v is not None}
    ok = all(live.values())
    for k, v in checks.items():
        print(f"  G-GUARDS {k}: {v}")
    _line("G-GUARDS", ok, f"{sum(bool(v) for v in live.values())}/{len(live)} guards "
          "fired", "all fire")
    return {"pass": bool(ok), **checks}


def gate_stat_only(eng, phi, nominal):
    """Risk R2 — chi2 must be the bare Poisson term, with no prior anywhere."""
    E = eng.expectation(phi, nominal)
    want = float(poisson_chi2(eng.obs[eng.few], E[eng.few]))
    got = eng.chi2(phi, nominal)
    good = (got == want)                       # exact: same kernel, same inputs
    # A prior would be Sum((theta-nom)/sigma)^2 >= 0 and would show up as a
    # nonzero offset the moment theta leaves nominal, so probe there too.
    th = np.array(nominal, float)
    j = eng.idx["tilt"]
    th[j] += 0.05
    E2 = eng.expectation(phi, th)
    want2 = float(poisson_chi2(eng.obs[eng.few], E2[eng.few]))
    got2 = eng.chi2(phi, th)
    good &= (got2 == want2)
    _line("G-STAT-ONLY", good,
          f"chi2 - poisson_chi2 = {got - want:.1e} at nominal, "
          f"{got2 - want2:.1e} off-nominal", "== 0.0 exactly (no prior term)")
    return {"pass": bool(good), "chi2_nominal": got, "poisson_only_nominal": want,
            "chi2_offnominal": got2, "poisson_only_offnominal": want2}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--response", required=True,
                    help="orca_response_flat900.npz — BUILD PRODUCT of "
                         "analysis/ORCA-binned-datafit/build_orca_binned_response.py")
    ap.add_argument("--mc", default=os.path.join(
        _REPO, "data", "ORCA", "ORCA_MC_dataverse_with_muons.parquet"),
        help="ORCA MC parquet (default: data/ORCA/ORCA_MC_dataverse_with_muons.parquet)")
    ap.add_argument("--data", default=os.path.join(
        _REPO, "data", "ORCA", "ORCA_data_dataverse.parquet"),
        help="ORCA data parquet (default: data/ORCA/ORCA_data_dataverse.parquet)")
    ap.add_argument("--xml", required=True,
                    help="ORCA analysis XML (ORCA_Atm_r2_fude_ccqe.xml); no "
                         "repo-relative default — it is not a repo file")
    ap.add_argument("--pynu-root", default=_REPO,
                    help="repo root providing `pynu` (default: this clone)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--n-theta", type=int, default=5)
    ap.add_argument("--response-v1", default=None,
                    help="the stale 300-bin v1 npz, for the R4 rejection guard")
    ap.add_argument("--only", default=None,
                    choices=["dials", "g1", "g2", "stat", "guards",
                             "local-identity"])
    a = ap.parse_args()

    res = {"response": os.path.abspath(a.response), "seed": a.seed, "gates": {}}
    names, nominal, sigma = read_manifest(a.xml)
    assert names == ORCA_MANIFEST, (
        f"XML manifest order != ORCA_MANIFEST\n xml: {names}\n eng: {ORCA_MANIFEST}")

    if a.only in (None, "dials"):
        print("=== dial-field unit tests vs the LIVE tunes ===")
        res["gates"]["G-DIALS"] = gate_dial_fields(a.pynu_root, a.xml, seed=a.seed)

    if a.only in (None, "g1", "g2", "stat", "guards", "local-identity"):
        eng, obs, mu, few = build_local_engine(a.response, a.mc, a.data, names)
        eng0, *_ = build_local_engine(a.response, a.mc, a.data, names, mu_zero=True)
        print("=== engine ===")
        for k, v in eng.summary().items():
            print(f"  {k}: {v}")
        phi, _ = synthetic_phi(eng, seed=a.seed)
        phi = scale_phi(eng, phi, nominal)
        res["engine"] = eng.summary()
        res["engine"]["obs_sum"] = float(obs.sum())
        res["engine"]["mu_sum"] = float(mu.sum())
        res["engine"]["n_few"] = int(few.sum())

        if a.only in (None, "guards"):
            print("=== G-GUARDS load/eval guards (risks R1, R3, R4) ===")
            res["gates"]["G-GUARDS"] = gate_guards(
                a.response, a.response_v1, a.mc, a.data, names, nominal)
        if a.only in (None, "local-identity"):
            print("=== G-ORCA-1L local surrogate of the FASRC identity gate ===")
            res["gates"]["G-ORCA-1L"] = gate_local_identity(
                a.response, a.mc, a.data, a.xml, a.pynu_root, names, nominal,
                sigma, seed=a.seed)
        if a.only in (None, "stat"):
            print("=== stat-only contract (risk R2) ===")
            res["gates"]["G-STAT-ONLY"] = gate_stat_only(eng, phi, nominal)
        if a.only in (None, "g1"):
            print("=== G-G1 per-dial analytic vs central FD ===")
            res["gates"]["G-G1"] = gate_g1(eng, phi, names, nominal, sigma,
                                           n_theta=a.n_theta, seed=a.seed)
            print("=== G-G1J model-Jacobian FD (better conditioned) ===")
            res["gates"]["G-G1J"] = gate_g1j(eng, phi, names, nominal, sigma)
        if a.only in (None, "g2"):
            print("=== G-G2 zero-gradient invariants (bitwise) ===")
            res["gates"]["G-G2"] = gate_g2(eng, eng0, phi, names, nominal, sigma)

    ok = all(g["pass"] for g in res["gates"].values())
    res["overall"] = ok
    print(f"GATE G-GRAD-ALL: {'PASS' if ok else 'FAIL'} "
          f"{sum(g['pass'] for g in res['gates'].values())}/{len(res['gates'])} "
          "(threshold all PASS)")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=2, default=str)
        print(f"artifact: {os.path.abspath(a.out)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
