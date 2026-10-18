#!/usr/bin/env python3
"""IC DeepCore release fit with the Pynu event engine — Stage 0 driver.

Differences from analysis/IC-DeepCore-datafit/run_ic_datafit_row_worker.py:
  * parameters listed in UNCONSTRAINED get no Gaussian penalty (paper Table IV:
    hole_ice_p0/p1, ice absorption/scattering, Aeff scale, atmospheric-mu scale);
  * explicit bounds per parameter, no clipping of negative-capable parameters at 0.01;
  * statistic selectable: 'bb' (Pynu Barlow-Beeston) or 'chi2mod' (paper Eq. 4:
    sum (E-O)^2/(E + sigma_sim^2) with sigma_sim^2 = MC weight variance + muon variance);
  * modes: point (nuisance fit at one oscillation point), free (Nelder-Mead over
    (sin2th23, dm2_31) with inner nuisance fit), profile (list of points);
  * results to JSON.

Usage examples:
  python icdc_fit.py --config IC_paper.xml --mode point --s23 0.51 --dm32 2.41e-3
  python icdc_fit.py --config IC_paper.xml --mode free --stat chi2mod
  python icdc_fit.py --config IC_paper.xml --mode profile --points points.json
"""
import os, sys, json, time, argparse
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
PYNU_DIR = os.environ.get("PYNU_DIR", os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, PYNU_DIR)
from pynu import PyNuFit
from pynu.Experiments.ICDeepCore import ICDeepCore_Atm

DM21 = 7.41e-5          # eV^2, NO: dm2_31 = dm2_32 + dm2_21
UNCONSTRAINED = {"hole_ice_p0", "hole_ice_p1", "bulk_ice_abs", "bulk_ice_scatter",
                 "normalization_above1GeV", "muon_norm"}
# explicit bounds for parameters whose +-5 sigma range is not the right box
BOUNDS = {"hole_ice_p0": (-1.0, 1.0), "hole_ice_p1": (-0.5, 0.5),
          "bulk_ice_abs": (0.5, 1.5), "bulk_ice_scatter": (0.5, 1.5),
          "normalization_above1GeV": (0.3, 3.0), "muon_norm": (0.05, 5.0),
          "NCoverCC": (0.2, 3.0), "XSecNuTau": (0.2, 3.0)}


def resolve_config(config):
    """Rewrite '../../data/...' style paths in the XML to absolute paths under PYNU_DIR
    (so the config works from any working directory). Returns a path to the XML to use."""
    import re, tempfile
    txt = open(config).read()
    def fix(m):
        p = m.group(2)
        if os.path.isabs(p) or os.path.exists(p):
            return m.group(0)
        cand = os.path.abspath(os.path.join(PYNU_DIR, p.replace("../../", "", 1)))
        return m.group(1) + cand + m.group(3)
    new = re.sub(r"((?:MCFiles|DataFiles) name=')([^']+)(')", fix, txt)
    if new == txt:
        return config
    tmp = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
    tmp.write(new); tmp.close()
    return tmp.name


class ICFit:
    def __init__(self, config, hs_dir, stat="bb"):
        self.stat = stat
        self.fit = PyNuFit(resolve_config(config), verbosity=False)
        self.exp = [e for e in self.fit.Experiments.values() if isinstance(e, ICDeepCore_Atm)][0]
        self.exp_name = [k for k, e in self.fit.Experiments.items() if e is self.exp][0]
        self.exp.load_hypersurfaces(hs_dir)
        A = self.fit.Analysis
        self.names = list(A.NuisanceList)
        self.nominal = np.array(A.NuisNominalList, float)
        self.sigma = np.array(A.NuisSigmaList, float)
        self.penalised = np.array([n not in UNCONSTRAINED for n in self.names])
        self.bounds = []
        for n, m, s in zip(self.names, self.nominal, self.sigma):
            if n in BOUNDS:
                self.bounds.append(BOUNDS[n])
            else:
                lo, hi = m - 5 * s, m + 5 * s
                if m > 0 and lo < 0.01 and n not in UNCONSTRAINED:
                    lo = 0.01
                self.bounds.append((lo, hi))
        self.hs_idx = {n: self.names.index(n) for n in self.exp.HS_SLOPE_NAMES if n in self.names}
        self.mu_idx = self.names.index("muon_norm") if "muon_norm" in self.names else None
        # nominal expectation, data observation, MC variance, muon template
        self._set_osc(0.51, 2.41e-3 + DM21)
        self.fit.ApplyNuisanceWeights(self.fit.Analysis.NuisNominalList)
        self.fit.SetExpectedWeights()
        self.fit.SetBinnedExpectedEvents()
        self.fit.SetBinnedMCVariance()
        self.fit.SetMuonBackground()
        self.exp.SetObservedBinned()
        self.obs = self.exp.GetObservedBinned()
        self.fit.Observation = {self.exp_name: self.obs}
        self.fit.set_likelihood("BarlowBeestonLikelihood")
        self.fit.LLH.observation[self.exp_name] = self.obs
        self.fit.LLH.set_muon_background(self.fit.MuonBackground)
        self.fit.LLH.set_mc_variance(self.fit.MCVariance)
        if self.mu_idx is not None:
            self.fit.LLH.set_muon_norm_index(self.mu_idx)
        self.mask = self.exp.FewEntries
        mu_counts, mu_var = self.exp.GetMuonBackground()
        self.mu_counts = mu_counts[self.mask] if mu_counts is not None else np.zeros(self.obs.size)
        self.mu_var = mu_var[self.mask] if mu_var is not None else np.zeros(self.obs.size)
        self.n_data = float(np.sum(self.obs))
        self._last_dm31 = None

    # ---- oscillations
    def _set_osc(self, s23, dm31):
        self.fit.StartPhysics()
        self.fit.StartNuisance()
        for pt in self.fit.physics_tunes.values():
            P = pt.OscillationTunes.Parameters
            P["Sin2Theta23"], P["Dm231"], P["Dm231_bar"] = s23, dm31, dm31
            if hasattr(pt.OscillationTunes, "reset_cache"):
                pt.OscillationTunes.reset_cache()
        self.fit.ApplyOscillations("Physics")
        self._last_dm31 = dm31

    # ---- expectation for a nuisance vector (neutrinos only, HS-corrected, masked)
    def expectation(self, nuis):
        self.fit.StartNuisance()
        self.fit.ApplyNuisanceWeights(nuis)
        self.fit.SetExpectedWeights()
        hs = {n: nuis[i] for n, i in self.hs_idx.items()}
        return self.exp.apply_hs_correction(self._last_dm31, hs)

    def mc_variance(self):
        v = self.fit.MCVariance.get(self.exp_name)
        return np.asarray(v) if v is not None else np.zeros(self.obs.size)

    def objective(self, nuis):
        E_nu = self.expectation(nuis)
        mu_scale = nuis[self.mu_idx] if self.mu_idx is not None else 1.0
        if self.stat == "bb":
            stat = self.fit.LLH.stats_only({self.exp_name: E_nu}, self.fit.MCVariance, mu_scale)
        else:  # paper Eq. (4)
            E = E_nu + mu_scale * self.mu_counts
            var_sim = self.mc_variance() + mu_scale ** 2 * self.mu_var
            denom = E + var_sim
            ok = denom > 0
            stat = float(np.sum((E[ok] - self.obs[ok]) ** 2 / denom[ok]))
        pen = float(np.sum((((nuis - self.nominal) / self.sigma) ** 2)[self.penalised]))
        return stat + pen

    def fit_point(self, s23, dm31, x0=None):
        self._set_osc(s23, dm31)
        x0 = self.nominal if x0 is None else np.asarray(x0)
        t0 = time.time()
        r = minimize(self.objective, x0, method="L-BFGS-B", bounds=self.bounds,
                     options={"ftol": 1e-6, "gtol": 1e-5, "maxiter": 300})
        return {"s23": s23, "dm31": dm31, "dm32": dm31 - DM21, "chi2": float(r.fun),
                "nuisance": r.x.tolist(), "pulls": ((r.x - self.nominal) / self.sigma).tolist(),
                "nit": int(r.nit), "converged": bool(r.success), "time_s": time.time() - t0}

    def components(self, nuis):
        """Per-component totals for the current oscillation point and nuisance vector."""
        self.fit.StartNuisance(); self.fit.ApplyNuisanceWeights(nuis); self.fit.SetExpectedWeights()
        hs = {n: nuis[i] for n, i in self.hs_idx.items()}
        slopes = self.exp.interpolate_hs(self._last_dm31)
        corr = {}
        for cat in self.exp.HS_CATEGORIES:
            c = slopes[cat]["intercept"].copy()
            for s in self.exp.HS_SLOPE_NAMES:
                c += slopes[cat][s] * (hs[s] - self.exp.HS_NOMINALS[s])
            corr[cat] = c
        w = self.exp.ExpectedWeight * self.exp.BaseWeight
        pdg, cc = np.abs(self.exp.nuPDG), self.exp.CC
        def binned(mask):
            parts = []
            for m in self.exp.Samples:
                sm = mask & (self.exp.Sample == m)
                h, _, _ = np.histogram2d(self.exp.EReco[sm], self.exp.CosZReco[sm],
                                         bins=[self.exp.EnergyBins[m], self.exp.CTBins[m]], weights=w[sm])
                parts.append(h.flatten())
            return np.concatenate(parts)
        mu_scale = nuis[self.mu_idx] if self.mu_idx is not None else 1.0
        out = {"numu_cc": float(np.sum(binned((pdg == 14) & (cc == 1)) * corr["numu_cc"])),
               "nue_cc": float(np.sum(binned((pdg == 12) & (cc == 1)) * corr["nc_nue_cc"])),
               "nutau_cc": float(np.sum(binned((pdg == 16) & (cc == 1)) * corr["nutau_cc"])),
               "nc": float(np.sum(binned(cc == 0) * corr["nc_nue_cc"])),
               "mu": float(mu_scale * np.sum(self.mu_counts))}
        out["total"] = sum(out.values())
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--hs-dir", default=os.path.join(PYNU_DIR, "data", "IceCube"))
    ap.add_argument("--stat", choices=["bb", "chi2mod"], default="bb")
    ap.add_argument("--mode", choices=["point", "free", "profile"], default="point")
    ap.add_argument("--s23", type=float, default=0.51)
    ap.add_argument("--dm32", type=float, default=2.41e-3)
    ap.add_argument("--points", help="JSON list of [s23, dm32] for profile mode")
    ap.add_argument("--maxfev", type=int, default=40)
    ap.add_argument("--out", default="icdc_fit_result.json")
    a = ap.parse_args()

    F = ICFit(a.config, a.hs_dir, a.stat)
    print(f"nuisances ({len(F.names)}): {F.names}")
    print(f"unconstrained: {[n for n in F.names if n in UNCONSTRAINED]}")
    print(f"data events {F.n_data:.0f}, muon template {np.sum(F.mu_counts):.1f}, stat={a.stat}")
    res = {"config": a.config, "stat": a.stat, "names": F.names,
           "nominal": F.nominal.tolist(), "sigma": F.sigma.tolist()}

    if a.mode == "point":
        r = F.fit_point(a.s23, a.dm32 + DM21)
        r["components"] = F.components(np.array(r["nuisance"]))
        res["point"] = r
        print(f"\nchi2={r['chi2']:.2f} nit={r['nit']} conv={r['converged']} ({r['time_s']:.0f}s)")
        for n, v, p in zip(F.names, r["nuisance"], r["pulls"]):
            print(f"  {n:26s} {v:9.4f}  pull {p:+6.2f}")
        print("  components:", {k: round(v) for k, v in r["components"].items()})

    elif a.mode == "profile":
        pts = json.load(open(a.points))
        res["profile"] = []
        x0 = None
        for s23, dm32 in pts:
            r = F.fit_point(s23, dm32 + DM21, x0)
            x0 = r["nuisance"]
            res["profile"].append(r)
            print(f"  s23={s23:.4f} dm32={dm32:.5e} chi2={r['chi2']:.3f} conv={r['converged']}", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)

    else:  # free
        hist, x0 = [], [None]
        def outer(p):
            s23, dm32 = p
            if not (0.3 < s23 < 0.7 and 2.0e-3 < dm32 < 3.0e-3):
                return 1e6
            r = F.fit_point(s23, dm32 + DM21, x0[0])
            x0[0] = r["nuisance"]
            hist.append(r)
            print(f"  s23={s23:.4f} dm32={dm32:.5e} chi2={r['chi2']:.3f}", flush=True)
            json.dump({**res, "history": hist}, open(a.out, "w"), indent=1)
            return r["chi2"]
        d0 = a.dm32
        r = minimize(outer, [a.s23, d0], method="Nelder-Mead",
                     options={"xatol": 2e-3, "fatol": 0.02, "maxfev": a.maxfev,
                              "initial_simplex": [[a.s23, d0], [a.s23 + 0.05, d0], [a.s23, d0 + 1.5e-4]]})
        best = min(hist, key=lambda h: h["chi2"])
        best["components"] = None
        F._set_osc(best["s23"], best["dm31"])
        best["components"] = F.components(np.array(best["nuisance"]))
        res.update({"best": best, "history": hist})
        print(f"\nbest: s23={best['s23']:.4f} dm2_32={best['dm32']:.5e} chi2={best['chi2']:.3f} "
              f"(paper 0.51+-0.05, 2.41e-3+-0.07e-3)")
        for n, v, p in zip(F.names, best["nuisance"], best["pulls"]):
            print(f"  {n:26s} {v:9.4f}  pull {p:+6.2f}")
        print("  components:", {k: round(v) for k, v in best["components"].items()})

    json.dump(res, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
