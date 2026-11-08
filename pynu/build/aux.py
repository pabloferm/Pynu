"""IC and ORCA auxiliary arrays, so the fit stage needs no event framework.

ic_aux.npz
  obs (200,)   ic_response.observed_200(IC_data.parquet)
  mu  (200,)   ic_response.muon_200(IC_MC.parquet)
  norm         ICDeepCore exp.NORM (FitExposure x seconds/year) from the IC XML
  min_entries  exp.MIN_ENTRIES (0.01)
  hs_dm31_grid, hs_<cat>_<key> (n_dm, 200) for cat in HS_CATEGORIES and key in
               intercept + HS_SLOPE_NAMES, as ICDeepCore.load_hypersurfaces stores
               them; hs_categories, hs_slope_names (their order)
  response_md5 md5 of the IC response the aux was built against
orca_aux.npz
  obs (900,)   orca_response.observed_900(ORCA_data_dataverse.parquet)
  mu  (900,)   orca_response.muon_900(ORCA_MC_dataverse_with_muons.parquet)
  norm         Orca exp.NORM (FitExposure x 1e4)
  few (900,)   exp.FewEntries after PyNuFit construction from the ORCA XML (expectation
               at the XML's oscillation point > MIN_ENTRIES 0.01; 430 bins True)
  min_entries  exp.MIN_ENTRIES (0.01)
  response_md5 md5 of the ORCA response
"""
import json

import numpy as np

from .events import load_pynufit, md5sum


def build_ic_aux(event_xml, ic_mc, ic_data, data_dir, ic_response, out_path):
    from .ic_response import observed_200, muon_200

    obs = observed_200(ic_data, data_dir)
    mu = muon_200(ic_mc, data_dir)
    _, _, exp, _ = load_pynufit(event_xml, {"IC_MC": ic_mc})
    exp.load_hypersurfaces(data_dir)
    cats = list(exp.HS_CATEGORIES.keys())
    keys = ["intercept"] + list(exp.HS_SLOPE_NAMES)
    hs = {f"hs_{c}_{k}": np.asarray(exp._hs_data[c][k], float) for c in cats for k in keys}
    out = dict(obs=obs, mu=mu, norm=np.float64(exp.NORM),
               min_entries=np.float64(exp.MIN_ENTRIES),
               hs_dm31_grid=np.asarray(exp._hs_dm31_grid, float),
               hs_categories=np.array(cats), hs_slope_names=np.array(exp.HS_SLOPE_NAMES),
               response_md5=np.array(md5sum(ic_response)),
               meta=json.dumps(dict(hs_categories_files=dict(exp.HS_CATEGORIES),
                                    hs_nominals=dict(exp.HS_NOMINALS))),
               **hs)
    np.savez_compressed(out_path, **out)
    print(f"[aux/ic] obs sum {obs.sum():.1f}  mu sum {mu.sum():.4f}  NORM {exp.NORM:.6e}  "
          f"MIN_ENTRIES {exp.MIN_ENTRIES}  HS {len(cats)} cats x {len(keys)} keys x "
          f"{np.asarray(exp._hs_dm31_grid).size} dm31 -> {out_path}")
    return out_path


def build_orca_aux(event_xml, orca_mc, orca_data, orca_response, out_path):
    from .orca_response import observed_900, muon_900

    obs = observed_900(orca_data)
    mu = muon_900(orca_mc)
    _, _, exp, _ = load_pynufit(event_xml, {"ORCA_MC": orca_mc})
    few = np.asarray(exp.FewEntries, bool)
    np.savez_compressed(out_path, obs=obs, mu=mu, norm=np.float64(exp.NORM), few=few,
                        min_entries=np.float64(exp.MIN_ENTRIES),
                        response_md5=np.array(md5sum(orca_response)))
    print(f"[aux/orca] obs sum {obs.sum():.1f}  mu sum {mu.sum():.4f}  NORM {exp.NORM:.6e}  "
          f"few {int(few.sum())}/{few.size} True -> {out_path}")
    return out_path
