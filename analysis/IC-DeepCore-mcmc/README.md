# IC DeepCore 9-year release: data fit, NC/CC bin study, MCMC

Branch `datafit-IC-mcmc` (albaburgos/Pynu), built on `datafit-SK-mcmc` @ `83374c9`.
Goal: (0) reproduce the PRD 108, 012014 νμ-disappearance fit with the Pynu event
engine on the public release; (1) extend the PID axis from {mixed, tracks} to
{NC-like, CC-like, mixed, tracks} and quantify, as an Asimov study, what a NC/CC
cascade classifier would add; (2) MCMC posterior in (sin²θ23, Δm²32).

Everything here runs on a laptop-class machine (nuSQuIDS from PyPI + the nuSQuIDS
`data/` directory at `/project/share/nuSQuIDS/`, nuflux, iminuit); grid scans and
long chains go to the cluster.

## Files

- `icdc_fit.py` — Stage 0 driver. Pynu event engine (`ICDeepCore_Atm`), hypersurfaces
  re-interpolated at each Δm²31, nuisance fit per oscillation point with L-BFGS-B.
  Paper-like treatment: `hole_ice_p0/p1`, `bulk_ice_abs/scatter`, Aeff scale
  (`normalization_above1GeV`) and `muon_norm` carry no Gaussian penalty (Table IV
  "unconstrained"), explicit bounds, no clipping of negative-capable parameters.
  Statistic: `--stat bb` (Pynu Barlow–Beeston) or `--stat chi2mod` (paper Eq. 4,
  Σ(E−O)²/(E+σ²_sim) with σ²_sim = MC weight variance + muon variance).
  Modes: `point`, `profile --points file.json`, `free` (Nelder–Mead over physics).
- `../AnalysisFiles/IC_Atm_datafit_paperlike.xml` — `IC_Atm_datafit.xml` with
  `NCoverCC` and `muon_norm` enabled and repo-relative data paths.
- `points_stage0.json` — the 5-point cross used for the first validation.
- `results/stage0_profile5_bb.json`, `results/stage0_profile5_chi2mod.json` — first results (see below).

```
cd analysis/IC-DeepCore-mcmc
python icdc_fit.py --config ../AnalysisFiles/IC_Atm_datafit_paperlike.xml \
    --mode profile --points points_stage0.json --stat bb --out results/stage0_profile5_bb.json
```
~70 s per point on one core (propagation + 14-parameter nuisance fit).

## Stage 0 results (2026-10-18)

Table V check at the paper's best-fit point with Table IV detector values, Aeff 0.82,
μ scale 1.39, no fit: Pynu/Table V = 0.973 (νμ CC), 1.011 (νe CC), 0.978 (ντ CC),
0.854 (NC; the paper's σNC/σCC +13% not applied), 1.001 (atm μ), total 0.971.
Livetime kept at the XML's 7.74 yr (Table V's rate implies 7.46 yr; the free Aeff
scale absorbs the difference).

5-point profile, BB statistic, 14 nuisances, 200 bins:

| sin²θ23 | Δm²32 [eV²] | χ² |
|---|---|---|
| 0.51 | 2.41e-3 | 179.36 |
| 0.45 | 2.41e-3 | 180.32 |
| 0.57 | 2.41e-3 | 181.06 |
| 0.51 | 2.30e-3 | 182.38 |
| 0.51 | 2.52e-3 | 180.90 |

Parabolic minimum: sin²θ23 = 0.502 ± 0.052, Δm²32 = (2.43 ± 0.07)e-3
(paper: 0.51 ± 0.05, (2.41 ± 0.07)e-3). The same five points with the paper's
χ²_mod statistic give χ² = 180.55 at the paper point and sin²θ23 = 0.502 ± 0.053,
Δm²32 = (2.44 ± 0.07)e-3: the two statistics agree to well within the errors. Nuisances at the paper point vs Table IV:
dom_eff 1.062 (1.06), hole_ice_p0 −0.23 (−0.27), hole_ice_p1 −0.04 (−0.04),
ice absorption 0.973 (0.97), ice scattering 0.989 (0.99), μ scale 1.39 (1.39),
Aeff 0.76 (0.82; Pynu's flux dials take part of it), σNC/σCC 1.07 (1.13).

## Known deviations from the paper (to close)

- Flux systematics are Pynu's `AtmoFlux` dials (normalisation, tilt, ν/ν̄, flavour,
  zenith up/down), not the paper's Δγ + Barr meson-yield blocks.
- `MaCCQE`/`MaCCRES` are absent: `utils/data_prep/ic_dataverse_to_parquet.py` drops the
  per-event `MaCCQE_*`/`MaCCRES_*` coefficients. Fix: carry the four columns into the
  parquet and add a dial (1 + a·δ + b·δ²) in `ICDeepCoreDetector` / `WaterXSection`.
- DIS CSMS cannot be reproduced from the release (the CSMS/GENIE ratio function is not
  published); paper pull 0.04σ.
- MC variance in the BB statistic is evaluated at nominal nuisances (as in the row worker).
- `run_ic_datafit_row_worker.py` clips every positive-nominal parameter at 0.01, which
  pins `hole_ice_p0` (nominal 0.10, paper best fit −0.27); not changed there, handled here.

## Next

1. A finer 1D profile in each parameter for proper
   Δχ² = 1 intervals (cluster, or ~15 points here).
2. 4-morphology switch in `ICDeepCore.py` ({NC-like, CC-like, mixed, tracks}, default
   off), parquet `pid` remap, mixed-bin hypersurface row reused for the three sub-bins,
   muon template kept in "mixed"; closure test (all in "mixed" ≡ 2-bin result).
3. Truth-split Asimov upper bound, then the WavePID OMSim P(NC)/P(CC) thresholds.
4. MCMC (emcee first) on the binned engine; compare to the profile contours.
