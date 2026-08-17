# ORCA binned engine — contract

`orca_binned_engine.ORCABinnedEngine` evaluates the ORCA arm as a binned
response-matrix model with an analytic dial gradient: the dial algebra runs once
per populated true cell (17,236) instead of once per event (592,099), and the
whole 30-long gradient comes out of a single adjoint pass instead of 28
finite-difference re-evaluations.

It is **not an approximation of the event path.** ORCA's MC parquet is
intrinsically binned — exactly one distinct `ETrue` and one distinct
`cos ZTrue` per true bin over all 592,099 rows — so a dial evaluated at the
cell's true coordinate reproduces the per-event value to the same float. The
engine differs from the event path only by float summation order, which is why
its identity gate is set at the *measured* event-path noise floor and a failure
there is a bug, never "binning".

## Modules

| Module | Role |
|---|---|
| `orca_binned_engine.py` | the engine: `cell_weights`, `expectation`, `chi2`, `chi2_and_grad`, `model_jacobian_dot` |
| `orca_binned_support.py` | flat900 grid constants + `_flat900`, `observed_900`, `muon_900`, `nu_cell_index`, `poisson_chi2`, and `binned_expectation` — the production reference model G-ORCA-1/2 certify the engine against |
| `orca_cell_phi.py` | exact cell-φ extraction off a live experiment (no tensor build, no nuSQuIDS) |
| `orca_binned_builder.py` | parquet → flat900 COO response `.npz` (pure reshape/reindex; no MC pass) |
| `binned_dial_fields.py` | shared 19 flux + 4 xsec dial forms and `build_cell_geometry` (also used by IC) |
| `binned_contract.py` | neutral COO contraction kernels (also used by IC) |

## The model

```
E_b = D_b * S_b + mu_b
S_b = SUM over nnz entries in bin b of  R_v[nnz] * NORM * W_cell(nnz)
W_c = phi(cell) * PROD over the 27 cell-axis dials of factor(dial, geom_c, x)
D_b = the per-bin morphology factor: f_HPT on pid block 1, f_Shower on block 0, 1 elsewhere
```

## Six things the contract pins

**1. Schema is flat900, and it is asserted.** 900 bins =
`pid(3) x reco_E(15) x reco_cz(20)`, flattened as
`(pid * 15 + i_Ereco) * 20 + i_czreco`. `pid` lives on the *bin* axis, not the
class axis, so `obs`, `mu900` and `few` are reused verbatim from the event
pipeline and the class axis is `(pdg, current)` only — 8 populated classes. The
constructor refuses anything whose `schema_version` is not
`orca_binned_response_v2_flat900`; a stale 300-bin v1 response cannot load
silently. Build with `orca_binned_builder.py --layout flat900`.

**2. `few` is an MC-SUPPORT mask of 430 bins — not `obs > 0`.** It is
`exp.FewEntries`, which is derived from MC support, so it holds 430 of the 900
bins, including 3 bins where the observation is empty. This differs from the 427
bins a naive `obs > 0` gives, and the difference is worth +2.89 χ² at nominal —
54x the identity tolerance. Production has always used 430; pass that mask.

**3. φ is oscillated FLUX, and there is NO SK NC override.** `phi` is
`InitialFlux x P_osc` (the live `exp.PhysicsWeight`), not bare survival
probability. SK forces `P = 1` on NC classes because SK's φ *is* bare
probability; doing that here would drop the flux on every NC cell. This is the
single most likely silent factor-of-flux bug in the port, and
`cell_weights`'s docstring says so at the point of use.

**4. Cell-φ is the MC-weight-weighted mean.** Oscillation averaging must be OFF
(`osc.osc_avg_scale is None`; the constructor asserts it if handed an `osc`
object) — with averaging on, `PhysicsWeight` stops being cell-constant and the
gather silently becomes approximate. Under averaging-off, gate G-ORCA-0 measures
the within-cell spread as bit-exactly zero. Where a residual stochastic spread
does appear, `orca_cell_phi.extract_cell_phi(..., how="weighted_mean")` is the
adopted reduction: it is the unique choice that conserves each cell's total
contribution (cell-total relative error 0.000e+00 exactly, versus 1.855e-03 for
a plain mean and 3.333e-03 for groupby-first).

**5. `E_shift` is pinned at 1.0 and hard-asserted.** The precomputed `R_b`
encodes `ENERGY_SCALE = 1`, so a moved `E_shift` would be *ignored* rather than
raise. `_theta_checked` refuses any `theta` whose `E_shift` slot is not exactly
1.0, and the gradient slot is written as a literal `0.0`.

**6. The arm is STAT-ONLY: pure Poisson χ², no prior.** `chi2` and
`chi2_and_grad` return the statistical term alone. The *driver* owns the single
Gaussian union prior over the dials. An arm-internal Gaussian would double-count
the prior, move every result, and still pass an identity gate — so it is the
driver's job by contract, not by convention. Likelihood is pure Poisson (campaign
convention since 2026-06-25); the constructor rejects `likelihood != "poisson"`.

## Manifest

30 active dials in the XML order of `ORCA_Atm_r2_fude_ccqe.xml`: 19 shared flux
+ 4 shared xsec + 7 detector (`f_all`, `f_HPT`, `f_Shower`, `f_tauCC`, `f_NC`,
`f_HE`, `E_shift`). `muon_norm` is `<status> 0 </status>` and absent. 28 are
movable (`E_shift` pinned locally, `nunubar_ratio` pinned by the union vector);
`f_HPT` and `f_Shower` are bin-axis, leaving 27 cell-axis dials.

Four dials have a structurally empty mask on this MC and therefore a *bitwise*
zero gradient: `flux_nuebar_subgev`, `flux_flavor_subgev`, `flux_numubar_subgev`
and `normalization_below1GeV` — the last because ORCA's minimum true energy is
1.1220184543 GeV, so its `E < 1` mask is empty too. Zeros there are correct
results, not failures.

## Certified tolerances

| Gate | What it compares | Result |
|---|---|---|
| **G-ORCA-1** | engine expectation vs the event-path expectation on flat900 | **PASS 55/55**, worst relative 5.647e-05 = 2.58x the measured event-path noise floor (draw-to-draw χ² spread 0.0118, relative 2.2e-5). The threshold is derived distributionally, predicted max ~5.5e-05. |
| **G-ORCA-2** | analytic gradient vs finite differences, all 30 dials | **PASS 58/58** under the derivation-backed 5e-3 bound; worst ratio-to-model-difference 1.52x, O(1) for every dial as predicted. |
| **G-ORCA-0** | within-cell spread of `PhysicsWeight` | bit-exactly 0, plus a hard assert that `osc_avg_scale is None`. |

Per-evaluation reference cost: **9.4 ms per gradient**.

Gate sources live in `test/binned_icorca/`. Provenance and the dev-tree source
map: `PROVENANCE_icorca_binned.md`. The IC sibling: `README_ic_binned.md`.
