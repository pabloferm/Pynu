# IC DeepCore binned engine — contract

`ic_binned_engine.ICBinnedEngine` is the sibling of `ORCABinnedEngine` under the
same contract: stat-only χ² plus an analytic dial gradient, evaluated once per
populated cell (127,757 at L3) instead of once per event (396,843), with the
whole 39-long gradient from one adjoint bincount. It replaces a finite-difference
loop over 38 dials that measured at 88.2% of a cold task.

Unlike ORCA, IC's MC is a **hybrid**: the true side is event-level (396,843
unique `(ETrue, CosZTrue)` pairs, no baked bin columns), so the true side must be
binned onto a resolution ladder and a genuine cell-centering residue exists. The
reco side is histogrammed at fit time onto the native edge files. That is why the
IC identity gate is quoted differently from ORCA's — see the tolerances below.

## Modules

| Module | Role |
|---|---|
| `ic_binned_engine.py` | the engine: cell weights, HS block, `chi2`, `chi2_and_grad` |
| `ic_binned_cells.py` | `ICCells` — sparse populated-cell structure and contraction, npz-only loader |
| `ic_dial_fields.py` | the 11 Mode-keyed IC dial forms + `ICCellGeom` |
| `ic_binned_builder.py` | parquet → ladder response `.npz` (`--mode-axis`, levels L0–L3) |
| `ic_binned_support.py` | 200-bin reco grid + `observed_200`, `muon_200`, `nu_index`, `POINTS`, `poisson_chi2`, and the HS reference model (`_hs_params_from_theta`, `_hs_correction_factor`, `_corrected_expectation`) G-IC-3 certifies the engine against |
| `binned_dial_fields.py` | shared 19 flux + 4 xsec forms, consumed FROZEN from the ORCA track |
| `binned_contract.py` | neutral COO contraction kernels |

## The model

```
E_b        = SUM over categories cat of  C_cat[b] * hist_cat[b]  +  mu_b
hist_cat[b] = SUM over entries whose cell is in cat of  entry_w * NORM * phi_cell * W_cell
W_cell      = PROD over cell dials of factor(dial, geom_cell, x)
C_cat[b]    = intercept_cat[b] + SUM over HS dials s of  slope_cat,s[b] * (x_s - nominal_s)
```

## Construction

```python
eng = ICBinnedEngine(
    response_npz   = ".../ic_response_modeaxis_L3.npz",   # see (1)
    obs200         = observed_200(data, data_dir),
    mu200          = muon_200(mc, data_dir),
    nuisance_names = list(fit.Analysis.NuisanceList),      # 39, manifest order
    norm           = FitExposure * SECONDS_PER_YEAR,       # see (4)
    hs_slopes      = exp.interpolate_hs(dm31_cell),        # see (2) — PER CELL
    pinned         = ("nunubar_ratio",))                   # see (5)

chi2         = eng.chi2(phi[ipt], theta)
chi2, grad39 = eng.chi2_and_grad(phi[ipt], theta)          # STAT-ONLY
```

## Five things the contract pins

**1. The response must be the mode-axis, no-snap build**
(`ic_response_modeaxis_L*`). The `|Mode|` class axis (47 classes) is required —
without it the 11 Mode-keyed dials are unrepresentable, and the constructor
**refuses** a 12-class response rather than producing quiet nonsense. It must
also stay *unsnapped* (nE 160 at L3) so the existing φ tensors index it;
`phi_cells` hard-fails on a shape mismatch. Build with
`ic_binned_builder.py --mode-axis --grids L3`.

**2. ★ `hs_slopes` IS Δm²-DEPENDENT AND MUST BE REFRESHED PER GRID CELL**, from
*that* cell's true grid Δm²:

```python
eng.hs_slopes = exp.interpolate_hs(dm31_cell)      # once per (i_dm, i_s23) cell
```

This is the one sharp usage edge in the whole engine. Reusing one cell's slopes
across a patch, or interpolating at a Δm² that came from anywhere other than the
grid, **silently fits against the wrong hypersurface while still converging**.
That defect invalidated the first G-IC-4 post-fit run — labels computed by an
independent centre-plus-step arithmetic were fed to `interpolate_hs` — and its
measured footprint was a per-bin `|dC|/C` up to 6.2e-3 for a one-grid-step Δm²
error. Derive `dm31_cell` and the φ row `ipt` from the SAME `(i_dm, i_s23)`:
`ipt = i_dm * grid_size + i_s23`. The engine carries a stale-slope assert; do not
route around it.

**3. The HS block is additive, not multiplicative-in-log.** The 5 hypersurface
dials are event-weight no-ops; their entire effect is the per-category linear
multiplier `C_cat[b]` above, so

```
dE_b/dx_s = SUM over cat of  slope_cat,s[b] * hist_cat[b]
```

with the slopes constant at fixed Δm². Writing them as a `d(ln W)` form would be
wrong twice over: `C_cat` can pass through zero, and the dials multiply a
category *histogram* rather than a per-event weight. The three HS categories
(`nc_nue_cc`, `numu_cc`, `nutau_cc`) are pure class masks, so they partition the
cell axis exactly.

**4. Muon is a 200-bin constant added AFTER the HS correction, with zero
gradient** (200 rows, Σw = 512.166). NORM is applied **at scan time**, not baked:
the response stores raw weight. Recover NORM from the live experiment rather than
recomputing it, so a config change cannot desync the two sides. The `few` mask is
`obs > MIN_ENTRIES` with `MIN_ENTRIES = 0.01`.

**5. STAT-ONLY, gradient in manifest order with the pinned slot present.** The
gradient is `len(nuisance_names)` long in manifest order with the pinned slot
written `0.0` — no reindexing at the call site. The driver keeps sole ownership
of the single Gaussian union prior; an arm-internal Gaussian would double-count
it silently *and* still pass an identity gate.

Two further invariants: φ is oscillated **flux** (flux x P), not bare
probability, and there is **no SK NC override** — the NC classes carry the same
flux treatment as CC (G-IC-3 is the detector for a violation, since it shows up
as an O(1) discrepancy at nominal dials). And `ccqe_shape_subgev_e_edges` stays
`None`, fixing `mu = -1.0164880658631577`; injecting the IC ladder edges would
redefine the dial.

φ is indexed by **integer cell indices only**, never float-edge matching: a
cluster rebuild reproduces `e_true_edges` to `allclose` (9.1e-13) but not
bitwise. `assert_phi_grid(phi_npz)` gives an explicit allclose check.

The engine never calls `Tune.Get`, so the coordinate-blind `@cache_method` on the
tune dispatcher cannot contaminate it — by construction, not by discipline.

## Certified tolerances

| Gate | What it compares | Result |
|---|---|---|
| **G-IC-3** | engine vs the production IC term at reference points | **PASS** at summation-order identity on both ladders: **4.688e-15 (L3)**, 4.236e-15 (L1), against a 1e-9 threshold. |
| **G-IC-4** | engine vs the live event path post-fit | **BOUNDS ≤ 0.053**: `max|Δχ²| ≤ 0.053`, spread ≤ 0.087, *both scatter-limited* at ftol 1e-5. The per-cell binning cost sits below the fit's own convergence scatter and is therefore **not resolved**. Do not propagate 0.087 as a systematic, and do not read the run-1→run-2 change 0.129→0.087 as an improvement — both are bounds. |
| **G-G1** | analytic gradient vs Richardson finite differences | worst 4.5e-08. |

Per-evaluation reference cost: **29 ms per gradient**.

## Known open item

A band-edge defect (B3) is outstanding: the 10 GeV and 1.33 GeV band thresholds
fall *inside* ladder cells rather than on edges. The builder's `--snap-e-edges`
flag exists to force thresholds onto the true-E edge set, but the **snapped v2
build deliberately does not ship** — the mode-axis unsnapped build is the
certified production response, and the B3 ruling is pending. Omit
`--snap-e-edges` for the shipped ladder.

Gate sources live in `test/binned_icorca/`. Provenance and the dev-tree source
map: `PROVENANCE_icorca_binned.md`. The ORCA sibling: `README_orca_binned.md`.
