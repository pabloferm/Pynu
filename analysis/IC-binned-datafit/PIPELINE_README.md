# IC DeepCore binned-engine pipeline

End-to-end recipe for the single-experiment IC DeepCore binned fit: build the
response matrix, build the oscillation tensors, run the scan, verify.

The engine (`pynu/Experiments/ic_binned_engine.py`) evaluates every dial once per
**populated true cell** (127,757 at L3) instead of once per event (396,843), and
gets the whole 39-long gradient from one adjoint bincount.

---

## ★ The rule that will bite you

**The hypersurface slopes are Δm²-dependent and must be refreshed per grid cell,
from that cell's grid Δm²:**

```python
eng.set_hs_slopes(exp.interpolate_hs(dm_cell), dm_cell)
```

Reusing one cell's slopes across a patch — or interpolating at a Δm² that came
from anywhere other than the grid — silently fits against the **wrong**
hypersurface while converging cleanly. No error, no warning. That defect
invalidated the first G-IC-4 postfit run; the measured footprint of a *one grid
step* Δm² error was a per-bin |dC|/C up to 6.2e-3.

`run_ic_binned_fit_worker.py` does this in `refresh_cell_state`, once per cell,
before the fit, and asserts it again after. `set_hs_slopes` (not the plain
attribute) records the Δm² the slopes came from, which is the only thing that
makes staleness detectable — the engine never sees the grid.

Its twin: **the φ row index and `dm31_cell` must derive from the same
`(i_dm, i_s23)`** — `ipt = i_dm * ns23 + i_s23`. Independent centre+step
arithmetic on the two sides is exactly how the one-step error happened.

---

## 0. What you need

| Input | Where | Notes |
|---|---|---|
| `data/IceCube/IC_MC.parquet` | in this repo | MC + muon background |
| `data/IceCube/_*_reco_bins.npy` | in this repo | reco bin edges |
| `data/IceCube/hs_*.csv` (3 files) | in this repo | hypersurfaces |
| IC DeepCore analysis XML | **not in this repo** | must carry the 39-dial manifest (19 flux + 15 xsec + 5 hypersurface) |
| nuSQuIDS | cluster/HPC env | step 2 only |

---

## 1. Build the response matrix (local, deterministic)

```bash
python3 analysis/IC-binned-datafit/build_ic_binned_response.py \
    --out-prefix /path/to/ic_response_modeaxis      # -> ..._L3.npz
```

Mode-axis and L3 are the defaults, and both matter:

- **`--mode-axis`** puts |NEUT mode| on the class axis (47 classes), which is
  what makes the 11 Mode-keyed cross-section dials representable at all. The
  engine **refuses** a 12-class response rather than producing quiet nonsense.
- **unsnapped** (i.e. no `--snap-e-edges`) keeps nE at the ladder's own value.
  The tensors index the response by integer cell, so a snapped response cannot
  be indexed by tensors built on the unsnapped grid; `phi_cells` hard-fails on
  the shape.

## 2. Build the oscillation tensors (cluster, nuSQuIDS)

One npz for the whole scan grid, rows in the worker's row-major order:

```bash
python3 analysis/IC-binned-datafit/build_ic_osc_tensors.py \
    --config   /path/to/IC_manifest.xml \
    --response /path/to/ic_response_modeaxis_L3.npz \
    --grid 20 2.3e-3 2.7e-3 20 0.40 0.65 \
    --dcp 1.36 \
    --out /path/to/ic_phi_L3.npz
```

`--grid NDM DM_MIN DM_MAX NS23 S23_MIN S23_MAX` emits the points in exactly the
order the worker reads (`ipt = i_dm*NS23 + i_s23`) and stores each row's
`(dm231, s23)`. The worker hard-checks those against the cell it is fitting, so a
mis-ordered point list is an error rather than a one-grid-step silent offset. If
you build with `--points` instead, that check is disabled and the worker says so.

IC's true side is event-level, so the tensor evaluated at ladder cell centres is
*approximate* per event — the cell-centering (Jensen) residue. That is a property
of the binning, quantified by the gate battery, not a bug in the build.

## 3. Run the scan

```bash
mkdir -p logs
sbatch --export=ALL,PYNU_TREE=/path/to/Pynu,CONFIG=/path/to/IC_manifest.xml,\
RESPONSE=/path/to/ic_response_modeaxis_L3.npz,PHI=/path/to/ic_phi_L3.npz,\
ENV_SETUP=/path/to/env.sh \
  analysis/IC-binned-datafit/submit_ic_binned_scan.sbatch
```

One array task per Δm² row; each writes `results/<tag>/<tag>_row<NNN>.json` with
per-cell `chi2`, the stat/prior split, the Δm² the hypersurface was interpolated
at, the convergence flag, and the full 39-dial vector. Locally:

```bash
python3 analysis/IC-binned-datafit/run_ic_binned_fit_worker.py \
    --row 0 --config ... --response ... --phi ... --outdir out/
```

### Where the prior lives

**In the worker, never in the engine.** `ICBinnedEngine.chi2_and_grad` is
stat-only pure-Poisson by contract; `run_ic_binned_fit_worker.py` adds
`Σ((θ−nominal)/σ)²` and its gradient `2(θ−nominal)/σ²`, with `(nominal, σ)` read
from the live `Analysis.NuisNominalList` / `NuisSigmaList`. An arm-internal
Gaussian would double-count silently *and* still pass an identity gate.

### Other conventions the worker preserves

- **NORM** (`FitExposure * SECONDS_PER_YEAR`) is applied at scan time — the
  response stores raw weight. It is read off the live experiment so a config
  change cannot desync the two sides.
- **Muon background**: a 200-bin constant added *after* the HS correction, with
  zero gradient. Taken from `exp.GetMuonBackground()`.
- **Mask**: `obs > MIN_ENTRIES` (0.01), the production definition. (This differs
  from the ORCA arm, whose mask is MC-support.)
- **Observation**: `exp.ObservedBinned` is masked in place by
  `Experiment.SetObservedBinned`, so the worker scatters it back to full length —
  exact, since χ² and the residual read `obs[few]` only. A `DataFit=False`
  experiment is refused loudly rather than silently self-fitting Asimov MC;
  `--observed-npz` is the explicit override.
- **Pinned dials**: the swept set is derived from `eng.pinned` (default
  `nunubar_ratio`), so the driver and the engine cannot disagree about which
  gradient slots are zeroed.

---

## 4. Verify

**Reference values come from the gate battery in `test/binned_icorca/`, not from
this README.** Run it against the artifacts you just built:

| Gate | Script | Certifies |
|---|---|---|
| G-IC-3 | `test/binned_icorca/gate_ic_engine.py` (+ `gate_ic_engine_L3.json`) | engine vs vendored reference values |
| G-IC-4 | `test/binned_icorca/gate_ic_identity.py` | engine vs the live event path at probe points |
| bench | `test/binned_icorca/bench_engines.py` | per-evaluation timing |

The gates carry their own tolerances and their own vendored reference numbers.
Do not compare a scan against a χ² quoted in prose — compare it against a gate.

---

## 5. Sanity floors the worker enforces per cell

- both per-cell states set from the same `(i_dm, i_s23)`, and the engine's
  recorded `hs_dm31` re-checked against the cell **after** the fit;
- the φ row's stored `(dm231, s23)` matches the cell being fitted;
- one bind-time ladder check (`cells.assert_phi_grid`) — the shape check alone
  cannot see a same-shape-different-ladder mix;
- `chi2` finite, stat term ≥ 0, prior term ≥ 0, and the split reconstructs the
  objective.

A failure raises and aborts the row rather than writing a json.
