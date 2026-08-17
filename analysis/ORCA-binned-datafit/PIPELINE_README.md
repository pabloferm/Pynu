# ORCA binned-engine pipeline

End-to-end recipe for the single-experiment ORCA binned fit: build the response
matrix, build the oscillation tensors, run the scan, verify.

The engine (`pynu/Experiments/orca_binned_engine.py`) evaluates the ORCA dial
algebra once per **populated true cell** instead of once per event, and gets the
whole 30-long dial gradient from a single adjoint pass. Steps 1 and 2 produce the
two artifacts it needs; step 3 is the fit.

---

## 0. What you need

| Input | Where | Notes |
|---|---|---|
| `data/ORCA/ORCA_MC_dataverse_with_muons.parquet` | in this repo | MC + muon background |
| `data/ORCA/ORCA_data_dataverse.parquet` | in this repo | the observation |
| `data/ORCA/_*_bins.npy` | in this repo | reco/true bin edges |
| ORCA analysis XML | **not in this repo** | must carry the 30-dial manifest (19 flux + 4 xsec + 7 detector, `E_shift` last) that the engine's `ORCA_MANIFEST` names, in that order |
| nuSQuIDS | cluster/HPC env | step 2 only |

Steps 1 and 3 need no nuSQuIDS. Step 2 does — the same constraint the SK binned
pipeline has.

---

## 1. Build the response matrix (local, deterministic)

```bash
python3 analysis/ORCA-binned-datafit/build_orca_binned_response.py \
    --out /path/to/orca_binned_response_flat900.npz
```

`--layout flat900` is the default and the only layout the engine loads; the
schema guard refuses anything else rather than producing quiet nonsense. The
build is one pass over the parquet and is reproducible array-for-array (npz file
hashes never match across machines — zip timestamps — so compare arrays, not
files).

## 2. Build the oscillation tensors (cluster, nuSQuIDS)

One tensor per grid point, named so the worker can find it:

```bash
for i in $(seq 0 19); do for j in $(seq 0 19); do
  dm=$(python3 -c "import numpy as np;print(np.linspace(2.3e-3,2.7e-3,20)[$i])")
  s23=$(python3 -c "import numpy as np;print(np.linspace(0.40,0.65,20)[$j])")
  python3 analysis/ORCA-binned-datafit/build_orca_osc_tensors.py \
      --config  /path/to/ORCA_manifest.xml \
      --response /path/to/orca_binned_response_flat900.npz \
      --dm231 "$dm" --s23 "$s23" --dcp-nodes 1.36 \
      --out "$PHI_DIR/$(printf 'orca_phi_%03d_%03d.npz' $i $j)"
done; done
```

The tensor is built on the response's **native quantized true centres**, so it is
exact per event — ORCA's true side is intrinsically binned and every event
carries its cell-centre true coordinate.

**Oscillation averaging must be off.** With averaging on, the per-cell flux stops
being an exact reduction of the event path; the engine asserts
`osc.osc_avg_scale is None` at construction.

Each tensor records the `(dm231, s23)` it was built at, and the worker checks
that against the cell it is fitting — a tensor directory indexed one grid step
off is an error, not a smooth wrong surface.

## 3. Run the scan

```bash
mkdir -p logs
sbatch --export=ALL,PYNU_TREE=/path/to/Pynu,CONFIG=/path/to/ORCA_manifest.xml,\
RESPONSE=/path/to/orca_binned_response_flat900.npz,PHI_DIR=$PHI_DIR,\
ENV_SETUP=/path/to/env.sh \
  analysis/ORCA-binned-datafit/submit_orca_binned_scan.sbatch
```

One array task per Δm² row; each writes `results/<tag>/<tag>_row<NNN>.json` with
per-cell `chi2`, the stat/prior split, convergence flag, and the full 30-dial
vector. Locally, a single row is just

```bash
python3 analysis/ORCA-binned-datafit/run_orca_binned_fit_worker.py \
    --row 0 --config ... --response ... --phi-dir $PHI_DIR --outdir out/
```

### Two conventions the worker preserves

1. **`E_shift` is pinned at 1.0.** The response encodes `ENERGY_SCALE = 1`, so a
   moved `E_shift` would be silently ignored. The engine hard-asserts the value
   and zeroes its gradient slot; the worker drops it from the swept set.
2. **The fit mask is MC-support, not `obs > MIN_ENTRIES`.** The worker reads
   `exp.FewEntries` off the live experiment — on the production XML (whose
   `<DataFiles>` block is status 0) that is "bins the ORCA MC can populate", 430
   of 900, rather than the 427 bins with data. The data still enters χ²: the
   worker supplies the observation itself from the data parquet and never reads
   `exp.ObservedBinned`.

### Where the prior lives

**In the worker, never in the engine.** `ORCABinnedEngine.chi2_and_grad` is
stat-only pure-Poisson by contract; `run_orca_binned_fit_worker.py` adds
`Σ((θ−nominal)/σ)²` and its gradient `2(θ−nominal)/σ²`, with `(nominal, σ)` read
from the live `Analysis.NuisNominalList` / `NuisSigmaList`. An arm-internal
Gaussian would double-count silently *and* still pass every identity gate.

### φ source

`--phi-source tensors` (default) is the reproducible route. `--phi-source live`
re-derives the cell flux from the live event path by the MC-weight-weighted cell
mean — the production convention. The two are **not** identical: nuSQuIDS
randomizes each event's production height, so the live per-cell flux is a
weighted mean over a stochastic sample while the tensor is one deterministic
evaluation at the cell centre. Use `live` to reproduce production numbers.

---

## 4. Verify

**Reference values come from the gate battery in `test/binned_icorca/`, not from
this README.** Run it against the artifacts you just built:

| Gate | Script | Certifies |
|---|---|---|
| G-ORCA-1 | `test/binned_icorca/gate_orca_identity.py` | engine expectation ≡ event-path expectation on the flat900 response |
| G-ORCA-2 | `test/binned_icorca/gate_orca_grad.py` | analytic gradient vs finite differences, all 30 dials |
| bench | `test/binned_icorca/bench_engines.py` | per-evaluation timing |

The gates carry their own tolerances and their own vendored reference numbers.
Do not compare a scan against a χ² quoted in prose — compare it against a gate.

---

## 5. Sanity floors the worker enforces per cell

- `chi2` finite, stat term ≥ 0, prior term ≥ 0;
- the reported stat + prior split reconstructs the objective;
- every φ tensor's stored `(dm231, s23)` matches the cell being fitted;
- `E_shift` nominal is exactly 1.0 (checked once, at setup, with a message
  pointing at the XML rather than the worker).

A failure raises and aborts the row rather than writing a json.
