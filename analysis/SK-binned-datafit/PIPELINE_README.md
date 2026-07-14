# SK binned 20×20 scan — integrated (PyNuFit-modular) engine

Two 20×20 (Δm²₃₁ × sin²θ₂₃) scans driven through the modular path:
arm `r2_fude_ccqe` (131 dials) and `r2_fude_ccqe_nmig` (133 dials). Grid: Δm²
linspace 2.0e-3→3.2e-3 ×20, sin²θ₂₃ linspace 0.30→0.70 ×20, δCP profiled per
cell over the 20-node tensor axis. Output jsons are schema-identical to the
standalone-engine scan rows (`{arm, arm_spec, row, dm231, n_dials,
nuisance_names, points[{dm231, sin2theta23, chi2, best_dcp_idx, nuisance}]}`,
`.tmp`+`os.replace` atomic writes, filenames `<arm>_row<rrr>.json`), so the two
paths can be compared cell-by-cell.

## Files

| file | role |
|---|---|
| `run_sk_binned_scan_row_worker.py` | row worker: ONE `PyNuFit` object, modular method vocabulary (incl. the S·F/F5 first-class `override_prior_sigma` / `nuisance_bounds` / `set_likelihood(binned_priors=True)` / `StagePhysics`), pure-Poisson LLH; the standalone `SKBinnedEngine.fit_point` scan protocol fit-for-fit |
| `build_sk_binned_response.py` | thin CLI over the native `PyNuFit.BuildBinnedResponse` — builds the response npz (see "Building the inputs") |
| `build_sk_osc_tensors.py` | thin CLI over the native `PyNuFit.BuildOscTensors` — builds one `osc_tensor_<i>_<j>.npz` node (see "Building the inputs") |
| `submit_sk_binned_scan.sbatch` | SLURM array [0-39], task = arm_idx·20 + row; arms parameterized at top; all paths `--export`-overridable |
| `compare_scan_outputs.py` | reference-vs-modular per-cell Δχ² table + acceptance verdict (EXACT / SCATTER regimes) |
| `PIPELINE_README.md` | this file |

## Building the inputs

The scan consumes two external data artifacts, both passed downstream by path:
the **response npz** (`--response`) and the **tensor dir** (`--tensors`, one
`osc_tensor_<i>_<j>.npz` per (Δm², s²θ₂₃) grid node). Two thin CLI wrappers
build them through the NATIVE, Gate-D-certified PyNuFit methods
(`PyNuFit.BuildBinnedResponse` / `PyNuFit.BuildOscTensors`,
`pynu/PyNuFit.py:1424/:1468`). They are the same methods the row worker calls
under `--build-missing`; the wrappers just make the build a standalone step.

```bash
# 1) response (production 400×80 — pass --n-etrue 400, method default is 200):
python build_sk_binned_response.py \
    --config SK2023_Atm_datafit_r2_fude_ccqe_full.xml \
    --output sk_response.npz --n-etrue 400 --n-cztrue 80

# 2) one osc-tensor node per (Δm², s²θ₂₃) grid cell (row=i, col=j):
python build_sk_osc_tensors.py \
    --config SK2023_Atm_datafit_r2_fude_ccqe_full.xml \
    --dm231 2.5e-3 --s23 0.5 --row 10 --col 5 \
    --outdir tensors/ --n-etrue 400 --n-cztrue 80
#   -> tensors/osc_tensor_010_005.npz  (dCP axis linspace(0,2π,20,endpoint=False))
```

Both need the SK MC + nuSQuIDS env (FASRC); local smoke is `--help` only
(imports are deferred). `--n-etrue`/`--n-cztrue` MUST match between the response
and every tensor node (production 400×80).

**Frozen-original convention.** The native builder methods have byte-parity
reference twins — the standalone SLURM scripts
`analysis/SuperK-datafit/sk_binned/build_sk_response.py` and
`build_osc_tensors.py`. Those are the frozen Gate-D reference and stay the
frozen cluster entry points until Track S **E7**; they **MUST NOT be modified**.
The wrappers here are the native-method equivalent for this pipeline (Gate D
certified the two paths produce byte-identical artifacts) — so nobody
"rediscovers" the two-directory split and edits the frozen originals.

## Every PyNuFit call in the worker, and why

| call | why |
|---|---|
| `PyNuFit(analysis_xml)` | the single object; full event-engine init (MC + nuSQuIDS) so the native builders can run on the same object |
| `pf.BuildBinnedResponse(exp, out_path=…)` | native response build when the npz is absent (`--build-missing`); byte-compatible with `build_sk_response.py`. A prebuilt artifact = fast path (default) |
| `pf.BuildOscTensors(dm, s23, dcp_nodes=…, out_path=…)` | native per-node tensor build for absent nodes of this row (`--build-missing`). dCP convention = `linspace(0, 2π, 20, endpoint=False)` |
| `pf.set_binned_engine(exp, BinnedConfig(...), analysis_xml)` | programmatic `<BinnedEngine>` opt-in: response/tensors paths, `likelihood='poisson'`, `migration='weighted'`, `interp='nodes'`, `osc_averaging='4pi'`, `nuisance_spec=<arm>` — returns the `BinnedExperiment` (its read-only surface == the former `BinnedBinding`) |
| `exp.override_prior_sigma(names, σ)` | seed `flux_ratio_sigma` prior override, in place on `binding.sigma` (== `engine.sigma`) — first-class (was deviation D1) |
| `exp.nuisance_bounds()` | L-BFGS-B production box (nominal±10σ, 0.01 clip, box dials incl. the pinned neutron-migration box) — the single source shared with `SKBinnedEngine.fit_point` (was deviation D2) |
| `pf.set_likelihood('PoissonLikelihood', binned_priors=True)` | pure-Poisson LLH from the binding's post-override nominal/σ + engine kernel — first-class (was deviation D3) |
| `pf.SetBinnedDcpNode(di)` | worker-level δCP profile loop |
| `pf.StagePhysics(dm, s23, di)` | stages the (Δm², s²θ₂₃, δCP-node) tensor slice; uses `ApplyPhysicsWeights(point)` when the analysis XML grid covers (dm, s23), else the `StageBinnedPhysics` fallback — first-class grid staging with fallback (was deviation D4) |
| `pf.resolve_binned_grid_index(dm, s23)` | grid-index lookup (or None) — the fallback half of the staging decision, also used for the staging-mode print |
| `pf.StartNuisance()` | per-evaluation staged-θ reset |
| `pf.ApplyNuisanceWeights(θ)` | stages θ on the PyNuFit object |
| `pf.SetExpectedWeights()` | binned no-op, kept for call-sequence parity |
| `pf.SetBinnedExpectedEvents()` | contracts the response → `pf.Expectation[exp]` (FewEntries-filtered) |
| `pf.LLH.stats_only / stats_and_systematics` | ftol seeding + per-cell certification (see below) |

**The L-BFGS-B (f,g) callable is `pf._binned_chi2_and_grad()`** (the PyNuFit-side
surface holding the staged (phi, theta) those methods set). Rationale: the
engine's one-pass analytic kernel is what the standalone `fit_point` minimizes,
and it is bit-identical to the vocabulary objective by construction (the binding
adds no numerics of its own). Running the full vocabulary per evaluation would double
the contraction cost; instead **every converged cell is re-evaluated through the
complete vocabulary + `PoissonLikelihood.stats_and_systematics` and must equal
the kernel value with diff == 0.0 or the task aborts** — each row json is
self-certifying.

Bit-parity ingredients replayed exactly from the standalone protocol
(`SKBinnedEngine.fit_point`): same seed json (θ₀), same bounds (nominal±10σ,
0.01 clip, box dials incl. the pinned neutron-migration box), same
ftol = max(1e-5, √(stat-χ²)·1e-5) per δCP node, δCP warm-chain, s23 warm-chain,
`--npolish 3` restart-polish with the `1e-3` acceptance, same output naming from
the seed basename.

## Two-XML / named-spec convention (explicit, never conflated)

Three distinct XML roles:

1. **`--analysis-xml`** = the FULL PyNuFit config (experiment blocks + MC).
   Every enabled nuisance in it must have an owning tune method — a bare
   enabled dial with no method crashes PyNuFit at init. The sbatch preflight
   fails loudly if the file is absent.
2. **`--arm-specs`** = engine NAMED specs (`r2_fude_ccqe`, `r2_fude_ccqe_nmig`).
   Path-like values are rejected.
3. **`--arm-xmls`** = nuisance-manifest .xml FILE PATHS (the
   `analysis/AnalysisFiles/SK2023_Atm_datafit_r2_fude_ccqe*.xml` activation
   manifests), mutually exclusive with `--arm-specs`. Existence-checked.

The worker additionally hard-checks that the resolved dial list is byte-equal
(names + order) to the seed json's `nuisance_names` — a spec/seed mismatch
aborts before any fit.

## Deviations — none (all dissolved into first-class methods, Track S·F / F5)

The worker previously carried four deviations (D1–D4) where a step could not go
through a PyNuFit method. Track S·F / F5 dissolved every one into a first-class
method on `PyNuFit` / `BinnedExperiment`; the worker now calls only the method
vocabulary and holds **no** worker-side deviation. For provenance:

| former | what it was | now owned by |
|---|---|---|
| D1 | seed `flux_ratio_sigma` written into `binding.sigma` (== `engine.sigma`) in place; no prior-override method existed | `BinnedExperiment.override_prior_sigma(names, σ)` |
| D2 | L-BFGS-B bounds transcribed from the engine's box constants; the engine exposed no bounds-building API | `BinnedExperiment.nuisance_bounds()` → `BinnedBinding.nuisance_bounds()` → `fitter.minimizer.binned_fit.build_nuisance_bounds` (the single source `SKBinnedEngine.fit_point` also calls) |
| D3 | `ft.PoissonLikelihood` constructed directly from the binding's post-override nominal/σ, because `set_likelihood` read the XML priors (which don't carry the D1 override) | `pf.set_likelihood('PoissonLikelihood', binned_priors=True)` → `BinnedExperiment.poisson_likelihood()` (sources the priors from the binding) |
| D4 | physics staging fell back to `pf.StageBinnedPhysics` when the analysis XML didn't declare the scan grid | `pf.StagePhysics(dm, s23, di)` (grid staging via `ApplyPhysicsWeights` when the XML grid covers the cell, `StageBinnedPhysics` fallback otherwise; `pf.resolve_binned_grid_index` is the lookup) |

The refactor is surface-only — every method preserves the exact numerics of the
former inline code (the bounds box is literally the same source `fit_point` uses;
the prior override writes the same array positions; the Poisson LLH is the same
class/`set_engine` wiring), so the row fit tuple is bit-identical.

Legacy seed knobs (`lump`, `xsec_tight_sigma`, `dirsmear_matrix`) are NOT
supported: the worker hard-errors if a seed requests them (both unset in the
production `r2_fude_ccqe*` seeds).

## Module homes after Track T (final)

The binned surface is fully distributed across the functional pynu
subdirectories; `pynu/binned/` was DELETED at Track T / T6. Homes: the dial
vocabulary + XML value authority in `pynu/analysis_reader/binned_dials.py`;
the `<BinnedEngine>` config in `pynu/analysis_reader/binned_config.py`
(reader-attached — `ParseXML.BinnedConfigs`); the engine + kernels + masks +
builders as `pynu/Experiments/sk_binned_*` beside `BinnedExperiment.py`, with
the two dial-value XMLs as package data there; the χ² kernels / `fit_point` /
`TensorStore` / `BinnedBinding` / φ-interpolator in `pynu/fitter/`; the
detector + energy-scale operator in `pynu/PhysicsTunes/Detector/` and the
cell-weight factor sourcing in `pynu/PhysicsTunes/TuneFactorSource.py`. This
worker imports only `pynu` (PyNuFit) and the two `analysis_reader` config
modules. Full map: `pynu/Experiments/README_sk_binned.md` +
`PROVENANCE_sk_binned.md`. Historical scripts that still import `pynu.binned`
run against the frozen reference tree (tag `certified/2a7b2ff`) via
`PYNU_ROOT`.

## How to submit

```bash
# from the directory you want logs/ and outputs relative to:
mkdir -p logs
sbatch <PYNU_TREE>/analysis/SK-binned-datafit/submit_sk_binned_scan.sbatch
# every path is overridable via --export, e.g.:
#   sbatch --export=ALL,PROJECT_DIR=/path/to/project,TAG=my_replay,RESPONSE=/path/to/sk_response.npz …
```

Then compare against a reference (standalone-engine) scan of the same grid:

```bash
python3 compare_scan_outputs.py \
  --canonical /path/to/reference/rows \
  --modular   /path/to/modular_replay_20x20 \
  --arms r2_fude_ccqe r2_fude_ccqe_nmig --out compare_report.txt
```

**Acceptance**: EXACT (Δχ² == 0.0 every cell) is the target and the expected
outcome — the kernel is the engine's own and the trajectory is replayed. The ±1
convergence-scatter tolerance is admissible ONLY if the warm-chain trajectory
demonstrably differs (the comparator's `first-diverge` column attributes this);
the report states which regime applies.

## Submission-time checklist (sanity floor)

- [ ] Analysis XML path confirmed (preflight enforces existence, not correctness).
- [ ] Response npz at `$RESPONSE`; tensor dir at `$TENSORS` complete (400 nodes, 20-node δCP axis) — the worker aborts if the tensor axes ≠ dm 2.0e-3→3.2e-3 ×20 / s23 0.30→0.70 ×20.
- [ ] Seed jsons `${SEEDS_DIR}/{r2_fude_ccqe,r2_fude_ccqe_nmig}.json` = the same files the reference scan consumed (when replaying one).
- [ ] Worker prints `cert=0.0` on every cell (vocabulary certification) — any task that aborts with "certification FAILED" is a bit-parity break, not a tolerance issue.
