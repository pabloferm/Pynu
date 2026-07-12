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
| `run_sk_binned_scan_row_worker.py` | row worker: ONE `PyNuFit` object, modular method vocabulary, `ft.PoissonLikelihood`; the standalone `SKBinnedEngine.fit_point` scan protocol fit-for-fit |
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
# 1) response (production 400x40 — pass --n-etrue 400, method default is 200):
python build_sk_binned_response.py \
    --config SK2023_Atm_datafit_r2_fude_ccqe_full.xml \
    --output sk_response.npz --n-etrue 400 --n-cztrue 40

# 2) one osc-tensor node per (Δm², s²θ₂₃) grid cell (row=i, col=j):
python build_sk_osc_tensors.py \
    --config SK2023_Atm_datafit_r2_fude_ccqe_full.xml \
    --dm231 2.5e-3 --s23 0.5 --row 10 --col 5 \
    --outdir tensors/ --n-etrue 400 --n-cztrue 40
#   -> tensors/osc_tensor_010_005.npz  (dCP axis linspace(0,2π,20,endpoint=False))
```

Both need the SK MC + nuSQuIDS env (FASRC); local smoke is `--help` only
(imports are deferred). `--n-etrue`/`--n-cztrue` MUST match between the response
and every tensor node (production 400×40).

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
| `pf.set_binned_engine(exp, BinnedConfig(...), analysis_xml)` | programmatic `<BinnedEngine>` opt-in: response/tensors paths, `likelihood='poisson'`, `migration='weighted'`, `interp='nodes'`, `osc_averaging='4pi'`, `nuisance_spec=<arm>` — returns the loaded adapter |
| `pf.SetBinnedDcpNode(di)` | worker-level δCP profile loop |
| `pf.ApplyPhysicsWeights(point)` | stages the (Δm², s²θ₂₃, δCP-node) tensor slice — used when the analysis XML declares the scan grid (see D4 otherwise) |
| `pf.StartNuisance()` | per-evaluation staged-θ reset |
| `pf.ApplyNuisanceWeights(θ)` | stages θ on the adapter |
| `pf.SetExpectedWeights()` | binned no-op, kept for call-sequence parity |
| `pf.SetBinnedExpectedEvents()` | contracts the response → `pf.Expectation[exp]` (FewEntries-filtered) |
| `ft.PoissonLikelihood.stats_only / stats_and_systematics` | ftol seeding + per-cell certification (see below) |

**The L-BFGS-B (f,g) callable is `adapter.chi2_and_grad_binned()`** (the adapter
is the PyNuFit-side surface those methods delegate to). Rationale: the engine's
one-pass analytic kernel is what the standalone `fit_point` minimizes, and it is
bit-identical to the vocabulary objective by construction (the adapter adds no
numerics of its own). Running the full vocabulary per evaluation would double
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
   `pynu/binned/SK2023_Atm_datafit_r2_fude_ccqe*.xml` activation manifests),
   mutually exclusive with `--arm-specs`. Existence-checked.

The worker additionally hard-checks that the resolved dial list is byte-equal
(names + order) to the seed json's `nuisance_names` — a spec/seed mismatch
aborts before any fit.

## Deviations — where the worker could NOT go through a PyNuFit method

| # | what | why no PyNuFit method | counterpart in the standalone path |
|---|---|---|---|
| D1 | seed `flux_ratio_sigma` (0.03 in the production seeds) written into `adapter.sigma` in place | no prior-override method exists | the standalone scan's seed prior-knob handling |
| D2 | L-BFGS-B bounds transcribed from `sk_binned_engine.fit_point:1828-1850` (box constants imported from the vendored module) | the engine exposes no bounds-building API | `fit_point` internal |
| D3 | `ft.PoissonLikelihood` constructed directly from the adapter's post-override nominal/σ (then `set_engine`, `pf.LLH = llh`) instead of `pf.set_likelihood('PoissonLikelihood')` | `set_likelihood` reads the XML priors, which don't carry the D1 flux-ratio override → penalty would differ | same class, same `set_engine` wiring |
| D4 | physics staging falls back to `adapter.apply_physics(dm, s23, di)` when the analysis XML doesn't declare the 20×20 grid | `ApplyPhysicsWeights(point)` resolves (Δm², s²θ₂₃) from the XML `FullPhysicsGrid`, which is not guaranteed to cover the scan | verbatim delegation target (`PyNuFit.py:493-504`) — numerically identical |

Legacy seed knobs (`lump`, `xsec_tight_sigma`, `dirsmear_matrix`) are NOT
supported: the worker hard-errors if a seed requests them (both unset in the
production `r2_fude_ccqe*` seeds).

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
