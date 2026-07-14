# SK binned mode — engine, builders, and the two-mode model

**The binned forward model lives here, beside the mode subclass that drives
it.** The former `pynu/binned/` package (shim + residents) was dissolved at
Track T: this README replaces its package README; the relocation history is
`PROVENANCE_sk_binned.md` (same directory).

## The two-mode model

"Binned" is a **per-experiment mode**, not a parallel framework. An analysis
XML that declares a `<BinnedEngine>` block on a `<NeutrinoExperiment>` causes
PyNuFit to wrap that experiment in a `BinnedExperiment(Experiment)`
(`BinnedExperiment.py`, this directory) whose state is a pre-built response
matrix (npz) plus oscillation tensors, instead of event arrays.
`BinnedExperiment` implements the SAME base method vocabulary as the
event-mode `Experiment`, so PyNuFit's fit loops are mode-agnostic and a single
fit can mix binned and event experiments. `<BinnedEngine>` parsing is part of
the analysis reader (`ParseXML` attaches `BinnedConfigs` from its own tree;
PyNuFit consumes them) — routing is automatic.

## Module map (all in this directory unless noted)

| Module / data | Role |
|---|---|
| `BinnedExperiment.py` | the mode subclass — the single seam PyNuFit drives |
| `sk_binned_engine.py` | engine shell: response contraction, nuisance dials, analytic-gradient χ² |
| `sk_binned_engine_core.py` | numerical kernels (expectation / contraction / gradient) |
| `sk_binned_masks.py` | mask / selector assembly |
| `sk_binned_builder.py` | native builders behind `PyNuFit.BuildBinnedResponse` / `BuildOscTensors` |
| `SK2023_Atm_datafit_r2_fude_ccqe_full.xml`, `SK2023_Atm_datafit_binned_extra_dials.xml` | the dial-VALUE XMLs (package data; sole authority for dial nominal/σ) |
| `pynu/analysis_reader/binned_dials.py` | the dial vocabulary + XML value authority (leaf: stdlib+numpy; `resolve_nuisance_spec`) |
| `pynu/analysis_reader/binned_config.py` | `<BinnedEngine>` block config (`BinnedConfig`) |
| `pynu/PhysicsTunes/TuneFactorSource.py` | cell-weight factor sourcing from the real PhysicsTunes methods |
| `pynu/PhysicsTunes/Detector/detector.py`, `escale_operator.py` | detector factors / energy-scale operator (shared with the event mode) |
| `pynu/fitter/binned_kernels.py`, `pynu/fitter/minimizer/binned_fit.py`, `pynu/fitter/inference/interp_engine.py` | χ² kernels · `fit_point`/`TensorStore`/`BinnedBinding` · φ-interpolation |

## Driving the engine

Three equivalent drive paths (identical kernels, bit-identical results on the
same inputs):

1. **Through `PyNuFit`, packaged** — the XML's `<BinnedEngine>` block
   (`<status>1</status>`, `<response>`, `<tensors>`, optional
   `likelihood`/`migration`/`nuisance_spec`/`interp`/`osc_averaging`);
   `PyNuFit.FitModel()` auto-routes. Reference config:
   `analysis/SuperK-datafit/SK2023_Atm_binned.xml`.
2. **Through `PyNuFit`, modular** — the base method vocabulary
   (`StagePhysics` / `StartNuisance` / `ApplyNuisanceWeights` /
   `SetExpectedWeights` / `SetBinnedExpectedEvents` / `BinnedChi2AndGrad`)
   drives the fit step-by-step; the production worker
   `analysis/SK-binned-datafit/run_sk_binned_scan_row_worker.py` is the full
   pattern (dCP profiled per cell, warm-chaining, restart-polish).
3. **Standalone** — construct `SKBinnedEngine` from
   `pynu.Experiments.sk_binned_engine` directly and call
   `engine.fit_point(phi, ...)` in your own (Δm²₃₁, sin²θ₂₃) grid loop.

`<response>`/`<tensors>` are multi-GB numpy caches you build once (per-node
nuSQuIDS/MC pass): natively via `PyNuFit.BuildBinnedResponse` /
`BuildOscTensors`, or fanned out via the thin CLI wrappers in
`analysis/SK-binned-datafit/`. Production SK grid: 400 true-E × 80 true-cz.

## Nuisance dials

Dial `<nominal>`/`<sigma>` values come SOLELY from the two value XMLs in this
directory (`XML_DIAL_VALUES` in `binned_dials.py`; the
`analysis/AnalysisFiles/` mirrors hard-fail on any byte divergence). A named
`nuisance_spec` (e.g. the production 131-dial `r2_fude_ccqe`) selects which
dials are active and in what order via the activation manifests in
`analysis/AnalysisFiles/`; `resolve_nuisance_spec` is the resolver.
