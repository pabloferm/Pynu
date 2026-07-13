# `pynu.binned` — back-compat shim + resident binned-engine modules

**This package is no longer the home of the binned forward model.** Its modules
have been re-homed into the functional subdirectories of the pynu tree, where a
reader looking for binned support finds it beside the event-mode code rather than
in a side package. What remains here is a **back-compatibility shim** plus the
modules that deliberately still live here.

`from pynu.binned import X` keeps working for every name it used to export — the
`__init__.py` is a PEP 562 lazy re-export surface that forwards to the new homes.
Existing scripts import unchanged. New code should import from the functional
homes directly (table below).

## Where things moved

| What | New home |
|---|---|
| `<BinnedEngine>` XML config (`BinnedConfig`, `parse_binned_config`) | `pynu/analysis_reader/binned_config.py` |
| χ² kernels (`poisson_chi2`, `bb_chi2`) | `pynu/fitter/binned_kernels.py` |
| `fit_point`, `TensorStore`, `BinnedBinding` | `pynu/fitter/minimizer/binned_fit.py` |
| φ-interpolation (`PhiInterpolator`, `detect_grid`, `JointSKFit`) | `pynu/fitter/inference/interp_engine.py` |
| Detector factors (`detector.py`) | `pynu/PhysicsTunes/Detector/detector.py` |
| Energy-scale operator (`escale_operator.py`) | `pynu/PhysicsTunes/Detector/escale_operator.py` |
| The binned **mode** itself | `pynu/Experiments/BinnedExperiment.py` |

## What still lives in `pynu/binned/`

| Module / data | Role |
|---|---|
| `__init__.py` | the back-compat re-export shim (PEP 562 lazy attributes) |
| `sk_binned_engine.py` | the engine shell — response contraction, nuisance dials, analytic-gradient χ² |
| `engine_core.py` | the Experiment-side numerical kernels (expectation / contraction / gradient) |
| `masks.py` | mask / selector assembly |
| `builder.py` | native builders for the engine's inputs (response npz + osc tensors), driven behind `PyNuFit.BuildBinnedResponse` / `PyNuFit.BuildOscTensors` |
| `grid_experiment.py` | grid-coordinate bridge that sources flux/xsec cell weights from the real PhysicsTunes methods |
| `SK2023_Atm_datafit_r2_fude_ccqe_full.xml`, `SK2023_Atm_datafit_binned_extra_dials.xml` | the two dial-VALUE XMLs (package data; the sole authority for dial nominal/σ) |

These modules are the engine internals plus the two builders (`builder.py` stays
behind the `PyNuFit.Build*` methods by design; `grid_experiment.py`'s final home
is still open). The value XMLs will follow the engine's home whenever that
settles. This package is scheduled for eventual deletion in a later cleanup pass
once the resident modules relocate; until then it stays as the shim.

## The two-mode model

"Binned" is a **per-experiment mode**, not a parallel framework. An analysis XML
that declares a `<BinnedEngine>` block on a `<NeutrinoExperiment>` causes PyNuFit
to wrap that experiment in a `BinnedExperiment(Experiment)`
(`pynu/Experiments/BinnedExperiment.py`) whose state is a pre-built response
matrix (npz) plus oscillation tensors, instead of event arrays.
`BinnedExperiment` implements the SAME base method vocabulary as the event-mode
`Experiment` (`SetObservedBinned` / `SetExpectedBinned` /
`StartNuisanceWeights` / `UpdatePhysicsWeights` / …), so PyNuFit's fit loops are
mode-agnostic and a single fit can mix binned and event experiments. Mode
selection happens at the **assemble** level, not per-dial at fit time:
`SetExpectedBinned()` does one fused tensor contraction — there is deliberately
no per-dial event-loop analogue (the fused kernel is the engine's speed).

## Import-ordering constraint

Reach the fitter/minimizer binned surface (`binned_fit`) through the
`pynu.binned` entry point rather than importing `pynu.fitter.minimizer.binned_fit`
first — the shim's re-export order is the circular-import guard. The destination
`__init__`s are non-eager (they import nothing heavy), so toggling the binned mode
OFF runs zero binned code and the surface imports cleanly in an environment
without nuSQuIDS / nuflux.

## Driving the engine

Three equivalent drive paths (unchanged by the re-homing):

1. **Standalone, direct import** — import `SKBinnedEngine` from
   `sk_binned_engine.py`, construct it, and call `engine.fit_point(...)` in your
   own (Δm²₃₁, sin²θ₂₃) grid loop.
2. **Through `PyNuFit`, packaged** — an analysis XML declares
   `<BinnedEngine><status>1</status>…` on a `<NeutrinoExperiment>` and
   `PyNuFit.FitModel()` auto-routes to `FitModelBinned()`.
3. **Through `PyNuFit`, modular** — the same base method vocabulary drives both
   event and binned fits with the same call sequence; a full production-shaped
   worker on this path is
   `analysis/SK-binned-datafit/run_sk_binned_scan_row_worker.py`.

All paths call the exact same kernels with the exact same numerics, so results
agree bit-for-bit given the same inputs.

### Quickstart: through `PyNuFit`

```python
from pynu.PyNuFit import PyNuFit

pynufit = PyNuFit("path/to/your_analysis.xml")
# If the XML's <NeutrinoExperiment> block for this experiment has a
# <BinnedEngine><status>1</status>...</BinnedEngine> child, every subsequent
# pynufit.FitModel(point) call for that experiment routes through the binned
# engine automatically.
chi2 = pynufit.FitModel({"Dm231": 2.5e-3, "Sin2Theta23": 0.55})
```

Programmatic opt-in (no XML block) uses `pynufit.set_binned_engine(exp_name,
BinnedConfig(...), analysis_xml)`, which returns the loaded `BinnedExperiment`
(its read-only surface is the former `BinnedBinding`).

Minimal `<BinnedEngine>` block (all fields but `<response>`/`<tensors>` are
optional and default as shown; the authoritative field list is
`pynu/analysis_reader/binned_config.py:BinnedConfig`):

```xml
<NeutrinoExperiment name="SuperK_2023">
    ...
    <BinnedEngine>
        <status> 1 </status>
        <response> ${SK_BINNED_DIR}/sk_response.npz </response>
        <tensors> ${SK_BINNED_DIR}/osc_tensors </tensors>
        <likelihood> poisson </likelihood>      <!-- poisson (default) | bb -->
        <migration> weighted </migration>       <!-- weighted (default) | rawcount -->
        <nuisance_spec> barr </nuisance_spec>   <!-- 'self' | a named spec | an .xml path -->
        <interp> nodes </interp>                <!-- nodes (default, exact) | cubic -->
        <osc_averaging> 4pi </osc_averaging>    <!-- provenance declaration only -->
    </BinnedEngine>
</NeutrinoExperiment>
```

**`<response>`/`<tensors>` are not committed to this repo.** They are multi-GB
numpy caches you build yourself (see "Building the response and tensor caches"
below) — point `$SK_BINNED_DIR` (or hardcode the paths) at wherever you put them.
There is no default value or repo-relative fallback; if it is unset, XML parsing
of `${SK_BINNED_DIR}/...` leaves the literal unexpanded string and fails
downstream. A full worked reference config is
`analysis/SuperK-datafit/SK2023_Atm_binned.xml`.

### Quickstart: standalone

```python
from sk_binned_engine import SKBinnedEngine
import numpy as np

eng = SKBinnedEngine(
    "path/to/sk_response.npz",
    migration_mode="weighted",
    likelihood="poisson",
    nuisance_spec="barr",
)
phi = np.load("path/to/osc_tensors/osc_tensor_<i>_<j>.npz")["phi"]  # one grid node
chi2, dcp_bf, nuisance_bf, n_evals, converged = eng.fit_point(phi, x0=None)
```

Your own script owns the (Δm²₃₁, sin²θ₂₃) grid loop and the SLURM array mapping;
`analysis/SK-binned-datafit/run_sk_binned_scan_row_worker.py` shows the full
pattern (warm-chaining the nuisance seed across grid points, a restart-polish
loop, δCP profiled per cell), driven through the modular path.

## What `fit_point` actually does

`fit_point` profiles δCP over a fixed set of discrete tensor nodes (13 or 20,
depending on how the tensor cache was built) and, at each node, minimizes only the
nuisance-parameter vector via `scipy.optimize.minimize(method="L-BFGS-B")` using
the engine's analytic gradient. **Oscillation parameters are never free inside
`fit_point`** — Δm²₃₁/sin²θ₂₃ are fixed by which pre-built tensor you pass in, and
δCP is a discrete profile scan, not a continuous fit variable. The (Δm²₃₁,
sin²θ₂₃) grid loop lives entirely in the calling script (standalone) or in
`PyNuFit`'s own point-by-point calling convention (framework path) — never inside
the engine itself.

## Nuisance dials: values come from the package value XMLs

The binned engine's dial `<nominal>`/`<sigma>` values are read from two **value
XMLs that ship as package data** in this directory: the production 131-dial
`SK2023_Atm_datafit_r2_fude_ccqe_full.xml` plus the supplementary
`SK2023_Atm_datafit_binned_extra_dials.xml` for the non-production dials some
specs resolve. These two files are the **sole authority** for dial values
(`resolve_nuisance_spec` reads them via `XML_DIAL_VALUES`). A named
`nuisance_spec` (or an activation manifest) only selects **which named dials are
active and in what order**; the values come from the value XMLs, so changing a
prior means editing a value XML, not the engine source. The named-spec activation
manifests (`SK2023_Atm_datafit_*.xml`) live in `analysis/AnalysisFiles/`.

When the analysis-tree mirror of a value XML is present
(`analysis/AnalysisFiles/SK2023_Atm_datafit_r2_fude_ccqe_full.xml` etc.), the
loader hard-fails on any byte-level divergence from the package copy — there is
no silent second authority.

For the event-level path, per-nuisance hard bounds are declared with an optional
`<box>lo hi</box>` child on the `<nuisance>` block (parsed into
`ParseXML.NuisBox`); an absent tag means unbounded, so existing XMLs parse
unchanged.

## Running the production systematics set (`r2_fude_ccqe`, incl. neutron tagging)

The current production systematics configuration is the 131-dial `r2_fude_ccqe`
set. The neutron-tagging dials (`neutron_tagging_subgev`,
`neutron_tagging_multigev`) are part of those 131 — selecting the set selects
them; there is no separate switch. Which XML setting selects it depends on the
drive path:

- **Binned engine** (standalone, or through `PyNuFit` packaged/modular): set the
  nuisance spec to the named spec `r2_fude_ccqe`, or equivalently to the manifest
  path `analysis/AnalysisFiles/SK2023_Atm_datafit_r2_fude_ccqe.xml` — the two
  resolve to the same 131 dials in the same order (`resolve_nuisance_spec` in
  `sk_binned_engine.py`). In a `<BinnedEngine>` block that is
  `<nuisance_spec> r2_fude_ccqe </nuisance_spec>`; standalone it is
  `SKBinnedEngine(..., nuisance_spec="r2_fude_ccqe")`.
- **Modular `PyNuFit` pipeline** (`analysis/SK-binned-datafit/`): the dial
  name/order/nominal/sigma source is
  `analysis/AnalysisFiles/SK2023_Atm_datafit_r2_fude_ccqe_full.xml` — all 131
  `<nuisance>` blocks in the flat production order (byte-equal to the manifest
  above), including the `<box>` hard bounds. Note that `PyNuFit` init
  additionally needs a FULL analysis config in which every enabled nuisance has
  an owning tune method (see the two-XML convention note in
  `analysis/SK-binned-datafit/PIPELINE_README.md`).
- **Optional variants** (binned engine only, off by default): the
  neutron-production 0n/1n migration extension ships as
  `SK2023_Atm_datafit_r2_fude_ccqe_nmig.xml` (trial, σ = 0.20) and
  `..._nmig_pinned.xml` (pinned, σ = 0.10 with hard physical box); the decay-e
  norm extension as `..._dcye.xml`; `..._nmig_dcye.xml` combines both
  (133/133/132/134 dials respectively). None of these is part of the production
  set — nothing activates them unless you select one explicitly.

## Building the response and tensor caches

`<tensors>` is a directory of `osc_tensor_<i>_<j>.npz` files, one oscillation
tensor per (Δm²₃₁, sin²θ₂₃) grid node; `<response>` is a single `sk_response.npz`
carrying the detector response/migration matrix and the observed data vector.
Building these requires running the full event-level nuSQuIDS/MC pipeline once per
grid node.

There are two equivalent ways to build them:

1. **Native methods (in-process)** — a constructed `PyNuFit` object builds the
   artifacts from its own live experiment + physics-tune state:

   ```python
   pf = PyNuFit("path/to/your_analysis.xml")
   pf.BuildBinnedResponse(exp_name="SuperK_2023", out_path="sk_response.npz",
                          n_etrue=400, n_cztrue=80)
   pf.BuildOscTensors(2.5e-3, 0.55, exp_name="SuperK_2023",
                      dcp_nodes=np.linspace(0, 2*np.pi, 20, endpoint=False),
                      out_path="osc_tensor_000_000.npz")
   ```

   Both delegate to `pynu/binned/builder.py`. Every MC convention is inherited
   from the live experiment (never re-implemented), class signatures come from
   evaluating the actual `WaterXSection` tunes, and the object's state is
   snapshot/restored `try/finally` byte-exact, so a subsequent event fit on the
   same object is unaffected. The npz outputs are byte-compatible with what the
   engine loads.

2. **Standalone build scripts (cluster fan-out)** — the per-node build is
   embarrassingly parallel over grid nodes, so large grids are usually built by
   SLURM array jobs calling the standalone scripts the builder was ported from.
   The `analysis/SK-binned-datafit/` row worker's `--build-missing` flag bridges
   the two: absent artifacts are built natively on the node, prebuilt artifacts
   are the fast path.

## Relationship to `analysis/SuperK-datafit/sk_binned/`

This repo also contains an older, separate, independently-built SK binned engine
under `analysis/SuperK-datafit/sk_binned/` (see that directory and the top-level
`README_datafit-SK.md`). It predates this package, is **not** interoperable with
it (its `sk_response.npz` uses an incompatible file schema despite the shared
filename — `allow_pickle=True` with JSON-blob keys, vs this engine's
`allow_pickle=False`), and was deliberately kept unmerged rather than reconciled
(see that commit's message). Don't point `$SK_BINNED_DIR` at that directory's
build output — it will silently fail to load or misbehave. If you're not sure
which one you're looking at: this package's engine lives at
`pynu/binned/sk_binned_engine.py`, not under `analysis/`.

## Provenance

`pynu/binned/` is native code (the Track S de-vendoring completed at E6; the
Track S·F re-homing distributed the surface into the functional subdirectories at
F1–F5). The engine shell, kernels, descriptor modules, and builders are owned by
this repo, and the SK dial values ship as package-data value XMLs. `PROVENANCE.md`
is the historical record of the former vendoring era (source commits + the
snapshot hash history) and the S·F re-homing.
