# Branch `datafit-SK` — Super-Kamiokande data-fit upgrades

This branch extends `pheno-CPT` with the Super-Kamiokande 2023 data-fit work:
fixes to the SK systematics implementation and minimizer loop, new opt-in
nuisance dials at the granularity of the published SK analysis,
and an experimental **binned forward engine** that reproduces the event-by-event
pipeline at true-grid resolution. It is structured to merge directly into
`main` (it carries the full `pheno-CPT` line; the only merge conflicts are the
two pre-existing ones from main-only commit `931c960`).

| Commit | Area | Summary |
|---|---|---|
| `ae5f6d6` | `pynu/Experiments/SuperK_Atm_Pheno.py` | Support `*_allvariables.h5` MC: CC-mask tolerant to `\|S1`/`\|S3` encodings; NEUT `mode` read from h5 (zeros fallback) |
| `de7066a` | `pynu/Experiments/SuperK_Atm_Pheno.py` | `weight_tune` re-enabled in `WMC`; `WMC` included in the Barlow-Beeston MC variance |
| `932224e` | `pynu/PhysicsTunes/Detector/SKCombinedDetector.py` | Migration ratios from **weighted rates** instead of raw event counts (+ regression test) |
| `0ef214a` | `pynu/PhysicsTunes/Flux/AtmoFlux.py`, `SKCombinedDetector.py` | Opt-in dials: `barr_zenith`, era-split neutron tagging (+ 3 XML variants) |
| `27fff9b` | `pynu/fitter/BinnedLogLikelihoodRatio.py` | Fix non-positive-expectation guard (`np.any(E) <= 0` typo → NaN → minimizer hang) |
| `573f4a9` | `analysis/SuperK-datafit/run_sk_datafit_row_worker.py` | Per-eval refresh of BB variance + analytic Jacobian; `StartPhysics()` reset in `set_physics_params` |
| `9f1eec2` | `analysis/SuperK-datafit/sk_binned/` | **Binned forward engine** (experimental) |
| `819e9b1` | `SKCombinedDetector.py` | Back-compat switch `MIGRATION_BASIS` (`"weighted"`/`"raw"`) |

Run the systematics regression test (pure numpy, no nuSQuIDS/MC needed):

```bash
python test/test_sk_weighted_migration.py    # 5/5
```

---

## 1. Systematics implementation: weighted-rate migration

### What changed

Every SK migration systematic (`multiring_*_separation`, `pc_stopthru_separation`,
`pi0_ring_separation`, `e_ring_separation`, `mu_ring_separation`,
`singlering_pid`, `multiring_pid`, `neutron_tagging`, `upmu_shower_separation`)
has the rate-conserving form

```
mr[donor]    = x
mr[acceptor] = 1 + r · (1 − x),      r = n_donor / n_acceptor
```

so that what leaves the donor samples reappears in the acceptor samples.
Previously `n_donor`, `n_acceptor` were **raw MC event counts**
(`np.sum(Sample == s)`). Raw counts are an artifact of the MC generation
multiplicities — the physically meaningful balance is between **expected
rates**. The ratio is now computed from

```
W = BaseWeight × PhysicsWeight          # per-event expected rate,
r = Σ W[donor] / Σ W[acceptor]          # pre-detector-nuisance
```

(the rate-weighted convention of the published SK analysis). With
weighted rates, the total *expected rate* — not the raw event count — is
conserved across the migration. The same basis is used for the DecayE
occupancy fractions in `decay_e_tagging`.

All of this is centralized in three helpers on `SuperK_Combined`
(`pynu/PhysicsTunes/Detector/SKCombinedDetector.py`): `_rate_weight`,
`_migration_ratio`, `_mask_ratio`. The migration algebra itself is unchanged.

### Why the analytic gradient stays exact

`W = BaseWeight × PhysicsWeight` is evaluated *before* detector nuisances are
applied and does not depend on the nuisance vector. The paired `diff_*`
derivatives treat `r` as a constant (acceptor term `−r`), so they remain
**exact** under the new basis — the L-BFGS-B fit keeps its analytic Jacobian
(no finite differences). This is also why the per-evaluation Jacobian refresh
in the row worker (commit `573f4a9`) is valid.

### New opt-in dials (granularity of the published SK analysis)

These are inactive unless enabled in the XML (`<status> 1 </status>`), so
existing configurations are unaffected:

- **`barr_zenith`** (`AtmoFlux`): Barr-style energy-damped up/down flux
  asymmetry, `w = (1 + env(E)·x)^tanh(3 cosθz)` with
  `env(E) = 0.07/(1+(E/0.5 GeV)²)`; prior N(0, 1). Two-sided and
  rate-preserving across the horizon — intended to *replace* the one-sided
  `zenith_up`/`zenith_down` pair, not run alongside them.
- **`neutron_tagging_subgev` / `neutron_tagging_multigev`**
  (`SKCombinedDetector`): era-split n-tag efficiencies (SubGeV
  {20,22}↔{21,23}, MultiGeV {25,27}↔{26,28}; prior N(1, 0.12)), where the
  shared `neutron_tagging` ties both eras to one parameter.

Ready-made configs: `analysis/AnalysisFiles/SK2023_Atm_datafit_xsec_barr.xml`,
`..._barr_ntag.xml`, `..._zen004.xml`.

### Back-compatibility: using the old raw-count migration

The basis is switchable through `SuperK_Combined.MIGRATION_BASIS`
(default `"weighted"`). `"raw"` makes `_rate_weight` return ones, which
reduces every sum to an unweighted event count — **bit-for-bit the original
implementation** for all migration tunes, neutron tagging (including the
era-split dials), and `decay_e_tagging`. Two ways to select it:

```bash
# 1. Environment variable (read at import time) — no code changes:
export PYNU_SK_MIGRATION_BASIS=raw
python run_sk_datafit_row_worker.py ...
```

```python
# 2. Class attribute, set any time before expectations are built:
from pynu.PhysicsTunes.Detector.SKCombinedDetector import SuperK_Combined
SuperK_Combined.MIGRATION_BASIS = "raw"
```

Notes:
- Both bases are nuisance-independent, so the analytic `diff_*` derivatives
  are exact either way; the switch is safe with the gradient-based fit.
- The switch covers the *migration rate basis only*. The other fixes on this
  branch (WMC/variance consistency, likelihood guard, worker refresh) are
  unconditional bug fixes with no legacy mode.
- The binned engine (below) hard-codes the weighted-rate analog; it does not
  follow this switch.

---

## 2. Worker/minimizer fixes (`analysis/SuperK-datafit/run_sk_datafit_row_worker.py`)

- **Per-evaluation refresh**: the nuisance minimization used to freeze the
  Barlow-Beeston MC variance at `x0` and evaluate the analytic Jacobian at
  *nominal* (`ComputeBinnedDiffExpectation()` with no argument defaults to
  `NuisNominalList`), so L-BFGS-B descended on a stale gradient. Expectation,
  `SetBinnedMCVariance()`, and `ComputeBinnedDiffExpectation(nuisance_vector=…)`
  are now recomputed at the current point on every evaluation, with `f` and
  `g` sharing one expectation build (`jac=True`).
- **`StartPhysics()` inside `set_physics_params`**: `ApplyOscillations`
  *multiplies onto* the existing `PhysicsWeight`; standalone callers that did
  not reset first got `Φ_osc(NOM) × Φ_osc(point)` — a double application that
  crushed the CC sector ~48× (NC, force-set to 1, was immune). The fit loop
  already reset per point, so **fit results are bit-identical**; the reset
  makes the helper correct for every caller (extraction/diagnostic scripts).

---

## 3. The binned forward engine (`analysis/SuperK-datafit/sk_binned/`, EXPERIMENTAL)

### What it is — and what it is not

It is **not** a new experiment class, and it does **not** plug into
`PyNuFit`/the row-worker machinery. It is a standalone two-stage pipeline:
the existing event-by-event framework is used **once, at build time**, to
derive binned response artifacts; fitting then runs in pure numpy/scipy with
no Pynu import at all. The event engine remains unchanged and authoritative —
the binned engine mirrors it at true-grid resolution and is validated against
it.

The forward model is

```
N_b = [ Σ_k  R_k[c,b]ᵀ · ( Φ_k(c) · flux(c,k) · xsec_k · axial(c_E)^CC_k ) ] · Π_det d_b
```

where `R_k[c,b]` is a sparse response matrix (sum of `BaseWeight` over events
of class `k` in true cell `c = (c_E, c_Z)` landing in reco bin `b`), `Φ_k(c)`
is the oscillated flux on the true grid, and the flux/xsec/detector tune
forms are ported verbatim from `AtmoFlux`, `WaterXSection`, and
`SKCombinedDetector`. Migration ratios use the binned analog of the weighted
basis (`N_phys = Σ_k R_kᵀ Φ_k` — exactly equal to the event-engine value,
since bins partition events).

### Does it need different row workers?

**No row workers at all for fitting.** The division of labor:

| Stage | Where it runs | Needs Pynu? | Script |
|---|---|---|---|
| 1. Build response matrices | cluster (one array job) | yes — one MC pass *through* `SuperK_2023` so all conventions (NC `w_no`, NORM, WMC, CC-mask) are inherited, never re-implemented | `build_sk_response.py` → `sk_response.npz` |
| 2. Build oscillation tensors | same array job | yes — Pynu oscillation machinery per grid point | `build_osc_tensors.py` → `osc_tensors/*.npz` |
| 3. Fit | **locally, minutes** | no — pure numpy/scipy | `fit_sk_binned.py` |
| 4. Validate vs event engine | locally | no | `validate_binned_vs_event.py` (Gate-B reference built by `eval_event_engine_vectors.py`) |

Once stage 1–2 artifacts exist, the entire grid scan is a local loop — that
is the point of the engine: the per-point cost collapses from a full
event-weight pipeline pass to a few sparse matrix contractions, so nuisance
studies / likelihood variants / grid scans iterate without SLURM.

The fitter uses the **SK-official Eq. 10 likelihood** (Poisson LLR + Gaussian
pulls, *no Barlow-Beeston*), with dCP profiled over 13 precomputed values per
grid point. This is deliberately the published-analysis likelihood, distinct
from the framework's `BarlowBeestonLikelihood` — useful for apples-to-apples
comparison with the SK paper.

### Usage

```bash
# one-time build (cluster; ~228-task array: response + 225 osc tensors + Gate-B ref)
sbatch analysis/SuperK-datafit/sk_binned/submit_sk_binned_build.sh

# fit named points or the full grid (local)
python analysis/SuperK-datafit/sk_binned/fit_sk_binned.py \
    --xml analysis/AnalysisFiles/SK2023_Atm_datafit_xsec_barr_ntag.xml \
    --points pointA pointB skbf        # or --grid

# validation gates vs the event engine
python analysis/SuperK-datafit/sk_binned/validate_binned_vs_event.py \
    --xml analysis/AnalysisFiles/SK2023_Atm_datafit_xsec_barr_ntag.xml
```

All result paths default to `analysis/SuperK-datafit/sk_binned/results/`;
the submit script resolves everything `SCRIPT_DIR`-relative. The build
scripts import `setup_pynufit_datafit` / `set_physics_params` from the
neighboring `run_sk_datafit_row_worker.py`.

### Status

**Experimental.** The Gate-A/Gate-B numerical-agreement targets (binned vs
event engine, < 0.1%) have not been signed off yet; `validate_binned_vs_event.py`
additionally expects event-engine reference files under
`results/pe_mc_comparison/` that are currently produced in the development
repo. Treat event-engine results as authoritative until the gates pass.

---

## 4. Other fixes

- **`BinnedLogLikelihoodRatio.stats`** (`27fff9b`): the non-positive-expectation
  guard was `np.any(E) <= 0` — a no-op typo for `np.any(E <= 0)` — so a
  nuisance step driving a bin's model ≤ 0 reached `O·log(O/E)`, produced NaN,
  and hung the minimizer. Relevant to any bare-Poisson fit (no per-bin BB β
  to keep bins positive).
- **`SuperK_2023` MC loading** (`ae5f6d6`, `de7066a`): see commit messages —
  the CC-mask encoding bug silently classified *all* events as NC with
  `|S1`-encoded files, disabling every CC xsec systematic; the WMC/variance
  fixes make the BB variance consistent with the model weight
  `BaseWeight = Weight × NORM × WMC`.
