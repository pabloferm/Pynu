# Provenance — IC & ORCA binned engines

Port basis: **AtmNuDataFit commit `8f6f098`** (branch `sk-pe-reproduction`),
landed on top of `datafit-SK-mcmc` at `da3ad34`. Specification of record:
`UPSTREAM_PORT_PLAN_icorca_binned_2026-08-17.md` sections 1 (rulings), 3
(per-file tables), 6 (gate C-1).

## The faithful-port rule

Every module below ships **byte-identical to its certified dev-tree source
except for one contiguous import-header block**, whose only edits are dropping
`sys.path` bootstrapping and rewriting cross-module imports to package-relative
form. No reformatting, no renamed symbols, no numeric changes. This is not
stylistic conservatism: the certifications in the table at the bottom were earned
by *these bodies*, and any restructuring would void them and force a full
re-gate.

`test/binned_icorca/compare_ported_modules.py` (gate **C-1**) mechanizes the
rule. For each module it asserts that (a) every line outside the declared header
region is byte-identical to the dev source, and (b) every statement inside the
declared region parses as import-header material — so a too-wide region cannot
hide a body edit. `orca_binned_support.py`, being an extraction rather than a
move, is compared region by region against its source. Run it from the repo root:

```
python3 test/binned_icorca/compare_ported_modules.py [--dev-root <AtmNuDataFit>]
```

## Source map

Dev-tree paths are relative to the AtmNuDataFit checkout at `8f6f098`:

- `A/` = `claude/2-atmospheric-oscillation/combined-fit/binned_arms/`
- `C/` = `claude/2-atmospheric-oscillation/combined-fit/`
- `M/` = `claude/2-atmospheric-oscillation/multi-experiment-systematics/`

| Upstream file (`pynu/Experiments/`) | Dev source | Lines | Diff vs source | Certifying gate |
|---|---|---:|---|---|
| `orca_binned_engine.py` | `A/orca_binned_engine.py` | 441 | header 49–58 (was 49–63): dropped `import sys` + two `sys.path.insert` + the `_HERE` anchor; `orca_exact_scan` → `.orca_binned_support`, `binned_contract` → `.binned_contract`, `binned_dial_fields` → relative | G-ORCA-1, G-ORCA-2 |
| `orca_binned_support.py` | extracted from `C/orca_exact_scan.py` | 120 | NEW module, new docstring + imports; the 7 extracted regions are byte-identical to source lines 48–49, 70–71, 74–79, 82–96, 99–122, 125–130, 185–204 | G-ORCA-1, G-ORCA-2 (it holds their reference model) |
| `ic_binned_support.py` | extracted from `M/ic_divergence_scan.py` | 154 | NEW module, new docstring + imports + one section-divider comment; the 12 extracted regions are byte-identical to source lines 93–94, 96–98, 108–111, 114–116, 119–120, 123–131, 134–143, 146–176, 179–184, 235–236, 239–252, 255–265 | G-IC-3 (it holds its reference model) |
| `orca_cell_phi.py` | `A/orca_cell_phi.py` | 206 | header 28–30 (was 28–34): dropped `import os`/`import sys` + the `sys.path.insert`; `orca_exact_scan` → `.orca_binned_support` | G-ORCA-0 |
| `orca_binned_builder.py` | `M/orca_binned_builder.py` | 274 | header 21–33 (was 21–35): dropped `import sys` + the `sys.path.insert`; `orca_exact_scan` → `.orca_binned_support`. CLI args are all `required=True`, so no dev-path defaults existed to retarget | C-2 (builder determinism) |
| `ic_binned_engine.py` | `A/ic_binned_engine.py` | 345 | header 104–108 (was 104–113): the dual script/package `try/except ImportError` collapsed to plain relative imports, and `ic_cells` → `ic_binned_cells` for the rename below | G-IC-3, G-IC-4 |
| `ic_binned_cells.py` | `A/ic_cells.py` | 257 | **renamed** for clarity; header 36–42 (was 36–45): `try/except ImportError` collapsed to `from .ic_dial_fields import ICCellGeom` | G-IC-3 |
| `ic_dial_fields.py` | `A/ic_dial_fields.py` | 334 | **verbatim** — no header edit | G-IC-3 |
| `ic_binned_builder.py` | `M/ic_binned_builder.py` | 358 | **verbatim** — no header edit; CLI args are all `required=True` or ladder-internal defaults, so nothing pointed at a dev path | C-2 |
| `binned_contract.py` | `A/binned_contract.py` | 84 | **verbatim** — pure numpy, zero experiment content | inherited by both engines' gates |
| `binned_dial_fields.py` | `A/binned_dial_fields.py` | 429 | **verbatim** | G-ORCA-2, G-IC-3 |

Placement follows the two user rulings of 2026-08-17: **D1** — ship
`binned_dial_fields.py` as the single certified file rather than splitting it
per-experiment (a split would duplicate ~23 dial forms and force re-certifying
both engines); **D2** — place `binned_contract.py` in `pynu/Experiments/`,
keeping `pynu/fitter/` untouched and the kernels next to their only consumers.
The per-experiment-separation ruling is honored at the engine, mask and builder
level: there is no unified engine core across SK / IC / ORCA.

## Certifying gates

Quoted from the port plan sections 1 and 6 and the certifying runs of
2026-08-17. These are the numbers the ported bodies earned; re-running them
upstream is gate **C-3**.

| Gate | What it certifies | Result |
|---|---|---|
| **G-ORCA-1** | engine expectation ≡ event-path expectation on the flat900 response | **PASS 55/55** (worst relative 5.647e-05 = 2.58x the measured event-path floor) |
| **G-ORCA-2** | analytic gradient vs finite differences, all 30 ORCA dials | **PASS 58/58** (worst ratio-to-model-difference 1.52x under a 5e-3 bound) |
| **G-IC-3** (`gate_ic_identity.py`) | IC engine vs the production IC term at nominal dials, 5 osc cells | **PASS**, summation-order class: **4.688e-15 (L3)**, 4.236e-15 (L1), threshold 1e-9 |
| **G-IC-4** (`ic_dial_divergence_scan.py`, **does not ship**) | post-fit dial-side divergence vs the live event path | **BOUNDS ≤ 0.053** — scatter-limited at ftol 1e-5; the binning cost itself is *not resolved*. This gate has no pass/fail threshold: it *produces* the number. |

Per-evaluation reference costs on the certifying hardware: IC 29 ms, ORCA 9.4 ms
per gradient.

## Amendment to plan section 3 — the two reference models ship

**Raised by the P3 gate port, 2026-08-17; both verified against the gate sources
before acting.**

The plan's section 3.1 O2 listed `binned_expectation` among the "scan-driver
parts that do not ship", and section 3.2 had no slot at all for an IC support
module. Both exclusions were wrong, for the same reason: these functions are not
driver conveniences, they are **the production reference models the engines are
certified against**. Excluding them would have shipped engines whose gates cannot
run upstream — precisely the reproducibility the port exists to deliver.

| Function | Where the gate needs it | Consequence of the exclusion |
|---|---|---|
| `orca_exact_scan.binned_expectation` | `gate_orca_identity.py` imports it at :70–73 and uses it as the reference model at :430 (χ² comparison), :500 (the `_RefArm` gradient reference) and :511 (per-bin model difference) | G-ORCA-1 **and** G-ORCA-2 cannot run at all |
| `ic_divergence_scan._corrected_expectation` + `_hs_correction_factor` + `_hs_params_from_theta` (and the `observed_200` / `muon_200` / `nu_index` / `POINTS` / `poisson_chi2` helpers they sit on) | `gate_ic_identity.py:57` does `import ic_divergence_scan as H` and calls `H.*` at :71–100 | G-IC-3 cannot run at all |

Both are now extracted under the same discipline as the rest of the port —
byte-identical bodies, region-compared by gate C-1 — into
`orca_binned_support.py` and the new `ic_binned_support.py`. Neither adds a
module-level pynu import: both take the live `PyNuFit` / experiment object as a
function argument, so the modules stay locally importable.

### Plan section 6's gate-name table is wrong; the source self-labels are authoritative

Section 6 of the port plan attributes **G-IC-3** to `gate_ic_engine.py` ("IC
engine vs reference values, 4.7e-15 class agreement", with
`gate_ic_engine_L3.json` as its "vendored reference numbers") and **G-IC-4** to
`gate_ic_identity.py`. Both attributions are wrong. Five independent checks
against the dev tree:

1. `gate_ic_identity.py` **self-labels G-IC-3** in four places — the module
   docstring (:1), the banner print (:61), the PASS line (:119), and the artifact
   schema `json.dump({"gate": "G-IC-3", ...})` (:122). Its :7 names G-IC-4 as a
   *different* gate whose dial-side residue vanishes at nominal dials.
2. `gate_ic_engine.py` contains **zero** `G-IC-*` labels (`grep -c` = 0). It
   self-labels G-G1 (per-dial Richardson gradient), G-G2 (zero invariants) and
   construction checks.
3. `gate_ic_engine.py` never reads `gate_ic_engine_L3.json` — there is no
   `json.load` in the file. That JSON is its `--out` **output artifact**, not a
   reference input, so "engine vs vendored reference numbers" does not describe
   it.
4. The 4.688e-15 / 4.236e-15 figures do not appear in `gate_ic_engine_L3.json`
   at all. Its headline is `worst_g1_rel = 4.490910190275778e-08`.
5. `ic_engine_identity.sbatch:10`, the submitter that runs
   `gate_ic_identity.py`, reads: *"G-IC-3 — ICBinnedEngine vs the production IC
   term at NOMINAL dials, 5 osc cells."*

So **`ic_binned_support.py`'s certifying gate is G-IC-3**, the summation-order
identity against the production IC term (4.688e-15 L3 / 4.236e-15 L1, threshold
1e-9) — which is exactly what the gate table above describes under that name.

**G-IC-4 is a different script that does not ship in this port**:
`combined-fit/ic_dial_divergence_scan.py` (:4, :331), the post-fit dial-side
divergence measurement. It has no pass/fail threshold — ":550 — G-IC-4 has NO
pass/fail threshold; this gate PRODUCES the number" — and the ≤ 0.053 figure is
its output, not a criterion. It is retained in the gate table above only because
it bounds the engine's post-fit behavior.

*Record of the error:* an earlier revision of this document relabeled
`gate_ic_identity.py` as G-IC-4 on the authority of plan section 6, which
contradicted both the gate sources and this document's own gate table. P3
supplied the evidence above and it verified in full. **Plan section 6 needs the
correction, not the gates.** The gate scripts stay source-faithful — relabeling
them would change printed gate identity and the `"gate"` JSON field, which is
neither a port concern nor within the porting mandate.

A third item raised at the same time needed no change: `ORCA_MANIFEST` is and
stays exported from `orca_binned_engine.py`. Note that it is the same 30-dial
*set* as the non-shipping worker's `SHARED_FLUX + SHARED_XSEC + ORCA_DET` in a
**different order** (`tilt` sits at index 2 here, index 4 there). Do not
"reconcile" them: the manifest order is the XML order that `_project_theta`
delivers theta in, and it is what the gradient's index map is certified against.

## Deliberate omissions

Recorded so their absence is not read as an oversight:

- **The 3-experiment combined fit does not ship.** `combined_3exp_fit_worker.py`,
  `combined_ic_orca_fit_worker.py`, `memo_store.py`, the grid minimizer and the
  FD-pool infrastructure are private analysis machinery (SK+IC+ORCA combined
  results are unpublished). Upstream gets the engines, the builders, and
  single-experiment drivers.
- **`add_pynu_root` is not extracted** into either support module: no extracted
  function calls it, and gate C-1 asserts on the AST that neither shipped module
  holds a reference to it.
- **`event_expectation` is not extracted** — no gate imports it.
- **The remaining scan-driver parts** of `orca_exact_scan.py` and
  `ic_divergence_scan.py` (`selftest`, `main`, `run_fasrc`, the draw and
  dial-class machinery) stay out of the package; their upstream equivalents
  belong under `analysis/ORCA-binned-datafit/` and `analysis/IC-binned-datafit/`.
- **The experiment XMLs** (`ORCA_Atm_r2_fude_ccqe.xml`,
  `IC_DeepCore_r2_fude_ccqe.xml`) are dev-tree-only and not in the ship list, so
  the gates' `--config` / `--xml` arguments stay required with no repo-relative
  default.
- **No derived `.npz` is committed** — responses, φ tensors and HS artifacts are
  rebuilt deterministically from `data/` by the shipped builders. (npz file
  hashes never match across machines anyway: zip timestamps.)
- **Snapped-ladder IC v2 responses** stay out pending the B3 band-edge ruling;
  the mode-axis unsnapped build is the shipped production response.
- **No SK file is touched.** The five SK binned modules are the project's
  most-audited artifact; this port adds files beside them and imports
  `sk_binned_builder`'s certified osc helpers read-only.

## Known stale reference

`ic_binned_engine.py`'s module docstring (line 4) still names
`ic_cells.ICCells`, the pre-rename module path. The docstring is a body line
under the faithful-port rule, so it was **not** edited — correcting it would put
a non-header diff into a certified module. The module is `ic_binned_cells.py`
upstream; fix the docstring in a later, separately-gated cleanup pass.

Two other pre-existing `sys.path` / dev-path mentions survive for the same
reason, and neither is a functional dev-path dependency:
`ic_dial_fields.py:249` calls `sys.path.insert(0, pynu_root)` inside its
`selftest()` local gate, where `pynu_root` is a CLI argument; and
`binned_contract.py:8` cites a dev-tree path in the docstring that records why
the kernels are a transcription rather than an import of the SK core module.
