# Provenance — historical record (the vendoring era ended at Track S / E6)

**Status: the `pynu/binned/` package is now native.** This file is kept as the
historical record of the vendoring era, not a description of the present state.

## History

`sk_binned_engine.py` and `interp_engine.py` began life as **vendored snapshots**
of the SK binned forward model and the oscillation-tensor interpolator, developed
in the maintainers' development repository and copied in so the Pynu package was
self-contained. During that era functional edits were never made in place —
changes landed as full-file resyncs from the development source, and the vendored
files were kept byte-verifiable against their source commit (the resync
discipline). Dial *values* lived in an in-code table (`CANONICAL_DIALS`), and an
`adapter.py` bridged a PyNuFit physics point to the standalone engine + minimizer.

The Track S de-vendoring (Phases E1–E6) retired that arrangement:

- **E1** — the structural/numerical kernels moved into the native
  `engine_core.py`; `interp_engine.py` was adopted as owned code.
- **E2/E3** — dial-value authority inverted from `CANONICAL_DIALS` to XML; the
  named-spec string dispatch became XML activation manifests.
- **E4/E5** — masks/selectors, the flux/xsec cell weights, and the detector
  factors moved to native descriptor modules (`masks.py`, `grid_experiment.py`,
  `detector.py`) running the real PhysicsTunes methods.
- **E6** — `CANONICAL_DIALS`, the string dispatch, the two hand-inlined parity
  shadows, and `adapter.py` were **deleted**. The two SK dial-value XMLs now ship
  as **package data** under `pynu/binned/` (the sole authority for dial values);
  the 26 activation manifests moved to `analysis/AnalysisFiles/`; φ lookup/caching
  became `engine_core.TensorStore`, and the former adapter's holding role became
  `engine_core.BinnedBinding` (staged (phi, theta) now live on PyNuFit's modular
  methods).

So `pynu/binned/` ends with **zero vendored files**. The package is pure `json` /
`numpy` / `scipy`; importing it needs no analysis data file (the value XMLs are
package data and always present in an installed package).

## Snapshot hash history (for the record)

These are the hashes of the last vendored snapshots, before the E1–E6 native
port rewrote the files. They are retained for traceability, not as a live
verification target.

| file (at the vendored tip) | source | hash |
|---|---|---|
| `sk_binned_engine.py` | maintainers' development repo, commit `22d1374`, comment-normalized for release (AST identical to the source modulo three docstring constants) | md5 `ac46cfa801e6049278bbe5b256cd5695`, sha256 `0bc1b18e1530aef4f12886bd562ca7865ee55bda8147b3fb11dd679c53ee13d6` |
| `interp_engine.py` | maintainers' development repo (frozen copy), comment-normalized for release (module docstring only — AST otherwise identical) | sha256 `8da7d65068f8e4e40392251d3f64ee9b0b148127091c6155a8ecb58c7cdddf46` |
| `SK2023_Atm_datafit_r2_fude_ccqe.xml` (now in `analysis/AnalysisFiles/`) | development repo, commit `e35921a` — activation manifest for the 131-dial `r2_fude_ccqe` spec | — |

## Certification

Numerics were certified throughout the port by the in-repo parity gates: the
modular PyNuFit path reproduces the direct engine chi2 / gradient / MC-variance
with diff = 0.0 (nominal + random nuisance draws), and every E-phase compared the
edited engine byte-for-byte against a frozen copy of the pre-port engine
(`frozen_referee_S`, md5 `ac46cfa801e6049278bbe5b256cd5695`). E6 re-ran the full
Gate-C rerun + E1 binding + E2 value-parity + E3 manifest resolver, all PASS with
diff = 0.0.
