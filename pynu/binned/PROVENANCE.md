# Provenance — vendored snapshots in `pynu/binned/`

`sk_binned_engine.py` and `interp_engine.py` are vendored snapshots of the SK
binned forward model and the oscillation-tensor interpolator, developed in the
maintainers' development repository and copied in so the Pynu package is
self-contained. Functional edits are never made to the vendored code in place;
changes land as full-file resyncs from the development source. The adapter
(`adapter.py`), config (`config.py`), builder (`builder.py`), energy-scale
operator (`escale_operator.py`), and package init are native to this package.

The vendored files are pure `json` / `numpy` / `scipy`; the `.xml` references in
`sk_binned_engine.py` are comments plus an *optional* nuisance-spec path
(`resolve_nuisance_spec` accepts a Pynu analysis XML), so no data file is needed
to import the module.

## Current snapshot

| file | source | hash |
|---|---|---|
| `sk_binned_engine.py` | maintainers' development repo, commit `22d1374`, comment-normalized for release (comments/docstrings only — the module AST is identical to the source modulo three docstring constants) | md5 `ac46cfa801e6049278bbe5b256cd5695`, sha256 `0bc1b18e1530aef4f12886bd562ca7865ee55bda8147b3fb11dd679c53ee13d6` |
| `SK2023_Atm_datafit_r2_fude_ccqe.xml` | development repo, commit `e35921a` — nuisance-activation manifest for the 131-dial `r2_fude_ccqe` spec | — |
| `SK2023_Atm_datafit_r2_fude_ccqe_{nmig,nmig_pinned,dcye,nmig_dcye}.xml` | development repo, commit `22d1374` — activation manifests for the four spec variants (133/133/132/134 dials) | — |
| `interp_engine.py` | maintainers' development repo (frozen copy), comment-normalized for release (module docstring only — AST identical to the source otherwise) | sha256 `8da7d65068f8e4e40392251d3f64ee9b0b148127091c6155a8ecb58c7cdddf46` |

Numerics are certified by the in-repo parity gate: the modular adapter/PyNuFit
path must reproduce the direct engine chi2 / gradient / MC-variance with
diff = 0.0 (nominal + random nuisance draws). The gate is re-run after every
resync and after the comment normalization above (PASS, diff = 0.0).

## Resync protocol

The adapter consumes only the engine's **stable surface**: the `SKBinnedEngine`
constructor (by KEYWORD — `response_path`, `migration_mode=`, `likelihood=`,
`nuisance_spec=`; extra optional positionals like `solar_mix_f` may be inserted
upstream without breaking keyword callers), `chi2`, `chi2_and_grad`, `fit_point`,
the `nuisance_names` / `nominal` / `sigma` attributes, and `resolve_nuisance_spec`;
from the interpolator: `PhiInterpolator` and `detect_grid`. So a resync is a
drop-in file replacement:

1. Verify the source file's hash against its commit BEFORE copying; STOP if it
   differs. Then copy (`cp <source> sk_binned_engine.py`) or
   `git show <commit>:<path> > …`.
2. Update the snapshot table above (source commit + hashes).
3. Re-run the parity gate (chi2 / gradient / mc_var diff = 0.0) after *every*
   engine resync (the gradient is what `fit_point` uses). Confirm the ctor is
   still keyword-compatible and any new named specs resolve.
