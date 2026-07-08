# Provenance — vendored snapshots in `pynu/binned/`

`sk_binned_engine.py` and `interp_engine.py` are **verbatim snapshots** of the SK
binned forward model + osc-tensor interpolator. They are copied in (not imported
from the `claude/...` subprojects) so the Pynu package is self-contained and the
integration surface is stable across the parallel investigation branches. No
functional edits are made to the vendored code — only the adapter (`adapter.py`),
config (`config.py`), and package init are new.

The vendored files are pure `json` / `numpy` / `scipy`; the `.xml` references in
`sk_binned_engine.py` are comments plus an *optional* nuisance-spec path
(`resolve_nuisance_spec` accepts a Pynu analysis XML), so no data file is needed
to import the module.

## Snapshot provenance (current — resynced 2026-07-08)

| file | source | md5 | sha256 |
|---|---|---|---|
| `sk_binned_engine.py` | `sk-pe-reproduction` **commit `e35921a`** ("SK binned: named spec + XML manifest for production r2_fude_ccqe arm") at `claude/2-atmospheric-oscillation/SuperK/binned-octant/src/sk_binned/sk_binned_engine.py` — the ADOPTED-PRODUCTION (`r2_fude_ccqe`, 131-dial) engine | `85b881ff9a660a61463042c9c3dd6a41` | `b85370fad4741e44cf477d34c01b9f6d057d80fd0231d06f13895f673710d564` |
| `SK2023_Atm_datafit_r2_fude_ccqe.xml` | same commit, sibling file — nuisance-activation manifest for the 131-dial `r2_fude_ccqe` spec (see contents note below) | — | — |
| `interp_engine.py` | `claude/2-atmospheric-oscillation/SuperK/binned-mcmc/src/sk_mcmc/interp_engine.py` (vendored frozen copy on this branch; untracked working-tree "committed-pending" interp pipeline) | — | `10fca075a4ac16709cf64875154c5f60996193f140a9657015247f2755785727` |

Verify at any time:

```
md5 -q Pynu/pynu/binned/sk_binned_engine.py     # 85b881ff9a660a61463042c9c3dd6a41
shasum -a 256 Pynu/pynu/binned/sk_binned_engine.py Pynu/pynu/binned/interp_engine.py
```

### Contents of the r2_fude_ccqe engine (vs the prior `e7e472b2` UBS snapshot, 07-03)

Superset of the 07-03 UBS snapshot, adding — this is the branch's first resync
to a COMMITTED source (`e35921a`, not a working-tree snapshot) since the
07-03 UBS resync, and folds in everything landed on `sk-pe-reproduction`
between commits `b10126c`→`4993319`→`e35921a`, all **default-OFF** unless the
`r2_fude_ccqe`/`R2FUDECCQE` spec (or its constituent named specs) is selected:
- Track H multi-GeV CCQE appearance freedom: `xsec_ccqe_shape` (global CCQE
  shape) + `xsec_ccqe_multigev_nue`/`xsec_ccqe_multigev_numu` (flavor norms,
  `MULTIGEV_CCQE_NAMES`) — the **first Δm²-moving arm** found in the campaign,
  now ADOPTED PRODUCTION (2026-07-06, 3D θ13-cube pergrid);
- Track U up/down energy-scale dials `updown_escale_sk{1,2,3,45}`
  (`UPDOWN_ESCALE_NAMES`) + the named spec `R2UDE` (124 dials) — Δm² NULL,
  thesis-vocabulary-complete;
- Track D direction-smearing dial (`DIR_SMEAR_NAME`) + the named spec `R2DS`
  (121 dials) — Δm² NULL, resolution axis closed;
- **NEW named spec `R2FUDECCQE`** (aliases `r2_fude_ccqe`, `ladder_r2_fude_ccqe`,
  131 dials) = `R2` + `flux_horizvert` + the UBS triplet + the UDE quartet +
  `xsec_ccqe_shape` + the two multi-GeV CCQE flavor norms — reproduces the
  exact production arm. Verified byte-identical, in order, to
  `claude-tmp/dm2/seeds_r2_fude_ccqe/r2_fude_ccqe.json:nuisance_names` (on
  `sk-pe-reproduction`, at commit time).
- **NEW file `SK2023_Atm_datafit_r2_fude_ccqe.xml`**: a nuisance-activation
  manifest (`<nuisance name='..'><status>1</status></nuisance>` blocks, no
  `<type>/<box>/<prior>` content) consumed by `resolve_nuisance_spec(<path>)`
  — gives a third, XML-driven way to select the same 131 dials, for anyone
  running off this branch without the seed-JSON pipeline. NOT a full
  event-level PyNuFit Analysis config (see header comment in the file).
None of these change prior behaviour unless explicitly activated.

### Resync history

- **2026-07-02** — original snapshot = committed `b10126c` (tip of
  `sk-pe-reproduction`, "SK dual-mode+IO round"), sha256 `313a3bc9…`. Chosen over
  the then-uncommitted live tree because a parity gate against a moving,
  unreviewed target is self-defeating.
- **2026-07-03** — resynced to the UBS working-tree engine (then-uncommitted) to
  run the `R2UBS` up-μ-background arm through `pynu.binned` (native fine-grid
  scan). The live UBS additions were Tier-3-verified default-OFF, so the
  resync was a reviewed drop-in; gate 2 + gate 3 re-run after the swap.
- **2026-07-08** — resynced to `sk-pe-reproduction@e35921a` (first COMMITTED
  resync source since 07-02) to bring the branch's binned engine and CCQE
  systematics up to the ADOPTED-PRODUCTION `r2_fude_ccqe` state, plus the new
  `R2FUDECCQE` named spec + XML manifest (both added in that same commit,
  specifically so this resync could happen). Gate 2 + gate 3 re-run against
  the phased response (`claude-tmp/skb_phased/sk_response.npz`, local); see
  gate log in the resync commit on this branch.

## Resync protocol

The adapter consumes only the engine's **stable surface**: the `SKBinnedEngine`
constructor (by KEYWORD — `response_path`, `migration_mode=`, `likelihood=`,
`nuisance_spec=`; extra optional positionals like `solar_mix_f` may be inserted
upstream without breaking keyword callers), `chi2`, `chi2_and_grad`, `fit_point`,
the `nuisance_names` / `nominal` / `sigma` attributes, and `resolve_nuisance_spec`;
from the interpolator: `PhiInterpolator` and `detect_grid`. So a resync is a
drop-in file replacement:

1. If resyncing from a working-tree (uncommitted) source, verify its md5 against
   the value the assigner provides BEFORE copying (another session may have
   touched it); STOP if it differs. Then copy
   (`cp <source> sk_binned_engine.py`) or `git show <commit>:<path> > …`.
2. Update the source / md5 / sha256 rows + the contents note above.
3. Rerun **gate 2** (`binned-native/tests/test_parity.py`, adapter-vs-direct) and
   **gate 3** (`test_parity.py` gradient block) against the new snapshot. Rerun
   gate 3 after *every* engine resync (the gradient is what `fit_point` uses).
   Confirm the ctor is still keyword-compatible and any new named specs resolve.
