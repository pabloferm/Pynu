# PyNu binned scans: Super-Kamiokande, IceCube-DeepCore and KM3NeT/ORCA6

This branch reproduces the oscillation-parameter scans of

> M. Jin, C. A. Argüelles, P. Fernández-Menéndez, I. Martínez-Soler,
> *Evidence for Normal Neutrino Ordering with Combined Atmospheric Neutrino Experiments*,
> arXiv:2610.12208.

It contains only what those scans need: the binned likelihood engines of the three
experiments, the code that builds their inputs from the public data releases, and
one configuration file per scan. The rest of PyNu lives on the other branches of
this repository.

## The four scans

| configuration | experiments | lattice region | ordering of record |
|---|---|---|---|
| `configs/SK.xml` | Super-Kamiokande I-V | 20 × 20 cells around each best fit, 10 δCP nodes | NO and IO |
| `configs/IC.xml` | IceCube-DeepCore (8 yr) | full 41 × 41 × 20 lattice | NO and IO |
| `configs/ORCA.xml` | KM3NeT/ORCA6 | full 41 × 41 × 20 lattice | NO and IO |
| `configs/Combined.xml` | SK + IC + ORCA | 20 × 20 cells around each best fit, 20 δCP nodes | NO and IO |

The lattice is |Δm²₃₁| ∈ [1.6, 3.0] × 10⁻³ eV² (41 values; the inverted ordering uses
Δm²₃₁(IO) = −(Δm²₃₁(NO) − Δm²₂₁) with Δm²₂₁ = 7.41 × 10⁻⁵ eV²), sin²θ₂₃ ∈ [0.30, 0.70]
(41 values) and δCP = k · 2π/20 (k = 0…19). sin²θ₁₃ = 0.022, sin²θ₁₂ = 0.303. At every
lattice node all nuisance parameters are fitted.

Each configuration is self-contained. It holds the input files with their checksums,
the lattice and region, the minimizer protocol, each experiment's parameter order, and
the full nuisance-parameter table. For each parameter that is the prior centre and width, the
fit bounds, the start value, and whether it is pinned. The code derives the bounds by
fixed rules and stops if they differ from the table, so the file states exactly what
was fitted. Each configuration also lists reference cells from the published scans.
`scan check` refits them.

## Layout

```
configs/            the four scan configurations
configs/events/     analysis XMLs read by the builders (MC, data, exposure; no fit parameters)
data/SuperK/        SK observed counts per analysis bin; FLUKA low-energy flux table
data/IceCube/       IceCube-DeepCore 8-year release files used here
data/ORCA/          KM3NeT/ORCA6 release files used here
pynu/binned/        fit stage: engines, fits, scan driver (numpy + scipy only)
pynu/build/         build stage: response matrices, oscillation tensors, inputs
pynu/               PyNu event framework, used by the build stage only
scripts/build.py    build-stage command line
scripts/scan.py     scan command line (same as python -m pynu.binned.scan)
tests/              test suite (tests/reference/: published values it checks against)
COPYING, COPYING.LESSER   license (LGPL-3.0-or-later)
```

## Installation

Python 3.11 or 3.12. The fit stage needs only numpy and scipy. The build stage also
needs nuSQuIDS, nuflux, pandas, pyarrow, h5py and boost-histogram.

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

nuflux publishes Linux wheels only; on macOS install it from conda-forge. nuSQuIDS
reads its Earth model and cross sections from `NUSQUIDS_DATA_PATH`; point it at the
`data/` directory of the nuSQuIDS source distribution of the installed version.

All built inputs and scan outputs go under one work directory, which the
configurations refer to as `${PYNU_WORK}`:

```
export PYNU_WORK=/path/to/work
W=$PYNU_WORK; mkdir -p $W/sk $W/ic $W/orca $W/fluxq
```

## External data

| file | where | checksum (md5) |
|---|---|---|
| `SuperK_fullMC_7_2_x100.h5`: Super-Kamiokande MC, 100 × the data exposure | Harvard Dataverse, doi:10.7910/DVN/E1NVLA | `2eb9e6deb027509c7a753e8a8b512673` |
| `sk_atm_2023.tar.gz`: Super-Kamiokande I-V data release, **version 1** | Zenodo, doi:10.5281/zenodo.8401262 (record 8401262) | `58f26b39ba0bb36cc3570b7723298e2b` |

The SK kernel of record is pinned to the version-1 release tables. Zenodo's "latest"
link now resolves to version 2 (2026-11-03), whose MC tables differ by up to 1 % per
bin. To reproduce the published numbers, download record 8401262 explicitly.

The IceCube-DeepCore and KM3NeT/ORCA6 files in `data/` are the parts of those
collaborations' public releases that the analysis reads. `data/SuperK/sk_2023_data_bins.csv`
is the observed SK count per analysis bin of the same release, in the format the SK
loader reads.

## Pipeline

**1. Response matrices and per-experiment inputs** (minutes each):

```
# Super-Kamiokande: response from the MC, the oscillation anchor, the kernel "Ninv"
python scripts/build.py sk-response --sk-mc SuperK_fullMC_7_2_x100.h5 --out $W/sk/sk_response.npz
python scripts/build.py tensors --experiment sk --single --out $W/sk/osc_tensor_pe_nominal.npz
python scripts/build.py sk-pin --response $W/sk/sk_response.npz --anchor $W/sk/osc_tensor_pe_nominal.npz \
       --release-dir sk_atm_2023/bins/normal --out $W/sk/sk_response_Ninv.npz
python scripts/build.py fluxq --experiment sk --response $W/sk/sk_response.npz --out $W/fluxq/fluxq_sk.npz

# IceCube-DeepCore and KM3NeT/ORCA6
python scripts/build.py ic-response   --out $W/ic/ic_response_modeaxis_L3.npz
python scripts/build.py orca-response --out $W/orca/orca_response_flat900.npz
python scripts/build.py fluxq --experiment ic   --response $W/ic/ic_response_modeaxis_L3.npz --out $W/fluxq/fluxq_ic.npz
python scripts/build.py fluxq --experiment orca --response $W/orca/orca_response_flat900.npz --out $W/fluxq/fluxq_orca.npz
python scripts/build.py aux --experiment ic   --response $W/ic/ic_response_modeaxis_L3.npz --out $W/ic/ic_aux.npz
python scripts/build.py aux --experiment orca --response $W/orca/orca_response_flat900.npz --out $W/orca/orca_aux.npz
```

**2. Oscillation tensors**: one file per (Δm²₃₁, sin²θ₂₃) lattice cell, holding all 20 δCP
nodes; task t is cell (t // 41, t % 41). A configuration lists the tasks its region needs:

```
for t in $(python -m pynu.binned.scan legs --config configs/Combined.xml --ordering NO --tensor-tasks); do
  python scripts/build.py tensors --experiment sk --ordering NO --task $t --out-dir $W/tensors/sk/NO
  python scripts/build.py tensors --experiment ic --ordering NO --task $t \
         --response $W/ic/ic_response_modeaxis_L3.npz --out-dir $W/tensors/ic/NO
  python scripts/build.py tensors --experiment orca --ordering NO --task $t \
         --response $W/orca/orca_response_flat900.npz --out-dir $W/tensors/orca/NO
done
```

On one machine a rebuilt tensor repeats itself bit for bit, across runs and thread
counts. Against the published files, SK tensors differ only by round-off (below 10⁻⁷).
IC and ORCA tensors are a different realization: with averaging off, nuSQuIDS draws the
neutrino production height at random on every evaluation, as in the published scans, and
its generator restarts with each process. This moves χ² by about 0.015 per δCP node
(largest seen 0.04), with no overall bias.

**3. Scans**: one task per (Δm²₃₁, sin²θ₂₃) cell of the region. Each δCP node is
written as one JSON.

```
python -m pynu.binned.scan legs     --config configs/Combined.xml --ordering NO        # number of tasks
python -m pynu.binned.scan run      --config configs/Combined.xml --ordering NO --task 189 --out $W/scans/combined/NO
python -m pynu.binned.scan assemble --config configs/Combined.xml --ordering NO \
       --cells $W/scans/combined/NO --out $W/scans/combined_NO.npz
```

`assemble` writes the 41 × 41 × 20 χ² cube, its profiles in Δm²₃₁, sin²θ₂₃, δCP and
(Δm²₃₁, sin²θ₂₃), and a summary with the best fit and the Δχ² = 1 intervals.

**Check against the published scans** (needs only the tensors of the reference cells):

```
python -m pynu.binned.scan check --config configs/Combined.xml
```

`check` refits each reference cell listed in the configuration, walking the δCP chain
up to it where the protocol carries the fit from node to node, and compares χ² with
the published value. `PASS` is within the minimizer-path tolerance `tol` of the
configuration. `PASS for rebuilt inputs` is within the larger `tol_rebuilt`, expected
when you built the inputs yourself. A kernel you built never has the published md5 (it
records its build time), so the scan prints a warning and continues; `tests/test_build.py`
compares it with the published file array by array.

### Verification of this release

Refits from a fresh install of `requirements.txt` on a workstation (Threadripper 7970X,
Linux, Python 3.12), compared with the published χ²:

| scan | reference cell | published χ² | published inputs | inputs rebuilt here |
|---|---|---|---|---|
| SK | NO (24, 22, 13) | 991.340733 | −3 × 10⁻⁷ | −4.0 × 10⁻⁴ |
| SK | IO (19, 22, 11) | 997.688399 | +2.6 × 10⁻⁵ | +4.9 × 10⁻⁵ |
| IC | NO (25, 21, 17) | 208.087946 | +4.7 × 10⁻⁵ | +2.8 × 10⁻² |
| IC | IO (15, 20, 6) | 209.787578 | +1.7 × 10⁻⁵ | +8.8 × 10⁻³ |
| ORCA | NO (15, 22, 17) | 507.848236 | +9 × 10⁻¹⁰ | +2.1 × 10⁻² |
| ORCA | IO (26, 22, 7) | 508.335112 | −1 × 10⁻⁹ | +3.0 × 10⁻² |
| combined | NO (24, 22, 13) | 1736.753190 | −5.8 × 10⁻⁴ | −3.3 × 10⁻³ |
| combined | IO (17, 22, 12) | 1747.783368 | −2.3 × 10⁻⁴ | −3.0 × 10⁻² |

Evaluated at the published fitted parameters on the published inputs (230 δCP nodes), χ²
agrees to better than 10⁻¹². The differences in the "published inputs" column come only
from the minimizer path (processor, BLAS, thread count); on the rebuilt inputs Δχ²(IO − NO)
is 11.004, against 11.030 published.

### Compute

| step | per ordering | measured on |
|---|---|---|
| SK tensors, 20 × 20 region | ≈ 400 files, ≈ 5.5 GB, ≈ 7 CPU-h (about 1 CPU-min per file) | workstation |
| IC tensors, full lattice | 1,681 files, 41 GB | computing cluster |
| ORCA avg5 tensors, full lattice | 1,681 files, 2.6 GB, ≈ 50 CPU-h | workstation |
| SK scan | 400 tasks | not yet measured |
| IC scan | 1,681 tasks, ≈ 23 CPU-h | computing cluster |
| ORCA scan | 1,681 tasks, ≈ 25 CPU-h | workstation |
| combined scan | 400 tasks, ≈ 1,100 CPU-h | computing cluster |

Each task is independent; run them as a scheduler array or with any local process pool.
Set `OMP_NUM_THREADS` (and `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`) to the cores per task.

## Tests

The test suite checks this code against the published scans at three levels:

| level | needs | what it checks | time |
|---|---|---|---|
| data-free | nothing | the four configurations; the paper's numbers from the published χ² cubes (in `tests/reference/`) | seconds |
| published inputs | the test inputs below | χ² of every part, and its gradient, at the published fitted parameters (230 δCP nodes); refits of the reference cells | ~15 min |
| your inputs | the test inputs and your `${PYNU_WORK}` | the inputs you built against the published ones, array by array; χ² on your inputs | ~1 min |

**Test inputs.** The published inputs of the cells the tests use (responses, the SK kernel,
auxiliary arrays, flux-ratio tables, and the oscillation tensors of 11 lattice cells) are
one archive on Harvard Dataverse:

| file | where | checksum (md5) |
|---|---|---|
| `eno_test_inputs.tar` (380 MB) | Harvard Dataverse, [doi:10.7910/DVN/YNWB3Y](https://doi.org/10.7910/DVN/YNWB3Y) | `9196dff9bfb40d5ef8cb8ea381662898` |

```
curl -L -o eno_test_inputs.tar https://dataverse.harvard.edu/api/access/datafile/14412137
tar xf eno_test_inputs.tar                       # unpacks to eno_test_inputs/
export PYNU_REFERENCE=$PWD/eno_test_inputs
```

The tests check each unpacked file against `tests/reference/inputs.md5` before using it.

**Running them:**

```
pip install pytest
pytest tests                          # data-free checks; the rest is skipped
pytest tests -m "not slow"            # with PYNU_REFERENCE set: also chi^2 at the published parameters
pytest tests                          # with PYNU_REFERENCE set: also the refits
PYNU_WORK=$W pytest tests             # also compares the inputs you built (steps 1-2 above)
```

Tests whose inputs are missing are skipped, so every command above runs. To compare your own
inputs, build at least the per-experiment inputs of step 1 and the tensors of the test cells:

```
for t in 1006; do python scripts/build.py tensors --experiment sk --ordering NO --task $t --out-dir $W/tensors/sk/NO; done
for t in 801 719; do python scripts/build.py tensors --experiment sk --ordering IO --task $t --out-dir $W/tensors/sk/IO; done
for t in 1046 1006; do python scripts/build.py tensors --experiment ic --ordering NO --task $t \
       --response $W/ic/ic_response_modeaxis_L3.npz --out-dir $W/tensors/ic/NO; done
for t in 635 719; do python scripts/build.py tensors --experiment ic --ordering IO --task $t \
       --response $W/ic/ic_response_modeaxis_L3.npz --out-dir $W/tensors/ic/IO; done
for t in 637 1006; do python scripts/build.py tensors --experiment orca --ordering NO --task $t \
       --response $W/orca/orca_response_flat900.npz --out-dir $W/tensors/orca/NO; done
for t in 1088 719; do python scripts/build.py tensors --experiment orca --ordering IO --task $t \
       --response $W/orca/orca_response_flat900.npz --out-dir $W/tensors/orca/IO; done
```

Your IC and ORCA tensors are a different random draw from the published ones (see step 2),
so χ² on your inputs is compared within that draw noise; everything else must agree to
round-off.

## Conventions

- Δχ² values are relative to each ordering's own minimum unless stated otherwise; the
  ordering test statistic is χ²(IO) − χ²(NO) between the two floors.
- Δm²₃₁ is the 3–1 splitting; values quoted elsewhere as Δm²₃₂ differ by Δm²₂₁.
- Results agree with the published cubes to within the minimizer tolerance: on the
  published inputs, χ² values can differ by up to 10⁻³ between processor families and
  thread counts. Rebuilt IC and ORCA tensors add the draw noise described under Pipeline.

## Event-by-event fits

This branch fits with the binned engines only; the PyNu event framework it carries is
used to build their inputs. The event-by-event engine, which reweights every MC
event at every fit step, is on the `datafit-SK-mcmc` branch of
https://github.com/pabloferm/Pynu. It includes event-level versions of the SK
systematics, checked against the binned engine at MC scale. Three parameters of the
final SK configuration have no event-level version there:
`xsec_ccqe_shape_multigev`, `elike_ringcount_lowp_sk13`, and `upmu_norm`, which is
pinned in the published fits.

For Super-Kamiokande the two engines were compared on the published configuration
at the combined and the SK-alone best fits:

- total predicted rate within 1 × 10⁻³;
- no analysis bin differs by more than 0.8 %, and 95 % of bins by less than 0.3 %;
- χ² differs by +0.16 at the combined best fit and −0.09 at the SK-alone best fit.

The binned engine evaluates the oscillation probability and the energy-dependent
systematics at the centres of its true-energy and zenith cells instead of per event;
that approximation accounts for 3 × 10⁻⁴ of the rate difference. No event-by-event
version of the IceCube-DeepCore, ORCA or combined fits was run for the paper.

## Citation

If you use this code, cite the paper above. The Super-Kamiokande MC set is
P. Fernández-Menéndez, C. A. Argüelles, M. Jin, I. Martínez-Soler,
*Super-Kamiokande Atmospheric Neutrino Simulation*, Harvard Dataverse (2026),
doi:10.7910/DVN/E1NVLA.
The test inputs are P. Fernández-Menéndez, M. Jin, *Evidence for Normal Neutrino Mass
Ordering from Atmospheric Combined Fit*, Harvard Dataverse (2026), doi:10.7910/DVN/YNWB3Y.

## License

This code is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version. See
`COPYING.LESSER`, which supplements the GNU General Public License in `COPYING`.

The files under `data/` are copied from the IceCube, KM3NeT/ORCA and Super-Kamiokande
public data releases and from the published FLUKA flux table, and remain under their
original terms.
