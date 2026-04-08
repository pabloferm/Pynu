# Data preparation scripts

Scripts that convert experiment-specific raw MC into the Pynu parquet format.
Run from a Pynu-aware Python environment. Outputs are written to `Pynu/data/<experiment>/`.

## IceCube DeepCore (9-year data release)

`ic_dataverse_to_parquet.py`

Converts the public IceCube DeepCore CSV release (Phys. Rev. D 108, 012014, 2023)
into Pynu-compatible parquet files plus hypersurface CSVs.

Inputs:
- `IC_data_release/` directory of CSV files from the IceCube release
- pre-binned `mc_mu.csv` (muon background)

Outputs (in `data/IceCube/`):
- `IC_MC.parquet` (neutrino MC events + fake muon events at bin centers)
- `IC_data.parquet` (fake data events at bin centers, weight = count)
- `_E_reco_bins.npy`, `_cosT_reco_bins.npy`
- `hs_nu_nc_nue_cc.csv`, `hs_numu_cc.csv`, `hs_nutau_cc.csv` (hypersurface slopes)

## ORCA-Full (matrix-based, response-matrix MC)

Three-stage pipeline:

1. `orcafull_build_response_matrices.py` — builds 4 response components
   (energy migration, zenith migration, PID classification, effective area)
   from digitized ORCA performance plots in `AtmNuCombination/sources/ORCA/`.
2. `orcafull_compute_morphed_pid.py` — computes the morphed PID classification
   probabilities from the response matrices.
3. `orcafull_generate_meta_parquet.py` — generates `ORCA_full_MC.parquet` in
   the same column format as the ORCA-6 data release. Each row is one
   `(true_E, true_cz, channel) -> (reco_E, reco_cz, pid)` matrix cell with a
   combined weight.

Outputs (in `data/ORCAFull/`):
- `ORCA_full_MC.parquet`
- `_E_reco_bins.npy`, `_cosT_reco_bins.npy`

> **Note**: this pipeline depends on digitized inputs that live outside the Pynu
> tree (`AtmNuCombination/sources/ORCA/`). Update paths inside the scripts if
> the source data moves.

## ICUpgrade and SuperK

These experiments do **not** need a parquet conversion. Their Pynu experiment
classes (`pynu/Experiments/ICUpgrade_Atm.py` and `pynu/Experiments/SuperK_Atm_Pheno.py`)
read directly from the released CSV / HDF5 MC files.
