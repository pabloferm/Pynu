# Analysis Pipeline Scripts

Each subdirectory contains a self-contained analysis: row worker, SLURM submit script, assembler, plotter, and a symlink to the XML configuration.

## Environment Setup

Before running any analysis, set the following:

### Required

1. **Software environment** -- load Python 3.11+ with nuSQuIDS, nuflux, numpy, scipy, h5py, boost-histogram:
   ```bash
   source ~/setup_rocky.sh   # or your equivalent
   ```

2. **nuSQuIDS data path** -- points to the nuSQuIDS cross-section and flux tables:
   ```bash
   export NUSQUIDS_DATA_PATH=/path/to/nuSQuIDS/share/nuSQuIDS/
   ```

3. **PYNU** (Super-Kamiokande only) -- needed for FLUKA flux file initialization:
   ```bash
   export PYNU=/path/to/Pynu/pynu
   ```

### Automatically Derived (no action needed)

- `PYNU_DIR` -- resolved from the script location (`SCRIPT_DIR/../..`)
- `PYTHONPATH` -- set by the submit script to include `PYNU_DIR`
- `OUTPUT_DIR` -- defaults to `results/` under the analysis directory; override with:
  ```bash
  export OUTPUT_DIR=/your/custom/output/path
  ```

## Running an Analysis

```bash
cd analysis/<name>/
sbatch submit_*.sh
```

After the array job completes, assemble and plot:
```bash
python assemble_*.py --input-dir results/<name>/rows --output-dir results/<name>/
python plot_*.py --input-dir results/<name>/
```

## Directory Layout

```
analysis/
├── README.md                    # This file
├── AnalysisFiles/               # XML configurations
│   ├── *.xml                    # Production configs
│   └── legacy-xml/              # Historical/template XMLs
│
├── ORCA-sensitivity/            # ORCA Asimov sensitivity (NTOA comparison)
├── ORCA-datafit/                # ORCA real-data fit (warm start)
├── IC-DeepCore-sensitivity/     # IC DeepCore Asimov (hypersurface systematics)
├── IC-DeepCore-datafit/         # IC DeepCore real-data fit (HS + muon_norm)
├── ICUpgrade-sensitivity/       # IceCube Upgrade Asimov projection
├── SuperK-sensitivity/          # Super-Kamiokande 2023 Asimov
├── SuperK-datafit/              # Super-Kamiokande 2023 real-data fit
└── ORCA-full-sensitivity/       # ORCA 115-string Asimov (2D + 4D scans)
```
