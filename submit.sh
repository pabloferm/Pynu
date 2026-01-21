#!/bin/bash
#SBATCH -c 2
#SBATCH -p arguelles_delgado
#SBATCH --mem 4096
#SBATCH -t 0-2:00
#SBATCH --output=logs/pynu_job_%A_%a.out
#SBATCH --error=logs/pynu_job_%A_%a.err
#SBATCH --array=0-440

# ============================================================================
# ORCA Atmospheric Neutrino Analysis - Pynu Framework
# SLURM Job Array Script
#
# Grid: 21 x 21 = 441 points
#   - Sin2Theta23: 0.4 to 0.6 (21 points)
#   - Dm231: 2.3e-3 to 2.7e-3 (21 points)
#
# Usage:
#   sbatch submit.sh
#
# Or to run a subset:
#   sbatch --array=0-99 submit.sh
# ============================================================================

# Setup environment
source ~/setup_rocky.sh

# Set nuSQuIDS data path
export NUSQUIDS_DATA_PATH=/n/holylfs05/LABS/arguelles_delgado_lab/Users/bskrzypek/software/GOLEMSOURCE/local/share/nuSQuIDS/

# Navigate to Pynu directory
PYNU_DIR=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit/Pynu
cd ${PYNU_DIR}

# Add pynu subdirectory to PYTHONPATH (needed for internal imports)
export PYTHONPATH=${PYNU_DIR}/pynu:${PYTHONPATH}
export PYNU=${PYNU_DIR}/pynu

# Create logs and results directories if they don't exist
mkdir -p logs
mkdir -p results

# Define output file (shared HDF5 file with file locking)
OUTFILE="results/ORCA_sensitivity_$(date +%m%d).hdf5"

# Run single point
echo "=============================================="
echo "Running Pynu analysis point ${SLURM_ARRAY_TASK_ID}"
echo "Output file: ${OUTFILE}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "=============================================="

python analysis_main.py \
    examples/AnalysisFiles/ORCA_Atm.xml \
    -p ${SLURM_ARRAY_TASK_ID} \
    -o ${OUTFILE}

echo "Point ${SLURM_ARRAY_TASK_ID} complete."
