#!/bin/bash
#SBATCH -J orca_ws
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit/Pynu/logs/orca_ws_%a_%j.out
#SBATCH -e /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit/Pynu/logs/orca_ws_%a_%j.err
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-00:30
#SBATCH -p arguelles_delgado
#SBATCH --array=0-40

echo "======================================================================"
echo "ORCA DATA FIT (no muon_norm, WARM START) - Array Task ${SLURM_ARRAY_TASK_ID}"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID | Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "======================================================================"

# Setup environment
source ~/setup_rocky.sh

# Set paths
export PROJECT_DIR=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit
export PYNU_DIR=${PROJECT_DIR}/Pynu
export NUSQUIDS_DATA_PATH=/n/holylfs05/LABS/arguelles_delgado_lab/Users/bskrzypek/software/GOLEMSOURCE/local/share/nuSQuIDS/
export PYTHONPATH=${PYNU_DIR}:${PYTHONPATH}

# Create directories
mkdir -p ${PYNU_DIR}/logs
mkdir -p ${PROJECT_DIR}/claude/muon-normalization-implementation/results/orca_datafit_warmstart_41x41/rows

# Run this row (warm-started minimization, no muon_norm)
python -u ${PROJECT_DIR}/claude/muon-normalization-implementation/scripts/run_orca_datafit_row_worker.py \
    --row-idx ${SLURM_ARRAY_TASK_ID} \
    --n-dm 41 --n-s23 41 \
    --config ${PYNU_DIR}/examples/AnalysisFiles/ORCA_Atm_datafit.xml \
    --output-dir ${PROJECT_DIR}/claude/muon-normalization-implementation/results/orca_datafit_warmstart_41x41/rows

echo ""
echo "======================================================================"
echo "Row ${SLURM_ARRAY_TASK_ID} complete!"
echo "Date: $(date)"
echo "======================================================================"
