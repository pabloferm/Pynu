#!/bin/bash
#SBATCH -J icup_sens
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit/Pynu/logs/icup_sens_%a_%j.out
#SBATCH -e /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit/Pynu/logs/icup_sens_%a_%j.err
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-01:00
#SBATCH -p arguelles_delgado
#SBATCH --array=0-40

echo "======================================================================"
echo "ICECUBE UPGRADE SENSITIVITY ROW WORKER - Array Task ${SLURM_ARRAY_TASK_ID}"
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
mkdir -p ${PROJECT_DIR}/claude/ICUp-implementation/results/icup_sensitivity_41x41/rows

# Run this row
python -u ${PROJECT_DIR}/claude/ICUp-implementation/scripts/run_icup_sensitivity_row_worker.py \
    --row-idx ${SLURM_ARRAY_TASK_ID} \
    --n-dm 41 --n-s23 41 \
    --config ${PYNU_DIR}/examples/AnalysisFiles/ICUp_Atm.xml \
    --output-dir ${PROJECT_DIR}/claude/ICUp-implementation/results/icup_sensitivity_41x41/rows

echo ""
echo "======================================================================"
echo "Row ${SLURM_ARRAY_TASK_ID} complete!"
echo "Date: $(date)"
echo "======================================================================"
