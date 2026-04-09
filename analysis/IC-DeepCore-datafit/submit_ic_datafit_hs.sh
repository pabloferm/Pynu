#!/bin/bash
#SBATCH -J ic_hs_data
#SBATCH -o logs/ic_hs_data_%a_%j.out
#SBATCH -e logs/ic_hs_data_%a_%j.err
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-01:00
#SBATCH -p arguelles_delgado
#SBATCH --array=0-40

echo "======================================================================"
echo "IC DEEPCORE DATA FIT (HS) ROW WORKER - Array Task ${SLURM_ARRAY_TASK_ID}"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID | Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname) | Date: $(date)"
echo "======================================================================"

# Setup environment
source ~/setup_rocky.sh

# Paths — edit these for your setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYNU_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH=${PYNU_DIR}:${PYTHONPATH}
export NUSQUIDS_DATA_PATH=/n/holylfs05/LABS/arguelles_delgado_lab/Users/bskrzypek/software/GOLEMSOURCE/local/share/nuSQuIDS/

# Output — set this to your desired results location
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/results/ic_datafit_hs_41x41/rows}
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

python -u ${SCRIPT_DIR}/run_ic_datafit_row_worker.py \
    --row-idx ${SLURM_ARRAY_TASK_ID} \
    --n-dm 41 --n-s23 41 \
    --config ${SCRIPT_DIR}/IC_Atm_datafit.xml \
    --hs-dir ${PYNU_DIR}/data/IceCube \
    --output-dir ${OUTPUT_DIR}

echo "Row ${SLURM_ARRAY_TASK_ID} complete! Date: $(date)"
