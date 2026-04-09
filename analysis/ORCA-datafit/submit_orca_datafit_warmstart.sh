#!/bin/bash
#SBATCH -J orca_df
#SBATCH -o logs/orca_df_%a_%j.out
#SBATCH -e logs/orca_df_%a_%j.err
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-02:00
#SBATCH -p arguelles_delgado
#SBATCH --array=0-40

echo "ORCA DATA FIT (warm start) - Array Task ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"

source ~/setup_rocky.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYNU_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH=${PYNU_DIR}:${PYTHONPATH}
export NUSQUIDS_DATA_PATH=/n/holylfs05/LABS/arguelles_delgado_lab/Users/bskrzypek/software/GOLEMSOURCE/local/share/nuSQuIDS/

OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/results/orca_datafit_warmstart_41x41/rows}
mkdir -p ${OUTPUT_DIR} logs

python -u ${SCRIPT_DIR}/run_orca_datafit_row_worker.py \
    --row-idx ${SLURM_ARRAY_TASK_ID} \
    --n-dm 41 --n-s23 41 \
    --config ${SCRIPT_DIR}/ORCA_Atm_datafit.xml \
    --output-dir ${OUTPUT_DIR}

echo "Row ${SLURM_ARRAY_TASK_ID} complete! Date: $(date)"
