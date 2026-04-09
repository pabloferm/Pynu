#!/bin/bash
#SBATCH -J of_4d
#SBATCH -o logs/of_4d_%a_%j.out
#SBATCH -e logs/of_4d_%a_%j.err
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-04:00
#SBATCH -p arguelles_delgado
#SBATCH --array=0-40

echo "ORCA-FULL 4D SENSITIVITY - Array Task ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"

source ~/setup_rocky.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYNU_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH=${PYNU_DIR}:${PYTHONPATH}
export NUSQUIDS_DATA_PATH=/n/holylfs05/LABS/arguelles_delgado_lab/Users/bskrzypek/software/GOLEMSOURCE/local/share/nuSQuIDS/

OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/results/orcafull_4d_matrix/rows}
mkdir -p ${OUTPUT_DIR} logs

cd ${PYNU_DIR}
python -u ${SCRIPT_DIR}/run_orcafull_4d_row_worker.py \
    --config ${SCRIPT_DIR}/ORCAFull_Atm.xml \
    --output-dir ${OUTPUT_DIR} \
    --row-idx ${SLURM_ARRAY_TASK_ID}

echo "Row ${SLURM_ARRAY_TASK_ID} complete! Date: $(date)"
