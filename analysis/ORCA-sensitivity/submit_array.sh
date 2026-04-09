#!/bin/bash
#SBATCH -J pynu_grid
#SBATCH -o logs/pynu_grid_%a_%j.out
#SBATCH -e logs/pynu_grid_%a_%j.err
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-02:00
#SBATCH -p arguelles_delgado
#SBATCH --array=0-19

source ~/setup_rocky.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYNU_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH=${PYNU_DIR}:${PYTHONPATH}
export NUSQUIDS_DATA_PATH=/n/holylfs05/LABS/arguelles_delgado_lab/Users/bskrzypek/software/GOLEMSOURCE/local/share/nuSQuIDS/

OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/results/pynu_grid/rows}
mkdir -p ${OUTPUT_DIR} logs

python -u ${SCRIPT_DIR}/run_pynu_grid_worker.py \
    --point-idx ${SLURM_ARRAY_TASK_ID} \
    --config ${SCRIPT_DIR}/ORCA_Atm_NTOA_grid.xml \
    --output-dir ${OUTPUT_DIR}

echo "Task ${SLURM_ARRAY_TASK_ID} complete! Date: $(date)"
