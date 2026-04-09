#!/bin/bash
#SBATCH -J sk_datafit
#SBATCH -o logs/sk_datafit_%a_%j.out
#SBATCH -e logs/sk_datafit_%a_%j.err
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-04:00
#SBATCH -p arguelles_delgado
#SBATCH --array=0-40

# SK 2023 Data Fit: 41x41 grid (Dm231 x Sin2Theta23)
# Post NC-weight fix. 32 nuisance params, dCP profiled over 13 values.
# Each row takes ~3.5 hrs (41 s23 points × 13 dCP × L-BFGS-B, warm start only).

echo "======================================================================"
echo "SK 2023 DATA FIT (post NC-fix) - Row ${SLURM_ARRAY_TASK_ID}"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID | Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "======================================================================"

source ~/setup_rocky.sh

PROJECT_DIR="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit"
PYNU_DIR="${PROJECT_DIR}/Pynu"
export NUSQUIDS_DATA_PATH=/n/holylfs05/LABS/arguelles_delgado_lab/Users/bskrzypek/software/GOLEMSOURCE/local/share/nuSQuIDS/
export PYTHONPATH="${PYNU_DIR}:${PYTHONPATH}"
export PYNU="${PYNU_DIR}/pynu"

cd "${PYNU_DIR}"
mkdir -p logs

RESULTS_DIR="${PROJECT_DIR}/claude/2-atmospheric-oscillation/SuperK/results/sk_datafit_ncfix/rows"
mkdir -p "${RESULTS_DIR}"

python -u "${PROJECT_DIR}/claude/2-atmospheric-oscillation/SuperK/scripts/run_sk_datafit_row_worker.py" \
    --row-idx ${SLURM_ARRAY_TASK_ID} \
    --n-dm 41 --n-s23 41 \
    --n-dcp 13 \
    --config "${PYNU_DIR}/examples/AnalysisFiles/SK2023_Atm_datafit.xml" \
    --output-dir "${RESULTS_DIR}"

echo ""
echo "======================================================================"
echo "Row ${SLURM_ARRAY_TASK_ID} complete: $(date)"
echo "======================================================================"
