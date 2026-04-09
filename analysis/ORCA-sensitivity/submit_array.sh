#!/bin/bash
#SBATCH -J pynu_grid           # Job name
#SBATCH -o logs/pynu_grid_%A_%a.out
#SBATCH -e logs/pynu_grid_%A_%a.err
#SBATCH -p arguelles_delgado
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-00:30             # 30 min per row (plenty)
#SBATCH --array=0-19           # 20 rows (one per theta value)

# =============================================================================
# Pynu Grid Array Job — Muon-Corrected NTOA Comparison
# Each array task handles one row (one theta, all dm values)
# =============================================================================

source ~/setup_rocky.sh
export NUSQUIDS_DATA_PATH=/n/holylfs05/LABS/arguelles_delgado_lab/Users/bskrzypek/software/GOLEMSOURCE/local/share/nuSQuIDS/

PROJECT_DIR=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit
SCRIPT_DIR=$PROJECT_DIR/claude/minimization-investigation/scripts
RESULTS_DIR=$PROJECT_DIR/claude/minimization-investigation/results/pynu_grid_v2

export PYTHONPATH=$PROJECT_DIR/Pynu:$PYTHONPATH
mkdir -p $PROJECT_DIR/logs $RESULTS_DIR

echo "=== Array task $SLURM_ARRAY_TASK_ID / Job $SLURM_ARRAY_JOB_ID ==="
echo "Started: $(date)"

python $SCRIPT_DIR/run_pynu_grid_worker.py \
    --row-idx $SLURM_ARRAY_TASK_ID \
    --n-theta 20 \
    --n-dm 20 \
    --output-dir $RESULTS_DIR

echo "Finished: $(date)"
