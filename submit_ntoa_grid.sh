#!/bin/bash
#SBATCH -c 2
#SBATCH -p arguelles_delgado
#SBATCH --mem 4096
#SBATCH -t 0-2:00
#SBATCH --output=logs/pynu_ntoa_%A_%a.out
#SBATCH --error=logs/pynu_ntoa_%A_%a.err
#SBATCH --array=0-399

# ============================================================================
# Pynu ORCA Analysis - NTOA Grid Match with Barlow-Beeston Likelihood
# Grid: 20 x 20 = 400 points (matching NTOA framework)
#   - Sin2Theta23: 0.3 to 0.7 (20 points)
#   - Dm231: 1.5e-3 to 3.0e-3 (20 points)
# Uses same MC file as NTOA with Barlow-Beeston likelihood and muons.
# ============================================================================

source ~/setup_rocky.sh
export NUSQUIDS_DATA_PATH=/n/holylfs05/LABS/arguelles_delgado_lab/Users/bskrzypek/software/GOLEMSOURCE/local/share/nuSQuIDS/

PYNU_DIR=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit/Pynu
cd ${PYNU_DIR}

export PYTHONPATH=${PYNU_DIR}/pynu:${PYTHONPATH}
export PYNU=${PYNU_DIR}/pynu

mkdir -p logs
mkdir -p results/grid_points

# Output file per task to avoid file locking
OUTFILE="results/grid_points/point_${SLURM_ARRAY_TASK_ID}.hdf5"

echo "============================================="
echo "Pynu analysis point ${SLURM_ARRAY_TASK_ID} / 399"
echo "Config: ORCA_Atm_NTOA_grid.xml"
echo "Mode: BarlowBeestonLikelihood (with muons)"
echo "Output: ${OUTFILE}"
echo "============================================="

python analysis_main.py \
    examples/AnalysisFiles/ORCA_Atm_NTOA_grid.xml \
    -p ${SLURM_ARRAY_TASK_ID} \
    -o ${OUTFILE} \
    --mode BarlowBeestonLikelihood

echo "Point ${SLURM_ARRAY_TASK_ID} complete."
