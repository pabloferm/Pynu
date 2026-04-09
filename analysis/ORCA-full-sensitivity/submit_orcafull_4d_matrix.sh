#!/bin/bash
#SBATCH -J of4d_mat
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit/Pynu/logs/of4d_mat_%A_%a.out
#SBATCH -e /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit/Pynu/logs/of4d_mat_%A_%a.err
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-6:00
#SBATCH -p arguelles_delgado_gpu_a100,arguelles_delgado_gpu_mixed
#SBATCH --array=0-40

# Environment setup
source ~/setup_rocky.sh
export NUSQUIDS_DATA_PATH=/n/holylfs05/LABS/arguelles_delgado_lab/Users/bskrzypek/software/GOLEMSOURCE/local/share/nuSQuIDS/

PROJECT_DIR=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit
cd ${PROJECT_DIR}/Pynu

echo "=== ORCAFull 4D Matrix-based: row ${SLURM_ARRAY_TASK_ID} ==="
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}, Node: $(hostname)"
echo "Started: $(date)"

python ${PROJECT_DIR}/claude/2-atmospheric-oscillation/ORCA-full/scripts/run_orcafull_4d_row_worker.py \
    --config ${PROJECT_DIR}/Pynu/examples/AnalysisFiles/ORCAFull_Atm.xml \
    --output-dir ${PROJECT_DIR}/claude/2-atmospheric-oscillation/ORCA-full/results/orcafull_4d_matrix_rows/ \
    --row-idx ${SLURM_ARRAY_TASK_ID} \
    --exposure 5.0 \
    --n-theta13 11 --theta13-min 0.018 --theta13-max 0.026 \
    --n-dcp 12 --dcp-min 0.0 --dcp-max 5.8853

echo "Finished: $(date)"
