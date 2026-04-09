#!/bin/bash
#SBATCH -J of2d_mat
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit/Pynu/logs/of2d_mat_%A_%a.out
#SBATCH -e /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit/Pynu/logs/of2d_mat_%A_%a.err
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 0-1:00
#SBATCH -p arguelles_delgado_gpu_a100,arguelles_delgado_gpu_mixed
#SBATCH --array=0-40

source ~/setup_rocky.sh
export NUSQUIDS_DATA_PATH=/n/holylfs05/LABS/arguelles_delgado_lab/Users/bskrzypek/software/GOLEMSOURCE/local/share/nuSQuIDS/

PROJECT_DIR=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit
cd ${PROJECT_DIR}/Pynu

python ${PROJECT_DIR}/claude/2-atmospheric-oscillation/ORCA-full/scripts/run_orcafull_2d_row_worker.py \
    --config ${PROJECT_DIR}/Pynu/examples/AnalysisFiles/ORCAFull_Atm.xml \
    --output-dir ${PROJECT_DIR}/claude/2-atmospheric-oscillation/ORCA-full/results/orcafull_2d_matrix_rows/ \
    --row-idx ${SLURM_ARRAY_TASK_ID} \
    --exposure 5.0
