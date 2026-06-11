#!/bin/bash
#SBATCH -J sk_binned_build
#SBATCH -o logs/sk_binned_build_%A_%a.out
#SBATCH -e logs/sk_binned_build_%A_%a.err
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-04:00
#SBATCH -p shared,sapphire,arguelles_delgado
#SBATCH --array=0-227

# P1 of the SK binned forward model (plan: response matrix + SK-official
# likelihood, 2026-06-11).
#
# Task 0:      build_sk_response.py  -> sk_response.npz (R, R+-2%, sumw2,
#              class table, true grid, data vector; self-check vs BinMC(ones))
# Tasks 1-226: build_osc_tensors.py --task (ID-1) -> per-grid-point Phi
#              tensors (1-225 -> grid 0-224) + specials (226 -> task 225:
#              no-osc / SK-bf / points A,B)
# Task 227:    eval_event_engine_vectors.py -> Gate-B reference (event-engine
#              expectations at 6 deterministic nuisance vectors at point A)

source ~/setup_rocky.sh
export NUSQUIDS_DATA_PATH=/n/holylfs05/LABS/arguelles_delgado_lab/Users/bskrzypek/software/GOLEMSOURCE/local/share/nuSQuIDS/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYNU_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"  # repo root (contains pynu/)
export PYTHONPATH="${PYNU_DIR}:${PYTHONPATH}"
export PYNU="${PYNU_DIR}/pynu"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

SKB="${SCRIPT_DIR}"
OUT="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"
CONFIG="${PYNU_DIR}/analysis/AnalysisFiles/SK2023_Atm_datafit_xsec_barr_ntag.xml"

mkdir -p "${OUT}/osc_tensors" logs
cd "${PYNU_DIR}"

echo "=== sk_binned_build task ${SLURM_ARRAY_TASK_ID} job ${SLURM_JOB_ID} $(date) ==="

if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
  python -u "${SKB}/build_sk_response.py" --config "${CONFIG}" \
      --n-etrue 400 --n-cztrue 80 \
      --output "${OUT}/sk_response.npz"
elif [ "${SLURM_ARRAY_TASK_ID}" -le 226 ]; then
  python -u "${SKB}/build_osc_tensors.py" --config "${CONFIG}" \
      --task $(( SLURM_ARRAY_TASK_ID - 1 )) \
      --n-etrue 400 --n-cztrue 80 \
      --output-dir "${OUT}/osc_tensors"
else
  python -u "${SKB}/eval_event_engine_vectors.py" --config "${CONFIG}" \
      --output "${OUT}/gateB_reference.npz"
fi

echo "=== task ${SLURM_ARRAY_TASK_ID} done $(date) ==="
