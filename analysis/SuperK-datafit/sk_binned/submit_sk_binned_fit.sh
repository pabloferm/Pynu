#!/bin/bash
#SBATCH -J sk_binned_fit
#SBATCH -o logs/sk_binned_fit_%j.out
#SBATCH -e logs/sk_binned_fit_%j.err
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 0-06:00
#SBATCH -p shared,sapphire,arguelles_delgado

# SK binned forward-model GRID FIT (SK-official Eq. 10 likelihood).
#
# Runs fit_sk_binned.py --grid: the full 15x15 (Dm231 x sin2th23) scan in ONE
# process, profiling dCP over the 13 precomputed values and minimizing all XML
# nuisances, warm-starting the nuisance vector across the grid. ~1 hr for the
# 225 points (the numpy fit phase is memory-bandwidth-bound). This is a SINGLE
# (non-array) job by design -- the warm-start chaining is sequential, so do NOT
# fan it out per-point.
#
# PREREQUISITE: run submit_sk_binned_build.sh first to produce the build
# artifacts this consumes: ${OUT}/sk_response.npz + ${OUT}/osc_tensors/.
#
# Optional env knobs:
#   CONFIG       analysis XML (default: SK2023_Atm_datafit_xsec_barr_ntag.xml)
#   OUTPUT_DIR   build-artifact + output dir (default: <this dir>/results)
#   MIN_ENTRIES  strict obs>MIN_ENTRIES bin cut (default 5 = SK production;
#                -1 keeps all 930 bins)
#   ENERGY_SCALE 1 (default) keeps the energy_scale nuisance; 0 -> --no-energy-scale
#
# Local (non-SLURM) equivalent:
#   python fit_sk_binned.py --xml <CONFIG> --response <OUT>/sk_response.npz \
#       --tensors <OUT>/osc_tensors --min-entries 5 --grid --output <OUT>/binned_fit_grid.json
# A single point instead of the grid: replace --grid with e.g. --points 5_6 skbf

source ~/setup_rocky.sh
export NUSQUIDS_DATA_PATH=/n/holylfs05/LABS/arguelles_delgado_lab/Users/bskrzypek/software/GOLEMSOURCE/local/share/nuSQuIDS/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYNU_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"  # repo root (contains pynu/)
export PYTHONPATH="${PYNU_DIR}:${PYTHONPATH}"
export PYNU="${PYNU_DIR}/pynu"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

SKB="${SCRIPT_DIR}"
OUT="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"
CONFIG="${CONFIG:-${PYNU_DIR}/analysis/AnalysisFiles/SK2023_Atm_datafit_xsec_barr_ntag.xml}"
MIN_ENTRIES="${MIN_ENTRIES:-5}"
ESFLAG=""; [ "${ENERGY_SCALE:-1}" = "0" ] && ESFLAG="--no-energy-scale"

mkdir -p logs
cd "${PYNU_DIR}"

echo "=== sk_binned_fit job ${SLURM_JOB_ID:-local} $(date) | config=${CONFIG} min_entries=${MIN_ENTRIES} energy_scale=${ENERGY_SCALE:-1} ==="
python -u "${SKB}/fit_sk_binned.py" \
    --xml "${CONFIG}" \
    --response "${OUT}/sk_response.npz" \
    --tensors "${OUT}/osc_tensors" \
    --min-entries "${MIN_ENTRIES}" \
    --grid ${ESFLAG} \
    --output "${OUT}/binned_fit_grid.json"
echo "=== sk_binned_fit done $(date) ==="
