#!/bin/bash
#SBATCH -J orca_binned_scan
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-04:00
#SBATCH -p shared,sapphire,arguelles_delgado
#SBATCH --array=0-19

# ============================================================================
# Single-experiment ORCA binned (Dm231 x Sin2Theta23) scan.
#
# One array task = one Dm231 row (NDM rows total); each task fits NS23 cells
# through ORCABinnedEngine's analytic 30-dial gradient. The Gaussian prior lives
# in the worker, not the engine (the engine is stat-only by contract).
#
# Log files go to logs/ relative to the submission directory — create it before
# the first sbatch. See PIPELINE_README.md for the full recipe.
# ============================================================================

# ---- grid (parameterize here; must match the tensor build) ------------------
NDM=20
NS23=20
DM_MIN=2.3e-3
DM_MAX=2.7e-3
S23_MIN=0.40
S23_MAX=0.65

# ---- roots (override any via --export) --------------------------------------
# PYNU_TREE = the Pynu checkout that carries this analysis dir + pynu/Experiments.
PYNU_TREE="${PYNU_TREE:?set PYNU_TREE to the Pynu checkout root}"
ANALYSIS_DIR="${PYNU_TREE}/analysis/ORCA-binned-datafit"

# ---- environment: source your cluster env first, then PIN PYTHONPATH to the
# Pynu tree EXPLICITLY (the pin is the authority on which tree runs). ---------
if [[ -n "${ENV_SETUP:-}" ]]; then source "${ENV_SETUP}"; fi
export PYTHONPATH="${PYNU_TREE}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# ---- inputs (override any via --export) -------------------------------------
CONFIG="${CONFIG:?set CONFIG to the ORCA 30-dial analysis XML}"
RESPONSE="${RESPONSE:?set RESPONSE to the flat900 response npz}"
PHI_DIR="${PHI_DIR:?set PHI_DIR to the per-point osc-tensor directory}"
MC="${MC:-${PYNU_TREE}/data/ORCA/ORCA_MC_dataverse_with_muons.parquet}"
DATA="${DATA:-${PYNU_TREE}/data/ORCA/ORCA_data_dataverse.parquet}"
TAG="${TAG:-orca_binned}"
OUT="${OUT:-${ANALYSIS_DIR}/results/${TAG}}"

# ---- preflight (fail loudly before burning queue time) ----------------------
for f in "${CONFIG}" "${RESPONSE}" "${MC}" "${DATA}" \
         "${ANALYSIS_DIR}/run_orca_binned_fit_worker.py"; do
    [[ -f "$f" ]] || { echo "PREFLIGHT FAIL: missing file $f" >&2; exit 64; }
done
[[ -d "${PHI_DIR}" ]] || { echo "PREFLIGHT FAIL: missing phi dir ${PHI_DIR}" >&2; exit 64; }
if (( SLURM_ARRAY_TASK_ID >= NDM )); then
    echo "PREFLIGHT FAIL: task ${SLURM_ARRAY_TASK_ID} >= NDM=${NDM}" >&2; exit 64
fi
# every tensor this row will read must already exist
for ((j=0; j<NS23; j++)); do
    p=$(printf "%s/orca_phi_%03d_%03d.npz" "${PHI_DIR}" "${SLURM_ARRAY_TASK_ID}" "${j}")
    [[ -f "$p" ]] || { echo "PREFLIGHT FAIL: missing phi tensor $p" >&2; exit 64; }
done

mkdir -p "${OUT}"

echo "=== orca_binned_scan task ${SLURM_ARRAY_TASK_ID} job ${SLURM_JOB_ID} $(date) ==="
echo "pynu tree : ${PYNU_TREE}"
echo "config    : ${CONFIG}"
echo "response  : ${RESPONSE}"
echo "phi dir   : ${PHI_DIR}"
echo "out       : ${OUT}"

python3 -u "${ANALYSIS_DIR}/run_orca_binned_fit_worker.py" \
    --row "${SLURM_ARRAY_TASK_ID}" \
    --config   "${CONFIG}" \
    --response "${RESPONSE}" \
    --mc       "${MC}" \
    --data     "${DATA}" \
    --phi-source tensors --phi-dir "${PHI_DIR}" \
    --ndm ${NDM}   --dm-min ${DM_MIN}   --dm-max ${DM_MAX} \
    --ns23 ${NS23} --s23-min ${S23_MIN} --s23-max ${S23_MAX} \
    --tag "${TAG}" --outdir "${OUT}"

echo "=== task ${SLURM_ARRAY_TASK_ID} done $(date) ==="
