#!/bin/bash
#SBATCH -J ic_binned_scan
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 0-06:00
#SBATCH -p shared,sapphire,arguelles_delgado
#SBATCH --array=0-19

# ============================================================================
# Single-experiment IC DeepCore binned (Dm231 x Sin2Theta23) scan.
#
# One array task = one Dm231 row (NDM rows total); each task fits NS23 cells
# through ICBinnedEngine's analytic 39-dial gradient. The Gaussian prior lives
# in the worker, not the engine (the engine is stat-only by contract).
#
# ★ The worker refreshes the Dm2-dependent hypersurface slopes PER GRID CELL and
#   asserts the refresh — see PIPELINE_README.md.
#
# Log files go to logs/ relative to the submission directory — create it before
# the first sbatch.
# ============================================================================

# ---- grid (parameterize here; MUST match the tensor build's --grid) ---------
NDM=20
NS23=20
DM_MIN=2.3e-3
DM_MAX=2.7e-3
S23_MIN=0.40
S23_MAX=0.65

# ---- roots (override any via --export) --------------------------------------
# PYNU_TREE = the Pynu checkout that carries this analysis dir + pynu/Experiments.
PYNU_TREE="${PYNU_TREE:?set PYNU_TREE to the Pynu checkout root}"
ANALYSIS_DIR="${PYNU_TREE}/analysis/IC-binned-datafit"

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
CONFIG="${CONFIG:?set CONFIG to the IC 39-dial analysis XML}"
RESPONSE="${RESPONSE:?set RESPONSE to the mode-axis, unsnapped ic_response npz}"
PHI="${PHI:?set PHI to the scan-grid osc-tensor npz}"
HS_DIR="${HS_DIR:-${PYNU_TREE}/data/IceCube}"
TAG="${TAG:-ic_binned}"
OUT="${OUT:-${ANALYSIS_DIR}/results/${TAG}}"

# ---- preflight (fail loudly before burning queue time) ----------------------
for f in "${CONFIG}" "${RESPONSE}" "${PHI}" \
         "${ANALYSIS_DIR}/run_ic_binned_fit_worker.py"; do
    [[ -f "$f" ]] || { echo "PREFLIGHT FAIL: missing file $f" >&2; exit 64; }
done
[[ -d "${HS_DIR}" ]] || { echo "PREFLIGHT FAIL: missing hs dir ${HS_DIR}" >&2; exit 64; }
for h in hs_nu_nc_nue_cc.csv hs_numu_cc.csv hs_nutau_cc.csv; do
    [[ -f "${HS_DIR}/${h}" ]] || {
        echo "PREFLIGHT FAIL: missing hypersurface ${HS_DIR}/${h}" >&2; exit 64; }
done
case "$(basename "${RESPONSE}")" in
    *modeaxis*) ;;
    *) echo "PREFLIGHT FAIL: ${RESPONSE} is not a *modeaxis* build; the engine "\
            "refuses a 12-class response" >&2; exit 64;;
esac
if (( SLURM_ARRAY_TASK_ID >= NDM )); then
    echo "PREFLIGHT FAIL: task ${SLURM_ARRAY_TASK_ID} >= NDM=${NDM}" >&2; exit 64
fi

mkdir -p "${OUT}"

echo "=== ic_binned_scan task ${SLURM_ARRAY_TASK_ID} job ${SLURM_JOB_ID} $(date) ==="
echo "pynu tree : ${PYNU_TREE}"
echo "config    : ${CONFIG}"
echo "response  : ${RESPONSE}"
echo "phi       : ${PHI}"
echo "hs dir    : ${HS_DIR}"
echo "out       : ${OUT}"

# OBSERVED_NPZ (optional): explicit 200-bin observation for XMLs whose
# <DataFiles> block is disabled — the worker refuses to self-fit Asimov MC
# without it (load_observation guard). Build it from the real data with
# pynu.Experiments.ic_binned_support.observed_200.
OBS_ARGS=()
if [[ -n "${OBSERVED_NPZ:-}" ]]; then
    [[ -f "${OBSERVED_NPZ}" ]] || { echo "PREFLIGHT FAIL: missing OBSERVED_NPZ ${OBSERVED_NPZ}" >&2; exit 64; }
    OBS_ARGS=(--observed-npz "${OBSERVED_NPZ}")
fi

python3 -u "${ANALYSIS_DIR}/run_ic_binned_fit_worker.py" \
    --row "${SLURM_ARRAY_TASK_ID}" \
    --config   "${CONFIG}" \
    --response "${RESPONSE}" \
    --phi      "${PHI}" \
    --hs-dir   "${HS_DIR}" \
    ${OBS_ARGS[@]+"${OBS_ARGS[@]}"} \
    --ndm ${NDM}   --dm-min ${DM_MIN}   --dm-max ${DM_MAX} \
    --ns23 ${NS23} --s23-min ${S23_MIN} --s23-max ${S23_MAX} \
    --tag "${TAG}" --outdir "${OUT}"
rc=$?
# propagate the worker's exit code — a guard abort must FAIL the task, not
# masquerade as COMPLETED (job 39856508 lesson: 20/20 tasks exited 0 with
# zero output because this echo was the last command).
if (( rc != 0 )); then
    echo "=== task ${SLURM_ARRAY_TASK_ID} FAILED rc=${rc} $(date) ===" >&2
    exit "${rc}"
fi

echo "=== task ${SLURM_ARRAY_TASK_ID} done $(date) ==="
