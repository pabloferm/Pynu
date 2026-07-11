#!/bin/bash
#SBATCH -J skb_modular_20x20
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0-06:00
#SBATCH -p shared,sapphire,arguelles_delgado
#SBATCH --array=0-39

# ============================================================================
# SK binned 20x20 (Dm231 x Sin2Theta23) scan through the integrated
# (PyNuFit-modular) engine.
#
# 2 arms x 20 dm rows = 40 tasks; task = arm_idx * 20 + dm_row (combined-array
# convention shared with the row worker).
#
# Resources: the 131/133-dial rows fit a 4 CPU / 16G / 8h envelope standalone;
# the modular worker additionally holds the full PyNuFit event-engine MC in
# memory, so this uses 8 CPU / 32G / 6h.
#
# Log files go to logs/ relative to the submission directory — create it
# before the first sbatch. See PIPELINE_README.md for the full how-to.
# ============================================================================

# ---- arms (parameterize here) ----------------------------------------------
# ARM_SPECS = engine NAMED specs (resolve_nuisance_spec names — NOT file paths;
# the worker rejects path-like values on --arm-specs). Order defines arm_idx.
ARM_SPECS=(r2_fude_ccqe r2_fude_ccqe_nmig)
NARMS=${#ARM_SPECS[@]}
NDM=20

# ---- roots (override any via --export) --------------------------------------
PROJECT_DIR="${PROJECT_DIR:-/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/AtmNuDataFit}"
# PYNU_TREE = the Pynu checkout that carries this analysis dir + pynu/binned.
PYNU_TREE="${PYNU_TREE:-${PROJECT_DIR}/backup_pynu/Pynu}"

# ---- environment: cluster python env, then PIN PYTHONPATH to the Pynu tree
# EXPLICITLY (the pin is the authority on which tree runs). ------------------
source "${PROJECT_DIR}/env_fasrc_r2.sh"
export PYTHONPATH="${PYNU_TREE}:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# ---- inputs (defaults; override any via --export) ---------------------------
# ANALYSIS_XML: MUST be a full PyNuFit config whose every enabled nuisance has
# an owning tune method (a bare enabled dial with no method crashes PyNuFit at
# init) — see PIPELINE_README.md for the SK-specific config note.
ANALYSIS_XML="${ANALYSIS_XML:-${PROJECT_DIR}/Pynu/examples/AnalysisFiles/SK2023_Atm_datafit_phased_gateb.xml}"
RESPONSE="${RESPONSE:-${PROJECT_DIR}/claude/2-atmospheric-oscillation/SuperK/results/sk_binned_phased/sk_response.npz}"
TENSORS="${TENSORS:-${PROJECT_DIR}/claude/2-atmospheric-oscillation/SuperK/results/sk_binned_avg4pi_20x20x20/osc_tensors}"
SEEDS_DIR="${SEEDS_DIR:-${PROJECT_DIR}/claude-tmp/dm2/seeds_r2_fude_ccqe}"
TAG="${TAG:-modular_replay_20x20}"
OUT="${OUT:-${PROJECT_DIR}/claude/2-atmospheric-oscillation/SuperK/results/sk_binned_phased/joint_min/${TAG}}"

# gbest seeds: one converged seed json per arm, in ARM_SPECS order
GBEST_LIST=()
for s in "${ARM_SPECS[@]}"; do GBEST_LIST+=("${SEEDS_DIR}/${s}.json"); done

# ---- preflight (fail loudly before burning queue time) ----------------------
for f in "${ANALYSIS_XML}" "${RESPONSE}" "${GBEST_LIST[@]}"; do
    [[ -f "$f" ]] || { echo "PREFLIGHT FAIL: missing $f" >&2; exit 64; }
done
[[ -d "${TENSORS}" ]] || { echo "PREFLIGHT FAIL: missing tensor dir ${TENSORS}" >&2; exit 64; }
if (( SLURM_ARRAY_TASK_ID >= NARMS * NDM )); then
    echo "PREFLIGHT FAIL: task ${SLURM_ARRAY_TASK_ID} >= ${NARMS}x${NDM}" >&2; exit 64
fi

mkdir -p "${OUT}"

echo "=== skb_modular_20x20 task ${SLURM_ARRAY_TASK_ID} job ${SLURM_JOB_ID} $(date) ==="
echo "pynu tree : ${PYNU_TREE}"
echo "xml       : ${ANALYSIS_XML}"
echo "response  : ${RESPONSE}"
echo "tensors   : ${TENSORS}"
echo "arms      : ${ARM_SPECS[*]}"
echo "seeds     : ${GBEST_LIST[*]}"

python3 -u "${PYNU_TREE}/analysis/SK-binned-datafit/run_sk_binned_scan_row_worker.py" \
    --task "${SLURM_ARRAY_TASK_ID}" \
    --analysis-xml "${ANALYSIS_XML}" \
    --arm-specs "${ARM_SPECS[@]}" \
    --gbest-list "${GBEST_LIST[@]}" \
    --response "${RESPONSE}" \
    --tensors  "${TENSORS}" \
    --outdir   "${OUT}" \
    --ndm ${NDM} --dm-min 2.0e-3 --dm-max 3.2e-3 \
    --ns23 20 --s23-min 0.30 --s23-max 0.70 \
    --osc-averaging 4pi

echo "=== task ${SLURM_ARRAY_TASK_ID} done $(date) ==="
