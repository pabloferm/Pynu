#!/bin/bash
#SBATCH --job-name=orca_phi_build
#SBATCH --partition=arguelles_delgado,shared,sapphire,serial_requeue
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-19
#SBATCH --output=orca_phi_build_%A_%a.out
# ============================================================================
# Per-point ORCA osc-tensor build for the binned scan (stage 2 of the pipeline:
# response -> TENSORS -> scan). One array task = one Dm231 row: builds the
# NS23 files `orca_phi_{row:03d}_{j:03d}.npz` that submit_orca_binned_scan.sbatch
# preflights, by calling build_orca_osc_tensors.py once per (dm231, s23) point.
#
# The (dm, s23) values come from the SAME np.linspace arithmetic the scan
# worker uses (run_orca_binned_fit_worker.py:382-383) — computed here in one
# python call and consumed verbatim, so a grid mismatch is impossible by
# construction. Requires nuSQuIDS (cluster env via ENV_SETUP).
# ============================================================================
set -u

# ---- grid (must match submit_orca_binned_scan.sbatch) -----------------------
NDM=${NDM:-20}
NS23=${NS23:-20}
DM_MIN=${DM_MIN:-2.3e-3}
DM_MAX=${DM_MAX:-2.7e-3}
S23_MIN=${S23_MIN:-0.40}
S23_MAX=${S23_MAX:-0.65}

# ---- oscillation setting (defaults = the fixed-dCP contour convention) ------
DCP=${DCP:-1.36}
S13=${S13:-0.022}

# ---- roots / environment (same pattern as the scan sbatch) ------------------
PYNU_TREE="${PYNU_TREE:?set PYNU_TREE to the Pynu checkout root}"
ANALYSIS_DIR="${PYNU_TREE}/analysis/ORCA-binned-datafit"
if [[ -n "${ENV_SETUP:-}" ]]; then source "${ENV_SETUP}"; fi
export PYTHONPATH="${PYNU_TREE}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# ---- inputs ------------------------------------------------------------------
CONFIG="${CONFIG:?set CONFIG to the ORCA 30-dial analysis XML}"
RESPONSE="${RESPONSE:?set RESPONSE to the flat900 response npz}"
PHI_DIR="${PHI_DIR:?set PHI_DIR to the output tensor directory}"

# ---- preflight (fail loudly before burning queue time) ----------------------
for f in "${CONFIG}" "${RESPONSE}" "${ANALYSIS_DIR}/build_orca_osc_tensors.py"; do
    [[ -f "$f" ]] || { echo "PREFLIGHT FAIL: missing file $f" >&2; exit 64; }
done
if (( SLURM_ARRAY_TASK_ID >= NDM )); then
    echo "PREFLIGHT FAIL: task ${SLURM_ARRAY_TASK_ID} >= NDM=${NDM}" >&2; exit 64
fi
mkdir -p "${PHI_DIR}"

I=${SLURM_ARRAY_TASK_ID}
# the scan worker's exact grid arithmetic (run_orca_binned_fit_worker.py:382-383)
read -r DM <<< "$(python3 -c "import numpy as np; print(repr(float(np.linspace(${DM_MIN},${DM_MAX},${NDM})[${I}])))")"
S23_LIST=$(python3 -c "import numpy as np; print(' '.join(repr(float(v)) for v in np.linspace(${S23_MIN},${S23_MAX},${NS23})))")

echo "=== orca_phi_build row ${I} job ${SLURM_JOB_ID} $(date) ==="
echo "pynu tree : ${PYNU_TREE}"
echo "dm231     : ${DM}   dcp ${DCP}   s13 ${S13}"
echo "phi dir   : ${PHI_DIR}"

J=0
for S23 in ${S23_LIST}; do
    OUTF=$(printf "%s/orca_phi_%03d_%03d.npz" "${PHI_DIR}" "${I}" "${J}")
    if [[ -f "${OUTF}" ]]; then
        echo "[${I},${J}] exists, skipping ${OUTF}"
    else
        python3 -u "${ANALYSIS_DIR}/build_orca_osc_tensors.py" \
            --config "${CONFIG}" --response "${RESPONSE}" \
            --dm231 "${DM}" --s23 "${S23}" --s13 "${S13}" \
            --dcp-nodes "${DCP}" \
            --out "${OUTF}" || { echo "BUILD FAIL at (${I},${J})" >&2; exit 1; }
    fi
    J=$((J+1))
done

echo "ROW_DONE ${I}: ${NS23} tensors in ${PHI_DIR}"
