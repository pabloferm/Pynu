"""COO response contraction kernels — TRANSCRIBED, not imported.

PROVENANCE. These bodies are a transcription of the two contraction kernels that
already exist in the SK code, which are functionally identical to each other:

  * `external/Pynu-upstream/pynu/Experiments/sk_binned_engine_core.py:46-56`
    (`contract` / `contract_var`, the eng-first kernel module), and
  * `claude/2-atmospheric-oscillation/SuperK/binned-octant/src/sk_binned/
     sk_binned_engine.py:1713-1722` (`SKBinnedEngine.contract` /
     `.contract_var`, the in-class copy the combined worker actually loads).

WHY TRANSCRIBED RATHER THAN LIFTED (design §3.1 is corrected by its own 2026-08-17
ADDENDUM item 4; scope §3.1):

  1. `sk_binned_engine_core.py:33-42` imports ERA_TAGS, MASK_TUNES,
     XSEC_VECTOR_NAMES, SUBGEV_NUE_NORM, MULTIGEV_CCQE_NORM, DIR_SMEAR_NAME,
     SOLAR_AMP and SOLAR_SCALE from `.sk_binned_engine`. Importing the "neutral"
     kernel module therefore drags the entire SK engine in with it.
  2. The SK engine the combined fit actually runs is not that module at all —
     `combined_3exp_fit_worker.py:134-137` points SKB at the 2,915-line
     binned-octant standalone, whose contract is an in-class method and which
     does not use sk_binned_engine_core.

Transcribing keeps the SK engine — the project's most-audited artifact — literally
untouched, which is the stated architecture rule.

ONE DELIBERATE DEVIATION from the SK bodies. SK contracts a DENSE cell array and
gathers it per nonzero with `W.ravel()[eng.R_widx]`. Following design §3.3, the
IC/ORCA engines evaluate dial fields on the POPULATED cell list only, so these
kernels take `Wflat` already gathered to the nonzero axis (length nnz). The
arithmetic is otherwise identical: `bincount(Rb, weights=Rv * Wflat)`.
"""
import numpy as np


def contract(Rb, Rv, Wflat, n_bins):
    """n_pre[b] = sum over nonzeros in bin b of R_v * W.

    Args:
      Rb:     (nnz,) int, reco-bin index of each nonzero.
      Rv:     (nnz,) float, the nonzero's response weight.
      Wflat:  (nnz,) float, the cell weight already gathered to the nonzero axis.
      n_bins: int, length of the output.
    """
    return np.bincount(Rb, weights=Rv * Wflat, minlength=n_bins)


def contract_class(Rk, Rb, Rv, Wflat, n_cls, n_bins):
    """n_pre[k, b], the class-marginal contraction -> (n_cls, n_bins).

    Sums over k to `contract` exactly (the class axis is a disjoint partition),
    which is the property the class-N gradient shortcut rests on. Same idea as
    the SK per-era marginal `contract_era` (core:58-64 / engine:1724-1729), with
    the class axis in place of the era axis.
    """
    return np.bincount(Rk * n_bins + Rb, weights=Rv * Wflat,
                       minlength=n_cls * n_bins).reshape(n_cls, n_bins)


def contract_var(S2b, S2v, Wsqflat, n_bins):
    """Sum of BaseWeight^2 * W^2 per bin (pre-detector) — the MC-variance twin.

    Present for schema completeness. UNUSED by the ORCA arm, whose likelihood is
    pure Poisson, not Barlow-Beeston (campaign convention since 2026-06-25).
    """
    return np.bincount(S2b, weights=S2v * Wsqflat, minlength=n_bins)


def adjoint_cells(cell_of_nnz, Rb, Rvn, rD, n_cell):
    """u_c = sum over the cell's nonzeros of R_vn * rD[bin] — ONE bincount, O(nnz).

    The adjoint (reverse-mode) accumulator. It is the whole performance story of
    the gradient: u_c is built once per evaluation, after which each cell-axis
    dial's derivative is a single length-n_cell dot with u_c * W_c, instead of
    its own O(nnz) contraction.

    Args:
      cell_of_nnz: (nnz,) int, populated-cell index of each nonzero.
      Rb:          (nnz,) int, reco-bin index of each nonzero.
      Rvn:         (nnz,) float, NORM-scaled response weight.
      rD:          (n_bins,) float, resid_b * D_b (zero off the `few` mask).
      n_cell:      int, number of populated cells.
    """
    return np.bincount(cell_of_nnz, weights=Rvn * rD[Rb], minlength=n_cell)
