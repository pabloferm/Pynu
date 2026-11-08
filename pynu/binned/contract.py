"""COO response contraction kernels of the ORCA engine."""
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


def adjoint_cells(cell_of_nnz, Rb, Rvn, rD, n_cell):
    """u_c = sum over the cell's nonzeros of R_vn * rD[bin], one bincount, O(nnz).

    The adjoint accumulator of the gradient: u_c is built once per evaluation, after
    which each cell-axis dial's derivative is a single length-n_cell dot with u_c * W_c.

    Args:
      cell_of_nnz: (nnz,) int, populated-cell index of each nonzero.
      Rb:          (nnz,) int, reco-bin index of each nonzero.
      Rvn:         (nnz,) float, NORM-scaled response weight.
      rD:          (n_bins,) float, resid_b * D_b (zero off the `few` mask).
      n_cell:      int, number of populated cells.
    """
    return np.bincount(cell_of_nnz, weights=Rvn * rD[Rb], minlength=n_cell)
