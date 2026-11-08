"""Read the oscillation tensors of one lattice cell.

Every experiment stores one file per (Delta m^2_31, sin^2 theta_23) lattice cell,
``<dir>/osc_tensor_{i_dm31:03d}_{i_s23:03d}.npz``, holding

    phi    float32 (n_dcp, 2, 3, nE, nZ)   oscillation tensor at every delta_CP node
    dm231  the cell's signed Delta m^2_31
    s23    the cell's sin^2 theta_23
    dcp    the delta_CP nodes in radians

``scripts/build.py tensors`` writes them. The reader checks that the labels in the
file are the lattice values the configuration asks for, so a tensor built on a
different grid or ordering cannot be used by mistake.
"""
import os

import numpy as np


class TensorDir:
    def __init__(self, directory, lattice, ordering):
        self.dir = directory
        self.ordering = ordering
        self.dm31 = lattice.dm31_axis(ordering)
        self.s23 = lattice.s23_axis()
        self.dcp = lattice.dcp_nodes()

    def path(self, i_dm31, i_s23):
        return os.path.join(self.dir, f"osc_tensor_{i_dm31:03d}_{i_s23:03d}.npz")

    def load(self, i_dm31, i_s23):
        """Return phi (n_dcp, 2, 3, nE, nZ) float32 after checking the labels."""
        p = self.path(i_dm31, i_s23)
        if not os.path.exists(p):
            raise FileNotFoundError(f"oscillation tensor missing: {p}")
        z = np.load(p)
        dm = float(np.ravel(z["dm231"])[0])
        s23 = float(np.ravel(z["s23"])[0])
        want_dm, want_s23 = float(self.dm31[i_dm31]), float(self.s23[i_s23])
        if abs(dm - want_dm) > 1e-4 * abs(want_dm) or abs(s23 - want_s23) > 1e-5:
            raise ValueError(f"{p}: labels (dm231={dm}, s23={s23}) are not lattice cell "
                             f"({want_dm}, {want_s23}) of ordering {self.ordering}")
        dcp = np.ravel(np.asarray(z["dcp"], float))
        if dcp.size != self.dcp.size or not np.allclose(dcp, self.dcp, rtol=1e-9,
                                                        atol=1e-12):
            raise ValueError(f"{p}: delta_CP nodes {dcp} are not the lattice nodes")
        return z["phi"]
