"""IC sparse populated-cell structure: the response side of `ICBinnedEngine`.

Loads `ic_response_modeaxis_*.npz` and builds the per-cell geometry the dial fields
evaluate on. There are two axes of different lengths:

    CELL axis   distinct (class, ie_true, iz_true). Dial factors are functions of
                (E_cell, cz_cell, pdg, cc, |Mode|). phi also lives here, indexed
                [ntype, flavor, ie, iz].
    ENTRY axis  distinct (class, ie, iz, reco_bin). The response matrix lives here. One
                cell fans out to several reco bins; `entry_cell` maps entry -> cell.

A dial factor is computed once per cell and gathered onto entries.

phi is the oscillated flux (flux x P), so NC classes carry the same flux treatment as CC.
"""
import numpy as np

from .ic_dials import ICCellGeom


class ICCells:
    """Sparse populated-cell view of an IC binned response.

    Attributes
      n_cell, n_entry           axis lengths (see the module docstring)
      geom                      ICCellGeom over the CELL axis
      cell_k/cell_ie/cell_iz    per-cell response coordinates
      ntype, flavor             per-cell phi indices (0 nu / 1 nubar; 0 e/1 mu/2 tau)
      entry_cell                (n_entry,) entry -> cell index
      entry_bin, entry_w        (n_entry,) reco bin and summed raw weight
    """

    def __init__(self, path):
        r = np.load(path, allow_pickle=True)
        axis = [str(a) for a in np.asarray(r["class_axis"]).tolist()]
        if axis != ["pdg", "current", "absmode"]:
            raise SystemExit(
                f"response class axis is {axis}; the engine needs the |Mode| axis. "
                "The 11 Mode-keyed dials are unrepresentable without it.")
        self.nE, self.nZ = int(r["n_etrue"]), int(r["n_cztrue"])
        classes = np.asarray(r["classes"], np.int64)
        e_c = np.asarray(r["e_true_centers"], float)
        cz_c = np.asarray(r["cz_true_centers"], float)

        R_k = np.asarray(r["R_k"], np.int64)
        R_e = np.asarray(r["R_e"], np.int64)
        R_z = np.asarray(r["R_z"], np.int64)
        self.entry_bin = np.asarray(r["R_b"], np.int64)
        self.entry_w = np.asarray(r["R_v"], float)
        self.n_entry = self.entry_w.size

        # cell axis = distinct (class, ie, iz); `inverse` is entry -> cell
        key = (R_k * self.nE + R_e) * self.nZ + R_z
        uniq, inverse = np.unique(key, return_inverse=True)
        self.entry_cell = np.asarray(inverse, np.int64).ravel()
        self.n_cell = uniq.size

        rem, self.cell_iz = np.divmod(uniq, self.nZ)
        self.cell_k, self.cell_ie = np.divmod(rem, self.nE)

        pdg = classes[self.cell_k, 0]
        cc = classes[self.cell_k, 1]
        absmode = classes[self.cell_k, 2]
        # `e_grid`: the response's true-energy grid, over which the multi-GeV CCQE shape
        # is centred (the cell list repeats each energy once per populated class)
        self.geom = ICCellGeom(e_c[self.cell_ie], cz_c[self.cell_iz],
                               pdg, cc, absmode, e_grid=e_c)

        # phi[ntype, flavor, ie, iz]
        self.ntype = (pdg < 0).astype(np.int64)
        self.flavor = (np.abs(pdg) // 2 - 6).astype(np.int64)

    def phi_cells(self, phi_point):
        """Gather phi[ntype, flavor, ie, iz] onto the cell axis, by integer index."""
        p = np.asarray(phi_point)
        if p.shape[-2:] != (self.nE, self.nZ):
            raise SystemExit(
                f"phi grid {p.shape[-2:]} != response grid ({self.nE}, {self.nZ}). "
                "The tensor and the response must be on the same grid.")
        return np.asarray(
            p[self.ntype, self.flavor, self.cell_ie, self.cell_iz], float)
