"""IC sparse populated-cell structure — the response side of `ICBinnedEngine`.

Loads `ic_response_modeaxis_L*.npz` into the SPARSE representation design §3.3
prescribes, and builds the per-cell geometry the dial fields evaluate on. Reads
the npz only: no pynu import, no live experiment, so it runs and gates locally.

WHY SPARSE, NOT DENSE. The dense class-cell grid at L3 is 47 x 160 x 320 =
2,406,400 entries, of which 127,757 are populated — 5.3% occupancy. Dense arrays
would spend 95% of every dial evaluation multiplying zeros, and the whole point of
the port is that fit-time cost tracks POPULATED cells (design §3.3, and its
occupancy row corrected per ADDENDUM item 8: the old text bounded this by nnz,
which is the wrong quantity and itself grows with the class axis).

THE TWO AXES ARE DIFFERENT LENGTHS, and conflating them is the easy mistake:

    CELL axis   (n_cell = 127,757 at L3)  distinct (class, ie_true, iz_true).
                Dials live here — every dial factor is a function of
                (E_cell, cz_cell, pdg, cc, |Mode|), constant across the events in
                a cell. phi also lives here, indexed [ntype, flavor, ie, iz].
    ENTRY axis  (nnz = 330,097 at L3)     distinct (class, ie, iz, reco_bin).
                The response matrix lives here. One cell fans out to several reco
                bins; `cell_of_entry` maps entry -> cell, and the contraction is
                a bincount over reco bin weighted by (entry weight x cell factor).

So a dial factor is computed ONCE per cell and gathered onto entries — that
gather is exactly the response-matrix contraction, and it is where the speedup
comes from relative to the 396,843-event path.

PHI CONVENTION: oscillated FLUX (flux x P), NOT bare probability, and NO SK NC
override (SPEC_trackR_ic_binned_matrices.md:240-241,247-252; design §6). The NC
classes carry the same flux treatment as CC. Gate G-IC-3 is the detector for a
violation, because it shows up as an O(1) discrepancy at nominal dials.

    python ic_cells.py --selftest --response <ic_response_modeaxis_L3.npz>
"""
import argparse
import json
import sys

import numpy as np

from .ic_dial_fields import ICCellGeom

# Measured floors, SCOPE §1.3 "mode axis, no snap" column.
CELL_FLOORS = {"L1": {"n_cell": 26040, "nnz": 192826, "nE": 40},
               "L3": {"n_cell": 127757, "nnz": 330097, "nE": 160}}
SUMW_BASELINE = 64066.0106465270
N_BINS = 200


class ICCells:
    """Sparse populated-cell view of an IC binned response.

    Attributes
      n_cell, n_entry           axis lengths (see the module docstring)
      geom                      ICCellGeom over the CELL axis
      cell_k/cell_ie/cell_iz    per-cell response coordinates
      ntype, flavor             per-cell phi indices (0 nu / 1 nubar; 0 e/1 mu/2 tau)
      entry_cell                (n_entry,) entry -> cell index
      entry_bin, entry_w        (n_entry,) reco bin and summed raw weight
      mu_bkg                    (200,) static muon histogram, ZERO gradient
    """

    def __init__(self, path):
        self.path = path
        r = np.load(path, allow_pickle=True)
        axis = [str(a) for a in np.asarray(r["class_axis"]).tolist()]
        if axis != ["pdg", "current", "absmode"]:
            raise SystemExit(
                f"response class axis is {axis}; the engine needs the |Mode| axis "
                "(build with --mode-axis). The 11 Mode-keyed dials are "
                "unrepresentable without it.")
        self.grid_label = str(r["grid_label"])
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
        self.geom = ICCellGeom(e_c[self.cell_ie], cz_c[self.cell_iz],
                               pdg, cc, absmode)

        # phi[ntype, flavor, ie, iz] — same convention as the divergence scan
        self.ntype = (pdg < 0).astype(np.int64)
        self.flavor = (np.abs(pdg) // 2 - 6).astype(np.int64)

        self.mu_bkg = np.asarray(r["mu_bkg"], float)
        self.meta = json.loads(r["meta"].item() if hasattr(r["meta"], "item")
                               else str(r["meta"]))
        self.classes = classes

    # -- the contraction -----------------------------------------------------
    def expectation(self, cell_factor, phi_cell, norm=1.0):
        """Per-reco-bin expectation from a per-CELL dial factor and per-CELL phi.

        n_bin = SUM_entries  entry_w * norm * phi[cell] * factor[cell]

        The gather `[...][entry_cell]` IS the response-matrix contraction: the
        dial is evaluated n_cell times (127,757) rather than n_event times
        (396,843), and every event in a cell shares the value exactly.
        """
        per_cell = np.asarray(phi_cell, float) * np.asarray(cell_factor, float)
        return np.bincount(self.entry_bin,
                           weights=self.entry_w * norm * per_cell[self.entry_cell],
                           minlength=N_BINS)

    def phi_cells(self, phi_point):
        """Gather phi[ntype, flavor, ie, iz] onto the cell axis.

        INTEGER indexing only — never float-edge matching. The response and the
        phi tensor are built on separate runs, and a cluster rebuild reproduced
        e_true_edges/centers to allclose (max diff 9.1e-13) but NOT bitwise, from
        summation order in geomspace. Integer cell indices are immune to that;
        float-edge matching would not be.
        """
        p = np.asarray(phi_point)
        if p.shape[-2:] != (self.nE, self.nZ):
            raise SystemExit(
                f"phi grid {p.shape[-2:]} != response grid ({self.nE}, {self.nZ}). "
                "These must be the SAME ladder — a snapped response (nE 162) "
                "cannot index a tensor built on the unsnapped grid (nE 160).")
        return np.asarray(
            p[self.ntype, self.flavor, self.cell_ie, self.cell_iz], float)

    def assert_phi_grid(self, phi_npz, rtol=1e-9):
        """Optional stronger check when the phi npz carries its own centres.

        Compares with ALLCLOSE, not array_equal: identical geomspace inputs can
        differ in the last ulp across BLAS/numpy builds (measured 9.1e-13 between
        the local and cluster rebuilds of the same response), which is physically
        irrelevant but would fail a bitwise test. A genuine ladder mismatch is
        orders of magnitude larger and still caught.
        """
        r = np.load(phi_npz, allow_pickle=True) if isinstance(phi_npz, str) else phi_npz
        if "e_true_centers" not in r:
            return None
        resp_e = np.load(self.path, allow_pickle=True)["e_true_centers"]
        d = float(np.max(np.abs(np.asarray(r["e_true_centers"], float)
                                - np.asarray(resp_e, float))))
        ok = np.allclose(r["e_true_centers"], resp_e, rtol=rtol, atol=0.0)
        if not ok:
            raise SystemExit(f"phi centres differ from the response by {d:.3e} "
                             f"(rtol {rtol:g}) — different ladders, not roundoff")
        return d


def selftest(path, label=None):
    print("=== IC sparse cells — construction gates ===")
    c = ICCells(path)
    label = label or c.grid_label
    f = CELL_FLOORS.get(label)
    ok = True

    print(f"  grid {c.grid_label}  nE={c.nE} nZ={c.nZ}  "
          f"classes={c.classes.shape}  cells={c.n_cell}  entries={c.n_entry}")
    if f:
        good = (c.n_cell == f["n_cell"] and c.n_entry == f["nnz"] and c.nE == f["nE"])
        print(f"GATE cell-floors {label}: {'PASS' if good else 'FAIL'} "
              f"n_cell {c.n_cell}/{f['n_cell']}  nnz {c.n_entry}/{f['nnz']}  "
              f"nE {c.nE}/{f['nE']}  (SCOPE §1.3 mode-axis-no-snap)")
        ok &= good

    # weight conservation through the sparse structure
    sw = float(c.entry_w.sum())
    d = abs(sw - SUMW_BASELINE) / SUMW_BASELINE
    print(f"GATE conservation: {'PASS' if d <= 1e-9 else 'FAIL'} "
          f"Sw={sw!r} vs baseline {SUMW_BASELINE!r} (|d|/S={d:.3e})")
    ok &= d <= 1e-9

    # the contraction must move ALL the weight into the 200 reco bins
    unit = np.ones(c.n_cell)
    n = c.expectation(unit, unit)
    d2 = abs(float(n.sum()) - sw) / sw
    print(f"GATE contraction-closure: {'PASS' if d2 <= 1e-12 else 'FAIL'} "
          f"sum(n_bin)={float(n.sum())!r} vs Sw (|d|/S={d2:.3e}); "
          f"{int((n > 0).sum())}/200 bins populated")
    ok &= d2 <= 1e-12

    # entry -> cell map is a partition: every cell reached, no entry orphaned
    reached = np.unique(c.entry_cell).size
    part_ok = (reached == c.n_cell and c.entry_cell.min() >= 0
               and c.entry_cell.max() == c.n_cell - 1)
    print(f"GATE cell-partition: {'PASS' if part_ok else 'FAIL'} "
          f"{reached}/{c.n_cell} cells reached by entries; "
          f"mean fan-out {c.n_entry / c.n_cell:.3f} reco bins/cell")
    ok &= part_ok

    # per-cell class columns must agree with the response's own class table
    cls_ok = (np.array_equal(c.geom.pdg, c.classes[c.cell_k, 0])
              and np.array_equal(c.geom.cc, c.classes[c.cell_k, 1])
              and np.array_equal(c.geom.absmode, c.classes[c.cell_k, 2]))
    modes = sorted(set(int(v) for v in np.unique(c.geom.absmode)))
    print(f"GATE class-columns: {'PASS' if cls_ok else 'FAIL'}  "
          f"|Mode| realized {modes} (expect [0, 1, 11, 26, 31])")
    ok &= cls_ok and modes == [0, 1, 11, 26, 31]

    # phi index ranges
    ph_ok = (c.ntype.min() >= 0 and c.ntype.max() <= 1
             and c.flavor.min() == 0 and c.flavor.max() == 2)
    print(f"GATE phi-index: {'PASS' if ph_ok else 'FAIL'} "
          f"ntype[{c.ntype.min()},{c.ntype.max()}] flavor[{c.flavor.min()},"
          f"{c.flavor.max()}]")
    ok &= ph_ok

    # muon block: constant, and it must carry ZERO gradient (design §6, G-G2)
    mu_ok = abs(c.mu_bkg.sum() - 512.166) < 1e-2
    print(f"GATE muon-block: {'PASS' if mu_ok else 'FAIL'} "
          f"Sw={c.mu_bkg.sum():.4f} (expect 512.166), 200 bins, zero gradient")
    ok &= mu_ok

    # a class-N dial contracted on cells must equal the same dial applied on
    # entries — the property that makes the engine a rewrite, not an approximation
    try:
        from ic_dial_fields import factor as ic_factor
    except ImportError:
        from .ic_dial_fields import factor as ic_factor
    fac = np.asarray(ic_factor("CCQE", c.geom, 1.37), float)
    lhs = c.expectation(fac, np.ones(c.n_cell))
    rhs = np.bincount(c.entry_bin,
                      weights=c.entry_w * fac[c.entry_cell], minlength=N_BINS)
    eq = np.array_equal(lhs, rhs)
    print(f"GATE dial-contraction (CCQE @1.37): {'PASS' if eq else 'FAIL'} "
          f"cell-gathered == entry-applied, bitwise {eq}")
    ok &= eq

    print(f"IC CELLS: {'ALL PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--response", required=True)
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.response, a.label) else 1)
    c = ICCells(a.response)
    print(f"{c.grid_label}: {c.n_cell} cells, {c.n_entry} entries")
