"""Refit the published reference cells from the configured start (minutes; ``-m slow``).

``scan check`` refits each reference cell of a configuration, walking the delta_CP
chain up to it where the fit carries its parameters from node to node, and every
node of the chain is compared with the published value. Unlike the fixed-parameter
tests, a refit follows the minimizer's path, which depends on the processor, BLAS
library and thread count; the tolerances below cover that (measured on a laptop and
a workstation against the cluster runs, largest at the best fits: SK 3e-5, IC 5e-5,
ORCA 2e-9, combined 6e-4; <= 2.5e-3 along the combined chains).
"""
import os

import pytest

from conftest import SCANS, inputs_present, load_config

TOL_REFERENCE = {"SK": 1e-3, "IC": 5e-4, "ORCA": 1e-4, "Combined": 2e-3}
TOL_CHAIN = 5e-3


@pytest.mark.slow
@pytest.mark.parametrize("scan", SCANS)
def test_refit_reference_cells(scan, reference_dir, reference_cells, tmp_path):
    from pynu.binned.scan import Fitter, run_leg
    published = {(c["ordering"], *c["cell"]): c["chi2"] for c in reference_cells[scan]}
    cfg = load_config(scan, reference_dir)
    for ref in cfg.references:
        i, j, d = ref.cell
        if not inputs_present(cfg, ref.ordering, [(i, j)]):
            pytest.skip(f"published inputs for {scan} {ref.ordering} not in PYNU_REFERENCE")
        F = Fitter(cfg, ref.ordering)
        order = cfg.dcp_order(ref.ordering)
        nodes = order[:order.index(d) + 1] if F.warm else [d]
        recs = run_leg(F, i, j, nodes, os.path.join(tmp_path, f"{ref.ordering}_{i}_{j}"),
                       log=lambda *a: None)
        got = recs[-1]["chi2"]
        print(f"{scan} {ref.ordering} {ref.cell}: chi2 {got:.6f} published {ref.chi2:.6f} "
              f"diff {got - ref.chi2:+.2e}")
        assert got == pytest.approx(ref.chi2, abs=TOL_REFERENCE[scan])
        for r in recs:
            key = (ref.ordering, *r["cell"])
            if key in published:
                assert r["chi2"] == pytest.approx(published[key], abs=TOL_CHAIN), key
