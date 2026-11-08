"""Inputs you built (PYNU_WORK) against the published ones (PYNU_REFERENCE).

Compared array by array, ignoring metadata (paths, file names, checksums). Weights
and auxiliary arrays must be identical. Tolerances apply only where a machine or
library version enters:

  grid coordinates    last bit (numpy versions round np.geomspace differently)
  flux-ratio tables   1e-13 relative (nuflux versions)
  SK tensors          1e-6 absolute. Against the published build, round-off moves about
                      0.3 % of these float32 elements (measured: 9e-8).
  Ninv kernel         1e-6 relative; inherits the above through the anchor tensor
                      (measured: 4e-8).

The IC and ORCA tensors are a different realization from the published ones. With
averaging off, nuSQuIDS draws the neutrino production height at random on every
evaluation, as in the published scans; its generator restarts with each process, so
rebuilding on one machine repeats your own draw bit for bit, but not the published
draw. They are checked through chi^2 at the published parameters, on all rebuilt
inputs together: within the draw noise (per node sd measured: IC 0.014, ORCA 0.016,
combined 0.024), and with no overall bias. The same chi^2 check on SK, which has no
draw noise, is much tighter (measured: 7e-7).
"""
import os

import numpy as np
import pytest

from conftest import SCANS, inputs_present, load_config
from test_published_points import _evaluate, _legs

pytestmark = pytest.mark.build

FILES = ["sk/sk_response.npz", "sk/osc_tensor_pe_nominal.npz", "sk/sk_response_Ninv.npz",
         "ic/ic_response_modeaxis_L3.npz",
         "orca/orca_response_flat900.npz", "ic/ic_aux.npz", "orca/orca_aux.npz",
         "fluxq/fluxq_sk.npz", "fluxq/fluxq_ic.npz", "fluxq/fluxq_orca.npz"]
GRID_KEYS = {"e_edges", "z_edges", "e_true_edges", "e_true_centers", "cz_true_edges",
             "cz_true_centers", "e_c"}
METADATA = {"meta", "provenance", "response_md5", "min_entries"}
RTOL_GRID, RTOL_Q = 1e-12, 1e-13
ATOL_PHI = 1e-6
RTOL_KERNEL = 1e-6
KERNEL_KEYS = {"R_v", "S2_v"}
MAX_SHIFT = {"SK": 1e-5, "IC": 0.1, "ORCA": 0.1, "Combined": 0.1}   # chi^2, any single node
MAX_BIAS = 0.02                                                      # chi^2, mean over nodes


def _compare(rel, built, ref):
    a, b = np.load(built, allow_pickle=True), np.load(ref, allow_pickle=True)
    published = set(b.files) - METADATA
    assert published <= set(a.files), (built, "missing", published - set(a.files))
    assert set(a.files) - published <= METADATA, (built, "extra", set(a.files) - published)
    for k in sorted(published):
        x, y = a[k], b[k]
        if k in GRID_KEYS:
            np.testing.assert_allclose(x, y, rtol=RTOL_GRID, atol=0, err_msg=k)
        elif k.startswith("q_"):
            np.testing.assert_allclose(x, y, rtol=RTOL_Q, atol=0, err_msg=k)
        elif k == "phi":
            np.testing.assert_allclose(x, y, rtol=0, atol=ATOL_PHI, err_msg=k)
        elif k in KERNEL_KEYS and rel.endswith("Ninv.npz"):
            np.testing.assert_allclose(x, y, rtol=RTOL_KERNEL, atol=0, err_msg=k)
        else:
            assert x.shape == y.shape and np.array_equal(x, y), f"{built}: {k} differs"


@pytest.mark.parametrize("rel", FILES)
def test_built_inputs_equal_published(rel, work_dir, reference_dir):
    built, ref = os.path.join(work_dir, rel), os.path.join(reference_dir, rel)
    if not (os.path.exists(built) and os.path.exists(ref)):
        pytest.skip(f"{rel} missing in PYNU_WORK or PYNU_REFERENCE")
    _compare(rel, built, ref)


def test_sk_tensors_match_published(work_dir, reference_dir, reference_cells):
    worst, n = 0.0, 0
    for (o, i, j) in _legs(reference_cells, "SK"):
        name = os.path.join("tensors", "sk", o, f"osc_tensor_{i:03d}_{j:03d}.npz")
        b, r = os.path.join(work_dir, name), os.path.join(reference_dir, name)
        if not (os.path.exists(b) and os.path.exists(r)):
            continue
        x, y = np.load(b)["phi"].astype(float), np.load(r)["phi"].astype(float)
        assert x.shape == y.shape, name
        worst = max(worst, float(np.abs(x - y).max()))
        n += 1
    if n == 0:
        pytest.skip("no rebuilt SK tensors of the test legs in PYNU_WORK")
    print(f"SK tensors: {n} cells, max |delta phi| = {worst:.2e}")
    assert worst < ATOL_PHI


@pytest.mark.parametrize("scan", SCANS)
def test_chi2_on_rebuilt_inputs(scan, work_dir, reference_cells):
    from pynu.binned.scan import Fitter
    shifts = []
    for (o, i, j), cells in _legs(reference_cells, scan).items():
        cfg = load_config(scan, work_dir)
        if not inputs_present(cfg, o, [(i, j)]):
            continue
        F = Fitter(cfg, o)
        phis = {e: t.load(i, j) for e, t in F.dirs.items()}
        for c in cells:
            f, *_ = _evaluate(scan, F, i, j, c["cell"][2], phis, np.asarray(c["theta"], float))
            shifts.append(f - c["chi2"])
    if not shifts:
        pytest.skip(f"no rebuilt {scan} inputs of the test legs in PYNU_WORK")
    shifts = np.array(shifts)
    print(f"{scan}: {shifts.size} nodes, chi2 shift mean {shifts.mean():+.2e} "
          f"sd {shifts.std():.2e} max |.| {np.abs(shifts).max():.2e}")
    assert np.abs(shifts).max() < MAX_SHIFT[scan]
    assert abs(shifts.mean()) < MAX_BIAS
