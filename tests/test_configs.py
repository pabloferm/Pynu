"""The four scan configurations: structure, parameter tables and bound rules.

No data needed.
"""
import numpy as np
import pytest

from conftest import SCANS, config_path, load_config
from pynu.binned.arms import driver_bounds

EXPECTED = {  # (parameters, free parameters) of each scan of record
    "SK": (136, 133),
    "IC": (40, 37),
    "ORCA": (30, 26),
    "Combined": (151, 143),
}
LEGS = {"SK": (400, 400), "IC": (1681, 1681), "ORCA": (1681, 1681), "Combined": (400, 400)}


@pytest.fixture(scope="module", params=SCANS)
def cfg(request):
    return load_config(request.param, "/nonexistent")


def test_parameter_counts(cfg):
    n, free = EXPECTED[cfg.name]
    assert len(cfg.dials) == n
    assert sum(not d.pinned for d in cfg.dials) == free


def test_pinned_parameters_sit_at_nominal(cfg):
    for d in cfg.dials:
        if d.pinned:
            assert d.lo == d.hi == d.nominal, d.name


@pytest.mark.parametrize("scan", ["IC", "ORCA", "Combined"])
def test_driver_bound_rule(scan):
    """IC, ORCA and the combined shared slots: nominal +- 5 sigma, floor 0.01. (SK's
    bounds are the engine box, checked when the engine is built.)"""
    cfg = load_config(scan, "/nonexistent")
    ds = [d for d in cfg.dials if not d.pinned and d.block in ("", "shared")]
    lo, hi = driver_bounds(np.array([d.nominal for d in ds]), np.array([d.sigma for d in ds]))
    assert list(lo) == [d.lo for d in ds]
    assert list(hi) == [d.hi for d in ds]


def test_combined_layout():
    cfg = load_config("Combined", "/nonexistent")
    shared = [d for d in cfg.dials if d.block == "shared"]
    sk = [d for d in cfg.dials if d.block == "sk"]
    assert (len(shared), len(sk)) == (47, 104)
    assert [d.name for d in cfg.dials] == [d.name for d in shared + sk]
    assert set(cfg.arms["SK"].dials) <= {d.name for d in cfg.dials}
    assert len(cfg.arms["SK"].dials) == 136
    assert len(cfg.arms["IC"].dials) == 40 and len(cfg.arms["ORCA"].dials) == 30
    # the two names with a pinned shared slot and SK's own pinned slot
    both = sorted({d.name for d in shared} & {d.name for d in sk})
    assert both == ["flavor_ratio", "xsec_ccqe_shape"]


def test_regions_and_tasks(cfg):
    for o in ("NO", "IO"):
        assert len(cfg.region_cells(o)) == LEGS[cfg.name][0 if o == "NO" else 1]
    for ref in cfg.references:
        i, j, d = ref.cell
        assert (i, j) in cfg.region_cells(ref.ordering)
        assert d in cfg.dcp_order(ref.ordering)


def test_lattice():
    cfg = load_config("Combined", "/nonexistent")
    la = cfg.lattice
    no, io = la.dm31_axis("NO"), la.dm31_axis("IO")
    assert np.allclose(io, -(no - la.dm21)[::-1], rtol=0, atol=1e-18)
    assert la.dcp_nodes().size == 20 and la.dcp_nodes()[1] == pytest.approx(np.pi / 10)


def test_paths_need_pynu_work():
    from pynu.binned import config as C
    cfg = C.load(config_path("SK"))
    assert "${PYNU_WORK}" in cfg.inputs["sk_response"]


def test_input_you_built_warns_but_runs(tmp_path):
    """A kernel you built differs from the published file in md5 (metadata, round-off)."""
    cfg = load_config("SK", str(tmp_path))
    p = tmp_path / "sk" / "sk_response_Ninv.npz"
    p.parent.mkdir()
    p.write_bytes(b"not the published file")
    with pytest.warns(UserWarning, match="sk_response"):
        assert cfg.input("sk_response") == str(p)


def test_check_verdicts():
    from pynu.binned.scan import _verdict
    ic = load_config("IC", "/nonexistent").references[0]
    sk = load_config("SK", "/nonexistent").references[0]
    assert _verdict(1e-5, ic) == "PASS"
    assert _verdict(0.03, ic).startswith("PASS for rebuilt")
    assert _verdict(0.5, ic) == "FAIL"
    assert _verdict(0.003, sk).startswith("PASS for rebuilt")
    assert _verdict(0.03, sk) == "FAIL"     # SK tensors carry no draw noise


def test_fit_stage_needs_no_event_framework():
    import subprocess
    import sys
    code = ("import sys, pynu.binned.scan, pynu.binned.combined; "
            "bad = [m for m in ('nuSQuIDS', 'nuflux', 'pandas', 'h5py', 'pynu.PyNuFit') "
            "if m in sys.modules]; sys.exit(1 if bad else 0)")
    from conftest import ROOT
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode == 0
