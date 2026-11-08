"""The paper's numbers from the published chi^2 cubes, through this code's summary.

``scan assemble`` reduces a chi^2 cube to the best fit and the Delta chi^2 = 1
intervals. Applied to the published cubes it must give the published floors,
best-fit cells and intervals, and the numbers quoted in the paper. No data needed.
"""
import pytest

from conftest import SCANS, load_config
from pynu.binned.scan import summarize


@pytest.mark.parametrize("scan", SCANS)
@pytest.mark.parametrize("ordering", ["NO", "IO"])
def test_floor_and_best_fit(scan, ordering, published_cubes, published_summary):
    cfg = load_config(scan, "/nonexistent")
    s = summarize(cfg, ordering, published_cubes[f"{scan}_{ordering}"])
    pub = published_summary[f"{scan}_{ordering}"]
    assert s["floor"] == pub["floor"]
    assert s["argmin"] == list(pub["argmin"])
    assert s["n_cells"] == s["n_expected"]


@pytest.mark.parametrize("ordering", ["NO", "IO"])
def test_combined_intervals(ordering, published_cubes, published_summary):
    cfg = load_config("Combined", "/nonexistent")
    s = summarize(cfg, ordering, published_cubes[f"Combined_{ordering}"])
    pub = published_summary[f"Combined_{ordering}"]
    for mine, theirs in ((s["dm31_1sigma"], pub["dm31_1sigma"]),
                         (s["s23_1sigma"], pub["s23_1sigma"])):
        assert mine["lo"] == pytest.approx(theirs["lo"], rel=1e-12)
        assert mine["hi"] == pytest.approx(theirs["hi"], rel=1e-12)
    d, pd = s["dcp_1sigma"], pub["dcp_1sigma"]
    assert d["best_deg"] == pytest.approx(pd["argmin_deg"])
    assert d["plus_deg"] == pytest.approx(pd["plus_deg"], rel=1e-12)
    assert d["minus_deg"] == pytest.approx(pd["minus_deg"], rel=1e-12)
    assert d["depth"] == pytest.approx(pd["depth"], rel=1e-12)


@pytest.mark.parametrize("ordering", ["NO", "IO"])
def test_sk_intervals(ordering, published_cubes, published_summary):
    cfg = load_config("SK", "/nonexistent")
    s = summarize(cfg, ordering, published_cubes[f"SK_{ordering}"])
    pub = published_summary[f"SK_{ordering}"]
    assert sorted([s["dm31_1sigma"]["lo"], s["dm31_1sigma"]["hi"]]) == \
        pytest.approx(sorted(pub["dm31_1sigma"]), rel=1e-12)
    assert [s["s23_1sigma"]["lo"], s["s23_1sigma"]["hi"]] == \
        pytest.approx(pub["s23_1sigma"], rel=1e-12)


def test_numbers_quoted_in_the_paper(published_cubes, published_summary):
    """Abstract of arXiv:2610.12208: Delta m^2_31 = 2.44 +0.06 -0.04 (1e-3 eV^2),
    sin^2 theta_23 = 0.52 +0.03 -0.04, delta_CP = 1.30 +0.24 -0.28 pi, and the
    ordering preference Delta chi^2(IO - NO) = 11.0."""
    cfg = load_config("Combined", "/nonexistent")
    no = summarize(cfg, "NO", published_cubes["Combined_NO"])
    io = summarize(cfg, "IO", published_cubes["Combined_IO"])
    paper = published_summary["paper"]
    dm, s23, dcp = no["dm31_1sigma"], no["s23_1sigma"], no["dcp_1sigma"]
    assert [round(dm["best"], 2), round(dm["plus"], 2), round(dm["minus"], 2)] == paper["dm31"]
    assert [round(s23["best"], 2), round(s23["plus"], 2), round(s23["minus"], 2)] == paper["s23"]
    assert [round(dcp["best_deg"] / 180, 2), round(dcp["plus_deg"] / 180, 2),
            round(dcp["minus_deg"] / 180, 2)] == paper["dcp_pi"]
    assert round(io["floor"] - no["floor"], 1) == paper["dchi2_io_minus_no"]
