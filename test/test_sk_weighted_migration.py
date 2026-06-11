"""Local regression test for Step 1 (Finding 6): weighted-rate migration in
SKCombinedDetector.SuperK_Combined.

Pure-numpy, no nuSQuIDS/MC needed. Verifies:
  1. CONTROL: every migration tune returns mr == 1 at nominal x=1 (=> Dchi2=0 at
     nominal regardless of the r definition).
  2. WEIGHTED r: the acceptor compensation uses W=BaseWeight*PhysicsWeight, not raw
     counts (raw would give a different factor when per-event weights vary).
  3. RATE CONSERVATION: total *weighted* rate is conserved across a migration (the
     physical property the raw-count version violated).
  4. forward/diff consistency: diff acceptor factor == -r (weighted).

Run: python test/test_sk_weighted_migration.py   (or pytest)
"""
import os, sys
import numpy as np
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pynu.PhysicsTunes.Detector.SKCombinedDetector import SuperK_Combined


def make_mock():
    """Mock experiment where per-event weights vary across samples, so weighted-rate
    ratios differ from raw-count ratios. Every reco sample 0..28 is populated (so all
    migration tunes have well-defined r), plus heavy/phys-weighted events that make the
    weighted rate diverge from raw counts."""
    # (sample, base_weight, phys_weight, decay_e)
    rows = [(s, 1.0, 1.0, s % 3) for s in range(29)]  # one unit event per sample
    rows += [
        (10, 3.0, 2.0, 0),  # heavy + phys-weighted -> weighted rate(10) >> raw count
        (17, 2.0, 1.0, 0),  # upmu non-shower extra weight
        (21, 2.0, 1.0, 0),  # neutron nn1 extra weight
    ]
    sample = np.array([r[0] for r in rows])
    base = np.array([r[1] for r in rows], dtype=float)
    phys = np.array([r[2] for r in rows], dtype=float)
    decay = np.array([r[3] for r in rows])
    return SimpleNamespace(
        Sample=sample, BaseWeight=base, PhysicsWeight=phys,
        DecayE=decay, NumberOfEvents=len(rows),
    )


MIGRATION_FWD = [
    "multiring_nunubar_separation", "multiring_emu_separation",
    "multiring_eother_separation", "pc_stopthru_separation", "pi0_ring_separation",
    "e_ring_separation", "mu_ring_separation", "singlering_pid", "multiring_pid",
    "neutron_tagging", "decay_e_tagging", "upmu_shower_separation",
]


def test_control_unity_at_nominal():
    """mr == 1 for every migration tune at x=1 (Dchi2=0 at nominal)."""
    sk, exp = SuperK_Combined(), make_mock()
    for name in MIGRATION_FWD:
        mr = getattr(sk, name)(exp, 1.0)
        mr = np.asarray(mr, dtype=float)
        assert np.allclose(mr, 1.0, atol=1e-12), f"{name}: mr != 1 at x=1 (max dev {np.max(np.abs(mr-1)):.2e})"


def test_weighted_ratio_not_raw():
    """multiring_nunubar acceptor factor uses weighted r, not raw counts."""
    sk, exp = SuperK_Combined(), make_mock()
    x = 0.8
    mr = np.asarray(sk.multiring_nunubar_separation(exp, x), dtype=float)
    W = exp.BaseWeight * exp.PhysicsWeight
    r_w = np.sum(W[exp.Sample == 10]) / np.sum(W[exp.Sample == 11])  # weighted
    r_raw = np.sum(exp.Sample == 10) / np.sum(exp.Sample == 11)      # raw counts
    acc_w = 1 + r_w * (1 - x)
    acc_raw = 1 + r_raw * (1 - x)
    assert not np.isclose(acc_w, acc_raw), "test mock too symmetric (weighted==raw)"
    got = mr[exp.Sample == 11][0]
    assert np.isclose(got, acc_w), f"acceptor {got:.4f} != weighted {acc_w:.4f} (raw would be {acc_raw:.4f})"
    assert np.isclose(mr[exp.Sample == 10][0], x), "donor factor must be x"


def test_weighted_rate_conserved():
    """Total WEIGHTED rate is conserved across the migration (raw version violated this)."""
    sk, exp = SuperK_Combined(), make_mock()
    W = exp.BaseWeight * exp.PhysicsWeight
    donor, acc = (exp.Sample == 10), (exp.Sample == 11)
    before = np.sum(W[donor]) + np.sum(W[acc])
    for x in (0.5, 0.8, 1.2):
        mr = np.asarray(sk.multiring_nunubar_separation(exp, x), dtype=float)
        after = np.sum((mr * W)[donor]) + np.sum((mr * W)[acc])
        assert np.isclose(before, after), f"weighted rate not conserved at x={x}: {before:.4f} -> {after:.4f}"


def test_diff_matches_minus_r():
    """diff acceptor factor == -r (weighted), consistent with forward 1+r(1-x)."""
    sk, exp = SuperK_Combined(), make_mock()
    W = exp.BaseWeight * exp.PhysicsWeight
    r_w = np.sum(W[exp.Sample == 10]) / np.sum(W[exp.Sample == 11])
    d = np.asarray(sk.diff_multiring_nunubar_separation(exp, 0.8), dtype=float)
    assert np.isclose(d[exp.Sample == 11][0], -r_w), "diff acceptor != -r_weighted"
    assert np.isclose(d[exp.Sample == 10][0], 1.0), "diff donor != 1"


def test_raw_basis_backcompat():
    """MIGRATION_BASIS='raw' reproduces the original raw-count ratios exactly."""
    sk, exp = SuperK_Combined(), make_mock()
    x = 0.8
    old_basis = SuperK_Combined.MIGRATION_BASIS
    try:
        SuperK_Combined.MIGRATION_BASIS = "raw"
        mr = np.asarray(sk.multiring_nunubar_separation(exp, x), dtype=float)
    finally:
        SuperK_Combined.MIGRATION_BASIS = old_basis
    r_raw = np.sum(exp.Sample == 10) / np.sum(exp.Sample == 11)
    assert np.isclose(mr[exp.Sample == 11][0], 1 + r_raw * (1 - x)), \
        "raw basis: acceptor factor != 1 + r_raw*(1-x)"
    assert np.isclose(mr[exp.Sample == 10][0], x), "raw basis: donor factor must be x"


if __name__ == "__main__":
    tests = [test_control_unity_at_nominal, test_weighted_ratio_not_raw,
             test_weighted_rate_conserved, test_diff_matches_minus_r,
             test_raw_basis_backcompat]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
