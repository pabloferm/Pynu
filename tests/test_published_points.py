"""chi^2 and its gradient at the published fitted parameters.

For every delta_CP node of the test legs (230 lattice nodes in all), the parameters
the published scan converged to are evaluated with this code on the published
inputs. The chi^2, and each part of it, must equal the published value: no
minimizer is involved, so the agreement is limited only by floating-point rounding.
The analytic gradients are checked against finite differences at the same points.
"""
import os
from collections import defaultdict

import numpy as np
import pytest

from conftest import SCANS, inputs_present, load_config

RTOL = 1e-9          # relative to chi^2 (published agreement seen: ~1e-15)


def _legs(cells, scan):
    out = defaultdict(list)
    for c in cells[scan]:
        out[(c["ordering"], c["cell"][0], c["cell"][1])].append(c)
    return out


def _model(scan, ordering, ref):
    from pynu.binned.scan import Fitter
    cfg = load_config(scan, ref)
    return cfg, Fitter(cfg, ordering)


def _evaluate(scan, F, i, j, d, phis, theta):
    """(chi^2, parts, gradient wrt the free parameters, value function) at theta."""
    m = F.model
    dm = float(F.dm31[i])
    if scan == "Combined":
        m.bind(dm, phis["SK"][d].astype(float), np.asarray(phis["IC"][d], float),
               np.asarray(phis["ORCA"][d], float))
        f, g, (ic, orca, sk, prior) = m.chi2_and_grad(theta)
        free = np.array([a < b for a, b in m.bounds])
        return f, dict(ic=ic, orca=orca, sk=sk, prior=prior), g, free, \
            lambda x: m.chi2_and_grad(x)[0]
    if scan == "SK":
        phi = phis["SK"][d].astype(float)
        f, g = m.eng.chi2_and_grad(phi, theta)
        return float(f), dict(sk=float(f)), np.asarray(g), m.free, \
            lambda x: float(m.eng.chi2(phi, x))
    m.arm.bind(dm)
    phi = np.asarray(phis[scan][d], float)
    eng, nom, sig = m.arm.eng, m.nominal, m.sigma
    c, g = eng.chi2_and_grad(phi, theta)
    prior = float(np.sum(((theta - nom) / sig) ** 2))
    g = np.asarray(g, float) + 2.0 * (theta - nom) / sig ** 2
    free = np.zeros(theta.size, bool)
    free[m.swept] = True

    def value(x):
        return float(eng.chi2(phi, x)) + float(np.sum(((x - nom) / sig) ** 2))
    return float(c) + prior, dict(stat=float(c), prior=prior), g, free, value


@pytest.mark.parametrize("scan", SCANS)
def test_chi2_at_published_parameters(scan, reference_dir, reference_cells):
    worst = 0.0
    for (o, i, j), cells in _legs(reference_cells, scan).items():
        cfg, F = _model(scan, o, reference_dir)
        if not inputs_present(cfg, o, [(i, j)]):
            pytest.skip(f"published inputs for {scan} {o} ({i},{j}) not in PYNU_REFERENCE")
        phis = {e: t.load(i, j) for e, t in F.dirs.items()}
        for c in cells:
            th = np.asarray(c["theta"], float)
            f, parts, _, _, _ = _evaluate(scan, F, i, j, c["cell"][2], phis, th)
            assert f == pytest.approx(c["chi2"], rel=RTOL, abs=0), (scan, o, c["cell"])
            for k, v in c["parts"].items():
                assert parts[k] == pytest.approx(v, rel=RTOL, abs=1e-9), (scan, o, c["cell"], k)
            worst = max(worst, abs(f - c["chi2"]))
    print(f"{scan}: max |chi2 - published| = {worst:.2e}")


@pytest.mark.parametrize("scan", SCANS)
def test_gradient_matches_finite_differences(scan, reference_dir, reference_cells):
    """Analytic gradient vs central differences, 8 free parameters per point, at the
    published best fit of each ordering."""
    rng = np.random.default_rng(20261104)
    legs = _legs(reference_cells, scan)
    for o in ("NO", "IO"):
        key = min((k for k in legs if k[0] == o),
                  key=lambda k: min(c["chi2"] for c in legs[k]))
        c = min(legs[key], key=lambda c: c["chi2"])
        cfg, F = _model(scan, o, reference_dir)
        if not inputs_present(cfg, o, [key[1:]]):
            pytest.skip(f"published inputs for {scan} {o} not in PYNU_REFERENCE")
        i, j, d = c["cell"]
        phis = {e: t.load(i, j) for e, t in F.dirs.items()}
        th = np.asarray(c["theta"], float)
        _, _, g, free, value = _evaluate(scan, F, i, j, d, phis, th)
        for k in rng.choice(np.flatnonzero(free), size=8, replace=False):
            h = 1e-6 * max(1.0, abs(th[k]))
            up, dn = th.copy(), th.copy()
            up[k] += h
            dn[k] -= h
            fd = (value(up) - value(dn)) / (2 * h)
            assert g[k] == pytest.approx(fd, rel=1e-3, abs=1e-3), (scan, o, k)
