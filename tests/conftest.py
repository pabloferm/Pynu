"""Shared fixtures of the test suite.

Inputs are located through two environment variables, both optional:

  PYNU_REFERENCE  the published inputs (responses, Ninv, aux, flux tables, tensors of
                  the test cells), laid out like ${PYNU_WORK}. Tests that compare with
                  the published results evaluate on these.
  PYNU_WORK       inputs you built yourself with scripts/build.py. When both are set,
                  tests/test_build.py compares them with the published ones.

Tests whose inputs are missing are skipped, so ``pytest tests`` always runs the
data-free checks. Markers: ``slow`` (refits, minutes), ``build`` (built vs published).
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
REF = os.path.join(ROOT, "tests", "reference")
SCANS = ("SK", "IC", "ORCA", "Combined")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: refits the published reference cells (minutes)")
    config.addinivalue_line("markers", "build: compares inputs you built with the published ones")


def config_path(scan):
    return os.path.join(ROOT, "configs", f"{scan}.xml")


def load_config(scan, work):
    """Load a scan configuration with ${PYNU_WORK} pointing at ``work``."""
    from pynu.binned import config as C
    old = os.environ.get("PYNU_WORK")
    os.environ["PYNU_WORK"] = work
    try:
        cfg = C.load(config_path(scan))
        cfg.inputs = {k: C.expand_path(v) for k, v in cfg.inputs.items()}
    finally:
        if old is None:
            del os.environ["PYNU_WORK"]
        else:
            os.environ["PYNU_WORK"] = old
    return cfg


def inputs_present(cfg, ordering, legs):
    """True if every input file (and the tensors of ``legs``) exists."""
    for k, p in cfg.inputs.items():
        if k.endswith("_tensors"):
            for i, j in legs:
                if not os.path.exists(os.path.join(p.replace("{ordering}", ordering),
                                                   f"osc_tensor_{i:03d}_{j:03d}.npz")):
                    return False
        elif not os.path.exists(p):
            return False
    return True


@pytest.fixture(scope="session")
def reference_cells():
    with open(os.path.join(REF, "cells.json")) as fh:
        return json.load(fh)["cells"]


@pytest.fixture(scope="session")
def published_summary():
    with open(os.path.join(REF, "published_summary.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def published_cubes():
    return dict(np.load(os.path.join(REF, "published_cubes.npz")))


@pytest.fixture(scope="session")
def reference_dir():
    d = os.environ.get("PYNU_REFERENCE")
    if not d or not os.path.isdir(d):
        pytest.skip("PYNU_REFERENCE (the published inputs) is not set")
    return d


@pytest.fixture(scope="session")
def work_dir():
    d = os.environ.get("PYNU_WORK")
    if not d or not os.path.isdir(d):
        pytest.skip("PYNU_WORK (your built inputs) is not set")
    return d
