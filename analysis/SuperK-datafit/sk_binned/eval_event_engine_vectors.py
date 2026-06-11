#!/usr/bin/env python3
"""Gate-B reference: event-engine binned expectations at deterministic test
nuisance vectors, for validation of the binned forward model.

At point A physics (the near-truth comparison point), evaluates the FULL
event-by-event expectation (ApplyNuisanceWeights over 3.8M events -> BinMC)
at: the nominal vector, and 5 seeded pseudo-random perturbations that exercise
every enabled nuisance (ratio-type params perturbed multiplicatively ~5-15%,
offset-type params (nominal==0, e.g. tilt/barr_zenith) shifted by ~0.3-0.5 sigma-ish
absolute steps). Saves vectors + per-bin expectations.

Usage (cluster):
    python eval_event_engine_vectors.py --config <xsec_barr_ntag.xml> --output gateB_reference.npz
"""
import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from run_sk_datafit_row_worker import setup_pynufit_datafit, set_physics_params  # noqa: E402

POINT_A = dict(dm231=np.linspace(2.0e-3, 3.5e-3, 15)[5],
               s23=np.linspace(0.40, 0.80, 15)[6],
               dcp=2.0 * np.pi * 6 / 13)  # the BB best dcp at point A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default="gateB_reference.npz")
    ap.add_argument("--n-vectors", type=int, default=5)
    args = ap.parse_args()

    pynufit, _, _ = setup_pynufit_datafit(args.config)
    exp_key = list(pynufit.Experiments.keys())[0]
    exp = pynufit.Experiments[exp_key]
    set_physics_params(pynufit, POINT_A["dm231"], POINT_A["s23"], POINT_A["dcp"])

    nominal = np.array(pynufit.Analysis.NuisNominalList, dtype=float)
    names = list(pynufit.Analysis.NuisanceList)
    rng = np.random.default_rng(20260611)

    vectors = [nominal.copy()]
    for _ in range(args.n_vectors):
        z = rng.standard_normal(nominal.size)
        v = np.where(nominal != 0.0,
                     nominal * (1.0 + 0.08 * z),   # ratio-type: ~8% relative
                     0.35 * z)                      # offset-type: ~0.35 absolute
        vectors.append(v)

    expectations = []
    for i, v in enumerate(vectors):
        pynufit.StartNuisance()
        pynufit.ApplyNuisanceWeights(v)
        pynufit.SetExpectedWeights()
        nb = np.asarray(exp.BinMC(exp.ExpectedWeight), float)
        expectations.append(nb)
        print(f"[gateB] vector {i}: n_model = {nb.sum():.2f}")

    # also the pure-physics binned vector (for migration-ratio r cross-checks)
    n_phys = np.asarray(exp.BinMC(np.asarray(exp.PhysicsWeight, float)), float)

    np.savez_compressed(
        args.output,
        vectors=np.array(vectors), expectations=np.array(expectations),
        n_phys=n_phys, names=np.array(names),
        meta=json.dumps({"point": {k: float(v) for k, v in POINT_A.items()},
                         "config": args.config}),
    )
    print(f"[gateB] wrote {args.output}")


if __name__ == "__main__":
    main()
