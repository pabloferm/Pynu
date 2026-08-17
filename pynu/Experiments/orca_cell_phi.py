"""Exact cell-phi extraction for the ORCA arm — no tensor build, no nuSQuIDS code.

★ THE KEY FINDING (design ADDENDUM item 5; scope §2.2). The production ORCA arm
does NOT get its physics from a tensor. It gets it from `exp.PhysicsWeight`, set
by `ApplyOscillations("Physics")` once per grid cell
(`combined_ic_orca_fit_worker.py:307-308`). PhysicsWeight is the PROPAGATED FLUX
— InitialFlux x P_osc — and is a function of (pdg, ETrue, cos ZTrue) only, with
oscillation averaging OFF (`orca_osc.osc_avg_scale = None`, `:365`).

Under true-cell uniqueness (gate G-C2: exactly one ETrue and one cos ZTrue value
per true bin, measured 1 and 1) PhysicsWeight is therefore CONSTANT within every
(pdg, ie, iz) group. So the (2, 3, n_etrue, n_cztrue) cell-phi the binned engine
needs is a groupby-FIRST gather off the live event array: exact, free, and with
no `orca_build_tensors.py` run in the loop.

That is the single design choice that makes G-ORCA-1's 1e-9 target reachable
rather than aspirational, so it is gated rather than trusted:

  G-ORCA-0  max within-group spread of PhysicsWeight == 0, BIT-EXACT (not a
            tolerance), plus a hard assert that osc_avg_scale is None. If
            averaging is ever switched on, PhysicsWeight stops being
            cell-constant and the port silently becomes approximate with no
            other symptom — that is risk R1, and this is its whole treatment.

Index convention is SK's, shared with `ic_divergence_scan.nu_index`
(`sk_binned_engine.py:215-216`):  ntype = (pdg < 0),  flavor = |pdg|//2 - 6.
"""
import numpy as np

from .orca_binned_support import nu_cell_index               # noqa: E402

N_NTYPE, N_FLAVOR = 2, 3


class ORCACellIndex:
    """Per-event true-cell indices in the Orca-loader row order.

    Fields are aligned 1:1 with `exp.ETrue` / `exp.PhysicsWeight` etc., which is
    what makes the gather exact. `nu_cell_index` applies the identical filter the
    loader does (MC_type == 1, then 0 <= ETrue < 1e5) — gate G-C0 asserts the
    second cut is a no-op, so the two row orders coincide.
    """

    __slots__ = ("ntype", "flavor", "ie", "iz", "pdg", "bin900", "weight",
                 "e_centers", "cz_centers", "n_etrue", "n_cztrue", "key")

    def __init__(self, ntype, flavor, ie, iz, pdg, bin900, e_centers, cz_centers,
                 weight):
        self.ntype, self.flavor, self.ie, self.iz = ntype, flavor, ie, iz
        self.pdg, self.bin900, self.weight = pdg, bin900, weight
        self.e_centers, self.cz_centers = e_centers, cz_centers
        self.n_etrue, self.n_cztrue = int(e_centers.size), int(cz_centers.size)
        self.key = (((ntype * N_FLAVOR + flavor) * self.n_etrue + ie)
                    * self.n_cztrue + iz)

    @property
    def n_events(self):
        return int(self.ie.size)


def build_cell_index(mc_parquet):
    """Build the per-event ORCA cell index from the MC parquet.

    Reuses `orca_exact_scan.nu_cell_index` for the bin/true indices and the
    quantized true centres (the production definitions), and reads `pdg` under
    the identical row filter.
    """
    import pandas as pd
    bin900, ie, iz, e_c, z_c = nu_cell_index(mc_parquet)
    df = pd.read_parquet(mc_parquet)
    nu = df[df["MC_type"] == 1]
    et = nu["true_energy"].values
    nu = nu[(et >= 0) & (et < 1e5)]                # the loader's cut, Orca.py:126-127
    pdg = nu["pdg"].values.astype(np.int64)
    # RAW parquet weight, the same quantity the response stores in R_v. NORM is a
    # constant factor and cancels out of every weighted mean, so it is not applied.
    weight = nu["weight"].values.astype(float)
    if pdg.size != ie.size:
        raise AssertionError(f"pdg rows {pdg.size} != cell-index rows {ie.size}")
    ntype = (pdg < 0).astype(np.int64)
    flavor = np.abs(pdg) // 2 - 6
    if not ((flavor >= 0) & (flavor < N_FLAVOR)).all():
        raise AssertionError("flavor index out of range — unexpected pdg in the ORCA MC")
    return ORCACellIndex(ntype, flavor, ie, iz, pdg, bin900, e_c, z_c, weight)


def extract_cell_phi(exp, cell_index, osc=None, how="weighted_mean"):
    """Reduce exp.PhysicsWeight to a (2, 3, n_etrue, n_cztrue) cell-phi.

    ★ WHY THIS IS A MEAN AND NOT A GATHER (2026-08-17, probe job 39685508).
    The original design took the group's FIRST event, on the premise that
    PhysicsWeight is cell-constant. That premise is FALSE on the live path:
    `AtmOsc._eval_flavor_weights` (AtmoFlux's sibling
    `Oscillations/AtmOsc.py:108-122`) passes `itertools.repeat(True)` as
    nuSQuIDS' `randomize_production_height`, so every event draws its own
    production height and two events in the same true cell get different
    oscillation weights BY DESIGN. Measured: draws are NOT reproducible across
    sweeps (TEST A, max rel diff 7.8), within-cell relative spread median
    2.5e-05 / p95 1.4e-02.

    ★ WHY THE WEIGHTING IS BY MC WEIGHT — a conservation argument, not a taste.
    The event path's bin content is
        n_b = sum_{i in b} w_i * phi_i * dials(cell_i)
    while the binned engine computes
        n_b = sum_c phi_c * dials(c) * (sum_{i in c and b} w_i).
    Matching those bin-by-bin would need phi_c to be the w-weighted mean over
    `c AND b`, which depends on b — so NO single phi_c is exact for every bin.
    But summing over bins, the two agree EXACTLY iff

        phi_c = sum_{i in c} w_i phi_i / sum_{i in c} w_i          (this function)

    i.e. the weight-weighted mean is the unique choice that conserves each
    cell's total contribution, and hence the grand total. A PLAIN mean gets the
    cell total wrong by the within-cell covariance between w_i and phi_i, and
    that covariance is not small here: within-cell weight dispersion is median
    0.72, p95 2.59 (relative sd), measured on the production MC.

    Residual, irreducible: the bin-by-bin mismatch above, which is the
    within-cell covariance between weight and the drawn height. It cannot be
    removed by any cell-phi, and it is bounded below by the reference's own
    draw noise anyway (probe TEST A floor, rel chi2 2.193e-05).

    Args:
      exp:        the live ORCA Experiment, AFTER ApplyOscillations("Physics").
      cell_index: an `ORCACellIndex` (see `build_cell_index`).
      osc:        optional OscillationTunes; its osc_avg_scale is HARD-ASSERTED
                  to be None. Still load-bearing: with averaging ON, AtmOsc takes
                  the other EvalFlavor overload entirely (`:124-127`) and every
                  number in this docstring changes.
      how:        "weighted_mean" (default, the conserving choice), "mean"
                  (unweighted — kept for the estimator comparison only), or
                  "first" (the retired pre-2026-08-17 behaviour).

    Returns:
      (phi, info) — info carries the G-ORCA-0 DIAGNOSTIC numbers: within-cell
      spread absolute AND relative (the failing gate reported only absolute, on a
      PhysicsWeight whose median is ~7e-9, which made it uninterpretable).
    """
    if osc is not None:
        scale = getattr(osc, "osc_avg_scale", None)
        if scale is not None:
            raise AssertionError(
                f"osc_avg_scale is {scale!r}, must be None. With averaging ON, "
                "AtmOsc takes the non-randomizing EvalFlavor overload "
                "(AtmOsc.py:124-127) and this module's whole calibration — the "
                "measured spread, the floor, the G-ORCA-1 tolerance — no longer "
                "applies.")

    pw = np.asarray(exp.PhysicsWeight, float)
    if pw.size != cell_index.n_events:
        raise AssertionError(
            f"PhysicsWeight rows {pw.size} != cell-index rows {cell_index.n_events} "
            "— the event array and the parquet index are not aligned")

    key = cell_index.key
    w = cell_index.weight
    n_flat = N_NTYPE * N_FLAVOR * cell_index.n_etrue * cell_index.n_cztrue
    cnt = np.bincount(key, minlength=n_flat).astype(float)
    phi_flat = np.zeros(n_flat, float)

    if how == "first":
        uniq, first = np.unique(key, return_index=True)
        phi_flat[uniq] = pw[first]
    elif how == "mean":
        s = np.bincount(key, weights=pw, minlength=n_flat)
        np.divide(s, cnt, out=phi_flat, where=cnt > 0)
    elif how == "weighted_mean":
        num = np.bincount(key, weights=w * pw, minlength=n_flat)
        den = np.bincount(key, weights=w, minlength=n_flat)
        np.divide(num, den, out=phi_flat, where=den > 0)
        # 53 populated cells carry zero TOTAL weight (189 events have weight
        # exactly 0). They contribute nothing to any bin, so their phi is
        # irrelevant to the model — but fall back to the plain mean rather than
        # leave a 0/0 that could propagate as a silent zero.
        fallback = (den <= 0) & (cnt > 0)
        if fallback.any():
            s = np.bincount(key, weights=pw, minlength=n_flat)
            phi_flat[fallback] = s[fallback] / cnt[fallback]
    else:
        raise ValueError(f"how={how!r} not in ('weighted_mean', 'mean', 'first')")

    # ---- G-ORCA-0, now a DIAGNOSTIC: how far is phi from cell-constant? ------
    back = phi_flat[key]
    dev = np.abs(pw - back)
    multi = cnt[key] > 1
    rel = dev / np.maximum(np.abs(back), 1e-300)
    info = {
        "how": how,
        # ⚠ read this one with care under a MEAN reduction: sum/sum carries
        # division roundoff, so a genuinely cell-constant phi still reports False
        # here (measured 2.8e-14 relative on a constant input). The *_rel_spread
        # fields are the unambiguous ones; this bool is only meaningful for
        # how="first".
        "cell_constant_bit_exact": bool(np.array_equal(pw, back)),
        "max_abs_spread": float(np.max(dev)) if dev.size else 0.0,
        "max_rel_spread": float(np.max(rel[multi])) if multi.any() else 0.0,
        "median_rel_spread": float(np.median(rel[multi])) if multi.any() else 0.0,
        "p95_rel_spread": float(np.percentile(rel[multi], 95)) if multi.any() else 0.0,
        "n_groups": int((cnt > 0).sum()),
        "n_groups_multi_event": int((cnt > 1).sum()),
        "n_zero_weight_cells": int(((np.bincount(key, weights=w, minlength=n_flat) <= 0)
                                    & (cnt > 0)).sum()),
        "n_events": int(pw.size),
    }
    phi = phi_flat.reshape(N_NTYPE, N_FLAVOR, cell_index.n_etrue, cell_index.n_cztrue)
    return phi, info
