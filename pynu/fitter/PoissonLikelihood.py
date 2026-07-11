"""PoissonLikelihood — a pure-Poisson binned likelihood that mirrors the
``BarlowBeestonLikelihood`` interface so the SK binned engine's modular fit path
reuses the event-engine worker loop verbatim: same constructor signature and
method names as ``BarlowBeestonLikelihood``
(``stats_and_systematics(expectation, nuisance, mc_var)``, ``gradient(...)``),
built by ``set_likelihood('PoissonLikelihood')`` from the XML lists.

Concretely, this class computes the *statistics* term with the binned engine's
own ``SKBinnedEngine.poisson_chi2`` kernel (so the number is bit-identical to
``engine.chi2``), and the *systematics* term with the same Gaussian penalty the
engine uses (``sum((theta - nominal)^2 / sigma^2)``). The expectation and its
Jacobian are handed in already-contracted (by ``PyNuFit.SetBinnedExpectedEvents``
/ ``ComputeBinnedDiffExpectation``), exactly as ``BarlowBeestonLikelihood``
receives them — so the modular worker signature is unchanged.

Gaussian-only: the engine penalty is pure Gaussian, so a non-normal distribution
is a hard error here (the constraint the design says "belongs in the likelihood
class"). ``muon_norm`` is intentionally unsupported (SK has no muon background;
the binned engine owns no such dial).
"""
import sys
from typing import List

import numpy as np


class PoissonLikelihood:
    """Pure-Poisson binned likelihood with a Gaussian nuisance penalty.

    Constructed with the same 4-tuple as ``BarlowBeestonLikelihood``
    (``observation``, ``nominal_nuisance``, ``sigma_nuisance``,
    ``dist_nuisance``). The statistics kernel is delegated to the binned
    engine (``SKBinnedEngine.poisson_chi2``), attached via
    :meth:`set_engine`; the systematics term is the engine's Gaussian penalty.
    """

    def __init__(self, observation, nominal_nuisance, sigma_nuisance,
                 dist_nuisance) -> None:
        self.observation = observation
        self.nominal_nuisance = list(nominal_nuisance)
        self.sigma_nuisance = list(sigma_nuisance)
        self.dist_nuisance = list(dist_nuisance)
        self.number_of_nuisance = len(self.nominal_nuisance)
        self._nominal = np.asarray(self.nominal_nuisance, dtype=float)
        self._sigma = np.asarray(self.sigma_nuisance, dtype=float)
        # Gaussian-only: enforce the engine's assumption at construction so the
        # penalty algebra below is unconditionally valid.
        for name_i, dist in enumerate(self.dist_nuisance):
            if str(dist).strip() != "normal":
                sys.exit(
                    "PoissonLikelihood only supports 'normal' nuisance "
                    f"distributions (engine penalty is pure Gaussian); nuisance "
                    f"index {name_i} has distribution {dist!r}")
        # The engine supplying the statistics kernel; set by set_likelihood via
        # set_engine(). mc_variance is accepted for BB-signature parity but is
        # unused (pure Poisson ignores MC variance).
        self.engine = None
        self.mc_variance = None
        self.muon_norm_index = None  # unsupported; present for signature parity

    # ---- BB-signature-parity setters ----
    def set_mc_variance(self, mc_variance):
        """Accepted for interface parity; pure Poisson ignores MC variance."""
        self.mc_variance = mc_variance

    def set_engine(self, engine):
        """Attach the binned engine whose ``poisson_chi2`` kernel supplies the
        statistics term. Delegation target for design §2.4."""
        self.engine = engine
        return self

    def set_muon_norm_index(self, index):
        """SK has no muon background; a non-None index is a configuration error."""
        if index is not None:
            sys.exit("PoissonLikelihood does not support muon_norm (the SK binned "
                     "engine owns no muon background)")
        self.muon_norm_index = None

    # ---- statistics kernel (delegated to the engine) ----
    def _poisson_stat(self, expectation) -> float:
        """Sum the engine's pure-Poisson statistic over every experiment's
        contracted expectation. Bit-identical to the ``stat`` term of
        ``SKBinnedEngine.chi2`` because it uses the same kernel on the same
        (already FewEntries-filtered) expectation and observation."""
        if self.engine is None:
            sys.exit("PoissonLikelihood.set_engine() was never called; the "
                     "statistics kernel is unavailable")
        X2 = 0.0
        for (exp_name, O), E in zip(self.observation.items(),
                                    expectation.values()):
            X2 += float(self.engine.poisson_chi2(np.asarray(O, dtype=float),
                                                 np.asarray(E, dtype=float)))
        return X2

    # ---- penalty (engine's Gaussian form) ----
    def nuisance_penalty(self, nuisance: List[float]) -> float:
        theta = np.asarray(nuisance, dtype=float)
        return float(np.sum((theta - self._nominal) ** 2 / self._sigma ** 2))

    def stats_only(self, expectation, mc_variance=None,
                   muon_scale=1.0) -> float:
        """Statistics-only chi2 (no penalty). ``mc_variance``/``muon_scale``
        accepted for BB-signature parity and ignored (pure Poisson)."""
        return self._poisson_stat(expectation)

    def stats_and_systematics(self, expectation, nuisance: List[float],
                              mc_variance=None) -> float:
        """Full chi2 = engine Poisson statistic + engine Gaussian penalty.

        Signature identical to ``BarlowBeestonLikelihood.stats_and_systematics``
        so the worker's ``minimize_nuisance`` loop is byte-reusable."""
        return self._poisson_stat(expectation) + self.nuisance_penalty(nuisance)

    def gradient(self, expectation, diff_expectation, nuisance: List[float],
                 mc_variance=None) -> np.ndarray:
        """Analytic gradient with the event engine's first-order convention
        (beta absent in pure Poisson; the migration Jacobian is held at the
        current point). Mirrors ``BarlowBeestonLikelihood.gradient``:
        ``dchi2/dp_i = penalty_grad_i + sum_bins 2*(1 - O/E) * dE/dp_i``.

        The statistics residual ``2*(1 - O/E)`` is exactly ``dchi2/dE`` of the
        engine's ``poisson_chi2`` (matching ``chi2_and_grad``:1566). The
        expectation and its per-nuisance derivative arrive already contracted,
        as in the BB path."""
        theta = np.asarray(nuisance, dtype=float)
        nabla = 2.0 * (theta - self._nominal) / self._sigma ** 2   # penalty grad

        for i, dE in enumerate(diff_expectation.values()):
            for (exp_name, O), E, dEdx in zip(self.observation.items(),
                                              expectation.values(),
                                              dE.values()):
                O = np.asarray(O, dtype=float)
                E = np.asarray(E, dtype=float)
                resid = 2.0 * (1.0 - O / np.maximum(E, 1e-9))
                nabla[i] += float(np.sum(resid * np.asarray(dEdx, dtype=float)))
        return nabla
