"""
IceCube DeepCore detector systematics for Pynu — Hypersurface (HS) version.

The 5 HS detector parameters (DOM efficiency, hole ice p0/p1, bulk ice
absorption/scattering) act on binned histograms, not on event-level weights.
Their correction is applied via ICDeepCore_Atm.apply_hs_correction().

These methods are no-ops that return ones (weights) and zeros (derivatives)
so the parameters stay in the nuisance vector for prior penalty terms without
modifying event-level weights.
"""

import numpy as np
from ..PhysicsTunes import Tune


class ICDeepCoreDetector(Tune):
    """
    IceCube DeepCore detector systematics (hypersurface no-ops).

    Each HS parameter returns np.ones for the weight and np.zeros for the
    derivative. The actual bin-level correction is applied externally via
    ICDeepCore_Atm.apply_hs_correction().
    """

    def dom_eff(self, experiment, x):
        return np.ones(experiment.NumberOfEvents)

    def diff_dom_eff(self, experiment, x):
        return np.zeros(experiment.NumberOfEvents)

    def hole_ice_p0(self, experiment, x):
        return np.ones(experiment.NumberOfEvents)

    def diff_hole_ice_p0(self, experiment, x):
        return np.zeros(experiment.NumberOfEvents)

    def hole_ice_p1(self, experiment, x):
        return np.ones(experiment.NumberOfEvents)

    def diff_hole_ice_p1(self, experiment, x):
        return np.zeros(experiment.NumberOfEvents)

    def bulk_ice_abs(self, experiment, x):
        return np.ones(experiment.NumberOfEvents)

    def diff_bulk_ice_abs(self, experiment, x):
        return np.zeros(experiment.NumberOfEvents)

    def bulk_ice_scatter(self, experiment, x):
        return np.ones(experiment.NumberOfEvents)

    def diff_bulk_ice_scatter(self, experiment, x):
        return np.zeros(experiment.NumberOfEvents)

    def muon_norm(self, experiment, x):
        """Muon normalization — no-op at event level (scaling in likelihood)."""
        return np.ones(experiment.NumberOfEvents)

    def diff_muon_norm(self, experiment, x):
        """Derivative of muon_norm — zero at event level."""
        return np.zeros(experiment.NumberOfEvents)
