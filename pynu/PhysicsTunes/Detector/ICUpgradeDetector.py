"""
IceCube Upgrade detector systematics for Pynu — no-op placeholder.

The ICUp data release does not include ice gradient files needed for
detector systematic corrections. This class is a clean no-op so that
PhysicsTunes routing works without crashing.

Detector nuisance params can be added later when gradient data becomes available.
"""

from ..PhysicsTunes import Tune


class ICUpgradeDetector(Tune):
    """IceCube Upgrade detector — no-op (no detector systematic files in data release)."""
    pass
