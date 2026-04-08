"""
ORCA-Full detector systematics — no-op placeholder.

The ORCA-Full response is built from digitized matrices with no
detector-level systematic gradients. Only flux systematics are used.
"""

from ..PhysicsTunes import Tune


class ORCAFullDetector(Tune):
    """ORCA-Full detector — no-op (response matrices have no detector systematics)."""
    pass
