"""Event-side energy-scale histogram-transfer operator.

The event side gets the SAME histogram-level operator as the binned engine,
applied post-binning to the binned expectation. The per-event weight-emulation
for the 4 ``energy_scale_sk*`` dials is RETIRED in the event path (kept dead
with a comment there); this module is the parity path.

The operator is transcribed VERBATIM from the binned engine so an event-side
binned expectation and the binned engine's ``_escale_migrate`` produce identical
output on identical input histograms:

  * geometry  — ``sk_binned_engine.py:812-821`` (``es_below`` / ``es_has_above``
    / ``es_has_below`` from ``sample_table``);
  * operator  — ``sk_binned_engine.py:1098-1114`` (``_escale_migrate``),
    ``N'(ie) = N(ie) + d*( N(ie-1)*[ie>0] - N(ie)*[ie<ne-1] )``, column-total
    conserving, ``var=True`` propagating the squared coefficients;
  * ordering  — ``sk_binned_engine.py:1442-1450``: applied per era to the
    pre-detector rate ``n_pre_e`` with ``delta = x - 1``.

Era masking follows the engine's ``_era_theta`` convention
(``sk_binned_engine.py:1137-1152``): each ``energy_scale_sk<N>`` dial governs the
bins of SK era N only (sk45 = eras >= 4). On the event side the era of each bin
is read from the baked ``sample_table`` era ownership (the same table the binned
geometry is built from), so a single (n_bins,) histogram is migrated with the
correct per-era delta on each era's bin subset.
"""
import numpy as np

# SK era tags, in the fixed production order (mirrors sk_binned_engine.ERA_TAGS).
ERA_TAGS = ("sk1", "sk2", "sk3", "sk45")


class EScaleHistogramOperator:
    """Reco-E adjacency transfer on a binned expectation histogram.

    Built once from the response's ``sample_table`` (the per-sample
    ``(offset, n_reco_E, n_reco_cz)`` geometry) plus a per-bin era ownership
    array. ``migrate(hist, deltas)`` applies the rate-conserving transfer with
    one delta per era; ``migrate(hist, deltas, var=True)`` propagates BB
    variances. ``delta_from_dials(theta_dict)`` maps dial values ``x`` to the
    ``delta = x - 1`` per-era vector.
    """

    def __init__(self, sample_table, n_bins, bin_era):
        """
        sample_table : {sample_id: (offset, n_reco_E, n_reco_cz)} — baked geometry.
        n_bins       : total number of reco bins (flat layout).
        bin_era      : (n_bins,) int array; era index (0..n_era-1) owning each bin.
                       sk45 bins carry the single lumped index for eras >= 4.
        """
        self.n_bins = int(n_bins)
        self.bin_era = np.asarray(bin_era, dtype=np.int64)
        self.n_era = int(self.bin_era.max()) + 1 if self.n_bins else 0

        # ---- reco-E adjacency geometry (transcribed sk_binned_engine.py:812-821)
        self.es_below = np.full(self.n_bins, -1, dtype=np.int64)
        self.es_has_above = np.zeros(self.n_bins)
        for s, (off, ne_, nz) in sample_table.items():
            off, ne_, nz = int(off), int(ne_), int(nz)
            for ie in range(ne_):
                base = off + ie * nz
                if ie > 0:
                    self.es_below[base:base + nz] = np.arange(base, base + nz) - nz
                if ie < ne_ - 1:
                    self.es_has_above[base:base + nz] = 1.0
        self.es_has_below = (self.es_below >= 0).astype(float)

        # per-era bin masks (1.0 on bins owned by era e), for the single-histogram
        # form. A bin only migrates under its own era's delta.
        self._era_mask = [(self.bin_era == e).astype(float)
                          for e in range(self.n_era)]

    # ---- dial -> delta ----
    @staticmethod
    def delta_from_dials(theta_dict, n_era):
        """Per-era ``delta = x - 1`` from ``{energy_scale_sk<tag>: x}`` (engine
        convention, sk_binned_engine.py:1446)."""
        return np.array([theta_dict[f"energy_scale_{ERA_TAGS[e]}"] - 1.0
                         for e in range(n_era)])

    # ---- the transfer operator ----
    def _migrate_uniform(self, N, d, var=False):
        """Single-delta migration of a full (n_bins,) array — the exact per-era
        body of ``_escale_migrate`` (sk_binned_engine.py:1104-1113), factored out
        so the era-masked path reuses it verbatim."""
        below = np.where(self.es_below >= 0, N[self.es_below], 0.0)   # N(ie-1)
        if not var:
            return N + d * (below * self.es_has_below - N * self.es_has_above)
        c_self = 1.0 - d * self.es_has_above
        c_below = d * self.es_has_below
        return c_self * c_self * N + c_below * c_below * below

    def migrate(self, hist, deltas, var=False):
        """Apply the reco-E energy-scale transfer to a single binned histogram
        ``hist`` (n_bins,), with one ``delta`` per era. Each era's delta is
        applied only to that era's bins (``_era_theta`` convention); the transfer
        stays inside a bin's own reco-cz column and conserves the per-column
        total. Returns a new array; ``hist`` is not mutated.

        Equivalence to the binned engine: on a per-era pre-detector array
        ``arr_e[e]`` (all bins belong to era e by construction) this reduces to
        ``_escale_migrate(arr_e, deltas)[e]`` bin-for-bin, because the era mask is
        all-ones there. On a mixed single histogram it applies the correct
        per-era delta to each era's bins."""
        deltas = np.asarray(deltas, dtype=float)
        hist = np.asarray(hist, dtype=float)
        out = hist.copy()
        for e in range(self.n_era):
            mig = self._migrate_uniform(hist, deltas[e], var=var)
            m = self._era_mask[e]
            # write the migrated value on era-e bins; the transfer of an era-e bin
            # only pulls from its own reco-E-below neighbour, which is same-era by
            # construction (adjacency never crosses a sample/era boundary).
            out = out * (1.0 - m) + mig * m
        return out

    def migrate_derivative(self, hist, era):
        """d(migrate)/d(delta_era) on a single histogram, evaluated on era's own
        bins (zero elsewhere). Linear in delta, so the derivative is the operator
        coefficient, independent of delta:
          dN'(ie)/ddelta = N(ie-1)*[ie>0]  -  N(ie)*[ie<ne-1]   (era's bins only).
        This is the analytic-gradient counterpart the design mandates ("the same
        linear operator applied to gradient histograms")."""
        hist = np.asarray(hist, dtype=float)
        below = np.where(self.es_below >= 0, hist[self.es_below], 0.0)
        d = below * self.es_has_below - hist * self.es_has_above
        return d * self._era_mask[era]

    def migrate_era_stack(self, arr_e, deltas, var=False):
        """Direct binned-engine form: migrate a (n_era, n_bins) per-era array,
        one delta per era. Byte-identical to ``SKBinnedEngine._escale_migrate``
        (this is that method's algebra, re-expressed via ``_migrate_uniform``)."""
        arr_e = np.asarray(arr_e, dtype=float)
        out = np.empty_like(arr_e)
        for e in range(arr_e.shape[0]):
            out[e] = self._migrate_uniform(arr_e[e], deltas[e], var=var)
        return out


def bin_era_from_sample_table(sample_table, sample_era, n_bins):
    """Build the (n_bins,) per-bin era ownership array from the baked
    ``sample_table`` geometry and a ``{sample_id: era_index}`` map. Bins not
    covered by any sample default to era 0 (they never migrate under any active
    energy-scale dial in production, where every reco bin belongs to a sample)."""
    bin_era = np.zeros(int(n_bins), dtype=np.int64)
    for s, (off, ne_, nz) in sample_table.items():
        e = int(sample_era[int(s)])
        off, ne_, nz = int(off), int(ne_), int(nz)
        bin_era[off:off + ne_ * nz] = e
    return bin_era
