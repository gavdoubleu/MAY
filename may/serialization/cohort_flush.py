"""Shared bookkeeping for mid-timeline property flushes to the canonical
HDF5 file (see docs/adr/0003, docs/adr/0004, docs/adr/0005).

One CohortFlushTracker per cohort (population, geo_units, or one per
venue_type) records the cohort's row-id sequence at first flush and
re-verifies it on every later flush call and at final export, so a cohort
that drifted (members added, removed or reordered) after being flushed is
caught rather than silently misaligning columns. Pure bookkeeping: no HDF5
I/O here, that stays in WorldSerializer.
"""


class CohortFlushTracker:
    """Tracks flushed property names and the recorded row-id sequence for
    one cohort. `label` identifies the cohort in error messages (e.g.
    'population', 'geo_units', or a venue_type string)."""

    def __init__(self, label):
        self.label = label
        self._flushed_names = set()
        self._recorded_cohort_ids = None
        self._flushed_subset_keys = None

    @property
    def flushed_names(self):
        return self._flushed_names

    def validate(self, property_names, configured_properties):
        """Raise ValueError if any of `property_names` aren't in
        `configured_properties`, or have already been flushed."""
        unknown = [name for name in property_names if name not in configured_properties]
        if unknown:
            raise ValueError(
                f"Cannot flush unknown {self.label} properties {unknown}: "
                f"not listed in the serialization config "
                f"({configured_properties})."
            )
        already_flushed = self._flushed_names.intersection(property_names)
        if already_flushed:
            raise ValueError(
                f"{self.label} properties flushed twice: {sorted(already_flushed)}."
            )

    def mark_flushed(self, property_names):
        self._flushed_names.update(property_names)

    def _drift_error(self, cohort_ids):
        return RuntimeError(
            f"{self.label!r} cohort changed since its earlier "
            f"flush_properties step ({len(self._recorded_cohort_ids)} "
            f"members then, {len(cohort_ids)} now): flushed columns "
            f"would not share a row order."
        )

    def record_flush(self, cohort_ids):
        """Record `cohort_ids` for a real flush_properties call, raising if
        they differ from an earlier flush's ids."""
        if self._recorded_cohort_ids is not None and cohort_ids != self._recorded_cohort_ids:
            raise self._drift_error(cohort_ids)
        self._recorded_cohort_ids = cohort_ids

    def check_drift(self, cohort_ids):
        """Raise if `cohort_ids` differs from the ids recorded by an
        earlier real flush_properties call (final-export re-verify). No-op
        if no flush ever happened: nothing was flushed, so there is
        nothing to re-verify against."""
        if self._recorded_cohort_ids is None:
            return
        if cohort_ids != self._recorded_cohort_ids:
            raise self._drift_error(cohort_ids)

    def record_subset_keys(self, subset_keys):
        """Record the `(Venue.id, subset_index)` keys staged by a real
        subsets flush, raising if a second subsets-flush for this cohort
        sends a different set."""
        subset_keys = frozenset(subset_keys)
        if (
            self._flushed_subset_keys is not None
            and subset_keys != self._flushed_subset_keys
        ):
            raise RuntimeError(
                f"{self.label!r} cohort subsets flushed twice with "
                f"different subset sets: recorded "
                f"{sorted(self._flushed_subset_keys)}, now "
                f"{sorted(subset_keys)}."
            )
        self._flushed_subset_keys = subset_keys

    def check_subset_drift(self, live_keys, all_keys):
        """Raise if this cohort's subsets were flushed but `live_keys`
        (non-gutted subsets on a venue of this cohort) is non-empty, or a
        recorded subset is missing from `all_keys` (current husks + live
        subsets). No-op if this cohort's subsets were never flushed."""
        if self._flushed_subset_keys is None:
            return
        if live_keys:
            raise RuntimeError(
                f"{self.label!r} cohort's subsets were flushed, but live "
                f"(non-gutted) subsets still exist for this venue type: "
                f"{sorted(live_keys)}. The cohort is closed to new "
                f"subsets once flushed."
            )
        missing = self._flushed_subset_keys - frozenset(all_keys)
        if missing:
            raise RuntimeError(
                f"{self.label!r} cohort's flushed subsets are missing at "
                f"export: {sorted(missing)}."
            )
