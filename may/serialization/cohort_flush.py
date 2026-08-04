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
        """Accumulate the `(Venue.id, subset_index)` keys staged by a real
        subsets flush. Subsets may be staged in several disjoint batches
        (one per geo unit, say), so keys add to the recorded set rather
        than replacing it; an overlapping batch raises."""
        subset_keys = frozenset(subset_keys)
        if self._flushed_subset_keys is None:
            self._flushed_subset_keys = set(subset_keys)
            return
        overlap = subset_keys & self._flushed_subset_keys
        if overlap:
            raise RuntimeError(
                f"{self.label!r} cohort subsets staged twice: "
                f"{sorted(overlap)} already staged by an earlier batch."
            )
        # A mutable set grown in place: the accumulated keys are the whole
        # run's, so rebuilding them per batch would cost O(total staged) per
        # flush. The incoming batch stays a frozenset — the caller's, not ours.
        self._flushed_subset_keys.update(subset_keys)

    @property
    def subsets_flushed(self):
        """True if a real subsets flush has happened for this cohort."""
        return self._flushed_subset_keys is not None

    def classify_subset_batch(self, subset_keys):
        """Classify a batch against what this cohort has already staged:
        'repeat' if every key was staged before (safe to skip), 'new' if
        none were. A partial overlap raises — batches must be disjoint or
        identical, otherwise a re-flush would double-stage some subsets."""
        subset_keys = frozenset(subset_keys)
        if self._flushed_subset_keys is None:
            return 'new'
        overlap = subset_keys & self._flushed_subset_keys
        if not overlap:
            return 'new'
        if overlap == subset_keys:
            return 'repeat'
        raise RuntimeError(
            f"{self.label!r} cohort subsets batch partially overlaps an "
            f"earlier batch: {sorted(overlap)} already staged, "
            f"{sorted(subset_keys - overlap)} not. Batches must be "
            f"disjoint or identical."
        )

    def check_subset_drift(self, live_keys, all_keys):
        """Raise if this cohort's subsets were flushed but `live_keys`
        (non-gutted subsets on a venue of this cohort) is non-empty, or a
        recorded subset is missing from `all_keys` (current husks + live
        subsets). The recorded set is the union of every staged batch. No-op
        if this cohort's subsets were never flushed."""
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
