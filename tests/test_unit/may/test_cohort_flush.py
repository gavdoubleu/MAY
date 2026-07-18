"""CohortFlushTracker: shared bookkeeping for mid-timeline property flushes
(see docs/adr/0003, docs/adr/0005). One tracker per cohort (population,
geo_units, or one per venue_type) records the cohort's row-id sequence at
first flush and re-verifies it on every later call, so a cohort that drifted
after being flushed is caught rather than silently misaligning columns.
"""

import pytest

from may.serialization.cohort_flush import CohortFlushTracker


class TestCheckDrift:

    def test_first_call_records_cohort_without_raising(self):
        tracker = CohortFlushTracker("population")
        tracker.check_drift([1, 2, 3])  # no raise

    def test_fully_removed_cohort_still_raises(self):
        """The cohort must be checked even when the 'current' ids are now
        empty (e.g. every member removed post-flush) — this is the bug
        class fixed in c6d6e4b for venues; the shared tracker must guarantee
        it for every flush target, not just venues."""
        tracker = CohortFlushTracker("household")
        tracker.check_drift([1, 2, 3])

        with pytest.raises(RuntimeError, match="household"):
            tracker.check_drift([])

    def test_repeated_call_with_same_cohort_does_not_raise(self):
        """Two flush steps for the same cohort (splitting properties across
        calls) must not spuriously conflict with each other."""
        tracker = CohortFlushTracker("geo_units")
        tracker.check_drift([1, 2, 3])
        tracker.check_drift([1, 2, 3])  # no raise


class TestValidate:

    def test_unknown_property_raises(self):
        tracker = CohortFlushTracker("population")
        with pytest.raises(ValueError, match="not_a_property"):
            tracker.validate(["not_a_property"], configured_properties=["ethnicity"])

    def test_valid_property_does_not_raise(self):
        tracker = CohortFlushTracker("population")
        tracker.validate(["ethnicity"], configured_properties=["ethnicity"])  # no raise

    def test_already_flushed_property_raises(self):
        tracker = CohortFlushTracker("population")
        tracker.mark_flushed(["ethnicity"])
        with pytest.raises(ValueError, match="twice"):
            tracker.validate(["ethnicity"], configured_properties=["ethnicity"])


class TestMarkFlushed:

    def test_flushed_names_reflects_marked_properties(self):
        tracker = CohortFlushTracker("population")
        assert tracker.flushed_names == set()
        tracker.mark_flushed(["ethnicity", "work_mode"])
        assert tracker.flushed_names == {"ethnicity", "work_mode"}
