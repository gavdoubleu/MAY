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
        tracker.record_flush([1, 2, 3])  # no raise

    def test_fully_removed_cohort_still_raises(self):
        """The cohort must be checked even when the 'current' ids are now
        empty (e.g. every member removed post-flush) — this is the bug
        class fixed in c6d6e4b for venues; the shared tracker must guarantee
        it for every flush target, not just venues."""
        tracker = CohortFlushTracker("household")
        tracker.record_flush([1, 2, 3])

        with pytest.raises(RuntimeError, match="household"):
            tracker.check_drift([])

    def test_repeated_call_with_same_cohort_does_not_raise(self):
        """Two flush steps for the same cohort (splitting properties across
        calls) must not spuriously conflict with each other."""
        tracker = CohortFlushTracker("geo_units")
        tracker.record_flush([1, 2, 3])
        tracker.record_flush([1, 2, 3])  # no raise

    def test_check_drift_without_any_flush_does_not_raise(self):
        """No flush_properties call ever happened, so a final-export
        check_drift() call has nothing to re-verify against — must be a
        no-op, not a spurious raise (see mission_statement_cohort_drift_reexport)."""
        tracker = CohortFlushTracker("population")
        tracker.check_drift([1, 2, 3])  # no raise

    def test_check_drift_without_flush_ignores_later_cohort_change(self):
        """Second export() call on a never-flushed serializer, with cohort
        membership changed in between — must not raise, since no flush ever
        happened to establish a baseline worth re-verifying."""
        tracker = CohortFlushTracker("geo_units")
        tracker.check_drift([1, 2, 3])
        tracker.check_drift([1, 2])  # no raise: never flushed

    def test_check_drift_after_real_flush_still_raises_on_drift(self):
        tracker = CohortFlushTracker("population")
        tracker.record_flush([1, 2, 3])
        with pytest.raises(RuntimeError, match="population"):
            tracker.check_drift([1, 2])


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


class TestRecordSubsetKeys:

    def test_first_call_records_without_raising(self):
        tracker = CohortFlushTracker("household")
        tracker.record_subset_keys({(1, 0), (1, 1)})  # no raise

    def test_second_call_with_different_keys_raises(self):
        tracker = CohortFlushTracker("household")
        tracker.record_subset_keys({(1, 0), (1, 1)})

        with pytest.raises(RuntimeError, match="household"):
            tracker.record_subset_keys({(1, 0)})

    def test_repeated_call_with_same_keys_does_not_raise(self):
        tracker = CohortFlushTracker("household")
        tracker.record_subset_keys({(1, 0), (1, 1)})
        tracker.record_subset_keys({(1, 0), (1, 1)})  # no raise


class TestSubsetKeysAlreadyFlushed:

    def test_false_before_any_flush(self):
        tracker = CohortFlushTracker("household")
        assert tracker.subset_keys_already_flushed({(1, 0)}) is False

    def test_true_on_exact_match(self):
        tracker = CohortFlushTracker("household")
        tracker.record_subset_keys({(1, 0), (1, 1)})
        assert tracker.subset_keys_already_flushed({(1, 0), (1, 1)}) is True

    def test_false_on_different_set(self):
        tracker = CohortFlushTracker("household")
        tracker.record_subset_keys({(1, 0), (1, 1)})
        assert tracker.subset_keys_already_flushed({(1, 0)}) is False


class TestCheckSubsetDrift:

    def test_without_any_subset_flush_does_not_raise(self):
        tracker = CohortFlushTracker("household")
        tracker.check_subset_drift(live_keys=set(), all_keys={(1, 0)})  # no raise

    def test_after_flush_with_no_live_subsets_and_husks_intact_does_not_raise(self):
        tracker = CohortFlushTracker("household")
        tracker.record_subset_keys({(1, 0), (1, 1)})

        tracker.check_subset_drift(live_keys=set(), all_keys={(1, 0), (1, 1)})  # no raise

    def test_live_subset_on_flushed_cohort_raises(self):
        tracker = CohortFlushTracker("household")
        tracker.record_subset_keys({(1, 0)})

        with pytest.raises(RuntimeError, match="household"):
            tracker.check_subset_drift(live_keys={(2, 0)}, all_keys={(1, 0), (2, 0)})

    def test_missing_husk_raises(self):
        tracker = CohortFlushTracker("household")
        tracker.record_subset_keys({(1, 0), (1, 1)})

        with pytest.raises(RuntimeError, match="household"):
            tracker.check_subset_drift(live_keys=set(), all_keys={(1, 0)})
