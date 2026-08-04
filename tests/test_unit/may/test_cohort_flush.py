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

    def test_disjoint_batches_accumulate(self):
        tracker = CohortFlushTracker("household")
        tracker.record_subset_keys({(1, 0), (1, 1)})
        tracker.record_subset_keys({(2, 0)})

        tracker.check_subset_drift(
            live_keys=set(), all_keys={(1, 0), (1, 1), (2, 0)}
        )  # no raise: the union is the recorded set

    def test_overlapping_batch_raises(self):
        tracker = CohortFlushTracker("household")
        tracker.record_subset_keys({(1, 0), (1, 1)})

        with pytest.raises(RuntimeError, match="already staged"):
            tracker.record_subset_keys({(1, 0), (2, 0)})

    def test_recording_a_batch_costs_the_batch_not_the_run(self):
        """The accumulated set grows in place.

        Held as a frozenset it could only grow by rebinding to a fresh copy
        of every key staged so far, so each flush cost O(total staged) and
        briefly held two copies — inside the mode whose whole purpose is to
        free memory. Identity is the only observable: a size assertion
        passes either way.
        """
        tracker = CohortFlushTracker("household")
        tracker.record_subset_keys({(1, 0)})
        accumulated = tracker._flushed_subset_keys

        tracker.record_subset_keys({(2, 0)})

        assert tracker._flushed_subset_keys is accumulated
        assert accumulated == {(1, 0), (2, 0)}


class TestClassifySubsetBatch:

    def test_new_before_any_flush(self):
        tracker = CohortFlushTracker("household")
        assert tracker.classify_subset_batch({(1, 0)}) == 'new'

    def test_repeat_on_exact_match(self):
        tracker = CohortFlushTracker("household")
        tracker.record_subset_keys({(1, 0), (1, 1)})
        assert tracker.classify_subset_batch({(1, 0), (1, 1)}) == 'repeat'

    def test_new_on_disjoint_batch(self):
        tracker = CohortFlushTracker("household")
        tracker.record_subset_keys({(1, 0), (1, 1)})
        assert tracker.classify_subset_batch({(2, 0)}) == 'new'

    def test_partial_overlap_raises(self):
        tracker = CohortFlushTracker("household")
        tracker.record_subset_keys({(1, 0), (1, 1)})

        with pytest.raises(RuntimeError, match="partially overlaps"):
            tracker.classify_subset_batch({(1, 0), (2, 0)})


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
