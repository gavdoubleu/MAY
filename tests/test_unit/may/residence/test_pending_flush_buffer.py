"""The pending-flush buffer in isolation: when it cuts, and what counts.

The integration tests drive the buffer through residence allocation, which
only ever offers one venue type and never offers an empty household. Both
of those are accidents of the caller, not of the buffer — its docstring
promises an interface callable from a parallel result drain — so the
counters and the budget are pinned here without a world.

Isolation is also the only place the budget's middle is testable. Cadence is
now decided by comparing a real resident set against the configured budget,
and a real resident set is not a quantity a test can state; injecting the
reading is what makes "held here, staged there" something to assert rather
than hope for. That seam is a system boundary — `/proc/self/statm` — not a
collaborator.
"""

import logging

from may.residence.pending_flush_buffer import PendingFlushBuffer


class RecordingFlusher:
    """Every `(venue_type, subset_keys)` handed to staging, in order."""

    def __init__(self):
        self.calls = []

    def __call__(self, venue_type, subset_keys):
        self.calls.append((venue_type, list(subset_keys)))


class FakeMemoryPressure:
    """A resident-set reading the test moves when it wants a flush."""

    def __init__(self, resident_set_size_mb=400.0):
        self.resident_set_size_mb = resident_set_size_mb

    def __call__(self):
        return self.resident_set_size_mb


def build_buffer(budget_mb=0, pressure=None):
    flusher = RecordingFlusher()
    buffer = PendingFlushBuffer(flusher, logging.getLogger("test_flush"),
                                budget_mb=budget_mb,
                                measure_resident_set_size_mb=pressure or FakeMemoryPressure())
    return buffer, flusher


class TestTheBudgetDecidesWhenToStage:

    def test_membership_is_held_while_the_process_fits_its_budget(self):
        """The point of the budget: a machine with room to spare does not pay
        for writes it does not need."""
        buffer, flusher = build_buffer(budget_mb=1000)

        buffer.offer("household", [(1, 0)], member_count=3)
        buffer.offer("household", [(2, 0)], member_count=4)

        assert flusher.calls == []

    def test_exceeding_the_budget_stages_everything_held(self):
        pressure = FakeMemoryPressure()
        buffer, flusher = build_buffer(budget_mb=1000, pressure=pressure)

        buffer.offer("household", [(1, 0)], member_count=3)
        pressure.resident_set_size_mb = 1200.0
        buffer.offer("household", [(2, 0)], member_count=4)

        assert flusher.calls == [("household", [(1, 0), (2, 0)])]
        assert buffer.members == 7

    def test_a_budget_of_zero_stages_at_every_opportunity(self):
        """No process fits in zero megabytes, so the "flush as often as
        possible" end of the dial needs no special case — it is the budget
        nothing can satisfy. This is the mode the deleted
        `pending_flush_member_bound: 1` used to name.
        """
        buffer, flusher = build_buffer(budget_mb=0)

        buffer.offer("household", [(1, 0)], member_count=3)
        buffer.offer("household", [(2, 0)], member_count=4)

        assert flusher.calls == [("household", [(1, 0)]),
                                 ("household", [(2, 0)])]

    def test_a_batch_never_splits_an_offer(self):
        """The budget is one-sided: it can only make flushes rarer than one
        per offer, never finer. Residence allocation offers a whole loop unit
        at a time because the excess, overflow and promotion steps all add
        people to households earlier steps of the same unit created, so
        crossing the budget mid-unit cannot cut into it.
        """
        pressure = FakeMemoryPressure()
        buffer, flusher = build_buffer(budget_mb=1000, pressure=pressure)

        first_unit = [(1, 0), (2, 0), (3, 0)]
        second_unit = [(4, 0), (5, 0)]
        buffer.offer("household", first_unit, member_count=9)
        pressure.resident_set_size_mb = 1200.0
        buffer.offer("household", second_unit, member_count=6)

        assert flusher.calls == [("household", first_unit + second_unit)]

    def test_the_residue_is_staged_on_drain_however_far_under_budget(self):
        """A budget that never binds still stages everything at end of
        allocation, so the end state does not depend on the value."""
        buffer, flusher = build_buffer(budget_mb=10 ** 9)

        buffer.offer("household", [(1, 0)], member_count=3)
        buffer.offer("household", [(2, 0)], member_count=4)
        assert flusher.calls == []

        buffer.drain()

        assert flusher.calls == [("household", [(1, 0), (2, 0)])]
        assert buffer.calls == 1


class TestBatchesAreCountedNotFlusherInvocations:

    def test_one_batch_spanning_two_venue_types_is_one_call(self):
        """`calls` is the cadence of the buffer, and a sweep fits per-call
        cost against per-member cost from it. A batch that happens to span
        two venue types costs one deferral, not two."""
        pressure = FakeMemoryPressure()
        buffer, flusher = build_buffer(budget_mb=1000, pressure=pressure)

        buffer.offer("household", [(1, 0)], member_count=4)
        pressure.resident_set_size_mb = 1200.0
        buffer.offer("care_home", [(2, 0)], member_count=6)

        assert len(flusher.calls) == 2, "one invocation per venue type"
        assert buffer.calls == 1
        assert buffer.members == 10

    def test_max_batch_members_is_a_batch_not_a_cross_type_sum(self):
        """The largest batch is what sizes the deferral, so it has to be a
        quantity that was actually held at once."""
        pressure = FakeMemoryPressure()
        buffer, _ = build_buffer(budget_mb=1000, pressure=pressure)

        buffer.offer("household", [(1, 0)], member_count=4)
        pressure.resident_set_size_mb = 1200.0
        buffer.offer("care_home", [(2, 0)], member_count=6)
        pressure.resident_set_size_mb = 400.0
        buffer.offer("household", [(3, 0)], member_count=12)
        pressure.resident_set_size_mb = 1200.0
        buffer.offer("household", [(4, 0)], member_count=0)

        assert buffer.calls == 2
        assert buffer.max_batch_members == 12

    def test_per_call_debug_lines_match_the_summary_count(self, caplog):
        """The summary is the per-call lines aggregated; a run whose two
        never agree cannot be read."""
        flusher = RecordingFlusher()
        logger = logging.getLogger("test_flush_debug")
        pressure = FakeMemoryPressure()
        buffer = PendingFlushBuffer(flusher, logger, budget_mb=1000,
                                    measure_resident_set_size_mb=pressure)

        with caplog.at_level(logging.DEBUG, logger="test_flush_debug"):
            buffer.offer("household", [(1, 0)], member_count=4)
            pressure.resident_set_size_mb = 1200.0
            buffer.offer("care_home", [(2, 0)], member_count=6)

        per_call = [message for message in caplog.messages
                    if message.startswith("[FLUSH] call=")]
        assert len(per_call) == buffer.calls == 1

    def test_the_peak_reading_is_reported_for_a_run(self):
        """A run's headline memory number comes from the same reading that
        drove its cadence, so a sweep can tell "the budget was never
        approached" from "the budget was ridden all the way"."""
        pressure = FakeMemoryPressure()
        buffer, _ = build_buffer(budget_mb=10 ** 9, pressure=pressure)

        buffer.offer("household", [(1, 0)], member_count=3)
        pressure.resident_set_size_mb = 900.0
        buffer.offer("household", [(2, 0)], member_count=4)
        pressure.resident_set_size_mb = 700.0
        buffer.offer("household", [(3, 0)], member_count=5)

        assert buffer.max_resident_set_size_mb == 900.0


class TestABudgetOfZeroStagesEveryOffer:

    def test_a_household_with_no_members_still_stages(self):
        """A budget of zero means "stage at every unit that created a
        household". An empty household is still a created household, and
        deferring it holds keys the caller was told were gone."""
        buffer, flusher = build_buffer(budget_mb=0)

        buffer.offer("household", [(1, 0)], member_count=0)

        assert flusher.calls == [("household", [(1, 0)])]
        assert buffer.calls == 1

    def test_an_offer_of_no_subsets_stages_nothing(self):
        """A unit that created no household is not an opportunity to
        flush, so it must not cut an empty batch."""
        buffer, flusher = build_buffer(budget_mb=0)

        buffer.offer("household", [], member_count=0)

        assert flusher.calls == []
        assert buffer.calls == 0


class TestTheReadingComesFromTheProcess:

    def test_the_default_reader_reports_this_process(self):
        """The injected reader is a test seam; the real one has to be the
        parent's own resident set, since that is the only thing staging a
        batch can shrink."""
        from may.residence.pending_flush_buffer import resident_set_size_mb

        assert resident_set_size_mb() > 1.0
