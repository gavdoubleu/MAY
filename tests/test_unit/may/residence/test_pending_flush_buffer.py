"""The pending-flush buffer in isolation: what a batch is, and what counts.

The integration tests drive the buffer through residence allocation, which
only ever offers one venue type and never offers an empty household. Both
of those are accidents of the caller, not of the buffer — its docstring
promises an interface callable from a parallel result drain — so the
counters and the bound are pinned here without a world.
"""

import logging

from may.residence.pending_flush_buffer import PendingFlushBuffer


class RecordingFlusher:
    """Every `(venue_type, subset_keys)` handed to staging, in order."""

    def __init__(self):
        self.calls = []

    def __call__(self, venue_type, subset_keys):
        self.calls.append((venue_type, list(subset_keys)))


def build_buffer(bound=1):
    flusher = RecordingFlusher()
    return PendingFlushBuffer(flusher, logging.getLogger("test_flush"),
                              bound=bound), flusher


class TestBatchesAreCountedNotFlusherInvocations:

    def test_one_batch_spanning_two_venue_types_is_one_call(self):
        """`calls` is the cadence of the buffer, and a sweep fits per-call
        cost against per-member cost from it. A batch that happens to span
        two venue types costs one deferral, not two."""
        buffer, flusher = build_buffer(bound=10)

        buffer.offer("household", [(1, 0)], member_count=4)
        buffer.offer("care_home", [(2, 0)], member_count=6)

        assert len(flusher.calls) == 2, "one invocation per venue type"
        assert buffer.calls == 1
        assert buffer.members == 10

    def test_max_batch_members_is_a_batch_not_a_cross_type_sum(self):
        """The largest batch is what sizes the deferral, so it has to be a
        quantity that was actually held at once."""
        buffer, _ = build_buffer(bound=10)

        buffer.offer("household", [(1, 0)], member_count=4)
        buffer.offer("care_home", [(2, 0)], member_count=6)
        buffer.offer("household", [(3, 0)], member_count=12)

        assert buffer.calls == 2
        assert buffer.max_batch_members == 12

    def test_per_call_debug_lines_match_the_summary_count(self, caplog):
        """The summary is the per-call lines aggregated; a run whose two
        never agree cannot be read."""
        flusher = RecordingFlusher()
        logger = logging.getLogger("test_flush_debug")
        buffer = PendingFlushBuffer(flusher, logger, bound=10)

        with caplog.at_level(logging.DEBUG, logger="test_flush_debug"):
            buffer.offer("household", [(1, 0)], member_count=4)
            buffer.offer("care_home", [(2, 0)], member_count=6)

        per_call = [message for message in caplog.messages
                    if message.startswith("[FLUSH] call=")]
        assert len(per_call) == buffer.calls == 1


class TestABoundOfOneStagesEveryOffer:

    def test_a_household_with_no_members_still_stages(self):
        """A bound of 1 means "stage at every unit that created a
        household". An empty household is still a created household, and
        deferring it holds keys the caller was told were gone."""
        buffer, flusher = build_buffer(bound=1)

        buffer.offer("household", [(1, 0)], member_count=0)

        assert flusher.calls == [("household", [(1, 0)])]
        assert buffer.calls == 1

    def test_an_offer_of_no_subsets_stages_nothing(self):
        """A unit that created no household is not an opportunity to
        flush, so it must not cut an empty batch."""
        buffer, flusher = build_buffer(bound=1)

        buffer.offer("household", [], member_count=0)

        assert flusher.calls == []
        assert buffer.calls == 0
