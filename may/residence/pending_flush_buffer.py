"""The pending-flush buffer: what sits between allocation and staging.

Allocation offers each completed geo unit's household subsets here; the
buffer holds them live until they total the bound, then cuts a `Flush batch`
and calls the flusher. The bound is in members because that is what the
deferral holds in memory and what the write costs (ADR-0008).

It is one-sided: an offer is atomic, so a bound smaller than a unit's
membership flushes every unit and no more often. Bounding therefore spans
"flush at every opportunity" to "never", and nothing finer.

Wrapping the flusher rather than accumulating in the loop keeps "cadence
cannot change the world" trivially true — the loop body offers every unit
every time regardless — and leaves the interface callable from a parallel
result drain, not only from a unit boundary.
"""

import time
from typing import Callable, Dict, List, Tuple


class PendingFlushBuffer:
    """Offers of complete subsets in, staging calls out, counters kept."""

    def __init__(self, flusher: Callable, logger, bound: int = 1):
        self._flusher = flusher
        self._logger = logger
        self._bound = bound
        self._pending: Dict[str, List[Tuple[int, int]]] = {}
        self._pending_members = 0
        self.calls = 0
        self.members = 0
        self.seconds = 0.0
        self.max_batch_members = 0

    def offer(self, venue_type: str, subset_keys: List[Tuple[int, int]],
              member_count: int) -> None:
        """Hand over subsets whose membership is final."""
        if not subset_keys:
            return
        self._pending.setdefault(venue_type, []).extend(subset_keys)
        self._pending_members += member_count
        # Held keys are a deferral too, not only held members: a non-empty
        # offer is an opportunity to flush even when the households it
        # carries are empty, so a bound of 1 stages every one of them.
        if max(self._pending_members, 1) >= self._bound:
            self._cut_batch()

    def drain(self) -> None:
        """Stage whatever is still held, bound or no bound."""
        self._cut_batch()

    def _cut_batch(self) -> None:
        """Stage everything held, in one call per venue type.

        Counted as one batch however many venue types it spans: the batch
        is what the deferral held and what the cadence is measured in, and
        a sweep fitting per-call against per-member cost needs the two to
        mean different things.
        """
        if not self._pending:
            return
        batch_members = self._pending_members
        started = time.perf_counter()
        for venue_type, subset_keys in self._pending.items():
            self._flusher(venue_type, subset_keys)
        elapsed = time.perf_counter() - started

        self._pending = {}
        self._pending_members = 0
        self.calls += 1
        self.members += batch_members
        self.seconds += elapsed
        self.max_batch_members = max(self.max_batch_members, batch_members)
        self._logger.debug(
            f"[FLUSH] call={self.calls} members={batch_members} "
            f"seconds={elapsed:.3f}")

    def log_summary(self) -> None:
        """One line per run, in the shape `collect_results.py` parses."""
        self._logger.info(
            f"[FLUSH] calls={self.calls} members={self.members} "
            f"seconds={self.seconds:.3f} "
            f"max_batch_members={self.max_batch_members}")
