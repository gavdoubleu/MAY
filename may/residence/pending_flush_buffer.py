"""The pending-flush buffer: what sits between allocation and staging.

Allocation offers each completed geo unit's household subsets here; the
buffer holds them live while the process still fits its memory budget, then
cuts a `Flush batch` and calls the flusher. The budget is in megabytes of
measured resident set, because that is the quantity the deferral actually
spends and the one a machine actually runs out of (ADR-0008, amended).

It is one-sided: an offer is atomic, so a budget the world cannot fit under
flushes every unit and no more often. Bounding therefore spans "flush at
every opportunity" to "never", and nothing finer. A budget of zero is the
former, with no special case — no process fits in zero.

Measuring rather than counting is the amendment. Members were a stand-in for
bytes chosen because bytes then meant members multiplied by a guessed
constant; reading the resident set removes the guess, so the operator states
the budget the machine gives them instead of a member count they derived from
a sweep and cannot transfer to another machine.

Wrapping the flusher rather than accumulating in the loop keeps "cadence
cannot change the world" trivially true — the loop body offers every unit
every time regardless — and leaves the interface callable from a parallel
result drain, not only from a unit boundary.
"""

import os
import time
from typing import Callable, Dict, List, Optional, Tuple

_PAGE_SIZE_MB = os.sysconf('SC_PAGE_SIZE') / (1024 * 1024)


def resident_set_size_mb() -> float:
    """This process's resident set, in megabytes.

    The parent creates every household as it applies each unit's plan, so its
    own resident set is exactly what staging a batch gives back. Read from
    `/proc/self/statm`, which is a two-field parse of a page already in
    memory — cheap enough to read at every offer, which is what lets the
    budget be a live reading rather than a periodic sample.
    """
    with open('/proc/self/statm', 'r') as statm:
        resident_pages = int(statm.read().split()[1])
    return resident_pages * _PAGE_SIZE_MB


class PendingFlushBuffer:
    """Offers of complete subsets in, staging calls out, counters kept."""

    def __init__(self, flusher: Callable, logger, budget_mb: float = 0,
                 measure_resident_set_size_mb: Optional[Callable] = None):
        self._flusher = flusher
        self._logger = logger
        self._budget_mb = budget_mb
        self._measure = measure_resident_set_size_mb or resident_set_size_mb
        self._pending: Dict[str, List[Tuple[int, int]]] = {}
        self._pending_members = 0
        self.calls = 0
        self.members = 0
        self.seconds = 0.0
        self.max_batch_members = 0
        self.max_resident_set_size_mb = 0.0

    def offer(self, venue_type: str, subset_keys: List[Tuple[int, int]],
              member_count: int) -> None:
        """Hand over subsets whose membership is final."""
        if not subset_keys:
            return
        self._pending.setdefault(venue_type, []).extend(subset_keys)
        self._pending_members += member_count
        resident_set_mb = self._measure()
        self.max_resident_set_size_mb = max(self.max_resident_set_size_mb,
                                            resident_set_mb)
        if resident_set_mb > self._budget_mb:
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
            f"max_batch_members={self.max_batch_members} "
            f"budget_mb={self._budget_mb} "
            f"max_resident_set_size_mb={self.max_resident_set_size_mb:.1f}")
