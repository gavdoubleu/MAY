"""Behaviour tests for the thin process-pool `parallel_map` substrate.

Tests exercise the public `parallel_map` / `Slice` interface only, never pool
internals, so they survive re-implementation.
"""
import numpy as np

from may.parallel import parallel_map, parallel_map_in_order, GeoUnitSlice


# Read-only shared state a worker reads without it being passed per Slice; set
# by the parent at runtime, so a worker seeing its value proves fork-COW
# inheritance (a spawn/re-import would see the module-definition None instead).
_SHARED_TABLE = None

# Set by the look-ahead test to a manager list workers can append to.
_COMPLETED = None


def _lookup_shared(unit_slice, rng):
    """Worker that reads the parent-built module-global by index."""
    return int(_SHARED_TABLE[int(unit_slice.unit_codes[0])])


def _count_codes(unit_slice, rng):
    """Trivial deterministic worker: ignores rng, returns Slice size."""
    return len(unit_slice.unit_codes)


def _sleep_if_slow(unit_slice, rng):
    """Worker that stalls only on the Slice named 'SLOW'."""
    import time
    if unit_slice.unit_codes[0] == 'SLOW':
        time.sleep(2.0)
    return unit_slice.unit_codes[0]


def _draw(unit_slice, rng):
    """Worker whose output depends entirely on its injected RNG stream."""
    return rng.integers(0, 1_000_000, size=5).tolist()


def test_runs_fn_per_slice_preserving_input_order():
    # More Slices than workers proves the bounded pool drains a queue rather
    # than needing one worker per Slice.
    slices = [GeoUnitSlice(['E1']),
              GeoUnitSlice(['E2', 'E3']),
              GeoUnitSlice(['E4', 'E5', 'E6']),
              GeoUnitSlice(['E7', 'E8', 'E9', 'E10'])]

    results = parallel_map(_count_codes, slices, workers=2)

    assert results == [1, 2, 3, 4]


def test_tier1_results_identical_across_worker_count():
    # Tier-1: a Slice's RNG stream is fixed by its stable identity, so the
    # world produced is bit-for-bit identical regardless of how many workers
    # the pool runs — the mechanical-correctness gate for the apply plumbing.
    slices = [GeoUnitSlice(['E1', 'E2']),
              GeoUnitSlice(['E3']),
              GeoUnitSlice(['E4', 'E5', 'E6']),
              GeoUnitSlice(['E7'])]

    single = parallel_map(_draw, slices, workers=1)
    many = parallel_map(_draw, slices, workers=4)

    assert single == many


def test_stream_seeded_by_identity_not_position():
    # ADR-0002: a Slice's stream derives from its members, not its index. So a
    # given Slice yields the same draw whether it is dispatched first or last.
    slices = [GeoUnitSlice(['E1']),
              GeoUnitSlice(['E2']),
              GeoUnitSlice(['E3'])]

    forward = parallel_map(_draw, slices, workers=2)
    reversed_result = parallel_map(_draw, list(reversed(slices)), workers=2)

    # E3 is last forward, first reversed — its draw must not change.
    assert forward[2] == reversed_result[0]
    assert forward[0] == reversed_result[2]


def test_population_slice_digest_is_order_independent():
    # A PopulationSlice is a *set* of stable person-ids; its seed identity must
    # not depend on the order the ids were listed in.
    from may.parallel import PopulationSlice
    a = PopulationSlice([7, 3, 42])
    b = PopulationSlice([42, 7, 3])
    c = PopulationSlice([7, 3, 99])

    assert a.digest == b.digest
    assert a.digest != c.digest


def test_worker_count_defaults_to_slurm_allocation(monkeypatch):
    from may.parallel import resolve_worker_count
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '8')
    # Explicit arg always wins over the environment.
    assert resolve_worker_count(3) == 3
    # Absent arg falls back to the SLURM CPU allocation.
    assert resolve_worker_count(None) == 8


def test_worker_count_falls_back_to_cpu_count_without_slurm(monkeypatch):
    import os
    from may.parallel import resolve_worker_count
    monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
    assert resolve_worker_count(None) == max(1, os.cpu_count())


def test_readonly_shared_state_rides_fork_cow():
    # A large read-only array built once in the parent must be visible to every
    # worker via copy-on-write, not re-pickled into each Slice payload.
    global _SHARED_TABLE
    _SHARED_TABLE = np.arange(1000) * 10  # index i -> i*10

    slices = [GeoUnitSlice([str(index)]) for index in (5, 17, 42, 3)]
    results = parallel_map(_lookup_shared, slices, workers=2)

    assert results == [50, 170, 420, 30]


_PID_MARKER = None


def _return_pid(unit_slice, rng):
    import os
    return os.getpid()


def test_single_worker_runs_in_process_no_fork():
    # workers=1 must run in the calling process (no child pids) so the large
    # existing suites that route through parallel_map stay cheap and fork-safe.
    import os
    slices = [GeoUnitSlice(['E1']), GeoUnitSlice(['E2']), GeoUnitSlice(['E3'])]
    pids = parallel_map(_return_pid, slices, workers=1)
    assert pids == [os.getpid(), os.getpid(), os.getpid()]


def test_worker_count_capped_at_slice_count():
    # No point forking more workers than Slices; one Slice runs in-process
    # however many workers were requested.
    import os
    pids = parallel_map(_return_pid, [GeoUnitSlice(['E1'])], workers=8)
    assert pids == [os.getpid()]


def test_in_order_yields_every_slice_tagged_with_its_index():
    # `parallel_map` returns only once every Slice is done, holding every
    # result live (ADR-0003 wants a streamed per-Slice apply). The streaming
    # sibling hands each result over one at a time, tagged with the Slice's
    # input index so a caller applying in canonical order knows what it has.
    slices = [GeoUnitSlice(['E1']),
              GeoUnitSlice(['E2', 'E3']),
              GeoUnitSlice(['E4', 'E5', 'E6']),
              GeoUnitSlice(['E7', 'E8', 'E9', 'E10'])]

    yielded = list(parallel_map_in_order(_count_codes, slices, workers=2))

    assert yielded == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_in_order_seeds_each_slice_exactly_as_parallel_map():
    # Streaming is a dispatch change, not a seeding one: the streamed results
    # must reproduce `parallel_map`'s list exactly, at any worker count.
    # Without this the restructure would move the world.
    slices = [GeoUnitSlice(['E1', 'E2']),
              GeoUnitSlice(['E3']),
              GeoUnitSlice(['E4', 'E5', 'E6']),
              GeoUnitSlice(['E7'])]

    ordered = parallel_map(_draw, slices, workers=1)
    for worker_count in (1, 4):
        streamed = dict(parallel_map_in_order(_draw, slices,
                                              workers=worker_count))
        assert [streamed[index] for index in range(len(slices))] == ordered


def test_in_order_can_be_forced_to_fork_at_one_worker():
    # Phase 5's geo loop mutates a world inside the worker and hands the parent
    # a plan to replay. In-process that double-applies — the parent's own world
    # would be mutated *and* replayed on top — so the caller must be able to
    # demand real isolation whatever the worker count. It is also what makes
    # the W=1-vs-W=N comparison a comparison of two forked runs.
    import os

    slices = [GeoUnitSlice(['E1']), GeoUnitSlice(['E2'])]

    pids = dict(parallel_map_in_order(_return_pid, slices, workers=1,
                                      force_fork=True))

    assert os.getpid() not in pids.values()


def test_in_order_hands_over_a_finished_slice_while_later_ones_run():
    # The point of the entry point: a caller must be able to apply a finished
    # Slice while a long-running sibling is still in flight. Under
    # `parallel_map`'s all-or-nothing return nothing at all would be available
    # until the slow one completed. Ordering does not cost this — only Slices
    # *behind* the one being handed over are waited on, never ones ahead.
    import time

    slices = [GeoUnitSlice(['E1']), GeoUnitSlice(['E2']), GeoUnitSlice(['SLOW'])]

    started = time.monotonic()
    stream = parallel_map_in_order(_sleep_if_slow, slices, workers=3)
    first_index, first_result = next(stream)
    elapsed = time.monotonic() - started

    assert (first_index, first_result) == (0, 'E1')
    assert elapsed < 2.0
    list(stream)  # drain so the pool shuts down cleanly


def test_in_order_yields_slices_in_input_order_however_they_finish():
    # Phase 6. The as-completed stream left the caller to reorder, and a caller
    # that must apply in input order has nowhere to hold an early result but an
    # unbounded buffer of its own. Yielding in input order moves that buffer
    # inside the pool, where dispatch can bound it. The slow Slice is first, so
    # a stream that yielded on completion would put it last.
    from may.parallel import parallel_map_in_order

    slices = [GeoUnitSlice(['SLOW']), GeoUnitSlice(['E2']), GeoUnitSlice(['E3'])]

    yielded = list(parallel_map_in_order(_sleep_if_slow, slices, workers=3))

    assert yielded == [(0, 'SLOW'), (1, 'E2'), (2, 'E3')]


def _record_completion(unit_slice, rng):
    """Worker that reports when it *finished*. A finished Slice whose turn to
    be applied has not come is exactly a result the parent is holding, so
    counting completions counts the buffer."""
    import time
    if unit_slice.unit_codes[0] == 'SLOW':
        time.sleep(2.0)
    _COMPLETED.append(unit_slice.unit_codes[0])
    return unit_slice.unit_codes[0]


def test_max_tasks_per_child_recycles_the_worker():
    # A worker retired after N tasks gets a fresh pid for its (N+1)th task —
    # this is the mechanism that bounds a forked worker's own resident-set
    # growth, distinct from the parent-side flush/look-ahead bounds.
    slices = [GeoUnitSlice([f'E{index}']) for index in range(4)]

    pids = [pid for _, pid in parallel_map_in_order(
        _return_pid, slices, workers=1, force_fork=True,
        max_tasks_per_child=2)]

    assert pids[0] == pids[1]
    assert pids[2] == pids[3]
    assert pids[0] != pids[2]


def test_max_tasks_per_child_does_not_change_results():
    # Recycling is a memory concern only; a Slice's stream is seeded from its
    # own identity (ADR-0002), so results must not depend on whether, or how
    # often, the worker running it gets retired and reforked.
    slices = [GeoUnitSlice(['E1', 'E2']),
              GeoUnitSlice(['E3']),
              GeoUnitSlice(['E4', 'E5', 'E6']),
              GeoUnitSlice(['E7'])]

    baseline = parallel_map(_draw, slices, workers=4)
    for max_tasks_per_child in (1, 2):
        recycled = parallel_map(_draw, slices, workers=4,
                                max_tasks_per_child=max_tasks_per_child)
        assert recycled == baseline


def test_in_order_dispatches_no_further_ahead_than_the_look_ahead():
    # Phase 6. `imap_unordered` drained its input eagerly in a feeder thread,
    # so "at most pool-size Slices in flight" was never true: behind one slow
    # Slice every later Slice ran to completion and its result waited in
    # memory to be applied (ADR-0010's accepted gap). Throttling dispatch is
    # what makes the bound real, and completions are how it shows — a Slice
    # that has finished but cannot be applied yet *is* the buffer.
    import multiprocessing as mp

    global _COMPLETED
    manager = mp.get_context('fork').Manager()
    _COMPLETED = manager.list()

    slices = [GeoUnitSlice(['SLOW'])] + [GeoUnitSlice([f'E{index}'])
                                         for index in range(40)]
    stream = parallel_map_in_order(_record_completion, slices, workers=4,
                                   look_ahead=3)
    try:
        assert next(stream) == (0, 'SLOW')
        # The head plus at most the two the look-ahead allowed past it.
        assert len(_COMPLETED) <= 3, list(_COMPLETED)
    finally:
        stream.close()
        manager.shutdown()
