"""Behaviour tests for the thin process-pool `parallel_map` substrate.

Tests exercise the public `parallel_map` / `Slice` interface only, never pool
internals, so they survive re-implementation.
"""
import numpy as np

from may.parallel import parallel_map, GeoUnitSlice


# Read-only shared state a worker reads without it being passed per Slice; set
# by the parent at runtime, so a worker seeing its value proves fork-COW
# inheritance (a spawn/re-import would see the module-definition None instead).
_SHARED_TABLE = None


def _lookup_shared(unit_slice, rng):
    """Worker that reads the parent-built module-global by index."""
    return int(_SHARED_TABLE[int(unit_slice.unit_codes[0])])


def _count_codes(unit_slice, rng):
    """Trivial deterministic worker: ignores rng, returns Slice size."""
    return len(unit_slice.unit_codes)


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
