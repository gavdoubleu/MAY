"""Thin process-based `parallel_map`: a bounded pool runs a worker function
over `Slice`s, each with its own RNG stream seeded from the Slice's stable
domain identity (ADR-0001 processes-not-threads; ADR-0002 identity seeding).
"""
import multiprocessing as mp
import os
from typing import Any, Callable, Iterator, List, Sequence, Tuple

import numpy as np

from .slicing import Slice

# Fork start method: read-only shared state built in the parent (population
# arrays, spatial indices) rides copy-on-write into workers for free, rather
# than being re-pickled per Slice (ADR-0003).
_FORK_CONTEXT = mp.get_context('fork')


def resolve_worker_count(workers: int = None) -> int:
    """Resolve the pool size. An explicit `workers` wins; otherwise fall back to
    the SLURM CPU allocation (`SLURM_CPUS_PER_TASK`), then the machine's CPU
    count. Never returns below 1. The measured-footprint sizing formula is
    deferred to Phase 6.
    """
    if workers is not None:
        return max(1, workers)
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        return max(1, int(slurm_cpus))
    return max(1, os.cpu_count() or 1)


def _seed_and_invoke_indexed(payload):
    """As `_seed_and_invoke`, but returns the Slice's input index alongside the
    result so an unordered stream stays attributable to its Slice.
    """
    slice_index, worker_payload = payload
    return slice_index, _seed_and_invoke(worker_payload)


def _seed_and_invoke(payload):
    """Worker-side entry point. Builds the Slice's RNG from its stable digest
    so the stream is identical regardless of which worker runs it or when, then
    calls the user function. Top-level (picklable) by necessity.
    """
    worker_fn, unit_slice, global_seed = payload
    rng = np.random.default_rng(
        np.random.SeedSequence((global_seed, unit_slice.digest)))
    return worker_fn(unit_slice, rng)


def parallel_map(worker_fn: Callable, slices: Sequence[Slice], *,
                 global_seed: int = 0, workers: int = None) -> List:
    """Run `worker_fn(slice, rng)` for each Slice across a bounded process pool,
    returning results in input order. Worker count is decoupled from Slice count
    — the pool drains the Slices from a shared queue.
    """
    payloads = [(worker_fn, unit_slice, global_seed) for unit_slice in slices]
    # Never fork more workers than there are Slices to run.
    worker_count = min(resolve_worker_count(workers), len(payloads)) if payloads else 0
    # A single worker runs in-process: no fork, identical seeding. Keeps callers
    # that route every invocation through parallel_map (incl. single-Slice
    # worlds) cheap and fork-safe, and gives the Tier-1 gate a genuine
    # in-process-vs-forked comparison.
    if worker_count <= 1:
        return [_seed_and_invoke(payload) for payload in payloads]
    with _FORK_CONTEXT.Pool(processes=worker_count) as pool:
        return pool.map(_seed_and_invoke, payloads)


def parallel_map_as_completed(worker_fn: Callable, slices: Sequence[Slice], *,
                              global_seed: int = 0,
                              workers: int = None) -> Iterator[Tuple[int, Any]]:
    """As `parallel_map`, but yields `(slice_index, result)` pairs as they
    complete rather than returning every result at once.

    `parallel_map` returns only once the slowest Slice is done, so the caller
    holds every result live before it can apply any of them. Phase 5's payloads
    are whole loop units of household mutations, where that is the wrong shape:
    ADR-0003 wants the parent applying each Slice's mutations as it arrives.

    Results arrive in completion order, not input order — the `slice_index` is
    what lets a caller that needs a canonical apply order reorder them itself.
    A separate entry point rather than a flag on `parallel_map` so Phase 3's
    callers keep their ordered-list contract untouched.
    """
    payloads = [(slice_index, (worker_fn, unit_slice, global_seed))
                for slice_index, unit_slice in enumerate(slices)]
    worker_count = min(resolve_worker_count(workers), len(payloads)) if payloads else 0
    # Single worker runs in-process, lazily — same seeding, no fork, and the
    # caller still sees results one at a time (here, in input order).
    if worker_count <= 1:
        for payload in payloads:
            yield _seed_and_invoke_indexed(payload)
        return
    with _FORK_CONTEXT.Pool(processes=worker_count) as pool:
        for indexed_result in pool.imap_unordered(_seed_and_invoke_indexed,
                                                  payloads):
            yield indexed_result
