"""Thin process-based `parallel_map`: a bounded pool runs a worker function
over `Slice`s, each with its own RNG stream seeded from the Slice's stable
domain identity (ADR-0001 processes-not-threads; ADR-0002 identity seeding).
"""
import multiprocessing as mp
import os
from collections import deque
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
    count. Never returns below 1.

    CPUs and nothing else. Phase 6 had planned to divide a memory budget by a
    measured per-Slice footprint, and dropped it: a Slice's footprint cannot be
    anticipated, so the divisor would have been a guess of the kind
    `pending_flush_member_bound` was retired for. Memory is defended where it
    is spent instead — by flushing against a measured budget (ADR-0008 as
    amended), and by bounding how far dispatch runs ahead of the apply.
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
                 global_seed: int = 0, workers: int = None,
                 max_tasks_per_child: int = None) -> List:
    """Run `worker_fn(slice, rng)` for each Slice across a bounded process pool,
    returning results in input order. Worker count is decoupled from Slice count
    — the pool drains the Slices from a shared queue.

    `max_tasks_per_child` retires and reforks a worker after that many Slices.
    A long-lived forked worker keeps whatever it wrote — module-level or
    process-global state a `worker_fn` accumulates never resets between
    Slices the same worker draws (see `CONTEXT.md`'s "Worker world copy").
    That is invisible and unreclaimable from the parent, unlike its own
    memory. `None` (the default) never retires a worker, matching today's
    behaviour.
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
    with _FORK_CONTEXT.Pool(processes=worker_count,
                            maxtasksperchild=max_tasks_per_child) as pool:
        return pool.map(_seed_and_invoke, payloads)


def parallel_map_in_order(worker_fn: Callable, slices: Sequence[Slice], *,
                          global_seed: int = 0, workers: int = None,
                          force_fork: bool = False,
                          look_ahead: int = None,
                          max_tasks_per_child: int = None) -> Iterator[Tuple[int, Any]]:
    """As `parallel_map`, but yields `(slice_index, result)` one at a time, in
    input order, with only a bounded number of Slices ever in flight.

    `parallel_map` returns only once the slowest Slice is done, so the caller
    holds every result live before it can apply any of them. Phase 5's payloads
    are whole loop units of household mutations, where that is the wrong shape:
    ADR-0003 wants the parent applying each Slice's mutations as it arrives.

    Input order, not completion order. A caller that must apply in a canonical
    order (ADR-0010) gains nothing from an early result it is not allowed to
    apply yet — it can only hold it, and holding it is what an unbounded
    reorder buffer is. Yielding in order puts that buffer here instead, where
    `look_ahead` bounds it: at most that many Slices are dispatched but not yet
    handed over, so at most that many results exist anywhere at once. It costs
    no parallelism, because Slices behind the one being yielded keep running.

    `look_ahead` defaults to twice the worker count, so the pool stays fed
    while the caller works on what it was just handed. Below the worker count
    it idles workers; a caller whose results are large may still want that.

    `force_fork` suppresses the in-process shortcut below. A worker that
    *mutates* a world and returns a description of what it did needs that world
    to be a throwaway copy: run in-process it would change the caller's world
    as well as report the change, and the caller would apply it twice. Such a
    caller must fork at every worker count, which also makes its one-worker run
    a genuine control for its many-worker run rather than a different code path.

    `max_tasks_per_child` retires and reforks a worker after that many Slices,
    bounding the memory a long-lived worker itself accumulates — see
    `parallel_map`'s docstring. `None` never retires a worker.
    """
    payloads = [(slice_index, (worker_fn, unit_slice, global_seed))
                for slice_index, unit_slice in enumerate(slices)]
    if not payloads:
        return
    worker_count = min(resolve_worker_count(workers), len(payloads))
    # Single worker runs in-process, lazily — same seeding, no fork, and the
    # caller still sees results one at a time, in the same order.
    if worker_count <= 1 and not force_fork:
        for payload in payloads:
            yield _seed_and_invoke_indexed(payload)
        return
    window = look_ahead if look_ahead is not None else worker_count * 2
    with _FORK_CONTEXT.Pool(processes=worker_count,
                            maxtasksperchild=max_tasks_per_child) as pool:
        # Dispatched and not yet handed over. Held as `AsyncResult`s in
        # dispatch order, which is input order, so the head is always the next
        # Slice to yield — `get()` on it waits for that Slice and no other,
        # while everything behind it carries on running.
        in_flight = deque()
        next_to_dispatch = 0
        while next_to_dispatch < len(payloads) or in_flight:
            while (next_to_dispatch < len(payloads)
                   and len(in_flight) < window):
                in_flight.append(pool.apply_async(_seed_and_invoke_indexed,
                                                  (payloads[next_to_dispatch],)))
                next_to_dispatch += 1
            yield in_flight.popleft().get()
