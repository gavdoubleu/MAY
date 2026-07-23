"""Thin process-based `parallel_map`: a bounded pool runs a worker function
over `Slice`s, each with its own RNG stream seeded from the Slice's stable
domain identity (ADR-0001 processes-not-threads; ADR-0002 identity seeding).
"""
import multiprocessing as mp
import os
from typing import Callable, List, Sequence

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
    with _FORK_CONTEXT.Pool(processes=resolve_worker_count(workers)) as pool:
        return pool.map(_seed_and_invoke, payloads)
