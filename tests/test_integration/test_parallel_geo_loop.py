"""The forked geo-unit loop must build exactly the world the sequential one did.

Phase 5 cycle 4. Each loop unit's steps now run in a forked worker over its own
throwaway copy of the world; the parent creates every household, mints every id
and applies each unit's `UnitMutationPlan` in the loop's own unit order.

Two things are being gated at once, and one test catches both. Against the
stored golden the run must be bit-for-bit what the sequential loop produced —
that is Tier-1 sequential equivalence. And the run must be *forked*: were the
body still running in the parent, the parent would both mutate its world and
replay the plan describing that mutation, so every household would appear
twice and the digest would not survive it.

`SGU` is the only stress loop level with more than one unit, so it is the only
one where worker count means anything.
"""
from tests.test_integration.golden_fingerprint import (
    RANDOM_SEED, _strategy_file_at_loop_level, build_world_with_stable_venue_ids,
    fingerprint_venues, load_golden, person_properties, resident_slots)

from may.residence.allocation_strategy import execute_allocation_strategy


def _run(strategy_file, workers):
    population, venues, household_distributor = build_world_with_stable_venue_ids()
    execute_allocation_strategy(
        population=population,
        venues=venues,
        household_distributor=household_distributor,
        strategy_file=strategy_file,
        random_seed=RANDOM_SEED,
        workers=workers,
    )
    return population, venues, household_distributor


def test_worker_count_does_not_change_the_world(tmp_path):
    strategy_file = _strategy_file_at_loop_level("SGU", str(tmp_path))

    single_population, single_venues, single_distributor = _run(strategy_file, 1)
    many_population, many_venues, many_distributor = _run(strategy_file, 4)

    assert fingerprint_venues(single_venues) == load_golden()["SGU"]
    assert fingerprint_venues(many_venues) == fingerprint_venues(single_venues)
    assert resident_slots(many_venues) == resident_slots(single_venues)
    assert person_properties(many_population) == person_properties(single_population)
    assert many_distributor.allocated_people == single_distributor.allocated_people
