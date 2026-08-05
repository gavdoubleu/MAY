"""Collect-then-apply must build exactly the world direct mutation builds.

Phase 5 cycle 3. The loop body stops being the thing that changes the world:
each loop unit now emits a `UnitMutationPlan` — a compact, object-free
description of everything that unit did — and a parent applies it. Cycle 4
moves the body into a forked worker, at which point the parent is the only
process whose world is real. This test is the gate for the restructure while
both halves still run in one process, so a failure here is a collector or
applier bug and nothing else.

`SGU` is the loop level with more than one unit on the stress geography (see
`golden_fingerprint`), so it is the one that exercises per-unit scoping.
"""
import pytest

from may.residence.allocation_strategy import execute_allocation_strategy
from may.residence.household_distributor import HouseholdError
from may.residence.unit_mutation_plan import (
    UnitMutationCollector, apply_unit_plan, index_venues_by_id)
from tests.test_integration.golden_fingerprint import (
    RANDOM_SEED, _strategy_file_at_loop_level, build_world_with_stable_venue_ids,
    fingerprint_venues, person_properties, resident_slots)


def _run(strategy_file, plan_sink=None):
    population, venues, household_distributor = build_world_with_stable_venue_ids()
    execute_allocation_strategy(
        population=population,
        venues=venues,
        household_distributor=household_distributor,
        strategy_file=strategy_file,
        random_seed=RANDOM_SEED,
        plan_sink=plan_sink,
    )
    return population, venues, household_distributor


def test_applying_collected_plans_rebuilds_the_directly_mutated_world(tmp_path):
    strategy_file = _strategy_file_at_loop_level("SGU", str(tmp_path))

    direct_population, direct_venues, direct_distributor = _run(strategy_file)

    plans = []
    _run(strategy_file, plan_sink=plans.append)
    assert plans, "the loop emitted no plans at all"
    # A record type nothing exercises is a record type nothing gates, so
    # assert the stress world actually produces each of them.
    assert any(plan.created_households for plan in plans)
    assert any(plan.existing_venue_members for plan in plans)
    assert any(plan.venue_property_appends for plan in plans)
    assert any(plan.person_property_updates for plan in plans)

    replay_population, replay_venues, replay_distributor = \
        build_world_with_stable_venue_ids()
    people_by_id = {person.id: person
                    for person in replay_population.get_all_people()}
    venues_by_id = index_venues_by_id(replay_venues)
    for plan in plans:
        apply_unit_plan(plan, replay_venues, replay_distributor,
                        people_by_id, venues_by_id)

    assert fingerprint_venues(replay_venues) == fingerprint_venues(direct_venues)
    assert resident_slots(replay_venues) == resident_slots(direct_venues)
    assert person_properties(replay_population) == person_properties(direct_population)
    assert (replay_distributor.allocated_people
            == direct_distributor.allocated_people)


def _collector_over_one_geo_unit():
    """A collector snapshotted over a single SGU of an unallocated world."""
    population, venues, household_distributor = build_world_with_stable_venue_ids()
    geo_unit_name = next(iter(
        household_distributor.geography.get_units_by_level("SGU")))
    household_distributor.set_geo_unit_scope({geo_unit_name})
    collector = UnitMutationCollector(venues, household_distributor,
                                      geo_unit_name, {geo_unit_name})
    return population, venues, household_distributor, geo_unit_name, collector


def test_a_person_property_with_no_shipping_rule_is_refused():
    # The whole point of the allowlist: a mutation site nobody has enumerated
    # must fail here, in one process, rather than vanish silently once the
    # body runs in a forked worker whose world is thrown away.
    _, _, household_distributor, _, collector = _collector_over_one_geo_unit()
    person = next(iter(household_distributor.people_in_scope()))
    person.properties['favourite_colour'] = 'blue'

    with pytest.raises(HouseholdError, match="favourite_colour"):
        collector.collect()


def test_a_pre_existing_venue_property_with_no_shipping_rule_is_refused():
    _, _, household_distributor, geo_unit_name, collector = \
        _collector_over_one_geo_unit()
    geo_unit = household_distributor.geography.get_unit(geo_unit_name)
    venue = next(iter(geo_unit.venues))
    venue.properties['occupancy_note'] = 'refurbished'

    with pytest.raises(HouseholdError, match="occupancy_note"):
        collector.collect()


def test_creating_anything_other_than_a_household_is_refused():
    # The parent mints ids for households and nothing else, so a venue of any
    # other type created inside the loop would never reach its world.
    _, venues, household_distributor, geo_unit_name, collector = \
        _collector_over_one_geo_unit()
    venues.create_venue(
        venue_type='care_home',
        geo_unit=household_distributor.geography.get_unit(geo_unit_name))

    with pytest.raises(HouseholdError, match="care_home"):
        collector.collect()
