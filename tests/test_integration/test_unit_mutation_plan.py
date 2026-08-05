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

from may.geography.venue import Venue
from may.residence.allocation_strategy import execute_allocation_strategy
from may.residence.household_distributor import HouseholdError
from may.residence.unit_mutation_plan import (
    UnitMutationCollector, apply_unit_plan, index_venues_by_id)
from tests.test_integration.golden_fingerprint import (
    RANDOM_SEED, _build_world, _strategy_file_at_loop_level, fingerprint_venues)


def _build_world_with_stable_venue_ids():
    """A stress world whose pre-existing venue ids do not depend on the session.

    A plan names pre-existing venues by id, and in production the collector and
    the applier share one world so that is exact. Here they are three separate
    worlds, and `Venue._next_venue_id` is a process-global that no fixture
    resets — so without this every build would number its care homes
    differently and the replay would not find them.
    """
    Venue.reset_id_counter()
    return _build_world()


def _run(strategy_file, plan_sink=None):
    population, venues, household_distributor = _build_world_with_stable_venue_ids()
    execute_allocation_strategy(
        population=population,
        venues=venues,
        household_distributor=household_distributor,
        strategy_file=strategy_file,
        random_seed=RANDOM_SEED,
        plan_sink=plan_sink,
    )
    return population, venues, household_distributor


def _resident_slots(venues):
    """Every venue's `residents` / `residents_<column>` lists, as person ids.

    `fingerprint_venues` digests subsets only, so these lists — which
    `venue_allocator` keeps in parallel with them, holding live people — would
    otherwise be outside the gate entirely.
    """
    return {venue.id: {key: [person.id for person in value]
                       for key, value in venue.properties.items()
                       if key == 'residents' or key.startswith('residents_')}
            for venues_by_id in venues.venues_by_type_and_id.values()
            for venue in venues_by_id.values()}


def _person_properties(population):
    return {person.id: person.properties
            for person in population.get_all_people()}


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
        _build_world_with_stable_venue_ids()
    people_by_id = {person.id: person
                    for person in replay_population.get_all_people()}
    venues_by_id = index_venues_by_id(replay_venues)
    for plan in plans:
        apply_unit_plan(plan, replay_venues, replay_distributor,
                        people_by_id, venues_by_id)

    assert fingerprint_venues(replay_venues) == fingerprint_venues(direct_venues)
    assert _resident_slots(replay_venues) == _resident_slots(direct_venues)
    assert _person_properties(replay_population) == _person_properties(direct_population)
    assert (replay_distributor.allocated_people
            == direct_distributor.allocated_people)


def _collector_over_one_geo_unit():
    """A collector snapshotted over a single SGU of an unallocated world."""
    population, venues, household_distributor = _build_world_with_stable_venue_ids()
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
