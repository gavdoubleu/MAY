"""Remaining-people statistics honour the active geo-unit scope.

`get_available_people_by_category` is called at the end of every allocation
step to log what is left. Under the geo_unit-outer loop that is once per
step *per unit*, so a whole-world scan turns an O(population) report into
O(population x units) — the dominant cost of a fine `loop_level`. Scoped to
the unit, the per-unit reports sum back to one whole-world pass.
"""

import pytest

from may.geography import Geography, GeographicalUnit
from may.geography.venue_manager import VenueManager
from may.population.person import Person
from may.population.population import PopulationManager
from may.residence.household_distributor import HouseholdDistributor


CONFIG = "tests/test_data/micro_world/households/test_households_config.yaml"


def make_geography(unit_names):
    geography = Geography(levels=['SGU'])
    geography.units = {}
    geography.units_by_level = {'SGU': {}}
    for index, name in enumerate(unit_names):
        unit = GeographicalUnit(id=index, name=name, level='SGU')
        geography.units[name] = unit
        geography.units_by_level['SGU'][name] = unit
        geography.units_by_id[unit.id] = unit
    return geography


@pytest.fixture
def distributor_with_people(tmp_path):
    """Two units, each holding one adult, wired into both the population
    manager and their geographical unit."""
    geography = make_geography(['A', 'B'])
    population = PopulationManager(geography=geography, data_dir=str(tmp_path))
    for unit_name in ['A', 'B']:
        unit = geography.get_unit(unit_name)
        person = Person(age=40, sex='m', geographical_unit=unit)
        unit.add_person(person)
        population.people.append(person)

    distributor = HouseholdDistributor(
        geography=geography,
        population=population,
        venue_manager=VenueManager(geography, filter_by_geography=False),
        data_dir=str(tmp_path),
        config_file=CONFIG,
    )
    return distributor


def total(counts):
    return sum(counts.values())


class TestScopedRemainingCounts:

    def test_unscoped_counts_cover_the_whole_world(self, distributor_with_people):
        """No scope set means the old behaviour: every person is counted."""
        assert total(distributor_with_people.get_available_people_by_category()) == 2

    def test_scoped_counts_cover_only_the_scoped_units(self, distributor_with_people):
        """With a scope active the report describes that unit, so the scan
        never touches people belonging to other units."""
        distributor_with_people.set_geo_unit_scope({'A'})

        assert total(distributor_with_people.get_available_people_by_category()) == 1

    def test_clearing_the_scope_restores_whole_world_counts(
            self, distributor_with_people):
        """Scope is per-unit state, not a permanent narrowing."""
        distributor_with_people.set_geo_unit_scope({'A'})
        distributor_with_people.set_geo_unit_scope(None)

        assert total(distributor_with_people.get_available_people_by_category()) == 2


class TestScopedRemainingIsCheap:
    """`total_people_remaining` is logged once per step per loop unit, so it
    cannot afford a scan. It is derived from a per-unit baseline and the
    allocated set, which only grows — never from person pool sizes, which are
    a deliberately stale snapshot.
    """

    def test_unscoped_remaining_covers_the_whole_world(self, distributor_with_people):
        assert distributor_with_people.get_available_people_count_in_scope() == 2

    def test_setting_a_scope_narrows_the_count(self, distributor_with_people):
        """The baseline is per-unit state: reading it before the scope is set
        must not pin the whole-world figure for the rest of the run."""
        assert distributor_with_people.get_available_people_count_in_scope() == 2

        distributor_with_people.set_geo_unit_scope({'A'})

        assert distributor_with_people.get_available_people_count_in_scope() == 1

    def test_allocating_in_scope_moves_both_counts(self, distributor_with_people):
        """`mark_people_as_allocated` is the external entry point and does not
        touch person pools, so a count read off pool sizes would miss this.
        Both counts follow the allocated set instead."""
        distributor_with_people.set_geo_unit_scope({'A'})
        person_in_scope = distributor_with_people.geography.get_unit('A').people[0]

        assert distributor_with_people.get_allocated_people_count_in_scope() == 0

        distributor_with_people.mark_people_as_allocated([person_in_scope])

        assert distributor_with_people.get_allocated_people_count_in_scope() == 1
        assert distributor_with_people.get_available_people_count_in_scope() == 0

    def test_people_allocated_before_the_unit_still_count_as_allocated(
            self, distributor_with_people):
        """A venue step may allocate someone before household allocation
        reaches their unit. The unit's allocated count describes the unit, not
        only what this loop iteration did, so those people are included."""
        person_in_scope = distributor_with_people.geography.get_unit('A').people[0]
        distributor_with_people.mark_people_as_allocated([person_in_scope])

        distributor_with_people.set_geo_unit_scope({'A'})

        assert distributor_with_people.get_allocated_people_count_in_scope() == 1
        assert distributor_with_people.get_available_people_count_in_scope() == 0


class TestScopedScanIsLazy:
    """The scan itself must not build a list of the people it counts.

    `get_available_people_by_category` only ever counts, so the intermediate
    list is pure allocation — once per step per loop unit, sized by the
    unit's population. Counting is order-independent, so consuming the scan
    lazily cannot change the result.
    """

    def test_people_in_scope_is_a_single_pass_scan(self, distributor_with_people):
        distributor_with_people.set_geo_unit_scope({'A'})

        people = distributor_with_people.people_in_scope()

        assert iter(people) is people
        assert [person.geographical_unit.name for person in people] == ['A']
