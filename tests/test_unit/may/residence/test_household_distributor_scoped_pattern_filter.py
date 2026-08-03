"""The pattern filter honours the active geo-unit scope.

`_filter_households_by_patterns` scans every household in the world on every
call. Its callers in `household_excess_handler` run inside the per-unit loop
body, so under a fine `loop_level` the scan repeats once per loop unit —
O(households x units) for a result both callers immediately narrow to the
scope anyway.

Order is load-bearing as everywhere else in this phase: the returned list is
consumed household-by-household against the random stream, so it must stay a
sub-sequence of the whole-world result, not merely the same set.
"""

import pytest

from may.geography import Geography, GeographicalUnit
from may.geography.venue_manager import VenueManager
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
def distributor(tmp_path):
    geography = make_geography(['A', 'B', 'C'])
    population = PopulationManager(geography=geography, data_dir=str(tmp_path))
    return HouseholdDistributor(
        geography=geography,
        population=population,
        venue_manager=VenueManager(geography, filter_by_geography=False),
        data_dir=str(tmp_path),
        config_file=CONFIG,
    )


def build_households(distributor, unit_names_in_creation_order, patterns):
    """One household per entry, so creation order and geographical unit
    deliberately do not line up."""
    created = []
    for unit_name, pattern in zip(unit_names_in_creation_order, patterns):
        unit = distributor.geography.get_unit(unit_name)
        household = distributor.venue_manager.create_venue('household', unit)
        household.properties['original_pattern'] = pattern
        created.append(household)
    return created


class TestScopedPatternFilter:

    def test_returns_only_in_scope_households_in_creation_order(self, distributor):
        build_households(
            distributor,
            ['A', 'B', 'A', 'C', 'B', 'A'],
            ['K0+', 'K0+', 'A2+', 'K0+', 'A2+', 'K0+'],
        )

        distributor.set_geo_unit_scope(None)
        whole_world = distributor._filter_households_by_patterns(['K0+'])

        distributor.set_geo_unit_scope({'A', 'C'})
        scoped = distributor._filter_households_by_patterns(['K0+'])

        assert scoped == [household for household in whole_world
                          if household.geographical_unit.name in {'A', 'C'}]
        assert scoped != whole_world
