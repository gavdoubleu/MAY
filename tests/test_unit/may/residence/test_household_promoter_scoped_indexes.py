"""The promoter's household indexes honour the active geo-unit scope.

`set_geo_unit_scope` invalidates both indexes, so under the geo_unit-outer
loop they are rebuilt once per loop unit. Building them from a whole-world
venue scan therefore costs O(units x households) — and quadratically so,
because the household set grows monotonically through the run. Scoped, the
per-unit rebuilds sum back to one whole-world pass.

The indexes are internal caches, but they are the seam the cost lives on and
their *order* is load-bearing: `promote_and_allocate` shuffles the cached
per-geo-unit list in place, so a list that differs in contents or order draws
the random stream differently and moves world output.
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


def build_households(distributor, unit_names_in_creation_order):
    """Create one household per entry, in the order given, so creation order
    and geographical unit deliberately do not line up."""
    created = []
    for unit_name in unit_names_in_creation_order:
        unit = distributor.geography.get_unit(unit_name)
        created.append(distributor.venue_manager.create_venue('household', unit))
    return created


class TestScopedGeoUnitIndex:

    def test_index_holds_only_in_scope_households(self, distributor):
        build_households(distributor, ['A', 'B', 'A', 'C', 'B'])

        distributor.set_geo_unit_scope({'A', 'B'})
        index = distributor.promoter._get_households_by_geo_unit()

        assert set(index) == {'A', 'B'}

    def test_per_code_order_is_unchanged_by_scoping(self, distributor):
        """Contents and order per geo code must match the whole-world build:
        the promoter shuffles these lists in place, so any difference changes
        how the random stream is consumed and moves world output.
        """
        build_households(distributor, ['A', 'B', 'A', 'C', 'B', 'A'])

        distributor.set_geo_unit_scope(None)
        whole_world = {code: list(households) for code, households
                       in distributor.promoter._get_households_by_geo_unit().items()}

        distributor.set_geo_unit_scope({'A', 'C'})
        scoped = distributor.promoter._get_households_by_geo_unit()

        assert scoped == {'A': whole_world['A'], 'C': whole_world['C']}


class TestScopedPatternIndex:

    def test_index_holds_only_in_scope_households_in_creation_order(self, distributor):
        """Same scoping as the geo_unit index, and for the same reason: the
        rule loop at `promote_and_allocate`'s pattern stage skips any household
        whose geo code is absent from `person_pool_by_geo_unit`, so out-of-scope
        entries were only ever iterated to be discarded. Order within a pattern
        still has to hold — the loop draws from the random stream per household
        it keeps.
        """
        households = build_households(distributor, ['A', 'B', 'A', 'C', 'B', 'A'])
        for index, household in enumerate(households):
            household.properties['allocation_pattern'] = 'K0+' if index % 2 else 'A2+'

        distributor.set_geo_unit_scope(None)
        whole_world = {pattern: list(matching) for pattern, matching
                       in distributor.promoter._get_households_by_pattern().items()}

        distributor.set_geo_unit_scope({'A', 'C'})
        scoped = distributor.promoter._get_households_by_pattern()

        in_scope = {'A', 'C'}
        assert scoped == {
            pattern: [household for household in matching
                      if household.geographical_unit.name in in_scope]
            for pattern, matching in whole_world.items()
        }
