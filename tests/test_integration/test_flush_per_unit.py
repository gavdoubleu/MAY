"""Per-unit household subset flushing during residence allocation.

With `flush_per_unit: true`, each geo unit's household subsets are staged
and gutted as soon as that unit's steps finish, so peak memory holds one
unit's membership rather than the whole world's. The exported world must
be indistinguishable from the same run without flushing.
"""

import pytest
import yaml

from may.residence.allocation_strategy import execute_allocation_strategy
from may.residence.household_distributor import HouseholdError
from tests.test_integration.test_allocation_strategy_integration import (  # noqa: F401
    STRESS_DATA,
    geography,
    hd,
    population_manager,
    venue_manager,
)

STRATEGY_FILE = f"{STRESS_DATA}/households/test_allocation_strategy.yaml"


def write_strategy(tmp_path, **overrides):
    """The stress strategy with top-level keys overridden."""
    with open(STRATEGY_FILE) as handle:
        strategy = yaml.safe_load(handle)
    strategy.update(overrides)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "strategy.yaml"
    with open(path, "w") as handle:
        yaml.safe_dump(strategy, handle)
    return str(path)


SERIALIZATION_CONFIG = "configs/2021/serialization_config.yaml"


def build_world(population_manager, venue_manager, geography, tmp_path):
    """A World over the stress fixtures with its canonical file created, so
    the serializer can stage subsets beside it."""
    from may.world import World

    world = World(geography=geography, population=population_manager,
                  venues=venue_manager)
    world.create_canonical_file(str(tmp_path / "world.h5"),
                                config_file=SERIALIZATION_CONFIG)
    return world


def household_subsets(venue_manager):
    return [(venue, subset)
            for venue in venue_manager.get_venues_by_type("household")
            for subset in venue.subsets.values()]


class TestFlushPerUnitRequiresSerialization:

    def test_without_a_flusher_raises(self, hd, population_manager,  # noqa: F811
                                      venue_manager, tmp_path):
        """Staging needs the canonical file; asking for per-unit flushing
        without serialization is a config error, not a silent no-op."""
        strategy_file = write_strategy(tmp_path, flush_per_unit=True)

        with pytest.raises(HouseholdError, match="serialization"):
            execute_allocation_strategy(
                population=population_manager,
                venues=venue_manager,
                household_distributor=hd,
                strategy_file=strategy_file,
            )


class TestFlushPerUnitRun:

    def test_every_household_subset_is_a_husk_with_staged_members(
        self, hd, population_manager, venue_manager, geography, tmp_path
    ):
        """After the run nothing holds live household membership in memory,
        but all of it is readable back out of staging."""
        from may.flushing import GuttedMembers

        world = build_world(population_manager, venue_manager, geography, tmp_path)
        strategy_file = write_strategy(tmp_path, loop_level="SGU",
                                       flush_per_unit=True)

        execute_allocation_strategy(
            population=population_manager,
            venues=venue_manager,
            household_distributor=hd,
            strategy_file=strategy_file,
            subset_flusher=world.flush_subsets_batch,
        )

        subsets = household_subsets(venue_manager)
        assert subsets, "the stress world should produce households"
        assert all(isinstance(subset.members, GuttedMembers)
                   for _, subset in subsets)

        staged_ids = set()
        for _, subset in subsets:
            staged_ids.update(world._serializer.get_staged_member_ids(subset))
        assert staged_ids == set(hd.allocated_people)


def build_fixtures():
    """Fresh stress-world managers, so two runs can be compared in one test
    (the module fixtures are function-scoped and shared)."""
    from may.geography import Geography
    from may.geography.venue_manager import VenueManager
    from may.population.population import PopulationManager
    from may.residence.household_distributor import HouseholdDistributor

    geo = Geography(data_dir=f"{STRESS_DATA}/geography",
                    levels=["SGU", "MGU", "LGU"])
    geo.load_from_csv()
    venues = VenueManager(geo, data_dir=f"{STRESS_DATA}/venues")
    venues.load_from_yaml_config("test_venues_config.yaml")
    population = PopulationManager(geography=geo,
                                   data_dir=f"{STRESS_DATA}/population")
    population.load_explicit_from_csv(
        "people.csv",
        column_mapping={"age": "age", "sex": "sex", "geo_unit": "location"},
    )
    distributor = HouseholdDistributor(
        geography=geo, population=population, venue_manager=venues,
        data_dir=f"{STRESS_DATA}/households",
        config_file="test_households_config.yaml",
        rules_file="relationship_rules.yaml",
    )
    distributor.load_household_data("households.csv")
    return geo, population, venues, distributor


def household_compositions(venue_manager, population, serializer=None):
    """Every household as (geo unit, sorted resident (age, sex) pairs).

    Compared instead of raw ids because Person and Venue ids come from
    process-global counters, so two runs in one process are offset.
    """
    compositions = []
    for venue in venue_manager.get_venues_by_type("household"):
        residents = []
        for subset in venue.subsets.values():
            if serializer is None:
                members = subset.members
            else:
                members = serializer.get_staged_members(subset, population)
            residents.extend((person.age, person.sex) for person in members)
        compositions.append((venue.geographical_unit.name, sorted(residents)))
    return sorted(compositions)


class TestFlushedRunMatchesUnflushedRun:

    def test_same_seed_same_households(self, tmp_path):
        """Flushing changes when membership leaves memory, not what it is.

        Loops at SGU so the run spans three units, i.e. three disjoint
        staging batches rather than one whole-world flush."""
        unflushed_strategy = write_strategy(tmp_path / "plain",
                                            loop_level="SGU")
        _, population, venues, distributor = build_fixtures()
        execute_allocation_strategy(
            population=population, venues=venues,
            household_distributor=distributor,
            strategy_file=unflushed_strategy, random_seed=42,
        )
        expected = household_compositions(venues, population)

        flushed_strategy = write_strategy(tmp_path / "flushed",
                                          loop_level="SGU", flush_per_unit=True)
        geo, population, venues, distributor = build_fixtures()
        world = build_world(population, venues, geo, tmp_path / "flushed")
        execute_allocation_strategy(
            population=population, venues=venues,
            household_distributor=distributor,
            strategy_file=flushed_strategy, random_seed=42,
            subset_flusher=world.flush_subsets_batch,
        )

        assert household_compositions(
            venues, population, serializer=world._serializer) == expected

    def test_export_writes_every_staged_household(self, tmp_path):
        """Final export re-indexes the staged batches by key and verifies
        their union covers the venue type (docs/adr/0005)."""
        import h5py

        strategy_file = write_strategy(tmp_path, loop_level="SGU",
                                       flush_per_unit=True)
        geo, population, venues, distributor = build_fixtures()
        world = build_world(population, venues, geo, tmp_path)
        execute_allocation_strategy(
            population=population, venues=venues,
            household_distributor=distributor,
            strategy_file=strategy_file, random_seed=42,
            subset_flusher=world.flush_subsets_batch,
        )
        expected_members = sum(
            len(subset.members) for _, subset in household_subsets(venues))

        export_path = str(tmp_path / "world.h5")
        world.export_to_hdf5(export_path, config_file=SERIALIZATION_CONFIG)

        with h5py.File(export_path, "r") as exported:
            member_counts = exported["venues/subsets/member_counts"][:]
            members_flat = exported["venues/subsets/members_flat"][:]
        assert int(member_counts.sum()) == expected_members
        assert len(members_flat) == expected_members


class TestBatchedFlushCostIsInTheBatch:
    """A batched flush must not enumerate the world.

    Building the candidate list by walking every venue of the type made each
    flush O(total households) whatever the batch size, so across a run it cost
    O(loop units x total households) — invisible at nine XLGU units, ~6,856x at
    MGU. The batch's keys are (Venue.id, subset_index) and the venue manager
    indexes venues by exactly that id, so the candidates are lookups.
    """

    def test_flushing_a_batch_never_enumerates_the_venue_type(self, tmp_path):
        """The contract is a call count, not a wall clock: a timing assertion
        at stress-world scale would be flaky, and the defect is precisely that
        a whole-type enumeration happens at all."""
        from may.geography.venue_manager import VenueManager

        enumerations = []
        inside_flush = False
        original_get_venues_by_type = VenueManager.get_venues_by_type

        def recording_get_venues_by_type(self, venue_type):
            if inside_flush:
                enumerations.append(venue_type)
            return original_get_venues_by_type(self, venue_type)

        strategy_file = write_strategy(tmp_path, loop_level="SGU",
                                       flush_per_unit=True)
        geo, population, venues, distributor = build_fixtures()
        world = build_world(population, venues, geo, tmp_path)

        def flusher(venue_type, subset_keys):
            nonlocal inside_flush
            inside_flush = True
            try:
                world.flush_subsets_batch(venue_type, subset_keys)
            finally:
                inside_flush = False

        VenueManager.get_venues_by_type = recording_get_venues_by_type
        try:
            execute_allocation_strategy(
                population=population, venues=venues,
                household_distributor=distributor,
                strategy_file=strategy_file, random_seed=42,
                subset_flusher=flusher,
            )
        finally:
            VenueManager.get_venues_by_type = original_get_venues_by_type

        assert enumerations == []

    def test_batches_are_staged_in_venue_creation_order(self, tmp_path):
        """Characterisation guard on the order the removed scan imposed.

        `_new_household_subset_keys` walks the venue manager's households
        backwards, so the caller's keys arrive in reverse creation order; the
        scan silently re-sorted them. Loop units run in creation order too, so
        staged venue ids are non-decreasing across the whole run — a batch
        staged in caller order would descend within each unit.
        """
        import h5py

        strategy_file = write_strategy(tmp_path, loop_level="SGU",
                                       flush_per_unit=True)
        geo, population, venues, distributor = build_fixtures()
        world = build_world(population, venues, geo, tmp_path)
        execute_allocation_strategy(
            population=population, venues=venues,
            household_distributor=distributor,
            strategy_file=strategy_file, random_seed=42,
            subset_flusher=world.flush_subsets_batch,
        )

        staging_path = world._serializer._staging_path("household")
        with h5py.File(staging_path, "r") as staged:
            venue_ids = staged["subsets/venue_ids"][:]
            subset_indices = staged["subsets/subset_indices"][:]

        assert len(venue_ids) > 1, "the stress world should stage many subsets"
        keys = list(zip(venue_ids.tolist(), subset_indices.tolist()))
        assert keys == sorted(keys)


class TestSetupHouseholdsPlumbing:

    def test_world_argument_enables_per_unit_flushing(self, tmp_path):
        """The config-driven path: setup_households builds the flusher from
        the world's serializer, so `flush_per_unit: true` works from YAML
        without the caller wiring anything up."""
        from may.flushing import GuttedMembers
        from may.world import setup_households

        strategy_file = write_strategy(tmp_path, loop_level="SGU",
                                       flush_per_unit=True)
        geo, population, venues, _ = build_fixtures()
        world = build_world(population, venues, geo, tmp_path)
        config = {"households": {
            "data_dir": f"{STRESS_DATA}/households",
            "config_file": "test_households_config.yaml",
            "rules_file": "relationship_rules.yaml",
            "data_file": "households.csv",
        }}

        setup_households(geo, population, venues, config,
                         strategy_file=strategy_file, world=world)

        assert all(isinstance(subset.members, GuttedMembers)
                   for _, subset in household_subsets(venues))
