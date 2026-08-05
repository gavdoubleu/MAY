"""Bounding the pending-flush buffer during residence allocation.

`pending_flush_member_bound` is how much membership a run chooses to hold
live rather than write. A bound of 1 cuts a batch at every opportunity; an
unset bound never cuts one; values between buy exactly as much memory relief
as a machine needs. The bound is one-sided — the excess, overflow and
promotion steps all add people to households earlier steps of the same unit
created, so no unit can be cut into — and it never changes the world it
builds, only when membership leaves memory.
"""

import pytest
import yaml

from may.residence.allocation_strategy import execute_allocation_strategy
from may.residence.household_distributor import HouseholdError
from tests.test_integration.h5_equivalence import assert_h5_equivalent
from tests.test_integration.test_allocation_strategy_integration import (  # noqa: F401
    STRESS_DATA,
    geography,
    hd,
    population_manager,
    venue_manager,
)

STRATEGY_FILE = f"{STRESS_DATA}/households/test_allocation_strategy.yaml"

SERIALIZATION_CONFIG = "configs/2021/serialization_config.yaml"


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


def batch_member_counts(bound, tmp_path):
    """Members carried by each staging call of a run at the given bound.

    Counted at the flusher, before the subsets are gutted, so it measures
    what was actually handed to staging rather than what the buffer says.
    """
    strategy_file = write_strategy(tmp_path, loop_level="SGU",
                                   pending_flush_member_bound=bound)
    geo, population, venues, distributor = build_fixtures()
    world = build_world(population, venues, geo, tmp_path)
    counts = []

    def recording_flusher(venue_type, subset_keys):
        batch = 0
        for venue_id, subset_index in subset_keys:
            venue = venues.get_venue_by_type_and_id(venue_type, venue_id)
            batch += sum(len(subset.members)
                         for subset in venue.subsets.values()
                         if subset.subset_index == subset_index)
        counts.append(batch)
        world.flush_subsets_batch(venue_type, subset_keys)

    execute_allocation_strategy(
        population=population, venues=venues,
        household_distributor=distributor,
        strategy_file=strategy_file, random_seed=42,
        subset_flusher=recording_flusher,
    )
    return counts


class TestBoundRequiresSerialization:

    def test_without_a_flusher_raises(self, hd, population_manager,  # noqa: F811
                                      venue_manager, tmp_path):
        """Staging needs the canonical file; asking for a bounded buffer
        without serialization is a config error, not a silent no-op."""
        strategy_file = write_strategy(tmp_path, pending_flush_member_bound=1)

        with pytest.raises(HouseholdError, match="serialization"):
            execute_allocation_strategy(
                population=population_manager,
                venues=venue_manager,
                household_distributor=hd,
                strategy_file=strategy_file,
            )


class TestBoundOfOne:

    def test_every_household_subset_is_a_husk_with_staged_members(
        self, hd, population_manager, venue_manager, geography, tmp_path
    ):
        """A bound of one member cuts a batch at every opportunity, so after
        the run nothing holds live household membership — but all of it is
        readable back out of staging."""
        from may.flushing import GuttedMembers

        world = build_world(population_manager, venue_manager, geography, tmp_path)
        strategy_file = write_strategy(tmp_path, loop_level="SGU",
                                       pending_flush_member_bound=1)

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


class TestUnsetBound:

    def test_membership_stays_live(
        self, hd, population_manager, venue_manager, geography, tmp_path
    ):
        """Unset is the default, and it wires no buffering at all: a machine
        that can hold the world unflushed pays nothing for the knob."""
        from may.flushing import GuttedMembers

        world = build_world(population_manager, venue_manager, geography, tmp_path)
        strategy_file = write_strategy(tmp_path, loop_level="SGU")

        execute_allocation_strategy(
            population=population_manager,
            venues=venue_manager,
            household_distributor=hd,
            strategy_file=strategy_file,
            subset_flusher=world.flush_subsets_batch,
        )

        subsets = household_subsets(venue_manager)
        assert subsets, "the stress world should produce households"
        assert not any(isinstance(subset.members, GuttedMembers)
                       for _, subset in subsets)


class TestResidualDrain:

    def test_a_bound_too_large_to_bind_still_stages_everything(
        self, hd, population_manager, venue_manager, geography, tmp_path
    ):
        """A set bound drains at end of allocation whether or not it ever
        bound, so the end state is the same for every value: every household
        a husk, and the terminal whole-world flush finds the cohort staged
        rather than building one large transient out of live membership."""
        from may.flushing import GuttedMembers

        world = build_world(population_manager, venue_manager, geography, tmp_path)
        strategy_file = write_strategy(tmp_path, loop_level="SGU",
                                       pending_flush_member_bound=10 ** 9)

        calls = []

        def recording_flusher(venue_type, subset_keys):
            calls.append(list(subset_keys))
            world.flush_subsets_batch(venue_type, subset_keys)

        execute_allocation_strategy(
            population=population_manager,
            venues=venue_manager,
            household_distributor=hd,
            strategy_file=strategy_file,
            subset_flusher=recording_flusher,
        )

        assert len(calls) == 1, "nothing crossed the bound, so only the drain"
        assert all(isinstance(subset.members, GuttedMembers)
                   for _, subset in household_subsets(venue_manager))


def cumulative(counts):
    running, totals = 0, []
    for count in counts:
        running += count
        totals.append(running)
    return totals


class TestIntermediateBound:

    def test_units_are_grouped_into_batches_never_split(self, tmp_path):
        """The bound is one-sided: it can only make flushes rarer than one
        per unit. Every batch it cuts is a whole number of units, because no
        household's membership is final until its unit's last step has run.
        """
        per_unit = batch_member_counts(1, tmp_path / "per_unit")
        assert len(per_unit) > 2, "the stress world should span several units"

        bound = per_unit[0] + 1
        grouped = batch_member_counts(bound, tmp_path / "grouped")

        assert len(grouped) < len(per_unit)
        assert sum(grouped) == sum(per_unit)
        assert set(cumulative(grouped)) <= set(cumulative(per_unit))


def export_at_bound(bound, tmp_path):
    """Export the stress world built at a given bound, ids reset first.

    Person and Venue ids come from process-global counters, so without the
    reset two runs in one process are offset and nothing downstream of the
    ids could be compared.
    """
    from may.geography.venue import Venue
    from may.population.person import Person

    Person.reset_counter()
    Venue.reset_id_counter()

    overrides = {} if bound is None else {"pending_flush_member_bound": bound}
    strategy_file = write_strategy(tmp_path, loop_level="SGU", **overrides)
    geo, population, venues, distributor = build_fixtures()
    world = build_world(population, venues, geo, tmp_path)
    execute_allocation_strategy(
        population=population, venues=venues,
        household_distributor=distributor,
        strategy_file=strategy_file, random_seed=42,
        subset_flusher=world.flush_subsets_batch,
    )
    # The canonical file the run staged beside — export finishes it in place
    export_path = str(tmp_path / "world.h5")
    world.export_to_hdf5(export_path, config_file=SERIALIZATION_CONFIG)
    return export_path


class TestBoundsAreInterchangeableForOutput:

    def test_every_bound_exports_the_same_world(self, tmp_path):
        """What licenses sweeping the bound on a target machine without
        re-validating the output: the bound changes which complete subsets
        are grouped into which batch, and the export re-indexes by key, so
        it cannot change the world. Tested rather than argued.
        """
        first_unit_members = batch_member_counts(1, tmp_path / "probe")[0]

        per_unit = export_at_bound(1, tmp_path / "per_unit")
        intermediate = export_at_bound(first_unit_members + 1,
                                       tmp_path / "intermediate")
        never = export_at_bound(None, tmp_path / "never")

        assert_h5_equivalent(per_unit, intermediate)
        assert_h5_equivalent(per_unit, never)


class TestInvalidBounds:

    @pytest.mark.parametrize("bound", [0, -1, 1.5, "many", True])
    def test_a_bound_that_is_not_a_positive_count_of_members_raises(
        self, bound, hd, population_manager, venue_manager,  # noqa: F811
        geography, tmp_path
    ):
        """The value is a member count, and nothing else means anything.

        `true` is rejected rather than read as 1: it is what the deleted
        `flush_per_unit` took, and silently honouring it would hide a stale
        config instead of pointing at it.
        """
        world = build_world(population_manager, venue_manager, geography, tmp_path)
        strategy_file = write_strategy(tmp_path, loop_level="SGU",
                                       pending_flush_member_bound=bound)

        with pytest.raises(HouseholdError, match="pending_flush_member_bound"):
            execute_allocation_strategy(
                population=population_manager,
                venues=venue_manager,
                household_distributor=hd,
                strategy_file=strategy_file,
                subset_flusher=world.flush_subsets_batch,
            )


class TestExportCoversTheStagedCohort:

    def test_export_writes_every_staged_household(self, tmp_path):
        """Final export re-indexes the staged batches by key and verifies
        their union covers the venue type (docs/adr/0005)."""
        import h5py

        strategy_file = write_strategy(tmp_path, loop_level="SGU",
                                       pending_flush_member_bound=1)
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
                                       pending_flush_member_bound=1)
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
        backwards, so the buffer's keys arrive in reverse creation order; the
        scan silently re-sorted them. Loop units run in creation order too, so
        staged venue ids are non-decreasing across the whole run — a batch
        staged in offer order would descend within each unit.
        """
        import h5py

        strategy_file = write_strategy(tmp_path, loop_level="SGU",
                                       pending_flush_member_bound=1)
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

    def test_world_argument_enables_the_bounded_buffer(self, tmp_path):
        """The config-driven path: setup_households builds the flusher from
        the world's serializer, so a bound works from YAML without the caller
        wiring anything up."""
        from may.flushing import GuttedMembers
        from may.world import setup_households

        strategy_file = write_strategy(tmp_path, loop_level="SGU",
                                       pending_flush_member_bound=1)
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
