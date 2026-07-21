"""Explicit staged-member read API (see docs/adr/0002, docs/adr/0005).

Gutting a flushed subset frees its `.members`, and `.members` stays
fail-loud — there is deliberately no read-through. Code that genuinely
needs a husk's membership asks the serializer for it, which reads the ids
back out of staging. These tests exercise that API through a real staged
world, one batch at a time.
"""

import pytest

from may.flushing import FlushedPropertyError
from may.serialization.world_serializer import StagedMembersUnavailableError
from tests.test_integration.test_timeline_serialization import minimal_world  # noqa: F401

VENUE_TYPE = "household"
SERIALIZATION_CONFIG = "configs/2021/serialization_config.yaml"


def build_households(world):
    """Two households in the minimal world, splitting its ten people."""
    from may.geography import Venue

    people = list(world.population.people)
    sgu = world.geography.units_by_id[3]
    first = Venue(name="household_a", venue_type=VENUE_TYPE, geographical_unit=sgu)
    second = Venue(name="household_b", venue_type=VENUE_TYPE, geographical_unit=sgu)
    world.venues.add_venue(first)
    world.venues.add_venue(second)
    for person in people[:5]:
        first.add_to_subset(person)
    for person in people[5:]:
        second.add_to_subset(person)
    return first, second


def subset_keys_of(venue):
    return [(venue.id, subset.subset_index) for subset in venue.subsets.values()]


def only_subset(venue):
    return list(venue.subsets.values())[0]


def member_ids_of(venue):
    return [person.id for person in only_subset(venue).members]


@pytest.fixture
def staged_world(minimal_world, tmp_path):  # noqa: F811
    """Minimal world with two households, canonical file created, ready to
    flush. Yields (world, household_a, household_b, expected_ids)."""
    first, second = build_households(minimal_world)
    expected = {
        first.id: member_ids_of(first),
        second.id: member_ids_of(second),
    }
    minimal_world.create_canonical_file(
        str(tmp_path / "staged_world.h5"), config_file=SERIALIZATION_CONFIG
    )
    return minimal_world, first, second, expected


class TestGetStagedMemberIds:

    def test_returns_pre_gut_membership_of_a_flushed_subset(self, staged_world):
        world, household, _, expected = staged_world
        world._serializer.flush_venue_subsets(
            world, VENUE_TYPE, subset_keys=subset_keys_of(household)
        )

        staged = world._serializer.get_staged_member_ids(only_subset(household))

        assert staged == expected[household.id]

    def test_subset_missing_from_every_batch_raises(self, staged_world):
        """household_b is still live: its members are on `.members`, not in
        staging, and asking staging for them must say so rather than
        KeyError out of an index lookup."""
        world, household_a, household_b, _ = staged_world
        world._serializer.flush_venue_subsets(
            world, VENUE_TYPE, subset_keys=subset_keys_of(household_a)
        )

        with pytest.raises(StagedMembersUnavailableError, match="never staged"):
            world._serializer.get_staged_member_ids(only_subset(household_b))

    def test_ids_stay_correct_across_multiple_batches(self, staged_world):
        """Reading between batches must not leave a stale index behind: the
        first read builds one over a one-batch file, then a second batch
        appends to it."""
        world, household_a, household_b, expected = staged_world
        serializer = world._serializer

        serializer.flush_venue_subsets(
            world, VENUE_TYPE, subset_keys=subset_keys_of(household_a)
        )
        assert serializer.get_staged_member_ids(only_subset(household_a)) == \
            expected[household_a.id]

        serializer.flush_venue_subsets(
            world, VENUE_TYPE, subset_keys=subset_keys_of(household_b)
        )

        assert serializer.get_staged_member_ids(only_subset(household_a)) == \
            expected[household_a.id]
        assert serializer.get_staged_member_ids(only_subset(household_b)) == \
            expected[household_b.id]

    def test_venue_type_never_flushed_raises(self, staged_world):
        """No batch of this venue_type ever ran, so there is no staging file
        — the caller should hear that, not an h5py file-open error."""
        world, household, _, _ = staged_world

        with pytest.raises(StagedMembersUnavailableError, match="never staged"):
            world._serializer.get_staged_member_ids(only_subset(household))

    def test_members_still_fails_loud_after_flush(self, staged_world):
        """The staged read API is an addition, not a read-through: the husk
        itself must keep refusing (docs/adr/0002)."""
        world, household, _, _ = staged_world
        world._serializer.flush_venue_subsets(
            world, VENUE_TYPE, subset_keys=subset_keys_of(household)
        )

        with pytest.raises(FlushedPropertyError):
            list(only_subset(household).members)


class TestGetStagedMembers:

    def test_resolves_staged_ids_to_people(self, staged_world):
        world, household, _, expected = staged_world
        world._serializer.flush_venue_subsets(
            world, VENUE_TYPE, subset_keys=subset_keys_of(household)
        )

        people = world._serializer.get_staged_members(
            only_subset(household), world.population
        )

        assert [person.id for person in people] == expected[household.id]
        assert all(person is world.population.get_person(person.id)
                   for person in people)

    def test_person_missing_from_population_raises(self, staged_world):
        """A staged id the population can no longer resolve means the two
        have drifted apart — silently dropping the member would corrupt the
        membership the caller asked for."""
        world, household, _, expected = staged_world
        world._serializer.flush_venue_subsets(
            world, VENUE_TYPE, subset_keys=subset_keys_of(household)
        )
        dropped_id = expected[household.id][0]
        del world.population.people_by_id[dropped_id]

        with pytest.raises(StagedMembersUnavailableError, match=str(dropped_id)):
            world._serializer.get_staged_members(
                only_subset(household), world.population
            )
