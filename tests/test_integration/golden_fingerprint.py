"""Golden fingerprint of the stress world's allocated households.

Phase 4b removes whole-world scans from the per-loop-unit body. Those are
performance defects only, so the world must come out bit-for-bit identical.
The unit tests each cover one site; this fingerprint is the composed-pipeline
guard that catches any of them perturbing RNG consumption or venue creation
order.

`SGU` is the loop level that matters: the stress geography puts all three SGUs
under one `MGU_S1` (`geography/hierarchy.csv`), so `MGU` and `LGU` both give a
single loop unit and exercise no scoping. `LGU` is captured alongside it as the
whole-world-scope control.

The strategy is `test_allocation_strategy_with_venues.yaml` — the stress
strategy plus two `venue` steps, so `venue_allocator`'s scoping is inside the
gate. It is a separate file because those steps change what the household steps
see, and the tests in `test_allocation_strategy_integration.py` assert the
household-only outcomes.
"""
import hashlib
import json
import os

from may.geography import Geography
from may.geography.venue import Venue
from may.geography.venue_manager import VenueManager
from may.population.population import PopulationManager
from may.residence.allocation_strategy import execute_allocation_strategy
from may.residence.household_distributor import HouseholdDistributor

STRESS_DATA = "tests/test_data/stress_world"
STRATEGY_FILE = f"{STRESS_DATA}/households/test_allocation_strategy_with_venues.yaml"
GOLDEN_PATH = f"{STRESS_DATA}/golden_fingerprint.json"

# SGU is the only level with more than one loop unit; LGU is the single-unit
# whole-world-scope control.
LOOP_LEVELS = ("SGU", "LGU")

RANDOM_SEED = 42


def _build_world():
    """A fresh stress world — every run must start from an unallocated one."""
    geography = Geography(data_dir=f"{STRESS_DATA}/geography",
                          levels=["SGU", "MGU", "LGU"],
                          hierarchy_file="hierarchy.csv")
    geography.load_from_csv()

    venue_manager = VenueManager(geography, data_dir=f"{STRESS_DATA}/venues")
    venue_manager.load_from_yaml_config("test_venues_config.yaml")

    population_manager = PopulationManager(
        geography=geography, data_dir=f"{STRESS_DATA}/population")
    population_manager.load_explicit_from_csv(
        "people.csv",
        column_mapping={"age": "age", "sex": "sex", "geo_unit": "location"})

    household_distributor = HouseholdDistributor(
        geography=geography,
        population=population_manager,
        venue_manager=venue_manager,
        data_dir=f"{STRESS_DATA}/households",
        config_file="test_households_config.yaml",
        rules_file="relationship_rules.yaml")
    household_distributor.load_household_data("households.csv")

    return population_manager, venue_manager, household_distributor


def build_world_with_stable_venue_ids():
    """A stress world whose pre-existing venue ids do not depend on the session.

    `Venue._next_venue_id` is a process-global that no fixture resets, so two
    worlds built in one session number their care homes differently. Any gate
    that compares two separately built worlds by venue id — a mutation plan
    names pre-existing venues that way — needs them to agree.
    """
    Venue.reset_id_counter()
    return _build_world()


def resident_slots(venue_manager):
    """Every venue's `residents` / `residents_<column>` lists, as person ids.

    `fingerprint_venues` digests subsets only, so these lists — which
    `venue_allocator` keeps in parallel with them, holding live people — would
    otherwise be outside every gate.
    """
    return {venue.id: {key: [person.id for person in value]
                       for key, value in venue.properties.items()
                       if key == 'residents' or key.startswith('residents_')}
            for venues_by_id in venue_manager.venues_by_type_and_id.values()
            for venue in venues_by_id.values()}


def person_properties(population_manager):
    return {person.id: person.properties
            for person in population_manager.get_all_people()}


def _strategy_file_at_loop_level(loop_level, tmp_dir):
    """Copy of the stress strategy with `loop_level` pinned.

    Written beside the original's data files so its relative paths still
    resolve; the file itself goes to `tmp_dir` to keep the fixture untouched.
    """
    import yaml

    with open(STRATEGY_FILE) as handle:
        strategy = yaml.safe_load(handle)
    strategy['loop_level'] = loop_level

    path = os.path.join(tmp_dir, f"strategy_{loop_level}.yaml")
    with open(path, 'w') as handle:
        yaml.safe_dump(strategy, handle, sort_keys=False)
    return path


def fingerprint_venues(venue_manager):
    """sha256 over every venue's identity, per-subset membership and pattern.

    Ordered by venue id, which is global creation order (`Venue._next_venue_id`
    is one monotonic counter), so a change in the order venues are created
    changes the digest — that is the point. Membership is broken out per
    subset key, so a person landing in the right venue under the wrong subset
    is still caught.

    The digest carries each venue's *rank* in that order, never the raw id:
    `_next_venue_id` is a process-global that no fixture resets, so raw ids
    depend on how many venues earlier tests in the same session happened to
    create. Rank captures the ordering this gate cares about and nothing else.

    Covers all venue types, not just households: the `venue` steps allocate
    into care homes and boarding schools, which no household digest would see.
    """
    digest = hashlib.sha256()
    all_venues = sorted(
        (venue for venue_type in sorted(venue_manager.venues_by_type_and_id)
         for venue in venue_manager.venues_by_type_and_id[venue_type].values()),
        key=lambda venue: venue.id)
    for rank, venue in enumerate(all_venues):
        digest.update("{}|{}|{}|{}\n".format(
            rank,
            venue.type,
            venue.geographical_unit.name,
            venue.properties.get('allocation_pattern', ''),
        ).encode())
        for subset_key in sorted(venue.subsets, key=repr):
            member_ids = sorted(
                person.id for person in venue.subsets[subset_key].members)
            digest.update("  {}|{}\n".format(
                subset_key,
                ",".join(str(member_id) for member_id in member_ids),
            ).encode())
    return digest.hexdigest()


def run_and_fingerprint(loop_level, tmp_dir):
    population_manager, venue_manager, household_distributor = _build_world()
    execute_allocation_strategy(
        population=population_manager,
        venues=venue_manager,
        household_distributor=household_distributor,
        strategy_file=_strategy_file_at_loop_level(loop_level, tmp_dir),
        random_seed=RANDOM_SEED,
    )
    return fingerprint_venues(venue_manager)


def build_fingerprints():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        return {loop_level: run_and_fingerprint(loop_level, tmp_dir)
                for loop_level in LOOP_LEVELS}


def load_golden():
    with open(GOLDEN_PATH) as handle:
        return json.load(handle)
