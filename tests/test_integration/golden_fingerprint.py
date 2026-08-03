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

Caveat: the stress strategy has no `venue` step, so this fingerprint does not
cover `venue_allocator`.
"""
import hashlib
import json
import os

from may.geography import Geography
from may.geography.venue_manager import VenueManager
from may.population.population import PopulationManager
from may.residence.allocation_strategy import execute_allocation_strategy
from may.residence.household_distributor import HouseholdDistributor

STRESS_DATA = "tests/test_data/stress_world"
STRATEGY_FILE = f"{STRESS_DATA}/households/test_allocation_strategy.yaml"
GOLDEN_PATH = f"{STRESS_DATA}/golden_fingerprint.json"

# SGU is the only level with more than one loop unit; LGU is the single-unit
# whole-world-scope control.
LOOP_LEVELS = ("SGU", "LGU")

RANDOM_SEED = 42


def _build_world():
    """A fresh stress world — every run must start from an unallocated one."""
    geography = Geography(data_dir=f"{STRESS_DATA}/geography",
                          levels=["SGU", "MGU", "LGU"])
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


def fingerprint_households(venue_manager):
    """sha256 over every household's identity, membership and pattern.

    Ordered by venue id, which is global creation order (`Venue._next_venue_id`
    is one monotonic counter), so a change in the order households are created
    changes the digest — that is the point.
    """
    digest = hashlib.sha256()
    households = sorted(venue_manager.get_venues_by_type("household"),
                        key=lambda venue: venue.id)
    for household in households:
        member_ids = sorted(person.id for person in household.get_all_members())
        digest.update("{}|{}|{}|{}\n".format(
            household.id,
            household.geographical_unit.name,
            household.properties.get('allocation_pattern', ''),
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
    return fingerprint_households(venue_manager)


def build_fingerprints():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        return {loop_level: run_and_fingerprint(loop_level, tmp_dir)
                for loop_level in LOOP_LEVELS}


def load_golden():
    with open(GOLDEN_PATH) as handle:
        return json.load(handle)
