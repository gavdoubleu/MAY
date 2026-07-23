"""Tier-1 bit-for-bit tests for geo_unit-partitioned parallel attribute
assignment, exercised through the public `AttributeAssigner.assign_all` entry
point (never the pool internals), so they survive the return-and-apply
restructuring.

The gate: 1 worker vs N workers over the *same* Slice partition must produce
identical assignments — mechanical correctness of the return-and-apply plumbing.
"""
import numpy as np

from may.attribute_assignment.assigner import AttributeAssigner
from may.attribute_assignment.assignment_config import AssignmentRule


# --- Compact real-object doubles ------------------------------------------

class _GeoUnit:
    def __init__(self, name):
        self.name = name
        self.level = "SGU"
        self.parent = None


class _Subset:
    def __init__(self, venue):
        self.subset_name = "Adults"
        self.venue = venue


class _Person:
    def __init__(self, person_id, geo_unit):
        self.id = person_id
        self.age = 30
        self.sex = "M"
        self.geographical_unit = geo_unit
        self.properties = {}
        self.activities = set()
        self.activity_map = {"residence": {"household": [_Subset(None)]}}


class _Population:
    def __init__(self, people):
        self._people = people

    def get_all_people(self):
        return self._people


class _GeoSource:
    """A weighted distribution keyed by residence geo_unit — the shape
    DrawStrategy ('probabilistic') samples from."""
    def __init__(self, weights_by_geo):
        self._weights_by_geo = weights_by_geo

    def lookup(self, person, household=None, context=None):
        geo_unit = person.geographical_unit.name
        return self._weights_by_geo[geo_unit]


class _DataManager:
    def __init__(self, sources):
        self._sources = sources

    def get_source(self, name):
        return self._sources.get(name)


class _PersonConfig:
    """Person-level config for a stochastic probabilistic attribute."""
    assignment_level = "person"

    def __init__(self):
        self.attribute_name = "shade"
        self.filters = {}
        self.required_attributes = {}
        self.settings = {}

    def get_person_assignment_rule(self):
        return AssignmentRule(
            role="person", priority=1, description="",
            assignment={"strategy": "probabilistic", "data_source": "geo_distribution"},
        )


def _build_population():
    """30 people spread across two geo_units (so >1 worker genuinely splits),
    with stable explicit ids used to seed each Slice's RNG stream."""
    geo_a, geo_b = _GeoUnit("E00000001"), _GeoUnit("E00000002")
    people = []
    for person_id in range(100, 130):
        geo_unit = geo_a if person_id % 2 == 0 else geo_b
        people.append(_Person(person_id, geo_unit))
    return _Population(people)


def _make_assigner():
    data_manager = _DataManager({
        "geo_distribution": _GeoSource({
            "E00000001": {"A": 0.5, "B": 0.5},
            "E00000002": {"A": 0.3, "B": 0.7},
        })
    })
    return AttributeAssigner(_PersonConfig(), data_manager)


def _run(workers):
    population = _build_population()
    assigner = _make_assigner()
    assigner.assign_all(venue_manager=None, population=population, workers=workers)
    return {person.id: person.properties.get("shade")
            for person in population.get_all_people()}


def test_tier1_person_assignment_identical_across_worker_count():
    single = _run(workers=1)
    many = _run(workers=4)

    # Every person assigned, and the assignment is independent of worker count.
    assert all(value is not None for value in single.values())
    assert single == many


# --- Residence-level (person_by_residence) Tier-1 --------------------------

from tests.test_unit.may.attribute_assignment.test_assigner_extended import (
    MinimalGeoUnit, MinimalPerson, MinimalVenue, MinimalVenueManager,
    SimpleGeoSource, SimpleDataManager, _build_family_config,
)


def _family_household(geo_unit):
    """A Family household (2 adults + 1 child) in a given geo_unit."""
    adults = [MinimalPerson(age=40, geographical_unit=geo_unit, category="Adults"),
              MinimalPerson(age=38, geographical_unit=geo_unit, category="Adults")]
    child = [MinimalPerson(age=8, geographical_unit=geo_unit, category="Kids")]
    household = MinimalVenue(venue_type="household", geographical_unit=geo_unit,
                            members=adults + child)
    household.properties["_cached_actual_pattern"] = "1 0 2 0"
    return household


def _build_residence_assigner_and_vm():
    # Rewind the shared id counters so both worker-count runs build an
    # identically-identified world — ids are what the comparison keys on.
    MinimalPerson._next_id = 7000
    MinimalVenue._next_id = 8000

    # Stochastic sources: primary adult drawn probabilistically, secondary via
    # partnership — so per-Slice seeding genuinely matters.
    geo_source = SimpleGeoSource(lookup_data={
        "E00000001": {"W": 0.5, "A": 0.5},
        "E00000002": {"W": 0.3, "A": 0.7},
    })
    pair_source = SimpleGeoSource(lookup_data={
        "E00000001": {"W": {"W": 0.6, "A": 0.4}, "A": {"W": 0.4, "A": 0.6}},
        "E00000002": {"W": {"W": 0.7, "A": 0.3}, "A": {"W": 0.2, "A": 0.8}},
    })
    config = _build_family_config(geo_source, pair_source)
    data_manager = SimpleDataManager(sources={
        "geo_distribution": geo_source, "pair_probabilities": pair_source})

    geo_a, geo_b = MinimalGeoUnit("E00000001"), MinimalGeoUnit("E00000002")
    households = ([_family_household(geo_a) for _ in range(4)]
                  + [_family_household(geo_b) for _ in range(4)])
    venue_manager = MinimalVenueManager(households)
    return AttributeAssigner(config, data_manager), venue_manager, households


def _run_residence(workers):
    assigner, venue_manager, households = _build_residence_assigner_and_vm()
    assigner.assign_all(venue_manager=venue_manager, workers=workers)
    return {member.id: member.properties.get("ethnicity")
            for household in households for member in household.get_all_members()}


def test_tier1_residence_assignment_identical_across_worker_count():
    single = _run_residence(workers=1)
    many = _run_residence(workers=4)

    assert all(value is not None for value in single.values())
    assert single == many
