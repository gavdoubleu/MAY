"""
The four call sites must agree on what a strategy's `_fail` means.

A `_fail` is a **Refusal** (see CONTEXT.md): the engine's verdict that a named
person cannot be assigned from the data given. It is counted and deferred to
the end-of-attribute raise. Anything else escaping a strategy is an **engine
fault** and aborts where it happened.

Covers, for `_assign_all_people_batch`, `_assign_all_people_sequential`,
`_assign_household` and `_assign_other_residences`:
- the same unassignable person produces the same outcome at every call site
- a Refusal never stops the people queued behind it
- an engine fault is never absorbed into a counter
- a Refusal is never written into `person.properties`

See docs/adr/0008-a-refusal-is-a-per-person-result-an-engine-fault-is-an-abort.md
"""
import numpy as np
import pytest

from may.attribute_assignment.assigner import AttributeAssigner, AttributeAssignmentError
from may.attribute_assignment.strategies import AssignmentStrategy

from tests.test_unit.may.attribute_assignment.test_assigner import (
    MinimalConfig,
    MinimalGeoUnit,
    MinimalPerson,
    MinimalVenue,
    MinimalVenueManager,
    SimpleDataManager,
    SimpleGeoSource,
)
from tests.test_unit.may.attribute_assignment.test_assigner_extended import (
    _build_family_config,
)


@pytest.fixture(autouse=True)
def reset_ids():
    MinimalPerson._next_id = 3000
    MinimalVenue._next_id = 5000


@pytest.fixture
def geo_unit():
    return MinimalGeoUnit("E00001234")


class RefusingStrategy(AssignmentStrategy):
    """Assigns `value`, except to the people whose position is in `refuse_at`,
    who are refused through the shared `_fail` primitive.

    Defines both `assign` and `assign_batch` off one refusal rule, so the two
    paths are being asked the identical question about the identical people.
    """

    def __init__(self, refuse_at=(), value="W"):
        super().__init__({'strategy': 'refusing_test'}, SimpleDataManager())
        self.refuse_at = set(refuse_at)
        self.value = value

    def _verdict(self, index, person):
        if index in self.refuse_at:
            return self._fail(person, "test refusal: no distribution for this person")
        return self.value

    def assign(self, person, household, context):
        return self._verdict(self._seen(person), person)

    def assign_batch(self, people, households, contexts):
        return [self._verdict(index, person)
                for index, person in enumerate(people)]

    def _seen(self, person):
        """Position of this person in the order they were first handed over."""
        if not hasattr(self, '_order'):
            self._order = []
        if person.id not in self._order:
            self._order.append(person.id)
        return self._order.index(person.id)


class BrokenStrategy(AssignmentStrategy):
    """An engine fault: a lookup the strategy author got wrong. Not a Refusal —
    nothing has been decided about the person, the engine is simply broken."""

    def __init__(self):
        super().__init__({'strategy': 'broken_test'}, SimpleDataManager())

    def assign(self, person, household, context):
        raise KeyError("column 'ethnic_group' missing from the loaded frame")

    def assign_batch(self, people, households, contexts):
        raise KeyError("column 'ethnic_group' missing from the loaded frame")


def _make_people(count, geo, venue=None):
    venue = venue or MinimalVenue(venue_type="household", geographical_unit=geo)
    return [MinimalPerson(age=30, geographical_unit=geo, residence_venue=venue)
            for _ in range(count)]


class TestCallSitesReportTheSameOutcome:
    """The divergence the audit found: same signal, four outcomes."""

    def test_batch_and_sequential_report_the_same_refusal_count(self, geo_unit):
        config = MinimalConfig(attribute_name="ethnicity")
        data_manager = SimpleDataManager()

        batch_assigner = AttributeAssigner(config, data_manager)
        batch_people = _make_people(3, geo_unit)
        batch_assigner._assign_all_people_batch(
            batch_people, RefusingStrategy(refuse_at=[1]))

        sequential_assigner = AttributeAssigner(config, data_manager)
        sequential_people = _make_people(3, geo_unit)
        np.random.seed(42)
        sequential_assigner._assign_all_people_sequential(
            sequential_people, RefusingStrategy(refuse_at=[1]))

        assert batch_assigner.stats['refusals'] == 1
        assert sequential_assigner.stats['refusals'] == 1
        assert batch_assigner.stats['assigned_people'] == 2
        assert sequential_assigner.stats['assigned_people'] == 2

    def test_a_refusal_does_not_stop_the_people_behind_it_in_a_batch(self, geo_unit):
        """One refused person must not cost the rest of the batch their
        assignment — the batch aborts the run today."""
        assigner = AttributeAssigner(MinimalConfig(attribute_name="ethnicity"),
                                     SimpleDataManager())
        people = _make_people(3, geo_unit)

        assigner._assign_all_people_batch(people, RefusingStrategy(refuse_at=[0]))

        assert 'ethnicity' not in people[0].properties
        assert people[1].properties['ethnicity'] == 'W'
        assert people[2].properties['ethnicity'] == 'W'

    def test_venue_site_reports_the_same_refusal_count_as_sequential(self, geo_unit):
        """`_assign_other_residences` must count refusals the way the person-level
        sites do, or the end-of-attribute completeness check never sees them."""
        config = MinimalConfig(
            attribute_name="ethnicity",
            venue_assignment_rules=[
                {'venue_types': ['care_home'],
                 'assignment': {'strategy': 'refusing_test'}}
            ],
        )
        assigner = AttributeAssigner(config, SimpleDataManager())
        strategy = RefusingStrategy(refuse_at=[1])
        assigner._get_or_create_strategy = lambda assignment_config: strategy

        residents = _make_people(3, geo_unit)
        venue = MinimalVenue(venue_type="care_home", members=residents,
                             geographical_unit=geo_unit)

        assigner._assign_other_residences([venue])

        assert assigner.stats['refusals'] == 1

    def test_household_site_reports_the_refusal(self, geo_unit):
        config = _build_family_config(SimpleGeoSource(fallback={"W": 1.0}))
        data_manager = SimpleDataManager(sources={
            "geo_distribution": SimpleGeoSource(fallback={"W": 1.0}),
            "pair_probabilities": SimpleGeoSource(),
        })
        assigner = AttributeAssigner(config, data_manager)
        strategy = RefusingStrategy(refuse_at=[0])
        assigner._get_or_create_strategy = lambda assignment_config: strategy

        adult = MinimalPerson(age=35, category="Adults", geographical_unit=geo_unit)
        household = MinimalVenue(venue_type="household", members=[adult],
                                 geographical_unit=geo_unit)
        household.properties["_cached_actual_pattern"] = "0 0 1 0"

        assigner._assign_household(household)

        assert assigner.stats['refusals'] == 1
        assert 'ethnicity' not in adult.properties


class TestRefusalDoesNotAbandonTheVenue:
    """The silent hole: the venue-wide `try` abandons every member queued
    behind the refused one, counting none of them."""

    def _venue_assigner(self, geo_unit, strategy, members):
        config = MinimalConfig(
            attribute_name="ethnicity",
            venue_assignment_rules=[
                {'venue_types': ['care_home'],
                 'assignment': {'strategy': 'refusing_test'}}
            ],
        )
        assigner = AttributeAssigner(config, SimpleDataManager())
        assigner._get_or_create_strategy = lambda assignment_config: strategy
        venue = MinimalVenue(venue_type="care_home", members=members,
                             geographical_unit=geo_unit)
        return assigner, venue

    def test_later_members_are_still_assigned(self, geo_unit):
        residents = _make_people(4, geo_unit)
        assigner, venue = self._venue_assigner(
            geo_unit, RefusingStrategy(refuse_at=[1]), residents)

        assigner._assign_other_residences([venue])

        assert residents[0].properties['ethnicity'] == 'W'
        assert 'ethnicity' not in residents[1].properties
        assert residents[2].properties['ethnicity'] == 'W'
        assert residents[3].properties['ethnicity'] == 'W'

    def test_abandoned_members_are_never_uncounted(self, geo_unit):
        """Whatever happens to them, they must not vanish: every eligible
        person is either assigned or refused."""
        residents = _make_people(4, geo_unit)
        assigner, venue = self._venue_assigner(
            geo_unit, RefusingStrategy(refuse_at=[1]), residents)

        assigner._assign_other_residences([venue])

        assigned = sum(1 for person in residents
                       if 'ethnicity' in person.properties)
        assert assigned + assigner.stats['refusals'] == len(residents)

    def test_venue_type_with_no_rule_refuses_each_member(self, geo_unit):
        """Scope belongs in the eligibility filters, not in a silent `continue`
        over a venue type nobody wrote a rule for."""
        config = MinimalConfig(
            attribute_name="ethnicity",
            venue_assignment_rules=[
                {'venue_types': ['care_home'],
                 'assignment': {'strategy': 'constant', 'value': 'W'}}
            ],
        )
        assigner = AttributeAssigner(config, SimpleDataManager())
        residents = _make_people(3, geo_unit)
        venue = MinimalVenue(venue_type="student_dorms", members=residents,
                             geographical_unit=geo_unit)

        assigner._assign_other_residences([venue])

        assert assigner.stats['refusals'] == 3


class TestEngineFaultAborts:
    """A broken engine is not a data verdict and must never be absorbed into
    a counter."""

    def test_sequential_aborts_unwrapped(self, geo_unit):
        assigner = AttributeAssigner(MinimalConfig(attribute_name="ethnicity"),
                                     SimpleDataManager())
        np.random.seed(42)
        with pytest.raises(KeyError):
            assigner._assign_all_people_sequential(
                _make_people(3, geo_unit), BrokenStrategy())

    def test_batch_aborts_unwrapped(self, geo_unit):
        """Today the batch site wraps everything as AttributeAssignmentError,
        which loses the distinction between a bug and a refusal."""
        assigner = AttributeAssigner(MinimalConfig(attribute_name="ethnicity"),
                                     SimpleDataManager())
        with pytest.raises(KeyError):
            assigner._assign_all_people_batch(
                _make_people(3, geo_unit), BrokenStrategy())

    def test_venue_site_aborts_unwrapped(self, geo_unit):
        config = MinimalConfig(
            attribute_name="ethnicity",
            venue_assignment_rules=[
                {'venue_types': ['care_home'],
                 'assignment': {'strategy': 'broken_test'}}
            ],
        )
        assigner = AttributeAssigner(config, SimpleDataManager())
        assigner._get_or_create_strategy = lambda assignment_config: BrokenStrategy()
        venue = MinimalVenue(venue_type="care_home",
                             members=_make_people(2, geo_unit),
                             geographical_unit=geo_unit)

        with pytest.raises(KeyError):
            assigner._assign_other_residences([venue])

    def test_household_site_aborts_unwrapped(self, geo_unit):
        config = _build_family_config(SimpleGeoSource(fallback={"W": 1.0}))
        assigner = AttributeAssigner(config, SimpleDataManager())
        assigner._get_or_create_strategy = lambda assignment_config: BrokenStrategy()

        adult = MinimalPerson(age=35, category="Adults", geographical_unit=geo_unit)
        household = MinimalVenue(venue_type="household", members=[adult],
                                 geographical_unit=geo_unit)
        household.properties["_cached_actual_pattern"] = "0 0 1 0"

        with pytest.raises(KeyError):
            assigner._assign_household(household)

    def test_unconstructable_strategy_aborts_the_venue_run(self, geo_unit):
        """No strategy means no member of that venue could ever be assigned,
        so there is no per-person verdict to give."""
        config = MinimalConfig(
            attribute_name="ethnicity",
            venue_assignment_rules=[
                {'venue_types': ['care_home'],
                 'assignment': {'strategy': 'no_such_strategy'}}
            ],
        )
        assigner = AttributeAssigner(config, SimpleDataManager())
        venue = MinimalVenue(venue_type="care_home",
                             members=_make_people(2, geo_unit),
                             geographical_unit=geo_unit)

        with pytest.raises(Exception) as caught:
            assigner._assign_other_residences([venue])
        assert not isinstance(caught.value, AttributeAssignmentError)


class TestRefusalReachesTheCompletenessCheck:
    """A Refusal is only useful if it eventually stops the run."""

    def test_refusals_from_a_venue_raise_at_end_of_attribute(self, geo_unit):
        config = MinimalConfig(
            attribute_name="ethnicity",
            venue_assignment_rules=[
                {'venue_types': ['care_home'],
                 'assignment': {'strategy': 'refusing_test'}}
            ],
        )
        assigner = AttributeAssigner(config, SimpleDataManager())
        strategy = RefusingStrategy(refuse_at=[0])
        assigner._get_or_create_strategy = lambda assignment_config: strategy

        venue = MinimalVenue(venue_type="care_home",
                             members=_make_people(2, geo_unit),
                             geographical_unit=geo_unit)

        with pytest.raises(AttributeAssignmentError):
            assigner.assign_all(MinimalVenueManager([venue]))

    def test_a_refusal_is_never_written_into_properties(self, geo_unit):
        """`venue_distributor` reads flags by truthiness, so a stored Refusal
        object would silently change how that person is treated downstream."""
        assigner = AttributeAssigner(MinimalConfig(attribute_name="ethnicity"),
                                     SimpleDataManager())
        people = _make_people(2, geo_unit)

        assigner._assign_all_people_batch(people, RefusingStrategy(refuse_at=[0]))

        assert people[0].properties == {}
