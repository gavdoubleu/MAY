"""
`assign_batch` hands back one verdict per person: a value or a `Refusal`.

An exception cannot carry a verdict for each of *n* people without becoming a
container, so the batch path returns refusals rather than raising them. Two
consequences are pinned here:

- one unassignable person costs only that person their assignment;
- a refusal names *the person it applies to*. `DrawStrategy` draws once per
  group of people sharing a distribution, so an empty distribution refuses
  every member of that group, not the member who happened to be sampled first.

The call sites that consume these returns are a later phase; this file asks the
strategies directly.

See docs/adr/0008-a-refusal-is-a-per-person-result-an-engine-fault-is-an-abort.md
"""
import pytest

from may.attribute_assignment.strategies import (
    CategoricalSamplerStrategy,
    CommutingLikelihoodStrategy,
    ConstantStrategy,
    Refusal,
)

from tests.test_unit.may.attribute_assignment.test_strategies_extended import (
    MinimalGeoUnit,
    MinimalPerson,
    MinimalVenue,
    MultiKeySource,
    ODMatrixSource,
    SimpleDataManager,
)


class PerPersonSource:
    """Returns a different distribution per person, so a batch splits into
    groups the strategy did not choose."""

    def __init__(self, by_person_id):
        self._by_person_id = by_person_id

    def lookup(self, person, household=None, context=None):
        return self._by_person_id[person.id]


class RedistributingODSource(ODMatrixSource):
    """An O-D source that declares a share of each origin's workers as bounced
    back in-boundary, which is what makes the batch attach the flag."""

    def __init__(self, redistributed_flag, redistributed_fraction, **kwargs):
        super().__init__(**kwargs)
        self._redistributed_flag = redistributed_flag
        self._redistributed_fraction = redistributed_fraction


@pytest.fixture(autouse=True)
def reset_person_ids():
    MinimalPerson._next_id = 7000


def _draw_strategy(source):
    return CategoricalSamplerStrategy(
        {"strategy": "categorical_sampler", "data_source": "probs"},
        SimpleDataManager(sources={"probs": source}),
    )


def _draw_batch(strategy, people):
    return strategy.assign_batch(people, [MinimalVenue()] * len(people),
                                 [{"attribute_name": "sector"}] * len(people))


class TestDrawStrategyRefusesEveryMemberOfTheGroup:
    """The group is a sampling optimisation, not a unit of judgement."""

    def test_an_empty_distribution_refuses_each_person_by_name(self):
        people = [MinimalPerson() for _ in range(3)]
        strategy = _draw_strategy(MultiKeySource(return_value={}))

        results = _draw_batch(strategy, people)

        assert all(isinstance(result, Refusal) for result in results)
        assert [result.person_id for result in results] == [p.id for p in people]

    def test_a_zero_total_distribution_refuses_each_person_by_name(self):
        people = [MinimalPerson() for _ in range(3)]
        strategy = _draw_strategy(MultiKeySource(return_value={"A": 0.0,
                                                              "B": 0.0}))

        results = _draw_batch(strategy, people)

        assert [result.person_id for result in results] == [p.id for p in people]
        assert all("zero total weight" in result.reason for result in results)

    def test_a_refused_group_does_not_cost_the_other_groups_their_draw(self):
        assignable = [MinimalPerson() for _ in range(2)]
        refused = [MinimalPerson() for _ in range(2)]
        strategy = _draw_strategy(PerPersonSource(
            {**{person.id: {"X": 1.0} for person in assignable},
             **{person.id: {} for person in refused}}
        ))

        results = _draw_batch(strategy, assignable + refused)

        assert results[:2] == ["X", "X"]
        assert all(isinstance(result, Refusal) for result in results[2:])

    def test_a_refusal_names_the_strategy_that_declined(self):
        strategy = _draw_strategy(MultiKeySource(return_value={}))

        results = _draw_batch(strategy, [MinimalPerson()])

        assert results[0].refused_by == "Strategy 'categorical_sampler'"

    def test_every_slot_is_filled(self):
        """A surviving `None` is an absence, which is precisely what a Refusal
        exists to replace."""
        people = [MinimalPerson() for _ in range(4)]
        strategy = _draw_strategy(PerPersonSource(
            {people[0].id: {"X": 1.0}, people[1].id: {},
             people[2].id: {"Y": 1.0}, people[3].id: {"A": 0.0}}
        ))

        assert all(result is not None for result in _draw_batch(strategy, people))


class TestCommutingBatchReturnsRefusals:

    OUTPUTS = {"workplace_location": "destination"}

    def _strategy(self, source):
        return CommutingLikelihoodStrategy(
            {"strategy": "commuting_likelihood", "data_source": "commuting_flows",
             "outputs": self.OUTPUTS},
            SimpleDataManager(sources={"commuting_flows": source}),
        )

    def _batch(self, strategy, people):
        return strategy.assign_batch(
            people, [MinimalVenue()] * len(people),
            [{"attribute_name": "workplace_location"}] * len(people))

    def test_an_unresolvable_origin_refuses_only_that_person(self):
        strategy = self._strategy(ODMatrixSource(
            lookup_data={"ORIGIN_A": [("DEST_1", {}, 1.0)]}))
        resolvable = MinimalPerson(geographical_unit=MinimalGeoUnit("ORIGIN_A"))
        unresolvable = MinimalPerson(geographical_unit=None)

        results = self._batch(strategy, [resolvable, unresolvable, resolvable])

        assert results[0] == "DEST_1"
        assert results[2] == "DEST_1"
        assert isinstance(results[1], Refusal)
        assert results[1].person_id == unresolvable.id
        assert "geographical_unit" in results[1].reason

    def test_an_origin_with_no_flow_row_refuses_each_of_its_people(self):
        strategy = self._strategy(ODMatrixSource(
            lookup_data={"ORIGIN_A": [("DEST_1", {}, 1.0)]}))
        assignable = MinimalPerson(geographical_unit=MinimalGeoUnit("ORIGIN_A"))
        stranded = [MinimalPerson(geographical_unit=MinimalGeoUnit("ORIGIN_B"))
                    for _ in range(2)]

        results = self._batch(strategy, [assignable] + stranded)

        assert results[0] == "DEST_1"
        assert [result.person_id for result in results[1:]] == \
            [person.id for person in stranded]
        assert all("no commuting-flow row" in result.reason
                   for result in results[1:])

    def test_a_missing_data_source_refuses_every_person(self):
        strategy = CommutingLikelihoodStrategy(
            {"strategy": "commuting_likelihood", "data_source": "commuting_flows",
             "outputs": self.OUTPUTS},
            SimpleDataManager(sources={}),
        )
        people = [MinimalPerson(geographical_unit=MinimalGeoUnit("ORIGIN_A"))
                  for _ in range(2)]

        results = self._batch(strategy, people)

        assert [result.person_id for result in results] == [p.id for p in people]
        assert all("commuting_flows" in result.reason for result in results)

    def test_a_refusal_is_never_promoted_by_the_redistributed_flag(self):
        """`_with_redistributed_flag` promotes a scalar result to a dict, and
        `venue_distributor` reads that flag by truthiness — so a flagged
        Refusal would change where a real person ends up."""
        strategy = self._strategy(RedistributingODSource(
            redistributed_flag="was_redistributed",
            redistributed_fraction={"ORIGIN_A": 1.0, "ORIGIN_B": 1.0},
            lookup_data={"ORIGIN_A": [("DEST_1", {}, 1.0)]},
        ))
        assignable = MinimalPerson(geographical_unit=MinimalGeoUnit("ORIGIN_A"))
        stranded = MinimalPerson(geographical_unit=MinimalGeoUnit("ORIGIN_B"))

        results = self._batch(strategy, [assignable, stranded])

        assert results[0] == {"workplace_location": "DEST_1",
                              "was_redistributed": True}
        assert isinstance(results[1], Refusal)

    def test_the_out_of_boundary_headcount_excludes_refusals(self, caplog):
        """Refused people were never offered a destination, so counting them in
        the denominator understates how many of the assigned drew the sentinel."""
        strategy = self._strategy(ODMatrixSource(
            lookup_data={"ORIGIN_A": [("OUTSIDE", {}, 1.0)]},
            out_of_boundary="outside", outside_value="OUTSIDE"))
        people = [MinimalPerson(geographical_unit=MinimalGeoUnit("ORIGIN_A"))
                  for _ in range(2)]
        people.append(MinimalPerson(geographical_unit=MinimalGeoUnit("ORIGIN_B")))

        with caplog.at_level("INFO"):
            self._batch(strategy, people)

        assert "2/2 people assigned the 'OUTSIDE' sentinel" in caplog.text


class TestConstantBatchReturnsRefusals:

    def test_no_configured_value_refuses_each_person(self):
        strategy = ConstantStrategy({"strategy": "constant"}, SimpleDataManager())
        people = [MinimalPerson() for _ in range(3)]

        results = strategy.assign_batch(people, [MinimalVenue()] * 3,
                                        [{"attribute_name": "attr"}] * 3)

        assert [result.person_id for result in results] == [p.id for p in people]
        assert all("constant strategy has no 'value'" in result.reason
                   for result in results)
