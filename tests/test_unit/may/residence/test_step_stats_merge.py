"""Merging one step's stats across the loop units that ran it.

Under the geo_unit-outer loop a step runs once per unit, and the per-step
stats a caller sees are the fold of those runs. Keys that describe work done
add up; keys that describe the world as a whole cannot.
"""

import logging

from may.residence.allocation_strategy import (
    _check_per_unit_counts_against_the_world, _merge_step_stats)


class TestPeopleCountsSumAcrossUnits:
    """`total_people_allocated` and `total_people_remaining` are per-unit
    counts, not world snapshots. Each unit reports its own scope, so the
    world-wide figure is the sum — keeping the last unit's value would
    describe one unit and label it the world.
    """

    def test_allocated_counts_add_up(self):
        accumulated = {}

        _merge_step_stats(accumulated, {'total_people_allocated': 30})
        _merge_step_stats(accumulated, {'total_people_allocated': 12})

        assert accumulated['total_people_allocated'] == 42

    def test_remaining_counts_add_up(self):
        """Each unit reports what is left in its own scope, so the leftovers
        of the world are the leftovers of its units added together."""
        accumulated = {}

        _merge_step_stats(accumulated, {'total_people_remaining': 5})
        _merge_step_stats(accumulated, {'total_people_remaining': 7})

        assert accumulated['total_people_remaining'] == 12

    def test_round_identity_still_takes_the_latest_value(self):
        """Only the people counts changed sides. Keys that name the step
        rather than count its work must not start accumulating."""
        accumulated = {}

        _merge_step_stats(accumulated, {'round_number': 2, 'round_name': 'A'})
        _merge_step_stats(accumulated, {'round_number': 2, 'round_name': 'A'})

        assert accumulated['round_number'] == 2
        assert accumulated['round_name'] == 'A'


class TestPerUnitCountsAreCheckedAgainstTheWorld:
    """The summed counts assume everything a unit allocates lies inside that
    unit's scope. Nothing enforces it — `mark_people_as_allocated` takes any
    person — so the end of the run compares the arithmetic against one exact
    whole-world scan and says so when they part company.
    """

    def test_silence_when_the_arithmetic_matches(self, caplog):
        all_stats = {'last step': {'step_number': 2, 'total_people_remaining': 3}}

        with caplog.at_level(logging.WARNING, logger='household'):
            _check_per_unit_counts_against_the_world(all_stats, 3)

        assert caplog.records == []

    def test_warns_when_the_arithmetic_disagrees(self, caplog):
        all_stats = {'last step': {'step_number': 2, 'total_people_remaining': 3}}

        with caplog.at_level(logging.WARNING, logger='household'):
            _check_per_unit_counts_against_the_world(all_stats, 5)

        assert len(caplog.records) == 1
        assert '3' in caplog.text and '5' in caplog.text

    def test_a_final_step_that_reports_no_counts_is_not_compared(self, caplog):
        """Venue steps carry no people counts. The last reporting step's total
        predates whatever ran after it, so comparing them would warn about a
        difference that is simply the run continuing."""
        all_stats = {
            'household step': {'step_number': 1, 'total_people_remaining': 3},
            'venue step': {'step_number': 2},
        }

        with caplog.at_level(logging.WARNING, logger='household'):
            _check_per_unit_counts_against_the_world(all_stats, 5)

        assert caplog.records == []
