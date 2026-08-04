"""Merging one step's stats across the loop units that ran it.

Under the geo_unit-outer loop a step runs once per unit, and the per-step
stats a caller sees are the fold of those runs. Keys that describe work done
add up; keys that describe the world as a whole cannot.
"""

from may.residence.allocation_strategy import _merge_step_stats


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
