"""`parallel.max_tasks_per_child` in the world config, read the same way by
both parallelised stages, sibling to `resolve_configured_workers`.

A forked worker is reused for every Slice it draws for the life of the pool;
nothing resets what it accumulated in between (`CONTEXT.md`'s "Worker world
copy"). Unset, that stays true — the default is unlimited, matching today's
behaviour until an operator opts in.
"""
import pytest

from may.residence.household_distributor import HouseholdError
from may.world import resolve_configured_max_tasks_per_child


def test_an_absent_parallel_block_leaves_recycling_unlimited():
    assert resolve_configured_max_tasks_per_child({}) is None
    assert resolve_configured_max_tasks_per_child({"parallel": {}}) is None


def test_a_configured_count_is_returned_for_both_stages():
    assert resolve_configured_max_tasks_per_child(
        {"parallel": {"max_tasks_per_child": 4}}) == 4


def test_a_max_tasks_per_child_that_is_not_a_positive_whole_number_raises():
    """`true` is rejected rather than read as 1, and zero rather than read as
    "no limit": a recycling cadence nobody can name is a config error, not a
    default to fall back to silently.
    """
    for max_tasks_per_child in (0, -1, 2.5, "many", True):
        with pytest.raises(HouseholdError, match="parallel.max_tasks_per_child"):
            resolve_configured_max_tasks_per_child(
                {"parallel": {"max_tasks_per_child": max_tasks_per_child}})
