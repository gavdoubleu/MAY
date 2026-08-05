"""One pool-size dial in the world config, read the same way by both
parallelised stages.

Phase 6. Worker count was reachable only as a Python argument, so a run
launched from a config file could not set it: residence allocation took its
own conservative default and attribute assignment took the whole machine, and
neither could be told otherwise from the sbatch that knew the allocation.
"""
import pytest

from may.residence.household_distributor import HouseholdError
from may.world import resolve_configured_workers


def test_an_absent_parallel_block_leaves_each_stage_on_its_own_default():
    """Unset means "unchanged", not "one" or "all". The two stages default
    differently on purpose — residence forks a whole world per worker and is
    held to one until its footprint is measured — and an operator who has not
    asked for anything should not have either default moved under them.
    """
    assert resolve_configured_workers({}) is None
    assert resolve_configured_workers({"parallel": {}}) is None


def test_a_configured_count_is_returned_for_both_stages():
    assert resolve_configured_workers({"parallel": {"workers": 8}}) == 8


def test_a_worker_count_that_is_not_a_positive_whole_number_raises():
    """`true` is rejected rather than read as 1, and zero rather than read as
    "no limit": a pool size nobody can name is a config error, not a default
    to fall back to silently.
    """
    for workers in (0, -1, 2.5, "many", True):
        with pytest.raises(HouseholdError, match="parallel.workers"):
            resolve_configured_workers({"parallel": {"workers": workers}})
