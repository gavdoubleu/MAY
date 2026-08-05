"""Per-run flush counters emitted by residence allocation.

Choosing how much membership to hold live rather than write is a tuning
exercise, and it needs numbers: how many staging calls a run made, how many
members they carried, and how long they took. One summary line at the end
of allocation carries them, in the shape `collect_results.py` already
regex-parses out of `[MEMORY] after <stage>: ...`.
"""

import logging
import re

import pytest

from may.residence.allocation_strategy import execute_allocation_strategy
from tests.test_integration.test_allocation_strategy_integration import (  # noqa: F401
    geography,
    hd,
    population_manager,
    venue_manager,
)
from tests.test_integration.test_pending_flush_buffer import (
    build_world,
    write_strategy,
)

FLUSH_FIELD_RE = re.compile(r"(\w+)=([\d.]+)")


def flush_fields(message):
    """The `key=value` pairs of a `[FLUSH]` line, ints where they are ints."""
    return {key: float(value) if "." in value else int(value)
            for key, value in FLUSH_FIELD_RE.findall(message)}


def summary_lines(caplog):
    return [message for message in caplog.messages
            if message.startswith("[FLUSH] calls=")]


def per_call_lines(caplog):
    return [message for message in caplog.messages
            if message.startswith("[FLUSH] call=")]


class TestFlushSummary:

    def test_summary_reports_the_totals_of_every_staging_call(
        self, hd, population_manager, venue_manager, geography,  # noqa: F811
        tmp_path, caplog
    ):
        """One line per run, and its counters are the run's own per-call
        counters aggregated — so a sweep can fit call cost against member
        cost from the summary alone."""
        world = build_world(population_manager, venue_manager, geography, tmp_path)
        strategy_file = write_strategy(tmp_path, loop_level="SGU",
                                       pending_flush_member_bound=1)

        with caplog.at_level(logging.DEBUG, logger="allocation_strategy"):
            execute_allocation_strategy(
                population=population_manager,
                venues=venue_manager,
                household_distributor=hd,
                strategy_file=strategy_file,
                subset_flusher=world.flush_subsets_batch,
            )

        summaries = summary_lines(caplog)
        assert len(summaries) == 1
        summary = flush_fields(summaries[0])

        per_call = [flush_fields(message) for message in per_call_lines(caplog)]
        assert per_call, "the stress world should stage at least one batch"
        assert summary["calls"] == len(per_call)
        assert summary["members"] == sum(call["members"] for call in per_call)
        assert summary["max_batch_members"] == max(call["members"]
                                                   for call in per_call)
        assert summary["seconds"] >= 0

    def test_members_counted_are_the_members_staged(
        self, hd, population_manager, venue_manager, geography,  # noqa: F811
        tmp_path, caplog
    ):
        """Every allocated person sits in exactly one household subset, so a
        run that flushes every unit stages each of them exactly once."""
        world = build_world(population_manager, venue_manager, geography, tmp_path)
        strategy_file = write_strategy(tmp_path, loop_level="SGU",
                                       pending_flush_member_bound=1)

        with caplog.at_level(logging.INFO, logger="allocation_strategy"):
            execute_allocation_strategy(
                population=population_manager,
                venues=venue_manager,
                household_distributor=hd,
                strategy_file=strategy_file,
                subset_flusher=world.flush_subsets_batch,
            )

        summary = flush_fields(summary_lines(caplog)[0])
        assert summary["members"] == len(hd.allocated_people)

    def test_calls_counts_the_units_that_staged_something(
        self, hd, population_manager, venue_manager, geography,  # noqa: F811
        tmp_path, caplog
    ):
        """Units that created no household make no call, so the count is of
        staging calls rather than of loop units."""
        world = build_world(population_manager, venue_manager, geography, tmp_path)
        strategy_file = write_strategy(tmp_path, loop_level="SGU",
                                       pending_flush_member_bound=1)

        with caplog.at_level(logging.INFO, logger="allocation_strategy"):
            execute_allocation_strategy(
                population=population_manager,
                venues=venue_manager,
                household_distributor=hd,
                strategy_file=strategy_file,
                subset_flusher=world.flush_subsets_batch,
            )

        flushed_units = [message for message in caplog.messages
                         if "to the pending-flush buffer" in message]
        summary = flush_fields(summary_lines(caplog)[0])
        assert summary["calls"] == len(flushed_units)

    def test_per_call_counters_are_debug_only(
        self, hd, population_manager, venue_manager, geography,  # noqa: F811
        tmp_path, caplog
    ):
        """The summary is for every run; the per-call detail is for someone
        already debugging, and would be one line per unit on England."""
        world = build_world(population_manager, venue_manager, geography, tmp_path)
        strategy_file = write_strategy(tmp_path, loop_level="SGU",
                                       pending_flush_member_bound=1)

        with caplog.at_level(logging.INFO, logger="allocation_strategy"):
            execute_allocation_strategy(
                population=population_manager,
                venues=venue_manager,
                household_distributor=hd,
                strategy_file=strategy_file,
                subset_flusher=world.flush_subsets_batch,
            )

        assert summary_lines(caplog)
        assert not per_call_lines(caplog)

    def test_a_flusher_that_is_never_called_still_reports_zero(
        self, hd, population_manager, venue_manager, geography,  # noqa: F811
        tmp_path, caplog
    ):
        """A run that deliberately never flushes is the measurement control
        the sweep is compared against, so it needs its own row."""
        world = build_world(population_manager, venue_manager, geography, tmp_path)
        strategy_file = write_strategy(tmp_path, loop_level="SGU")

        with caplog.at_level(logging.INFO, logger="allocation_strategy"):
            execute_allocation_strategy(
                population=population_manager,
                venues=venue_manager,
                household_distributor=hd,
                strategy_file=strategy_file,
                subset_flusher=world.flush_subsets_batch,
            )

        summary = flush_fields(summary_lines(caplog)[0])
        assert summary["calls"] == 0
        assert summary["members"] == 0
        assert summary["max_batch_members"] == 0

    def test_no_flusher_means_no_flush_line(
        self, hd, population_manager, venue_manager,  # noqa: F811
        tmp_path, caplog
    ):
        """Nothing could have been staged, so there is nothing to report."""
        strategy_file = write_strategy(tmp_path, loop_level="SGU")

        with caplog.at_level(logging.DEBUG, logger="allocation_strategy"):
            execute_allocation_strategy(
                population=population_manager,
                venues=venue_manager,
                household_distributor=hd,
                strategy_file=strategy_file,
            )

        assert not [message for message in caplog.messages
                    if message.startswith("[FLUSH]")]
