"""Facts from earlier config blocks that later preflight checks depend on."""

from dataclasses import dataclass
from typing import Any, FrozenSet, Optional


@dataclass(frozen=True)
class PreflightContext:
    """Immutable. A fact is `None` when the block that supplies it failed, and the
    checks that depend on it are skipped, so one root cause gives one finding.

    `geo_levels` is the levels `Geography` parsed. `households` is the
    settings `HouseholdDistributor` parsed, not the raw block. `households_block_present`
    is whether a `households:` block is written at all, parsed or not."""
    geo_levels: Optional[FrozenSet[str]] = None
    households: Optional[Any] = None
    households_block_present: bool = False


def check_geo_level(key, geo_level, context, report):
    """A geo level, when set, must be a name; a known one once the geography is known."""
    if geo_level is None:
        return
    if not isinstance(geo_level, str):
        report.problem(key, f"must be a geo level name; got {geo_level!r}")
        return
    if context.geo_levels is not None and geo_level not in context.geo_levels:
        report.problem(key, f"unknown geo level {geo_level!r}; expected one of {sorted(context.geo_levels)}")
