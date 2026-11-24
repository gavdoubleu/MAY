"""Facts from earlier config blocks that later preflight checks depend on."""

from dataclasses import dataclass
from typing import Any, FrozenSet, Optional


@dataclass(frozen=True)
class PreflightContext:
    """Immutable. A fact is `None` when the block that supplies it failed, and the
    checks that depend on it are skipped, so one root cause gives one finding.

    `geo_levels` stays `None` until a geography preflight exists. `households` is the
    settings `HouseholdDistributor` parsed, not the raw block. `households_block_present`
    is whether a `households:` block is written at all, parsed or not."""
    geo_levels: Optional[FrozenSet[str]] = None
    households: Optional[Any] = None
    households_block_present: bool = False
