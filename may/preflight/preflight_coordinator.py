"""Preflight coordinator: run every stage's config check in build order."""

import logging
from typing import Optional

from may import timeline
from may.geography.geography import Geography
from may.geography.venue_manager import VenueManager
from may.population.population import PopulationManager
from may.preflight.context import ConfigContext, PreflightContext
from may.preflight.report import PreflightError, PreflightReport, not_mapping_message
from may.residence.household_distributor import HouseholdDistributor
from may.serialization.world_serializer import WorldSerializer

logger = logging.getLogger("preflight")


def check_world_config(config: dict) -> PreflightReport:
    """Check a whole world config without loading data. Pure: no logging, no raising."""
    block_report = PreflightReport()
    households_block = (
        _mapping_block(config, "households", block_report)
        if _has_residence_step(config)
        else None
    )
    household_config = None
    # Temporary adapter until the coordinator is a context-driven driver (phase 11).
    context = ConfigContext()
    geography_config, geography_report = Geography.preflight(config.get("geography", {}), context)
    context = context.with_result(Geography, geography_config)
    population_config, population_report = PopulationManager.preflight(config.get("population", {}), context)
    context = context.with_result(PopulationManager, population_config)
    venue_config, venue_report = VenueManager.preflight(config.get("venues", {}), context)
    context = context.with_result(VenueManager, venue_config)
    _, serialisation_report = WorldSerializer.preflight(config.get("serialization", {}), context)
    report = PreflightReport.merge(
        block_report, geography_report, population_report, venue_report, serialisation_report
    )
    geo_levels = frozenset(geography_config.levels) if geography_config is not None else None
    if households_block is not None:
        household_config, household_report = HouseholdDistributor.parse_config(households_block)
        report = PreflightReport.merge(
            report, household_report.under("households", sep=".")
        )
    timeline_context = PreflightContext(
        geo_levels=geo_levels,
        households=household_config,
        households_block_present=bool(config.get("households")),
    )
    return PreflightReport.merge(report, timeline.preflight(config.get("timeline"), timeline_context))


def run_preflight(config: dict) -> PreflightReport:
    """Check a world config, log the report at INFO; `PreflightError` if any problem."""
    report = check_world_config(config)
    logger.info(report.format())
    if report.has_problems:
        raise PreflightError(report)
    return report


def _has_residence_step(config: dict) -> bool:
    """Whether the timeline allocates residences, the only reader of `households`.

    A malformed timeline is `False`; the timeline preflight reports it."""
    timeline_block = config.get("timeline")
    steps = timeline_block.get("steps") if isinstance(timeline_block, dict) else None
    return isinstance(steps, list) and any(
        isinstance(step, dict) and step.get("type") == "residence_allocation"
        for step in steps
    )


def _mapping_block(config: dict, name: str, report: PreflightReport) -> Optional[dict]:
    """The `name` block, `{}` if absent or blank; `None` with a problem if not a mapping."""
    block = config.get(name)
    if block is None:
        return {}
    if not isinstance(block, dict):
        report.problem(name, not_mapping_message(block))
        return None
    return block

