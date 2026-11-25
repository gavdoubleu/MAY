"""Preflight coordinator: run every stage's config check and the cross-block checks."""

import logging
from typing import Optional

from may import timeline
from may.geography.geography import Geography
from may.geography.venue_manager import VenueConfig, VenueManager
from may.population.population import PopulationManager
from may.preflight.context import ConfigContext, PreflightContext
from may.preflight.report import PreflightError, PreflightReport, not_mapping_message
from may.residence.household_distributor import HouseholdDistributor
from may.preflight.venue_property_gaps import venue_property_gaps
from may.serialization.world_serializer import SerialisationSettings, WorldSerializer

logger = logging.getLogger("preflight")


def check_world_config(config: dict) -> PreflightReport:
    """Check a whole world config without loading data. Pure: no logging, no raising."""
    block_report = PreflightReport()
    venues_block = _mapping_block(config, "venues", block_report)
    population_block = _mapping_block(config, "population", block_report)
    serialization_block = _mapping_block(config, "serialization", block_report)
    households_block = (
        _mapping_block(config, "households", block_report)
        if _has_residence_step(config)
        else None
    )
    venue_config = None
    household_config = None
    # Temporary adapter until the coordinator is a context-driven driver (phase 11).
    geography_config, geography_report = Geography.preflight(config.get("geography", {}), ConfigContext())
    report = PreflightReport.merge(block_report, geography_report)
    geo_levels = frozenset(geography_config.levels) if geography_config is not None else None
    if population_block is not None:
        population_report = PopulationManager.preflight(population_block, geo_levels)
        report = PreflightReport.merge(report, population_report.under("population", sep="."))
    if venues_block is not None:
        venue_config, venue_report = VenueManager.parse_config(venues_block, geo_levels)
        report = PreflightReport.merge(report, venue_report.under("venues", sep="."))
    if households_block is not None:
        household_config, household_report = HouseholdDistributor.parse_config(households_block)
        report = PreflightReport.merge(
            report, household_report.under("households", sep=".")
        )
    serialisation_settings = None
    if serialization_block is not None:
        serialisation_settings, serialisation_report = WorldSerializer.parse_config(
            serialization_block
        )
        report = PreflightReport.merge(
            report, serialisation_report.under("serialization", sep=".")
        )
    if (
        venue_config is not None
        and serialisation_settings is not None
        and serialisation_settings.enabled
    ):
        report = PreflightReport.merge(
            report, _venue_property_gap_report(serialisation_settings, venue_config)
        )
    context = PreflightContext(
        geo_levels=geo_levels,
        households=household_config,
        households_block_present=bool(config.get("households")),
    )
    return PreflightReport.merge(report, timeline.preflight(config.get("timeline"), context))


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


def _venue_property_gap_report(
    settings: SerialisationSettings, venue_config: VenueConfig
) -> PreflightReport:
    """Schema venue properties absent from the venue files, as warnings."""
    report = PreflightReport()
    file_key = f"serialization.config_file ({settings.schema_path})"
    try:
        gaps = venue_property_gaps(settings.schema, venue_config)
    except Exception as error:
        # Advisory check: an unreadable venue file must not
        # stop preflight, but the user is told the check did not run.
        report.warning(file_key, f"venue property check skipped: {error}")
        return report
    for venue_type, property_name, columns in gaps:
        report.warning(
            f"{file_key}: venues.types.{venue_type}.properties",
            f"{property_name!r} not in the venue files; they carry {columns}",
        )
    return report
