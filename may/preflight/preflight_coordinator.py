"""Preflight coordinator: run every stage's config check and the cross-block checks."""

import logging
from typing import Optional

import yaml

from may import timeline
from may.geography.geography import Geography
from may.geography.venue_manager import VenueConfig, VenueManager
from may.population.population import PopulationManager
from may.preflight.context import PreflightContext
from may.preflight.report import PreflightError, PreflightReport, not_mapping_message
from may.residence.household_distributor import HouseholdDistributor
from may.serialization.preflight import venue_property_gaps
from may.utils import path_resolver as pr
from may.utils.yaml_loader import invalid_yaml_message, load_yaml, unreadable_file_message

logger = logging.getLogger("preflight")

SERIALIZATION_CONFIG_FILE_KEY = "serialization.config_file"


def check_world_config(config: dict) -> PreflightReport:
    """Check a whole world config without loading data. Pure: no logging, no raising."""
    block_report = PreflightReport()
    geography_block = _mapping_block(config, "geography", block_report)
    venues_block = _mapping_block(config, "venues", block_report)
    population_block = _mapping_block(config, "population", block_report)
    serialization_block = _mapping_block(config, "serialization", block_report)
    households_block = (
        _mapping_block(config, "households", block_report)
        if _has_residence_step(config)
        else None
    )
    geography_config = None
    venue_config = None
    household_config = None
    report = block_report
    if geography_block is not None:
        geography_config, geography_report = Geography.parse_config(geography_block)
        report = PreflightReport.merge(report, geography_report.under("geography", sep="."))
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
    loaded_schema = None
    if serialization_block is not None:
        schema_report = PreflightReport()
        loaded_schema = _load_serialization_schema(serialization_block, schema_report)
        report = PreflightReport.merge(report, schema_report)
    if venue_config is not None and loaded_schema is not None:
        report = PreflightReport.merge(
            report, _venue_property_gap_report(*loaded_schema, venue_config)
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


def _load_serialization_schema(
    serialization_block: dict, report: PreflightReport
) -> Optional[tuple]:
    """(path, parsed schema) if serialisation enabled with a readable config file, else `None`.

    A missing, unreadable, empty or non-mapping config file is a problem in `report`: export would
    fail after the build.
    """
    if not serialization_block.get("enabled", True):
        return None
    config_file = serialization_block.get("config_file")
    if not config_file:
        report.problem(SERIALIZATION_CONFIG_FILE_KEY, "not set; serialisation is enabled")
        return None
    path = pr.resolve(config_file)
    try:
        schema = load_yaml(path)
    except FileNotFoundError:
        report.problem(SERIALIZATION_CONFIG_FILE_KEY, f"{path} not found")
        return None
    except (OSError, UnicodeDecodeError) as error:
        report.problem(SERIALIZATION_CONFIG_FILE_KEY, unreadable_file_message(path, error))
        return None
    except yaml.YAMLError as error:
        report.problem(SERIALIZATION_CONFIG_FILE_KEY, invalid_yaml_message(path, error))
        return None
    # SerializationConfig rejects an empty schema, so export would fail after the build.
    if not schema:
        report.problem(SERIALIZATION_CONFIG_FILE_KEY, f"{path} is empty")
        return None
    if not isinstance(schema, dict):
        report.problem(f"{SERIALIZATION_CONFIG_FILE_KEY} ({path})", not_mapping_message(schema))
        return None
    return path, schema


def _venue_property_gap_report(path: str, schema, venue_config: VenueConfig) -> PreflightReport:
    report = PreflightReport()
    file_key = f"{SERIALIZATION_CONFIG_FILE_KEY} ({path})"
    try:
        gaps = venue_property_gaps(schema, venue_config.venue_types, venue_config.data_dir)
    except Exception as error:
        # Advisory check: a malformed schema or unreadable venue file must not
        # stop preflight, but the user is told the check did not run.
        report.warning(file_key, f"venue property check skipped: {error}")
        return report
    for venue_type, property_name, columns in gaps:
        report.warning(
            f"{file_key}: venues.types.{venue_type}.properties",
            f"{property_name!r} not in the venue files; they carry {columns}",
        )
    return report
