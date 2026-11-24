"""
Serialisation schema: which properties and attributes the HDF5 export writes.

The schema is parsed once, before the build, so a malformed section is reported
up front rather than crashing the export after the build has been paid for.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import yaml

from may.preflight.report import PreflightReport, not_list_message, not_mapping_message
from may.utils import path_resolver as pr
from may.utils.yaml_loader import invalid_yaml_message, load_yaml, unreadable_file_message

logger = logging.getLogger("serialization_config")

KNOWN_SECTIONS = ("population", "geography", "venues", "relationships", "output")
VENUE_GLOBAL_SWITCHES = ("include_coordinates", "include_is_residence")


class SerialisationError(Exception):
    """The serialisation config or schema cannot be used; `.report` lists every problem."""


@dataclass(frozen=True)
class SerialisationConfig:
    """Parsed serialisation schema. Build with `parse_schema` or `from_file`."""

    population_properties: List[str] = field(default_factory=list)
    geography_include_coordinates: bool = True
    geography_properties: List[str] = field(default_factory=list)
    venue_global_settings: Dict = field(default_factory=dict)
    venue_type_properties: Dict[str, List[str]] = field(default_factory=dict)
    relationships: Dict = field(default_factory=dict)
    output_settings: Dict = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str) -> "SerialisationConfig":
        """Load and parse a schema file; `SerialisationError` listing every problem."""
        schema, report = load_schema_file(path)
        report.raise_if_problems(
            SerialisationError, f"Serialisation schema {path} cannot be used:"
        )
        for warning in report.warnings:
            logger.info(f"Serialisation schema warning: {warning.key}: {warning.message}")
        return schema


def load_schema_file(path: str) -> Tuple[Optional[SerialisationConfig], PreflightReport]:
    """Find (`find_config_file`), load and parse a schema file.

    File findings are keyed `""`; schema findings `"(<found path>): <section>"`.
    """
    report = PreflightReport()
    try:
        found_path = pr.find_config_file(path, None)
    except pr.ConfigFileNotFoundError as error:
        report.problem("", str(error))
        return None, report
    try:
        loaded = load_yaml(found_path)
    except yaml.YAMLError as error:
        report.problem("", invalid_yaml_message(found_path, error))
        return None, report
    except (OSError, UnicodeDecodeError) as error:
        report.problem("", unreadable_file_message(found_path, error))
        return None, report
    if not loaded:
        report.problem("", f"{found_path} is empty")
        return None, report
    schema, schema_report = parse_schema(loaded)
    return schema, PreflightReport.merge(report, schema_report.under(f"({found_path})"))


def parse_schema(schema) -> Tuple[Optional[SerialisationConfig], PreflightReport]:
    """Parse a loaded schema; keys relative to its root. Config is `None` if any problem."""
    report = PreflightReport()
    if not schema:
        report.problem("", "is empty")
        return None, report
    if not isinstance(schema, dict):
        report.problem("", not_mapping_message(schema))
        return None, report

    for section in schema:
        if section not in KNOWN_SECTIONS:
            report.warning(
                section, f"unknown section; ignored (known: {', '.join(KNOWN_SECTIONS)})"
            )

    population = _section(report, schema, "population", "population")
    geography = _section(report, schema, "geography", "geography")
    venues = _section(report, schema, "venues", "venues")
    relationships = _section(report, schema, "relationships", "relationships")
    output = _section(report, schema, "output", "output")

    population_properties = _names(report, population, "properties", "population")
    geography_properties = _names(report, geography, "properties", "geography")
    _switch(report, geography, "include_coordinates", "geography")

    venue_global = _section(report, venues, "global", "venues.global")
    for switch in VENUE_GLOBAL_SWITCHES:
        _switch(report, venue_global, switch, "venues.global")
    venue_types = _section(report, venues, "types", "venues.types")
    venue_type_properties = {}
    for venue_type in venue_types or {}:
        type_settings = _section(report, venue_types, venue_type, f"venues.types.{venue_type}")
        venue_type_properties[venue_type] = _names(
            report, type_settings, "properties", f"venues.types.{venue_type}"
        )

    _switch(report, relationships, "include_activity_map", "relationships")
    _switch(report, output, "include_metadata", "output")
    _names(report, output, "metadata", "output")
    if output and "compression_level" in output:
        level = output["compression_level"]
        if isinstance(level, bool) or not isinstance(level, int):
            report.problem("output.compression_level", f"must be a whole number; got {level!r}")

    if report.has_problems:
        return None, report
    return (
        SerialisationConfig(
            population_properties=population_properties,
            geography_include_coordinates=geography.get("include_coordinates", True),
            geography_properties=geography_properties,
            venue_global_settings=venue_global,
            venue_type_properties=venue_type_properties,
            relationships=relationships,
            output_settings=output,
        ),
        report,
    )


def _section(report: PreflightReport, parent, name: str, key: str) -> Optional[dict]:
    """`parent[name]` as a mapping (`{}` if absent or blank); `None` with a problem otherwise.

    A `None` parent (itself malformed, already reported) gives `None` silently."""
    if parent is None:
        return None
    value = parent.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        report.problem(key, not_mapping_message(value))
        return None
    return value


def _names(report: PreflightReport, section, name: str, section_key: str) -> List[str]:
    """`section[name]` as a list of strings (`[]` if absent or blank); problems keyed by index."""
    if section is None:
        return []
    value = section.get(name)
    if value is None:
        return []
    key = f"{section_key}.{name}"
    if not isinstance(value, list):
        report.problem(key, not_list_message(value))
        return []
    for index, entry in enumerate(value):
        if not isinstance(entry, str):
            report.problem(f"{key}[{index}]", f"must be a name; got {entry!r}")
    return value


def _switch(report: PreflightReport, section, name: str, section_key: str) -> None:
    """Problem if `section[name]` is set and not true/false."""
    if section is None or name not in section:
        return
    value = section[name]
    if not isinstance(value, bool):
        report.problem(f"{section_key}.{name}", f"must be true or false; got {value!r}")
