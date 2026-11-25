"""
Serialisation schema: which properties and attributes the HDF5 export writes.

The schema is parsed once, before the build, so a malformed section is reported
up front rather than crashing the export after the build has been paid for.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from may.preflight.config_reader import FAILED, ConfigReader
from may.preflight.report import PreflightReport

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
        """Find, load and parse a schema file; `SerialisationError` listing every problem.

        Findings are keyed `schema` (the file) and `schema (<found path>): <section>`."""
        report = PreflightReport()
        file_reader = ConfigReader({"schema": path}, "", report).config_file("schema", data_dir=None)
        schema = parse_schema(file_reader)
        report.raise_if_problems(
            SerialisationError, f"Serialisation schema {path} cannot be used:"
        )
        for warning in report.warnings:
            logger.info(f"Serialisation schema warning: {warning.key}: {warning.message}")
        return schema


def parse_schema(reader: ConfigReader) -> SerialisationConfig:
    """Read a schema through `reader`, its root; findings go to the reader's report.

    Fields of a section with problems may be `FAILED`; callers drop a schema with problems.
    `venue_type_properties` is `FAILED` when any venue type's `properties` failed."""
    section_names = reader.keys()
    if section_names is not FAILED:
        if not section_names:
            reader.problem("is empty; give at least one section")
        for section in section_names:
            if section not in KNOWN_SECTIONS:
                reader.warning(
                    f"unknown section; ignored (known: {', '.join(KNOWN_SECTIONS)})", at=section
                )

    population = reader.section("population", default={})
    geography = reader.section("geography", default={})
    venues = reader.section("venues", default={})
    relationships = reader.section("relationships", default={})
    output = reader.section("output", default={})

    venue_global = venues.section("global", default={})
    for switch in VENUE_GLOBAL_SWITCHES:
        venue_global.true_or_false(switch, default=None)
    relationships.true_or_false("include_activity_map", default=None)
    output.true_or_false("include_metadata", default=None)
    output.names("metadata", default=[])
    output.whole_number("compression_level", default=None)

    return SerialisationConfig(
        population_properties=population.names("properties", default=[]),
        geography_include_coordinates=geography.true_or_false("include_coordinates", default=True),
        geography_properties=geography.names("properties", default=[]),
        venue_global_settings=venue_global.value(),
        venue_type_properties=_venue_type_properties(venues),
        relationships=relationships.value(),
        output_settings=output.value(),
    )


def _venue_type_properties(venues: ConfigReader):
    """Type name → property names; `FAILED` if the types or any type's properties failed."""
    type_readers = venues.entries("types", default={})
    if type_readers is FAILED:
        return FAILED
    properties = {
        venue_type: type_reader.names("properties", default=[])
        for venue_type, type_reader in type_readers.items()
    }
    if any(names is FAILED for names in properties.values()):
        return FAILED
    return properties
