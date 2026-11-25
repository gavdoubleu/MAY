"""Serialisation block checks: every malformed key is found before the build."""

import pytest
import yaml

from may.geography.venue_manager import VenueConfig, VenueManager
from may.preflight.context import StageNotParsed
from may.serialization.world_serializer import WorldSerializer
from may.utils import path_resolver
from tests.test_unit.may.config_contexts import geography_context

SCHEMA_2021 = "configs/2021/serialization_config.yaml"
GIVE_SCHEMA = "give a schema file, or set enabled: false"


@pytest.fixture(autouse=True)
def reset_resolver(monkeypatch):
    monkeypatch.setattr(path_resolver, "_resolver", None)


def block(**overrides):
    """A block shaped like the `configs/2021/config.yaml` serialization block."""
    base = {
        "enabled": True,
        "config_file": SCHEMA_2021,
        "output_dir": "output/2021",
        "filename": "world_state.h5",
    }
    base.update(overrides)
    return base


def venue_config(venue_paths=None):
    return VenueConfig(
        data_dir="data", config_path="venues.yaml", venue_types={},
        filter_by_geography=None, venue_paths=venue_paths or {},
    )


def venue_context(config="no venue files"):
    """Context with geography and a `VenueConfig`; `config=None` = venues failed."""
    if config == "no venue files":
        config = venue_config()
    return geography_context(extra_results={VenueManager: config})


def preflight(serialisation_block, context=None):
    return WorldSerializer.preflight(serialisation_block, context or venue_context())


def problem_keys(report):
    return [finding.key for finding in report.problems]


def problem_lines(report):
    return [(finding.key, finding.message) for finding in report.problems]


def warning_lines(report):
    return [(finding.key, finding.message) for finding in report.warnings]


def test_clean_block_has_empty_report_and_parsed_settings():
    settings, report = preflight(block())
    assert report.problems == [] and report.warnings == []
    assert settings.enabled is True
    assert settings.schema_path == SCHEMA_2021
    assert settings.schema is not None
    assert settings.output_dir == "output/2021"
    assert settings.filename == "world_state.h5"


@pytest.mark.parametrize(
    "value, message",
    [(None, "left blank; give a value"), ("schema.yaml", "must be a mapping; got text 'schema.yaml'")],
    ids=["blank", "text"],
)
def test_blank_or_not_mapping_block_is_one_problem(value, message):
    settings, report = preflight(value)
    assert settings is None
    assert problem_lines(report) == [("serialization", message)]


class TestEnabled:
    def test_not_true_or_false_is_problem(self):
        settings, report = preflight(block(enabled="yes"))
        assert settings is None
        assert problem_keys(report) == ["serialization.enabled"]

    def test_disabled_checks_nothing(self):
        settings, report = preflight(
            {"enabled": False, "config_file": 7, "output_dir": 3, "filename": []}
        )
        assert report.problems == [] and report.warnings == []
        assert settings.enabled is False
        assert settings.schema is None

    def test_unset_means_enabled(self):
        unset = block()
        del unset["enabled"]
        settings, report = preflight(unset)
        assert report.problems == []
        assert settings.enabled is True


class TestConfigFile:
    @pytest.mark.parametrize(
        "unset, message",
        [(None, f"left blank; {GIVE_SCHEMA}"), ("", f"must not be empty; {GIVE_SCHEMA}")],
        ids=["blank", "empty"],
    )
    def test_blank_or_empty_while_enabled_is_problem(self, unset, message):
        settings, report = preflight(block(config_file=unset))
        assert settings is None
        assert problem_lines(report) == [("serialization.config_file", message)]

    def test_absent_while_enabled_is_problem(self):
        absent = block()
        del absent["config_file"]
        _, report = preflight(absent)
        assert problem_lines(report) == [("serialization.config_file", f"not set; {GIVE_SCHEMA}")]

    def test_not_text_is_problem(self):
        _, report = preflight(block(config_file=["a.yaml"]))
        assert problem_lines(report) == [("serialization.config_file", "must be text; got a list")]

    def test_not_found_names_tried_path(self):
        _, report = preflight(block(config_file="nowhere/schema.yaml"))
        assert problem_keys(report) == ["serialization.config_file"]
        assert "nowhere/schema.yaml" in report.problems[0].message

    def test_config_root_template_resolves(self):
        path_resolver.init("configs/2021", None, "output")
        settings, report = preflight(block(config_file="${config_root}/serialization_config.yaml"))
        assert report.problems == []
        assert settings.schema_path == SCHEMA_2021

    @pytest.mark.parametrize(
        "text, expected",
        [("population: [\n", "YAML"), ("", "empty"), ("{}\n", "empty"), ("- a\n- b\n", "mapping")],
        ids=["bad_yaml", "empty_file", "empty_mapping", "list"],
    )
    def test_unloadable_schema_file_is_one_problem(self, tmp_path, text, expected):
        schema_file = tmp_path / "schema.yaml"
        schema_file.write_text(text)
        _, report = preflight(block(config_file=str(schema_file)))
        assert len(report.problems) == 1
        assert problem_keys(report)[0].startswith("serialization.config_file")
        assert expected in report.problems[0].message

    def test_schema_finding_keyed_under_config_file_and_found_path(self, tmp_path):
        schema_file = tmp_path / "schema.yaml"
        schema_file.write_text("population: 3\n")
        _, report = preflight(block(config_file=str(schema_file)))
        assert problem_keys(report) == [f"serialization.config_file ({schema_file}): population"]


class TestOutput:
    @pytest.mark.parametrize("key", ["output_dir", "filename"])
    @pytest.mark.parametrize(
        "value, message",
        [
            (3.5, "must be text; got a number 3.5; put it in quotes to use it as text"),
            ("", "must not be empty; give a path"),
            (["a"], "must be text; got a list"),
            (None, "left blank; give a value, or remove the key to use the default"),
        ],
        ids=["number", "empty", "list", "blank"],
    )
    def test_not_a_path_is_problem(self, key, value, message):
        settings, report = preflight(block(**{key: value}))
        assert settings is None
        assert problem_lines(report) == [(f"serialization.{key}", message)]

    def test_whole_number_read_as_text(self):
        settings, report = preflight(block(output_dir=2021))
        assert report.problems == []
        assert settings.output_dir == "2021"

    def test_absent_takes_defaults(self):
        settings, report = preflight({"config_file": SCHEMA_2021})
        assert report.problems == []
        assert settings.output_dir == "."
        assert settings.filename == "world_state.h5"

    def test_output_root_template_resolves(self):
        path_resolver.init("configs/2021", None, "output")
        settings, _ = preflight(block(output_dir="${output_root}/2021"))
        assert settings.output_dir == "output/2021"

    def test_every_block_problem_reported_together(self):
        _, report = preflight(block(config_file=None, output_dir=3.5, filename=[]))
        assert sorted(problem_keys(report)) == [
            "serialization.config_file", "serialization.filename", "serialization.output_dir",
        ]

    def test_unknown_key_is_no_finding(self):
        _, report = preflight(block(compression="gzip"))
        assert report.problems == [] and report.warnings == []


@pytest.mark.parametrize("config_dir", ["configs/2021", "configs/1911"])
def test_repo_blocks_are_clean(config_dir):
    path_resolver.init(config_dir, None, "output")
    with open(f"{config_dir}/config.yaml") as config_file:
        serialisation_block = yaml.safe_load(config_file)["serialization"]
    _, report = preflight(serialisation_block)
    assert report.problems == [] and report.warnings == []


class TestVenuePropertyGaps:
    @pytest.fixture
    def company_file(self, tmp_path):
        path = tmp_path / "companies.csv"
        path.write_text("geo_unit,industry_code\nA,C\n")
        return path

    @pytest.fixture
    def schema_file(self, tmp_path):
        path = tmp_path / "serialization.yaml"
        path.write_text("venues:\n  types:\n    company:\n      properties: [work_sector]\n")
        return path

    def company_context(self, company_file):
        return venue_context(venue_config({"company": [str(company_file)]}))

    def test_property_missing_from_venue_files_is_a_warning(self, company_file, schema_file):
        settings, report = preflight(
            block(config_file=str(schema_file)), self.company_context(company_file)
        )
        assert settings is not None and report.problems == []
        assert warning_lines(report) == [
            (
                f"serialization.config_file ({schema_file}): venues.types.company.properties",
                "'work_sector' not in the venue files; they carry ['geo_unit', 'industry_code']",
            )
        ]

    def test_venues_failed_gives_no_gap_findings(self, schema_file):
        _, report = preflight(block(config_file=str(schema_file)), venue_context(None))
        assert report.problems == [] and report.warnings == []

    def test_disabled_gives_no_gap_findings(self, company_file, schema_file):
        _, report = preflight(
            block(enabled=False, config_file=str(schema_file)), self.company_context(company_file)
        )
        assert report.problems == [] and report.warnings == []

    def test_unreadable_venue_file_is_one_skipped_warning(self, company_file, schema_file):
        company_file.unlink()
        company_file.mkdir()
        _, report = preflight(
            block(config_file=str(schema_file)), self.company_context(company_file)
        )
        assert report.problems == []
        assert [finding.key for finding in report.warnings] == [
            f"serialization.config_file ({schema_file})"
        ]
        assert report.warnings[0].message.startswith("venue property check skipped: ")

    @pytest.mark.parametrize(
        "schema_yaml, section",
        [
            ("venues: [a, b]\n", "venues"),
            ("venues:\n  types: company\n", "venues.types"),
            ("venues:\n  types:\n    company:\n      properties: work_sector\n", "venues.types.company.properties"),
        ],
        ids=["venues_list", "types_text", "properties_text"],
    )
    def test_malformed_schema_venues_is_problem_and_skips_gap_check(
        self, company_file, schema_file, schema_yaml, section
    ):
        schema_file.write_text(schema_yaml)
        _, report = preflight(
            block(config_file=str(schema_file)), self.company_context(company_file)
        )
        assert problem_keys(report) == [f"serialization.config_file ({schema_file}): {section}"]
        assert report.warnings == []

    def test_schema_type_absent_from_venues_is_accepted(self, company_file, schema_file):
        """Child and route venue types exist only once the build creates them."""
        schema_file.write_text("venues:\n  types:\n    classroom:\n      properties: [capacity]\n")
        _, report = preflight(
            block(config_file=str(schema_file)), self.company_context(company_file)
        )
        assert report.problems == [] and report.warnings == []

    def test_venues_not_parsed_raises(self):
        with pytest.raises(StageNotParsed):
            preflight(block(), geography_context())
