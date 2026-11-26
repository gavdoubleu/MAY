"""Serialisation schema checks: every malformed section is found before the build."""

import logging

import pytest

from may.preflight.config_reader import ConfigReader
from may.preflight.report import PreflightReport
from may.serialization.serialization_config import (
    SerialisationConfig,
    SerialisationError,
    parse_schema,
)
from may.utils import path_resolver

REPO_SCHEMAS = [
    "configs/2021/serialization_config.yaml",
    "configs/1911/serialization_config.yaml",
]


@pytest.fixture(autouse=True)
def reset_resolver(monkeypatch):
    monkeypatch.setattr(path_resolver, "_resolver", None)


def schema(**overrides):
    """A schema shaped like `configs/2021/serialization_config.yaml`."""
    base = {
        "population": {"properties": ["ethnicity", "friendships"]},
        "geography": {"include_coordinates": True, "properties": []},
        "venues": {
            "global": {"include_coordinates": True, "include_is_residence": True},
            "types": {
                "household": {"properties": ["capacity"]},
                "university": {"properties": []},
            },
        },
        "relationships": {"include_activity_map": True},
        "output": {
            "compression": "gzip",
            "compression_level": 4,
            "include_metadata": True,
            "metadata": ["creation_timestamp"],
        },
    }
    base.update(overrides)
    return base


def parse(mapping):
    """(parsed schema, report), keys relative to the schema root."""
    report = PreflightReport()
    parsed = parse_schema(ConfigReader(mapping, "", report))
    return parsed, report


def problem_keys(report):
    return [finding.key for finding in report.problems]


def warning_keys(report):
    return [finding.key for finding in report.warnings]


class TestParseSchema:
    def test_clean_schema_has_empty_report_and_parsed_fields(self):
        parsed, report = parse(schema())
        assert report.problems == [] and report.warnings == []
        assert parsed.population_properties == ["ethnicity", "friendships"]
        assert parsed.geography_include_coordinates is True
        assert parsed.geography_properties == []
        assert parsed.venue_global_settings == {
            "include_coordinates": True,
            "include_is_residence": True,
        }
        assert parsed.venue_type_properties == {"household": ["capacity"], "university": []}
        assert parsed.relationships == {"include_activity_map": True}
        assert parsed.output_settings["compression_level"] == 4

    def test_parsed_config_is_frozen(self):
        parsed, _ = parse(schema())
        with pytest.raises(Exception):
            parsed.population_properties = []

    def test_absent_sections_take_defaults(self):
        parsed, report = parse({"venues": {"types": {"pub": {}}}})
        assert report.problems == []
        assert parsed.population_properties == []
        assert parsed.geography_include_coordinates is True
        assert parsed.venue_type_properties == {"pub": []}
        assert parsed.relationships == {}
        assert parsed.output_settings == {}

    @pytest.mark.parametrize(
        "overrides, key",
        [
            ({"population": None}, "population"),
            ({"venues": {"types": None}}, "venues.types"),
            ({"venues": {"types": {"pub": None}}}, "venues.types.pub"),
            ({"population": {"properties": None}}, "population.properties"),
            ({"output": {"metadata": None}}, "output.metadata"),
        ],
    )
    def test_blank_section_or_name_list_is_problem(self, overrides, key):
        _, report = parse(schema(**overrides))
        assert problem_keys(report) == [key]

    def test_empty_schema_is_problem(self):
        _, report = parse({})
        assert problem_keys(report) == [""]
        assert "empty" in report.problems[0].message

    @pytest.mark.parametrize(
        "overrides, key",
        [
            ({"population": ["ethnicity"]}, "population"),
            ({"geography": "yes"}, "geography"),
            ({"venues": ["household"]}, "venues"),
            ({"venues": {"global": True}}, "venues.global"),
            ({"venues": {"types": ["household"]}}, "venues.types"),
            ({"venues": {"types": {"household": ["capacity"]}}}, "venues.types.household"),
            ({"relationships": True}, "relationships"),
            ({"output": "gzip"}, "output"),
        ],
    )
    def test_section_not_mapping_is_problem(self, overrides, key):
        _, report = parse(schema(**overrides))
        assert problem_keys(report) == [key]
        assert "must be a mapping" in report.problems[0].message

    @pytest.mark.parametrize(
        "overrides, key",
        [
            ({"population": {"properties": "ethnicity"}}, "population.properties"),
            ({"geography": {"properties": {"a": 1}}}, "geography.properties"),
            (
                {"venues": {"types": {"school": {"properties": "Gender"}}}},
                "venues.types.school.properties",
            ),
            ({"output": {"metadata": "creation_timestamp"}}, "output.metadata"),
        ],
    )
    def test_name_list_not_list_is_problem(self, overrides, key):
        _, report = parse(schema(**overrides))
        assert problem_keys(report) == [key]
        assert "must be a list" in report.problems[0].message

    def test_name_list_bad_items_are_problems_counted_from_one(self):
        _, report = parse(schema(population={"properties": ["ethnicity", 5.5, None]}))
        assert problem_keys(report) == ["population.properties[2]", "population.properties[3]"]

    def test_whole_number_names_read_as_text(self):
        parsed, report = parse(
            schema(population={"properties": [2021]}, venues={"types": {7: {"properties": [1]}}})
        )
        assert report.problems == []
        assert parsed.population_properties == ["2021"]
        assert parsed.venue_type_properties == {"7": ["1"]}

    @pytest.mark.parametrize(
        "overrides, key",
        [
            ({"geography": {"include_coordinates": "yes"}}, "geography.include_coordinates"),
            (
                {"venues": {"global": {"include_coordinates": 1}}},
                "venues.global.include_coordinates",
            ),
            (
                {"venues": {"global": {"include_is_residence": "no"}}},
                "venues.global.include_is_residence",
            ),
            (
                {"relationships": {"include_activity_map": None}},
                "relationships.include_activity_map",
            ),
            ({"output": {"include_metadata": "true"}}, "output.include_metadata"),
        ],
    )
    def test_switch_not_true_or_false_is_problem(self, overrides, key):
        _, report = parse(schema(**overrides))
        assert problem_keys(report) == [key]

    @pytest.mark.parametrize("value", ["4", 4.5, True])
    def test_compression_level_not_whole_number_is_problem(self, value):
        _, report = parse(schema(output={"compression_level": value}))
        assert problem_keys(report) == ["output.compression_level"]

    def test_unknown_top_level_section_is_warning(self):
        parsed, report = parse(schema(subsets={"properties": []}))
        assert parsed is not None
        assert report.problems == []
        assert warning_keys(report) == ["subsets"]

    def test_every_problem_reported_in_one_pass(self):
        _, report = parse(schema(population="x", geography={"include_coordinates": "y"}, output=[]))
        assert set(problem_keys(report)) == {"population", "geography.include_coordinates", "output"}


class TestFromFile:
    @pytest.mark.parametrize("path", REPO_SCHEMAS)
    def test_repo_schemas_parse(self, path):
        parsed = SerialisationConfig.from_file(path)
        assert isinstance(parsed, SerialisationConfig)
        assert parsed.population_properties

    def test_path_object_reads_as_its_text(self, tmp_path):
        path = tmp_path / "schema.yaml"
        path.write_text("population: {properties: [age_band]}\n")
        parsed = SerialisationConfig.from_file(path)
        assert parsed.population_properties == ["age_band"]

    def test_raises_listing_every_problem(self, tmp_path):
        path = tmp_path / "schema.yaml"
        path.write_text("population: [ethnicity]\noutput: {compression_level: high}\n")
        with pytest.raises(SerialisationError) as excinfo:
            SerialisationConfig.from_file(str(path))
        message = str(excinfo.value)
        assert message.startswith(f"Serialisation schema {path} cannot be used:")
        assert f"schema ({path}): population" in message
        assert f"schema ({path}): output.compression_level" in message
        assert excinfo.value.report.has_problems

    @pytest.mark.parametrize(
        "text, expected",
        [(None, "not found"), ("population: [unclosed\n", "not valid YAML"), ("# nothing\n", "empty")],
        ids=["missing", "bad_yaml", "empty"],
    )
    def test_unloadable_file_raises_one_problem(self, tmp_path, text, expected):
        path = tmp_path / "schema.yaml"
        if text is not None:
            path.write_text(text)
        with pytest.raises(SerialisationError) as excinfo:
            SerialisationConfig.from_file(str(path))
        problems = excinfo.value.report.problems
        assert [finding.key for finding in problems] == ["schema"]
        assert expected in problems[0].message

    def test_config_root_substituted(self, tmp_path):
        (tmp_path / "schema.yaml").write_text("population: {properties: [age_band]}\n")
        path_resolver.init(config_root=str(tmp_path), data_root=None, output_root=str(tmp_path))
        parsed = SerialisationConfig.from_file("${config_root}/schema.yaml")
        assert parsed.population_properties == ["age_band"]

    def test_logs_warnings_at_info(self, tmp_path, caplog):
        path = tmp_path / "schema.yaml"
        path.write_text("population: {properties: []}\nsubsets: {}\n")
        with caplog.at_level(logging.INFO):
            SerialisationConfig.from_file(str(path))
        assert any(
            "subsets" in record.getMessage() and record.levelno == logging.INFO
            for record in caplog.records
        )
