"""Serialisation schema checks: every malformed section is found before the build."""

import logging

import pytest

from may.serialization.serialization_config import (
    SerialisationConfig,
    SerialisationError,
    load_schema_file,
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


def problem_keys(report):
    return [finding.key for finding in report.problems]


def warning_keys(report):
    return [finding.key for finding in report.warnings]


class TestParseSchema:
    def test_clean_schema_has_empty_report_and_parsed_fields(self):
        parsed, report = parse_schema(schema())
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
        parsed, _ = parse_schema(schema())
        with pytest.raises(Exception):
            parsed.population_properties = []

    def test_absent_and_blank_sections_take_defaults(self):
        parsed, report = parse_schema({"population": None, "venues": {"types": {"pub": None}}})
        assert report.problems == []
        assert parsed.population_properties == []
        assert parsed.geography_include_coordinates is True
        assert parsed.venue_type_properties == {"pub": []}
        assert parsed.relationships == {}
        assert parsed.output_settings == {}

    @pytest.mark.parametrize("value", [None, {}])
    def test_empty_schema_is_problem(self, value):
        parsed, report = parse_schema(value)
        assert parsed is None
        assert problem_keys(report) == [""]
        assert "empty" in report.problems[0].message

    @pytest.mark.parametrize("value", [["population"], "population"])
    def test_non_mapping_schema_is_problem(self, value):
        parsed, report = parse_schema(value)
        assert parsed is None
        assert problem_keys(report) == [""]
        assert "must be a mapping" in report.problems[0].message

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
        parsed, report = parse_schema(schema(**overrides))
        assert parsed is None
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
        parsed, report = parse_schema(schema(**overrides))
        assert parsed is None
        assert problem_keys(report) == [key]
        assert "must be a list" in report.problems[0].message

    def test_name_list_entry_not_string_is_problem_naming_index(self):
        parsed, report = parse_schema(
            schema(population={"properties": ["ethnicity", 5, None]})
        )
        assert parsed is None
        assert problem_keys(report) == ["population.properties[1]", "population.properties[2]"]

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
        parsed, report = parse_schema(schema(**overrides))
        assert parsed is None
        assert problem_keys(report) == [key]
        assert "true or false" in report.problems[0].message

    @pytest.mark.parametrize("value", ["4", 4.5, True])
    def test_compression_level_not_whole_number_is_problem(self, value):
        parsed, report = parse_schema(schema(output={"compression_level": value}))
        assert parsed is None
        assert problem_keys(report) == ["output.compression_level"]

    def test_unknown_top_level_section_is_warning(self):
        parsed, report = parse_schema(schema(subsets={"properties": []}))
        assert parsed is not None
        assert report.problems == []
        assert warning_keys(report) == ["subsets"]

    def test_every_problem_reported_in_one_pass(self):
        _, report = parse_schema(
            schema(population="x", geography={"include_coordinates": "y"}, output=[])
        )
        assert set(problem_keys(report)) == {"population", "geography.include_coordinates", "output"}


class TestLoadSchemaFile:
    @pytest.mark.parametrize("path", REPO_SCHEMAS)
    def test_repo_schemas_are_clean(self, path):
        parsed, report = load_schema_file(path)
        assert report.problems == [] and report.warnings == []
        assert isinstance(parsed, SerialisationConfig)

    def test_missing_file_is_problem_naming_path(self, tmp_path):
        missing = str(tmp_path / "absent.yaml")
        parsed, report = load_schema_file(missing)
        assert parsed is None
        assert problem_keys(report) == [""]
        assert missing in report.problems[0].message

    def test_invalid_yaml_is_problem(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("population: [unclosed\n")
        parsed, report = load_schema_file(str(path))
        assert parsed is None
        assert problem_keys(report) == [""]
        assert "not valid YAML" in report.problems[0].message

    def test_empty_file_is_problem(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("# everything commented out\n")
        parsed, report = load_schema_file(str(path))
        assert parsed is None
        assert problem_keys(report) == [""]
        assert str(path) in report.problems[0].message
        assert "empty" in report.problems[0].message

    def test_schema_findings_keyed_under_found_path(self, tmp_path):
        path = tmp_path / "schema.yaml"
        path.write_text("population: [ethnicity]\nsubsets: {}\n")
        parsed, report = load_schema_file(str(path))
        assert parsed is None
        assert problem_keys(report) == [f"({path}): population"]
        assert warning_keys(report) == [f"({path}): subsets"]

    def test_config_root_substituted(self, tmp_path):
        (tmp_path / "schema.yaml").write_text("population: {properties: [age_band]}\n")
        path_resolver.init(config_root=str(tmp_path), data_root=None, output_root=str(tmp_path))
        parsed, report = load_schema_file("${config_root}/schema.yaml")
        assert report.problems == []
        assert parsed.population_properties == ["age_band"]


class TestFromFile:
    def test_returns_parsed_schema(self):
        parsed = SerialisationConfig.from_file(REPO_SCHEMAS[0])
        assert "ethnicity" in parsed.population_properties

    def test_raises_listing_every_problem(self, tmp_path):
        path = tmp_path / "schema.yaml"
        path.write_text("population: [ethnicity]\noutput: {compression_level: high}\n")
        with pytest.raises(SerialisationError) as excinfo:
            SerialisationConfig.from_file(str(path))
        message = str(excinfo.value)
        assert "population" in message and "output.compression_level" in message
        assert excinfo.value.report.has_problems

    def test_logs_warnings_at_info(self, tmp_path, caplog):
        path = tmp_path / "schema.yaml"
        path.write_text("population: {properties: []}\nsubsets: {}\n")
        with caplog.at_level(logging.INFO):
            SerialisationConfig.from_file(str(path))
        assert any(
            "subsets" in record.getMessage() and record.levelno == logging.INFO
            for record in caplog.records
        )
