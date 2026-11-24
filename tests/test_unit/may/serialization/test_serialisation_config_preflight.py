"""Serialisation block checks: every malformed key is found before the build."""

import pytest

from may.serialization.world_serializer import WorldSerializer
from may.utils import path_resolver

SCHEMA_2021 = "configs/2021/serialization_config.yaml"


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


def problem_keys(report):
    return [finding.key for finding in report.problems]


def test_clean_block_has_empty_report_and_parsed_settings():
    settings, report = WorldSerializer.parse_config(block())
    assert report.problems == [] and report.warnings == []
    assert settings.enabled is True
    assert settings.schema_path == SCHEMA_2021
    assert settings.schema is not None
    assert settings.output_dir == "output/2021"
    assert settings.filename == "world_state.h5"


class TestEnabled:
    def test_not_true_or_false_is_problem(self):
        settings, report = WorldSerializer.parse_config(block(enabled="yes"))
        assert settings is None
        assert problem_keys(report) == ["enabled"]

    def test_disabled_checks_nothing(self):
        settings, report = WorldSerializer.parse_config(
            {"enabled": False, "config_file": 7, "output_dir": 3, "filename": []}
        )
        assert report.problems == [] and report.warnings == []
        assert settings.enabled is False
        assert settings.schema is None

    def test_unset_means_enabled(self):
        unset = block()
        del unset["enabled"]
        settings, report = WorldSerializer.parse_config(unset)
        assert report.problems == []
        assert settings.enabled is True


def messages(report):
    return {finding.key: finding.message for finding in report.problems}


class TestConfigFile:
    @pytest.mark.parametrize("unset", [None, ""])
    def test_unset_or_blank_while_enabled_is_problem(self, unset):
        settings, report = WorldSerializer.parse_config(block(config_file=unset))
        assert settings is None
        assert messages(report) == {
            "config_file": "not set; serialisation is enabled"
        }

    def test_absent_while_enabled_is_problem(self):
        absent = block()
        del absent["config_file"]
        _, report = WorldSerializer.parse_config(absent)
        assert problem_keys(report) == ["config_file"]

    def test_not_a_path_is_problem(self):
        _, report = WorldSerializer.parse_config(block(config_file=["a.yaml"]))
        assert messages(report) == {"config_file": "must be a path; got ['a.yaml']"}

    def test_not_found_names_tried_path(self):
        _, report = WorldSerializer.parse_config(block(config_file="nowhere/schema.yaml"))
        assert problem_keys(report) == ["config_file"]
        assert "nowhere/schema.yaml" in messages(report)["config_file"]

    def test_config_root_template_resolves(self):
        path_resolver.init("configs/2021", None, "output")
        settings, report = WorldSerializer.parse_config(
            block(config_file="${config_root}/serialization_config.yaml")
        )
        assert report.problems == []
        assert settings.schema_path == SCHEMA_2021

    @pytest.mark.parametrize(
        "text, expected",
        [("population: [\n", "YAML"), ("", "empty"), ("- a\n- b\n", "mapping")],
    )
    def test_unloadable_schema_file_is_problem(self, tmp_path, text, expected):
        schema_file = tmp_path / "schema.yaml"
        schema_file.write_text(text)
        _, report = WorldSerializer.parse_config(block(config_file=str(schema_file)))
        assert problem_keys(report)[0].startswith("config_file")
        assert expected in " ".join(messages(report).values())

    def test_schema_finding_keyed_under_config_file_and_found_path(self, tmp_path):
        schema_file = tmp_path / "schema.yaml"
        schema_file.write_text("population: 3\n")
        _, report = WorldSerializer.parse_config(block(config_file=str(schema_file)))
        assert problem_keys(report) == [f"config_file ({schema_file}): population"]


class TestOutput:
    @pytest.mark.parametrize("key", ["output_dir", "filename"])
    @pytest.mark.parametrize("value", [3, "", ["a"]])
    def test_not_a_path_is_problem(self, key, value):
        settings, report = WorldSerializer.parse_config(block(**{key: value}))
        assert settings is None
        assert messages(report) == {key: f"must be a path; got {value!r}"}

    def test_unset_takes_defaults(self):
        settings, report = WorldSerializer.parse_config(
            {"config_file": SCHEMA_2021, "output_dir": None}
        )
        assert report.problems == []
        assert settings.output_dir == "."
        assert settings.filename == "world_state.h5"

    def test_output_root_template_resolves(self):
        path_resolver.init("configs/2021", None, "output")
        settings, _ = WorldSerializer.parse_config(block(output_dir="${output_root}/2021"))
        assert settings.output_dir == "output/2021"

    def test_every_block_problem_reported_together(self):
        _, report = WorldSerializer.parse_config(
            block(config_file=None, output_dir=3, filename=4)
        )
        assert sorted(problem_keys(report)) == ["config_file", "filename", "output_dir"]

    def test_unknown_key_is_no_finding(self):
        _, report = WorldSerializer.parse_config(block(compression="gzip"))
        assert report.problems == [] and report.warnings == []


@pytest.mark.parametrize("config_dir", ["configs/2021", "configs/1911"])
def test_repo_blocks_are_clean(config_dir):
    import yaml

    path_resolver.init(config_dir, None, "output")
    with open(f"{config_dir}/config.yaml") as config_file:
        serialisation_block = yaml.safe_load(config_file)["serialization"]
    _, report = WorldSerializer.parse_config(serialisation_block)
    assert report.problems == [] and report.warnings == []


def test_preflight_returns_parse_report():
    report = WorldSerializer.preflight(block(enabled="yes"))
    assert problem_keys(report) == ["enabled"]
