import os

import pytest

from may.geography.venue_manager import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_DATA_DIR,
    VenueManager,
)
from may.utils import path_resolver


@pytest.fixture(autouse=True)
def reset_resolver(monkeypatch):
    monkeypatch.setattr(path_resolver, "_resolver", None)


@pytest.fixture
def working_dir(tmp_path, monkeypatch):
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    return working


@pytest.fixture
def write_config(working_dir):
    """Write venue YAML in the working dir, and empty `data_files` under data/;
    return the block pointing at them."""

    def write(text, config_file="venues.yaml", data_files=()):
        (working_dir / config_file).write_text(text)
        (working_dir / "data").mkdir(exist_ok=True)
        for name in data_files:
            (working_dir / "data" / name).write_text("geo_unit\n")
        return {"data_dir": "data", "config_file": config_file}

    return write


def test_empty_venue_types_is_clean_and_parsed(write_config):
    block = write_config("venue_types: {}\n")
    venue_config, report = VenueManager.parse_config(block)
    assert not report.problems and not report.warnings
    assert venue_config.venue_types == {}
    assert venue_config.config_path == "venues.yaml"
    assert venue_config.data_dir == "data"


def test_blank_venue_types_is_clean_and_parsed_as_empty(write_config):
    block = write_config("venue_types:\n  # pub:\n  #   filename: pubs.csv\n")
    venue_config, report = VenueManager.parse_config(block)
    assert not report.problems and not report.warnings
    assert venue_config.venue_types == {}


def problem_lines(report):
    return [(finding.key, finding.message) for finding in report.problems]


def test_config_file_not_found_is_problem(working_dir):
    venue_config, report = VenueManager.parse_config(
        {"data_dir": "data", "config_file": "missing.yaml"}
    )
    assert venue_config is None
    assert problem_lines(report) == [
        ("config_file", "not found; tried missing.yaml, data/missing.yaml")
    ]


def test_invalid_yaml_is_problem_with_line_and_column(write_config):
    block = write_config("venue_types:\n  pub: [unclosed\n")
    venue_config, report = VenueManager.parse_config(block)
    assert venue_config is None
    assert problem_lines(report) == [
        ("config_file", "venues.yaml is not valid YAML at line 3, column 1")
    ]


def test_empty_file_is_problem(write_config):
    block = write_config("# nothing here\n")
    venue_config, report = VenueManager.parse_config(block)
    assert venue_config is None
    assert problem_lines(report) == [("config_file", "venues.yaml is empty")]


def test_top_level_not_mapping_is_problem(write_config):
    block = write_config("- pub\n- school\n")
    venue_config, report = VenueManager.parse_config(block)
    assert venue_config is None
    assert problem_lines(report) == [
        ("config_file (venues.yaml)", "must be a mapping; got sequence (list)")
    ]


@pytest.mark.parametrize("key", ["data_dir", "config_file"])
@pytest.mark.parametrize("value", [["venues.yaml"], {"path": "venues.yaml"}], ids=["list", "mapping"])
def test_a_path_that_is_a_list_or_mapping_is_problem(write_config, key, value):
    block = write_config("venue_types: {}\n")
    block[key] = value
    venue_config, report = VenueManager.parse_config(block)
    assert venue_config is None
    assert [finding.key for finding in report.problems] == [key]


def test_missing_venue_types_is_problem(write_config):
    block = write_config("settings: {}\n")
    venue_config, report = VenueManager.parse_config(block)
    assert venue_config is None
    assert problem_lines(report) == [
        ("config_file (venues.yaml): venue_types", "not set")
    ]


def test_venue_types_not_mapping_is_problem(write_config):
    block = write_config("venue_types:\n  - pub\n")
    venue_config, report = VenueManager.parse_config(block)
    assert venue_config is None
    assert problem_lines(report) == [
        (
            "config_file (venues.yaml): venue_types",
            "must be a mapping; got sequence (list)",
        )
    ]


def test_type_entry_not_mapping_is_problem(write_config):
    block = write_config("venue_types:\n  pub: pubs.csv\n")
    venue_config, report = VenueManager.parse_config(block)
    assert venue_config is None
    assert problem_lines(report) == [
        ("config_file (venues.yaml): venue_types.pub", "must be a mapping; got 'pubs.csv'")
    ]


def test_settings_not_mapping_is_problem(write_config):
    block = write_config("venue_types: {}\nsettings:\n  - filter_by_geography\n")
    venue_config, report = VenueManager.parse_config(block)
    assert venue_config is None
    assert problem_lines(report) == [
        (
            "config_file (venues.yaml): settings",
            "must be a mapping; got sequence (list)",
        )
    ]


@pytest.mark.parametrize("flag", ["enabled", "batch_mode"])
def test_type_flag_not_bool_is_problem(write_config, flag):
    block = write_config(
        f"venue_types:\n  pub:\n    filename: pubs.csv\n    {flag}: 'true'\n",
        data_files=["pubs.csv"],
    )
    venue_config, report = VenueManager.parse_config(block)
    assert venue_config is None
    assert problem_lines(report) == [
        (
            f"config_file (venues.yaml): venue_types.pub.{flag}",
            "must be true or false; got 'true'",
        )
    ]


def test_filter_by_geography_not_bool_is_problem(write_config):
    block = write_config("venue_types: {}\nsettings:\n  filter_by_geography: 1\n")
    venue_config, report = VenueManager.parse_config(block)
    assert venue_config is None
    assert problem_lines(report) == [
        (
            "config_file (venues.yaml): settings.filter_by_geography",
            "must be true or false; got 1",
        )
    ]


def test_filter_by_geography_unset_is_none(write_config):
    venue_config, _ = VenueManager.parse_config(write_config("venue_types: {}\n"))
    assert venue_config.filter_by_geography is None


def test_filter_by_geography_set_is_value_yaml_no_counts_as_bool(write_config):
    block = write_config("venue_types: {}\nsettings:\n  filter_by_geography: no\n")
    venue_config, report = VenueManager.parse_config(block)
    assert not report.problems
    assert venue_config.filter_by_geography is False


def test_enabled_type_without_filename_is_problem(write_config):
    block = write_config("venue_types:\n  pub:\n    description: pubs\n")
    venue_config, report = VenueManager.parse_config(block)
    assert venue_config is None
    assert problem_lines(report) == [
        ("config_file (venues.yaml): venue_types.pub.filename", "not set; type is enabled")
    ]


def test_batch_filename_list_is_problem(write_config):
    block = write_config(
        "venue_types:\n  pub:\n    batch_mode: true\n    level: MGU\n    filename: ['a_{unit}.csv', 'b_{unit}.csv']\n"
    )
    venue_config, report = VenueManager.parse_config(block)
    assert venue_config is None
    assert problem_lines(report) == [
        (
            "config_file (venues.yaml): venue_types.pub.filename",
            "batch_mode takes one pattern, not a list; got ['a_{unit}.csv', 'b_{unit}.csv']",
        )
    ]


def test_batch_filename_without_unit_placeholder_is_problem(write_config):
    block = write_config(
        "venue_types:\n  pub:\n    batch_mode: true\n    level: MGU\n    filename: pubs.csv\n"
    )
    venue_config, report = VenueManager.parse_config(block)
    assert venue_config is None
    assert problem_lines(report) == [
        (
            "config_file (venues.yaml): venue_types.pub.filename",
            "batch_mode pattern must contain '{unit}'; got 'pubs.csv'",
        )
    ]


def warning_lines(report):
    return [(finding.key, finding.message) for finding in report.warnings]


def test_filter_column_without_values_warns_and_still_parses(write_config):
    block = write_config(
        "venue_types:\n  pub:\n    filename: pubs.csv\n    filter_column: kind\n",
        data_files=["pubs.csv"],
    )
    venue_config, report = VenueManager.parse_config(block)
    assert not report.problems
    assert set(venue_config.venue_types) == {"pub"}
    assert warning_lines(report) == [
        (
            "config_file (venues.yaml): venue_types.pub.filter_values",
            "not set; filter_column ignored",
        )
    ]


def test_filter_values_without_column_warns_and_still_parses(write_config):
    block = write_config(
        "venue_types:\n  pub:\n    filename: pubs.csv\n    filter_values: [A, B]\n",
        data_files=["pubs.csv"],
    )
    venue_config, report = VenueManager.parse_config(block)
    assert not report.problems
    assert set(venue_config.venue_types) == {"pub"}
    assert warning_lines(report) == [
        (
            "config_file (venues.yaml): venue_types.pub.filter_column",
            "not set; filter_values ignored",
        )
    ]


def test_several_findings_in_one_file_all_reported(write_config):
    block = write_config(
        "venue_types:\n"
        "  pub:\n    enabled: maybe\n    filename: pubs.csv\n"
        "  school:\n    description: schools\n"
        "  gym:\n    filename: gyms.csv\n    filter_column: kind\n"
        "settings: []\n",
        data_files=["pubs.csv", "gyms.csv"],
    )
    venue_config, report = VenueManager.parse_config(block)
    assert venue_config is None
    assert problem_lines(report) == [
        ("config_file (venues.yaml): venue_types.pub.enabled", "must be true or false; got 'maybe'"),
        ("config_file (venues.yaml): venue_types.school.filename", "not set; type is enabled"),
        ("config_file (venues.yaml): settings", "must be a mapping; got sequence (list)"),
    ]
    assert warning_lines(report) == [
        ("config_file (venues.yaml): venue_types.gym.filter_values", "not set; filter_column ignored")
    ]


def test_disabled_type_without_filename_is_valid(write_config):
    block = write_config("venue_types:\n  pub:\n    enabled: false\n")
    venue_config, report = VenueManager.parse_config(block)
    assert not report.problems
    assert set(venue_config.venue_types) == {"pub"}


def test_disabled_batch_type_skips_pattern_and_level_checks(write_config):
    block = write_config(
        "venue_types:\n  pub:\n    enabled: false\n    batch_mode: true\n    filename: pubs.csv\n"
    )
    venue_config, report = VenueManager.parse_config(block)
    assert not report.problems
    assert set(venue_config.venue_types) == {"pub"}


def test_config_only_under_data_dir_is_found(working_dir):
    (working_dir / "data").mkdir()
    (working_dir / "data" / "venues.yaml").write_text("venue_types: {}\n")
    venue_config, report = VenueManager.parse_config(
        {"data_dir": "data", "config_file": "venues.yaml"}
    )
    assert not report.problems
    assert venue_config.config_path == os.path.join("data", "venues.yaml")


def test_empty_block_uses_defaults(working_dir):
    default_dir = working_dir / DEFAULT_DATA_DIR
    default_dir.mkdir(parents=True)
    (default_dir / DEFAULT_CONFIG_FILE).write_text("venue_types: {}\n")
    venue_config, report = VenueManager.parse_config({})
    assert not report.problems
    assert venue_config.data_dir == DEFAULT_DATA_DIR
    assert venue_config.config_path == os.path.join(DEFAULT_DATA_DIR, DEFAULT_CONFIG_FILE)


def test_preflight_returns_parse_config_report(write_config):
    block = write_config(
        "venue_types:\n  pub:\n    description: pubs\n"
        "  gym:\n    filename: gyms.csv\n    filter_values: [A]\n"
    )
    _, parse_report = VenueManager.parse_config(block)
    assert VenueManager.preflight(block) == parse_report


@pytest.mark.parametrize("key", ["data_dir", "config_file"])
def test_blank_block_key_uses_default(working_dir, key):
    (working_dir / DEFAULT_DATA_DIR).mkdir(parents=True)
    (working_dir / DEFAULT_DATA_DIR / DEFAULT_CONFIG_FILE).write_text("venue_types: {}\n")
    block = {"data_dir": DEFAULT_DATA_DIR, "config_file": DEFAULT_CONFIG_FILE, key: None}
    venue_config, report = VenueManager.parse_config(block)
    assert not report.problems
    assert venue_config.data_dir == DEFAULT_DATA_DIR
    assert venue_config.config_path == os.path.join(DEFAULT_DATA_DIR, DEFAULT_CONFIG_FILE)


def test_blank_settings_is_clean_and_uses_defaults(write_config):
    block = write_config("venue_types: {}\nsettings:\n  # filter_by_geography: false\n")
    venue_config, report = VenueManager.parse_config(block)
    assert not report.problems and not report.warnings
    assert venue_config.filter_by_geography is None


def test_directory_as_config_file_is_problem(working_dir):
    (working_dir / "venues.yaml").mkdir()
    venue_config, report = VenueManager.parse_config(
        {"data_dir": "data", "config_file": "venues.yaml"}
    )
    assert venue_config is None
    assert [key for key, _ in problem_lines(report)] == ["config_file"]
    assert problem_lines(report)[0][1].startswith("venues.yaml could not be read: ")


def test_non_utf8_config_file_is_problem(working_dir):
    (working_dir / "venues.yaml").write_bytes(b"venue_types: \xff\xfe\n")
    venue_config, report = VenueManager.parse_config(
        {"data_dir": "data", "config_file": "venues.yaml"}
    )
    assert venue_config is None
    assert problem_lines(report) == [
        ("config_file", "venues.yaml could not be read: not UTF-8 text")
    ]


BATCH_PUB = "venue_types:\n  pub:\n    batch_mode: true\n    filename: 'pubs_{unit}.csv'\n"
PUB_KEY = "config_file (venues.yaml): venue_types.pub"
GEO_LEVELS = frozenset({"SGU", "MGU"})


def test_batch_type_without_level_is_problem(write_config):
    block = write_config(BATCH_PUB)
    venue_config, report = VenueManager.parse_config(block, GEO_LEVELS)
    assert venue_config is None
    assert problem_lines(report) == [
        (f"{PUB_KEY}.level", "not set; batch_mode needs the geo level {unit} names")
    ]


def test_batch_level_unknown_to_geography_is_problem(write_config):
    block = write_config(BATCH_PUB + "    level: LSOA\n")
    venue_config, report = VenueManager.parse_config(block, GEO_LEVELS)
    assert venue_config is None
    assert problem_lines(report) == [
        (f"{PUB_KEY}.level", "unknown geo level 'LSOA'; expected one of ['MGU', 'SGU']")
    ]


def test_batch_level_not_a_name_is_problem(write_config):
    block = write_config(BATCH_PUB + "    level: [MGU]\n")
    _, report = VenueManager.parse_config(block, GEO_LEVELS)
    assert problem_lines(report) == [
        (f"{PUB_KEY}.level", "must be a geo level name; got ['MGU']")
    ]


def test_batch_level_unchecked_without_geo_levels(write_config):
    block = write_config(BATCH_PUB + "    level: LSOA\n")
    venue_config, report = VenueManager.parse_config(block)
    assert not report.problems and not report.warnings
    assert venue_config.venue_types["pub"]["level"] == "LSOA"


def test_level_on_non_batch_type_warns_unused(write_config):
    block = write_config(
        "venue_types:\n  pub:\n    filename: pubs.csv\n    level: MGU\n",
        data_files=["pubs.csv"],
    )
    venue_config, report = VenueManager.parse_config(block, GEO_LEVELS)
    assert not report.problems
    assert venue_config is not None
    assert warning_lines(report) == [
        (f"{PUB_KEY}.level", "unused; batch_mode is not true")
    ]


def test_1911_shaped_batch_type_is_clean(write_config):
    block = write_config(
        "venue_types:\n"
        "  household:\n"
        "    enabled: true\n"
        "    batch_mode: true\n"
        "    level: MGU\n"
        "    filename: '{unit}_loc.csv'\n"
        "    filter_column: BTCode\n"
        "    filter_values: [C]\n"
        "    is_residence: true\n"
    )
    _, report = VenueManager.parse_config(block, GEO_LEVELS)
    assert not report.problems and not report.warnings


def test_enabled_type_with_missing_file_is_problem(write_config):
    block = write_config("venue_types:\n  pub:\n    filename: pubs.csv\n")
    venue_config, report = VenueManager.parse_config(block)
    assert venue_config is None
    assert problem_lines(report) == [
        (f"{PUB_KEY}.filename", f"not found: {os.path.abspath('data/pubs.csv')}")
    ]


def test_disabled_type_with_missing_file_is_valid(write_config):
    block = write_config("venue_types:\n  pub:\n    enabled: false\n    filename: pubs.csv\n")
    venue_config, report = VenueManager.parse_config(block)
    assert not report.problems
    assert venue_config.venue_paths == {}


def test_venue_paths_hold_absolute_lists_for_enabled_file_types_only(write_config):
    block = write_config(
        "venue_types:\n"
        "  school:\n    filename: [ew.csv, sct.csv]\n"
        "  pub:\n    enabled: false\n    filename: pubs.csv\n"
        "  household:\n    batch_mode: true\n    level: MGU\n    filename: '{unit}.csv'\n",
        data_files=["ew.csv", "sct.csv"],
    )
    venue_config, report = VenueManager.parse_config(block, GEO_LEVELS)
    assert not report.problems
    assert venue_config.venue_paths == {
        "school": [os.path.abspath("data/ew.csv"), os.path.abspath("data/sct.csv")]
    }
