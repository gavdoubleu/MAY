import os

import pytest

from may.geography.venue_manager import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_DATA_DIR,
    VenueManager,
)
from may.preflight.context import ConfigContext, StageNotParsed
from may.utils import path_resolver
from tests.test_unit.may.config_contexts import geography_context

BLANK_OPTIONAL = "left blank; give a value, or remove the key to use the default"
GEO_LEVELS = ("SGU", "MGU")


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


def preflight(block, context=None):
    return VenueManager.preflight(block, context or geography_context(levels=GEO_LEVELS))


def problem_lines(report):
    return [(finding.key, finding.message) for finding in report.problems]


def warning_lines(report):
    return [(finding.key, finding.message) for finding in report.warnings]


FILE_KEY = "venues.config_file (venues.yaml)"
TYPES_KEY = f"{FILE_KEY}: venue_types"
PUB_KEY = f"{TYPES_KEY}.pub"
BATCH_PUB = "venue_types:\n  pub:\n    batch_mode: true\n    filename: 'pubs_{unit}.csv'\n"


# --- block and file ------------------------------------------------------------------


def test_empty_venue_types_is_clean_and_parsed(write_config):
    venue_config, report = preflight(write_config("venue_types: {}\n"))
    assert not report.problems and not report.warnings
    assert venue_config.venue_types == {}
    assert venue_config.config_path == "venues.yaml"
    assert venue_config.data_dir == "data"


def test_blank_venue_types_is_problem(write_config):
    block = write_config("venue_types:\n  # pub:\n  #   filename: pubs.csv\n")
    venue_config, report = preflight(block)
    assert venue_config is None
    assert problem_lines(report) == [(TYPES_KEY, "left blank; give a mapping of venue types")]


def test_blank_block_is_problem():
    venue_config, report = preflight(None)
    assert venue_config is None
    assert problem_lines(report) == [("venues", "left blank; give a value")]


def test_block_not_mapping_is_problem():
    _, report = preflight(["venues.yaml"])
    assert problem_lines(report) == [("venues", "must be a mapping; got a list")]


def test_config_file_not_found_is_problem(working_dir):
    venue_config, report = preflight({"data_dir": "data", "config_file": "missing.yaml"})
    assert venue_config is None
    assert problem_lines(report) == [
        ("venues.config_file", "not found; tried missing.yaml, data/missing.yaml")
    ]


def test_invalid_yaml_is_problem_with_line_and_column(write_config):
    venue_config, report = preflight(write_config("venue_types:\n  pub: [unclosed\n"))
    assert venue_config is None
    assert problem_lines(report) == [
        ("venues.config_file", "venues.yaml is not valid YAML at line 3, column 1")
    ]


def test_empty_file_is_problem(write_config):
    venue_config, report = preflight(write_config("# nothing here\n"))
    assert venue_config is None
    assert problem_lines(report) == [("venues.config_file", "venues.yaml is empty")]


def test_top_level_not_mapping_is_problem(write_config):
    venue_config, report = preflight(write_config("- pub\n- school\n"))
    assert venue_config is None
    assert problem_lines(report) == [
        ("venues.config_file", "venues.yaml must hold a mapping; got a list")
    ]


@pytest.mark.parametrize("key", ["data_dir", "config_file"])
@pytest.mark.parametrize("value", [["venues.yaml"], {"path": "venues.yaml"}], ids=["list", "mapping"])
def test_a_path_that_is_a_list_or_mapping_is_problem(write_config, key, value):
    block = write_config("venue_types: {}\n")
    block[key] = value
    venue_config, report = preflight(block)
    assert venue_config is None
    assert [finding.key for finding in report.problems] == [f"venues.{key}"]


@pytest.mark.parametrize("key", ["data_dir", "config_file"])
def test_blank_block_key_is_problem(write_config, key):
    block = write_config("venue_types: {}\n")
    block[key] = None
    venue_config, report = preflight(block)
    assert venue_config is None
    assert problem_lines(report) == [(f"venues.{key}", BLANK_OPTIONAL)]


def test_empty_data_dir_is_problem(write_config):
    block = write_config("venue_types: {}\n")
    block["data_dir"] = ""
    _, report = preflight(block)
    assert [finding.key for finding in report.problems] == ["venues.data_dir"]


def test_whole_number_data_dir_and_config_file_read_as_text(working_dir):
    (working_dir / "2021").mkdir()
    (working_dir / "2021" / "1911").write_text("venue_types: {}\n")
    venue_config, report = preflight({"data_dir": 2021, "config_file": 1911})
    assert not report.problems
    assert venue_config.data_dir == "2021"
    assert venue_config.config_path == os.path.join("2021", "1911")


def test_config_only_under_data_dir_is_found(working_dir):
    (working_dir / "data").mkdir()
    (working_dir / "data" / "venues.yaml").write_text("venue_types: {}\n")
    venue_config, report = preflight({"data_dir": "data", "config_file": "venues.yaml"})
    assert not report.problems
    assert venue_config.config_path == os.path.join("data", "venues.yaml")


def test_empty_block_uses_defaults(working_dir):
    default_dir = working_dir / DEFAULT_DATA_DIR
    default_dir.mkdir(parents=True)
    (default_dir / DEFAULT_CONFIG_FILE).write_text("venue_types: {}\n")
    venue_config, report = preflight({})
    assert not report.problems
    assert venue_config.data_dir == DEFAULT_DATA_DIR
    assert venue_config.config_path == os.path.join(DEFAULT_DATA_DIR, DEFAULT_CONFIG_FILE)


def test_directory_as_config_file_is_problem(working_dir):
    (working_dir / "venues.yaml").mkdir()
    venue_config, report = preflight({"data_dir": "data", "config_file": "venues.yaml"})
    assert venue_config is None
    assert [key for key, _ in problem_lines(report)] == ["venues.config_file"]
    assert problem_lines(report)[0][1].startswith("venues.yaml could not be read: ")


def test_non_utf8_config_file_is_problem(working_dir):
    (working_dir / "venues.yaml").write_bytes(b"venue_types: \xff\xfe\n")
    venue_config, report = preflight({"data_dir": "data", "config_file": "venues.yaml"})
    assert venue_config is None
    assert problem_lines(report) == [
        ("venues.config_file", "venues.yaml could not be read: not UTF-8 text")
    ]


def test_missing_venue_types_is_problem(write_config):
    venue_config, report = preflight(write_config("settings: {}\n"))
    assert venue_config is None
    assert problem_lines(report) == [(TYPES_KEY, "not set; give a mapping of venue types")]


def test_venue_types_not_mapping_is_problem(write_config):
    venue_config, report = preflight(write_config("venue_types:\n  - pub\n"))
    assert venue_config is None
    assert problem_lines(report) == [(TYPES_KEY, "must be a mapping; got a list")]


def test_type_entry_not_mapping_is_problem(write_config):
    venue_config, report = preflight(write_config("venue_types:\n  pub: pubs.csv\n"))
    assert venue_config is None
    assert problem_lines(report) == [(PUB_KEY, "must be a mapping; got text 'pubs.csv'")]


def test_whole_number_type_name_read_as_text(write_config):
    block = write_config("venue_types:\n  1911:\n    enabled: false\n")
    venue_config, report = preflight(block)
    assert not report.problems
    assert set(venue_config.venue_types) == {"1911"}


# --- settings ------------------------------------------------------------------------


def test_settings_not_mapping_is_problem(write_config):
    block = write_config("venue_types: {}\nsettings:\n  - filter_by_geography\n")
    venue_config, report = preflight(block)
    assert venue_config is None
    assert problem_lines(report) == [(f"{FILE_KEY}: settings", "must be a mapping; got a list")]


def test_blank_settings_is_problem(write_config):
    block = write_config("venue_types: {}\nsettings:\n  # filter_by_geography: false\n")
    venue_config, report = preflight(block)
    assert venue_config is None
    assert problem_lines(report) == [(f"{FILE_KEY}: settings", BLANK_OPTIONAL)]


def test_filter_by_geography_not_bool_is_problem(write_config):
    block = write_config("venue_types: {}\nsettings:\n  filter_by_geography: 1\n")
    venue_config, report = preflight(block)
    assert venue_config is None
    assert problem_lines(report) == [
        (f"{FILE_KEY}: settings.filter_by_geography", "must be true or false; got a whole number 1")
    ]


def test_filter_by_geography_unset_is_none(write_config):
    venue_config, _ = preflight(write_config("venue_types: {}\n"))
    assert venue_config.filter_by_geography is None


def test_filter_by_geography_set_is_value_yaml_no_counts_as_bool(write_config):
    block = write_config("venue_types: {}\nsettings:\n  filter_by_geography: no\n")
    venue_config, report = preflight(block)
    assert not report.problems
    assert venue_config.filter_by_geography is False


# --- venue types ---------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["enabled", "batch_mode"])
def test_type_flag_not_bool_is_problem(write_config, flag):
    block = write_config(
        f"venue_types:\n  pub:\n    filename: pubs.csv\n    {flag}: 'true'\n",
        data_files=["pubs.csv"],
    )
    venue_config, report = preflight(block)
    assert venue_config is None
    assert problem_lines(report) == [
        (f"{PUB_KEY}.{flag}", "must be true or false; got text 'true'")
    ]


def test_enabled_type_without_filename_is_problem(write_config):
    venue_config, report = preflight(write_config("venue_types:\n  pub:\n    description: pubs\n"))
    assert venue_config is None
    assert problem_lines(report) == [
        (f"{PUB_KEY}.filename", "not set; give a path or a list of paths")
    ]


def test_whole_number_filename_read_as_text(write_config):
    block = write_config("venue_types:\n  pub:\n    filename: 2021\n", data_files=["2021"])
    venue_config, report = preflight(block)
    assert not report.problems
    assert venue_config.venue_paths == {"pub": [os.path.abspath("data/2021")]}


def test_filename_bad_shape_is_problem(write_config):
    _, report = preflight(write_config("venue_types:\n  pub:\n    filename: {a: 1}\n"))
    assert problem_lines(report) == [
        (f"{PUB_KEY}.filename", "must be a path or a non-empty list of paths; got a mapping")
    ]


def test_batch_filename_list_is_problem(write_config):
    block = write_config(
        "venue_types:\n  pub:\n    batch_mode: true\n    level: MGU\n    filename: ['a_{unit}.csv', 'b_{unit}.csv']\n"
    )
    venue_config, report = preflight(block)
    assert venue_config is None
    assert problem_lines(report) == [
        (
            f"{PUB_KEY}.filename",
            "batch_mode takes one pattern, not a list; got ['a_{unit}.csv', 'b_{unit}.csv']",
        )
    ]


def test_batch_filename_without_unit_placeholder_is_problem(write_config):
    block = write_config(
        "venue_types:\n  pub:\n    batch_mode: true\n    level: MGU\n    filename: pubs.csv\n"
    )
    venue_config, report = preflight(block)
    assert venue_config is None
    assert problem_lines(report) == [
        (f"{PUB_KEY}.filename", "batch_mode pattern must contain '{unit}'; got 'pubs.csv'")
    ]


def test_whole_number_batch_pattern_read_as_text(write_config):
    block = write_config("venue_types:\n  pub:\n    batch_mode: true\n    level: MGU\n    filename: 12\n")
    _, report = preflight(block)
    assert problem_lines(report) == [
        (f"{PUB_KEY}.filename", "batch_mode pattern must contain '{unit}'; got '12'")
    ]


def test_batch_without_filename_is_problem(write_config):
    _, report = preflight(write_config("venue_types:\n  pub:\n    batch_mode: true\n    level: MGU\n"))
    assert problem_lines(report) == [
        (f"{PUB_KEY}.filename", "not set; give a file pattern containing {unit}")
    ]


def test_filter_column_without_values_warns_and_still_parses(write_config):
    block = write_config(
        "venue_types:\n  pub:\n    filename: pubs.csv\n    filter_column: kind\n",
        data_files=["pubs.csv"],
    )
    venue_config, report = preflight(block)
    assert not report.problems
    assert set(venue_config.venue_types) == {"pub"}
    assert warning_lines(report) == [(f"{PUB_KEY}.filter_values", "not set; filter_column ignored")]


def test_filter_values_without_column_warns_and_still_parses(write_config):
    block = write_config(
        "venue_types:\n  pub:\n    filename: pubs.csv\n    filter_values: [A, B]\n",
        data_files=["pubs.csv"],
    )
    venue_config, report = preflight(block)
    assert not report.problems
    assert set(venue_config.venue_types) == {"pub"}
    assert warning_lines(report) == [(f"{PUB_KEY}.filter_column", "not set; filter_values ignored")]


def test_several_findings_in_one_file_all_reported(write_config):
    block = write_config(
        "venue_types:\n"
        "  pub:\n    enabled: maybe\n    filename: pubs.csv\n"
        "  school:\n    description: schools\n"
        "  gym:\n    filename: gyms.csv\n    filter_column: kind\n"
        "settings: []\n",
        data_files=["pubs.csv", "gyms.csv"],
    )
    venue_config, report = preflight(block)
    assert venue_config is None
    assert problem_lines(report) == [
        (f"{PUB_KEY}.enabled", "must be true or false; got text 'maybe'"),
        (f"{TYPES_KEY}.school.filename", "not set; give a path or a list of paths"),
        (f"{FILE_KEY}: settings", "must be a mapping; got a list"),
    ]
    assert warning_lines(report) == [
        (f"{TYPES_KEY}.gym.filter_values", "not set; filter_column ignored")
    ]


def test_failed_flag_skips_filename_checks(write_config):
    _, report = preflight(write_config("venue_types:\n  pub:\n    batch_mode: 3\n"))
    assert [finding.key for finding in report.problems] == [f"{PUB_KEY}.batch_mode"]


def test_disabled_type_without_filename_is_valid(write_config):
    venue_config, report = preflight(write_config("venue_types:\n  pub:\n    enabled: false\n"))
    assert not report.problems
    assert set(venue_config.venue_types) == {"pub"}


def test_disabled_batch_type_skips_pattern_and_level_checks(write_config):
    block = write_config(
        "venue_types:\n  pub:\n    enabled: false\n    batch_mode: true\n    filename: pubs.csv\n"
    )
    venue_config, report = preflight(block)
    assert not report.problems
    assert set(venue_config.venue_types) == {"pub"}


# --- batch level against the geography ----------------------------------------------


def test_batch_type_without_level_is_problem(write_config):
    venue_config, report = preflight(write_config(BATCH_PUB, data_files=["pubs_M1.csv"]))
    assert venue_config is None
    assert problem_lines(report) == [
        (f"{PUB_KEY}.level", "not set; give the geo level {unit} names")
    ]


def test_batch_level_unknown_to_geography_is_problem(write_config):
    block = write_config(BATCH_PUB + "    level: LSOA\n", data_files=["pubs_M1.csv"])
    venue_config, report = preflight(block)
    assert venue_config is None
    assert problem_lines(report) == [
        (f"{PUB_KEY}.level", "unknown geo level 'LSOA'; expected one of ['MGU', 'SGU']")
    ]


def test_batch_level_not_a_name_is_problem(write_config):
    block = write_config(BATCH_PUB + "    level: [MGU]\n", data_files=["pubs_M1.csv"])
    _, report = preflight(block)
    assert problem_lines(report) == [(f"{PUB_KEY}.level", "must be text; got a list")]


def test_whole_number_batch_level_read_as_text(write_config):
    block = write_config(BATCH_PUB + "    level: 2021\n", data_files=["pubs_M1.csv"])
    venue_config, report = preflight(block, geography_context(levels=("2021",)))
    assert not report.problems
    assert venue_config.venue_types["pub"]["level"] == "2021"


def test_batch_level_unchecked_when_geography_failed(write_config):
    block = write_config(BATCH_PUB + "    level: LSOA\n", data_files=["pubs_M1.csv"])
    venue_config, report = preflight(block, geography_context(levels=None))
    assert not report.problems and not report.warnings
    assert venue_config.venue_types["pub"]["level"] == "LSOA"


def test_context_without_geography_raises(write_config):
    block = write_config(BATCH_PUB + "    level: MGU\n", data_files=["pubs_M1.csv"])
    with pytest.raises(StageNotParsed):
        VenueManager.preflight(block, ConfigContext())


def test_level_on_non_batch_type_warns_unused(write_config):
    block = write_config(
        "venue_types:\n  pub:\n    filename: pubs.csv\n    level: MGU\n",
        data_files=["pubs.csv"],
    )
    venue_config, report = preflight(block)
    assert not report.problems
    assert venue_config is not None
    assert warning_lines(report) == [(f"{PUB_KEY}.level", "unused; batch_mode is not true")]


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
        "    is_residence: true\n",
        data_files=["M1_loc.csv"],
    )
    _, report = preflight(block)
    assert not report.problems and not report.warnings


# --- data files ----------------------------------------------------------------------


def test_enabled_type_with_missing_file_is_problem(write_config):
    venue_config, report = preflight(write_config("venue_types:\n  pub:\n    filename: pubs.csv\n"))
    assert venue_config is None
    assert problem_lines(report) == [
        (f"{PUB_KEY}.filename", f"not found: {os.path.abspath('data/pubs.csv')}")
    ]


def test_disabled_type_with_missing_file_is_valid(write_config):
    block = write_config("venue_types:\n  pub:\n    enabled: false\n    filename: pubs.csv\n")
    venue_config, report = preflight(block)
    assert not report.problems
    assert venue_config.venue_paths == {}


def test_venue_paths_hold_absolute_lists_for_enabled_file_types_only(write_config):
    block = write_config(
        "venue_types:\n"
        "  school:\n    filename: [ew.csv, sct.csv]\n"
        "  pub:\n    enabled: false\n    filename: pubs.csv\n"
        "  household:\n    batch_mode: true\n    level: MGU\n    filename: '{unit}.csv'\n",
        data_files=["ew.csv", "sct.csv", "M1.csv"],
    )
    venue_config, report = preflight(block)
    assert not report.problems
    assert venue_config.venue_paths == {
        "school": [os.path.abspath("data/ew.csv"), os.path.abspath("data/sct.csv")]
    }


def test_batch_pattern_matching_no_files_is_problem(write_config):
    block = write_config(BATCH_PUB + "    level: MGU\n", data_files=["other.csv"])
    venue_config, report = preflight(block)
    assert venue_config is None
    assert problem_lines(report) == [
        (f"{PUB_KEY}.filename", f"pattern 'pubs_{{unit}}.csv' matched no files in {os.path.abspath('data')}")
    ]


def test_batch_pattern_with_missing_data_dir_is_problem(write_config):
    block = write_config(BATCH_PUB + "    level: MGU\n")
    block["data_dir"] = "gone"
    _, report = preflight(block)
    assert [finding.key for finding in report.problems] == [f"{PUB_KEY}.filename"]


def test_batch_pattern_glob_characters_are_literal(write_config):
    block = write_config(
        "venue_types:\n  pub:\n    batch_mode: true\n    level: MGU\n    filename: '[x]_{unit}.csv'\n",
        data_files=["x_M1.csv"],
    )
    _, report = preflight(block)
    assert [finding.key for finding in report.problems] == [f"{PUB_KEY}.filename"]
