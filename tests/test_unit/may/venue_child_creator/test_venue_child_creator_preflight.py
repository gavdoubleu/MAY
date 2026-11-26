"""
`VenueChildCreator` is a config stage whose config is a child creator YAML path:
`preflight` reports every problem without building; `from_config` raises on the same ones.
"""
import glob

import pytest

from may.preflight.config_stage import ConfigError
from may.preflight.context import ConfigContext
from may.venue_child_creator.venue_child_creator import (
    ChildCreatorConfigError,
    VenueChildCreator,
    VenueChildCreatorError,
)

VALID = "parent_venue_type: school\nchild_venue_type: classroom\n"
KEY_PATH = "timeline.steps[4].config"
EMPTY = ConfigContext()


def write(tmp_path, text, name="rooms.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def preflight(path):
    return VenueChildCreator.preflight(path, EMPTY, key_path=KEY_PATH)


def keys(report, path):
    """Finding keys with the file prefix taken off; a finding at the naming key stays whole."""
    prefix = f"{KEY_PATH} ({path}): "
    return [finding.key.removeprefix(prefix) for finding in report.problems]


def test_a_valid_file_gives_the_settings(tmp_path):
    settings, report = preflight(write(tmp_path, VALID + "distribution_strategy: fill\nchild_max_size: 10\n"))

    assert report.problems == [] and report.warnings == []
    assert settings["parent_venue_type"] == "school"
    assert settings["child_venue_type"] == "classroom"
    assert settings["distribution_strategy"] == "fill"
    assert settings["child_max_size"] == 10
    assert settings["remove_from_parent"] is False


def test_absent_optional_keys_take_their_defaults(tmp_path):
    settings, _ = preflight(write(tmp_path, VALID))

    assert settings["child_max_size"] == 30
    assert settings["distribution_strategy"] == "even"
    assert settings["replace_parent_activity"] is True
    assert settings["member_filters"] is None


@pytest.mark.parametrize("path", sorted(glob.glob("configs/*/venue_child_creators/*.yaml"))
                         + ["configs/1911/school_classrooms.yaml"])
def test_shipped_child_creator_configs_pass(path):
    _, report = preflight(path)

    assert report.problems == []


def test_preflight_is_callable_without_a_timeline(tmp_path):
    path = write(tmp_path, VALID + "child_max_size: 0\n")

    _, report = VenueChildCreator.preflight(path, EMPTY, key_path="config")

    assert [finding.key for finding in report.problems] == [f"config ({path}): child_max_size"]


def test_a_missing_file_is_one_problem_at_the_naming_key(tmp_path):
    path = str(tmp_path / "absent.yaml")

    settings, report = preflight(path)

    assert settings is None
    assert keys(report, path) == [KEY_PATH]
    assert "absent.yaml" in report.problems[0].message


@pytest.mark.parametrize("text, message", [
    ("parent_venue_type: [school\n", "not valid YAML"),
    ("", "is empty"),
    ("# all commented out\n", "is empty"),
    ("- school\n- classroom\n", "must hold a mapping"),
], ids=["invalid_yaml", "empty", "only_comments", "not_mapping"])
def test_an_unusable_file_is_one_problem(tmp_path, text, message):
    path = write(tmp_path, text)

    _, report = preflight(path)

    assert keys(report, path) == [KEY_PATH]
    assert message in report.problems[0].message


@pytest.mark.parametrize("missing_key", ["parent_venue_type", "child_venue_type"])
@pytest.mark.parametrize("written, message", [
    ("absent", "not set; give a venue type name"),
    ("null", "left blank; give a venue type name"),
    ("empty", "must not be empty; give a venue type name"),
])
def test_a_required_key_not_set_is_a_problem(tmp_path, missing_key, written, message):
    lines = {"parent_venue_type": "school", "child_venue_type": "classroom"}
    text = "".join(f"{key}: {value}\n" for key, value in lines.items() if key != missing_key)
    if written == "null":
        text += f"{missing_key}: null\n"
    elif written == "empty":
        text += f"{missing_key}: ''\n"
    path = write(tmp_path, text)

    _, report = preflight(path)

    assert keys(report, path) == [missing_key]
    assert report.problems[0].message == message


def test_a_venue_type_that_is_not_text_is_a_problem(tmp_path):
    path = write(tmp_path, "parent_venue_type: school\nchild_venue_type: [classroom]\n")

    _, report = preflight(path)

    assert keys(report, path) == ["child_venue_type"]
    assert report.problems[0].message == "must be text; got a list"


def test_a_whole_number_venue_type_is_read_as_text(tmp_path):
    settings, report = preflight(write(tmp_path, "parent_venue_type: 1911\nchild_venue_type: classroom\n"))

    assert report.problems == []
    assert settings["parent_venue_type"] == "1911"


def test_an_unknown_distribution_strategy_names_the_valid_ones(tmp_path):
    path = write(tmp_path, VALID + "distribution_strategy: evne\n")

    _, report = preflight(path)

    assert keys(report, path) == ["distribution_strategy"]
    assert report.problems[0].message == "must be one of even, fill; got text 'evne'"


@pytest.mark.parametrize("value, message", [
    ("0", "must be at least 1; got 0"),
    ("-5", "must be at least 1; got -5"),
    ("thirty", "must be a whole number; got text 'thirty'"),
    ("2.5", "must be a whole number; got a number 2.5"),
    ("true", "must be a whole number; got true"),
])
def test_child_max_size_must_be_a_positive_whole_number(tmp_path, value, message):
    path = write(tmp_path, VALID + f"child_max_size: {value}\n")

    _, report = preflight(path)

    assert keys(report, path) == ["child_max_size"]
    assert report.problems[0].message == message


def test_member_filters_must_be_a_list(tmp_path):
    path = write(tmp_path, VALID + "member_filters:\n  attribute: age\n")

    _, report = preflight(path)

    assert keys(report, path) == ["member_filters"]
    assert report.problems[0].message == "must be a list; got a mapping"


def test_blank_member_filters_is_a_problem(tmp_path):
    path = write(tmp_path, VALID + "member_filters:\n")

    _, report = preflight(path)

    assert keys(report, path) == ["member_filters"]
    assert report.problems[0].message.startswith("left blank")


@pytest.mark.parametrize("filter_yaml, key", [
    ("  - age\n", "member_filters[1]"),
    ("  - type: numerical\n    min: 3\n", "member_filters[1].attribute"),
    ("  - attribute: age\n    type: ordinal\n", "member_filters[1].type"),
], ids=["not_mapping", "no_attribute", "unknown_type"])
def test_a_bad_member_filter_is_named_by_position(tmp_path, filter_yaml, key):
    path = write(tmp_path, VALID + "member_filters:\n" + filter_yaml)

    _, report = preflight(path)

    assert keys(report, path) == [key]


def test_member_filters_keep_their_written_settings(tmp_path):
    settings, report = preflight(write(tmp_path, VALID + "member_filters:\n  - attribute: age\n    min: 3\n"))

    assert report.problems == []
    assert settings["member_filters"] == [{"attribute": "age", "min": 3}]


def test_whole_number_member_filter_attribute_reaches_the_build_as_text(tmp_path):
    settings, report = preflight(write(tmp_path, VALID + "member_filters:\n  - attribute: 7\n    min: 3\n"))

    assert report.problems == []
    assert settings["member_filters"] == [{"attribute": "7", "min": 3}]


def test_exclude_and_include_subset_keys_together_is_a_problem(tmp_path):
    path = write(tmp_path, VALID + "exclude_subset_keys: [guest]\ninclude_subset_keys: [student]\n")

    _, report = preflight(path)

    assert keys(report, path) == ["include_subset_keys"]
    assert "exclude_subset_keys" in report.problems[0].message


def test_any_parent_venue_type_name_is_accepted(tmp_path):
    """Steps create venue types at runtime, so preflight cannot know every type."""
    _, report = preflight(write(tmp_path, "parent_venue_type: made_at_runtime\nchild_venue_type: classroom\n"))

    assert report.problems == []


def test_several_problems_in_one_file_are_all_reported(tmp_path):
    text = (
        "parent_venue_type: scool\n"
        "distribution_strategy: evne\n"
        "child_max_size: 0\n"
        "member_filters:\n  - attribute: age\n    type: ordinal\n"
    )
    path = write(tmp_path, text)

    _, report = preflight(path)

    assert keys(report, path) == [
        "child_venue_type",
        "distribution_strategy",
        "child_max_size",
        "member_filters[1].type",
    ]


def test_the_config_error_is_a_child_creator_error():
    """The timeline's `expected_errors` still match a config failure."""
    assert issubclass(ChildCreatorConfigError, ConfigError)
    assert issubclass(ChildCreatorConfigError, VenueChildCreatorError)


@pytest.mark.parametrize("text", [
    "",
    "child_venue_type: classroom\n",
    VALID + "distribution_strategy: evne\n",
    VALID + "child_max_size: 0\n",
    VALID + "member_filters:\n  - attribute: age\n    type: ordinal\n",
    VALID + "exclude_subset_keys: [guest]\ninclude_subset_keys: [student]\n",
], ids=["empty", "no_parent", "strategy", "max_size", "filter_type", "subset_keys"])
def test_from_config_raises_where_preflight_reports_a_problem(tmp_path, text):
    path = write(tmp_path, text)
    _, report = preflight(path)

    with pytest.raises(ChildCreatorConfigError) as excinfo:
        VenueChildCreator.from_config(path, EMPTY, key_path=KEY_PATH)

    assert report.problems
    assert str(excinfo.value).startswith("Child creator config cannot be used:")
    for finding in report.problems:
        assert f"{finding.key}: {finding.message}" in str(excinfo.value)


def test_from_config_raises_on_a_missing_file(tmp_path):
    with pytest.raises(ChildCreatorConfigError, match="absent.yaml"):
        VenueChildCreator.from_config(str(tmp_path / "absent.yaml"), EMPTY, key_path=KEY_PATH)


def test_from_config_builds_from_the_settings(tmp_path):
    creator = VenueChildCreator.from_config(
        write(tmp_path, VALID + "child_max_size: 12\n"), EMPTY, key_path=KEY_PATH
    )

    assert isinstance(creator, VenueChildCreator)
    assert creator.child_max_size == 12
