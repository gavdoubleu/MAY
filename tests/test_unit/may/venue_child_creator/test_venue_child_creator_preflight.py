"""
`VenueChildCreator.preflight` reports every problem in a child creator YAML without
building anything; `from_config` parses the same way and raises on the same problems.
"""
import glob

import pytest

from may.preflight.context import PreflightContext
from may.venue_child_creator.venue_child_creator import VenueChildCreator, VenueChildCreatorError

VALID = "parent_venue_type: school\nchild_venue_type: classroom\n"


def keys(findings):
    return [finding.key for finding in findings]


def write(tmp_path, text, name="rooms.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def preflight(path, context=PreflightContext()):
    return VenueChildCreator.preflight(path, context)


def test_a_valid_file_has_no_findings(tmp_path):
    report = preflight(write(tmp_path, VALID + "distribution_strategy: fill\nchild_max_size: 10\n"))

    assert report.problems == []
    assert report.warnings == []


@pytest.mark.parametrize("path", sorted(glob.glob("configs/*/venue_child_creators/*.yaml"))
                         + ["configs/1911/school_classrooms.yaml"])
def test_shipped_child_creator_configs_pass(path):
    assert preflight(path).problems == []


def test_a_missing_file_is_a_problem(tmp_path):
    path = str(tmp_path / "absent.yaml")

    report = preflight(path)

    assert keys(report.problems) == ["config"]
    assert "absent.yaml" in report.problems[0].message


@pytest.mark.parametrize("text, message", [
    ("parent_venue_type: [school\n", "not valid YAML"),
    ("", "is empty"),
    ("# all commented out\n", "is empty"),
    ("- school\n- classroom\n", "must be a mapping"),
], ids=["invalid_yaml", "empty", "only_comments", "not_mapping"])
def test_an_unusable_file_is_one_problem(tmp_path, text, message):
    report = preflight(write(tmp_path, text))

    assert keys(report.problems) == ["config"]
    assert message in report.problems[0].message


@pytest.mark.parametrize("missing_key", ["parent_venue_type", "child_venue_type"])
@pytest.mark.parametrize("written", ["absent", "null", "blank"])
def test_a_required_key_not_set_is_a_problem(tmp_path, missing_key, written):
    lines = {"parent_venue_type": "school", "child_venue_type": "classroom"}
    text = "".join(f"{key}: {value}\n" for key, value in lines.items() if key != missing_key)
    if written == "null":
        text += f"{missing_key}: null\n"
    elif written == "blank":
        text += f"{missing_key}: ''\n"

    report = preflight(write(tmp_path, text))

    assert keys(report.problems) == [missing_key]
    assert "not set" in report.problems[0].message


def test_a_venue_type_that_is_not_a_string_is_a_problem(tmp_path):
    report = preflight(write(tmp_path, "parent_venue_type: school\nchild_venue_type: [classroom]\n"))

    assert keys(report.problems) == ["child_venue_type"]


def test_an_unknown_distribution_strategy_names_the_valid_ones(tmp_path):
    report = preflight(write(tmp_path, VALID + "distribution_strategy: evne\n"))

    assert keys(report.problems) == ["distribution_strategy"]
    message = report.problems[0].message
    assert "'evne'" in message and "even" in message and "fill" in message


@pytest.mark.parametrize("value", ["0", "-5", "thirty", "2.5", "true"])
def test_child_max_size_must_be_a_positive_whole_number(tmp_path, value):
    report = preflight(write(tmp_path, VALID + f"child_max_size: {value}\n"))

    assert keys(report.problems) == ["child_max_size"]


def test_member_filters_must_be_a_list(tmp_path):
    report = preflight(write(tmp_path, VALID + "member_filters:\n  attribute: age\n"))

    assert keys(report.problems) == ["member_filters"]
    assert "must be a list" in report.problems[0].message


@pytest.mark.parametrize("filter_yaml, key", [
    ("  - age\n", "member_filters[1]"),
    ("  - type: numerical\n    min: 3\n", "member_filters[1].attribute"),
    ("  - attribute: age\n    type: ordinal\n", "member_filters[1].type"),
], ids=["not_mapping", "no_attribute", "unknown_type"])
def test_a_bad_member_filter_is_named_by_position(tmp_path, filter_yaml, key):
    report = preflight(write(tmp_path, VALID + "member_filters:\n" + filter_yaml))

    assert keys(report.problems) == [key]


def test_a_member_filter_type_defaults_to_numerical(tmp_path):
    report = preflight(write(tmp_path, VALID + "member_filters:\n  - attribute: age\n    min: 3\n"))

    assert report.problems == []


def test_exclude_and_include_subset_keys_together_is_a_problem(tmp_path):
    report = preflight(write(
        tmp_path, VALID + "exclude_subset_keys: [guest]\ninclude_subset_keys: [student]\n"
    ))

    assert keys(report.problems) == ["include_subset_keys"]
    assert "exclude_subset_keys" in report.problems[0].message


def test_any_parent_venue_type_name_is_accepted(tmp_path):
    """Steps create venue types at runtime, so preflight cannot know every type."""
    report = preflight(write(tmp_path, "parent_venue_type: made_at_runtime\nchild_venue_type: classroom\n"))

    assert report.problems == []


def test_several_problems_in_one_file_are_all_reported(tmp_path):
    text = (
        "parent_venue_type: scool\n"
        "distribution_strategy: evne\n"
        "child_max_size: 0\n"
        "member_filters:\n  - attribute: age\n    type: ordinal\n"
    )

    report = preflight(write(tmp_path, text))

    assert keys(report.problems) == [
        "child_venue_type",
        "distribution_strategy",
        "child_max_size",
        "member_filters[1].type",
    ]


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
    report = VenueChildCreator.preflight(path, PreflightContext())

    with pytest.raises(VenueChildCreatorError) as excinfo:
        VenueChildCreator.from_config(path, {})

    assert report.problems
    for finding in report.problems:
        assert f"{finding.key}: {finding.message}" in str(excinfo.value)


def test_from_config_raises_on_a_missing_file(tmp_path):
    with pytest.raises(VenueChildCreatorError, match="absent.yaml"):
        VenueChildCreator.from_config(str(tmp_path / "absent.yaml"), {})


