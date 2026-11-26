"""
`ResidentLinkedDistributor.check_config` records every problem in a resident-linked distributor
YAML without building anything; `DistributorStep.from_config` raises on the same problems, then builds it.
"""
import pytest

from may.venue_distributor import DistributorConfigError
from may.venue_distributor.resident_linked_distributor import ResidentLinkedDistributor
from tests.test_unit.may.venue_distributor.distributor_checks import KNOWN_GEOGRAPHY, UNKNOWN_GEOGRAPHY, check_config, from_config, keys, write

VALID = "distributor_type: resident_linked\ntarget_venue_type: care_home\n"


def preflight(path, context=UNKNOWN_GEOGRAPHY):
    return check_config(ResidentLinkedDistributor, path, context)


def test_a_valid_file_has_no_findings(tmp_path):
    report = preflight(write(tmp_path, VALID))

    assert report.problems == []
    assert report.warnings == []


def test_a_target_venue_type_that_is_not_a_name_is_a_problem(tmp_path):
    report = preflight(write(tmp_path, VALID.replace("care_home", "[care_home]")))

    assert keys(report.problems) == ["target_venue_type"]


def test_an_unknown_link_level_is_a_problem(tmp_path):
    report = preflight(write(tmp_path, VALID + "link_level: households\n"))

    assert keys(report.problems) == ["link_level"]
    assert "must be one of" in report.problems[0].message


@pytest.mark.parametrize("multiplier", ["-1", "two", "true"])
def test_a_multiplier_that_is_not_a_non_negative_number_is_a_problem(tmp_path, multiplier):
    report = preflight(write(tmp_path, VALID + f"multiplier: {multiplier}\n"))

    assert keys(report.problems) == ["multiplier"]


def test_a_fractional_multiplier_is_accepted(tmp_path):
    assert preflight(write(tmp_path, VALID + "multiplier: 1.5\n")).problems == []


def test_an_unknown_geography_level_is_a_problem(tmp_path):
    context = KNOWN_GEOGRAPHY

    report = preflight(write(tmp_path, VALID + "geography_level: LSOA\n"), context)

    assert keys(report.problems) == ["geography_level"]


def test_geography_level_checks_are_skipped_when_geography_is_unknown(tmp_path):
    assert preflight(write(tmp_path, VALID + "geography_level: LSOA\n")).problems == []


@pytest.mark.parametrize("text, key", [
    ("visitor_eligibility:\n  global_filters:\n    - attribute: age\n      type: ordinal\n",
     "visitor_eligibility.global_filters[1].type"),
    ("visitor_eligibility:\n  global_filters:\n    - type: categorical\n",
     "visitor_eligibility.global_filters[1].attribute"),
    ("visitor_eligibility: household\n", "visitor_eligibility"),
], ids=["filter_type", "filter_attribute", "not_mapping"])
def test_a_bad_visitor_filter_is_a_problem(tmp_path, text, key):
    assert keys(preflight(write(tmp_path, VALID + text)).problems) == [key]


def test_several_problems_in_one_file_are_all_reported(tmp_path):
    text = (
        "target_venue_type: [care_home]\nlink_level: households\nmultiplier: -1\n"
        "visitor_eligibility:\n  global_filters:\n    - attribute: age\n      type: ordinal\n"
    )

    report = preflight(write(tmp_path, text))

    assert sorted(keys(report.problems)) == [
        "link_level", "multiplier", "target_venue_type", "visitor_eligibility.global_filters[1].type",
    ]


def test_the_shipped_care_home_visits_config_passes():
    path = "configs/2021/distributors/care_home_visits_distributor.yaml"

    assert check_config(ResidentLinkedDistributor, path).problems == []


@pytest.mark.parametrize("text", [
    "",
    VALID + "link_level: households\n",
    VALID + "multiplier: two\n",
    VALID + "visitor_eligibility:\n  global_filters:\n    - attribute: age\n      type: ordinal\n",
], ids=["empty", "link_level", "multiplier", "filter_type"])
def test_from_config_raises_where_preflight_reports_a_problem(tmp_path, text):
    path = write(tmp_path, text)
    report = check_config(ResidentLinkedDistributor, path)

    with pytest.raises(DistributorConfigError) as excinfo:
        from_config(path)

    assert report.problems
    for finding in report.problems:
        assert f"{finding.key}: {finding.message}" in str(excinfo.value)


def test_from_config_builds_a_valid_file(tmp_path):
    distributor = from_config(write(tmp_path, VALID))

    assert distributor.target_venue_type == "care_home"


@pytest.mark.parametrize("value", ["[MGU]", "{name: MGU}"], ids=["list", "mapping"])
def test_a_geography_level_that_is_not_a_name_is_a_problem(tmp_path, value):
    context = KNOWN_GEOGRAPHY

    report = preflight(write(tmp_path, VALID + f"geography_level: {value}\n"), context)

    assert keys(report.problems) == ["geography_level"]
