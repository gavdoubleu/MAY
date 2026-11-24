"""
`SocialContactVisitDistributor.preflight` reports every problem in a social contact visit
YAML without building anything; `from_config` parses the same way and raises on the same problems.
"""
import pytest

from may.preflight.context import PreflightContext
from may.venue_distributor import DistributorConfigError
from may.venue_distributor.social_contact_visit_distributor import SocialContactVisitDistributor

VALID = (
    "distributor_type: social_contact_visit\n"
    "source:\n  property_key: social_contacts\n"
    "max_contacts: 5\n"
    "eligibility:\n  global_filters:\n    - attribute: age\n      min: 16\n"
)


def keys(findings):
    return [finding.key for finding in findings]


def write(tmp_path, text, name="distributor.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def preflight(path):
    return SocialContactVisitDistributor.preflight(path, PreflightContext())


def test_a_valid_file_has_no_findings(tmp_path):
    report = preflight(write(tmp_path, VALID))

    assert report.problems == []
    assert report.warnings == []


def test_a_file_that_is_not_a_mapping_is_a_problem(tmp_path):
    assert preflight(write(tmp_path, "- social_contacts\n")).problems


@pytest.mark.parametrize("text, key", [
    ("source: social_contacts\n", "source"),
    ("source:\n  property_key: ''\n", "source.property_key"),
], ids=["source_not_mapping", "property_key_empty"])
def test_a_bad_source_is_a_problem(tmp_path, text, key):
    assert keys(preflight(write(tmp_path, text)).problems) == [key]


@pytest.mark.parametrize("max_contacts", ["0", "-2", "2.5", "true", "many"])
def test_max_contacts_must_be_a_positive_whole_number(tmp_path, max_contacts):
    report = preflight(write(tmp_path, f"max_contacts: {max_contacts}\n"))

    assert keys(report.problems) == ["max_contacts"]


@pytest.mark.parametrize("text, key", [
    ("eligibility: adults\n", "eligibility"),
    ("eligibility:\n  global_filters: age\n", "eligibility.global_filters"),
    ("eligibility:\n  global_filters:\n    - age\n", "eligibility.global_filters[1]"),
    ("eligibility:\n  global_filters:\n    - min: 16\n", "eligibility.global_filters[1].attribute"),
    ("eligibility:\n  global_filters:\n    - attribute: age\n", "eligibility.global_filters[1]"),
], ids=["eligibility_not_mapping", "filters_not_list", "filter_not_mapping", "no_attribute", "no_condition"])
def test_a_bad_eligibility_filter_is_a_problem(tmp_path, text, key):
    assert keys(preflight(write(tmp_path, text)).problems) == [key]


def test_several_problems_in_one_file_are_all_reported(tmp_path):
    text = "source: social_contacts\nmax_contacts: 0\neligibility:\n  global_filters:\n    - min: 16\n"

    report = preflight(write(tmp_path, text))

    assert sorted(keys(report.problems)) == ["eligibility.global_filters[1].attribute", "max_contacts", "source"]


@pytest.mark.parametrize("text", [
    "",
    "source: social_contacts\n",
    "max_contacts: 0\n",
    "eligibility:\n  global_filters:\n    - attribute: age\n",
], ids=["empty", "source", "max_contacts", "no_condition"])
def test_from_config_raises_where_preflight_reports_a_problem(tmp_path, text):
    path = write(tmp_path, text)
    report = preflight(path)

    with pytest.raises(DistributorConfigError) as excinfo:
        SocialContactVisitDistributor.from_config(path, {})

    assert report.problems
    for finding in report.problems:
        assert f"{finding.key}: {finding.message}" in str(excinfo.value)


def test_from_config_builds_a_valid_file(tmp_path):
    distributor = SocialContactVisitDistributor.from_config(write(tmp_path, VALID), {})

    assert distributor.max_contacts == 5
