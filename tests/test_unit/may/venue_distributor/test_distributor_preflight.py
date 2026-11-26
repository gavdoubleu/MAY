"""
`distributor_preflight` reads `distributor_type` and hands the file to that class's
`preflight`; `distributor_from_yaml` parses the same way and raises on the same problems.
"""
import pytest

from may.preflight.context import PreflightContext
from may.venue_distributor import DistributorConfigError, distributor_from_yaml, distributor_preflight

NO_CONTEXT = PreflightContext()


def keys(findings):
    return [finding.key for finding in findings]


def write(tmp_path, text, name="distributor.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_an_unknown_distributor_type_is_a_problem(tmp_path):
    report = distributor_preflight(write(tmp_path, "distributor_type: multivenue\n"), NO_CONTEXT)

    assert keys(report.problems) == ["distributor_type"]
    assert "'multivenue'" in report.problems[0].message
    assert "single_venue" in report.problems[0].message


@pytest.mark.parametrize("value", ["[single_venue]", "{single_venue: 1}"], ids=["list", "mapping"])
def test_a_distributor_type_that_is_a_list_or_mapping_is_a_problem(tmp_path, value):
    report = distributor_preflight(write(tmp_path, f"distributor_type: {value}\n"), NO_CONTEXT)

    assert keys(report.problems) == ["distributor_type"]


def test_distributor_from_yaml_raises_on_an_unknown_type(tmp_path):
    path = write(tmp_path, "distributor_type: multivenue\n")

    with pytest.raises(DistributorConfigError, match="distributor_type: unknown distributor type 'multivenue'"):
        distributor_from_yaml(path)


@pytest.mark.parametrize("type_line", ["", "distributor_type: single_venue\n"], ids=["absent", "explicit"])
def test_a_single_venue_file_gets_the_venue_distributor_checks(tmp_path, type_line):
    report = distributor_preflight(write(tmp_path, type_line + "venue_type: [school]\n"), NO_CONTEXT)

    assert keys(report.problems) == ["venue_type"]


def test_a_missing_file_is_one_problem(tmp_path):
    report = distributor_preflight(str(tmp_path / "absent.yaml"), NO_CONTEXT)

    assert keys(report.problems) == ["config"]


def test_distributor_from_yaml_raises_where_the_class_preflight_reports(tmp_path):
    path = write(tmp_path, "venue_type: school\nallocation:\n  strategy: clsest\n")

    with pytest.raises(DistributorConfigError, match="allocation.strategy"):
        distributor_from_yaml(path)


BLANK_SECTIONS = {
    "venue_selection": "venue_type: school\nvenue_selection:\n",
    "allocation": "venue_type: school\nallocation:\n",
    "eligibility": "venue_type: school\neligibility:\n",
    "eligibility.global_filters": "venue_type: school\neligibility:\n  global_filters:\n",
    "special_cases": "venue_type: school\nspecial_cases:\n",
    "settings": "venue_type: school\nsettings:\n",
    "validation": "venue_type: school\nvalidation:\n",
}


@pytest.mark.parametrize("key", BLANK_SECTIONS, ids=list(BLANK_SECTIONS))
def test_a_section_left_blank_is_one_problem(tmp_path, key):
    report = distributor_preflight(write(tmp_path, BLANK_SECTIONS[key]), NO_CONTEXT)

    assert keys(report.problems) == [key]
    assert "blank" in report.problems[0].message


def test_distributor_from_yaml_raises_on_a_blank_section_instead_of_building(tmp_path):
    path = write(tmp_path, BLANK_SECTIONS["venue_selection"])

    with pytest.raises(DistributorConfigError, match="venue_selection: left blank"):
        distributor_from_yaml(path)


def test_settings_that_are_not_a_mapping_are_one_problem(tmp_path):
    report = distributor_preflight(write(tmp_path, "venue_type: school\nsettings: 3\n"), NO_CONTEXT)

    assert keys(report.problems) == ["settings"]
