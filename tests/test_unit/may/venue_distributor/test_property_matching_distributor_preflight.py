"""
`PropertyMatchingDistributor.check_config` records every problem in a property-matching
distributor YAML without building anything; `DistributorStep.from_config` raises
on the same problems, then builds it.
"""
import pytest

from may.venue_distributor import DistributorConfigError
from may.venue_distributor.property_matching_distributor import PropertyMatchingDistributor
from tests.test_unit.may.venue_distributor.distributor_checks import UNKNOWN_GEOGRAPHY, check_config, from_config, keys, write

VALID = (
    "distributor_type: property_matching\n"
    "target_venue_type: [household, prison]\n"
    "mapping_key: HID\nvenue_property: HID\nactivity_name: residence\n"
)


def preflight(path, context=UNKNOWN_GEOGRAPHY):
    return check_config(PropertyMatchingDistributor, path, context)


def test_a_valid_file_has_no_findings(tmp_path):
    report = preflight(write(tmp_path, VALID))

    assert report.problems == []
    assert report.warnings == []


@pytest.mark.parametrize("key", ["mapping_key", "venue_property", "activity_name"])
def test_a_required_key_not_set_is_a_problem(tmp_path, key):
    text = "".join(line + "\n" for line in VALID.splitlines() if not line.startswith(f"{key}:"))

    assert keys(preflight(write(tmp_path, text)).problems) == [key]


def test_a_single_target_venue_type_is_accepted(tmp_path):
    text = VALID.replace("[household, prison]", "prison")

    assert preflight(write(tmp_path, text)).problems == []


def test_target_venue_type_defaults_to_household(tmp_path):
    text = VALID.replace("target_venue_type: [household, prison]\n", "")

    assert preflight(write(tmp_path, text)).problems == []


def test_a_target_venue_type_that_is_not_a_name_is_named_by_position(tmp_path):
    report = preflight(write(tmp_path, VALID.replace("prison]", "[prison]]")))

    assert keys(report.problems) == ["target_venue_type[2]"]


def test_several_problems_in_one_file_are_all_reported(tmp_path):
    text = "target_venue_type: [household, [prison]]\nmapping_key: HID\n"

    report = preflight(write(tmp_path, text))

    assert sorted(keys(report.problems)) == ["activity_name", "target_venue_type[2]", "venue_property"]


def test_the_shipped_residence_config_passes():
    path = "configs/1911/residence_distribution_config.yaml"

    assert check_config(PropertyMatchingDistributor, path).problems == []


@pytest.mark.parametrize("text", [
    "",
    VALID.replace("activity_name: residence\n", ""),
    VALID.replace("[household, prison]", "{}"),
], ids=["empty", "no_activity_name", "target_not_name"])
def test_from_config_raises_where_preflight_reports_a_problem(tmp_path, text):
    path = write(tmp_path, text)
    report = check_config(PropertyMatchingDistributor, path)

    with pytest.raises(DistributorConfigError) as excinfo:
        from_config(path)

    assert report.problems
    for finding in report.problems:
        assert f"{finding.key}: {finding.message}" in str(excinfo.value)


def test_from_config_builds_a_valid_file(tmp_path):
    distributor = from_config(write(tmp_path, VALID))

    assert distributor.target_venue_types == ["household", "prison"]
