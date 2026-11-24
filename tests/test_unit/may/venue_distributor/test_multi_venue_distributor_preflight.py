"""
`MultiVenueDistributor.preflight` reports every problem in a multi-venue distributor YAML
without building anything; `from_config` parses the same way and raises on the same problems.
"""
import glob

import pytest
import yaml

from may.preflight.context import PreflightContext
from may.utils import path_resolver as pr
from may.venue_distributor import DistributorConfigError
from may.venue_distributor.multi_venue_distributor import MultiVenueDistributor

VALID = "distributor_type: multi_venue\nactivity_map_key: leisure\nvenue_types: [cinema, gym]\n"


def keys(findings):
    return [finding.key for finding in findings]


def write(tmp_path, text, name="distributor.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def preflight(path, context=PreflightContext()):
    return MultiVenueDistributor.preflight(path, context)


def participation(tmp_path, lines):
    """A gym participation filter; `DATA` in `lines` is replaced by an existing CSV."""
    (tmp_path / "gym.csv").write_text("")
    body = "".join(f"      {line}\n" for line in lines).replace("DATA", str(tmp_path / "gym.csv"))
    return VALID + "venue_type_config:\n  gym:\n    participation_filter:\n" + body


PARTICIPATION = ["data_file: DATA", "probability_column:", "  column_name: rate"]
FILTER_KEY = "venue_type_config.gym.participation_filter"


def test_a_valid_file_has_no_findings(tmp_path):
    report = preflight(write(tmp_path, VALID))

    assert report.problems == []
    assert report.warnings == []


def test_activity_map_key_not_set_is_a_problem(tmp_path):
    report = preflight(write(tmp_path, VALID.replace("activity_map_key: leisure\n", "")))

    assert keys(report.problems) == ["activity_map_key"]


@pytest.mark.parametrize("venue_types", ["", "venue_types: []\n", "venue_types: cinema\n"],
                         ids=["absent", "empty", "not_list"])
def test_venue_types_must_be_a_non_empty_list(tmp_path, venue_types):
    text = VALID.replace("venue_types: [cinema, gym]\n", venue_types)

    assert keys(preflight(write(tmp_path, text)).problems) == ["venue_types"]


def test_a_venue_type_that_is_not_a_name_is_named_by_position(tmp_path):
    report = preflight(write(tmp_path, VALID.replace("gym", "[gym]")))

    assert keys(report.problems) == ["venue_types[2]"]


def test_an_unknown_consider_by_is_a_problem(tmp_path):
    report = preflight(write(tmp_path, VALID + "venue_selection:\n  consider_by: distance\n"))

    assert keys(report.problems) == ["venue_selection.consider_by"]
    assert "expected one of" in report.problems[0].message


@pytest.mark.parametrize("consider_by, strategy", [("geo_unit", "random"), ("geo_unit", "closest_balanced")])
def test_a_strategy_is_known_when_considering_by_geo_unit(tmp_path, consider_by, strategy):
    text = VALID + f"venue_selection:\n  consider_by: {consider_by}\nallocation:\n  strategy: {strategy}\n"

    assert preflight(write(tmp_path, text)).problems == []


@pytest.mark.parametrize("consider_by, strategy", [
    ("geo_unit", "closest"), ("geo_unit", None), ("count", "random"),
], ids=["unknown_with_geo_unit", "missing_with_geo_unit", "set_with_count"])
def test_a_strategy_that_does_not_fit_consider_by_is_a_problem(tmp_path, consider_by, strategy):
    text = VALID + f"venue_selection:\n  consider_by: {consider_by}\n"
    if strategy:
        text += f"allocation:\n  strategy: {strategy}\n"

    assert keys(preflight(write(tmp_path, text)).problems) == ["allocation.strategy"]


def test_venue_type_config_for_a_type_not_in_venue_types_is_a_problem(tmp_path):
    report = preflight(write(tmp_path, VALID + "venue_type_config:\n  pub:\n    count: 1\n"))

    assert keys(report.problems) == ["venue_type_config.pub"]
    assert "venue_types" in report.problems[0].message


def test_a_participation_filter_with_its_file_present_has_no_findings(tmp_path):
    assert preflight(write(tmp_path, participation(tmp_path, PARTICIPATION))).problems == []


@pytest.mark.parametrize("lines, key", [
    (PARTICIPATION[1:], "data_file"),
    (["data_file: absent.csv"] + PARTICIPATION[1:], "data_file"),
    (["data_file: [gym.csv]"] + PARTICIPATION[1:], "data_file"),
    (["data_file: {path: gym.csv}"] + PARTICIPATION[1:], "data_file"),
    (PARTICIPATION[:1], "probability_column"),
    (PARTICIPATION[:2] + ["  person_attribute: sex"], "probability_column"),
    (PARTICIPATION + ["row_filters:", "  - person_attribute: age", "    match_type: fuzzy"],
     "row_filters[1].match_type"),
], ids=["no_data_file", "missing_data_file", "list_data_file", "mapping_data_file", "no_probability_column", "no_column_rule", "match_type"])
def test_a_bad_participation_filter_is_a_problem(tmp_path, lines, key):
    report = preflight(write(tmp_path, participation(tmp_path, lines)))

    assert keys(report.problems) == [f"{FILTER_KEY}.{key}"]


def test_several_problems_in_one_file_are_all_reported(tmp_path):
    text = (
        "venue_types: [cinema, [gym]]\n"
        "venue_selection:\n  consider_by: distance\n"
        "venue_type_config:\n  pub:\n    count: 1\n"
    )

    report = preflight(write(tmp_path, text))

    assert sorted(keys(report.problems)) == [
        "activity_map_key", "venue_selection.consider_by", "venue_type_config.pub", "venue_types[2]",
    ]


@pytest.fixture
def scenario_roots():
    """Path tokens resolved as `create_world` does, from each scenario's `data_root`."""
    previous = pr._resolver
    yield lambda data_root: pr.init(config_root="configs", data_root=data_root, output_root="output")
    pr._resolver = previous


def _multi_venue_configs():
    paths = sorted(glob.glob("configs/*/distributors/*.yaml") + glob.glob("configs/1911/*distributor*.yaml"))
    return [path for path in paths if yaml.safe_load(open(path)).get("distributor_type") == "multi_venue"]


@pytest.mark.parametrize("path", _multi_venue_configs())
def test_shipped_multi_venue_configs_pass(path, scenario_roots):
    scenario_roots("data" if path.startswith("configs/2021") else None)

    assert MultiVenueDistributor.preflight(path, PreflightContext()).problems == []


@pytest.mark.parametrize("text", [
    "",
    VALID.replace("activity_map_key: leisure\n", ""),
    VALID + "venue_selection:\n  consider_by: distance\n",
    VALID + "venue_type_config:\n  pub:\n    count: 1\n",
    VALID + "venue_type_config:\n  gym:\n    participation_filter:\n      data_file: absent.csv\n",
], ids=["empty", "no_activity_map_key", "consider_by", "stray_type_config", "participation_file"])
def test_from_config_raises_where_preflight_reports_a_problem(tmp_path, text):
    path = write(tmp_path, text)
    report = MultiVenueDistributor.preflight(path, PreflightContext())

    with pytest.raises(DistributorConfigError) as excinfo:
        MultiVenueDistributor.from_config(path, {})

    assert report.problems
    for finding in report.problems:
        assert f"{finding.key}: {finding.message}" in str(excinfo.value)


def test_from_config_builds_a_valid_file(tmp_path):
    distributor = MultiVenueDistributor.from_config(write(tmp_path, VALID), {})

    assert distributor.venue_types == ["cinema", "gym"]
