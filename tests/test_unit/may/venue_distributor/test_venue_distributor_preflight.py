"""
`VenueDistributor.check_config` records every problem in a single-venue distributor YAML
without building anything; `DistributorStep.from_config` raises on the same problems, then builds it.
"""
import glob

import pytest
import yaml

from may.utils import path_resolver as pr
from may.venue_distributor import DistributorConfigError
from may.venue_distributor.venue_distributor import VenueDistributor
from tests.test_unit.may.venue_distributor.distributor_checks import KNOWN_GEOGRAPHY, UNKNOWN_GEOGRAPHY, check_config, from_config, keys, write

VALID = "venue_type: school\nactivity_map_key: primary_activity\n"


def preflight(path, context=UNKNOWN_GEOGRAPHY):
    return check_config(VenueDistributor, path, context)


def test_a_valid_file_has_no_findings(tmp_path):
    report = preflight(write(tmp_path, VALID))

    assert report.problems == []
    assert report.warnings == []


def test_venue_type_not_set_is_a_problem(tmp_path):
    report = preflight(write(tmp_path, "activity_map_key: primary_activity\n"))

    assert keys(report.problems) == ["venue_type"]


def test_any_venue_type_name_is_accepted(tmp_path):
    """Steps create venue types at runtime, so preflight cannot know every type."""
    assert preflight(write(tmp_path, "venue_type: made_at_runtime\n")).problems == []


@pytest.mark.parametrize("key", ["venue_geo_level", "batch_geo_level"])
def test_an_unknown_geo_level_is_a_problem(tmp_path, key):
    context = KNOWN_GEOGRAPHY
    text = VALID + f"venue_selection:\n  {key}: LSOA\n"

    report = preflight(write(tmp_path, text), context)

    assert keys(report.problems) == [f"venue_selection.{key}"]
    assert "'LSOA'" in report.problems[0].message
    assert "['MGU', 'SGU']" in report.problems[0].message


def test_geo_level_checks_are_skipped_when_geography_is_unknown(tmp_path):
    report = preflight(write(tmp_path, VALID + "venue_selection:\n  venue_geo_level: LSOA\n"))

    assert report.problems == []


GROUP = "eligibility:\n  priority_allocation:\n    groups:\n      - name: young\n        filters:\n"


@pytest.mark.parametrize("text, key", [
    ("eligibility:\n  global_filters:\n    - attribute: age\n      type: ordinal\n",
     "eligibility.global_filters[1].type"),
    ("eligibility:\n  global_filters:\n    - type: numerical\n",
     "eligibility.global_filters[1].attribute"),
    ("eligibility:\n  global_filters:\n    - age\n", "eligibility.global_filters[1]"),
    ("eligibility:\n  global_filters: age\n", "eligibility.global_filters"),
    (GROUP + "          - attribute: age\n            type: ordinal\n",
     "eligibility.priority_allocation.groups[1].filters[1].type"),
    ("eligibility:\n  attributes:\n    - name: sex\n      type: ordinal\n",
     "eligibility.attributes[1].type"),
    ("eligibility:\n  attributes:\n    - name: sex\n", "eligibility.attributes[1].type"),
    ("eligibility:\n  attributes:\n    - type: categorical\n", "eligibility.attributes[1].name"),
], ids=["filter_type", "filter_attribute", "filter_not_mapping", "filters_not_list",
        "group_filter_type", "rule_type", "rule_type_unset", "rule_name"])
def test_a_bad_filter_is_named_by_position(tmp_path, text, key):
    report = preflight(write(tmp_path, VALID + text))

    assert keys(report.problems) == [key]


def test_a_filter_type_defaults_to_numerical(tmp_path):
    text = VALID + "eligibility:\n  global_filters:\n    - attribute: age\n      min: 5\n"

    assert preflight(write(tmp_path, text)).problems == []


SPECIAL = "special_cases:\n  - name: boarders\n    allocation_rule:\n"


@pytest.mark.parametrize("text, key", [
    ("allocation:\n  strategy: clsest\n", "allocation.strategy"),
    ("allocation:\n  when_full: spill\n", "allocation.when_full"),
    ("allocation:\n  batch_by: msoa\n", "allocation.batch_by"),
    ("allocation:\n  capacity_handling:\n    if_missing: guess\n", "allocation.capacity_handling.if_missing"),
    ("allocation:\n  capacity_handling:\n    if_zero: default\n", "allocation.capacity_handling.if_zero"),
    ("venue_selection:\n  consider_by: radius\n", "venue_selection.consider_by"),
    ("fallback:\n  strategy: retry\n", "fallback.strategy"),
    (SPECIAL + "      strategy: nearest\n", "special_cases[1].allocation_rule.strategy"),
    (SPECIAL + "      if_no_match: ignore\n", "special_cases[1].allocation_rule.if_no_match"),
    (SPECIAL + "      match_by:\n        - source: person.residence.name\n          match_type: fuzzy\n",
     "special_cases[1].allocation_rule.match_by[1].match_type"),
], ids=["strategy", "when_full", "batch_by", "if_missing", "if_zero", "consider_by",
        "fallback", "special_strategy", "if_no_match", "match_type"])
def test_an_unknown_value_names_the_valid_ones(tmp_path, text, key):
    report = preflight(write(tmp_path, VALID + text))

    assert keys(report.problems) == [key]
    assert "must be one of" in report.problems[0].message


@pytest.mark.parametrize("text, key", [
    ("fallback:\n  strategy: [skip]\n", "fallback.strategy"),
    ("allocation:\n  strategy: {random: 1}\n", "allocation.strategy"),
    (SPECIAL + "      strategy: [closest]\n", "special_cases[1].allocation_rule.strategy"),
], ids=["list", "mapping", "special_case_list"])
def test_a_closed_vocabulary_value_that_is_a_list_or_mapping_is_a_problem(tmp_path, text, key):
    report = preflight(write(tmp_path, VALID + text))

    assert keys(report.problems) == [key]
    assert "must be one of" in report.problems[0].message


def test_capacity_proportional_needs_batching_by_geo_unit(tmp_path):
    text = VALID + "allocation:\n  strategy: capacity_proportional\n  batch_by: none\n"

    report = preflight(write(tmp_path, text))

    assert keys(report.problems) == ["allocation.strategy"]
    assert "batch_by" in report.problems[0].message


def test_capacity_proportional_is_known_when_batching_by_geo_unit(tmp_path):
    text = VALID + "allocation:\n  strategy: capacity_proportional\n"

    assert preflight(write(tmp_path, text)).problems == []


def test_a_special_case_with_strategy_and_match_by_is_a_problem(tmp_path):
    text = VALID + SPECIAL + "      strategy: closest\n      match_by:\n        - source: person.residence.name\n"

    report = preflight(write(tmp_path, text))

    assert keys(report.problems) == ["special_cases[1].allocation_rule"]


def probability_group(probability_config):
    return VALID + (
        "eligibility:\n  priority_allocation:\n    enabled: true\n    groups:\n"
        "      - name: young\n        probability_config:\n"
        + "".join(f"          {line}\n" for line in probability_config)
    )


PROBABILITY_KEY = "eligibility.priority_allocation.groups[1].probability_config"


def test_probability_files_that_exist_have_no_findings(tmp_path):
    (tmp_path / "rates.csv").write_text("not, a, real\ncsv\x00")
    text = probability_group(["type: file", f"file_path: {tmp_path / 'rates.csv'}", "probability_column: rate"])

    assert preflight(write(tmp_path, text)).problems == []


def test_a_missing_probability_file_is_a_problem(tmp_path):
    text = probability_group(["type: file", f"file_path: [{tmp_path / 'absent.csv'}]", "probability_column: rate"])

    report = preflight(write(tmp_path, text))

    assert keys(report.problems) == [f"{PROBABILITY_KEY}.file_path"]
    assert "absent.csv" in report.problems[0].message


@pytest.mark.parametrize("lines, key", [
    (["type: file", "probability_column: rate"], "file_path"),
    (["type: file", "file_path: rates.csv"], "probability_column"),
    (["type: file", "file_path: []", "probability_column: rate"], "file_path"),
], ids=["no_file_path", "no_column", "empty_file_list"])
def test_a_probability_file_setting_not_set_is_a_problem(tmp_path, lines, key):
    (tmp_path / "rates.csv").write_text("")
    report = preflight(write(tmp_path, probability_group(
        [line.replace("rates.csv", str(tmp_path / "rates.csv")) for line in lines]
    )))

    assert keys(report.problems) == [f"{PROBABILITY_KEY}.{key}"]


def test_a_retired_probability_key_is_a_problem(tmp_path):
    (tmp_path / "rates.csv").write_text("")
    text = probability_group([
        "type: file", f"file_path: {tmp_path / 'rates.csv'}", "probability_column: rate", "default: 0.5",
    ])

    report = preflight(write(tmp_path, text))

    assert keys(report.problems) == [f"{PROBABILITY_KEY}.default"]


def test_probability_files_are_not_checked_when_priority_allocation_is_off(tmp_path):
    text = probability_group(["type: file", "file_path: absent.csv"]).replace("enabled: true", "enabled: false")

    assert preflight(write(tmp_path, text)).problems == []


def _single_venue_configs():
    paths = sorted(glob.glob("configs/*/distributors/*.yaml") + glob.glob("configs/1911/*distributor*.yaml"))
    return [path for path in paths if "distributor_type" not in yaml.safe_load(open(path))]


@pytest.fixture
def scenario_roots():
    """Path tokens resolved as `create_world` does, from each scenario's `data_root`."""
    previous = pr._resolver
    yield lambda data_root: pr.init(config_root="configs", data_root=data_root, output_root="output")
    pr._resolver = previous


@pytest.mark.parametrize("path", _single_venue_configs())
def test_shipped_single_venue_configs_pass(path, scenario_roots):
    scenario_roots("data" if path.startswith("configs/2021") else None)

    assert check_config(VenueDistributor, path).problems == []


def test_several_problems_in_one_file_are_all_reported(tmp_path):
    text = (
        "venue_type: [school]\n"
        "allocation:\n  strategy: clsest\n"
        "fallback:\n  strategy: retry\n"
        "eligibility:\n  global_filters:\n    - attribute: age\n      type: ordinal\n"
    )

    report = preflight(write(tmp_path, text))

    assert keys(report.problems) == [
        "eligibility.global_filters[1].type",
        "venue_type",
        "allocation.strategy",
        "fallback.strategy",
    ]


@pytest.mark.parametrize("text", [
    "",
    "activity_map_key: primary_activity\n",
    VALID + "allocation:\n  strategy: clsest\n",
    VALID + "eligibility:\n  global_filters:\n    - attribute: age\n      type: ordinal\n",
    VALID + SPECIAL + "      if_no_match: ignore\n",
    probability_group(["type: file", "file_path: absent.csv", "probability_column: rate"]),
], ids=["empty", "no_venue_type", "strategy", "filter_type", "if_no_match", "probability_file"])
def test_from_config_raises_where_preflight_reports_a_problem(tmp_path, text):
    path = write(tmp_path, text)
    report = check_config(VenueDistributor, path)

    with pytest.raises(DistributorConfigError) as excinfo:
        from_config(path)

    assert report.problems
    for finding in report.problems:
        assert f"{finding.key}: {finding.message}" in str(excinfo.value)


def test_from_config_builds_a_valid_file(tmp_path):
    distributor = from_config(write(tmp_path, VALID))

    assert distributor.venue_type == "school"


@pytest.mark.parametrize("key", ["venue_geo_level", "batch_geo_level"])
@pytest.mark.parametrize("value", ["[MGU]", "{name: MGU}"], ids=["list", "mapping"])
def test_a_geo_level_that_is_not_a_name_is_a_problem(tmp_path, key, value):
    context = KNOWN_GEOGRAPHY

    report = preflight(write(tmp_path, VALID + f"venue_selection:\n  {key}: {value}\n"), context)

    assert keys(report.problems) == [f"venue_selection.{key}"]
