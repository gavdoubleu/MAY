"""
`RouteDistributor.preflight` reports every problem in a route distributor YAML without
building anything; `from_config` parses the same way and raises on the same problems.
"""
import glob

import pytest

from may.preflight.context import PreflightContext
from may.utils import path_resolver as pr
from may.venue_distributor import DistributorConfigError
from may.venue_distributor.route_distributor import RouteDistributor

SOURCES = (
    "origin_source:\n  type: ancestor\n  from: geographical_unit\n  level: MGU\n"
    "destination_source:\n  type: ancestor\n  from: properties.workplace_mgu\n  level: MGU\n"
)
POOL = (
    "pool_rule:\n  corridor_level: LGU\n  speed_kmh: 30\n"
    "  min_duration_min: 5\n  max_duration_min: 60\n"
)
VALID = "distributor_type: route\nleg_venue_type: bus_line\n" + SOURCES + POOL
KNOWN_GEOGRAPHY = PreflightContext(geo_levels=frozenset({"SGU", "MGU", "LGU"}))


def keys(findings):
    return [finding.key for finding in findings]


def write(tmp_path, text, name="distributor.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def preflight(path, context=PreflightContext()):
    return RouteDistributor.preflight(path, context)


def table(tmp_path, extra=""):
    """A table-routed train config whose legs table exists."""
    (tmp_path / "legs.csv").write_text("")
    return (
        f"distributor_type: route\nleg_venue_type: train_line\nlegs_table: {tmp_path / 'legs.csv'}\n"
        + SOURCES + extra
    )


def test_a_valid_pool_rule_file_has_no_findings(tmp_path):
    report = preflight(write(tmp_path, VALID), KNOWN_GEOGRAPHY)

    assert report.problems == []
    assert report.warnings == []


def test_a_valid_legs_table_file_has_no_findings(tmp_path):
    text = table(tmp_path, "catchment:\n  max_access_km: 10\n  access_speed_kmh: 30\n")

    assert preflight(write(tmp_path, text)).problems == []


def test_a_leg_venue_type_that_is_not_a_name_is_a_problem(tmp_path):
    report = preflight(write(tmp_path, VALID.replace("bus_line", "[bus_line]")))

    assert keys(report.problems) == ["leg_venue_type"]


@pytest.mark.parametrize("value", ["[legs.csv]", "{path: legs.csv}"], ids=["list", "mapping"])
def test_a_legs_table_that_is_a_list_or_mapping_is_a_problem(tmp_path, value):
    text = table(tmp_path).replace(str(tmp_path / "legs.csv"), value)

    assert keys(preflight(write(tmp_path, text)).problems) == ["legs_table"]


def test_a_missing_legs_table_is_a_problem(tmp_path):
    text = table(tmp_path).replace(str(tmp_path / "legs.csv"), str(tmp_path / "absent.csv"))

    report = preflight(write(tmp_path, text))

    assert keys(report.problems) == ["legs_table"]
    assert "absent.csv" in report.problems[0].message


def test_a_legs_table_beside_a_pool_rule_is_a_problem(tmp_path):
    report = preflight(write(tmp_path, VALID + "legs_table: legs.csv\n"))

    assert keys(report.problems) == ["legs_table"]
    assert "pool_rule" in report.problems[0].message


@pytest.mark.parametrize("setting", ["speed_kmh", "min_duration_min", "max_duration_min"])
def test_a_pool_rule_setting_not_set_is_a_problem(tmp_path, setting):
    text = "".join(line + "\n" for line in VALID.splitlines() if not line.strip().startswith(f"{setting}:"))

    assert keys(preflight(write(tmp_path, text)).problems) == [f"pool_rule.{setting}"]


def test_a_pool_rule_setting_that_is_not_a_number_is_a_problem(tmp_path):
    report = preflight(write(tmp_path, VALID.replace("speed_kmh: 30", "speed_kmh: fast")))

    assert keys(report.problems) == ["pool_rule.speed_kmh"]


@pytest.mark.parametrize("setting", ["max_access_km", "access_speed_kmh"])
def test_a_catchment_setting_not_a_number_is_a_problem(tmp_path, setting):
    other = "access_speed_kmh" if setting == "max_access_km" else "max_access_km"
    text = table(tmp_path, f"catchment:\n  {other}: 10\n")

    assert keys(preflight(write(tmp_path, text)).problems) == [f"catchment.{setting}"]


@pytest.mark.parametrize("text, key", [
    (VALID.replace("LGU", "Region"), "pool_rule.corridor_level"),
    (VALID.replace("  level: MGU\ndestination", "  level: msoa\ndestination"), "origin_source.level"),
], ids=["corridor_level", "source_level"])
def test_an_unknown_geo_level_is_a_problem(tmp_path, text, key):
    assert keys(preflight(write(tmp_path, text), KNOWN_GEOGRAPHY).problems) == [key]


def test_geo_level_checks_are_skipped_when_geography_is_unknown(tmp_path):
    assert preflight(write(tmp_path, VALID.replace("LGU", "Region"))).problems == []


def test_an_unknown_source_type_is_a_problem(tmp_path):
    text = VALID.replace("  type: ancestor\n  from: properties", "  type: ancestors\n  from: properties")

    report = preflight(write(tmp_path, text))

    assert keys(report.problems) == ["destination_source.type"]
    assert "expected one of" in report.problems[0].message


def test_a_source_that_is_not_a_mapping_is_a_problem(tmp_path):
    text = VALID.replace(
        "origin_source:\n  type: ancestor\n  from: geographical_unit\n  level: MGU\n", "origin_source: MGU\n"
    )

    assert keys(preflight(write(tmp_path, text)).problems) == ["origin_source"]


def test_several_problems_in_one_file_are_all_reported(tmp_path):
    text = (
        VALID.replace("bus_line", "[bus_line]").replace("speed_kmh: 30", "speed_kmh: fast")
        + "legs_table: legs.csv\n"
    )

    report = preflight(write(tmp_path, text))

    assert sorted(keys(report.problems)) == ["leg_venue_type", "legs_table", "pool_rule.speed_kmh"]


@pytest.fixture
def scenario_roots():
    """Path tokens resolved as `create_world` does, from each scenario's `data_root`."""
    previous = pr._resolver
    yield lambda data_root: pr.init(config_root="configs", data_root=data_root, output_root="output")
    pr._resolver = previous


@pytest.mark.parametrize("path", sorted(glob.glob("configs/2021/distributors/route_*.yaml")))
def test_shipped_route_configs_pass(path, scenario_roots):
    scenario_roots("data")

    assert RouteDistributor.preflight(path, PreflightContext()).problems == []


@pytest.mark.parametrize("text", [
    "",
    VALID.replace("speed_kmh: 30", "speed_kmh: fast"),
    VALID + "catchment:\n  max_access_km: far\n  access_speed_kmh: 30\n",
    "distributor_type: route\nlegs_table: absent.csv\n",
], ids=["empty", "pool_speed", "catchment", "missing_table"])
def test_from_config_raises_where_preflight_reports_a_problem(tmp_path, text):
    path = write(tmp_path, text)
    report = RouteDistributor.preflight(path, PreflightContext())

    with pytest.raises(DistributorConfigError) as excinfo:
        RouteDistributor.from_config(path, {})

    assert report.problems
    for finding in report.problems:
        assert f"{finding.key}: {finding.message}" in str(excinfo.value)


def test_from_config_builds_a_valid_file(tmp_path):
    distributor = RouteDistributor.from_config(write(tmp_path, VALID), {})

    assert distributor.leg_venue_type == "bus_line"
