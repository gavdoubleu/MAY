"""
`AllocationStrategy` checks an allocation-strategy YAML before any household data is
loaded, and the run builds through the same parse. The file is found with
`find_config_file(path, households.data_dir)`; there is no default path.
"""
from unittest.mock import MagicMock, patch

import pytest

from may.preflight.context import PreflightContext
from may.residence.allocation_strategy import AllocationStrategy, execute_allocation_strategy
from may.residence.household_distributor import HouseholdConfig, HouseholdError
from may.utils import path_resolver

GOOD_STRATEGY = "steps:\n  - {name: everyone, type: household}\n"


@pytest.fixture(autouse=True)
def reset_resolver(monkeypatch):
    monkeypatch.setattr(path_resolver, "_resolver", None)


@pytest.fixture
def working_dir(tmp_path, monkeypatch):
    working = tmp_path / "working"
    (working / "data").mkdir(parents=True)
    monkeypatch.chdir(working)
    return working


def households(data_dir="data", strategy_path=None):
    return HouseholdConfig(
        data_dir=data_dir, config_path="households.yaml", config={}, rules_path=None,
        data_file="households.csv", column_policy="strict", strategy_path=strategy_path,
    )


def context(**household_overrides):
    return PreflightContext(households=households(**household_overrides), households_block_present=True)


def lines(findings):
    return [(finding.key, finding.message) for finding in findings]


def test_clean_strategy_has_no_findings(working_dir):
    (working_dir / "strategy.yaml").write_text(GOOD_STRATEGY)

    report = AllocationStrategy.preflight("strategy.yaml", context())

    assert report.problems == [] and report.warnings == []


def test_strategy_is_found_under_households_data_dir(working_dir):
    (working_dir / "data" / "strategy.yaml").write_text(GOOD_STRATEGY)

    parsed, report = AllocationStrategy.parse_config("strategy.yaml", "data")

    assert report.problems == []
    assert parsed.path == "data/strategy.yaml"


def test_missing_strategy_file_is_problem(working_dir):
    report = AllocationStrategy.preflight("gone.yaml", context())

    assert lines(report.problems) == [("config", "not found; tried gone.yaml, data/gone.yaml")]


def test_invalid_yaml_is_problem(working_dir):
    (working_dir / "strategy.yaml").write_text("steps: [\n")

    report = AllocationStrategy.preflight("strategy.yaml", context())

    assert [finding.key for finding in report.problems] == ["config"]


def test_steps_not_a_list_is_problem(working_dir):
    (working_dir / "strategy.yaml").write_text("steps: household\n")

    report = AllocationStrategy.preflight("strategy.yaml", context())

    assert lines(report.problems) == [("steps", "must be a list; got scalar (str)")]


def test_no_steps_is_warning(working_dir):
    (working_dir / "strategy.yaml").write_text("enabled: true\n")

    report = AllocationStrategy.preflight("strategy.yaml", context())

    assert report.problems == []
    assert [finding.key for finding in report.warnings] == ["steps"]


def test_unknown_step_type_and_non_mapping_step_are_problems_in_one_run(working_dir):
    (working_dir / "strategy.yaml").write_text(
        "steps:\n  - {name: a, type: houshold}\n  - just a string\n  - {name: c, type: venue}\n"
    )

    report = AllocationStrategy.preflight("strategy.yaml", context())

    assert [finding.key for finding in report.problems] == ["steps[1].type", "steps[2]"]
    assert "'houshold'" in report.problems[0].message


def test_interpretation_without_mixture_is_problem(working_dir):
    (working_dir / "strategy.yaml").write_text(
        "steps:\n  - {name: a, type: household, interpretation: couple}\n"
    )

    report = AllocationStrategy.preflight("strategy.yaml", context())

    assert [finding.key for finding in report.problems] == ["mixture"]


def test_missing_mixture_file_is_problem(working_dir):
    (working_dir / "strategy.yaml").write_text(
        GOOD_STRATEGY + "mixture: {file: shares.csv, geo_level: MSOA}\n"
    )

    report = AllocationStrategy.preflight("strategy.yaml", context())

    assert [finding.key for finding in report.problems] == ["mixture.file"]


def test_mixture_file_is_checked_for_existence_only(working_dir):
    (working_dir / "shares.csv").write_text("")
    (working_dir / "strategy.yaml").write_text(
        GOOD_STRATEGY + "mixture: {file: shares.csv, geo_level: MSOA}\n"
    )

    with patch("builtins.open", wraps=open) as opened:
        report = AllocationStrategy.preflight("strategy.yaml", context())

    assert report.problems == []
    assert not any("shares.csv" in str(call.args[0]) for call in opened.call_args_list)


def test_check_is_skipped_when_households_block_failed(working_dir):
    report = AllocationStrategy.preflight(
        "gone.yaml", PreflightContext(households=None, households_block_present=True)
    )

    assert report.problems == [] and report.warnings == []


def test_households_strategy_file_naming_another_file_is_warning_naming_both_keys(working_dir):
    (working_dir / "strategy.yaml").write_text(GOOD_STRATEGY)
    (working_dir / "other.yaml").write_text(GOOD_STRATEGY)

    report = AllocationStrategy.preflight("strategy.yaml", context(strategy_path="other.yaml"))

    assert report.problems == []
    assert [finding.key for finding in report.warnings] == ["config"]
    message = report.warnings[0].message
    assert "households.strategy_file" in message and "other.yaml" in message
    assert "step's file is used" in message


def test_households_strategy_file_naming_the_same_file_does_not_warn(working_dir):
    (working_dir / "data" / "strategy.yaml").write_text(GOOD_STRATEGY)

    report = AllocationStrategy.preflight(
        "data/strategy.yaml", context(strategy_path="data/strategy.yaml")
    )

    assert report.warnings == []


def test_from_config_raises_where_preflight_reports_a_problem(working_dir):
    with pytest.raises(HouseholdError, match="gone.yaml"):
        AllocationStrategy.from_config("gone.yaml", {"households": {"data_dir": "data"}})


def test_from_config_finds_the_file_under_households_data_dir(working_dir):
    (working_dir / "data" / "strategy.yaml").write_text(GOOD_STRATEGY)

    built = AllocationStrategy.from_config("strategy.yaml", {"households": {"data_dir": "data"}})

    assert built.strategy_path == "data/strategy.yaml"


def test_allocate_sets_up_households_with_the_step_file(working_dir):
    (working_dir / "strategy.yaml").write_text(GOOD_STRATEGY)
    world_config = {"households": {"data_dir": "data"}}
    world = MagicMock()

    with patch("may.world.setup_households") as setup_households:
        AllocationStrategy.from_config("strategy.yaml", world_config).allocate(world)

    setup_households.assert_called_once_with(
        world.geography, world.population, world.venues, world_config,
        strategy_file="strategy.yaml",
    )
    assert world.household_distributor is setup_households.return_value


def test_execute_requires_a_strategy_file():
    with pytest.raises(TypeError):
        execute_allocation_strategy(MagicMock(), MagicMock(), MagicMock())


def test_execute_finds_the_file_under_the_distributor_data_dir(working_dir):
    (working_dir / "data" / "strategy.yaml").write_text("enabled: false\n")
    distributor = MagicMock(data_dir="data")

    assert execute_allocation_strategy(MagicMock(), MagicMock(), distributor, "strategy.yaml") == {}


def test_execute_raises_on_a_missing_file_without_a_data_prefix_retry(working_dir):
    (working_dir / "data" / "strategy.yaml").write_text("enabled: false\n")
    distributor = MagicMock(data_dir=None)

    with pytest.raises(HouseholdError, match="not found; tried strategy.yaml"):
        execute_allocation_strategy(MagicMock(), MagicMock(), distributor, "strategy.yaml")


@pytest.mark.parametrize("value", ["[a.csv]", "{path: a.csv}", "3"])
def test_mixture_file_that_is_not_a_path_is_problem(working_dir, value):
    (working_dir / "strategy.yaml").write_text(GOOD_STRATEGY + f"mixture: {{file: {value}, geo_level: MSOA}}\n")

    report = AllocationStrategy.preflight("strategy.yaml", context())

    assert [finding.key for finding in report.problems] == ["mixture.file"]


def test_disabled_strategy_steps_and_mixture_are_not_checked(working_dir):
    (working_dir / "strategy.yaml").write_text(
        "enabled: false\nsteps:\n  - {name: a, type: legacy_round, interpretation: couple}\n"
    )

    report = AllocationStrategy.preflight("strategy.yaml", context())

    assert report.problems == []
