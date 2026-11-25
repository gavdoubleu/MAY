"""Acceptance: the shipped 2021 config passes preflight, and a broken copy of it has
every planted step finding reported in one run, named by its key."""
import shutil

import pytest
import yaml

from may.preflight.preflight_coordinator import check_world_config
from may.timeline import TIMELINE_STEPS
from may.utils import path_resolver as pr

SCENARIO_DIR = "configs/2021"

# One planted finding per timeline step type: (step file, step number, mutate, key).
PLANTED_FINDINGS = {
    "residence_allocation": (
        "households/allocation_strategy.yaml", 1,
        lambda config: config["steps"][0].update(type="householdz"),
        "steps[1].type",
    ),
    "attribute": (
        "attributes/attribute_assignment.yaml", 2,
        lambda config: config["step"].update(assignment_level="by_guesswork"),
        "step.assignment_level",
    ),
    "distributor": (
        "distributors/school_distributor.yaml", 4,
        lambda config: config.setdefault("allocation", {}).update(strategy="clsest"),
        "allocation.strategy",
    ),
    "child_creator": (
        "venue_child_creators/school_classrooms.yaml", 5,
        lambda config: config.update(distribution_strategy="evens"),
        "distribution_strategy",
    ),
}


@pytest.fixture
def scenario_config():
    """The 2021 world config, with path tokens resolved against a chosen config root."""
    previous_resolver = pr._resolver
    with open(f"{SCENARIO_DIR}/config.yaml", encoding="utf-8-sig") as config_file:
        world_config = yaml.safe_load(config_file)

    def with_config_root(config_root):
        pr.init(config_root=str(config_root), data_root=world_config.get("data_root"), output_root="output")
        return world_config

    yield with_config_root
    pr._resolver = previous_resolver


def _plant(scenario_copy, relative_path, mutate):
    path = scenario_copy / relative_path
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    mutate(config)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


# Steps migrated to config stages key findings at `config`, after the found file.
STAGE_STEP_TYPES = {"child_creator"}


def _planted_key(step_type, scenario_copy, relative_path, step_number, key):
    if step_type in STAGE_STEP_TYPES:
        return f"timeline.steps[{step_number}].config ({scenario_copy / relative_path}): {key}"
    return f"timeline.steps[{step_number}] (${{config_root}}/{relative_path}): {key}"


def test_a_finding_is_planted_for_every_step_type():
    assert set(PLANTED_FINDINGS) == set(TIMELINE_STEPS)


def test_the_shipped_2021_config_passes_preflight(scenario_config):
    report = check_world_config(scenario_config(SCENARIO_DIR))

    assert report.problems == []


def test_every_planted_step_finding_is_reported_in_one_run(scenario_config, tmp_path):
    scenario_copy = tmp_path / "2021"
    shutil.copytree(SCENARIO_DIR, scenario_copy)
    for relative_path, _, mutate, _ in PLANTED_FINDINGS.values():
        _plant(scenario_copy, relative_path, mutate)

    report = check_world_config(scenario_config(scenario_copy))

    problem_keys = {finding.key for finding in report.problems}
    expected_keys = {
        _planted_key(step_type, scenario_copy, relative_path, step_number, key)
        for step_type, (relative_path, step_number, _, key) in PLANTED_FINDINGS.items()
    }
    assert expected_keys <= problem_keys
    assert len(report.problems) == len(PLANTED_FINDINGS)
