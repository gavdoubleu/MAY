"""
The timeline preflight reports every structure problem in a timeline block, and every
step's own findings, as findings named `timeline.steps[N] (<file>)`. `run_timeline`
runs the same parse and raises the same text when preflight is skipped.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from may import timeline
from may.preflight.context import PreflightContext
from may.preflight.report import PreflightReport


# An attribute step whose config passes preflight.
PASSING_STEP = {"type": "attribute", "config": str(Path(__file__).parents[3] / "test_data" / "preflight" / "constant_attribute.yaml")}


def keys(findings):
    return [finding.key for finding in findings]


def lines(findings):
    return [f"{finding.key}: {finding.message}" for finding in findings]


def enabled(*steps):
    return {"enabled": True, "steps": list(steps)}


class FlaggingStep:
    """Step class whose preflight reports one problem under an inner key."""
    @staticmethod
    def preflight(config_path, context):
        report = PreflightReport()
        report.problem("distribution_strategy", f"unknown in {config_path}")
        return report

    @staticmethod
    def from_config(config_path, world_config):
        raise AssertionError("not built in preflight")


def flagging_registry():
    return {
        step_type: timeline.TimelineStep(step_class=FlaggingStep, run="allocate")
        for step_type in timeline.TIMELINE_STEPS
    }


@pytest.mark.parametrize("timeline_block, key", [
    (None, "timeline"),
    ({"enabled": False, "steps": [PASSING_STEP]},
     "timeline.enabled"),
    ({"enabled": True, "steps": []}, "timeline.steps"),
    ({"enabled": True}, "timeline.steps"),
    ({"enabled": True, "steps": {"type": "attribute"}}, "timeline.steps"),
    ("timeline", "timeline"),
], ids=["absent", "disabled", "empty", "no_steps", "steps_not_list", "not_mapping"])
def test_an_unrunnable_timeline_is_a_problem_not_an_error(timeline_block, key):
    report = timeline.preflight(timeline_block, PreflightContext())

    assert keys(report.problems) == [key]


@pytest.mark.parametrize("step", ["attribute", None, ["attribute", "age.yaml"]],
                         ids=["string", "blank", "list"])
def test_a_step_that_is_not_a_mapping_is_a_problem(step):
    report = timeline.preflight(enabled(step), PreflightContext())

    assert keys(report.problems) == ["timeline.steps[1]"]
    assert "must be a mapping" in report.problems[0].message


def test_an_unknown_step_type_names_it_and_lists_the_valid_ones():
    report = timeline.preflight(
        enabled({"type": "attribut", "config": "age.yaml"}), PreflightContext()
    )

    assert keys(report.problems) == ["timeline.steps[1] (age.yaml): type"]
    message = report.problems[0].message
    assert "'attribut'" in message
    for step_type in timeline.TIMELINE_STEPS:
        assert step_type in message


@pytest.mark.parametrize("step_type", [["attribute"], {"attribute": 1}], ids=["list", "mapping"])
def test_a_step_type_that_is_a_list_or_mapping_is_a_problem(step_type):
    report = timeline.preflight(enabled({"type": step_type, "config": "age.yaml"}), PreflightContext())

    assert keys(report.problems) == ["timeline.steps[1] (age.yaml): type"]


@pytest.mark.parametrize("config", [["age.yaml"], {"file": "age.yaml"}], ids=["list", "mapping"])
def test_a_config_that_is_a_list_or_mapping_is_a_problem(config):
    report = timeline.preflight(enabled({"type": "attribute", "config": config}), PreflightContext())

    assert keys(report.problems) == ["timeline.steps[1]: config"]
    assert "must be set to a YAML file" in report.problems[0].message


@pytest.mark.parametrize("step_type", sorted(timeline.TIMELINE_STEPS))
@pytest.mark.parametrize("empty", ["absent", None, ""])
def test_every_step_type_reports_a_missing_config(step_type, empty):
    step = {"type": step_type} if empty == "absent" else {"type": step_type, "config": empty}

    report = timeline.preflight(enabled(step), PreflightContext())

    assert keys(report.problems) == ["timeline.steps[1]: config"]
    assert "must be set to a YAML file" in report.problems[0].message


def test_residence_allocation_keeps_its_example_path():
    report = timeline.preflight(enabled({"type": "residence_allocation"}), PreflightContext())

    assert "allocation_strategy.yaml" in report.problems[0].message


def test_several_structure_problems_are_all_reported():
    timeline_block = {"enabled": False, "steps": [
        PASSING_STEP,
        "distributor",
        {"type": "attribut", "config": "income.yaml"},
        {"type": "distributor"},
    ]}

    report = timeline.preflight(timeline_block, PreflightContext())

    assert keys(report.problems) == [
        "timeline.enabled",
        "timeline.steps[2]",
        "timeline.steps[3] (income.yaml): type",
        "timeline.steps[4]: config",
    ]


def test_step_findings_are_prefixed_with_position_and_file():
    timeline_block = enabled(
        {"type": "distributor", "config": "pubs.yaml"},
        {"type": "child_creator", "config": "rooms/classrooms.yaml"},
    )

    with patch.dict(timeline.TIMELINE_STEPS, flagging_registry()):
        report = timeline.preflight(timeline_block, PreflightContext())

    assert keys(report.problems) == [
        "timeline.steps[1] (pubs.yaml): distribution_strategy",
        "timeline.steps[2] (rooms/classrooms.yaml): distribution_strategy",
    ]


def test_a_step_with_a_structure_problem_is_not_preflighted():
    with patch.dict(timeline.TIMELINE_STEPS, flagging_registry()):
        report = timeline.preflight(enabled({"type": "distributor"}), PreflightContext())

    assert keys(report.problems) == ["timeline.steps[1]: config"]


def test_a_households_block_without_a_residence_step_is_a_warning():
    context = PreflightContext(households_block_present=True)

    report = timeline.preflight(enabled(PASSING_STEP), context)

    assert report.problems == []
    assert keys(report.warnings) == ["households"]
    assert "residence_allocation" in report.warnings[0].message


def test_a_households_block_with_a_residence_step_does_not_warn():
    context = PreflightContext(households_block_present=True)

    report = timeline.preflight(
        enabled({"type": "residence_allocation", "config": "s.yaml"}), context
    )

    assert report.warnings == []


def test_no_households_block_does_not_warn():
    report = timeline.preflight(
        enabled({"type": "attribute", "config": "age.yaml"}), PreflightContext()
    )

    assert report.warnings == []


@pytest.mark.parametrize("timeline_block", [
    None,
    {"enabled": False, "steps": [PASSING_STEP]},
    enabled("distributor", {"type": "attribut", "config": "age.yaml"}, {"type": "distributor"}),
    enabled({"type": ["attribute"], "config": "age.yaml"}, {"type": "attribute", "config": ["age.yaml"]}),
], ids=["absent", "disabled", "several_step_problems", "list_values"])
def test_run_timeline_without_preflight_raises_the_same_findings(timeline_block):
    report = timeline.preflight(timeline_block, PreflightContext())

    with pytest.raises(timeline.TimelineError) as excinfo:
        timeline.run_timeline(MagicMock(), {"timeline": timeline_block})

    for line in lines(report.problems):
        assert line in str(excinfo.value)


def creator_step(tmp_path, name, parent, child):
    path = tmp_path / name
    path.write_text(f"parent_venue_type: {parent}\nchild_venue_type: {child}\n")
    return {"type": "child_creator", "config": str(path)}


def test_a_venue_type_made_by_a_later_step_is_accepted(tmp_path):
    distributor = tmp_path / "teachers.yaml"
    distributor.write_text("venue_type: classroom\n")
    timeline_block = enabled(
        {"type": "distributor", "config": str(distributor)},
        creator_step(tmp_path, "classrooms.yaml", "school", "classroom"),
    )

    report = timeline.preflight(timeline_block, PreflightContext())

    assert report.problems == []
