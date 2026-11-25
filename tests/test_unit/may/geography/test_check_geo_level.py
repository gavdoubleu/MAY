"""`check_geo_level` reads a geo level through a reader and checks it against `context[Geography]`."""

import pytest

from may.geography.geography import check_geo_level
from may.preflight.config_reader import FAILED, ConfigReader
from may.preflight.context import ConfigContext, StageNotParsed
from may.preflight.report import PreflightReport
from tests.test_unit.may.config_contexts import geography_context


def check(mapping, context, key="level"):
    report = PreflightReport()
    level = check_geo_level(ConfigReader(mapping, "block", report), key, context)
    return level, [(finding.key, finding.message) for finding in report.problems]


def test_known_level_is_returned_with_no_finding():
    assert check({"level": "MGU"}, geography_context()) == ("MGU", [])


def test_unknown_level_is_problem_listing_levels():
    level, problems = check({"level": "OA"}, geography_context())
    assert level is FAILED
    assert problems == [("block.level", "unknown geo level 'OA'; expected one of ['LGU', 'MGU', 'SGU']")]


@pytest.mark.parametrize(
    "value, message",
    [(["MGU"], "must be text; got a list"), ("", "must not be empty; give a geo level name")],
)
def test_not_a_name_is_problem(value, message):
    assert check({"level": value}, geography_context()) == (FAILED, [("block.level", message)])


def test_whole_number_level_is_read_as_text():
    assert check({"level": 3}, geography_context(levels=("1", "2", "3"))) == ("3", [])


def test_absent_level_is_silent():
    assert check({}, geography_context()) == (None, [])


def test_failed_reader_is_silent():
    report = PreflightReport()
    inert = ConfigReader({"block": 3}, "", report).section("block")
    report_size = len(report.problems)
    assert check_geo_level(inert, "level", geography_context()) is FAILED
    assert len(report.problems) == report_size


def test_failed_geography_runs_only_the_name_check():
    assert check({"level": "OA"}, geography_context(levels=None)) == ("OA", [])
    assert check({"level": [1]}, geography_context(levels=None))[1] == [("block.level", "must be text; got a list")]


def test_geography_not_parsed_raises():
    with pytest.raises(StageNotParsed):
        check({"level": "MGU"}, ConfigContext())
