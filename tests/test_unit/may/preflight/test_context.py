"""`check_geo_level` reports a geo level that is not a known level name, and never raises."""
import pytest

from may.preflight.context import PreflightContext, check_geo_level
from may.preflight.report import PreflightReport

KNOWN_GEOGRAPHY = PreflightContext(geo_levels=frozenset({"SGU", "MGU"}))
NOT_NAMES = [["MGU"], {"name": "MGU"}, 3, True]


def check(geo_level, context=KNOWN_GEOGRAPHY):
    report = PreflightReport()
    check_geo_level("some.level", geo_level, context, report)
    return report


def test_a_known_level_has_no_findings():
    assert check("MGU").problems == []


def test_an_unset_level_has_no_findings():
    assert check(None).problems == []


def test_an_unknown_level_is_a_problem():
    report = check("LSOA")

    assert [finding.key for finding in report.problems] == ["some.level"]
    assert "'LSOA'" in report.problems[0].message
    assert "['MGU', 'SGU']" in report.problems[0].message


@pytest.mark.parametrize("geo_level", NOT_NAMES, ids=["list", "mapping", "int", "bool"])
@pytest.mark.parametrize("context", [KNOWN_GEOGRAPHY, PreflightContext()], ids=["known", "unknown"])
def test_a_level_that_is_not_a_name_is_a_problem(geo_level, context):
    report = check(geo_level, context)

    assert [finding.key for finding in report.problems] == ["some.level"]
    assert "must be a geo level name" in report.problems[0].message
