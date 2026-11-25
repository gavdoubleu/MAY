import logging
from pathlib import Path

import pytest

from may.preflight.preflight_coordinator import check_world_config, run_preflight
from may.preflight.report import PreflightError
from may.utils import path_resolver


@pytest.fixture(autouse=True)
def reset_resolver(monkeypatch):
    monkeypatch.setattr(path_resolver, "_resolver", None)


@pytest.fixture
def venues_dir(tmp_path):
    directory = tmp_path / "venues"
    directory.mkdir()
    return directory


def keys(findings):
    return [finding.key for finding in findings]


# Passes the timeline preflight, so tests of other blocks see only their own findings.
CONSTANT_ATTRIBUTE = Path(__file__).parents[3] / "test_data" / "preflight" / "constant_attribute.yaml"
VALID_TIMELINE = {"enabled": True, "steps": [{"type": "attribute", "config": str(CONSTANT_ATTRIBUTE)}]}
# Passes the geography preflight; there the file must exist but is not opened.
PREFLIGHT_DATA = Path(__file__).parents[3] / "test_data" / "preflight"
VALID_GEOGRAPHY = {
    "levels": ["SGU", "MGU"],
    "data_dir": str(PREFLIGHT_DATA),
    "hierarchy_file": "hierarchy.csv",
}
# Passes the population preflight with no warning; an absent block warns.
VALID_POPULATION = {"type": "matrix", "data_dir": str(PREFLIGHT_DATA)}


def test_venue_problems_are_prefixed_with_venues(venues_dir):
    config = {
        "venues": {"data_dir": str(venues_dir), "config_file": "missing.yaml"},
        "serialization": {"enabled": False},
        "geography": VALID_GEOGRAPHY,
        "population": VALID_POPULATION,
        "timeline": VALID_TIMELINE,
    }
    report = check_world_config(config)
    assert keys(report.problems) == ["venues.config_file"]


def test_absent_venues_block_checks_the_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = check_world_config({"geography": VALID_GEOGRAPHY, "population": VALID_POPULATION, "serialization": {"enabled": False}, "timeline": VALID_TIMELINE})
    assert keys(report.problems) == ["venues.config_file"]
    assert "venues_config.yaml" in report.problems[0].message
    assert "data/venues" in report.problems[0].message


def world_config(venues_dir, venues_yaml, serialization_path, serialization_yaml):
    (venues_dir / "venues.yaml").write_text(venues_yaml)
    (venues_dir / "companies.csv").write_text("geo_unit,industry_code\nA,C\n")
    serialization_path.write_text(serialization_yaml)
    return {
        "venues": {"data_dir": str(venues_dir), "config_file": "venues.yaml"},
        "serialization": {"enabled": True, "config_file": str(serialization_path)},
        "geography": VALID_GEOGRAPHY,
        "population": VALID_POPULATION,
        "timeline": VALID_TIMELINE,
    }


COMPANY_VENUES = "venue_types:\n  company:\n    filename: companies.csv\n"
WORK_SECTOR_SCHEMA = "venues:\n  types:\n    company:\n      properties: [work_sector]\n"


def test_property_missing_from_venue_files_is_a_warning(venues_dir, tmp_path):
    serialization_path = tmp_path / "serialization.yaml"
    config = world_config(venues_dir, COMPANY_VENUES, serialization_path, WORK_SECTOR_SCHEMA)
    report = check_world_config(config)
    assert report.problems == []
    assert [(finding.key, finding.message) for finding in report.warnings] == [
        (
            f"serialization.config_file ({serialization_path}): "
            "venues.types.company.properties",
            "'work_sector' not in the venue files; they carry ['geo_unit', 'industry_code']",
        )
    ]


def test_venue_problems_skip_the_gap_check(venues_dir, tmp_path):
    broken_venues = "venue_types:\n  company:\n    enabled: 'yes'\n    filename: companies.csv\n"
    config = world_config(
        venues_dir, broken_venues, tmp_path / "serialization.yaml", WORK_SECTOR_SCHEMA
    )
    report = check_world_config(config)
    assert keys(report.problems) == [
        f"venues.config_file ({venues_dir / 'venues.yaml'}): venue_types.company.enabled"
    ]
    assert report.warnings == []


@pytest.mark.parametrize(
    "serialization_block",
    [{"enabled": False, "config_file": "missing.yaml"}, {"enabled": False}],
    ids=["disabled_with_missing_file", "disabled_without_file"],
)
def test_no_serialization_findings_without_an_enabled_schema(
    venues_dir, tmp_path, serialization_block
):
    config = world_config(
        venues_dir, COMPANY_VENUES, tmp_path / "serialization.yaml", WORK_SECTOR_SCHEMA
    )
    config["serialization"] = serialization_block
    report = check_world_config(config)
    assert report.problems == []
    assert report.warnings == []


@pytest.mark.parametrize(
    "serialization_block",
    [None, {}, {"enabled": True}, {"output_dir": "out"}, {"config_file": ""}],
    ids=["blank", "empty", "enabled", "output_dir_only", "empty_config_file"],
)
def test_enabled_serialization_without_config_file_is_problem(
    venues_dir, tmp_path, serialization_block
):
    config = world_config(
        venues_dir, COMPANY_VENUES, tmp_path / "serialization.yaml", WORK_SECTOR_SCHEMA
    )
    config["serialization"] = serialization_block
    report = check_world_config(config)
    assert [(finding.key, finding.message) for finding in report.problems] == [
        ("serialization.config_file", "not set; serialisation is enabled")
    ]


def test_serialization_absent_is_problem(venues_dir, tmp_path):
    config = world_config(
        venues_dir, COMPANY_VENUES, tmp_path / "serialization.yaml", WORK_SECTOR_SCHEMA
    )
    del config["serialization"]
    report = check_world_config(config)
    assert keys(report.problems) == ["serialization.config_file"]


@pytest.mark.parametrize(
    "block_name, block_value, expected_message",
    [
        ("venues", "configs/venues_config.yaml", "must be a mapping; got scalar (str)"),
        ("venues", ["a"], "must be a mapping; got sequence (list)"),
        ("serialization", "schema.yaml", "must be a mapping; got scalar (str)"),
        ("serialization", ["a"], "must be a mapping; got sequence (list)"),
    ],
    ids=["venues_string", "venues_list", "serialization_string", "serialization_list"],
)
def test_block_not_mapping_is_problem(
    venues_dir, tmp_path, block_name, block_value, expected_message
):
    config = world_config(
        venues_dir, COMPANY_VENUES, tmp_path / "serialization.yaml", WORK_SECTOR_SCHEMA
    )
    config[block_name] = block_value
    report = check_world_config(config)
    assert [(finding.key, finding.message) for finding in report.problems] == [
        (block_name, expected_message)
    ]
    assert report.warnings == []


def test_missing_schema_file_is_problem(venues_dir, tmp_path):
    config = world_config(
        venues_dir, COMPANY_VENUES, tmp_path / "serialization.yaml", WORK_SECTOR_SCHEMA
    )
    missing_path = tmp_path / "missing.yaml"
    config["serialization"]["config_file"] = str(missing_path)
    report = check_world_config(config)
    assert [(finding.key, finding.message) for finding in report.problems] == [
        ("serialization.config_file", f"not found; tried {missing_path}")
    ]


def test_invalid_schema_yaml_is_problem(venues_dir, tmp_path):
    serialization_path = tmp_path / "serialization.yaml"
    config = world_config(venues_dir, COMPANY_VENUES, serialization_path, "venues: [unclosed\n")
    report = check_world_config(config)
    assert [(finding.key, finding.message) for finding in report.problems] == [
        (
            "serialization.config_file",
            f"{serialization_path} is not valid YAML at line 2, column 1",
        )
    ]


def test_unreadable_schema_is_problem_even_when_venues_are_broken(venues_dir, tmp_path):
    config = {
        "venues": {"data_dir": str(venues_dir), "config_file": "missing.yaml"},
        "serialization": {"enabled": True, "config_file": str(tmp_path / "missing.yaml")},
        "geography": VALID_GEOGRAPHY,
        "population": VALID_POPULATION,
        "timeline": VALID_TIMELINE,
    }
    report = check_world_config(config)
    assert keys(report.problems) == ["venues.config_file", "serialization.config_file"]


def test_gap_check_failure_is_warning(venues_dir, tmp_path):
    serialization_path = tmp_path / "serialization.yaml"
    config = world_config(venues_dir, COMPANY_VENUES, serialization_path, WORK_SECTOR_SCHEMA)
    (venues_dir / "companies.csv").unlink()
    (venues_dir / "companies.csv").mkdir()
    report = check_world_config(config)
    assert report.problems == []
    assert keys(report.warnings) == [f"serialization.config_file ({serialization_path})"]
    assert report.warnings[0].message.startswith("venue property check skipped: ")


def test_bad_enabled_is_problem_prefixed_with_serialization(venues_dir, tmp_path):
    config = world_config(
        venues_dir, COMPANY_VENUES, tmp_path / "serialization.yaml", WORK_SECTOR_SCHEMA
    )
    config["serialization"]["enabled"] = "yes"
    report = check_world_config(config)
    assert keys(report.problems) == ["serialization.enabled"]
    assert report.warnings == []


@pytest.mark.parametrize(
    "schema_yaml, section",
    [
        ("venues: [a, b]\n", "venues"),
        ("venues:\n  types: company\n", "venues.types"),
        ("venues:\n  types:\n    company:\n      properties: work_sector\n", "venues.types.company.properties"),
    ],
    ids=["venues_list", "types_string", "properties_string"],
)
def test_malformed_schema_is_problem_and_skips_gap_check(
    venues_dir, tmp_path, schema_yaml, section
):
    serialization_path = tmp_path / "serialization.yaml"
    config = world_config(venues_dir, COMPANY_VENUES, serialization_path, schema_yaml)
    report = check_world_config(config)
    assert keys(report.problems) == [f"serialization.config_file ({serialization_path}): {section}"]
    assert report.warnings == []


def test_unknown_schema_section_is_warning(venues_dir, tmp_path):
    serialization_path = tmp_path / "serialization.yaml"
    config = world_config(
        venues_dir, COMPANY_VENUES, serialization_path, "extras: {}\n" + WORK_SECTOR_SCHEMA
    )
    (venues_dir / "companies.csv").write_text("geo_unit,work_sector\nA,C\n")
    report = check_world_config(config)
    assert report.problems == []
    assert keys(report.warnings) == [f"serialization.config_file ({serialization_path}): extras"]


def test_schema_venue_type_absent_from_venues_config_is_accepted(venues_dir, tmp_path):
    """Child and route venue types exist only once the build creates them."""
    schema_yaml = "venues:\n  types:\n    classroom:\n      properties: [capacity]\n"
    config = world_config(
        venues_dir, COMPANY_VENUES, tmp_path / "serialization.yaml", schema_yaml
    )
    report = check_world_config(config)
    assert report.problems == []
    assert report.warnings == []


def test_run_preflight_raises_on_problems_after_logging(venues_dir, caplog):
    config = {
        "venues": {"data_dir": str(venues_dir), "config_file": "missing.yaml"},
        "serialization": {"enabled": False},
        "geography": VALID_GEOGRAPHY,
        "population": VALID_POPULATION,
        "timeline": VALID_TIMELINE,
    }
    with caplog.at_level(logging.INFO, logger="preflight"):
        with pytest.raises(PreflightError) as raised:
            run_preflight(config)
    assert keys(raised.value.report.problems) == ["venues.config_file"]
    assert [record.levelno for record in caplog.records] == [logging.INFO]
    assert caplog.records[0].getMessage() == raised.value.report.format()


def test_run_preflight_returns_report_on_warnings_only(venues_dir, tmp_path, caplog):
    config = world_config(
        venues_dir, COMPANY_VENUES, tmp_path / "serialization.yaml", WORK_SECTOR_SCHEMA
    )
    with caplog.at_level(logging.INFO, logger="preflight"):
        report = run_preflight(config)
    assert len(report.warnings) == 1
    assert caplog.records[0].getMessage() == report.format()


RESIDENCE_TIMELINE = {
    "enabled": True,
    "steps": [{"type": "residence_allocation", "config": "allocation_strategy.yaml"}],
}


def households_config(venues_dir, households_block, timeline=RESIDENCE_TIMELINE):
    (venues_dir / "venues.yaml").write_text("venue_types: {}\n")
    return {
        "venues": {"data_dir": str(venues_dir), "config_file": "venues.yaml"},
        "serialization": {"enabled": False},
        "households": households_block,
        "geography": VALID_GEOGRAPHY,
        "population": VALID_POPULATION,
        "timeline": timeline,
    }


def test_household_problems_are_prefixed_with_households(venues_dir):
    config = households_config(venues_dir, {"config_file": "c.yaml", "data_file": "h.csv"})
    report = check_world_config(config)
    assert ("households.data_dir", "not set; there is no default") in [
        (finding.key, finding.message) for finding in report.problems
    ]


@pytest.mark.parametrize(
    "timeline",
    [None, {"steps": []}, {"steps": [{"type": "venue_allocation"}, "not a step"]}],
    ids=["no_timeline", "no_steps", "other_steps"],
)
def test_households_not_checked_without_residence_step(venues_dir, timeline):
    config = households_config(venues_dir, "not a mapping", timeline=timeline)
    report = check_world_config(config)
    assert "households" not in keys(report.problems)


def test_households_not_mapping_is_problem(venues_dir):
    config = households_config(venues_dir, ["a"])
    report = check_world_config(config)
    assert [(finding.key, finding.message) for finding in report.problems] == [
        ("households", "must be a mapping; got sequence (list)")
    ]


@pytest.mark.parametrize(
    "timeline, expected_key, expected_message",
    [
        (["a"], "timeline", "must be a mapping; got sequence (list)"),
        ({"steps": "residence_allocation"}, "timeline.steps", "must be a list; got scalar (str)"),
        ({"steps": {"type": "residence_allocation"}}, "timeline.steps", "must be a list; got mapping (dict)"),
    ],
    ids=["timeline_list", "steps_string", "steps_mapping"],
)
def test_malformed_timeline_is_problem_and_skips_households(
    venues_dir, timeline, expected_key, expected_message
):
    config = households_config(venues_dir, "not a mapping", timeline=timeline)
    report = check_world_config(config)
    findings = [(finding.key, finding.message) for finding in report.problems]
    assert findings.count((expected_key, expected_message)) == 1
    assert "households" not in keys(report.problems)


@pytest.mark.parametrize("schema_yaml", ["", "# nothing\n", "{}\n"], ids=["empty", "comment", "empty_mapping"])
def test_empty_schema_is_problem(venues_dir, tmp_path, schema_yaml):
    serialization_path = tmp_path / "serialization.yaml"
    config = world_config(venues_dir, COMPANY_VENUES, serialization_path, schema_yaml)
    report = check_world_config(config)
    assert [(finding.key, finding.message) for finding in report.problems] == [
        ("serialization.config_file", f"{serialization_path} is empty")
    ]


def test_schema_not_mapping_is_problem(venues_dir, tmp_path):
    serialization_path = tmp_path / "serialization.yaml"
    config = world_config(venues_dir, COMPANY_VENUES, serialization_path, "- population\n")
    report = check_world_config(config)
    assert [(finding.key, finding.message) for finding in report.problems] == [
        (
            f"serialization.config_file ({serialization_path})",
            "must be a mapping; got sequence (list)",
        )
    ]
    assert report.warnings == []


def test_non_utf8_schema_is_problem(venues_dir, tmp_path):
    serialization_path = tmp_path / "serialization.yaml"
    config = world_config(venues_dir, COMPANY_VENUES, serialization_path, "")
    serialization_path.write_bytes(b"venues: \xff\xfe\n")
    report = check_world_config(config)
    assert [(finding.key, finding.message) for finding in report.problems] == [
        ("serialization.config_file", f"{serialization_path} could not be read: not UTF-8 text")
    ]


SCHOOL_VENUES = "venue_types:\n  school:\n    filename: schools.csv\n"


def child_creator_config(venues_dir, tmp_path, venues_yaml, parent_venue_type):
    (venues_dir / "venues.yaml").write_text(venues_yaml)
    (venues_dir / "schools.csv").write_text("geo_unit\nA\n")
    creator_path = tmp_path / "classrooms.yaml"
    creator_path.write_text(f"parent_venue_type: {parent_venue_type}\nchild_venue_type: classroom\n")
    return {
        "venues": {"data_dir": str(venues_dir), "config_file": "venues.yaml"},
        "serialization": {"enabled": False},
        "geography": VALID_GEOGRAPHY,
        "population": VALID_POPULATION,
        "timeline": {
            "enabled": True,
            "steps": [{"type": "child_creator", "config": str(creator_path)}],
        },
    }, creator_path


def test_timeline_step_problem_is_reported_with_step_prefix(venues_dir, tmp_path):
    config, creator_path = child_creator_config(venues_dir, tmp_path, SCHOOL_VENUES, "school")
    creator_path.write_text("parent_venue_type: school\n")
    report = check_world_config(config)
    assert keys(report.problems) == [f"timeline.steps[1] ({creator_path}): child_venue_type"]


def test_valid_child_creator_step_passes(venues_dir, tmp_path):
    config, _ = child_creator_config(venues_dir, tmp_path, SCHOOL_VENUES, "school")
    report = check_world_config(config)
    assert report.problems == []


def test_venue_type_not_in_venues_config_is_accepted(venues_dir, tmp_path):
    """Steps create venue types at runtime, so the timeline accepts any name."""
    disabled_school = "venue_types:\n  school:\n    enabled: false\n    filename: schools.csv\n"
    config, _ = child_creator_config(venues_dir, tmp_path, disabled_school, "made_at_runtime")
    report = check_world_config(config)
    assert report.problems == []


def test_broken_venues_and_broken_step_are_both_reported(venues_dir, tmp_path):
    config, creator_path = child_creator_config(venues_dir, tmp_path, SCHOOL_VENUES, "school")
    config["venues"]["config_file"] = "missing.yaml"
    creator_path.write_text("parent_venue_type: school\n")
    report = check_world_config(config)
    assert keys(report.problems) == [
        "venues.config_file",
        f"timeline.steps[1] ({creator_path}): child_venue_type",
    ]


def test_absent_timeline_is_problem(venues_dir, tmp_path):
    config, _ = child_creator_config(venues_dir, tmp_path, SCHOOL_VENUES, "school")
    del config["timeline"]
    report = check_world_config(config)
    assert keys(report.problems) == ["timeline"]


def test_households_block_without_residence_step_is_warning(venues_dir, tmp_path):
    config, _ = child_creator_config(venues_dir, tmp_path, SCHOOL_VENUES, "school")
    config["households"] = {"data_dir": "h"}
    report = check_world_config(config)
    assert report.problems == []
    assert keys(report.warnings) == ["households"]


STRATEGY = "steps:\n  - {name: everyone, type: household}\n"


def residence_config(venues_dir, tmp_path, **households_overrides):
    """A config whose households block and residence step both pass preflight."""
    households_dir = tmp_path / "households"
    households_dir.mkdir()
    (households_dir / "households.yaml").write_text(
        "categories:\n  - {name: Adults, symbol: A, attribute: age, type: numerical,"
        " numerical: {min: 18, max: null}}\n"
    )
    (households_dir / "rules.yaml").write_text("enabled: true\n")
    (households_dir / "step_strategy.yaml").write_text(STRATEGY)
    (households_dir / "h.csv").write_text("geo_unit\n")
    block = {
        "data_dir": str(households_dir), "data_file": "h.csv",
        "config_file": "households.yaml", "rules_file": "rules.yaml",
        **households_overrides,
    }
    timeline = {"enabled": True, "steps": [{"type": "residence_allocation", "config": "step_strategy.yaml"}]}
    return households_config(venues_dir, block, timeline=timeline), households_dir


def test_residence_step_found_under_households_data_dir_passes(venues_dir, tmp_path):
    config, _ = residence_config(venues_dir, tmp_path)
    report = check_world_config(config)
    assert report.problems == [] and report.warnings == []


def test_households_strategy_file_naming_another_file_warns_naming_both_keys(venues_dir, tmp_path):
    config, households_dir = residence_config(venues_dir, tmp_path, strategy_file="other.yaml")
    (households_dir / "other.yaml").write_text(STRATEGY)
    report = check_world_config(config)
    assert report.problems == []
    assert keys(report.warnings) == ["timeline.steps[1] (step_strategy.yaml): config"]
    assert "households.strategy_file" in report.warnings[0].message


def test_dead_households_strategy_file_is_problem_even_when_a_step_overrides_it(venues_dir, tmp_path):
    config, _ = residence_config(venues_dir, tmp_path, strategy_file="gone.yaml")
    report = check_world_config(config)
    assert keys(report.problems) == ["households.strategy_file"]


def test_missing_residence_step_file_is_problem(venues_dir, tmp_path):
    config, households_dir = residence_config(venues_dir, tmp_path)
    (households_dir / "step_strategy.yaml").unlink()
    report = check_world_config(config)
    assert keys(report.problems) == ["timeline.steps[1] (step_strategy.yaml): config"]


def geography_config(tmp_path, geography_block, distributor_yaml="venue_type: school\nactivity_map_key: primary_activity\n"):
    """A config whose only possible findings come from geography and one distributor step."""
    venues_dir = tmp_path / "venues"
    venues_dir.mkdir()
    (venues_dir / "venues.yaml").write_text("venue_types: {}\n")
    distributor_path = tmp_path / "distributor.yaml"
    distributor_path.write_text(distributor_yaml)
    return {
        "geography": geography_block,
        "population": VALID_POPULATION,
        "venues": {"data_dir": str(venues_dir), "config_file": "venues.yaml"},
        "serialization": {"enabled": False},
        "timeline": {"enabled": True, "steps": [{"type": "distributor", "config": str(distributor_path)}]},
    }, distributor_path


@pytest.mark.parametrize("geography_block", [None, {}], ids=["absent", "empty"])
def test_geography_is_always_checked(tmp_path, geography_block):
    config, _ = geography_config(tmp_path, geography_block)
    report = check_world_config(config)
    assert keys(report.problems) == ["geography.levels", "geography.hierarchy_file"]


def test_geography_problems_are_prefixed_with_geography(tmp_path):
    config, _ = geography_config(tmp_path, {**VALID_GEOGRAPHY, "load_all": "yes"})
    report = check_world_config(config)
    assert keys(report.problems) == ["geography.load_all"]


def test_geography_warnings_are_prefixed_with_geography(tmp_path):
    config, _ = geography_config(tmp_path, {**VALID_GEOGRAPHY, "filter": {"codes": ["A"]}})
    report = check_world_config(config)
    assert report.problems == []
    assert keys(report.warnings) == ["geography.filter.level"]


def test_geography_not_mapping_is_problem(tmp_path):
    config, _ = geography_config(tmp_path, ["SGU"])
    report = check_world_config(config)
    assert [(finding.key, finding.message) for finding in report.problems] == [
        ("geography", "must be a mapping; got sequence (list)")
    ]


UNKNOWN_LEVEL_DISTRIBUTOR = (
    "venue_type: school\nactivity_map_key: primary_activity\n"
    "venue_selection:\n  venue_geo_level: LSOA\n"
)


def test_step_naming_unknown_geo_level_is_problem(tmp_path):
    config, distributor_path = geography_config(tmp_path, VALID_GEOGRAPHY, UNKNOWN_LEVEL_DISTRIBUTOR)
    report = check_world_config(config)
    assert keys(report.problems) == [
        f"timeline.steps[1] ({distributor_path}): venue_selection.venue_geo_level"
    ]
    assert "['MGU', 'SGU']" in report.problems[0].message


def test_step_naming_known_geo_level_passes(tmp_path):
    known_level = UNKNOWN_LEVEL_DISTRIBUTOR.replace("LSOA", "MGU")
    config, _ = geography_config(tmp_path, VALID_GEOGRAPHY, known_level)
    report = check_world_config(config)
    assert report.problems == []


@pytest.mark.parametrize(
    "geography_block",
    [["SGU"], {"hierarchy_file": "hierarchy.csv"}, {**VALID_GEOGRAPHY, "load_all": "yes"}],
    ids=["not_mapping", "no_levels", "other_problem"],
)
def test_bad_geography_skips_geo_level_checks(tmp_path, geography_block):
    config, _ = geography_config(tmp_path, geography_block, UNKNOWN_LEVEL_DISTRIBUTOR)
    report = check_world_config(config)
    assert all(key.startswith("geography") for key in keys(report.problems))


def test_population_problems_are_prefixed_with_population(tmp_path):
    config, _ = geography_config(tmp_path, VALID_GEOGRAPHY)
    config["population"] = {"type": "census"}
    report = check_world_config(config)
    assert keys(report.problems) == ["population.type"]


def test_population_not_mapping_is_problem(tmp_path):
    config, _ = geography_config(tmp_path, VALID_GEOGRAPHY)
    config["population"] = "matrix"
    report = check_world_config(config)
    assert [(finding.key, finding.message) for finding in report.problems] == [
        ("population", "must be a mapping; got scalar (str)")
    ]


@pytest.mark.parametrize("population_block", [None, {}], ids=["absent", "empty"])
def test_absent_population_is_defaults_warning(tmp_path, monkeypatch, population_block):
    default_population_dir = tmp_path / "data" / "population"
    default_population_dir.mkdir(parents=True)
    for name in ("demographics_male.csv", "demographics_female.csv"):
        (default_population_dir / name).write_text("")
    monkeypatch.chdir(tmp_path)
    config, _ = geography_config(tmp_path, VALID_GEOGRAPHY)
    config["population"] = population_block
    report = check_world_config(config)
    assert report.problems == []
    assert keys(report.warnings) == ["population"]
    assert "defaults in use" in report.warnings[0].message


BATCH_POPULATION_UNKNOWN_LEVEL = {"type": "explicit_batch", "level": "LSOA"}


def test_population_batch_level_unknown_to_geography_is_problem(tmp_path):
    config, _ = geography_config(tmp_path, VALID_GEOGRAPHY)
    config["population"] = BATCH_POPULATION_UNKNOWN_LEVEL
    report = check_world_config(config)
    assert keys(report.problems) == ["population.level"]
    assert "['MGU', 'SGU']" in report.problems[0].message


@pytest.mark.parametrize(
    "geography_block",
    [["SGU"], {"hierarchy_file": "hierarchy.csv"}, {**VALID_GEOGRAPHY, "load_all": "yes"}],
    ids=["not_mapping", "no_levels", "other_problem"],
)
def test_bad_geography_skips_population_level_check(tmp_path, geography_block):
    config, _ = geography_config(tmp_path, geography_block)
    config["population"] = BATCH_POPULATION_UNKNOWN_LEVEL
    report = check_world_config(config)
    assert not any(key.startswith("population") for key in keys(report.problems))


BATCH_VENUES_UNKNOWN_LEVEL = (
    "venue_types:\n  pub:\n    batch_mode: true\n    level: LSOA\n    filename: 'pubs_{unit}.csv'\n"
)


def test_venue_batch_level_unknown_to_geography_is_problem(tmp_path):
    config, _ = geography_config(tmp_path, VALID_GEOGRAPHY)
    venues_yaml = tmp_path / "venues" / "venues.yaml"
    venues_yaml.write_text(BATCH_VENUES_UNKNOWN_LEVEL)
    report = check_world_config(config)
    assert keys(report.problems) == [f"venues.config_file ({venues_yaml}): venue_types.pub.level"]
    assert "['MGU', 'SGU']" in report.problems[0].message


@pytest.mark.parametrize(
    "geography_block",
    [["SGU"], {"hierarchy_file": "hierarchy.csv"}, {**VALID_GEOGRAPHY, "load_all": "yes"}],
    ids=["not_mapping", "no_levels", "other_problem"],
)
def test_bad_geography_skips_venue_level_check(tmp_path, geography_block):
    config, _ = geography_config(tmp_path, geography_block)
    (tmp_path / "venues" / "venues.yaml").write_text(BATCH_VENUES_UNKNOWN_LEVEL)
    report = check_world_config(config)
    assert not any(key.startswith("venues") for key in keys(report.problems))
