"""
`AttributeAssignmentConfig.preflight` reports every problem in an attribute YAML without
loading its data; `from_config` parses the same way and raises on the same problems.
"""
import builtins
import glob
from unittest.mock import MagicMock

import pytest
import yaml

from may.attribute_assignment import AttributeAssignmentConfig, AttributeAssignmentError
from may.preflight.context import PreflightContext
from may.utils import path_resolver as pr

KNOWN_GEOGRAPHY = PreflightContext(geo_levels=frozenset({"SGU", "MGU", "LGU"}))
RULE_KEY = "assignment_rules.Family.rules[1].assignment"


def keys(findings):
    return [finding.key for finding in findings]


def valid_config(tmp_path):
    """A small config whose data files exist."""
    (tmp_path / "geo.csv").write_text("geo_unit,W\nA,1\n")
    (tmp_path / "pair.csv").write_text("geo_unit,first,W\nA,W,1\n")
    return {
        "attributes": [{"name": "ethnicity"}],
        "step": {"assignment_level": "person_by_residence"},
        "required_attributes": [{"name": "age"}],
        "data_sources": {
            "geo": {
                "type": "csv_lookup",
                "format": "multi_key",
                "files": [{
                    "path": str(tmp_path / "geo.csv"),
                    "key_columns": {"sgu": {
                        "attribute": "geographical_unit", "type": "ancestor_lookup", "level": "SGU",
                    }},
                }],
            },
            "pair": {
                "type": "csv_lookup",
                "format": "pair",
                "files": [{"path": str(tmp_path / "pair.csv")}],
            },
        },
        "assignment_rules": {"Family": {"rules": [
            {"role": "primary_adult", "assignment": {"strategy": "probabilistic", "data_source": "geo"}},
            {"role": "secondary_adult", "assignment": {
                "strategy": "partnership", "data_source": "pair", "partner_role": "primary_adult",
            }},
        ]}},
    }


def write(tmp_path, config, name="attributes.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config))
    return str(path)


def preflight(tmp_path, config, context=KNOWN_GEOGRAPHY):
    return AttributeAssignmentConfig.preflight(write(tmp_path, config), context)


def first_assignment(config):
    return config["assignment_rules"]["Family"]["rules"][0]["assignment"]


def test_a_valid_file_has_no_findings(tmp_path):
    report = preflight(tmp_path, valid_config(tmp_path))

    assert report.problems == []
    assert report.warnings == []


def test_a_missing_file_is_a_problem(tmp_path):
    report = AttributeAssignmentConfig.preflight(str(tmp_path / "absent.yaml"), KNOWN_GEOGRAPHY)

    assert keys(report.problems) == ["config"]


def test_the_retired_attribute_block_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    config["attribute"] = {"name": "ethnicity"}

    report = preflight(tmp_path, config)

    assert keys(report.problems) == ["attribute"]
    assert "retired" in report.problems[0].message


@pytest.mark.parametrize("attributes", [None, [], {"name": "ethnicity"}], ids=["absent", "empty", "mapping"])
def test_attributes_must_be_a_non_empty_list(tmp_path, attributes):
    config = valid_config(tmp_path)
    config["attributes"] = attributes

    assert keys(preflight(tmp_path, config).problems) == ["attributes"]


def test_an_attribute_without_a_name_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    config["attributes"] = [{"name": "ethnicity"}, {"description": "no name"}]

    assert keys(preflight(tmp_path, config).problems) == ["attributes[2].name"]


def test_an_unknown_assignment_level_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    config["step"]["assignment_level"] = "household"

    report = preflight(tmp_path, config)

    assert keys(report.problems) == ["step.assignment_level"]
    assert "person_by_residence" in report.problems[0].message


def test_a_step_that_is_not_a_mapping_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    config["step"] = ["person"]

    assert keys(preflight(tmp_path, config).problems) == ["step"]


def test_required_attributes_in_the_mapping_form_are_a_problem(tmp_path):
    config = valid_config(tmp_path)
    config["required_attributes"] = {"age": {"required": True}}

    assert keys(preflight(tmp_path, config).problems) == ["required_attributes"]


def test_a_required_attribute_without_a_name_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    config["required_attributes"] = [{"required": True}]

    assert keys(preflight(tmp_path, config).problems) == ["required_attributes[1].name"]


def test_an_unknown_strategy_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    first_assignment(config)["strategy"] = "probabilistc"

    report = preflight(tmp_path, config)

    assert keys(report.problems) == [f"{RULE_KEY}.strategy"]
    assert "probabilistic" in report.problems[0].message


def test_an_assignment_without_a_strategy_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    del first_assignment(config)["strategy"]

    assert keys(preflight(tmp_path, config).problems) == [f"{RULE_KEY}.strategy"]


def test_an_assignment_that_is_not_a_mapping_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    config["assignment_rules"]["Family"]["rules"][0]["assignment"] = "probabilistic"

    assert keys(preflight(tmp_path, config).problems) == [RULE_KEY]


def test_a_key_the_strategy_does_not_read_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    first_assignment(config)["context"] = "household.geo_unit"

    report = preflight(tmp_path, config)

    assert keys(report.problems) == [f"{RULE_KEY}.context"]


def test_probabilistic_conditions_needs_a_selection_method(tmp_path):
    config = valid_config(tmp_path)
    first_assignment(config)["strategy"] = "probabilistic_conditions"

    assert keys(preflight(tmp_path, config).problems) == [f"{RULE_KEY}.selection_method"]


def test_an_unknown_selection_method_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    first_assignment(config).update(strategy="probabilistic_conditions", selection_method="coin_flip")

    assert keys(preflight(tmp_path, config).problems) == [f"{RULE_KEY}.selection_method"]


LIST_OR_MAPPING = pytest.mark.parametrize(
    "value", [["primary_adult"], {"name": "primary_adult"}], ids=["list", "mapping"]
)


@LIST_OR_MAPPING
def test_a_strategy_that_is_a_list_or_mapping_is_a_problem(tmp_path, value):
    config = valid_config(tmp_path)
    first_assignment(config)["strategy"] = value

    assert keys(preflight(tmp_path, config).problems) == [f"{RULE_KEY}.strategy"]


@LIST_OR_MAPPING
def test_a_selection_method_that_is_a_list_or_mapping_is_a_problem(tmp_path, value):
    config = valid_config(tmp_path)
    first_assignment(config).update(strategy="probabilistic_conditions", selection_method=value)

    assert keys(preflight(tmp_path, config).problems) == [f"{RULE_KEY}.selection_method"]


@LIST_OR_MAPPING
def test_a_data_source_that_is_a_list_or_mapping_is_a_problem(tmp_path, value):
    config = valid_config(tmp_path)
    first_assignment(config)["data_source"] = value

    assert keys(preflight(tmp_path, config).problems) == [f"{RULE_KEY}.data_source"]


@LIST_OR_MAPPING
def test_a_partner_role_that_is_a_list_or_mapping_is_a_problem(tmp_path, value):
    config = valid_config(tmp_path)
    config["assignment_rules"]["Family"]["rules"][1]["assignment"]["partner_role"] = value

    assert keys(preflight(tmp_path, config).problems) == [
        "assignment_rules.Family.rules[2].assignment.partner_role"
    ]


def test_a_role_that_is_a_mapping_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    config["assignment_rules"]["Family"]["rules"][0]["role"] = {"name": "primary_adult"}

    assert keys(preflight(tmp_path, config).problems) == ["assignment_rules.Family.rules[1].role"]


def test_a_role_list_entry_that_is_not_a_name_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    config["assignment_rules"]["Family"]["rules"][0]["role"] = [["primary_adult"]]

    assert keys(preflight(tmp_path, config).problems) == ["assignment_rules.Family.rules[1].role[1]"]


def test_a_role_list_of_names_is_accepted(tmp_path):
    config = valid_config(tmp_path)
    config["assignment_rules"]["Family"]["rules"][0]["role"] = ["primary_adult", "lone_adult"]

    assert preflight(tmp_path, config).problems == []


def inheritance(logic):
    return {"strategy": "inheritance", "inherit_from": {"role": "primary_adult"}, "logic": logic}


def child_rule(config, logic):
    config["assignment_rules"]["Family"]["rules"].append({"role": "child", "assignment": inheritance(logic)})


CHILD_KEY = "assignment_rules.Family.rules[3].assignment"


def test_an_unknown_logic_entry_key_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    child_rule(config, [{"when": {"unique_count": 1}, "then": "values[0]", "otherwise": "W"}])

    assert keys(preflight(tmp_path, config).problems) == [f"{CHILD_KEY}.logic[1]"]


def test_an_unknown_then_key_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    child_rule(config, [{"when": {"unique_count": 1}, "then": {"value": "W", "weight": 2}}])

    assert keys(preflight(tmp_path, config).problems) == [f"{CHILD_KEY}.logic[1].then"]


def test_an_unknown_when_predicate_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    child_rule(config, [{"when": {"unique_counts": 1}, "then": "values[0]"}])

    report = preflight(tmp_path, config)

    assert keys(report.problems) == [f"{CHILD_KEY}.logic[1].when"]
    assert "unknown inheritance 'when' predicate" in report.problems[0].message


def test_an_unknown_then_block_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    child_rule(config, [{"when": {"unique_count": 1}, "then": {"value": "W", "exclude": []}}])

    report = preflight(tmp_path, config)

    assert keys(report.problems) == [f"{CHILD_KEY}.logic[1].then"]
    assert "unknown inheritance 'then' block" in report.problems[0].message


@LIST_OR_MAPPING
def test_an_exclude_entry_that_is_a_list_or_mapping_is_a_problem(tmp_path, value):
    config = valid_config(tmp_path)
    child_rule(config, [{"when": {"unique_count": 1}, "then": {"strategy": "probabilistic",
                                                             "data_source": "geo", "exclude": [value]}}])

    assert keys(preflight(tmp_path, config).problems) == [f"{CHILD_KEY}.logic[1].then.exclude[1]"]


def test_an_undeclared_data_source_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    first_assignment(config)["data_source"] = "goe"

    report = preflight(tmp_path, config)

    assert keys(report.problems) == [f"{RULE_KEY}.data_source"]
    assert "['geo', 'pair']" in report.problems[0].message


def test_a_venue_assignment_rule_is_checked(tmp_path):
    config = valid_config(tmp_path)
    config["venue_assignment_rules"] = [{"venue_type": "care_home", "assignment": {"strategy": "nope"}}]

    assert keys(preflight(tmp_path, config).problems) == ["venue_assignment_rules[1].assignment.strategy"]


def test_a_reference_to_an_undefined_role_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    del config["assignment_rules"]["Family"]["rules"][0]

    report = preflight(tmp_path, config)

    assert keys(report.problems) == ["assignment_rules.Family"]
    assert "undefined role 'primary_adult'" in report.problems[0].message


def test_cyclic_role_references_are_a_problem(tmp_path):
    config = valid_config(tmp_path)
    first_assignment(config).update(strategy="partnership", partner_role="secondary_adult", data_source="pair")

    report = preflight(tmp_path, config)

    assert keys(report.problems) == ["assignment_rules.Family"]
    assert "cycle" in report.problems[0].message


def test_an_unknown_data_source_type_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    config["data_sources"]["geo"]["type"] = "sql"

    assert keys(preflight(tmp_path, config).problems) == ["data_sources.geo.type"]


def test_a_constant_data_source_is_not_checked_further(tmp_path):
    config = valid_config(tmp_path)
    config["data_sources"]["fixed"] = {"type": "constant", "value": "W"}

    assert preflight(tmp_path, config).problems == []


@pytest.mark.parametrize("data_format", [None, "geo"], ids=["absent", "unknown"])
def test_a_csv_source_needs_a_known_format(tmp_path, data_format):
    config = valid_config(tmp_path)
    config["data_sources"]["geo"]["format"] = data_format

    report = preflight(tmp_path, config)

    assert keys(report.problems) == ["data_sources.geo.format"]
    assert "multi_key" in report.problems[0].message


def test_a_source_that_is_not_a_mapping_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    config["data_sources"]["geo"] = "geo.csv"

    assert keys(preflight(tmp_path, config).problems) == ["data_sources.geo"]


def test_files_must_be_a_list(tmp_path):
    config = valid_config(tmp_path)
    config["data_sources"]["pair"]["files"] = {"path": "pair.csv"}

    assert keys(preflight(tmp_path, config).problems) == ["data_sources.pair.files"]


def test_a_file_without_a_path_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    config["data_sources"]["pair"]["files"] = [{"key_columns": {}}]

    assert keys(preflight(tmp_path, config).problems) == ["data_sources.pair.files[1].path"]


def test_a_missing_data_file_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    config["data_sources"]["pair"]["files"][0]["path"] = str(tmp_path / "absent.csv")

    report = preflight(tmp_path, config)

    assert keys(report.problems) == ["data_sources.pair.files[1].path"]
    assert "absent.csv" in report.problems[0].message


def geo_file(config):
    return config["data_sources"]["geo"]["files"][0]


def test_an_unknown_key_column_level_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    geo_file(config)["key_columns"]["sgu"]["level"] = "XLGU"

    report = preflight(tmp_path, config)

    assert keys(report.problems) == ["data_sources.geo.files[1].key_columns.sgu.level"]
    assert "XLGU" in report.problems[0].message


def test_an_unknown_destination_level_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    geo_file(config)["destination_level"] = "LAD"

    assert keys(preflight(tmp_path, config).problems) == ["data_sources.geo.files[1].destination_level"]


def test_an_unknown_geographical_unit_column_level_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    geo_file(config)["geographical_unit_column"] = {"name": "MGU", "level": "MSOA"}

    assert keys(preflight(tmp_path, config).problems) == [
        "data_sources.geo.files[1].geographical_unit_column.level"
    ]


@pytest.mark.parametrize("value", [["LGU"], {"name": "LGU"}], ids=["list", "mapping"])
def test_a_geo_level_that_is_not_a_name_is_a_problem(tmp_path, value):
    config = valid_config(tmp_path)
    geo_file(config)["key_columns"]["sgu"]["level"] = value

    assert keys(preflight(tmp_path, config).problems) == ["data_sources.geo.files[1].key_columns.sgu.level"]


def test_geo_levels_are_not_checked_while_unknown(tmp_path):
    config = valid_config(tmp_path)
    geo_file(config)["destination_level"] = "LAD"

    assert preflight(tmp_path, config, PreflightContext()).problems == []


def test_several_problems_in_one_file_are_all_reported(tmp_path):
    config = valid_config(tmp_path)
    config["step"]["assignment_level"] = "household"
    first_assignment(config)["strategy"] = "probabilistc"
    config["data_sources"]["pair"]["files"][0]["path"] = str(tmp_path / "absent.csv")
    geo_file(config)["key_columns"]["sgu"]["level"] = "XLGU"

    report = preflight(tmp_path, config)

    assert sorted(keys(report.problems)) == sorted([
        "step.assignment_level",
        f"{RULE_KEY}.strategy",
        "data_sources.pair.files[1].path",
        "data_sources.geo.files[1].key_columns.sgu.level",
    ])


def test_data_files_are_not_opened(tmp_path, monkeypatch):
    path = write(tmp_path, valid_config(tmp_path))
    opened = []
    real_open = builtins.open

    def recording_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", recording_open)
    AttributeAssignmentConfig.preflight(path, KNOWN_GEOGRAPHY)

    assert not [name for name in opened if name.endswith(".csv")]


def test_from_config_raises_on_a_problem_preflight_reports(tmp_path):
    config = valid_config(tmp_path)
    first_assignment(config)["strategy"] = "probabilistc"

    with pytest.raises(AttributeAssignmentError, match="strategy"):
        AttributeAssignmentConfig.from_config(write(tmp_path, config), {})


def test_from_config_builds_a_valid_file(tmp_path):
    built = AttributeAssignmentConfig.from_config(write(tmp_path, valid_config(tmp_path)), {})

    assert built.attribute_name == "ethnicity"


def test_the_built_config_assigns_its_attributes_into_the_world(tmp_path):
    path = write(tmp_path, valid_config(tmp_path))
    world = MagicMock()

    AttributeAssignmentConfig.from_config(path, {}).assign_attributes(world)

    world.assign_attributes.assert_called_once_with(path)


@pytest.fixture
def scenario_roots():
    """Path tokens resolved as `create_world` does, from each scenario's `data_root`."""
    previous = pr._resolver
    yield lambda data_root: pr.init(config_root="configs", data_root=data_root, output_root="output")
    pr._resolver = previous


@pytest.mark.parametrize("path", sorted(glob.glob("configs/2021/attributes/*.yaml")))
def test_shipped_attribute_configs_pass(path, scenario_roots):
    scenario_roots("data")

    assert AttributeAssignmentConfig.preflight(path, PreflightContext()).problems == []


@pytest.mark.parametrize("value", [None, "primary_adult", {"name": "primary_adult"}])
def test_inherit_from_roles_that_is_not_a_list_is_a_problem(tmp_path, value):
    config = valid_config(tmp_path)
    child_rule(config, [{"when": {"unique_count": 1}, "then": "values[0]"}])
    config["assignment_rules"]["Family"]["rules"][2]["assignment"]["inherit_from"] = {"roles": value}

    assert keys(preflight(tmp_path, config).problems) == [f"{CHILD_KEY}.inherit_from.roles"]


@pytest.mark.parametrize("value", ["primary_adult.ethnicity", {"role": "primary_adult"}])
def test_an_exclude_that_is_not_a_list_is_a_problem(tmp_path, value):
    config = valid_config(tmp_path)
    child_rule(config, [{"when": {"unique_count": 1}, "then": {"strategy": "probabilistic",
                                                             "data_source": "geo", "exclude": value}}])

    assert keys(preflight(tmp_path, config).problems) == [f"{CHILD_KEY}.logic[1].then.exclude"]
