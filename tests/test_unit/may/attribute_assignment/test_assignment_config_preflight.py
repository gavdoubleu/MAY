"""
`AttributeAssignmentConfig` is a config stage whose config is an attribute YAML path:
`preflight` reports every problem without loading its data; `from_config` raises on the
same ones, then builds from the loaded mapping without reading the file again.
"""
import builtins
import glob
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from may.attribute_assignment import AttributeAssignmentConfig, AttributeAssignmentError
from may.attribute_assignment import assignment_config as assignment_config_module
from may.attribute_assignment.assignment_config import AttributeConfigError, ParsedAttributeConfig
from may.preflight.config_stage import ConfigError
from may.utils import path_resolver as pr
from tests.test_unit.may.config_contexts import geography_context

KNOWN_GEOGRAPHY = geography_context()
UNKNOWN_GEOGRAPHY = geography_context(levels=None)
KEY_PATH = "timeline.steps[2].config"
RULE_KEY = "assignment_rules.Family.rules[1].assignment"


def keys(findings):
    """Finding keys with the file prefix taken off; a finding at the naming key stays whole."""
    return [finding.key.split("): ", 1)[1] if "): " in finding.key else finding.key for finding in findings]


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
    _, report = AttributeAssignmentConfig.preflight(write(tmp_path, config), context, key_path=KEY_PATH)
    return report


def first_assignment(config):
    return config["assignment_rules"]["Family"]["rules"][0]["assignment"]


def test_a_valid_file_has_no_findings(tmp_path):
    report = preflight(tmp_path, valid_config(tmp_path))

    assert report.problems == []
    assert report.warnings == []


def test_a_valid_file_gives_its_mapping_and_found_path(tmp_path):
    path = write(tmp_path, valid_config(tmp_path))

    parsed, _ = AttributeAssignmentConfig.preflight(path, KNOWN_GEOGRAPHY, key_path=KEY_PATH)

    assert parsed.config_path == path
    assert parsed.config_mapping["attributes"] == [{"name": "ethnicity"}]


def test_a_path_object_reads_as_its_text(tmp_path):
    path = write(tmp_path, valid_config(tmp_path))

    parsed, report = AttributeAssignmentConfig.preflight(Path(path), KNOWN_GEOGRAPHY, key_path=KEY_PATH)

    assert report.problems == []
    assert parsed.config_path == path


def test_a_missing_file_is_a_problem_at_the_naming_key(tmp_path):
    _, report = AttributeAssignmentConfig.preflight(
        str(tmp_path / "absent.yaml"), KNOWN_GEOGRAPHY, key_path=KEY_PATH
    )

    assert keys(report.problems) == [KEY_PATH]


def test_findings_are_keyed_under_the_found_file(tmp_path):
    config = valid_config(tmp_path)
    config["step"]["assignment_level"] = "household"
    path = write(tmp_path, config)

    _, report = AttributeAssignmentConfig.preflight(path, KNOWN_GEOGRAPHY, key_path=KEY_PATH)

    assert [finding.key for finding in report.problems] == [f"{KEY_PATH} ({path}): step.assignment_level"]


def test_the_retired_attribute_block_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    config["attribute"] = {"name": "ethnicity"}

    report = preflight(tmp_path, config)

    assert keys(report.problems) == ["attribute"]
    assert "retired" in report.problems[0].message


@pytest.mark.parametrize("attributes", ["absent", None, [], {"name": "ethnicity"}],
                         ids=["absent", "blank", "empty", "mapping"])
def test_attributes_must_be_a_non_empty_list(tmp_path, attributes):
    config = valid_config(tmp_path)
    if attributes == "absent":
        del config["attributes"]
    else:
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


PARTNER_KEY = "assignment_rules.Family.rules[2].assignment"


def partner_assignment(config):
    return config["assignment_rules"]["Family"]["rules"][1]["assignment"]


def test_an_undeclared_marginal_data_source_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    partner_assignment(config)["marginal_data_source"] = "goe"

    report = preflight(tmp_path, config)

    assert keys(report.problems) == [f"{PARTNER_KEY}.marginal_data_source"]
    assert "['geo', 'pair']" in report.problems[0].message


@LIST_OR_MAPPING
def test_a_marginal_data_source_that_is_a_list_or_mapping_is_a_problem(tmp_path, value):
    config = valid_config(tmp_path)
    partner_assignment(config)["marginal_data_source"] = value

    assert keys(preflight(tmp_path, config).problems) == [f"{PARTNER_KEY}.marginal_data_source"]


def test_whole_number_marginal_data_source_is_read_as_text(tmp_path):
    config = valid_config(tmp_path)
    config["data_sources"][2021] = config["data_sources"].pop("geo")
    first_assignment(config)["data_source"] = 2021
    partner_assignment(config)["marginal_data_source"] = 2021
    path = write(tmp_path, config)

    parsed, report = AttributeAssignmentConfig.preflight(path, KNOWN_GEOGRAPHY, key_path=KEY_PATH)

    assert report.problems == []
    assert partner_assignment(parsed.config_mapping)["marginal_data_source"] == "2021"


def test_whole_number_logic_data_source_is_read_as_text(tmp_path):
    config = valid_config(tmp_path)
    config["data_sources"][2021] = config["data_sources"].pop("geo")
    first_assignment(config)["data_source"] = 2021
    child_rule(config, [{"when": {"unique_count": 1}, "then": {"strategy": "probabilistic", "data_source": 2021}}])
    path = write(tmp_path, config)

    parsed, report = AttributeAssignmentConfig.preflight(path, KNOWN_GEOGRAPHY, key_path=KEY_PATH)

    assert report.problems == []
    child_assignment = parsed.config_mapping["assignment_rules"]["Family"]["rules"][2]["assignment"]
    assert child_assignment["logic"][0]["then"]["data_source"] == "2021"


def test_whole_number_venue_rule_data_source_is_read_as_text(tmp_path):
    config = valid_config(tmp_path)
    config["data_sources"][2021] = config["data_sources"].pop("geo")
    first_assignment(config)["data_source"] = 2021
    config["venue_assignment_rules"] = [
        {"venue_type": "care_home", "assignment": {"strategy": "probabilistic", "data_source": 2021}}
    ]
    path = write(tmp_path, config)

    parsed, report = AttributeAssignmentConfig.preflight(path, KNOWN_GEOGRAPHY, key_path=KEY_PATH)

    assert report.problems == []
    assert parsed.config_mapping["venue_assignment_rules"][0]["assignment"]["data_source"] == "2021"


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

    assert preflight(tmp_path, config, UNKNOWN_GEOGRAPHY).problems == []


def test_whole_number_attribute_and_source_names_are_read_as_text(tmp_path):
    config = valid_config(tmp_path)
    config["attributes"] = [{"name": 7}]
    config["required_attributes"] = [{"name": 8}]
    config["data_sources"][5] = config["data_sources"].pop("geo")
    first_assignment(config)["data_source"] = 5
    path = write(tmp_path, config)

    parsed, report = AttributeAssignmentConfig.preflight(path, KNOWN_GEOGRAPHY, key_path=KEY_PATH)

    assert report.problems == []
    mapping = parsed.config_mapping
    assert mapping["attributes"][0]["name"] == "7"
    assert mapping["required_attributes"][0]["name"] == "8"
    assert "5" in mapping["data_sources"]
    assert first_assignment(mapping)["data_source"] == "5"



WHOLE_NUMBER_LEVELS = geography_context(levels=("1", "2"))


def test_whole_number_geo_levels_reach_the_build_as_text(tmp_path):
    config = valid_config(tmp_path)
    geo_file(config)["destination_level"] = 2
    geo_file(config)["geographical_unit_column"] = {"name": "sgu", "level": 1}
    geo_file(config)["key_columns"]["sgu"]["level"] = 1
    path = write(tmp_path, config)

    parsed, report = AttributeAssignmentConfig.preflight(path, WHOLE_NUMBER_LEVELS, key_path=KEY_PATH)

    assert report.problems == []
    file_config = geo_file(parsed.config_mapping)
    assert file_config["destination_level"] == "2"
    assert file_config["geographical_unit_column"]["level"] == "1"
    assert file_config["key_columns"]["sgu"]["level"] == "1"


def test_whole_number_data_file_path_reaches_the_build_as_text(tmp_path, monkeypatch):
    config = valid_config(tmp_path)
    (tmp_path / "2021").write_text("geo_unit,W\nA,1\n")
    monkeypatch.chdir(tmp_path)
    geo_file(config)["path"] = 2021
    path = write(tmp_path, config)

    parsed, report = AttributeAssignmentConfig.preflight(path, KNOWN_GEOGRAPHY, key_path=KEY_PATH)

    assert report.problems == []
    assert geo_file(parsed.config_mapping)["path"] == "2021"

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
    AttributeAssignmentConfig.preflight(path, KNOWN_GEOGRAPHY, key_path=KEY_PATH)

    assert not [name for name in opened if name.endswith(".csv")]


def from_config(path, context=KNOWN_GEOGRAPHY):
    return AttributeAssignmentConfig.from_config(path, context, key_path=KEY_PATH)


def test_from_config_raises_on_a_problem_preflight_reports(tmp_path):
    config = valid_config(tmp_path)
    first_assignment(config)["strategy"] = "probabilistc"

    with pytest.raises(AttributeConfigError, match="strategy") as raised:
        from_config(write(tmp_path, config))

    assert isinstance(raised.value, AttributeAssignmentError)
    assert isinstance(raised.value, ConfigError)
    assert str(raised.value).startswith("Attribute config cannot be used: 1 problem.")


def test_from_config_raises_on_an_unknown_geo_level(tmp_path):
    config = valid_config(tmp_path)
    geo_file(config)["destination_level"] = "LAD"

    with pytest.raises(AttributeConfigError, match="destination_level"):
        from_config(write(tmp_path, config))


def test_from_config_builds_a_valid_file(tmp_path):
    built = from_config(write(tmp_path, valid_config(tmp_path)))

    assert built.attribute_name == "ethnicity"


def test_the_file_is_loaded_once(tmp_path, monkeypatch):
    path = write(tmp_path, valid_config(tmp_path))
    loads = []
    real_load = pr.find_config_file

    def counting_find(*args, **kwargs):
        loads.append(args)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(pr, "find_config_file", counting_find)
    from_config(path)

    assert len(loads) == 1


def test_build_reads_no_file(tmp_path, monkeypatch):
    parsed = ParsedAttributeConfig(config_mapping=valid_config(tmp_path), config_path="nowhere/attributes.yaml")

    def no_reading(*args, **kwargs):
        raise AssertionError("build read a file")

    monkeypatch.setattr(assignment_config_module, "load_yaml", no_reading, raising=False)
    monkeypatch.setattr(pr, "resolve", no_reading)

    built = AttributeAssignmentConfig.build(parsed)

    assert built.attribute_name == "ethnicity"
    assert str(built.config_path) == "nowhere/attributes.yaml"


def test_from_yaml_is_gone():
    assert not hasattr(AttributeAssignmentConfig, "from_yaml")


def test_the_built_config_assigns_its_attributes_into_the_world(tmp_path):
    world = MagicMock()

    built = from_config(write(tmp_path, valid_config(tmp_path)))
    built.assign_attributes(world)

    world.assign_attributes.assert_called_once_with(built)


@pytest.fixture
def scenario_roots():
    """Path tokens resolved as `create_world` does, from each scenario's `data_root`."""
    previous = pr._resolver
    yield lambda data_root: pr.init(config_root="configs", data_root=data_root, output_root="output")
    pr._resolver = previous


@pytest.mark.parametrize("path", sorted(glob.glob("configs/2021/attributes/*.yaml")))
def test_shipped_attribute_configs_pass(path, scenario_roots):
    scenario_roots("data")

    _, report = AttributeAssignmentConfig.preflight(path, UNKNOWN_GEOGRAPHY, key_path=KEY_PATH)

    assert report.problems == []


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


def _blank(config, path):
    """Set the entry at dotted `path` to None, as a YAML key left blank."""
    *parents, name = path.split(".")
    container = config
    for parent in parents:
        container = container[parent]
    container[name] = None


@pytest.mark.parametrize("path", [
    "step", "required_attributes", "data_sources", "assignment_rules",
    "assignment_rules.Family.rules", "venue_assignment_rules",
])
def test_a_section_left_blank_is_one_problem(tmp_path, path):
    config = valid_config(tmp_path)
    config.setdefault("venue_assignment_rules", [])
    _blank(config, path)

    report = preflight(tmp_path, config)

    assert keys(report.problems) == [path]
    assert "blank" in report.problems[0].message


def test_a_blank_data_source_files_list_is_a_problem(tmp_path):
    config = valid_config(tmp_path)
    config["data_sources"]["pair"]["files"] = None

    assert keys(preflight(tmp_path, config).problems) == ["data_sources.pair.files"]
