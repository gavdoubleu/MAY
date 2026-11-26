"""
`DistributorStep` is the distributor timeline step's config stage. Its config is a
distributor YAML path: `parse` loads the file once, reads `distributor_type` and runs the
chosen class's `check_config`; `build` constructs that class from the loaded mapping.
"""
import builtins
from pathlib import Path

import pytest

from may.preflight.config_stage import ConfigError
from may.venue_distributor import (
    DISTRIBUTOR_TYPES,
    DistributorConfigError,
    DistributorStep,
    ParsedDistributor,
    SocialContactVisitDistributor,
    VenueDistributor,
)
from may.venue_distributor import base_distributor as base_distributor_module
from tests.test_unit.may.config_contexts import geography_context

KNOWN_GEOGRAPHY = geography_context()
KEY_PATH = "timeline.steps[4].config"


def keys(findings):
    """Finding keys with the file prefix taken off; a finding at the naming key stays whole."""
    return [finding.key.split("): ", 1)[1] if "): " in finding.key else finding.key for finding in findings]


def write(tmp_path, text, name="distributor.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def preflight(path, context=KNOWN_GEOGRAPHY):
    return DistributorStep.preflight(path, context, key_path=KEY_PATH)


def from_config(path, context=KNOWN_GEOGRAPHY):
    return DistributorStep.from_config(path, context, key_path=KEY_PATH)


def test_an_unknown_distributor_type_lists_every_type_sorted(tmp_path):
    _, report = preflight(write(tmp_path, "distributor_type: multivenue\n"))

    assert keys(report.problems) == ["distributor_type"]
    assert report.problems[0].message == (
        f"must be one of {', '.join(sorted(DISTRIBUTOR_TYPES))}; got text 'multivenue'"
    )


@pytest.mark.parametrize("value", ["[single_venue]", "{single_venue: 1}"], ids=["list", "mapping"])
def test_a_distributor_type_that_is_a_list_or_mapping_is_a_problem(tmp_path, value):
    _, report = preflight(write(tmp_path, f"distributor_type: {value}\n"))

    assert keys(report.problems) == ["distributor_type"]


def test_an_unknown_type_runs_no_class_checks(tmp_path):
    _, report = preflight(write(tmp_path, "distributor_type: multivenue\nvenue_type: [school]\n"))

    assert keys(report.problems) == ["distributor_type"]


@pytest.mark.parametrize("type_line", ["", "distributor_type: single_venue\n"], ids=["absent", "explicit"])
def test_a_single_venue_file_gets_the_venue_distributor_checks(tmp_path, type_line):
    _, report = preflight(write(tmp_path, type_line + "venue_type: [school]\n"))

    assert keys(report.problems) == ["venue_type"]


def test_each_type_dispatches_to_its_class_check(tmp_path, monkeypatch):
    called = []
    for distributor_class in set(DISTRIBUTOR_TYPES.values()):
        monkeypatch.setattr(
            distributor_class, "check_config",
            classmethod(lambda cls, reader, context: called.append(cls)),
        )

    for distributor_type, distributor_class in DISTRIBUTOR_TYPES.items():
        called.clear()
        preflight(write(tmp_path, f"distributor_type: {distributor_type}\n"))
        assert called == [distributor_class], distributor_type


def test_findings_are_keyed_under_the_step_and_file(tmp_path):
    path = write(tmp_path, "venue_type: [school]\n")

    _, report = preflight(path)

    assert [finding.key for finding in report.problems] == [f"{KEY_PATH} ({path}): venue_type"]


def test_a_missing_file_is_one_problem_at_the_naming_key(tmp_path):
    result, report = preflight(str(tmp_path / "absent.yaml"))

    assert result is None
    assert keys(report.problems) == [KEY_PATH]


def test_a_section_of_the_wrong_type_is_one_problem_not_a_cascade(tmp_path):
    _, report = preflight(write(tmp_path, "venue_type: school\neligibility: [a, b]\n"))

    assert keys(report.problems) == ["eligibility"]


def test_a_valid_file_parses_to_its_class_mapping_and_found_path(tmp_path):
    path = write(tmp_path, "distributor_type: social_contact_visit\nmax_contacts: 3\n")

    result, report = preflight(path)

    assert report.problems == []
    assert result == ParsedDistributor(
        distributor_class=SocialContactVisitDistributor,
        config_mapping={"distributor_type": "social_contact_visit", "max_contacts": 3},
        config_path=path,
    )


def test_a_path_from_code_is_accepted(tmp_path):
    result, report = preflight(Path(write(tmp_path, "venue_type: school\n")))

    assert report.problems == []
    assert result.distributor_class is VenueDistributor


def test_the_error_is_a_config_error_and_a_value_error(tmp_path):
    with pytest.raises(DistributorConfigError) as raised:
        from_config(write(tmp_path, "venue_type: school\nallocation:\n  strategy: clsest\n"))

    assert isinstance(raised.value, ConfigError)
    assert isinstance(raised.value, ValueError)
    assert str(raised.value).startswith("Distributor config cannot be used: 1 problem.")
    assert "allocation.strategy" in str(raised.value)


def test_from_config_raises_on_an_unknown_geo_level(tmp_path):
    path = write(tmp_path, "venue_type: school\nvenue_selection:\n  venue_geo_level: LSOA\n")

    with pytest.raises(DistributorConfigError, match="venue_selection.venue_geo_level"):
        from_config(path)


def test_from_config_builds_the_chosen_class(tmp_path):
    path = write(tmp_path, "venue_type: school\nvenue_selection:\n  venue_geo_level: SGU\n")

    distributor = from_config(path)

    assert type(distributor) is VenueDistributor
    assert distributor.venue_type == "school"
    assert distributor.venue_geo_level == "SGU"
    assert distributor.config_path == Path(path)


def test_the_file_is_read_once(tmp_path, monkeypatch):
    path = write(tmp_path, "venue_type: school\n")
    opened = []
    real_open = builtins.open

    def recording_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", recording_open)
    from_config(path)

    assert opened.count(path) == 1


def test_build_reads_no_file(monkeypatch):
    parsed = ParsedDistributor(
        distributor_class=VenueDistributor,
        config_mapping={"venue_type": "school"},
        config_path="nowhere/distributor.yaml",
    )

    def no_reading(*args, **kwargs):
        raise AssertionError("build read a file")

    monkeypatch.setattr(base_distributor_module, "load_yaml", no_reading, raising=False)

    distributor = DistributorStep.build(parsed)

    assert distributor.venue_type == "school"
    assert distributor.config_path == Path("nowhere/distributor.yaml")


def test_the_module_level_dispatch_functions_are_gone():
    import may.venue_distributor as package

    for name in ("distributor_from_yaml", "distributor_preflight", "_distributor_class"):
        assert not hasattr(package, name), name


@pytest.mark.parametrize("distributor_class", sorted(set(DISTRIBUTOR_TYPES.values()), key=lambda cls: cls.__name__))
def test_every_class_drops_its_own_parse_entry_points(distributor_class):
    for name in ("parse_config", "preflight", "from_config", "_check_config"):
        assert not hasattr(distributor_class, name), name


BLANK_SECTIONS = {
    "venue_selection": "venue_type: school\nvenue_selection:\n",
    "allocation": "venue_type: school\nallocation:\n",
    "eligibility": "venue_type: school\neligibility:\n",
    "eligibility.global_filters": "venue_type: school\neligibility:\n  global_filters:\n",
    "special_cases": "venue_type: school\nspecial_cases:\n",
    "settings": "venue_type: school\nsettings:\n",
    "validation": "venue_type: school\nvalidation:\n",
}


@pytest.mark.parametrize("key", BLANK_SECTIONS, ids=list(BLANK_SECTIONS))
def test_a_section_left_blank_is_one_problem(tmp_path, key):
    _, report = preflight(write(tmp_path, BLANK_SECTIONS[key]))

    assert keys(report.problems) == [key]
    assert "left blank" in report.problems[0].message


def test_settings_that_are_not_a_mapping_are_one_problem(tmp_path):
    _, report = preflight(write(tmp_path, "venue_type: school\nsettings: 3\n"))

    assert keys(report.problems) == ["settings"]
