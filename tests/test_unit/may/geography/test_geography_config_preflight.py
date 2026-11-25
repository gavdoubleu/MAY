import pytest

from may.geography.geography import DEFAULT_DATA_DIR, Geography
from may.preflight.context import ConfigContext
from may.utils import path_resolver


@pytest.fixture(autouse=True)
def reset_resolver(monkeypatch):
    monkeypatch.setattr(path_resolver, "_resolver", None)


@pytest.fixture
def working_dir(tmp_path, monkeypatch):
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    (working / "data").mkdir()
    for name in DATA_FILES:
        (working / "data" / name).write_text("")
    return working


DATA_FILES = ["EW_hierarchy.csv", "EW_coord_sgu.csv", "EW_coord_mgu.csv"]
LEVELS_TEXT = "['SGU', 'MGU', 'LGU', 'XLGU']"


def data_path(working_dir, name):
    return str(working_dir / "data" / name)


def clean_block(**overrides):
    """A block shaped like `configs/2021`'s `geography:` block."""
    block = {
        "data_dir": "data",
        "hierarchy_file": "EW_hierarchy.csv",
        "coord_files": {"SGU": "EW_coord_sgu.csv", "MGU": "EW_coord_mgu.csv"},
        "levels": ["SGU", "MGU", "LGU", "XLGU"],
        "load_all": False,
        "filter": {"level": "LGU", "codes": ["County Durham", "Darlington"]},
    }
    block.update(overrides)
    return block


def without(block, *keys):
    for key in keys:
        del block[key]
    return block


def parse(block):
    return Geography.preflight(block, ConfigContext())


def warning_lines(report):
    return [(finding.key, finding.message) for finding in report.problems + report.warnings]


def assert_single_problem(block, key):
    geography_config, report = parse(block)
    assert geography_config is None
    assert [finding.key for finding in report.problems] == [f"geography.{key}"]
    assert not report.warnings
    return report.problems[0].message


def test_clean_block_is_clean_and_parsed(working_dir):
    geography_config, report = parse(clean_block())
    assert not report.problems and not report.warnings
    assert geography_config.data_dir == "data"
    assert geography_config.levels == ["SGU", "MGU", "LGU", "XLGU"]
    assert geography_config.hierarchy_file == [data_path(working_dir, "EW_hierarchy.csv")]
    assert geography_config.coord_files == {
        "SGU": [data_path(working_dir, "EW_coord_sgu.csv")],
        "MGU": [data_path(working_dir, "EW_coord_mgu.csv")],
    }
    assert geography_config.filters == {
        "level": "LGU",
        "codes": ["County Durham", "Darlington"],
    }


def test_key_path_can_be_overridden(working_dir):
    _, report = Geography.preflight(without(clean_block(), "levels"), ConfigContext(), key_path="world.geo")
    assert [finding.key for finding in report.problems] == ["world.geo.levels"]


@pytest.mark.parametrize(
    "block, message",
    [(None, "left blank; give a value"), (["SGU"], "must be a mapping; got a list")],
    ids=["blank", "not_mapping"],
)
def test_blank_or_not_mapping_block_is_one_problem(working_dir, block, message):
    geography_config, report = parse(block)
    assert geography_config is None
    assert [(finding.key, finding.message) for finding in report.problems] == [("geography", message)]


def test_paths_are_template_resolved(working_dir, tmp_path):
    data_root = tmp_path / "data_root"
    (data_root / "geography").mkdir(parents=True)
    for name in ["a.csv", "sgu.csv", "geography/b.csv"]:
        (data_root / name).write_text("")
    path_resolver.init("/configs", str(data_root), "/output")
    geography_config, report = parse(
        clean_block(
            data_dir="${data_root}/geography",
            hierarchy_file=["${data_root}/a.csv", "b.csv"],
            coord_files={"SGU": "${data_root}/sgu.csv"},
        )
    )
    assert not report.problems
    assert geography_config.data_dir == f"{data_root}/geography"
    assert geography_config.hierarchy_file == [
        f"{data_root}/a.csv",
        f"{data_root}/geography/b.csv",
    ]
    assert geography_config.coord_files == {"SGU": [f"{data_root}/sgu.csv"]}


# data_dir


def test_absent_data_dir_falls_back_to_default(working_dir):
    default_dir = working_dir / DEFAULT_DATA_DIR
    default_dir.mkdir(parents=True)
    for name in DATA_FILES:
        (default_dir / name).write_text("")
    geography_config, report = parse(without(clean_block(), "data_dir"))
    assert not report.problems
    assert geography_config.data_dir == DEFAULT_DATA_DIR


@pytest.mark.parametrize(
    "blank, message",
    [
        (None, "left blank; give a value, or remove the key to use the default"),
        ("", "must not be empty; give a path, or remove the key to use the default"),
    ],
    ids=["blank", "empty"],
)
def test_blank_data_dir_is_problem(working_dir, blank, message):
    assert assert_single_problem(clean_block(data_dir=blank), "data_dir") == message


def test_data_dir_not_a_path_is_problem(working_dir):
    message = assert_single_problem(clean_block(data_dir=["data"]), "data_dir")
    assert message == "must be text; got a list"


def test_whole_number_data_dir_is_read_as_text(working_dir):
    (working_dir / "7").mkdir()
    for name in DATA_FILES:
        (working_dir / "7" / name).write_text("")
    geography_config, report = parse(clean_block(data_dir=7))
    assert not report.problems
    assert geography_config.data_dir == "7"


# levels


def test_levels_missing_is_problem(working_dir):
    message = assert_single_problem(without(clean_block(), "levels"), "levels")
    assert message == "not set; give a list of level names"


def test_levels_blank_is_problem(working_dir):
    message = assert_single_problem(clean_block(levels=None), "levels")
    assert message == "left blank; give a list of level names"


def test_levels_empty_is_problem(working_dir):
    message = assert_single_problem(clean_block(levels=[]), "levels")
    assert message == "must not be empty; give a list of level names"


def test_levels_not_list_is_problem(working_dir):
    message = assert_single_problem(clean_block(levels="SGU"), "levels")
    assert message == "must be a list; got text 'SGU'"


def test_levels_bad_item_is_problem_at_item(working_dir):
    message = assert_single_problem(clean_block(levels=["SGU", "", "LGU", "XLGU"]), "levels[2]")
    assert message == "must not be empty; give a name"


def test_levels_bad_item_skips_level_checks(working_dir):
    """A level left out is not then reported as unknown in coord_files or filter."""
    _, report = parse(clean_block(levels=["SGU", None, "LGU", "XLGU"]))
    assert [finding.key for finding in report.problems] == ["geography.levels[2]"]


def test_whole_number_level_is_read_as_text(working_dir):
    geography_config, report = parse(clean_block(levels=["SGU", "MGU", "LGU", 4], coord_files={}))
    assert not report.problems
    assert geography_config.levels == ["SGU", "MGU", "LGU", "4"]


def test_levels_duplicates_is_problem(working_dir):
    message = assert_single_problem(
        clean_block(levels=["SGU", "MGU", "SGU", "LGU", "XLGU"]), "levels"
    )
    assert message == "duplicate level names: ['SGU']"


# hierarchy_file


def test_hierarchy_file_missing_is_problem(working_dir):
    message = assert_single_problem(without(clean_block(), "hierarchy_file"), "hierarchy_file")
    assert message == "not set; give a path or a list of paths"


def test_hierarchy_file_blank_is_problem(working_dir):
    message = assert_single_problem(clean_block(hierarchy_file=None), "hierarchy_file")
    assert message == "left blank; give a path or a list of paths"


@pytest.mark.parametrize(
    "bad_path, shown",
    [("", "text ''"), (True, "true"), ([], "a list"), (["a.csv", 2.5], "a list"), ({"a": "b"}, "a mapping")],
)
def test_hierarchy_file_not_a_path_is_problem(working_dir, bad_path, shown):
    message = assert_single_problem(clean_block(hierarchy_file=bad_path), "hierarchy_file")
    assert message == f"must be a path or a non-empty list of paths; got {shown}"


def test_whole_number_hierarchy_file_read_as_text(working_dir):
    message = assert_single_problem(clean_block(hierarchy_file=1911), "hierarchy_file")
    assert message == f"not found: {data_path(working_dir, '1911')}"


def test_hierarchy_file_missing_on_disk_is_problem(working_dir):
    message = assert_single_problem(
        clean_block(hierarchy_file=["EW_hierarchy.csv", "gone.csv"]), "hierarchy_file"
    )
    assert message == f"not found: {data_path(working_dir, 'gone.csv')}"


def test_absolute_hierarchy_file_not_joined(working_dir, tmp_path):
    absolute = tmp_path / "elsewhere.csv"
    absolute.write_text("")
    geography_config, report = parse(clean_block(hierarchy_file=str(absolute)))
    assert not report.problems
    assert geography_config.hierarchy_file == [str(absolute)]


# coord_files


def test_coord_files_absent_is_clean_and_empty(working_dir):
    geography_config, report = parse(without(clean_block(), "coord_files"))
    assert not report.problems
    assert geography_config.coord_files == {}


def test_coord_files_blank_is_problem(working_dir):
    message = assert_single_problem(clean_block(coord_files=None), "coord_files")
    assert message == "left blank; give a value, or remove the key to use the default"


def test_coord_files_not_mapping_is_problem(working_dir):
    message = assert_single_problem(clean_block(coord_files=["a.csv"]), "coord_files")
    assert message == "must be a mapping; got a list"


def test_coord_files_unknown_level_is_problem(working_dir):
    message = assert_single_problem(clean_block(coord_files={"OA": "oa.csv"}), "coord_files.OA")
    assert message == f"not a configured level; levels are {LEVELS_TEXT}"


def test_coord_files_whole_number_level_is_read_as_text(working_dir):
    geography_config, report = parse(
        clean_block(levels=["SGU", "MGU", "LGU", 4], coord_files={4: "EW_coord_sgu.csv"})
    )
    assert not report.problems
    assert geography_config.coord_files == {"4": [data_path(working_dir, "EW_coord_sgu.csv")]}


@pytest.mark.parametrize(
    "bad_path, message",
    [
        (None, "left blank; give a path or a list of paths"),
        (2.5, "must be a path or a non-empty list of paths; got a number 2.5"),
        ([], "must be a path or a non-empty list of paths; got a list"),
        (["a.csv", None], "must be a path or a non-empty list of paths; got a list"),
    ],
)
def test_coord_files_value_not_a_path_is_problem(working_dir, bad_path, message):
    assert assert_single_problem(clean_block(coord_files={"SGU": bad_path}), "coord_files.SGU") == message


def test_coord_file_missing_on_disk_is_problem(working_dir):
    message = assert_single_problem(
        clean_block(coord_files={"SGU": "EW_coord_sgu.csv", "LGU": "gone.csv"}),
        "coord_files.LGU",
    )
    assert message == f"not found: {data_path(working_dir, 'gone.csv')}"


def test_level_absent_from_coord_files_is_not_checked(working_dir):
    geography_config, report = parse(clean_block(coord_files={"SGU": "EW_coord_sgu.csv"}))
    assert not report.problems
    assert set(geography_config.coord_files) == {"SGU"}


def test_unknown_coord_level_file_not_checked_for_existence(working_dir):
    message = assert_single_problem(clean_block(coord_files={"OA": "gone.csv"}), "coord_files.OA")
    assert message.startswith("not a configured level")


def test_bad_data_dir_skips_existence_checks(working_dir):
    message = assert_single_problem(
        clean_block(data_dir=["data"], hierarchy_file="gone.csv"), "data_dir"
    )
    assert message == "must be text; got a list"


# load_all


def test_load_all_not_bool_is_problem(working_dir):
    message = assert_single_problem(clean_block(load_all="yes"), "load_all")
    assert message == "must be true or false; got text 'yes'"


def test_load_all_blank_is_problem(working_dir):
    message = assert_single_problem(clean_block(load_all=None), "load_all")
    assert message == "left blank; give a value, or remove the key to use the default"


def test_load_all_true_with_filter_has_no_finding_and_no_filters(working_dir):
    geography_config, report = parse(clean_block(load_all=True, filter={"level": "OA", "codes": "bad"}))
    assert not report.problems and not report.warnings
    assert geography_config.filters is None


# filter


def test_filter_absent_loads_all(working_dir):
    geography_config, report = parse(without(clean_block(), "filter"))
    assert not report.problems and not report.warnings
    assert geography_config.filters is None


def test_filter_blank_is_problem(working_dir):
    message = assert_single_problem(clean_block(filter=None), "filter")
    assert message == "left blank; give a value, or remove the key to use the default"


def test_filter_not_mapping_is_problem(working_dir):
    message = assert_single_problem(clean_block(filter="LGU"), "filter")
    assert message == "must be a mapping; got text 'LGU'"


def test_filter_level_not_configured_is_problem(working_dir):
    message = assert_single_problem(clean_block(filter={"level": "OA", "codes": ["x"]}), "filter.level")
    assert message == f"'OA' is not a configured level; levels are {LEVELS_TEXT}"


def test_filter_level_empty_is_problem(working_dir):
    message = assert_single_problem(clean_block(filter={"level": "", "codes": ["x"]}), "filter.level")
    assert message == "must not be empty; give a level name"


def test_filter_codes_not_list_is_problem(working_dir):
    message = assert_single_problem(
        clean_block(filter={"level": "LGU", "codes": "Darlington"}), "filter.codes"
    )
    assert message == "must be a list; got text 'Darlington'"


def test_filter_codes_blank_is_problem(working_dir):
    message = assert_single_problem(clean_block(filter={"level": "LGU", "codes": None}), "filter.codes")
    assert message == "left blank; give a value, or remove the key to use the default"


def test_filter_codes_whole_numbers_read_as_text(working_dir):
    geography_config, report = parse(clean_block(filter={"level": "LGU", "codes": [1001, "E02"]}))
    assert not report.problems
    assert geography_config.filters == {"level": "LGU", "codes": ["1001", "E02"]}


def test_filter_file_found_in_working_dir(working_dir):
    (working_dir / "codes.txt").write_text("# comment\nDarlington\n\nCounty Durham\n")
    geography_config, report = parse(clean_block(filter={"level": "LGU", "file": "codes.txt"}))
    assert not report.problems and not report.warnings
    assert geography_config.filters == {
        "level": "LGU",
        "codes": ["County Durham", "Darlington"],
    }


def test_filter_file_found_under_data_dir(working_dir):
    (working_dir / "data" / "codes.txt").write_text("Darlington\n")
    geography_config, report = parse(clean_block(filter={"level": "LGU", "file": "codes.txt"}))
    assert not report.problems
    assert geography_config.filters == {"level": "LGU", "codes": ["Darlington"]}


def test_filter_file_not_found_names_both_paths(working_dir):
    message = assert_single_problem(
        clean_block(filter={"level": "LGU", "file": "missing.txt"}), "filter.file"
    )
    assert message == "not found; tried missing.txt, data/missing.txt"


def test_filter_file_unreadable_is_problem(working_dir):
    (working_dir / "codes.txt").mkdir()
    message = assert_single_problem(clean_block(filter={"level": "LGU", "file": "codes.txt"}), "filter.file")
    assert message.startswith("codes.txt could not be read")


def test_filter_file_not_a_path_is_problem(working_dir):
    message = assert_single_problem(
        clean_block(filter={"level": "LGU", "file": ["codes.txt"]}), "filter.file"
    )
    assert message == "must be text; got a list"


def test_bad_data_dir_skips_filter_file_search(working_dir):
    message = assert_single_problem(
        clean_block(data_dir=["data"], filter={"level": "LGU", "file": "missing.txt"}), "data_dir"
    )
    assert message == "must be text; got a list"


# filter warnings


def test_filter_level_with_empty_codes_warns_and_loads_all(working_dir):
    geography_config, report = parse(clean_block(filter={"level": "LGU", "codes": []}))
    assert warning_lines(report) == [("geography.filter", "level set but no codes; loads all units")]
    assert geography_config.filters is None


def test_filter_level_without_codes_warns_and_loads_all(working_dir):
    geography_config, report = parse(clean_block(filter={"level": "LGU"}))
    assert warning_lines(report) == [("geography.filter", "level set but no codes; loads all units")]
    assert geography_config.filters is None


def test_filter_file_with_no_codes_warns_and_loads_all(working_dir):
    (working_dir / "codes.txt").write_text("# only a comment\n")
    geography_config, report = parse(clean_block(filter={"level": "LGU", "file": "codes.txt"}))
    assert warning_lines(report) == [("geography.filter", "level set but no codes; loads all units")]
    assert geography_config.filters is None


def test_filter_codes_and_file_warns_file_wins(working_dir):
    (working_dir / "codes.txt").write_text("Darlington\n")
    geography_config, report = parse(
        clean_block(filter={"level": "LGU", "codes": ["County Durham"], "file": "codes.txt"})
    )
    assert warning_lines(report) == [("geography.filter.codes", "ignored; filter.file is set")]
    assert geography_config.filters == {"level": "LGU", "codes": ["Darlington"]}


def test_filter_codes_without_level_warns_and_loads_all(working_dir):
    geography_config, report = parse(clean_block(filter={"codes": ["Darlington"]}))
    assert warning_lines(report) == [
        ("geography.filter.level", "not set; filter ignored, loads all units")
    ]
    assert geography_config.filters is None


def test_every_problem_reported_at_once(working_dir):
    geography_config, report = parse({"data_dir": [3], "levels": "SGU", "load_all": "no", "filter": []})
    assert geography_config is None
    assert [finding.key for finding in report.problems] == [
        "geography.data_dir",
        "geography.levels",
        "geography.hierarchy_file",
        "geography.load_all",
        "geography.filter",
    ]
