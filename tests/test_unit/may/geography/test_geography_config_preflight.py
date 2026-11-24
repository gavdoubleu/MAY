import pytest

from may.geography.geography import DEFAULT_DATA_DIR, Geography
from may.utils import path_resolver


@pytest.fixture(autouse=True)
def reset_resolver(monkeypatch):
    monkeypatch.setattr(path_resolver, "_resolver", None)


@pytest.fixture
def working_dir(tmp_path, monkeypatch):
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    return working


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


def problem_lines(report):
    return [(finding.key, finding.message) for finding in report.problems]


def warning_lines(report):
    return [(finding.key, finding.message) for finding in report.warnings]


def assert_single_problem(block, key):
    geography_config, report = Geography.parse_config(block)
    assert geography_config is None
    assert [finding.key for finding in report.problems] == [key]
    assert not report.warnings
    return report.problems[0].message


def test_clean_block_is_clean_and_parsed(working_dir):
    geography_config, report = Geography.parse_config(clean_block())
    assert not report.problems and not report.warnings
    assert geography_config.data_dir == "data"
    assert geography_config.levels == ["SGU", "MGU", "LGU", "XLGU"]
    assert geography_config.hierarchy_file == "EW_hierarchy.csv"
    assert geography_config.coord_files == {
        "SGU": "EW_coord_sgu.csv",
        "MGU": "EW_coord_mgu.csv",
    }
    assert geography_config.filters == {
        "level": "LGU",
        "codes": ["County Durham", "Darlington"],
    }


def test_preflight_returns_parse_report(working_dir):
    report = Geography.preflight(clean_block(levels=None))
    assert [finding.key for finding in report.problems] == ["levels"]


def test_paths_are_template_resolved(working_dir):
    path_resolver.init("/configs", "/data", "/output")
    geography_config, report = Geography.parse_config(
        clean_block(
            data_dir="${data_root}/geography",
            hierarchy_file=["${data_root}/a.csv", "b.csv"],
            coord_files={"SGU": "${data_root}/sgu.csv"},
        )
    )
    assert not report.problems
    assert geography_config.data_dir == "/data/geography"
    assert geography_config.hierarchy_file == ["/data/a.csv", "b.csv"]
    assert geography_config.coord_files == {"SGU": "/data/sgu.csv"}


# data_dir


@pytest.mark.parametrize("blank", [None, ""])
def test_blank_data_dir_falls_back_to_default(working_dir, blank):
    geography_config, report = Geography.parse_config(clean_block(data_dir=blank))
    assert not report.problems
    assert geography_config.data_dir == DEFAULT_DATA_DIR


def test_data_dir_not_a_path_is_problem(working_dir):
    message = assert_single_problem(clean_block(data_dir=["data"]), "data_dir")
    assert message == "must be a path; got ['data']"


# levels


def test_levels_missing_is_problem(working_dir):
    block = clean_block()
    del block["levels"]
    assert assert_single_problem(block, "levels") == "not set"


@pytest.mark.parametrize("empty", [None, []])
def test_levels_empty_is_problem(working_dir, empty):
    assert assert_single_problem(clean_block(levels=empty), "levels") == "not set"


@pytest.mark.parametrize("bad_levels", ["SGU", ["SGU", 3], ["SGU", ""]])
def test_levels_not_list_of_strings_is_problem(working_dir, bad_levels):
    message = assert_single_problem(clean_block(levels=bad_levels), "levels")
    assert message == f"must be a list of level names; got {bad_levels!r}"


def test_levels_duplicates_is_problem(working_dir):
    message = assert_single_problem(
        clean_block(levels=["SGU", "MGU", "SGU", "LGU", "XLGU"]), "levels"
    )
    assert message == "duplicate level names: ['SGU']"


# hierarchy_file


@pytest.mark.parametrize("missing", [None, ""])
def test_hierarchy_file_missing_is_problem(working_dir, missing):
    message = assert_single_problem(clean_block(hierarchy_file=missing), "hierarchy_file")
    assert message == "not set"


@pytest.mark.parametrize("bad_path", [3, [], ["a.csv", 3], {"a": "b"}])
def test_hierarchy_file_not_a_path_is_problem(working_dir, bad_path):
    message = assert_single_problem(clean_block(hierarchy_file=bad_path), "hierarchy_file")
    assert message == f"must be a path or non-empty list of paths; got {bad_path!r}"


# coord_files


def test_coord_files_absent_is_clean_and_empty(working_dir):
    geography_config, report = Geography.parse_config(clean_block(coord_files=None))
    assert not report.problems
    assert geography_config.coord_files == {}


def test_coord_files_not_mapping_is_problem(working_dir):
    message = assert_single_problem(clean_block(coord_files=["a.csv"]), "coord_files")
    assert message == "must be a mapping; got sequence (list)"


def test_coord_files_unknown_level_is_problem(working_dir):
    message = assert_single_problem(
        clean_block(coord_files={"OA": "oa.csv"}), "coord_files.OA"
    )
    assert message == "not a configured level; levels are ['SGU', 'MGU', 'LGU', 'XLGU']"


@pytest.mark.parametrize("bad_path", [None, 3, [], ["a.csv", None]])
def test_coord_files_value_not_a_path_is_problem(working_dir, bad_path):
    message = assert_single_problem(
        clean_block(coord_files={"SGU": bad_path}), "coord_files.SGU"
    )
    assert message == f"must be a path or non-empty list of paths; got {bad_path!r}"


# load_all


def test_load_all_not_bool_is_problem(working_dir):
    message = assert_single_problem(clean_block(load_all="yes"), "load_all")
    assert message == "must be true or false; got 'yes'"


def test_load_all_true_with_filter_has_no_finding_and_no_filters(working_dir):
    geography_config, report = Geography.parse_config(
        clean_block(load_all=True, filter={"level": "OA", "codes": "bad"})
    )
    assert not report.problems and not report.warnings
    assert geography_config.filters is None


# filter


def test_filter_absent_loads_all(working_dir):
    geography_config, report = Geography.parse_config(clean_block(filter=None))
    assert not report.problems and not report.warnings
    assert geography_config.filters is None


def test_filter_not_mapping_is_problem(working_dir):
    message = assert_single_problem(clean_block(filter="LGU"), "filter")
    assert message == "must be a mapping; got scalar (str)"


def test_filter_level_not_configured_is_problem(working_dir):
    message = assert_single_problem(
        clean_block(filter={"level": "OA", "codes": ["x"]}), "filter.level"
    )
    assert message == "'OA' is not a configured level; levels are ['SGU', 'MGU', 'LGU', 'XLGU']"


def test_filter_codes_not_list_is_problem(working_dir):
    message = assert_single_problem(
        clean_block(filter={"level": "LGU", "codes": "Darlington"}), "filter.codes"
    )
    assert message == "must be a list; got scalar (str)"


def test_filter_codes_numbers_cast_to_strings(working_dir):
    geography_config, report = Geography.parse_config(
        clean_block(filter={"level": "LGU", "codes": [1001, "E02"]})
    )
    assert not report.problems
    assert geography_config.filters == {"level": "LGU", "codes": ["1001", "E02"]}


def test_filter_file_found_in_working_dir(working_dir):
    (working_dir / "codes.txt").write_text("# comment\nDarlington\n\nCounty Durham\n")
    geography_config, report = Geography.parse_config(
        clean_block(filter={"level": "LGU", "file": "codes.txt"})
    )
    assert not report.problems and not report.warnings
    assert geography_config.filters == {
        "level": "LGU",
        "codes": ["County Durham", "Darlington"],
    }


def test_filter_file_found_under_data_dir(working_dir):
    (working_dir / "data").mkdir()
    (working_dir / "data" / "codes.txt").write_text("Darlington\n")
    geography_config, report = Geography.parse_config(
        clean_block(filter={"level": "LGU", "file": "codes.txt"})
    )
    assert not report.problems
    assert geography_config.filters == {"level": "LGU", "codes": ["Darlington"]}


def test_filter_file_not_found_names_both_paths(working_dir):
    message = assert_single_problem(
        clean_block(filter={"level": "LGU", "file": "missing.txt"}), "filter.file"
    )
    assert message == "not found; tried missing.txt, data/missing.txt"


def test_filter_file_unreadable_is_problem(working_dir):
    (working_dir / "codes.txt").mkdir()
    message = assert_single_problem(
        clean_block(filter={"level": "LGU", "file": "codes.txt"}), "filter.file"
    )
    assert message.startswith("codes.txt could not be read")


def test_filter_file_not_a_path_is_problem(working_dir):
    message = assert_single_problem(
        clean_block(filter={"level": "LGU", "file": ["codes.txt"]}), "filter.file"
    )
    assert message == "must be a path; got ['codes.txt']"


# filter warnings


@pytest.mark.parametrize("codes", [None, []])
def test_filter_level_without_codes_warns_and_loads_all(working_dir, codes):
    geography_config, report = Geography.parse_config(
        clean_block(filter={"level": "LGU", "codes": codes})
    )
    assert not report.problems
    assert warning_lines(report) == [("filter", "level set but no codes; loads all units")]
    assert geography_config.filters is None


def test_filter_file_with_no_codes_warns_and_loads_all(working_dir):
    (working_dir / "codes.txt").write_text("# only a comment\n")
    geography_config, report = Geography.parse_config(
        clean_block(filter={"level": "LGU", "file": "codes.txt"})
    )
    assert not report.problems
    assert warning_lines(report) == [("filter", "level set but no codes; loads all units")]
    assert geography_config.filters is None


def test_filter_codes_and_file_warns_file_wins(working_dir):
    (working_dir / "codes.txt").write_text("Darlington\n")
    geography_config, report = Geography.parse_config(
        clean_block(filter={"level": "LGU", "codes": ["County Durham"], "file": "codes.txt"})
    )
    assert not report.problems
    assert warning_lines(report) == [("filter.codes", "ignored; filter.file is set")]
    assert geography_config.filters == {"level": "LGU", "codes": ["Darlington"]}


def test_filter_codes_without_level_warns_and_loads_all(working_dir):
    geography_config, report = Geography.parse_config(
        clean_block(filter={"codes": ["Darlington"]})
    )
    assert not report.problems
    assert warning_lines(report) == [
        ("filter.level", "not set; filter ignored, loads all units")
    ]
    assert geography_config.filters is None


def test_every_problem_reported_at_once(working_dir):
    geography_config, report = Geography.parse_config(
        {"data_dir": 3, "levels": "SGU", "load_all": "no", "filter": []}
    )
    assert geography_config is None
    assert [finding.key for finding in report.problems] == [
        "data_dir",
        "levels",
        "hierarchy_file",
        "load_all",
        "filter",
    ]
