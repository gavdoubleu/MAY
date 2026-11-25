import os

import pytest

from may.preflight.report import PreflightReport
from may.utils import path_resolver
from may.utils.path_resolver import (
    ConfigFileNotFoundError,
    DataFileNotFoundError,
    DataFileSpecError,
    as_path_list,
    check_data_files,
    data_file_paths,
    find_config_file,
)


@pytest.fixture(autouse=True)
def reset_resolver(monkeypatch):
    monkeypatch.setattr(path_resolver, "_resolver", None)


@pytest.fixture
def working_dir(tmp_path, monkeypatch):
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    return working


@pytest.fixture
def data_dir(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    return str(data)


def test_found_in_working_dir_returned_as_written(working_dir, data_dir):
    (working_dir / "venues.yaml").write_text("")
    assert find_config_file("venues.yaml", data_dir) == "venues.yaml"


def test_falls_back_to_data_dir(working_dir, data_dir):
    open(os.path.join(data_dir, "venues.yaml"), "w").close()
    assert find_config_file("venues.yaml", data_dir) == os.path.join(
        data_dir, "venues.yaml"
    )


def test_working_dir_wins_over_data_dir(working_dir, data_dir):
    (working_dir / "venues.yaml").write_text("")
    open(os.path.join(data_dir, "venues.yaml"), "w").close()
    assert find_config_file("venues.yaml", data_dir) == "venues.yaml"


def test_absolute_path_present_returned_as_written(tmp_path, data_dir):
    absolute = tmp_path / "venues.yaml"
    absolute.write_text("")
    assert find_config_file(str(absolute), data_dir) == str(absolute)


def test_absolute_path_absent_tries_only_itself(tmp_path, data_dir):
    absolute = str(tmp_path / "missing.yaml")
    with pytest.raises(ConfigFileNotFoundError) as caught:
        find_config_file(absolute, data_dir)
    assert caught.value.tried == [absolute]


def test_substitutes_roots_in_path_and_data_dir(tmp_path, working_dir, data_dir):
    path_resolver.init(config_root=str(tmp_path), data_root=data_dir, output_root="")
    (tmp_path / "venues.yaml").write_text("")
    assert find_config_file("${config_root}/venues.yaml", None) == str(
        tmp_path / "venues.yaml"
    )
    open(os.path.join(data_dir, "rules.yaml"), "w").close()
    assert find_config_file("rules.yaml", "${data_root}") == os.path.join(
        data_dir, "rules.yaml"
    )


@pytest.mark.parametrize("empty_data_dir", [None, ""])
def test_no_data_dir_tries_working_dir_only(working_dir, empty_data_dir):
    with pytest.raises(ConfigFileNotFoundError) as caught:
        find_config_file("venues.yaml", empty_data_dir)
    assert caught.value.tried == ["venues.yaml"]


def test_miss_raises_with_both_tried_paths(working_dir, data_dir):
    joined = os.path.join(data_dir, "venues.yaml")
    with pytest.raises(ConfigFileNotFoundError) as caught:
        find_config_file("venues.yaml", data_dir)
    error = caught.value
    assert isinstance(error, FileNotFoundError)
    assert error.path == "venues.yaml"
    assert error.tried == ["venues.yaml", joined]
    assert str(error) == f"not found; tried venues.yaml, {joined}"


def test_a_path_object_miss_raises_not_found_naming_it(working_dir, data_dir):
    from pathlib import Path

    with pytest.raises(ConfigFileNotFoundError, match="missing.yaml") as caught:
        find_config_file(Path("nope/missing.yaml"), data_dir)
    assert caught.value.tried[0] == os.path.join("nope", "missing.yaml")


def test_a_path_object_is_found_as_text(working_dir, data_dir):
    from pathlib import Path

    (working_dir / "venues.yaml").write_text("")
    assert find_config_file(Path("venues.yaml"), data_dir) == "venues.yaml"


class TestAsPathList:
    def test_single_string_becomes_one_element_list(self):
        assert as_path_list("a.csv", "x") == ["a.csv"]

    def test_list_passes_through(self):
        assert as_path_list(["a.csv", "b.csv"], "x") == ["a.csv", "b.csv"]

    def test_empty_list_raises(self):
        with pytest.raises(DataFileSpecError, match="empty"):
            as_path_list([], "households.data_file")

    def test_non_string_entry_raises(self):
        with pytest.raises(DataFileSpecError, match="path string"):
            as_path_list(["a.csv", 3], "x")

    def test_non_path_value_raises(self):
        with pytest.raises(DataFileSpecError, match="expected a path"):
            as_path_list({"a": 1}, "x")

    @pytest.mark.parametrize("spec", ["", ["a.csv", ""]])
    def test_empty_path_raises(self, spec):
        with pytest.raises(DataFileSpecError, match="households.data_file"):
            as_path_list(spec, "households.data_file")

    def test_spec_error_is_a_value_error(self):
        with pytest.raises(ValueError):
            as_path_list(3, "x")


class TestDataFilePaths:
    def _touch(self, directory, name):
        path = os.path.join(directory, name)
        open(path, "w").close()
        return path

    def test_single_relative_path_joined_onto_data_dir(self, data_dir):
        joined = self._touch(data_dir, "a.csv")
        assert data_file_paths("a.csv", data_dir, "x") == [joined]

    def test_list_keeps_order(self, data_dir):
        second = self._touch(data_dir, "b.csv")
        first = self._touch(data_dir, "a.csv")
        assert data_file_paths(["b.csv", "a.csv"], data_dir, "x") == [second, first]

    def test_relative_data_dir_gives_absolute_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        os.mkdir("data")
        self._touch("data", "a.csv")
        assert data_file_paths("a.csv", "data", "x") == [str(tmp_path / "data" / "a.csv")]

    def test_absolute_path_kept_as_written(self, tmp_path, data_dir):
        absolute = self._touch(str(tmp_path), "elsewhere.csv")
        assert data_file_paths(absolute, data_dir, "x") == [absolute]

    def test_substitutes_roots_in_path_and_data_dir(self, tmp_path, data_dir):
        path_resolver.init(config_root=str(tmp_path), data_root=data_dir, output_root="")
        absolute = self._touch(str(tmp_path), "a.csv")
        joined = self._touch(data_dir, "b.csv")
        assert data_file_paths("${config_root}/a.csv", None, "x") == [absolute]
        assert data_file_paths("b.csv", "${data_root}", "x") == [joined]

    def test_a_path_object_is_taken_as_text(self, data_dir):
        from pathlib import Path

        joined = self._touch(data_dir, "a.csv")
        assert data_file_paths(Path("a.csv"), data_dir, "x") == [joined]

    @pytest.mark.parametrize("spec", ["", [], ["a.csv", 3], {"a": 1}, None])
    def test_bad_shape_raises_spec_error_naming_label(self, data_dir, spec):
        with pytest.raises(DataFileSpecError, match="geography.hierarchy_file"):
            data_file_paths(spec, data_dir, "geography.hierarchy_file")

    def test_missing_files_all_listed(self, data_dir):
        self._touch(data_dir, "present.csv")
        missing_first = os.path.join(data_dir, "gone_a.csv")
        missing_second = os.path.join(data_dir, "gone_b.csv")
        with pytest.raises(DataFileNotFoundError) as caught:
            data_file_paths(["gone_a.csv", "present.csv", "gone_b.csv"], data_dir, "x")
        error = caught.value
        assert isinstance(error, FileNotFoundError)
        assert error.missing == [missing_first, missing_second]
        assert str(error) == f"not found: {missing_first}, {missing_second}"


class TestCheckDataFiles:
    def test_present_files_returned_without_problem(self, data_dir):
        open(os.path.join(data_dir, "a.csv"), "w").close()
        report = PreflightReport()
        paths = check_data_files(report, "hierarchy_file", "a.csv", data_dir)
        assert paths == [os.path.join(data_dir, "a.csv")]
        assert not report.problems

    @pytest.mark.parametrize("spec", ["", [], ["a.csv", 3], {"a": 1}, None])
    def test_bad_shape_is_one_problem_under_key(self, data_dir, spec):
        report = PreflightReport()
        assert check_data_files(report, "coord_files.SGU", spec, data_dir) is None
        assert [(finding.key, finding.message) for finding in report.problems] == [
            ("coord_files.SGU", f"must be a path or non-empty list of paths; got {spec!r}")
        ]

    def test_missing_files_are_one_problem_listing_all(self, data_dir):
        report = PreflightReport()
        assert check_data_files(report, "hierarchy_file", ["a.csv", "b.csv"], data_dir) is None
        missing = [os.path.join(data_dir, "a.csv"), os.path.join(data_dir, "b.csv")]
        assert [(finding.key, finding.message) for finding in report.problems] == [
            ("hierarchy_file", f"not found: {missing[0]}, {missing[1]}")
        ]

    def test_no_data_dir_checks_shape_only(self):
        report = PreflightReport()
        assert check_data_files(report, "hierarchy_file", "a.csv", None) is None
        assert not report.problems
        check_data_files(report, "hierarchy_file", 3, None)
        assert [finding.key for finding in report.problems] == ["hierarchy_file"]
