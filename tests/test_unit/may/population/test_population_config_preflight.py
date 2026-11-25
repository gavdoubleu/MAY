import pytest

from may.population.population import (
    DEFAULT_DATA_DIR,
    DEFAULT_FEMALE_FILE,
    DEFAULT_MALE_FILE,
    DEFAULT_TYPE,
    PopulationManager,
)
from may.utils import path_resolver

GEO_LEVELS = frozenset({"SGU", "MGU", "LGU", "XLGU"})


@pytest.fixture(autouse=True)
def reset_resolver(monkeypatch):
    monkeypatch.setattr(path_resolver, "_resolver", None)


DATA_FILES = [
    "EW_demographics_male.csv",
    "EW_demographics_female.csv",
    DEFAULT_MALE_FILE,
    DEFAULT_FEMALE_FILE,
    "people.csv",
    "a.csv",
    "b.csv",
]


@pytest.fixture(autouse=True)
def working_dir(tmp_path, monkeypatch):
    """Every data file the blocks below name exists under `data/population`."""
    working = tmp_path / "working"
    population_dir = working / "data" / "population"
    population_dir.mkdir(parents=True)
    monkeypatch.chdir(working)
    for name in DATA_FILES:
        (population_dir / name).write_text("")
    return working


def data_path(working_dir, name):
    return str(working_dir / "data" / "population" / name)


def matrix_block(**overrides):
    """A block shaped like `configs/2021`'s `population:` block."""
    block = {
        "data_dir": "data/population",
        "demographics_male_file": "EW_demographics_male.csv",
        "demographics_female_file": "EW_demographics_female.csv",
    }
    block.update(overrides)
    return block


def batch_block(**overrides):
    """A block shaped like `configs/1911`'s `population:` block, with `level`."""
    block = {
        "type": "explicit_batch",
        "data_dir": "1911_data/population",
        "column_mapping": {"age": "Age", "sex": "Sex", "Occode": "Occode"},
        "level": "MGU",
    }
    block.update(overrides)
    return block


def explicit_block(**overrides):
    block = {
        "type": "explicit",
        "data_dir": "data/population",
        "filename": "people.csv",
        "column_mapping": {"age": "Age"},
    }
    block.update(overrides)
    return block


def assert_single_problem(block, key, geo_levels=None):
    population_config, report = PopulationManager.parse_config(block, geo_levels)
    assert population_config is None
    assert [finding.key for finding in report.problems] == [key]
    assert not report.warnings
    return report.problems[0].message


def assert_single_warning(block, key, geo_levels=None):
    population_config, report = PopulationManager.parse_config(block, geo_levels)
    assert population_config is not None
    assert not report.problems
    assert [finding.key for finding in report.warnings] == [key]
    return report.warnings[0].message


# Clean blocks


def test_clean_matrix_block_is_clean_and_parsed(working_dir):
    population_config, report = PopulationManager.parse_config(matrix_block())
    assert not report.problems and not report.warnings
    assert population_config.data_dir == "data/population"
    assert population_config.type == "matrix"
    assert population_config.male_file == [data_path(working_dir, "EW_demographics_male.csv")]
    assert population_config.female_file == [data_path(working_dir, "EW_demographics_female.csv")]
    assert population_config.column_mapping == {}
    assert population_config.filename is None
    assert population_config.level is None


def test_clean_batch_block_is_clean_and_parsed():
    population_config, report = PopulationManager.parse_config(batch_block(), GEO_LEVELS)
    assert not report.problems and not report.warnings
    assert population_config.type == "explicit_batch"
    assert population_config.data_dir == "1911_data/population"
    assert population_config.column_mapping == {"age": "Age", "sex": "Sex", "Occode": "Occode"}
    assert population_config.level == "MGU"


def test_clean_explicit_block_is_clean_and_parsed(working_dir):
    population_config, report = PopulationManager.parse_config(explicit_block())
    assert not report.problems and not report.warnings
    assert population_config.type == "explicit"
    assert population_config.filename == [data_path(working_dir, "people.csv")]


def test_preflight_returns_parse_report():
    report = PopulationManager.preflight(batch_block(level="NOPE"), GEO_LEVELS)
    assert [finding.key for finding in report.problems] == ["level"]


def test_paths_are_resolved(working_dir):
    data_root = str(working_dir / "data")
    path_resolver.init(config_root="/config", data_root=data_root, output_root="/out")
    population_config, report = PopulationManager.parse_config(
        matrix_block(
            data_dir="${data_root}/population",
            demographics_male_file=["${data_root}/population/a.csv", "b.csv"],
            demographics_female_file="${data_root}/population/people.csv",
        )
    )
    assert not report.problems
    assert population_config.data_dir == f"{data_root}/population"
    assert population_config.male_file == [
        data_path(working_dir, "a.csv"),
        data_path(working_dir, "b.csv"),
    ]
    assert population_config.female_file == [data_path(working_dir, "people.csv")]


# Row: block absent or blank → defaults warning


@pytest.mark.parametrize("block", [{}, None])
def test_absent_or_blank_block_warns_defaults_in_use(block, working_dir):
    message = assert_single_warning(block, "")
    assert "default" in message
    population_config, _ = PopulationManager.parse_config(block)
    assert population_config.type == DEFAULT_TYPE == "matrix"
    assert population_config.data_dir == DEFAULT_DATA_DIR
    assert population_config.male_file == [data_path(working_dir, DEFAULT_MALE_FILE)]
    assert population_config.female_file == [data_path(working_dir, DEFAULT_FEMALE_FILE)]


# Row: data_dir


@pytest.mark.parametrize("data_dir", [5, ["a"], {"a": 1}])
def test_data_dir_not_a_path(data_dir):
    assert_single_problem(matrix_block(data_dir=data_dir), "data_dir")


@pytest.mark.parametrize("data_dir", [None, ""])
def test_blank_data_dir_falls_back_to_default(data_dir):
    population_config, report = PopulationManager.parse_config(matrix_block(data_dir=data_dir))
    assert not report.problems and not report.warnings
    assert population_config.data_dir == DEFAULT_DATA_DIR


# Row: type


@pytest.mark.parametrize("population_type", ["matrx", 3, ["matrix"]])
def test_unknown_type(population_type):
    message = assert_single_problem(matrix_block(type=population_type), "type")
    assert "explicit_batch" in message


def test_unset_type_is_matrix():
    population_config, _ = PopulationManager.parse_config(matrix_block())
    assert population_config.type == "matrix"


# Row: column_mapping


@pytest.mark.parametrize("column_mapping", [["age"], "age", 5])
def test_column_mapping_not_a_mapping(column_mapping):
    assert_single_problem(batch_block(column_mapping=column_mapping), "column_mapping", GEO_LEVELS)


@pytest.mark.parametrize("column_name", [5, None, "", ["Age"]])
def test_column_mapping_value_not_a_column_name(column_name):
    assert_single_problem(
        batch_block(column_mapping={"age": column_name}), "column_mapping.age", GEO_LEVELS
    )


# Row: explicit filename


def test_explicit_filename_unset():
    block = explicit_block()
    del block["filename"]
    assert_single_problem(block, "filename")


@pytest.mark.parametrize("filename", ["", 5, ["a.csv"]])
def test_explicit_filename_not_a_path(filename):
    assert_single_problem(explicit_block(filename=filename), "filename")


def test_explicit_filename_missing_file(working_dir):
    (working_dir / "data" / "population" / "people.csv").unlink()
    message = assert_single_problem(explicit_block(), "filename")
    assert "people.csv" in message


# Row: matrix demographics files


@pytest.mark.parametrize("key", ["demographics_male_file", "demographics_female_file"])
@pytest.mark.parametrize("spec", [5, [], ["a.csv", 3], ""])
def test_matrix_demographics_file_not_a_path_spec(key, spec):
    assert_single_problem(matrix_block(**{key: spec}), key)


def test_matrix_demographics_file_list_accepted(working_dir):
    population_config, report = PopulationManager.parse_config(
        matrix_block(demographics_male_file=["a.csv", "b.csv"])
    )
    assert not report.problems
    assert population_config.male_file == [data_path(working_dir, "a.csv"), data_path(working_dir, "b.csv")]


def test_matrix_unset_demographics_files_use_defaults(working_dir):
    block = matrix_block()
    del block["demographics_male_file"]
    del block["demographics_female_file"]
    population_config, report = PopulationManager.parse_config(block)
    assert not report.problems and not report.warnings
    assert population_config.male_file == [data_path(working_dir, DEFAULT_MALE_FILE)]
    assert population_config.female_file == [data_path(working_dir, DEFAULT_FEMALE_FILE)]


@pytest.mark.parametrize(
    "key, name",
    [
        ("demographics_male_file", "EW_demographics_male.csv"),
        ("demographics_female_file", "EW_demographics_female.csv"),
    ],
)
def test_matrix_demographics_file_missing(working_dir, key, name):
    (working_dir / "data" / "population" / name).unlink()
    message = assert_single_problem(matrix_block(), key)
    assert name in message


@pytest.mark.parametrize(
    "key, name",
    [
        ("demographics_male_file", DEFAULT_MALE_FILE),
        ("demographics_female_file", DEFAULT_FEMALE_FILE),
    ],
)
def test_matrix_default_demographics_file_missing(working_dir, key, name):
    (working_dir / "data" / "population" / name).unlink()
    block = matrix_block()
    del block["demographics_male_file"]
    del block["demographics_female_file"]
    message = assert_single_problem(block, key)
    assert name in message


def test_matrix_stacked_demographics_lists_every_missing_file(working_dir):
    message = assert_single_problem(
        matrix_block(demographics_male_file=["a.csv", "gone_1.csv", "gone_2.csv"]),
        "demographics_male_file",
    )
    assert "gone_1.csv" in message and "gone_2.csv" in message


def test_batch_files_not_existence_checked():
    population_config, report = PopulationManager.parse_config(batch_block(), GEO_LEVELS)
    assert population_config is not None
    assert not report.problems


# Rows: explicit_batch level


def test_batch_level_unset():
    block = batch_block()
    del block["level"]
    assert_single_problem(block, "level", GEO_LEVELS)


def test_batch_level_unknown():
    message = assert_single_problem(batch_block(level="NOPE"), "level", GEO_LEVELS)
    assert "NOPE" in message


def test_batch_level_not_a_name():
    assert_single_problem(batch_block(level=3), "level", GEO_LEVELS)


def test_batch_level_check_skipped_without_geography():
    population_config, report = PopulationManager.parse_config(batch_block(level="NOPE"), None)
    assert not report.problems
    assert population_config.level == "NOPE"


# Row: key unused by chosen type → warning


@pytest.mark.parametrize(
    "block, key",
    [
        (matrix_block(filename="people.csv"), "filename"),
        (batch_block(filename="people.csv"), "filename"),
        (explicit_block(demographics_male_file="a.csv"), "demographics_male_file"),
        (batch_block(demographics_female_file="a.csv"), "demographics_female_file"),
        (matrix_block(column_mapping={"age": "Age"}), "column_mapping"),
        (matrix_block(level="MGU"), "level"),
        (explicit_block(level="MGU"), "level"),
    ],
)
def test_key_unused_by_type_warns(block, key):
    message = assert_single_warning(block, key, GEO_LEVELS)
    assert "ignored" in message


# Row: unknown keys → no finding


def test_unknown_key_gives_no_finding():
    population_config, report = PopulationManager.parse_config(matrix_block(colour="blue"))
    assert population_config is not None
    assert not report.problems and not report.warnings


def test_every_problem_reported_together():
    _, report = PopulationManager.parse_config(
        batch_block(data_dir=5, column_mapping=[], level="NOPE"), GEO_LEVELS
    )
    assert [finding.key for finding in report.problems] == ["data_dir", "column_mapping", "level"]
