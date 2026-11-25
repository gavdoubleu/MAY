import pytest

from may.population.population import (
    DEFAULT_DATA_DIR,
    DEFAULT_FEMALE_FILE,
    DEFAULT_MALE_FILE,
    DEFAULT_TYPE,
    PopulationManager,
)
from may.preflight.context import ConfigContext, StageNotParsed
from may.utils import path_resolver
from tests.test_unit.may.config_contexts import geography_context

BLANK_OPTIONAL = "left blank; give a value, or remove the key to use the default"


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
    batch_dir = working / "1911_data" / "population"
    batch_dir.mkdir(parents=True)
    (batch_dir / "M1_pop.csv").write_text("")
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


def without(block, *keys):
    for key in keys:
        del block[key]
    return block


def parse(block, context=None):
    return PopulationManager.preflight(block, context or geography_context())


def problem_lines(report):
    return [(finding.key, finding.message) for finding in report.problems]


def assert_single_problem(block, key, context=None):
    population_config, report = parse(block, context)
    assert population_config is None
    assert [finding.key for finding in report.problems] == [f"population.{key}"]
    assert not report.warnings
    return report.problems[0].message


def assert_single_warning(block, key):
    population_config, report = parse(block)
    assert population_config is not None
    assert not report.problems
    assert [finding.key for finding in report.warnings] == [f"population.{key}" if key else "population"]
    return report.warnings[0].message


# Clean blocks


def test_clean_matrix_block_is_clean_and_parsed(working_dir):
    population_config, report = parse(matrix_block())
    assert not report.problems and not report.warnings
    assert population_config.data_dir == "data/population"
    assert population_config.type == "matrix"
    assert population_config.male_file == [data_path(working_dir, "EW_demographics_male.csv")]
    assert population_config.female_file == [data_path(working_dir, "EW_demographics_female.csv")]
    assert population_config.column_mapping == {}
    assert population_config.filename is None
    assert population_config.level is None


def test_clean_batch_block_is_clean_and_parsed():
    population_config, report = parse(batch_block())
    assert not report.problems and not report.warnings
    assert population_config.type == "explicit_batch"
    assert population_config.data_dir == "1911_data/population"
    assert population_config.column_mapping == {"age": "Age", "sex": "Sex", "Occode": "Occode"}
    assert population_config.level == "MGU"


def test_clean_explicit_block_is_clean_and_parsed(working_dir):
    population_config, report = parse(explicit_block())
    assert not report.problems and not report.warnings
    assert population_config.type == "explicit"
    assert population_config.filename == [data_path(working_dir, "people.csv")]


def test_key_path_can_be_overridden():
    _, report = PopulationManager.preflight(
        batch_block(level="NOPE"), geography_context(), key_path="world.people"
    )
    assert [finding.key for finding in report.problems] == ["world.people.level"]


def test_paths_are_resolved(working_dir):
    data_root = str(working_dir / "data")
    path_resolver.init(config_root="/config", data_root=data_root, output_root="/out")
    population_config, report = parse(
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


# Row: block empty → defaults warning; blank or not a mapping → problem


def test_empty_block_warns_defaults_in_use(working_dir):
    message = assert_single_warning({}, "")
    assert "default" in message
    population_config, _ = parse({})
    assert population_config.type == DEFAULT_TYPE == "matrix"
    assert population_config.data_dir == DEFAULT_DATA_DIR
    assert population_config.male_file == [data_path(working_dir, DEFAULT_MALE_FILE)]
    assert population_config.female_file == [data_path(working_dir, DEFAULT_FEMALE_FILE)]


@pytest.mark.parametrize(
    "block, message",
    [(None, "left blank; give a value"), ("matrix", "must be a mapping; got text 'matrix'")],
    ids=["blank", "not_mapping"],
)
def test_blank_or_not_mapping_block_is_one_problem(block, message):
    population_config, report = parse(block)
    assert population_config is None
    assert problem_lines(report) == [("population", message)]


# Row: data_dir


@pytest.mark.parametrize(
    "data_dir, message",
    [
        (["a"], "must be text; got a list"),
        ({"a": 1}, "must be text; got a mapping"),
        (None, BLANK_OPTIONAL),
        ("", "must not be empty; give a path, or remove the key to use the default"),
    ],
    ids=["list", "mapping", "blank", "empty"],
)
def test_data_dir_problem(data_dir, message):
    population_config, report = parse(matrix_block(data_dir=data_dir))
    assert population_config is None
    assert problem_lines(report) == [("population.data_dir", message)]


def test_data_dir_whole_number_is_read_as_text(working_dir):
    (working_dir / "5").mkdir()
    for name in ("EW_demographics_male.csv", "EW_demographics_female.csv"):
        (working_dir / "5" / name).write_text("")
    population_config, report = parse(matrix_block(data_dir=5))
    assert not report.problems
    assert population_config.data_dir == "5"


def test_failed_data_dir_checks_file_shape_only():
    population_config, report = parse(matrix_block(data_dir=["a"], demographics_male_file={"a": 1}))
    assert population_config is None
    assert [finding.key for finding in report.problems] == [
        "population.data_dir",
        "population.demographics_male_file",
    ]


# Row: type


@pytest.mark.parametrize(
    "population_type, shown",
    [("matrx", "text 'matrx'"), (3, "a whole number 3"), (["matrix"], "a list")],
)
def test_unknown_type(population_type, shown):
    message = assert_single_problem(matrix_block(type=population_type), "type")
    assert message == f"must be one of explicit, explicit_batch, matrix; got {shown}"


def test_blank_type_is_problem():
    message = assert_single_problem(matrix_block(type=None), "type")
    assert message == BLANK_OPTIONAL


def test_unset_type_is_matrix():
    population_config, _ = parse(matrix_block())
    assert population_config.type == "matrix"


def test_unknown_type_still_checks_data_dir():
    _, report = parse(matrix_block(type="census", data_dir=["a"]))
    assert [finding.key for finding in report.problems] == ["population.data_dir", "population.type"]


# Row: column_mapping


@pytest.mark.parametrize(
    "column_mapping, message",
    [
        (["age"], "must be a mapping; got a list"),
        ("age", "must be a mapping; got text 'age'"),
        (None, BLANK_OPTIONAL),
    ],
    ids=["list", "text", "blank"],
)
def test_column_mapping_not_a_mapping(column_mapping, message):
    population_config, report = parse(batch_block(column_mapping=column_mapping))
    assert population_config is None
    assert problem_lines(report) == [("population.column_mapping", message)]


@pytest.mark.parametrize(
    "column_name, message",
    [
        (None, "left blank; give a column name"),
        ("", "must not be empty; give a column name"),
        (["Age"], "must be text; got a list"),
    ],
    ids=["blank", "empty", "list"],
)
def test_column_mapping_value_not_a_column_name(column_name, message):
    population_config, report = parse(batch_block(column_mapping={"age": column_name}))
    assert population_config is None
    assert problem_lines(report) == [("population.column_mapping.age", message)]


def test_column_mapping_whole_number_column_is_read_as_text():
    population_config, report = parse(batch_block(column_mapping={"age": 5}))
    assert not report.problems
    assert population_config.column_mapping == {"age": "5"}


def test_column_mapping_bad_entry_leaves_siblings_read():
    _, report = parse(batch_block(column_mapping={"age": None, "sex": ""}))
    assert [finding.key for finding in report.problems] == [
        "population.column_mapping.age",
        "population.column_mapping.sex",
    ]


# Row: explicit filename


def test_explicit_filename_unset():
    message = assert_single_problem(without(explicit_block(), "filename"), "filename")
    assert message == "not set; give a path"


@pytest.mark.parametrize(
    "filename, message",
    [
        ("", "must be a path or a non-empty list of paths; got text ''"),
        (["a.csv"], "must be a path; got a list"),
        (None, "left blank; give a path"),
    ],
    ids=["empty", "list", "blank"],
)
def test_explicit_filename_not_a_path(filename, message):
    assert assert_single_problem(explicit_block(filename=filename), "filename") == message


def test_explicit_filename_missing_file(working_dir):
    (working_dir / "data" / "population" / "people.csv").unlink()
    message = assert_single_problem(explicit_block(), "filename")
    assert "people.csv" in message


# Row: matrix demographics files


@pytest.mark.parametrize("key", ["demographics_male_file", "demographics_female_file"])
@pytest.mark.parametrize("spec", [5, [], ["a.csv", 3], ""])
def test_matrix_demographics_file_not_a_path_spec(key, spec):
    assert_single_problem(matrix_block(**{key: spec}), key)


@pytest.mark.parametrize("key", ["demographics_male_file", "demographics_female_file"])
def test_matrix_blank_demographics_file_is_problem(key):
    assert assert_single_problem(matrix_block(**{key: None}), key) == BLANK_OPTIONAL


def test_matrix_demographics_file_list_accepted(working_dir):
    population_config, report = parse(matrix_block(demographics_male_file=["a.csv", "b.csv"]))
    assert not report.problems
    assert population_config.male_file == [data_path(working_dir, "a.csv"), data_path(working_dir, "b.csv")]


def test_matrix_unset_demographics_files_use_defaults(working_dir):
    block = without(matrix_block(), "demographics_male_file", "demographics_female_file")
    population_config, report = parse(block)
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
    block = without(matrix_block(), "demographics_male_file", "demographics_female_file")
    message = assert_single_problem(block, key)
    assert name in message


def test_matrix_stacked_demographics_lists_every_missing_file():
    message = assert_single_problem(
        matrix_block(demographics_male_file=["a.csv", "gone_1.csv", "gone_2.csv"]),
        "demographics_male_file",
    )
    assert "gone_1.csv" in message and "gone_2.csv" in message


def test_batch_data_dir_missing_is_problem():
    message = assert_single_problem(batch_block(data_dir="gone"), "data_dir")
    assert "not found" in message


def test_batch_data_dir_without_unit_files_is_problem():
    message = assert_single_problem(batch_block(data_dir="data/population"), "data_dir")
    assert "_pop.csv" in message


# Rows: explicit_batch level


def test_batch_level_unset():
    message = assert_single_problem(without(batch_block(), "level"), "level")
    assert message == "not set; give a geo level name"


def test_batch_level_blank():
    message = assert_single_problem(batch_block(level=None), "level")
    assert message == "left blank; give a geo level name"


def test_batch_level_unknown():
    message = assert_single_problem(batch_block(level="NOPE"), "level")
    assert message == "unknown geo level 'NOPE'; expected one of ['LGU', 'MGU', 'SGU']"


def test_batch_level_not_a_name():
    message = assert_single_problem(batch_block(level=["MGU"]), "level")
    assert message == "must be text; got a list"


def test_batch_level_whole_number_is_read_as_text():
    population_config, report = parse(batch_block(level=1), geography_context(levels=("1", "2")))
    assert not report.problems
    assert population_config.level == "1"


def test_batch_level_check_skipped_when_geography_failed():
    population_config, report = parse(batch_block(level="NOPE"), geography_context(levels=None))
    assert not report.problems
    assert population_config.level == "NOPE"


def test_batch_level_without_parsed_geography_raises():
    with pytest.raises(StageNotParsed):
        parse(batch_block(), ConfigContext())


def test_matrix_needs_no_geography_in_context():
    population_config, report = parse(matrix_block(), ConfigContext())
    assert population_config is not None and not report.problems


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
    message = assert_single_warning(block, key)
    assert "ignored" in message


# Row: unknown keys → no finding


def test_unknown_key_gives_no_finding():
    population_config, report = parse(matrix_block(colour="blue"))
    assert population_config is not None
    assert not report.problems and not report.warnings


def test_every_problem_reported_together():
    _, report = parse(batch_block(data_dir=["a"], column_mapping=[], level="NOPE"))
    assert [finding.key for finding in report.problems] == [
        "population.data_dir",
        "population.column_mapping",
        "population.level",
    ]
