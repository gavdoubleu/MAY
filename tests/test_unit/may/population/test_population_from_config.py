import logging

import pytest

from may.geography import Geography
from may.population.population import PopulationError, PopulationManager
from may.utils import path_resolver

MICRO_WORLD = "tests/test_data/micro_world"
COLUMN_MAPPING = {"age": "age", "sex": "sex", "geo_unit": "location"}


@pytest.fixture(autouse=True)
def reset_resolver(monkeypatch):
    monkeypatch.setattr(path_resolver, "_resolver", None)


@pytest.fixture
def geography():
    loaded = Geography(
        data_dir=f"{MICRO_WORLD}/geography", levels=["SGU", "MGU", "LGU"], hierarchy_file="hierarchy.csv"
    )
    loaded.load_from_csv()
    return loaded


def test_problems_raise_population_error_with_every_line(geography):
    block = {"type": "explicit", "data_dir": 3, "column_mapping": {"age": 5}}
    with pytest.raises(PopulationError) as raised:
        PopulationManager.from_config(block, geography)
    message = str(raised.value)
    assert message.startswith("Population config has 3 problem(s):")
    for key in ("data_dir", "column_mapping.age", "filename"):
        assert key in message


def test_explicit_block_builds_population(geography):
    block = {
        "type": "explicit",
        "data_dir": f"{MICRO_WORLD}/population",
        "filename": "people.csv",
        "column_mapping": COLUMN_MAPPING,
    }
    population = PopulationManager.from_config(block, geography)
    assert len(population) == 16
    assert population.geography is geography


def test_explicit_batch_block_builds_population(geography):
    block = {
        "type": "explicit_batch",
        "data_dir": f"{MICRO_WORLD}/population/batch",
        "level": "MGU",
        "column_mapping": COLUMN_MAPPING,
    }
    population = PopulationManager.from_config(block, geography)
    assert len(population) == 4


def test_matrix_block_generates_population(geography):
    block = {"type": "matrix", "data_dir": f"{MICRO_WORLD}/population/matrix"}
    population = PopulationManager.from_config(block, geography)
    assert len(population) == 10


def test_batch_level_unknown_to_loaded_geography_is_a_problem(geography):
    block = {"type": "explicit_batch", "data_dir": f"{MICRO_WORLD}/population/batch", "level": "REGION"}
    with pytest.raises(PopulationError, match="level"):
        PopulationManager.from_config(block, geography)


def test_warnings_are_logged_at_info(geography, caplog):
    block = {
        "type": "matrix",
        "data_dir": f"{MICRO_WORLD}/population/matrix",
        "filename": "people.csv",
    }
    with caplog.at_level(logging.INFO, logger="population"):
        PopulationManager.from_config(block, geography)
    warning_lines = [
        record for record in caplog.records
        if record.levelno == logging.INFO and "filename" in record.getMessage()
    ]
    assert len(warning_lines) == 1
    assert "ignored by population type 'matrix'" in warning_lines[0].getMessage()
