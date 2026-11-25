import logging

import pytest

from may.geography import Geography
from may.population.population import PopulationConfigError, PopulationError, PopulationManager
from may.preflight.config_stage import ConfigError
from may.utils import path_resolver
from tests.test_unit.may.config_contexts import geography_context

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
    block = {"type": "explicit", "data_dir": [3], "column_mapping": {"age": None}}
    with pytest.raises(PopulationConfigError) as raised:
        PopulationManager.from_config(block, geography_context(), geography=geography)
    assert isinstance(raised.value, ConfigError) and isinstance(raised.value, PopulationError)
    message = str(raised.value)
    assert message.startswith("Population config cannot be used: 3 problems.")
    for key in ("population.data_dir", "population.column_mapping.age", "population.filename"):
        assert key in message


def test_explicit_block_builds_population(geography):
    block = {
        "type": "explicit",
        "data_dir": f"{MICRO_WORLD}/population",
        "filename": "people.csv",
        "column_mapping": COLUMN_MAPPING,
    }
    population = PopulationManager.from_config(block, geography_context(), geography=geography)
    assert len(population) == 16
    assert population.geography is geography


def test_explicit_batch_block_builds_population(geography):
    block = {
        "type": "explicit_batch",
        "data_dir": f"{MICRO_WORLD}/population/batch",
        "level": "MGU",
        "column_mapping": COLUMN_MAPPING,
    }
    population = PopulationManager.from_config(block, geography_context(), geography=geography)
    assert len(population) == 4


def test_matrix_block_generates_population(geography):
    block = {"type": "matrix", "data_dir": f"{MICRO_WORLD}/population/matrix"}
    population = PopulationManager.from_config(block, geography_context(), geography=geography)
    assert len(population) == 10


def test_batch_level_unknown_to_loaded_geography_is_a_problem(geography):
    block = {"type": "explicit_batch", "data_dir": f"{MICRO_WORLD}/population/batch", "level": "REGION"}
    with pytest.raises(PopulationConfigError, match="population.level"):
        PopulationManager.from_config(block, geography_context(), geography=geography)


def test_warnings_are_logged_at_warning(geography, caplog):
    block = {
        "type": "matrix",
        "data_dir": f"{MICRO_WORLD}/population/matrix",
        "filename": "people.csv",
    }
    with caplog.at_level(logging.INFO, logger="may.population.population"):
        PopulationManager.from_config(block, geography_context(), geography=geography)
    warning_records = [
        record for record in caplog.records if "population.filename" in record.getMessage()
    ]
    assert [record.levelno for record in warning_records] == [logging.WARNING]
    assert "ignored by population type 'matrix'" in warning_records[0].getMessage()
