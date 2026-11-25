import logging

import pytest

from may.geography.venue_manager import VenueConfig, VenueManager
from may.preflight.config_stage import ConfigError
from may.serialization.serialization_config import SerialisationError
from may.serialization.world_serializer import SerialisationConfigError, WorldSerializer
from may.utils import path_resolver
from tests.test_unit.may.config_contexts import geography_context

SCHEMA_2021 = "configs/2021/serialization_config.yaml"


@pytest.fixture(autouse=True)
def reset_resolver(monkeypatch):
    monkeypatch.setattr(path_resolver, "_resolver", None)


def from_config(block):
    venue_config = VenueConfig(
        data_dir="data", config_path="venues.yaml", venue_types={}, filter_by_geography=None
    )
    return WorldSerializer.from_config(block, geography_context(extra_results={VenueManager: venue_config}))


def test_error_is_config_and_serialisation_error():
    assert issubclass(SerialisationConfigError, ConfigError)
    assert issubclass(SerialisationConfigError, SerialisationError)


def test_problems_raise_serialisation_config_error_with_every_line():
    block = {"enabled": True, "config_file": None, "output_dir": 3.5, "filename": []}
    with pytest.raises(SerialisationConfigError) as raised:
        from_config(block)
    message = str(raised.value)
    assert message.startswith("Serialisation config cannot be used: 3 problems.")
    for key in ("config_file", "output_dir", "filename"):
        assert f"serialization.{key}" in message
    assert len(raised.value.report.problems) == 3


def test_warnings_logged_at_warning(tmp_path, caplog):
    schema_file = tmp_path / "schema.yaml"
    schema_file.write_text("population:\n  properties: []\nextras: {}\n")
    with caplog.at_level(logging.INFO, logger="may.serialization.world_serializer"):
        from_config({"config_file": str(schema_file)})
    warnings = [record for record in caplog.records if "extras" in record.getMessage()]
    assert warnings and all(record.levelno == logging.WARNING for record in warnings)


def test_disabled_gives_settings_without_schema():
    settings = from_config({"enabled": False, "config_file": 7})
    assert settings.enabled is False
    assert settings.schema is None


def test_defaults_applied():
    settings = from_config({"config_file": SCHEMA_2021})
    assert settings.enabled is True
    assert settings.schema_path == SCHEMA_2021
    assert settings.schema.population_properties
    assert settings.output_dir == "."
    assert settings.filename == "world_state.h5"
