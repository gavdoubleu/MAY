import logging

import pytest

from may.serialization.serialization_config import SerialisationError
from may.serialization.world_serializer import WorldSerializer
from may.utils import path_resolver

SCHEMA_2021 = "configs/2021/serialization_config.yaml"


@pytest.fixture(autouse=True)
def reset_resolver(monkeypatch):
    monkeypatch.setattr(path_resolver, "_resolver", None)


def test_problems_raise_serialisation_error_with_every_line():
    block = {"enabled": True, "config_file": None, "output_dir": 3, "filename": 4}
    with pytest.raises(SerialisationError) as raised:
        WorldSerializer.from_config(block)
    message = str(raised.value)
    assert message.startswith("Serialisation config has 3 problem(s):")
    for key in ("config_file", "output_dir", "filename"):
        assert key in message
    assert len(raised.value.report.problems) == 3


def test_warnings_logged_at_info(tmp_path, caplog):
    schema_file = tmp_path / "schema.yaml"
    schema_file.write_text("population:\n  properties: []\nextras: {}\n")
    with caplog.at_level(logging.INFO, logger="world_serializer"):
        WorldSerializer.from_config({"config_file": str(schema_file)})
    warnings = [record for record in caplog.records if "extras" in record.getMessage()]
    assert warnings and all(record.levelno == logging.INFO for record in warnings)


def test_disabled_gives_settings_without_schema():
    settings = WorldSerializer.from_config({"enabled": False, "config_file": 7})
    assert settings.enabled is False
    assert settings.schema is None


def test_defaults_applied():
    settings = WorldSerializer.from_config({"config_file": SCHEMA_2021})
    assert settings.enabled is True
    assert settings.schema_path == SCHEMA_2021
    assert settings.schema.population_properties
    assert settings.output_dir == "."
    assert settings.filename == "world_state.h5"
