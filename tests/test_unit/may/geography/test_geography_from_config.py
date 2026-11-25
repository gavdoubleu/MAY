import logging

import pytest

from may.geography.geography import Geography, GeographyError
from may.utils import path_resolver

LEVELS = ["SGU", "MGU", "LGU"]


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
def geography_dir(tmp_path):
    """Two LGUs, one MGU each, two SGUs per MGU; coordinates for SGUs only."""
    directory = tmp_path / "geography"
    directory.mkdir()
    hierarchy_rows = ["SGU,MGU,LGU"]
    coord_rows = ["SGU,latitude,longitude"]
    for lgu in ["L_keep", "L_drop"]:
        for sgu_index in range(2):
            sgu = f"{lgu}_S{sgu_index}"
            hierarchy_rows.append(f"{sgu},{lgu}_M0,{lgu}")
            coord_rows.append(f"{sgu},54.{sgu_index},-1.5")
    (directory / "hierarchy.csv").write_text("\n".join(hierarchy_rows) + "\n")
    (directory / "coord_sgu.csv").write_text("\n".join(coord_rows) + "\n")
    return directory


def block_for(geography_dir, **overrides):
    block = {
        "data_dir": str(geography_dir),
        "levels": LEVELS,
        "hierarchy_file": "hierarchy.csv",
        "coord_files": {"SGU": "coord_sgu.csv"},
    }
    block.update(overrides)
    return block


def test_problems_raise_geography_error_with_every_line(working_dir, geography_dir):
    block = block_for(geography_dir, coord_files={"BOGUS": "x.csv"})
    del block["levels"]
    del block["hierarchy_file"]
    with pytest.raises(GeographyError) as raised:
        Geography.from_config(block)
    message = str(raised.value)
    assert message.startswith("Geography config has 3 problem(s):")
    assert "levels" in message
    assert "hierarchy_file" in message
    assert "coord_files.BOGUS: not found" in message
    assert len(raised.value.report.problems) == 3


def test_warnings_are_logged_at_info(working_dir, geography_dir, caplog):
    block = block_for(geography_dir, filter={"codes": ["L_keep"]})
    with caplog.at_level(logging.INFO, logger="geography"):
        Geography.from_config(block)
    warning_records = [
        record for record in caplog.records if "filter.level" in record.getMessage()
    ]
    assert [record.levelno for record in warning_records] == [logging.INFO]
    assert "filter ignored" in warning_records[0].getMessage()


def test_builds_and_loads_filtered_units(working_dir, geography_dir):
    block = block_for(geography_dir, filter={"level": "LGU", "codes": ["L_keep"]})
    geography = Geography.from_config(block)
    assert geography.levels == LEVELS
    assert list(geography.get_units_by_level("LGU")) == ["L_keep"]
    assert len(geography.get_units_by_level("SGU")) == 2
    sgu = next(iter(geography.get_units_by_level("SGU").values()))
    assert sgu.coordinates is not None


def test_filter_file_in_data_dir_is_used(working_dir, geography_dir):
    (geography_dir / "keep.txt").write_text("# kept LGUs\nL_keep\n")
    block = block_for(geography_dir, filter={"level": "LGU", "file": "keep.txt"})
    geography = Geography.from_config(block)
    assert list(geography.get_units_by_level("LGU")) == ["L_keep"]


@pytest.fixture
def numeric_geography_dir(tmp_path):
    """FIPS-style hierarchy: pandas reads every level column as int64."""
    directory = tmp_path / "numeric_geography"
    directory.mkdir()
    hierarchy_rows = ["SGU,MGU,LGU"]
    for county in [6037, 6059]:
        for tract_index in range(2):
            hierarchy_rows.append(f"{county}{tract_index:02d},{county}9,{county}")
    (directory / "hierarchy.csv").write_text("\n".join(hierarchy_rows) + "\n")
    return directory


def numeric_block_for(numeric_geography_dir, **overrides):
    return block_for(numeric_geography_dir, coord_files={}, **overrides)


def test_numeric_inline_codes_match_numeric_units(working_dir, numeric_geography_dir):
    block = numeric_block_for(
        numeric_geography_dir, filter={"level": "LGU", "codes": [6037]}
    )
    geography = Geography.from_config(block)
    assert list(geography.get_units_by_level("LGU")) == [6037]
    assert len(geography.get_units_by_level("SGU")) == 2


def test_numeric_file_codes_match_numeric_units(working_dir, numeric_geography_dir):
    (numeric_geography_dir / "keep.txt").write_text("6037\n")
    block = numeric_block_for(
        numeric_geography_dir, filter={"level": "LGU", "file": "keep.txt"}
    )
    geography = Geography.from_config(block)
    assert list(geography.get_units_by_level("LGU")) == [6037]


def test_numeric_codes_match_a_column_read_as_float(working_dir, numeric_geography_dir):
    """A row blank at a level makes pandas read that column as float (6037.0); it is
    dropped, but the column stays float."""
    hierarchy = numeric_geography_dir / "hierarchy.csv"
    hierarchy.write_text(hierarchy.read_text() + "999901,99999,\n")
    block = numeric_block_for(
        numeric_geography_dir, filter={"level": "LGU", "codes": [6037]}
    )
    geography = Geography.from_config(block)
    assert len(geography.get_units_by_level("LGU")) == 1
    assert len(geography.get_units_by_level("SGU")) == 2


def test_a_missing_hierarchy_file_raises_geography_error(working_dir, geography_dir):
    block = block_for(geography_dir, hierarchy_file="absent.csv")
    with pytest.raises(GeographyError, match="absent.csv"):
        Geography.from_config(block)


def test_filter_matching_no_units_raises(working_dir, geography_dir):
    block = block_for(geography_dir, filter={"level": "LGU", "codes": ["Nowhere"]})
    with pytest.raises(GeographyError, match="matched no LGU units") as raised:
        Geography.from_config(block)
    assert "Nowhere" in str(raised.value)


def test_unmatched_filter_codes_are_warned(working_dir, geography_dir, caplog):
    block = block_for(
        geography_dir, filter={"level": "LGU", "codes": ["L_keep", "Nowhere"]}
    )
    with caplog.at_level(logging.WARNING, logger="geography"):
        geography = Geography.from_config(block)
    assert list(geography.get_units_by_level("LGU")) == ["L_keep"]
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "Nowhere" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "1 of 2" in warnings[0]
