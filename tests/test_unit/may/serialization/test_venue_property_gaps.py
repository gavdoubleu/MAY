"""
Contract tests for the schema comparison that runs before the world is built.

Reading one header per venue file is enough to tell which configured properties
name a column the data supplies. Running it at the start of a build puts that
answer in the log while there is still time to act on it.
"""

import os

import pytest

from may.geography.venue_manager import VenueConfig
from may.serialization.venue_property_gaps import venue_property_gaps
from may.serialization.serialization_config import SerialisationConfig

VENUE_TYPES = {
    "company": {"filename": "companies.csv"},
    "school": {"filename": ["ew_schools.csv", "sct_schools.csv"]},
    "household": {"description": "built during the run, no file"},
}


@pytest.fixture
def venue_dir(tmp_path):
    (tmp_path / "companies.csv").write_text("geo_unit,industry_code,sizeband\nA,C,1\n")
    (tmp_path / "ew_schools.csv").write_text("geo_unit,NumberOfPupils\nA,10\n")
    (tmp_path / "sct_schools.csv").write_text("geo_unit,Gender\nB,Mixed\n")
    return str(tmp_path)


def schema(venue_type_properties):
    return SerialisationConfig(venue_type_properties=venue_type_properties)


def venues(data_dir):
    """Parsed config for VENUE_TYPES: the file-read types carry absolute paths."""
    venue_paths = {
        "company": [os.path.join(data_dir, "companies.csv")],
        "school": [
            os.path.join(data_dir, "ew_schools.csv"),
            os.path.join(data_dir, "sct_schools.csv"),
        ],
    }
    return VenueConfig(
        data_dir=data_dir, config_path="venues.yaml", venue_types=VENUE_TYPES,
        filter_by_geography=None, venue_paths=venue_paths,
    )


class TestVenuePropertyGaps:
    def test_a_column_the_files_carry_is_matched(self, venue_dir):
        gaps = venue_property_gaps(
            schema({"company": ["industry_code"]}), venues(venue_dir)
        )
        assert gaps == []

    def test_a_name_absent_from_the_headers_is_a_gap(self, venue_dir):
        gaps = venue_property_gaps(
            schema({"company": ["work_sector"]}), venues(venue_dir)
        )
        assert [(t, p) for t, p, _ in gaps] == [("company", "work_sector")]

    def test_gap_reports_the_columns_the_files_carry(self, venue_dir):
        gaps = venue_property_gaps(
            schema({"company": ["work_sector"]}), venues(venue_dir)
        )
        assert "industry_code" in gaps[0][2]

    def test_a_column_in_any_stacked_file_counts(self, venue_dir):
        """Nations arrive as separate files, and a column in any one of them
        counts as supplied."""
        gaps = venue_property_gaps(
            schema({"school": ["Gender", "NumberOfPupils"]}),
            venues(venue_dir),
        )
        assert gaps == []

    def test_types_the_run_assembles_are_left_to_the_export(self, venue_dir):
        """The run assembles household venues, so their properties appear
        once the build reaches them."""
        gaps = venue_property_gaps(
            schema({"household": ["capacity"]}), venues(venue_dir)
        )
        assert gaps == []

    def test_types_without_venue_paths_are_skipped(self, venue_dir):
        """Batch and disabled types have no venue_paths; the export checks them."""
        config = VenueConfig(
            data_dir=venue_dir, config_path="venues.yaml",
            venue_types={"company": {"filename": "companies.csv", "enabled": False}},
            filter_by_geography=None,
        )
        gaps = venue_property_gaps(schema({"company": ["anything"]}), config)
        assert gaps == []

