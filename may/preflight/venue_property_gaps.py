"""
Compare the serialisation schema with the venue data before the world is built.

The HDF5 export runs last, so a property the schema names and the venue files
omit surfaces once the rest of the run has been paid for. Reading one header per
venue file at the start costs a few milliseconds and reports the same mismatch
while there is still time to act on it.

The comparison covers venue types loaded from a file, where both sides of it
exist on disk. Types assembled during the run, and the person properties, take
their values from the build itself and are checked as the export writes them.
"""

import os

import pandas as pd

from may.geography.venue_manager import VenueConfig
from may.serialization.serialization_config import SerialisationConfig
from may.utils.stacked_input import as_path_list


def venue_property_gaps(schema: SerialisationConfig, venue_config: VenueConfig):
    """
    Configured venue properties whose name is missing from the venue files.

    Args:
        schema: Parsed serialisation schema.
        venue_config: Parsed venue config; filenames are relative to its `data_dir`.

    Returns:
        List of (venue_type, property, columns_available), one entry per
        property whose name is absent from the header of every file that
        supplies the type. columns_available holds the names those headers do
        carry, sorted.
    """
    gaps = []
    for venue_type, properties in schema.venue_type_properties.items():
        spec = venue_config.venue_types.get(venue_type)
        if not isinstance(spec, dict) or not spec.get("filename"):
            # The run assembles this type, so its properties appear once the
            # build reaches them.
            continue

        paths = [
            os.path.join(venue_config.data_dir, path)
            for path in as_path_list(spec["filename"], f"venue type '{venue_type}'")
        ]
        present = [path for path in paths if os.path.exists(path)]
        if not present:
            continue

        columns = set()
        for path in present:
            columns |= set(pd.read_csv(path, nrows=0).columns)

        for property_name in properties:
            if property_name not in columns:
                gaps.append((venue_type, property_name, sorted(columns)))
    return gaps
