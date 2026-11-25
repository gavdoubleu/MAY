from __future__ import annotations

"""
Population manager for MAY.

Handles population generation and distribution across geographical units.
"""

import glob
import os
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .person import Person
from may.geography.geography import check_geo_level
from may.preflight.config_reader import FAILED, ConfigReader
from may.preflight.config_stage import ConfigError, ConfigStage
from may.utils import path_resolver as pr

logger = logging.getLogger("population")

DEFAULT_DATA_DIR = "data/population"
DEFAULT_TYPE = "matrix"
VALID_TYPES = ("matrix", "explicit", "explicit_batch")
DEFAULT_MALE_FILE = "demographics_male.csv"
DEFAULT_FEMALE_FILE = "demographics_female.csv"
BATCH_FILE_PATTERN = "*_pop.csv"

# Keys each type reads; any other of these set under that type is ignored.
_TYPE_KEYS = {
    "matrix": {"demographics_male_file", "demographics_female_file"},
    "explicit": {"filename", "column_mapping"},
    "explicit_batch": {"level", "column_mapping"},
}
_TYPE_SPECIFIC_KEYS = set().union(*_TYPE_KEYS.values())


class PopulationError(Exception):
    """Population data is missing, empty, or unloadable; raising this fails the
    run loudly. ``create_world`` is the only place that turns this into a
    process exit."""


class PopulationConfigError(ConfigError, PopulationError):
    """The `population` config block cannot be used; `.report` holds the findings."""


@dataclass(frozen=True)
class PopulationConfig:
    """Parsed `population` config block; read by `PopulationManager.parse`.

    `data_dir` is `${...}`-resolved. Data files (`filename`, `male_file`,
    `female_file`) are lists of absolute paths, checked to exist. Keys the chosen
    type ignores are `None` (`{}` for `column_mapping`). Fields of a block with
    problems may be `FAILED`."""

    data_dir: str
    type: str
    column_mapping: dict = field(default_factory=dict)
    filename: Optional[List[str]] = None
    male_file: Optional[List[str]] = None
    female_file: Optional[List[str]] = None
    level: Optional[str] = None


def _parse_data_dir(reader: ConfigReader):
    """Resolved data dir, default when absent; `FAILED` if not a path or empty."""
    data_dir = reader.path("data_dir", default=DEFAULT_DATA_DIR)
    if data_dir == "":
        reader.problem("must not be empty; give a path, or remove the key to use the default", at="data_dir")
        return FAILED
    return data_dir


def _check_batch_data_dir(reader: ConfigReader, data_dir) -> None:
    """`explicit_batch` reads `<unit>_pop.csv` per unit; the units are unknown before
    the geography loads, so at least one such file must exist."""
    if data_dir is FAILED:
        return
    if not os.path.isdir(data_dir):
        reader.problem(f"{os.path.abspath(data_dir)} not found", at="data_dir")
    elif not glob.glob(os.path.join(glob.escape(data_dir), BATCH_FILE_PATTERN)):
        reader.problem(
            f"no {BATCH_FILE_PATTERN} files in {os.path.abspath(data_dir)}; "
            "type 'explicit_batch' reads one per unit of `level`",
            at="data_dir",
        )


def _parse_column_mapping(reader: ConfigReader):
    """Field → column name; a bad column is a problem at its field, siblings still read."""
    mapping_reader = reader.section("column_mapping", default={})
    field_names = mapping_reader.keys()
    if field_names is FAILED:
        return FAILED
    column_mapping = {}
    for field_name in field_names:
        column_name = mapping_reader.name(field_name, give="a column name")
        if column_name is not FAILED:
            column_mapping[field_name] = column_name
    if len(column_mapping) != len(field_names):
        return FAILED
    return column_mapping


def _parse_explicit_filename(reader: ConfigReader, data_dir):
    """The one population file, as a one-item path list; a list is not accepted."""
    if isinstance(reader.value("filename"), list):
        reader.problem("must be a path; got a list", at="filename")
        return FAILED
    return reader.data_file("filename", data_dir, give="a path")


def _parse_batch_level(reader: ConfigReader, context):
    """The required `level`, known to the geography once it parsed."""
    level = reader.name("level", give="a geo level name")
    if level is FAILED:
        return FAILED
    return check_geo_level(reader, "level", context)


def _warn_unused_keys(reader: ConfigReader, population_type: str) -> None:
    written_keys = set(reader.keys())
    for key in sorted(_TYPE_SPECIFIC_KEYS - _TYPE_KEYS[population_type]):
        if key in written_keys:
            reader.warning(f"ignored by population type {population_type!r}", at=key)


class PopulationManager(ConfigStage):
    """
    Manages population generation and distribution.

    This class loads demographic data and creates Person objects distributed
    across geographical units according to specified distributions.
    """

    def __init__(self, geography, data_dir):
        """
        Initialize the PopulationManager.

        Args:
            geography (Geography): Geography object containing geographical units
            data_dir (str): Directory containing population data files
        """
        self.geography = geography
        self.data_dir = data_dir
        self.people = []
        self.people_by_id = {}

        # Precise demographics: geo_unit -> age -> sex -> count
        self.precise_demographics = {}

    def __len__(self):
        return len(self.people)

    display_name = "Population"
    config_error = PopulationConfigError
    config_key = "population"

    @classmethod
    def parse(cls, reader: ConfigReader, context) -> PopulationConfig:
        """Read a `population` block; `explicit_batch` `level` is checked against
        `context[Geography]`."""
        if not reader.keys():
            reader.warning(
                f"no population settings; defaults in use: type {DEFAULT_TYPE!r}, "
                f"data_dir {DEFAULT_DATA_DIR!r}, files {DEFAULT_MALE_FILE!r} / "
                f"{DEFAULT_FEMALE_FILE!r}"
            )
        data_dir = _parse_data_dir(reader)
        population_type = reader.one_of("type", VALID_TYPES, default=DEFAULT_TYPE)
        if population_type is FAILED:
            return PopulationConfig(data_dir=data_dir, type=FAILED)
        _warn_unused_keys(reader, population_type)

        column_mapping = {}
        if population_type != "matrix":
            column_mapping = _parse_column_mapping(reader)

        filename = male_file = female_file = level = None
        if population_type == "explicit":
            filename = _parse_explicit_filename(reader, data_dir)
        elif population_type == "matrix":
            male_file = reader.data_file(
                "demographics_male_file", data_dir, default=DEFAULT_MALE_FILE
            )
            female_file = reader.data_file(
                "demographics_female_file", data_dir, default=DEFAULT_FEMALE_FILE
            )
        else:  # explicit_batch
            level = _parse_batch_level(reader, context)
            _check_batch_data_dir(reader, data_dir)

        return PopulationConfig(
            data_dir=data_dir,
            type=population_type,
            column_mapping=column_mapping,
            filename=filename,
            male_file=male_file,
            female_file=female_file,
            level=level,
        )

    @classmethod
    def build(cls, parsed: PopulationConfig, *, geography) -> "PopulationManager":
        """Load the population the parsed block names onto the live `geography`."""
        population = cls(geography=geography, data_dir=parsed.data_dir)
        if parsed.type == "explicit_batch":
            population.load_batch_explicit_from_csv(
                data_dir=parsed.data_dir, column_mapping=parsed.column_mapping, level=parsed.level
            )
        elif parsed.type == "explicit":
            population.load_explicit_from_csv(
                filename=parsed.filename, column_mapping=parsed.column_mapping
            )
        else:  # matrix
            population.load_demographics_from_csv(parsed.male_file, parsed.female_file)
            population.generate_population()
        return population

    @staticmethod
    def _create_nested_defaultdict():
        """
        Create a nested defaultdict for demographics storage.

        Defined as a named function so the object stays pickle-compatible.
        Returns a defaultdict(dict) for storing age -> sex -> count mappings.
        """
        return defaultdict(dict)

    def load_demographics_from_csv(
        self, male_file="demographics_male.csv", female_file="demographics_female.csv"
    ):
        """
        Load precise population demographics from matrix-style CSV files.

        Expected format (separate files for male/female):
            geo_unit,0,1,2,3,...,100
            E00004320,2,2,1,3,...,0
            E00004321,1,3,2,2,...,1
            ...

        Rows = geo units
        Columns = ages (1-year bins from 0 to 100)

        Args:
            male_file: Filename or list of filenames for male demographics
            female_file: Filename or list of filenames for female demographics
        """
        from may.utils.stacked_input import load_stacked_csv

        try:
            male_paths = pr.data_file_paths(
                male_file, self.data_dir, "population.demographics_male_file"
            )
            female_paths = pr.data_file_paths(
                female_file, self.data_dir, "population.demographics_female_file"
            )
        except pr.DataFileNotFoundError as error:
            raise PopulationError(f"Demographics {error}") from error

        # Get the smallest geographical level from the loaded geography
        # to filter demographics to only relevant geo units
        smallest_level = self.geography.levels[0]
        smallest_units_dict = self.geography.get_units_by_level(smallest_level)

        if not smallest_units_dict:
            raise PopulationError(
                f"No {smallest_level} units found in geography. Cannot load demographics."
            )

        # Create a set of geo unit names that exist in our geography for fast lookup
        valid_geo_units = set(smallest_units_dict.keys())
        logger.info(
            f"Filtering demographics to {len(valid_geo_units)} {smallest_level}s in loaded geography"
        )

        try:
            male_df = load_stacked_csv(
                male_paths, label="male demographics", key_column="geo_unit"
            )
            female_df = load_stacked_csv(
                female_paths, label="female demographics", key_column="geo_unit"
            )
        except Exception as e:
            raise PopulationError(str(e)) from e

        # Ignore index column if it exists
        for _df in [male_df, female_df]:
            if "index" in _df.columns:
                _df.drop(columns=["index"], inplace=True)

        # Every loaded geo unit must have a demographics row: a silent gap here
        # would build a plausible-looking world with people quietly missing.
        for sex_label, df in (("male", male_df), ("female", female_df)):
            missing = valid_geo_units - set(df["geo_unit"].astype(str))
            if missing:
                examples = sorted(missing)[:10]
                raise PopulationError(
                    f"{len(missing)} {smallest_level}(s) in the loaded geography "
                    f"have no {sex_label} demographics row, e.g. {examples}. "
                    f"The geography filter is the only way to scope a world; "
                    f"demographics must cover every loaded unit."
                )

        # Filter to only geo units in our geography BEFORE processing
        male_df = male_df[male_df["geo_unit"].isin(valid_geo_units)]
        female_df = female_df[female_df["geo_unit"].isin(valid_geo_units)]

        logger.info(
            f"Filtered to {len(male_df)} male geo units and {len(female_df)} female geo units"
        )

        # Load into nested dict structure: geo_unit -> age -> sex -> count
        # Named function keeps this pickle-compatible
        self.precise_demographics = defaultdict(self._create_nested_defaultdict)
        total_people = 0

        logger.info("Processing male demographics...")
        # Convert male dataframe to long format for efficient processing
        male_melted = male_df.melt(
            id_vars=["geo_unit"], var_name="age", value_name="count"
        )
        male_melted["age"] = male_melted["age"].astype(int)
        male_melted["count"] = male_melted["count"].fillna(0).astype(int)
        # Filter out zero counts for efficiency
        male_melted = male_melted[male_melted["count"] > 0]

        logger.info("Processing female demographics...")
        # Convert female dataframe to long format
        female_melted = female_df.melt(
            id_vars=["geo_unit"], var_name="age", value_name="count"
        )
        female_melted["age"] = female_melted["age"].astype(int)
        female_melted["count"] = female_melted["count"].fillna(0).astype(int)
        # Filter out zero counts for efficiency
        female_melted = female_melted[female_melted["count"] > 0]

        logger.info("Building demographic dictionary...")
        # Convert to numpy arrays for much faster iteration
        male_values = male_melted.values  # [[geo_unit, age, count], ...]
        female_values = female_melted.values

        # Build nested dictionary
        for row in male_values:
            geo_unit = str(row[0])
            age = int(row[1])
            count = int(row[2])

            self.precise_demographics[geo_unit][age]["male"] = count
            total_people += count

        for row in female_values:
            geo_unit = str(row[0])
            age = int(row[1])
            count = int(row[2])

            self.precise_demographics[geo_unit][age]["female"] = count
            total_people += count

        logger.info(
            f"Loaded precise demographics for {len(self.precise_demographics)} geographical units"
        )
        logger.info(f"Total people in demographics: {total_people:,}")

    def load_explicit_from_csv(self, filename, column_mapping: Dict[str, str]):
        """
        Load individual-level population data from one CSV file.

        `filename` is a path, or a one-item list of paths; relative to `data_dir`.
        """
        try:
            paths = pr.data_file_paths(filename, self.data_dir, "population.filename")
        except pr.DataFileNotFoundError as error:
            raise PopulationError(f"Explicit population {error}") from error
        if len(paths) != 1:
            raise PopulationError(f"Explicit population takes one file; got {filename!r}")
        (path,) = paths

        logger.info(f"Loading explicit population from {path}")
        df = pd.read_csv(path)

        # Reset ID counter for consistency (at the entry point)
        Person.reset_counter()

        self.load_explicit_from_df(df, column_mapping)

    def load_explicit_from_df(self, df: pd.DataFrame, column_mapping: Dict[str, str]):
        """
        Internal method to load population from a DataFrame.
        """
        target_to_csv = column_mapping

        # Identify geographical column
        # Priority: 1. mapped 'geo_unit', 2. literal 'geo_unit', 3. any configured level label
        geo_levels = set(self.geography.levels)
        geo_cols = {"geo_unit"}.union(geo_levels)

        mapped_geo_col = target_to_csv.get("geo_unit")
        actual_geo_col = None

        if mapped_geo_col in df.columns:
            actual_geo_col = mapped_geo_col
        else:
            actual_geo_col = next((col for col in df.columns if col in geo_cols), None)

        if actual_geo_col is None:
            raise ValueError(
                f"Missing required geographical column ('geo_unit' or one of {sorted(geo_levels)}) in population data"
            )

        people_count = 0

        for row in df.itertuples(index=False):
            row_dict = row._asdict()
            properties = {}
            age = 0
            sex = "unknown"

            # 1. Determine geographical unit
            geo_unit_name = row_dict.get(actual_geo_col)
            geo_unit = self.geography.get_unit(geo_unit_name) if geo_unit_name else None

            if not geo_unit:
                logger.warning(
                    f"No geographical unit found for person in row (col: {actual_geo_col}, val: {geo_unit_name}). Skipping."
                )
                continue

            # Extract known attributes
            for target, csv_col in target_to_csv.items():
                if csv_col not in row_dict:
                    continue

                val = row_dict[csv_col]
                if target == "age":
                    try:
                        age = int(float(val))
                    except (ValueError, TypeError):
                        age = 0
                elif target == "sex":
                    sex = str(val).lower().strip() if pd.notna(val) else "unknown"
                    # Normalize common sex strings
                    if sex in ["m", "1", "male"]:
                        sex = "male"
                    elif sex in ["f", "2", "female"]:
                        sex = "female"
                elif target == "geo_unit":
                    unit_name = str(val).strip()
                    found_unit = self.geography.get_unit(unit_name)
                    if found_unit:
                        geo_unit = found_unit
                else:
                    # Treat as a generic property
                    properties[target] = val

            # Add all other columns not in mapping to properties.
            # The geographical column drives `geographical_unit` and is kept out
            # of the property dict (parallel with VenueManager, which keeps its
            # geo column out of properties).
            mapped_csv_cols = set(target_to_csv.values())
            reserved_cols = mapped_csv_cols | {actual_geo_col}
            for col, val in row_dict.items():
                if col not in reserved_cols:
                    properties[col] = val

            # Create and add person
            person = Person(
                age=age, sex=sex, geographical_unit=geo_unit, properties=properties
            )
            self.add_person(person)

            if geo_unit:
                geo_unit.add_person(person)

            people_count += 1

        logger.info(f"Successfully loaded {people_count:,} people from explicit data.")

    def generate_population(self, **kwargs):
        """
        Generate population from precise demographics data.

        Creates exact number of people per age/sex/geo_unit as specified
        in the demographics file. People are created in age order globally
        (person 0 is the youngest person across all smallest geographical units).

        Assumes demographics are provided for the smallest geographical level
        (first level in the hierarchy, e.g., SGU, village, census block, etc.).

        Args:
          **kwargs:
            Arbitrary keyword arguments to be passed to the creation of Person.
            Supported keys include:
              * activities (list[str], optional): list of activity names for each Person.
              * properties (dict, optional): a dict of properties of the Person, e.g. 'ethnicity', 'compliance', 'taste'.
              * activity_map (DefaultDict[str,list[Subset]], optional):
                a dict mapping an activity (same string as in activities) to a list of potential Subsets the Person would
                join to fulfil that activity.

        """
        if not self.precise_demographics:
            raise PopulationError(
                "No demographics data loaded. Cannot generate population."
            )

        logger.info("Generating population from precise demographics...")
        Person.reset_counter()

        # Get the smallest geographical level (first in the hierarchy)
        smallest_level = self.geography.levels[0]
        smallest_units_dict = self.geography.get_units_by_level(smallest_level)

        if not smallest_units_dict:
            raise PopulationError(
                f"No {smallest_level} units found in geography. Cannot generate population."
            )

        # Collect all (age, sex, geo_unit, count) tuples and sort by age
        all_age_sex_geo = []

        for unit in smallest_units_dict.values():
            if unit.name not in self.precise_demographics:
                logger.debug(f"No demographic data for {unit.name}, skipping")
                continue

            age_sex_data = self.precise_demographics[unit.name]

            for age, sex_counts in age_sex_data.items():
                for sex, count in sex_counts.items():
                    all_age_sex_geo.append((age, sex, unit, count))

        # Sort by age first, then sex (for consistent ordering)
        all_age_sex_geo.sort(key=lambda x: (x[0], x[1]))

        # Now create people in age order across all smallest units
        total_people = 0
        geo_units_with_data = len(set(item[2] for item in all_age_sex_geo))

        for age, sex, unit, count in all_age_sex_geo:
            for _ in range(count):
                person = Person(age=age, sex=sex, geographical_unit=unit, **kwargs)
                self.add_person(person)
                # Add person to their geographical unit's people list
                unit.add_person(person)
                total_people += 1

        logger.info(
            f"Generated {total_people:,} people across {geo_units_with_data} {smallest_level}s"
        )
        if geo_units_with_data > 0:
            logger.info(
                f"Average: {total_people / geo_units_with_data:.1f} people per {smallest_level}"
            )

    def add_person(self, person: Person):
        self.people.append(person)
        self.people_by_id[person.id] = person

    def add_people(self, people: list[Person]):
        for person in people:
            self.add_person(person)

    def get_person(self, person_id):
        """
        Get a person by their ID.

        Args:
            person_id (int): ID of the person

        Returns:
            Person: The person object, or None if not found
        """
        return self.people_by_id.get(person_id)

    def get_all_people(self):
        """
        Get all people as a list.

        Returns:
            list: List of all Person objects
        """
        return self.people

    def get_people_by_age_range(self, min_age, max_age):
        """
        Get all people within an age range.

        Args:
            min_age (int): Minimum age (inclusive)
            max_age (int): Maximum age (inclusive)

        Returns:
            list: List of Person objects in age range
        """
        return [p for p in self.people if min_age <= p.age <= max_age]

    def get_people_by_sex(self, sex):
        """
        Get all people of a specific sex.

        Args:
            sex (str): Sex category

        Returns:
            list: List of Person objects
        """
        return [p for p in self.people if p.sex == sex]

    def get_people_by_activity(self, activity):
        """
        Get all people with a specific activity.

        Args:
            activity (str): Activity name

        Returns:
            list: List of Person objects with this activity
        """
        return [p for p in self.people if p.has_activity(activity)]

    def get_people_by_geo_unit(self, geo_unit_code):
        """
        Get all people in a specific geographical unit.

        Args:
            geo_unit_code (str): Name/code of the geographical unit

        Returns:
            list: List of Person objects in this geo_unit
        """
        unit = self.geography.get_unit(geo_unit_code)
        if unit is None:
            return []
        return unit.people if hasattr(unit, "people") else []

    def get_statistics(self):
        """
        Get basic statistics about the population.

        Returns:
            dict: Dictionary of statistics
        """
        if not self.people:
            return {}

        ages = [p.age for p in self.people]
        sexes = [p.sex for p in self.people]

        # Count sex categories
        sex_counts = {}
        for sex in sexes:
            sex_counts[sex] = sex_counts.get(sex, 0) + 1

        # Collect all activities
        all_activities = set()
        for p in self.people:
            all_activities.update(p.activities)

        activity_counts = {}
        for activity in all_activities:
            activity_counts[activity] = len(self.get_people_by_activity(activity))

        return {
            "total_population": len(self.people),
            "mean_age": np.mean(ages),
            "median_age": np.median(ages),
            "min_age": np.min(ages),
            "max_age": np.max(ages),
            "sex_distribution": sex_counts,
            "activity_counts": activity_counts,
        }

    def load_batch_explicit_from_csv(
        self, data_dir: str, column_mapping: Dict[str, str], level: str
    ):
        """
        Load individual-level population data from one `<unit>_pop.csv` per unit of `level`.
        """
        unit_names = set(self.geography.get_units_by_level(level).keys())

        # Reset ID counter once for the whole batch
        Person.reset_counter()

        logger.info(
            f"Starting batch explicit population load for {len(unit_names)} {level} units"
        )

        total_files = 0
        for unit_name in unit_names:
            filename = f"{unit_name}_pop.csv"
            path = os.path.join(data_dir, filename)
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            total_files += 1

            # Filter rows by geographical unit to only keep what is in our geography
            # Check for any valid geo level column ('geo_unit' or any configured level)
            geo_levels = set(self.geography.levels)
            geo_cols = {"geo_unit"}.union(geo_levels)
            actual_geo_col = next((col for col in df.columns if col in geo_cols), None)

            if actual_geo_col and actual_geo_col in df.columns:
                # We filter by whatever geographical units are currently loaded in the geography
                loaded_units = set(self.geography.get_all_units().keys())
                df = df[df[actual_geo_col].isin(loaded_units)]

            self.load_explicit_from_df(df, column_mapping)

        if total_files == 0:
            raise PopulationError(
                f"Batch explicit population: no population files in {data_dir} for "
                f"any of the {len(unit_names)} {level} units (expected <unit>_pop.csv); "
                "check `level` and `data_dir`."
            )
        logger.info(f"Batch load complete. Processed {total_files} files.")
