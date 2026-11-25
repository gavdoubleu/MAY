import glob
import logging
import pandas as pd
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .venue import Venue
from may.geography.geography import check_geo_level
from may.preflight.config_reader import FAILED, ConfigReader
from may.preflight.config_stage import ConfigError, ConfigStage
from may.utils import path_resolver as pr

logger = logging.getLogger("venuemanager")

DEFAULT_DATA_DIR = "data/venues"
DEFAULT_CONFIG_FILE = "venues_config.yaml"


def _parse_data_dir(reader: ConfigReader):
    """Resolved data dir, default when absent; `FAILED` if not a path or empty."""
    data_dir = reader.path("data_dir", default=DEFAULT_DATA_DIR)
    if data_dir == "":
        reader.problem("must not be empty; give a path, or remove the key to use the default", at="data_dir")
        return FAILED
    return data_dir


def _warn_unpaired_filter(type_reader: ConfigReader) -> None:
    if type_reader.value("filter_column") and not type_reader.value("filter_values"):
        type_reader.warning("not set; filter_column ignored", at="filter_values")
    if type_reader.value("filter_values") and not type_reader.value("filter_column"):
        type_reader.warning("not set; filter_values ignored", at="filter_column")


def _parse_batch_pattern(type_reader: ConfigReader, data_dir):
    """The one `{unit}` file pattern; it must match a file for some unit. Units are
    unknown before the geography loads, so `{unit}` matches anything."""
    raw = type_reader.value("filename")
    if isinstance(raw, list):
        type_reader.problem(f"batch_mode takes one pattern, not a list; got {raw!r}", at="filename")
        return FAILED
    pattern = type_reader.name("filename", give="a file pattern containing {unit}")
    if pattern is FAILED:
        return FAILED
    if "{unit}" not in pattern:
        type_reader.problem(f"batch_mode pattern must contain '{{unit}}'; got {pattern!r}", at="filename")
        return FAILED
    if data_dir is FAILED:
        return pattern
    unit_glob = "*".join(glob.escape(part) for part in pattern.split("{unit}"))
    if not glob.glob(os.path.join(glob.escape(data_dir), unit_glob)):
        type_reader.problem(
            f"pattern {pattern!r} matched no files in {os.path.abspath(data_dir)}", at="filename"
        )
        return FAILED
    return pattern


def _parse_venue_type(type_reader: ConfigReader, data_dir, context):
    """`(type config, file paths)`: the written mapping with a batch type's pattern and level
    as text; paths of an enabled non-batch type's files, else None."""
    type_config = dict(type_reader.value())
    enabled = type_reader.true_or_false("enabled", default=True)
    batch_mode = type_reader.true_or_false("batch_mode", default=False)
    _warn_unpaired_filter(type_reader)
    if enabled is not True or batch_mode is FAILED:
        # Disabled types are never loaded; failed flags leave their needs unknown.
        return type_config, None
    if batch_mode is False:
        if "level" in type_reader.keys():
            type_reader.warning("unused; batch_mode is not true", at="level")
        paths = type_reader.data_file("filename", data_dir, give="a path or a list of paths")
        return type_config, None if paths is FAILED else paths
    type_config["filename"] = _parse_batch_pattern(type_reader, data_dir)
    level = type_reader.name("level", give="the geo level {unit} names")
    if level is not FAILED:
        level = check_geo_level(type_reader, "level", context)
    type_config["level"] = level
    return type_config, None


@dataclass(frozen=True)
class VenueConfig:
    """Parsed `venues` block and its YAML file; read by `VenueManager.parse`.

    `venue_types` maps each type name (as text) to its written mapping; a batch type's
    `filename` and `level` are text. Fields of a block with problems may be `FAILED`."""

    data_dir: str
    config_path: str
    venue_types: dict
    filter_by_geography: Optional[bool]
    # Absolute paths of each enabled non-batch type's files; batch types have none.
    venue_paths: Dict[str, List[str]] = field(default_factory=dict)


class VenueError(Exception):
    """Raised when venue configuration or data is missing/invalid. Mirrors
    PopulationError: the engine works on complete data or fails loudly."""


class VenueConfigError(ConfigError, VenueError):
    """The `venues` config block cannot be used; `.report` holds the findings."""


class VenueManager(ConfigStage):
    """
    Manages venues and their relationship to geographical units.
    """

    def __init__(self, geography, data_dir=None, filter_by_geography=True):
        self.geography = geography  # Reference to Geography object
        self.data_dir = data_dir
        self.venues_by_type_and_id = defaultdict(
            dict
        )  # Venues by type and ID: {type: {id: Venue}}

        self.filter_by_geography = (
            filter_by_geography  # Only load venues in loaded geo units
        )

        # Per-type counter used only for generating human-readable venue names
        self._venue_number_by_type = defaultdict(int)  # {venue_type: next_number}

        # Get set of loaded geographical unit names for filtering
        self._loaded_geo_units = set(self.geography.get_all_units().keys())

        # Store full venue type configurations from YAML
        self.venue_configs = {}  # {venue_type: full_config_dict}

        # Capacity configurations per venue type, lazily populated by
        # allocation steps (residence venue_allocator) at runtime.
        self.capacity_configs = {}  # {venue_type: capacity_config_dict}

        # Lossless name index: {type: {name: [id, ...]}}. Preserves all ids for
        # duplicate-named venues; get_venue / get_venue_by_type_and_name are lossy
        # (first-match) wrappers over this structure for backwards compatibility.
        self.type_and_name_to_id: defaultdict = defaultdict(lambda: defaultdict(list))

    def _get_venue_number(self, venue_type: str) -> int:
        """Return next sequential number for naming venues of this type."""
        number = self._venue_number_by_type[venue_type]
        self._venue_number_by_type[venue_type] += 1
        return number

    def add_venue(self, venue):
        """Adds a venue to the VenueManager in the appropriate place and relates it with the geography object"""
        self.venues_by_type_and_id[venue.type][venue.id] = venue
        # Add venue to its geographical unit
        venue.geographical_unit.add_venue(venue)
        self.type_and_name_to_id[venue.type][venue.name].append(venue.id)

    def create_venue(self, venue_type, geo_unit, properties=None):
        """
        Create a venue and add it to the manager.
        ID is auto-generated per venue type.

        Args:
            venue_type: Type of venue (e.g., "household", "hospital", "school")
            geo_unit: GeographicalUnit where venue is located
            properties: Venue-specific properties dict

        Returns:
            Venue object
        """
        venue_number = self._get_venue_number(venue_type)

        # Prepare properties, adding is_residence from config if available
        venue_properties = properties or {}

        # Add properties from venue_configs if available
        if venue_type in self.venue_configs:
            config = self.venue_configs[venue_type]
            venue_properties["is_residence"] = config.get("is_residence", False)

            # Explicitly copy subset configuration
            if "subset_categories" in config:
                venue_properties["subset_categories"] = config["subset_categories"]
            if "subset_key" in config:
                venue_properties["subset_key"] = config["subset_key"]

        venue = Venue(
            name=f"{venue_type}_{venue_number}",
            venue_type=venue_type,
            geographical_unit=geo_unit,
            properties=venue_properties,
        )

        # Add to manager
        self.add_venue(venue)

        return venue

    def remove_venue(self, venue):
        """
        Remove a venue from the VenueManager and its geographical_unit.

        Mirror of add_venue. The venue must be a leaf (no children) and must
        have no remaining subsets, so call migrate_subsets_to first if needed.

        Args:
            venue: Venue object to remove.

        Raises:
            ValueError: if venue has children, or still has subsets.
        """
        if venue.children:
            raise ValueError(
                f"Cannot remove {venue}: has child venues. Remove each child first."
            )
        if venue.subsets:
            raise ValueError(
                f"Cannot remove {venue}: has subsets. Call migrate_subsets_to first."
            )

        if venue.parent is not None:
            venue.parent.children.remove(venue)

        self.venues_by_type_and_id[venue.type].pop(venue.id, None)

        name_ids = self.type_and_name_to_id[venue.type].get(venue.name)
        if name_ids is not None:
            name_ids.remove(venue.id)
            if not name_ids:
                del self.type_and_name_to_id[venue.type][venue.name]

        venue.geographical_unit.venues.discard(venue)

    def create_child_venue(
        self, parent_venue, child_venue_type, properties=None, geo_unit=None
    ):
        """
        Create a child venue and attach it to a parent venue.
        The child inherits the geographical_unit from the parent if not specified.

        Args:
            parent_venue: Parent Venue object
            child_venue_type: Type of child venue (e.g., "classroom", "office")
            properties: Venue-specific properties dict
            geo_unit: Optional GeographicalUnit (defaults to parent's geo_unit)

        Returns:
            Child Venue object

        Example:
            >>> school = venue_manager.create_venue("school", some_sgu)
            >>> classroom = venue_manager.create_child_venue(school, "classroom",
            ...                                               properties={'grade': 1, 'capacity': 25})
        """
        # Use parent's geographical_unit if not specified
        if geo_unit is None:
            geo_unit = parent_venue.geographical_unit

        # Create the child venue
        child = self.create_venue(
            venue_type=child_venue_type, geo_unit=geo_unit, properties=properties
        )

        # Establish parent-child relationship
        parent_venue.add_child_venue(child)

        return child

    def create_venue_with_children(
        self, parent_type, geo_unit, children_spec, properties=None
    ):
        """
        Create a parent venue with multiple child venues in one call.

        Args:
            parent_type: Type of parent venue (e.g., "school", "company")
            geo_unit: GeographicalUnit where parent venue is located
            children_spec: List of dicts specifying children, each with:
                - 'type': Child venue type (required)
                - 'count': Number of children to create (default: 1)
                - 'properties': Properties dict for the child (optional)
            properties: Properties for the parent venue

        Returns:
            Tuple of (parent_venue, list_of_child_venues)

        Example:
            >>> school, classrooms = venue_manager.create_venue_with_children(
            ...     parent_type="school",
            ...     geo_unit=some_sgu,
            ...     children_spec=[
            ...         {'type': 'classroom', 'count': 20, 'properties': {'capacity': 25}},
            ...         {'type': 'gym', 'count': 1, 'properties': {'capacity': 100}}
            ...     ],
            ...     properties={'capacity': 500}
            ... )
        """
        # Create parent venue
        parent = self.create_venue(
            venue_type=parent_type, geo_unit=geo_unit, properties=properties
        )

        # Create child venues
        children = []
        for spec in children_spec:
            child_type = spec["type"]
            count = spec.get("count", 1)
            child_properties = spec.get("properties", {})

            for i in range(count):
                # Add index to properties if creating multiple of same type
                props = child_properties.copy()
                if count > 1:
                    props["index"] = i

                child = self.create_child_venue(
                    parent_venue=parent, child_venue_type=child_type, properties=props
                )
                children.append(child)

        return parent, children

    def load_venue_type_from_df(
        self, venue_type, venue_df, filter_column=None, filter_values=None
    ):
        """Creates venues from a given dataframe"""
        if filter_column and filter_values:
            original_count = len(venue_df)
            # Ensure filtering works regardless of type (strip and stringify, and uppercase for consistency)
            target_values = [str(v).strip().upper() for v in filter_values]
            venue_df = venue_df[
                venue_df[filter_column]
                .astype(str)
                .str.strip()
                .str.upper()
                .isin(target_values)
            ]
            logger.info(
                f"Filtered {venue_type} venues by {filter_column}: {len(venue_df)} rows kept (from {original_count})"
            )

        # Required columns - 'geo_unit' or any level label defined in geography
        geo_levels = set(self.geography.levels)
        geo_cols = {"geo_unit"}.union(geo_levels)
        actual_geo_col = next(
            (col for col in venue_df.columns if col in geo_cols), None
        )

        if actual_geo_col is None:
            raise ValueError(
                f"Missing required geographical column ('geo_unit' or one of {sorted(geo_levels)}) in file for {venue_type}"
            )

        # Optional coordinate columns (check both lowercase and capitalized)
        lat_col = None
        lon_col = None

        # Check for latitude column (case-insensitive)
        for col in venue_df.columns:
            if col.lower() == "latitude":
                lat_col = col
            elif col.lower() == "longitude":
                lon_col = col

        has_coords = lat_col is not None and lon_col is not None

        # Detect a 'name' column (case-insensitive), treating it as the venue name only
        # if it actually exists. Otherwise the venue keeps its auto-generated name.
        name_col = next(
            (col for col in venue_df.columns if col.lower() == "name"), None
        )

        # Get additional property columns
        reserved_cols = {"name", "geo_unit", "latitude", "longitude"}.union(geo_cols)
        property_cols = [
            col
            for col in venue_df.columns
            if col.lower() not in reserved_cols and col not in reserved_cols
        ]

        # Filter DataFrame upfront if geography filtering is enabled
        venues_skipped = 0
        if self.filter_by_geography and actual_geo_col in venue_df.columns:
            original_count = len(venue_df)
            venue_df = venue_df[venue_df[actual_geo_col].isin(self._loaded_geo_units)]
            venues_skipped = original_count - len(venue_df)
            logger.info(
                f"Pre-filtered {venue_type} venues: {len(venue_df)} venues in loaded geography ({venues_skipped} filtered out using {actual_geo_col})"
            )

        # Create venues
        venues_created = 0
        for row in venue_df.itertuples():
            # Only treat 'name' as a CSV-supplied venue name when the column truly exists
            # AND the value is non-null. Otherwise the venue keeps its auto-generated
            # `{venue_type}_{id}` name.
            csv_name = None
            if name_col is not None:
                raw = getattr(row, name_col, None)
                if raw is not None and pd.notna(raw):
                    csv_name = str(raw)

            geo_unit = None
            if actual_geo_col:
                geo_unit_name = getattr(row, actual_geo_col)
                geo_unit = self.geography.get_unit(geo_unit_name)

            if not geo_unit:
                logger.warning(
                    f"Geographical unit not found for {venue_type} venue "
                    f"'{csv_name if csv_name else f'<row {row.Index}>'}'. Skipping."
                )
                continue

            # Get coordinates if provided
            coordinates = None
            if has_coords:
                lat_val = getattr(row, lat_col, None)
                lon_val = getattr(row, lon_col, None)
                if (
                    lat_val is not None
                    and lon_val is not None
                    and pd.notna(lat_val)
                    and pd.notna(lon_val)
                ):
                    coordinates = (lat_val, lon_val)

            # Add additional properties
            properties = {}
            for prop_col in property_cols:
                prop_val = getattr(row, prop_col, None)
                if prop_val is not None and pd.notna(prop_val):
                    properties[prop_col] = prop_val

            # Create venue (ID auto-generated per type)
            venue = self.create_venue(
                venue_type=venue_type, geo_unit=geo_unit, properties=properties
            )
            venues_created += 1

            # Override name if the CSV provided one. Update type_and_name_to_id
            # to remove the auto-generated name entry and add the csv name.
            if csv_name is not None:
                auto_name = venue.name
                self.type_and_name_to_id[venue.type][auto_name].remove(venue.id)
                if not self.type_and_name_to_id[venue.type][auto_name]:
                    del self.type_and_name_to_id[venue.type][auto_name]
                venue.name = csv_name
                self.type_and_name_to_id[venue.type][csv_name].append(venue.id)

            # Set coordinates if available
            if coordinates:
                venue.coordinates = coordinates

        logger.info(f"Created {venues_created} {venue_type} venues")

    display_name = "Venue"
    config_error = VenueConfigError
    config_key = "venues"

    @classmethod
    def parse(cls, reader: ConfigReader, context) -> VenueConfig:
        """Read a `venues` block and its YAML file; batch `level`s are checked against
        `context[Geography]`."""
        data_dir = _parse_data_dir(reader)
        file_reader = reader.config_file("config_file", data_dir=data_dir, default=DEFAULT_CONFIG_FILE)
        type_readers = file_reader.entries("venue_types", give="a mapping of venue types")
        venue_types = venue_paths = FAILED
        if type_readers is not FAILED:
            venue_types, venue_paths = {}, {}
            for venue_type, type_reader in type_readers.items():
                type_config, paths = _parse_venue_type(type_reader, data_dir, context)
                venue_types[venue_type] = type_config
                if paths is not None:
                    venue_paths[venue_type] = paths
        settings_reader = file_reader.section("settings", default={})
        return VenueConfig(
            data_dir=data_dir,
            config_path=file_reader.found_path,
            venue_types=venue_types,
            filter_by_geography=settings_reader.true_or_false("filter_by_geography", default=None),
            venue_paths=venue_paths,
        )

    @classmethod
    def build(cls, parsed: VenueConfig, *, geography) -> "VenueManager":
        """Load the venues the parsed block names onto the live `geography`."""
        logger.info(f"Loading venue configuration from {parsed.config_path}")
        settings = {}
        if parsed.filter_by_geography is not None:
            settings["filter_by_geography"] = parsed.filter_by_geography
        venue_manager = cls(geography, data_dir=parsed.data_dir, **settings)
        venue_manager._load(parsed)
        return venue_manager

    def load_venue_type_from_csv(
        self, venue_type, filename, filter_column=None, filter_values=None, dtype=None
    ):
        """
        Load venues of a specific type from CSV (relative paths join data_dir;
        absolute paths are used as given).
        ``filename`` may be a single file or a list of files stacked into one
        table; all files must share the same columns.

        Expected columns:
        - name (optional): Name of the venue
        - a geographical column ('geo_unit' or any configured level label)
        - latitude / longitude (optional): coordinates
        - All other columns become properties specific to this venue type

        A missing file is a hard error (VenueError), so callers that tolerate
        absent files (e.g. batch mode) must check existence before calling.

        ``dtype`` maps a column to the type it is read as in every file. Pandas
        infers per file, so a column whose values happen to be numeric in one
        file and mixed in another arrives with two types and fails the stacking
        check; naming the type here settles it for the whole stack.
        """
        from may.utils.stacked_input import load_stacked_csv

        if self.data_dir is None:
            raise VenueError(
                f"Cannot load venue type '{venue_type}' from CSV: VenueManager "
                "has no data_dir; build it with VenueManager.from_config."
            )
        try:
            venue_paths = pr.data_file_paths(
                filename, self.data_dir, f"venue type '{venue_type}'"
            )
            read_kwargs = {"dtype": dtype} if dtype else {}
            venue_df = load_stacked_csv(
                venue_paths, label=f"{venue_type} venues", **read_kwargs
            )
        except Exception as e:
            raise VenueError(str(e)) from e
        logger.info(f"Loading {venue_type} venues from {venue_paths}")

        self.load_venue_type_from_df(
            venue_type,
            venue_df,
            filter_column=filter_column,
            filter_values=filter_values,
        )

    def _load(self, venue_config: VenueConfig) -> None:
        """Load every enabled venue type of a parsed config; records all type configs."""
        if not venue_config.venue_types:
            logger.info("venue_types is empty; loading no venues.")
            return

        enabled_types = []
        disabled_types = []
        for venue_type, type_config in venue_config.venue_types.items():
            self.venue_configs[venue_type] = type_config
            if not type_config.get("enabled", True):
                disabled_types.append(venue_type)
                continue
            enabled_types.append(venue_type)

        if disabled_types:
            logger.info(f"Skipping disabled venue types: {disabled_types}")
        logger.info(f"Loading {len(enabled_types)} venue types from YAML config")

        for venue_type in enabled_types:
            type_config = self.venue_configs[venue_type]
            filename = type_config["filename"]
            load_options = {
                "filter_column": type_config.get("filter_column"),
                "filter_values": type_config.get("filter_values"),
                "dtype": type_config.get("dtype"),
            }
            if type_config.get("batch_mode", False):
                self._load_batch_type(
                    venue_type, filename, type_config["level"], load_options
                )
            else:
                self.load_venue_type_from_csv(
                    venue_type, venue_config.venue_paths[venue_type], **load_options
                )

        self._log_total_created()
        self._log_summary()

    def _load_batch_type(self, venue_type, filename, level, load_options) -> None:
        """One file per unit of `level`, named by substituting {unit} into the
        filename. Absent per-unit files are skipped (partial geographies are
        routine), but zero matches is a hard error."""
        units = self.geography.get_units_by_level(level)
        matched = 0
        for unit_name in units.keys():
            unit_filename = filename.replace("{unit}", unit_name)
            if not os.path.exists(os.path.join(self.data_dir, unit_filename)):
                continue
            self.load_venue_type_from_csv(venue_type, unit_filename, **load_options)
            matched += 1
        if matched == 0:
            raise VenueError(
                f"Batch venue type '{venue_type}' matched no files in "
                f"{self.data_dir} (pattern {filename!r})."
            )

    def extend(self, other: "VenueManager"):
        """Adds all the venues from another instance of the VenueManager class into this instance.

        Created so that if multiple VenueManager child classes are made (e.g. to change the specifics of how they load venues)
        it is easy to combine them into one single object at the end.

        Args:
          other (VenueManager): another instance of the VenueManager class.

        """
        # Should add something to check that self.geography and other.geography are equal.
        for venue_type, id_dict in other.venues_by_type_and_id.items():
            self.venues_by_type_and_id[venue_type].update(id_dict)

        # Advance naming counters past imported venues to avoid duplicate names
        for venue_type, other_number in other._venue_number_by_type.items():
            if other_number > self._venue_number_by_type[venue_type]:
                self._venue_number_by_type[venue_type] = other_number

        for venue_type, name_dict in other.type_and_name_to_id.items():
            for name, ids in name_dict.items():
                self.type_and_name_to_id[venue_type][name].extend(ids)

    def get_venue(self, name):
        """Lossy: returns the first venue found with this name across all types."""
        for venue_type, name_dict in self.type_and_name_to_id.items():
            if name in name_dict:
                return self.get_venue_by_type_and_id(venue_type, name_dict[name][0])
        return None

    def get_venue_by_type_and_name(self, venue_type, name):
        """Lossy: returns the first registered venue of this type with this name."""
        ids = self.type_and_name_to_id.get(venue_type, {}).get(name)
        return self.get_venue_by_type_and_id(venue_type, ids[0]) if ids else None

    def get_venue_by_type_and_id(self, venue_type, venue_id):
        """
        Get a venue by its type and ID.

        Args:
            venue_type: Type of venue (e.g., "household", "hospital")
            venue_id: ID within that venue type

        Returns:
            Venue object or None if not found
        """
        return self.venues_by_type_and_id.get(venue_type, {}).get(venue_id)

    def get_venues_by_type(self, venue_type):
        """Get all venues of a specific type as a list."""
        return list(self.venues_by_type_and_id.get(venue_type, {}).values())

    def get_all_venues_list(self):
        """Get all venues as a flat list from venues_by_type_and_id (authoritative source)."""
        all_venues = []
        for id_dict in self.venues_by_type_and_id.values():
            all_venues.extend(id_dict.values())
        return all_venues

    def get_venue_types(self):
        """Get list of all venue types"""
        return list(self.venues_by_type_and_id.keys())

    def get_capacity_config(self, venue_type):
        """
        Get capacity configuration for a venue type.

        Args:
            venue_type: Type of venue (e.g., "care_home")

        Returns:
            dict: Capacity configuration or None if not defined
        """
        return self.capacity_configs.get(venue_type)

    def get_residence_types(self):
        """
        Get all venue types marked as residences.

        Returns:
            List of residence type strings (e.g., ['household', 'care_home', ...])

        Example:
            >>> venue_manager.get_residence_types()
            ['household', 'care_home', 'student_dorms', 'boarding_school']
        """
        residence_types = []

        # Get all residence types from venue_configs
        for venue_type, config in self.venue_configs.items():
            if config.get("is_residence", False):
                residence_types.append(venue_type)

        return residence_types

    def is_residence_type(self, venue_type: str) -> bool:
        """
        Check if a venue type is a residence.

        Args:
            venue_type: Venue type string (e.g., 'household', 'school')

        Returns:
            True if this venue type is marked as a residence

        Example:
            >>> venue_manager.is_residence_type('household')
            True
            >>> venue_manager.is_residence_type('school')
            False
        """
        # Check config
        config = self.venue_configs.get(venue_type, {})
        return config.get("is_residence", False)

    def get_all_residences(self):
        """
        Get all residence venues across all residence types.

        Returns:
            List of all residence Venue objects

        Example:
            >>> residences = venue_manager.get_all_residences()
            >>> len(residences)
            15234  # All households, care homes, dorms, etc.
        """
        residences = []
        for venue_type in self.get_residence_types():
            residences.extend(self.get_venues_by_type(venue_type))
        return residences

    def _log_total_created(self):
        total = sum(len(vs) for vs in self.venues_by_type_and_id.values())
        logger.info(f"Total venues created: {total}")

    def _log_summary(self):
        """Log summary statistics about venues"""
        for venue_type in sorted(self.venues_by_type_and_id.keys()):
            count = len(self.venues_by_type_and_id[venue_type])
            logger.info(f"  {venue_type}: {count} venues")

    def __repr__(self):
        return f"<VenueManager: {sum(len(d) for d in self.venues_by_type_and_id.values())} venues, {len(self.venues_by_type_and_id)} types>"
