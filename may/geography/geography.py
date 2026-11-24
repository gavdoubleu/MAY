import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from .geographical_unit import GeographicalUnit
from may.preflight.report import PreflightReport, not_list_message, not_mapping_message
from may.utils import path_resolver as pr
from may.utils.stacked_input import as_path_list, load_stacked_csv
from may.utils.yaml_loader import unreadable_file_message

logger = logging.getLogger("geography")

DEFAULT_DATA_DIR = "data/geography"


class GeographyError(Exception):
    """Raised when the `geography` config block is invalid. Mirrors HouseholdError."""


@dataclass(frozen=True)
class GeographyConfig:
    """Parsed `geography` config block; checked by `Geography.parse_config`.

    Paths are `${...}`-resolved. `filters` is `{"level", "codes"}`, or `None` to load all."""

    data_dir: str
    levels: list
    hierarchy_file: object
    coord_files: dict
    filters: Optional[dict]


def _resolve_spec(spec):
    """Apply ${...} templating to a path or each path in a list."""
    if isinstance(spec, list):
        return [pr.resolve(path) for path in spec]
    return pr.resolve(spec)


def _is_path(value) -> bool:
    return isinstance(value, str) and bool(value)


def _check_path_spec(report: PreflightReport, key: str, spec) -> None:
    """Problem unless `spec` is a path or a non-empty list of paths."""
    if _is_path(spec):
        return
    if isinstance(spec, list) and spec and all(_is_path(path) for path in spec):
        return
    report.problem(key, f"must be a path or non-empty list of paths; got {spec!r}")


def _parse_data_dir(block: dict, report: PreflightReport) -> Optional[str]:
    """Resolved data dir; a blank key falls back to the default. `None` if not a path."""
    data_dir = block.get("data_dir")
    if data_dir is not None and not isinstance(data_dir, str):
        report.problem("data_dir", f"must be a path; got {data_dir!r}")
        return None
    return pr.resolve(data_dir or DEFAULT_DATA_DIR)


def _parse_levels(block: dict, report: PreflightReport) -> Optional[list]:
    """The level labels, or `None` with a problem; checks needing levels then skip."""
    levels = block.get("levels")
    if not levels:
        report.problem("levels", "not set")
        return None
    if not isinstance(levels, list) or not all(_is_path(level) for level in levels):
        report.problem("levels", f"must be a list of level names; got {levels!r}")
        return None
    duplicates = sorted({level for level in levels if levels.count(level) > 1})
    if duplicates:
        report.problem("levels", f"duplicate level names: {duplicates}")
        return None
    return levels


def _parse_coord_files(block: dict, levels, report: PreflightReport) -> dict:
    coord_files = block.get("coord_files")
    if coord_files is None:
        return {}
    if not isinstance(coord_files, dict):
        report.problem("coord_files", not_mapping_message(coord_files))
        return {}
    for level, spec in coord_files.items():
        key = f"coord_files.{level}"
        if levels is not None and level not in levels:
            report.problem(key, f"not a configured level; levels are {levels}")
        _check_path_spec(report, key, spec)
    return coord_files


def _parse_filter(filter_block, levels, data_dir, report: PreflightReport) -> Optional[dict]:
    """`{"level", "codes"}`, or `None` to load all units (with any warning why)."""
    if filter_block is None:
        return None
    if not isinstance(filter_block, dict):
        report.problem("filter", not_mapping_message(filter_block))
        return None
    level = filter_block.get("level")
    if level is not None and levels is not None and level not in levels:
        report.problem(
            "filter.level", f"{level!r} is not a configured level; levels are {levels}"
        )
    inline_codes = filter_block.get("codes")
    if inline_codes is not None and not isinstance(inline_codes, list):
        report.problem("filter.codes", not_list_message(inline_codes))
        return None
    codes_file = filter_block.get("file")
    if codes_file and inline_codes:
        report.warning("filter.codes", "ignored; filter.file is set")
    if codes_file:
        codes = _read_filter_file(codes_file, data_dir, report)
    else:
        # YAML reads numeric codes as numbers; unit names are strings.
        codes = [str(code) for code in inline_codes or []]
    if not level:
        if codes_file or inline_codes:
            report.warning("filter.level", "not set; filter ignored, loads all units")
        return None
    if codes is None:
        return None
    if not codes:
        report.warning("filter", "level set but no codes; loads all units")
        return None
    return {"level": level, "codes": codes}


def _read_filter_file(codes_file, data_dir, report: PreflightReport) -> Optional[list]:
    """Sorted codes in the filter file, found in the working dir then `data_dir`."""
    if not isinstance(codes_file, str):
        report.problem("filter.file", f"must be a path; got {codes_file!r}")
        return None
    try:
        found_path = pr.find_config_file(codes_file, data_dir)
    except pr.ConfigFileNotFoundError as error:
        report.problem("filter.file", str(error))
        return None
    try:
        return sorted(Geography.load_codes_from_file(found_path))
    except (OSError, UnicodeDecodeError) as error:
        report.problem("filter.file", unreadable_file_message(found_path, error))
        return None


class Geography:
    """
    Main geography container. Loads and manages hierarchical geographical units.
    Generic implementation that works with any geography structure.
    """

    def __init__(
        self,
        data_dir="data/geography",
        levels=None,
        filters=None,
        hierarchy_file=None,
        coord_files=None,
    ):
        self.data_dir = data_dir
        self.units = {}  # All units by name: {name: GeographicalUnit}
        self.units_by_id = {}  # All units by ID: {id: GeographicalUnit}

        # Hierarchy levels (most granular to least granular). Required: callers
        # pass the scenario's configured labels.
        if not levels:
            raise ValueError(
                "Geography requires 'levels' — the ordered list of level labels "
                "(smallest to largest) from geography config. There is no default."
            )
        self.levels = levels

        # Separate lookups by level for efficiency
        self.units_by_level = {level: {} for level in self.levels}

        # Filters: dict with 'level' and 'names' keys
        # Example: {'level': 'MGU', 'codes': ['E02000173', 'E02000187']}
        # Note: the 'codes' key holds names
        self.filters = filters

        # Input files, named explicitly: a hierarchy file (or list of files
        # to stack) and a {level: file-or-list} coordinate mapping. Relative
        # paths resolve against data_dir. A level absent from coord_files
        # simply has no coordinates.
        self.hierarchy_file = hierarchy_file
        self.coord_files = coord_files or {}

        # ID counter for generating unique IDs
        self._next_id = 0

    def _generate_id(self):
        """
        Generate a unique sequential ID for a geographical unit.

        Returns:
            Unique integer ID
        """
        id = self._next_id
        self._next_id += 1
        return id

    @staticmethod
    def load_codes_from_file(file_path):
        """Set of codes in a text file, one per line; blank and `#` lines skipped.

        Pure: raises `OSError` / `UnicodeDecodeError` for the caller to report."""
        with open(file_path, "r", encoding="utf-8-sig") as codes_file:
            stripped_lines = (line.strip() for line in codes_file)
            return {line for line in stripped_lines if line and not line.startswith("#")}

    @staticmethod
    def parse_config(block: dict) -> Tuple[Optional[GeographyConfig], PreflightReport]:
        """Parse a `geography` config block. Config is `None` if any problem."""
        report = PreflightReport()
        data_dir = _parse_data_dir(block, report)
        levels = _parse_levels(block, report)
        hierarchy_file = block.get("hierarchy_file")
        if hierarchy_file is None or hierarchy_file == "":
            report.problem("hierarchy_file", "not set")
        else:
            _check_path_spec(report, "hierarchy_file", hierarchy_file)
        coord_files = _parse_coord_files(block, levels, report)
        load_all = block.get("load_all", False)
        if not isinstance(load_all, bool):
            report.problem("load_all", f"must be true or false; got {load_all!r}")
        # `load_all: true` ignores the filter, silently, so it is not checked.
        filters = None
        if load_all is not True:
            filters = _parse_filter(block.get("filter"), levels, data_dir, report)
        if report.has_problems:
            return None, report
        geography_config = GeographyConfig(
            data_dir=data_dir,
            levels=levels,
            hierarchy_file=_resolve_spec(hierarchy_file),
            coord_files={level: _resolve_spec(spec) for level, spec in coord_files.items()},
            filters=filters,
        )
        return geography_config, report

    @staticmethod
    def preflight(block: dict) -> PreflightReport:
        """Check a `geography` config block without loading geography data."""
        _, report = Geography.parse_config(block)
        return report

    @classmethod
    def from_config(cls, block: dict) -> "Geography":
        """Build and load from a `geography` block; `GeographyError` on any problem."""
        geography_config, report = cls.parse_config(block)
        report.raise_if_problems(
            GeographyError, f"Geography config has {len(report.problems)} problem(s):"
        )
        for finding in report.warnings:
            logger.info(f"Geography config warning: {finding.key}: {finding.message}")
        geography = cls(
            data_dir=geography_config.data_dir,
            levels=geography_config.levels,
            filters=geography_config.filters,
            hierarchy_file=geography_config.hierarchy_file,
            coord_files=geography_config.coord_files,
        )
        geography.load_from_csv()
        return geography

    def load_from_csv(self):
        """
        Load geography data from the configured hierarchy and coordinate CSVs.
        Each may be a single file or a list of files stacked into one table.
        """
        import os

        logger.info(f"Loading geography from {self.data_dir}")

        # 1. Load hierarchy file(s) (define parent-child relationships)
        if not self.hierarchy_file:
            raise ValueError(
                "geography requires 'hierarchy_file' (a path or list of paths) "
                "and optionally 'coord_files' (a {level: path-or-list} mapping); "
                "relative paths resolve against geography data_dir."
            )
        hierarchy_paths = [
            os.path.join(self.data_dir, p)
            for p in as_path_list(self.hierarchy_file, "geography.hierarchy_file")
        ]
        # Leaf units are the row keys: the same smallest-level unit defined in
        # two files (or twice in one) is a data error, not something to merge.
        hierarchy_df = load_stacked_csv(
            hierarchy_paths, label="geography hierarchy", key_column=self.levels[0]
        )

        logger.info(f"Loaded hierarchy with {len(hierarchy_df)} entries")

        # Verify that all configured levels exist in the CSV
        missing_levels = [lvl for lvl in self.levels if lvl not in hierarchy_df.columns]
        if missing_levels:
            logger.error(
                f"Hierarchy file is missing columns for configured levels: {missing_levels}"
            )
            logger.info(f"Available columns: {hierarchy_df.columns.tolist()}")
            raise ValueError(f"Missing columns {missing_levels} in {hierarchy_paths}")

        # Reject rows that are missing a value at any configured level; a
        # blank/NaN cell would create a ghost unit named "nan" and corrupt the
        # parent chain.
        invalid_mask = hierarchy_df[self.levels].isna().any(axis=1) | hierarchy_df[
            self.levels
        ].apply(lambda col: col.astype(str).str.strip() == "", axis=0).any(axis=1)
        if invalid_mask.any():
            invalid_count = int(invalid_mask.sum())
            logger.warning(
                f"Dropping {invalid_count} hierarchy row(s) with blank/NaN values "
                f"in configured level columns {self.levels}"
            )
            hierarchy_df = hierarchy_df[~invalid_mask].reset_index(drop=True)

        # 2. Apply filters if specified
        if self.filters and self.filters.get("codes"):
            filter_level = self.filters["level"]
            filter_names = set(self.filters["codes"])  # 'codes' key holds names

            if filter_level not in hierarchy_df.columns:
                logger.error(
                    f"Filter level '{filter_level}' not found in hierarchy columns."
                )
                raise ValueError(f"Invalid filter level: {filter_level}")

            # Codes are text (from YAML or a file); pandas may read the column
            # as numbers, so match on its text form.
            filter_codes = {str(code) for code in filter_names}
            column_as_text = hierarchy_df[filter_level].astype(str)
            original_size = len(hierarchy_df)
            hierarchy_df = hierarchy_df[column_as_text.isin(filter_codes)]

            unmatched_codes = sorted(filter_codes - set(column_as_text))
            if len(unmatched_codes) == len(filter_codes):
                raise ValueError(
                    f"Geography filter matched no {filter_level} units; codes "
                    f"{unmatched_codes[:5]} are not in the hierarchy's "
                    f"{filter_level} column, e.g. {list(column_as_text.unique()[:5])}."
                )
            if unmatched_codes:
                logger.warning(
                    f"{len(unmatched_codes)} of {len(filter_codes)} {filter_level} "
                    f"filter codes match no unit: {unmatched_codes[:5]}"
                )

            logger.info(
                f"Applied {filter_level} filter: {len(filter_names)} names specified, "
                f"reduced from {original_size} to {len(hierarchy_df)} rows"
            )

        # 3. Load coordinates for each level with configured files, restricted
        # to names present in the (post-filter) hierarchy. A level without an
        # entry in coord_files has no coordinates, by declaration.
        unknown_levels = [lvl for lvl in self.coord_files if lvl not in self.levels]
        if unknown_levels:
            raise ValueError(
                f"coord_files references unknown level(s) {unknown_levels}; "
                f"configured levels are {self.levels}."
            )
        names_per_level = {
            level: set(hierarchy_df[level].unique()) for level in self.levels
        }
        coords = {level: {} for level in self.levels}
        for level, file_spec in self.coord_files.items():
            coord_paths = [
                os.path.join(self.data_dir, p)
                for p in as_path_list(file_spec, f"geography.coord_files.{level}")
            ]
            coord_df = load_stacked_csv(coord_paths, label=f"{level} coordinates")
            self._validate_coord_columns(coord_df, coord_paths, level)
            for _candidate in ("geo_unit", level.lower(), level):
                if _candidate in coord_df.columns:
                    name_col = _candidate
                    break
            else:
                name_col = coord_df.columns[0]

            duplicated = coord_df[name_col].dropna()
            duplicated = duplicated[duplicated.duplicated()].unique()
            if len(duplicated) > 0:
                raise ValueError(
                    f"{level} coordinates define {len(duplicated)} unit(s) more "
                    f"than once across {coord_paths}, e.g. {list(duplicated[:5])}."
                )

            wanted = names_per_level[level]
            if wanted:
                coord_df = coord_df[coord_df[name_col].isin(wanted)]
            coords[level] = dict(
                zip(
                    coord_df[name_col], zip(coord_df["latitude"], coord_df["longitude"])
                )
            )
            logger.info(f"Loaded {len(coords[level])} coordinates for {level}")

        # 4. Create all units from hierarchy
        for level in self.levels:
            unique_names = hierarchy_df[level].unique()
            for name in unique_names:
                if name in self.units_by_level[level]:
                    continue
                coordinates = coords[level].get(name, None)
                unit = GeographicalUnit(
                    id=self._generate_id(),
                    name=name,
                    level=level,
                    coordinates=coordinates,
                )
                self.add_geo_unit(unit)

        logger.info(f"Created {len(self.units_by_id)} total units")

        # 5. Build parent-child relationships, vectorized per level pair.
        # Each (child, parent) pair appears once after drop_duplicates. A child
        # mapping to two different parents, in one file or across files, is
        # a hard error: silently keeping one parent would corrupt the tree.
        for i in range(len(self.levels) - 1):
            child_level = self.levels[i]
            parent_level = self.levels[i + 1]
            pairs = hierarchy_df[[child_level, parent_level]].drop_duplicates()
            conflicted = pairs[pairs[child_level].duplicated(keep=False)]
            if len(conflicted) > 0:
                examples = [
                    f"{child!r} -> {sorted(group[parent_level].unique())}"
                    for child, group in list(conflicted.groupby(child_level))[:5]
                ]
                raise ValueError(
                    f"{conflicted[child_level].nunique()} {child_level} unit(s) "
                    f"map to more than one {parent_level} parent: {examples}. "
                    f"Every unit must have exactly one parent."
                )
            child_index = self.units_by_level[child_level]
            parent_index = self.units_by_level[parent_level]
            for child_name, parent_name in pairs.itertuples(index=False, name=None):
                parent_index[parent_name].add_child(child_index[child_name])

        logger.info("Built hierarchical relationships")
        self._log_summary()

    @staticmethod
    def _validate_coord_columns(coord_df, coord_file, level):
        """
        Verify a coordinate CSV has the columns we rely on. Without this, a
        typo in the header surfaces as a mid-load KeyError far from the cause.
        """
        required = {"latitude", "longitude"}
        missing = required.difference(coord_df.columns)
        if missing:
            raise ValueError(
                f"Coordinate file {coord_file} for level {level} is missing "
                f"required column(s) {sorted(missing)}; found {list(coord_df.columns)}"
            )
        if len(coord_df.columns) < 3:
            raise ValueError(
                f"Coordinate file {coord_file} for level {level} must have a "
                f"name column followed by latitude and longitude; "
                f"found columns {list(coord_df.columns)}"
            )

    def add_geo_unit(self, unit: "GeographicalUnit"):
        existing = self.units.get(unit.name)
        if existing is None:
            self.units[unit.name] = unit
        elif existing.level != unit.level:
            logger.warning(
                f"Name collision across levels: '{unit.name}' exists at "
                f"{existing.level} (id={existing.id}) and {unit.level} (id={unit.id}); "
                f"get_unit('{unit.name}') will continue to return the {existing.level} unit. "
                f"Use get_units_by_level() to disambiguate."
            )
        elif existing.id != unit.id:
            logger.warning(
                f"Duplicate unit at {unit.level}: '{unit.name}' already registered with "
                f"id={existing.id}; ignoring new id={unit.id}."
            )
        self.units_by_id[unit.id] = unit
        self.units_by_level[unit.level][unit.name] = unit

    def add_geo_units(self, units):
        for unit in units:
            self.add_geo_unit(unit)

    def get_unit(self, name):
        """Get a unit by its name (or `code` in datasets that use codes as names)."""
        return self.units.get(name)

    def get_unit_by_id(self, id):
        """Get a unit by its numeric ID"""
        return self.units_by_id.get(id)

    def get_units_by_level(self, level):
        """Get all units at a specific level"""
        return self.units_by_level.get(level, {})

    def get_all_units(self):
        """Get all geographical units (returns dict of name -> unit)"""
        return self.units

    def get_all_units_list(self):
        """Get all geographical units as a list, sorted by ID"""
        return sorted(self.units.values(), key=lambda u: u.id)

    def get_roots(self):
        """Get all root units (units with no parent)"""
        return [unit for unit in self.units.values() if unit.parent is None]

    def _log_summary(self):
        """Log summary statistics about the geography"""
        for level in self.levels:
            count = len(self.units_by_level[level])
            logger.info(f"  {level}: {count} units")

    def __repr__(self):
        return f"<Geography: {len(self.units)} units across {len(self.levels)} levels>"

    def __eq__(self, other):
        if not isinstance(other, Geography):
            return NotImplemented
        for attribute in ["units", "levels"]:
            if getattr(self, attribute) != getattr(other, attribute):
                return False
        return True

    def __hash__(self):
        return hash((self.data_dir, tuple(self.levels)))
