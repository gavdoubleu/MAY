"""
Data source loaders for attribute assignment system.

This module handles loading demographic data from CSV files with:
- Regional routing (England/Wales, Scotland, Northern Ireland)
- Caching for performance
- Normalization of probability distributions
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from may.common.code_table import CodeTable
from may.utils import path_resolver as pr
from may.utils.attribute_access import get_person_attribute, get_nested_value

logger = logging.getLogger("may.attribute_assignment.data_sources")

class DataSource:
    """
    Base class for data sources.

    Data sources load demographic data from CSV files and provide
    probability distributions based on context (e.g., geographical unit code).
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize data source.

        Args:
            name: Identifier for this data source
            config: Configuration dict from YAML
        """
        self.name = name
        self.config = config
        self.cache: Dict[str, Any] = {}
        self._data_loaded = False
        # Loaded geo unit names keyed by hierarchy level, set by the manager
        # before load_data. Only sources that declare a level for their values
        # read it.
        self.geo_units_by_level: Optional[Dict[str, set]] = None

    def load_data(self, geo_units: Optional[set] = None):
        """
        Load data from CSV files.

        Args:
            geo_units: Optional set of geographical unit codes to filter by (for efficiency)
        """
        raise NotImplementedError("Subclasses must implement load_data()")

    def lookup(self, *args, **kwargs) -> Dict[str, float]:
        """
        Look up probability distribution for given context.

        Returns:
            Dict mapping attribute values to probabilities
        """
        raise NotImplementedError("Subclasses must implement lookup()")

    def _normalize_probabilities(self, probs: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize probabilities to ensure they sum to 1.0.

        Args:
            probs: Dictionary of probabilities

        Returns:
            Normalized probabilities that sum to 1.0
        """
        if not probs:
            raise ValueError(
                f"Empty probability distribution in source '{self.name}'. No fallbacks. "
                "Fix the data/config."
            )

        # Clamp negative values to 0, since negative probabilities are invalid
        has_negatives = False
        for v in probs.values():
            if v < 0:
                has_negatives = True
                break

        if has_negatives:
            neg_keys = [k for k, v in probs.items() if v < 0]
            logger.warning(
                f"Negative probability values in source '{self.name}' "
                f"for keys {neg_keys} — clamping to 0"
            )
            probs = {k: max(0.0, v) for k, v in probs.items()}

        total = sum(probs.values())

        if abs(total - 1.0) < 1e-10:  # Already normalized
            return probs
        elif total > 0:
            return {k: v / total for k, v in probs.items()}
        else:
            raise ValueError(
                f"All-zero probability distribution in source '{self.name}' "
                f"({len(probs)} keys). No fallbacks. Fix the data."
            )


class MissingDeclaredColumns(Exception):
    """A file does not have a column its own file block declares.

    Kept distinct from the generic per-file load failure so the loaders can
    re-raise it unwrapped: this is a config-contract violation, not a failure
    to read a file, and disguising it as one loses both the type and the
    carefully-built message.
    """


def _require_declared_columns(df: pd.DataFrame, source_name: str, file_path: Path,
                              key_columns: List[str],
                              value_columns: Dict[str, str]) -> None:
    """Every column a file block declares must exist in that file.

    Checked once per file, straight after `read_csv`, before the geo filter or
    any row parsing: per-row it would report the same fault thousands of times,
    and after the filter a missing key column has already skipped it silently.

    A missing *value* column is the reason this exists. The row loop would drop
    it and `_normalize_probabilities` would scale the survivors back to 1.0,
    inventing the absent category's probability mass into its siblings — a
    complete, well-formed, confidently wrong distribution that nothing
    downstream can detect. Key columns are checked in the same pass because
    their absence silently skips the geo-unit filter.
    """
    present = set(df.columns)
    missing = [f"'{column}' (key_columns)"
               for column in key_columns if column not in present]
    missing += [f"'{column}' (value_columns['{output_key}'])"
                for output_key, column in value_columns.items()
                if column not in present]
    if not missing:
        return

    raise MissingDeclaredColumns(
        f"source '{source_name}': {file_path} does not have declared column(s) "
        f"{', '.join(missing)}. The file has {sorted(present)}. Every declared "
        "column must exist: a missing value column would be dropped and the "
        "survivors renormalised, inventing its probability mass into the "
        "other categories. Fix the config or the file."
    )


def _ordered_key_columns(file_config: Dict[str, Any], source_name: str,
                         *, expected: Optional[int] = None) -> List[str]:
    """
    Read the canonical `key_columns` mapping and return its column names in order.

    `key_columns` is always a mapping; a single key is a one-entry mapping. The
    mapping's values carry per-key resolution config for MultiKey / OD sources;
    positional sources (geo distribution, diversity, pair) use only the column
    names, in declaration order. A singular `key_column` fails loudly.
    """
    if 'key_column' in file_config:
        raise ValueError(
            f"source '{source_name}': 'key_column' is retired. Use "
            "'key_columns' (a mapping; a single key is a one-entry mapping)."
        )
    key_columns = file_config.get('key_columns')
    if not key_columns:
        raise ValueError(
            f"source '{source_name}' needs 'key_columns' (a mapping of CSV key "
            "column name to optional resolution)."
        )
    if not isinstance(key_columns, dict):
        raise ValueError(
            f"source '{source_name}': 'key_columns' must be a mapping, got "
            f"{type(key_columns).__name__}. A list is the retired form."
        )
    columns = list(key_columns)
    if expected is not None and len(columns) != expected:
        raise ValueError(
            f"source '{source_name}': expected {expected} key column(s), got "
            f"{len(columns)}: {columns}."
        )
    return columns


def _merge_disjoint(target: Dict, new: Dict, source_name: str, file_path) -> None:
    """
    Merge one file's lookup into the accumulated lookup. A key appearing in
    two files of one source is a data error. Merging would have to pick a
    winner silently, so overlap instead fails loud, naming examples.
    """
    overlap = target.keys() & new.keys()
    if overlap:
        examples = sorted(overlap, key=str)[:5]
        raise ValueError(
            f"source '{source_name}': {len(overlap)} key(s) in {file_path} "
            f"were already loaded from an earlier file, e.g. {examples}. "
            f"Files within one source must cover disjoint keys."
        )
    target.update(new)


def _group_by_metadata(metadata_slices: Dict[str, np.ndarray],
                       likelihoods: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Merge rows carrying identical metadata, summing their likelihood mass.

    Returns the per-group metadata columns and the summed mass per group, in
    one pass over a structured view rather than a Python dict keyed on row
    tuples. Values are compared as stored — raw for numeric metadata columns,
    as CodeTable codes for the rest — which is exact either way because a code
    is one-to-one with its value.

    With no metadata columns every row shares the same (empty) key, so this
    collapses to a single group carrying the total.
    """
    if not metadata_slices:
        return {}, np.array([likelihoods.sum()], dtype=likelihoods.dtype)
    keys = np.empty(len(likelihoods),
                    dtype=[(key, column.dtype) for key, column in metadata_slices.items()])
    for key, column in metadata_slices.items():
        keys[key] = column
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    masses = np.bincount(inverse.ravel(), weights=likelihoods, minlength=len(unique_keys))
    return ({key: unique_keys[key] for key in metadata_slices},
            masses.astype(likelihoods.dtype))


class GeoDistributionSource(DataSource):
    """
    Data source for geographical unit-specific attribute distributions.

    Loads attribute distributions from CSV file.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize geo distribution source.

        Args:
            name: Source name
            config: Configuration with files and fallback
        """
        super().__init__(name, config)

        # Data lookup: geo_unit -> {ethnicity -> probability}
        self._lookup: Dict[str, Dict[str, float]] = {}

        # Parse file configurations
        self._file_configs = config.get('files', [])

    def load_data(self, geo_units: Optional[set] = None):
        """
        Load geographical unit distribution data from CSV file.

        Args:
            geo_units: Set of geographical unit codes to load (for efficiency)
        """
        logger.info(f"Loading data for source '{self.name}'...")

        # Each file contributes its own geo units; the merged lookup must be
        # key-disjoint across files.
        merged: Dict[str, Dict[str, float]] = {}
        for file_config in self._file_configs:
            file_path = Path(pr.resolve(file_config['path']))

            # Load and process CSV
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)

                    key_column = _ordered_key_columns(file_config, self.name, expected=1)[0]
                    value_columns = file_config.get('value_columns', {})
                    _require_declared_columns(df, self.name, file_path,
                                              [key_column], value_columns)

                    # Filter to needed areas
                    if geo_units:
                        df = df[df[key_column].isin(geo_units)]

                    total_column = file_config.get('total_column')
                    lookup = self._parse_dataframe(
                        df, key_column, value_columns, total_column
                    )

                    logger.info(f"  ✓ Loaded {len(lookup)} geographical units from {file_path.name}")

                except MissingDeclaredColumns:
                    raise
                except Exception as e:
                    # Fail loud on a load/parse error.
                    raise RuntimeError(
                        f"failed to load data source file {file_path}: {e}"
                    ) from e
                _merge_disjoint(merged, lookup, self.name, file_path)
            else:
                raise FileNotFoundError(f"data source file not found: {file_path}")
        self._lookup = merged

        self._data_loaded = True

    def _parse_dataframe(self, df: pd.DataFrame, key_column: str,
                        value_columns: Dict[str, str],
                        total_column: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Parse DataFrame into lookup dictionary.

        Args:
            df: DataFrame to parse
            key_column: Column with geographical unit codes
            value_columns: Mapping of output keys to DataFrame columns
            total_column: Optional column with totals (for normalization)

        Returns:
            Dictionary mapping geographical unit codes to probability distributions
        """
        lookup = {}

        for _, row in df.iterrows():
            geo_unit = row[key_column]

            # Get total if available
            if total_column and total_column in df.columns:
                total = row[total_column]
            else:
                total = None

            # Build probability distribution
            probs = {}
            for output_key, df_column in value_columns.items():
                if df_column in df.columns:
                    value = row[df_column]

                    # Normalize by total if provided
                    if total is not None and total > 0:
                        probs[output_key] = value / total
                    else:
                        probs[output_key] = value

            # Normalize probabilities
            probs = self._normalize_probabilities(probs)
            lookup[geo_unit] = probs

        return lookup

    def lookup(self, person, household=None, context=None) -> Dict[str, float]:
        """
        Look up the distribution for a person's residence geographical unit.

        Resolves the key itself: the residence venue's geo unit first,
        then the person's own.
        """
        if not self._data_loaded:
            raise RuntimeError(
                f"Data not loaded for source '{self.name}'. No fallbacks. "
                "Fix the source/data so it loads."
            )

        geo_unit = None
        if household is not None and getattr(household, 'geographical_unit', None):
            geo_unit = household.geographical_unit.name
        if not geo_unit and getattr(person, 'geographical_unit', None):
            geo_unit = person.geographical_unit.name
        if not geo_unit:
            raise KeyError(
                f"Source '{self.name}': no residence geographical_unit for person "
                f"{person.id} (no venue geo and no person-level geo). No fallbacks."
            )

        if geo_unit in self._lookup:
            return self._lookup[geo_unit]

        raise KeyError(
            f"Source '{self.name}' has no row for geo unit '{geo_unit}'. No fallbacks. "
            "The data must cover every keyed unit, or it's a real gap."
        )


class DiversitySource(DataSource):
    """
    Data source for venue diversity (single vs mixed attribute values).

    Provides probabilities for whether a venue has:
    - Single attribute value (all members same)
    - Two attribute values
    - Three or more attribute values
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize diversity source."""
        super().__init__(name, config)
        self._lookup: Dict[str, Dict[str, float]] = {}
        self._file_configs = config.get('files', [])

    def load_data(self, geo_units: Optional[set] = None):
        """Load diversity data from CSV file."""
        logger.info(f"Loading data for source '{self.name}'...")

        merged: Dict[str, Dict[str, float]] = {}
        for file_config in self._file_configs:
            file_path = Path(pr.resolve(file_config['path']))

            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    key_column = _ordered_key_columns(file_config, self.name, expected=1)[0]
                    value_columns = file_config.get('value_columns', {})
                    _require_declared_columns(df, self.name, file_path,
                                              [key_column], value_columns)

                    if geo_units:
                        df = df[df[key_column].isin(geo_units)]

                    lookup = self._parse_diversity_dataframe(df, key_column, value_columns)

                    logger.info(f"  ✓ Loaded {len(lookup)} geographical units from {file_path.name}")

                except MissingDeclaredColumns:
                    raise
                except Exception as e:
                    # Fail loud on a load/parse error.
                    raise RuntimeError(
                        f"failed to load data source file {file_path}: {e}"
                    ) from e
                _merge_disjoint(merged, lookup, self.name, file_path)
            else:
                raise FileNotFoundError(f"data source file not found: {file_path}")
        self._lookup = merged

        self._data_loaded = True

    def _parse_diversity_dataframe(self, df: pd.DataFrame, key_column: str,
                                   value_columns: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """Parse diversity DataFrame."""
        lookup = {}

        for _, row in df.iterrows():
            geo_unit = row[key_column]

            # Get diversity counts
            counts = {}
            for output_key, df_column in value_columns.items():
                if df_column in df.columns:
                    counts[output_key] = float(row[df_column])

            # Normalize to probabilities
            total = sum(counts.values())
            if total <= 0:
                raise ValueError(
                    f"Source '{self.name}': zero-total diversity counts for geo unit "
                    f"'{geo_unit}'. No fallbacks. Fix the data."
                )
            probs = {k: v / total for k, v in counts.items()}

            lookup[geo_unit] = self._normalize_probabilities(probs)

        return lookup

    def lookup(self, geo_unit: str) -> Dict[str, float]:
        """Look up diversity probabilities for a geographical unit."""
        if not self._data_loaded:
            raise RuntimeError(
                f"Data not loaded for source '{self.name}'. No fallbacks."
            )

        if geo_unit in self._lookup:
            return self._lookup[geo_unit]

        raise KeyError(
            f"Source '{self.name}' has no diversity row for geo unit '{geo_unit}'. "
            "No fallbacks."
        )


class PairProbabilitySource(DataSource):
    """
    Data source for pair probabilities.

    Provides conditional probabilities: given first person's attribute value,
    what is the probability of second person having each value?
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize pair probability source."""
        super().__init__(name, config)
        # Nested lookup: geo_unit -> first_ethnicity -> partner_ethnicity -> probability
        self._lookups: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._file_configs = config.get('files', [])

    def load_data(self, geo_units: Optional[set] = None):
        """Load pair probability data."""
        logger.info(f"Loading data for source '{self.name}'...")

        merged: Dict[str, Dict[str, Dict[str, float]]] = {}
        for file_config in self._file_configs:
            file_path = Path(pr.resolve(file_config['path']))

            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)

                    key_columns = _ordered_key_columns(file_config, self.name, expected=2)
                    value_columns = file_config.get('value_columns', {})
                    _require_declared_columns(df, self.name, file_path,
                                              key_columns, value_columns)

                    # Filter to needed areas if specified
                    if geo_units:
                        df = df[df[key_columns[0]].isin(geo_units)]

                    lookups = self._parse_pair_dataframe(df, key_columns, value_columns)

                    logger.info(f"  ✓ Loaded {len(lookups)} geographical units from {file_path.name}")

                except MissingDeclaredColumns:
                    raise
                except Exception as e:
                    # Fail loud on a load/parse error.
                    raise RuntimeError(
                        f"failed to load data source file {file_path}: {e}"
                    ) from e
                _merge_disjoint(merged, lookups, self.name, file_path)
            else:
                raise FileNotFoundError(f"data source file not found: {file_path}")
        self._lookups = merged

        self._data_loaded = True

    def _parse_pair_dataframe(self, df: pd.DataFrame, key_columns: List[str],
                              value_columns: Dict[str, str]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Parse pair probability DataFrame into nested lookup."""
        lookup = {}

        geo_col, first_value_col = key_columns[0], key_columns[1]

        for _, row in df.iterrows():
            geo_unit = row[geo_col]
            first_value = row[first_value_col]

            # Build second person probability distribution
            second_probs = {}
            for output_key, df_column in value_columns.items():
                if df_column in df.columns:
                    second_probs[output_key] = float(row[df_column])

            # Normalize
            second_probs = self._normalize_probabilities(second_probs)

            # Store in nested structure
            if geo_unit not in lookup:
                lookup[geo_unit] = {}
            lookup[geo_unit][first_value] = second_probs

        return lookup

    def lookup(self, geo_unit: str, first_value: str) -> Dict[str, float]:
        """
        Look up pair probabilities.

        Args:
            geo_unit: Geographical unit code
            first_value: Attribute value of first person

        Returns:
            Probability distribution for second person's attribute value
        """
        if not self._data_loaded:
            raise RuntimeError(
                f"Data not loaded for source '{self.name}'. No fallbacks."
            )

        # Look up geographical unit
        if geo_unit in self._lookups:
            # Look up first value within unit
            if first_value in self._lookups[geo_unit]:
                return self._lookups[geo_unit][first_value]

        raise KeyError(
            f"Source '{self.name}' has no pair row for (geo='{geo_unit}', "
            f"first='{first_value}'). No fallbacks. The pair data must cover "
            "every (unit, first-value) combination the model produces."
        )


class MultiKeyLookupSource(DataSource):
    """
    Data source for multi-key CSV lookups.

    Supports lookups based on multiple keys (e.g., sex + age + ethnicity + region).
    Uses a pure Python dictionary for maximum performance.
    """

    def __init__(self, name: str, config: Dict[str, Any], assignment_config):
        """
        Initialize multi-key lookup source.

        Args:
            name: Source name
            config: Configuration with files and fallback
            assignment_config: Parent AttributeAssignmentConfig for category lookups
        """
        super().__init__(name, config)
        self.assignment_config = assignment_config
        self._file_configs = config.get('files', [])
        self._lookup_dict = {}  # Dict mapping tuple keys to value dicts
        self._key_columns = []
        self._value_columns = {}
        self._key_columns_config = None  # Cache key columns config for fast lookup

        # Cache for lookup results and key resolution
        self._lookup_cache = {}  # Cache for lookup() results by tuple key
        self._key_value_cache = {}  # Cache for _resolve_key_value() results

    def load_data(self, geo_units: Optional[set] = None):
        """Load CSV data and convert to dictionary for fast lookups."""
        logger.info(f"Loading data for source '{self.name}'...")

        merged: Dict = {}
        for file_config in self._file_configs:
            file_path = Path(pr.resolve(file_config['path']))

            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)

                    _require_declared_columns(
                        df, self.name, file_path,
                        list(file_config.get('key_columns', {}).keys()),
                        file_config.get('value_columns', {}))

                    # Apply row filters if specified
                    row_filter = file_config.get('row_filter', {})
                    if row_filter:
                        for col, value in row_filter.items():
                            if col in df.columns:
                                df = df[df[col] == value]
                                logger.info(f"  Applied filter: {col} == '{value}' ({len(df)} rows remaining)")

                    # Key/value column config must agree across a source's
                    # files, so the lookup has one shape.
                    key_columns = list(file_config.get('key_columns', {}).keys())
                    if self._key_columns and key_columns != self._key_columns:
                        raise ValueError(
                            f"key_columns differ between files: {self._key_columns} "
                            f"vs {key_columns}."
                        )
                    self._key_columns = key_columns
                    self._key_columns_config = file_config.get('key_columns', {})
                    self._value_columns = file_config.get('value_columns', {})

                    # Build dictionary: {(key1, key2, ...): {col1: val1, col2: val2, ...}}
                    logger.info(f"  Building lookup dictionary from {len(df)} rows...")

                    file_lookup = {}
                    for _, row in df.iterrows():
                        # Build key tuple
                        key = tuple(row[col] for col in self._key_columns)

                        # Build value dict
                        values = {name: float(row[csv_col]) for name, csv_col in self._value_columns.items()}

                        file_lookup[key] = values

                    logger.info(f"  ✓ Loaded {len(file_lookup)} rows from {file_path.name} into dictionary")
                except MissingDeclaredColumns:
                    raise
                except Exception as e:
                    # Fail loud on a load/parse error.
                    raise RuntimeError(
                        f"failed to load data source file {file_path}: {e}"
                    ) from e
                _merge_disjoint(merged, file_lookup, self.name, file_path)
            else:
                raise FileNotFoundError(f"data source file not found: {file_path}")
        self._lookup_dict = merged

        self._data_loaded = True

    def lookup(self, person, household=None, context=None) -> Dict[str, float]:
        """
        Look up probabilities based on person demographics using dictionary lookup.

        Uses caching for repeated lookups with same keys.

        Args:
            person: Person object
            household: Optional household object
            context: Optional additional context

        Returns:
            Dict of value columns (e.g., {'cvd': 0.05, 'crd': 0.03, ...})
        """
        debug = context and context.get('debug', False)

        if not self._lookup_dict:
            raise RuntimeError(
                f"Source '{self.name}' has no data loaded. No fallbacks."
            )

        # Build key tuple directly (faster than building intermediate dict)
        key_values = []
        for csv_col_name, col_config in self._key_columns_config.items():
            value = self._resolve_key_value_cached(col_config, person, household, context)
            if value is None:
                raise KeyError(
                    f"Source '{self.name}': could not resolve key column "
                    f"'{csv_col_name}' for person {person.id}. No fallbacks. "
                    "The person is missing an attribute the key needs, or the key config "
                    "is wrong."
                )
            key_values.append(value)

        # Direct dictionary lookup with tuple key
        lookup_key = tuple(key_values)

        # Check cache first
        if lookup_key in self._lookup_cache:
            return self._lookup_cache[lookup_key]

        if debug:
            logger.debug(f"    [LOOKUP] Key: {lookup_key}")

        # dictionary lookup
        result = self._lookup_dict.get(lookup_key)

        if result is None:
            raise KeyError(
                f"Source '{self.name}' has no row for key {lookup_key}. No fallbacks. "
                "The data must cover every demographic combination the "
                "model produces, or this is a real gap."
            )

        if debug:
            logger.debug(f"    [LOOKUP] ✓ Found data: {list(result.keys())[:3]}...")

        # Normalize the result (convert counts to probabilities)
        result = self._normalize_probabilities(result)

        # Cache the normalized result
        self._lookup_cache[lookup_key] = result

        return result

    def _resolve_key_value_cached(self, col_config, person, household, context):
        """
        Caches results for person attributes that don't change.

        Args:
            col_config: Column configuration dict
            person: Person object
            household: Optional household object
            context: Optional context dict

        Returns:
            Resolved value or None if can't resolve
        """
        attr_name = col_config.get('attribute')
        col_type = col_config.get('type', 'direct')

        # Create cache key based on person ID, attribute name, and type
        # For person-level attributes (sex, age), these are immutable so we can cache
        if col_type == 'direct':
            cache_key = (person.id, attr_name, 'direct')
            if cache_key in self._key_value_cache:
                return self._key_value_cache[cache_key]

        # Call the actual resolution
        result = self._resolve_key_value(col_config, person, household, context)

        # Cache the result (only for direct lookups to avoid complexity)
        if col_type == 'direct':
            cache_key = (person.id, attr_name, 'direct')
            self._key_value_cache[cache_key] = result

        return result

    def _resolve_key_value(self, col_config, person, household, context):
        """
        Resolve a key value based on column configuration.

        Args:
            col_config: Column configuration dict
            person: Person object
            household: Optional household object
            context: Optional context dict

        Returns:
            Resolved value or None if can't resolve
        """
        attr_name = col_config.get('attribute')
        col_type = col_config.get('type', 'direct')

        if col_type == 'direct':
            return get_person_attribute(person, attr_name)

        elif col_type == 'category_lookup':
            # Get attribute value, find matching category
            value = get_person_attribute(person, attr_name)
            category = self.assignment_config.get_category_for_value(value, attr_name)
            return category.get('csv_value') if category else None

        elif col_type == 'ancestor_lookup':
            # Traverse hierarchy
            geo_unit = get_person_attribute(person, attr_name)
            if geo_unit is None and household:
                # Use the household's geo unit when the person has none
                geo_unit = get_nested_value(household, attr_name)

            if geo_unit is None:
                return None

            level = col_config.get('level')
            ancestor = geo_unit.get_ancestor_by_level(level)

            if ancestor is None:
                return None

            property_name = col_config.get('property', 'name')
            return getattr(ancestor, property_name)

        return None


class OriginDestinationMatrixSource(DataSource):
    """
    Data source for origin-destination flow matrices.

    Used for commuting patterns, migration flows, etc.
    Returns all possible destinations for a given origin with associated likelihoods.
    """

    #: Default rows per chunk for the chunked CSV read, matching
    #: RouteLegsIndex (may/venue_distributor/route_legs_index.py) — the
    #: precedent this columnar design follows. Overridable per file_config
    #: via 'chunk_size'.
    _DEFAULT_CHUNK_SIZE = 500_000

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize O-D matrix source."""
        super().__init__(name, config)
        # Columnar storage, populated by load_data(). Empty/absent until then
        # — see module note on OriginDestinationMatrixSource for why this is
        # not a Dict[str, List[Tuple[...]]] any more (memory at ~92M-row
        # scale). self._origin_code_table maps origin string -> dense int32
        # code; self._group_starts/_group_stops (indexed by that code) give
        # the row range in self._dest_codes/_likelihoods/_metadata_arrays for
        # that origin.
        self._origin_code_table = CodeTable()
        self._dest_code_table = CodeTable()
        self._metadata_code_tables: Dict[str, CodeTable] = {}
        # Per metadata key: stored raw (numeric) or as CodeTable codes. Decided
        # by the first chunk that carries the column and held for the whole
        # source, so files cannot disagree silently.
        self._metadata_is_numeric: Dict[str, bool] = {}
        self._dest_codes = np.empty(0, dtype=np.int32)
        self._likelihoods = np.empty(0, dtype=np.float64)
        self._metadata_arrays: Dict[str, np.ndarray] = {}
        # Per-row: does this row's metadata read back as the outside sentinel
        # rather than as stored values? True only for sentinel rows minted with
        # no source row behind them (redistribute + on_empty='outside').
        self._group_starts = np.empty(0, dtype=np.int64)
        self._group_stops = np.empty(0, dtype=np.int64)
        self._file_configs = config.get('files', [])

        # Out-of-boundary destination policy. Required, with no default, because
        # a destination drawn outside the loaded world boundary is routine in
        # region runs and impossible in whole-country runs, so the engine refuses
        # to guess what to do with it.
        self._out_of_boundary = config.get('out_of_boundary')
        if self._out_of_boundary is None:
            raise ValueError(
                f"O-D source '{self.name}' must declare 'out_of_boundary' "
                "(error | redistribute | outside). It is required, with no "
                "default. Silence is never interpreted."
            )
        if self._out_of_boundary not in ('error', 'redistribute', 'outside'):
            raise ValueError(
                f"O-D source '{self.name}': out_of_boundary='{self._out_of_boundary}' "
                "is not one of error | redistribute | outside."
            )
        self._outside_value = config.get('outside_value')
        self._on_empty = config.get('on_empty')
        if self._out_of_boundary == 'redistribute':
            if self._on_empty not in ('error', 'outside'):
                raise ValueError(
                    f"O-D source '{self.name}': out_of_boundary='redistribute' "
                    "requires 'on_empty' (error | outside) for origins whose entire "
                    "distribution is out-of-boundary."
                )
        # The sentinel value is required whenever the 'outside' outcome can occur.
        outside_can_occur = (
            self._out_of_boundary == 'outside'
            or (self._out_of_boundary == 'redistribute' and self._on_empty == 'outside')
        )
        if outside_can_occur and not self._outside_value:
            raise ValueError(
                f"O-D source '{self.name}': 'outside_value' is required when the "
                "'outside' outcome can occur."
            )

        # Which metadata columns survive the boundary. Out-of-boundary rows
        # collapse onto sentinel rows grouped by exactly these columns, so the
        # list also bounds the sentinel count: one row per distinct combination
        # of preserved values, not one per out-of-boundary row. Required (no
        # default) whenever the sentinel can occur — preserving everything is
        # what makes the collapse degenerate, so it must not be what you get by
        # not thinking. An empty list is legal and means one sentinel per origin.
        self._sentinel_preserves = config.get('sentinel_preserves')
        if outside_can_occur:
            if self._sentinel_preserves is None:
                raise ValueError(
                    f"O-D source '{self.name}': 'sentinel_preserves' is required "
                    "when the 'outside' outcome can occur. It names the metadata "
                    "columns an out-of-boundary row keeps; [] means keep none."
                )
            if not isinstance(self._sentinel_preserves, list):
                raise ValueError(
                    f"O-D source '{self.name}': 'sentinel_preserves' must be a "
                    f"list, got {type(self._sentinel_preserves).__name__}."
                )
            declared = set()
            for file_config in self._file_configs:
                declared.update(file_config.get('metadata_columns', {}))
            unknown = [k for k in self._sentinel_preserves if k not in declared]
            if unknown:
                raise ValueError(
                    f"O-D source '{self.name}': sentinel_preserves names "
                    f"{unknown}, which is not a declared metadata column "
                    f"({sorted(declared)})."
                )
        self._sentinel_preserves = list(self._sentinel_preserves or [])
        # Destination code of the sentinel, allocated on demand while applying
        # the boundary policy. Identifies sentinel rows on read.
        self._outside_dest_code: Optional[int] = None

        # Optional marker for redistributed assignments. Names the person
        # property set true when an assignment was bounced back in-boundary, so
        # the venue layer can deprioritise it. Config-named, so the engine carries
        # whatever the scenario calls it. Only meaningful under 'redistribute':
        # flagging it elsewhere raises (no silent no-ops).
        self._redistributed_flag = config.get('redistributed_flag')
        if self._redistributed_flag is not None and self._out_of_boundary != 'redistribute':
            raise ValueError(
                f"O-D source '{self.name}': 'redistributed_flag' is only valid under "
                f"out_of_boundary='redistribute', not '{self._out_of_boundary}'."
            )
        # Per-origin fraction of flow that was bounced back in-boundary, populated
        # by the redistribute branch of _apply_boundary_policy. Empty otherwise.
        self._redistributed_fraction: Dict[str, float] = {}

    def load_data(self, geo_units: Optional[set] = None):
        """Load origin-destination flow data from CSV, chunked and columnar
        (never a full-file DataFrame or a per-row Python object) so a
        ~92M-row table stays memory-bounded. See RouteLegsIndex
        (may/venue_distributor/route_legs_index.py) for the precedent this
        follows.

        Every file of one source contributes to ONE lookup: the census
        publishes commuting flow as separate national tables, and a run
        spanning them must see a single origin space. Files must cover
        disjoint origins (an origin in two files has no single answer) and
        must declare the same metadata keys (the columnar store holds one
        dense array per key, which cannot be ragged).
        """
        logger.info(f"Loading data for source '{self.name}'...")

        self._require_uniform_metadata_columns()

        # Code tables are shared across the files of a source, so a value
        # first seen in the second file gets one code, not a per-file one.
        self._origin_code_table = CodeTable()
        self._dest_code_table = CodeTable()
        self._metadata_code_tables = {}
        self._metadata_is_numeric: Dict[str, bool] = {}

        origin_code_chunks: List[np.ndarray] = []
        dest_code_chunks: List[np.ndarray] = []
        likelihood_chunks: List[np.ndarray] = []
        metadata_chunks: Dict[str, List[np.ndarray]] = {}

        for file_config in self._file_configs:
            file_path = Path(pr.resolve(file_config['path']))

            if not file_path.exists():
                raise FileNotFoundError(f"data source file not found: {file_path}")

            origins_before = len(self._origin_code_table)
            try:
                origin_codes, dest_codes, likelihoods, metadata_arrays = self._load_file(
                    file_path, file_config, geo_units
                )
            except Exception as e:
                # Fail loud on a load/parse error.
                raise RuntimeError(
                    f"failed to load data source file {file_path}: {e}"
                ) from e

            self._require_disjoint_origins(origin_codes, origins_before, file_path)

            if not len(origin_codes):
                continue
            origin_code_chunks.append(origin_codes)
            dest_code_chunks.append(dest_codes)
            likelihood_chunks.append(likelihoods)
            for meta_key, array in metadata_arrays.items():
                metadata_chunks.setdefault(meta_key, []).append(array)

        self._build_groups(origin_code_chunks, dest_code_chunks,
                           likelihood_chunks, metadata_chunks)

        # Apply the out-of-boundary policy AFTER parsing and OUTSIDE the
        # per-file try/except above, since policy violations must fail loud.
        self._apply_boundary_policy(geo_units)

        self._data_loaded = True

    def _require_uniform_metadata_columns(self) -> None:
        """Every file of a source must declare the same metadata keys.

        Storage is one dense array per key covering all rows of the source, so
        a key present in one file and absent from another has no representable
        value on the rows it is missing from. Declaring fewer keys on one file
        is therefore a config error, not a licence to leave holes.
        """
        key_sets = [frozenset(fc.get('metadata_columns', {})) for fc in self._file_configs]
        if len(set(key_sets)) <= 1:
            return
        union = sorted(set().union(*key_sets))
        per_file = {
            str(fc.get('path')): sorted(keys)
            for fc, keys in zip(self._file_configs, key_sets)
        }
        raise ValueError(
            f"O-D source '{self.name}': files declare different "
            f"'metadata_columns' keys {per_file}. Every file must declare all "
            f"of {union}, or the source must stop declaring the missing ones."
        )

    def _require_disjoint_origins(self, origin_codes: np.ndarray,
                                   origins_before: int, file_path: Path) -> None:
        """An origin appearing in two files of one source is a data error.

        Codes are assigned in first-appearance order, so any code this file
        produced that is below the table size before the file was read came
        from an earlier file. Merging would have to pick a winner silently.
        """
        if not len(origin_codes):
            return
        seen = np.unique(origin_codes)
        overlap = seen[seen < origins_before]
        if not len(overlap):
            return
        examples = [self._origin_code_table[int(c)] for c in overlap[:5]]
        raise ValueError(
            f"O-D source '{self.name}': {len(overlap)} origin(s) in {file_path} "
            f"were already loaded from an earlier file, e.g. {sorted(examples)}. "
            "Files within one source must cover disjoint origins."
        )

    def _load_file(self, file_path: Path, file_config: Dict[str, Any],
                    geo_units: Optional[set]) -> Tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, Dict[str, np.ndarray]]:
        origin_column = _ordered_key_columns(file_config, self.name, expected=1)[0]
        destination_column = file_config.get('destination_column')
        likelihood_column = file_config.get('likelihood_column')
        metadata_columns = file_config.get('metadata_columns', {})
        exclude_destinations = set(file_config.get('exclude_destinations', []))
        chunk_size = file_config.get('chunk_size', self._DEFAULT_CHUNK_SIZE)

        # Overlap pre-pass: only filter by geo_units if origin values in this
        # file actually overlap with it (some O-D files are keyed at a level
        # geo_units never matches, e.g. LGU origins under OA/SGU geo_units —
        # they must stay unfiltered). Reads only the origin column, chunked.
        should_filter = False
        if geo_units:
            origin_values = set()
            for chunk in pd.read_csv(file_path, usecols=[origin_column], chunksize=chunk_size):
                origin_values.update(chunk[origin_column].unique())
            overlap = origin_values & geo_units
            should_filter = bool(overlap)
            if should_filter:
                logger.info(f"  Filtering O-D matrix based on {len(overlap)} matching origins")

        origin_code_table = self._origin_code_table
        dest_code_table = self._dest_code_table
        metadata_code_tables = self._metadata_code_tables
        metadata_is_numeric = self._metadata_is_numeric

        origin_code_chunks: List[np.ndarray] = []
        dest_code_chunks: List[np.ndarray] = []
        likelihood_chunks: List[np.ndarray] = []
        metadata_chunks: Dict[str, List[np.ndarray]] = {}
        n_rows = 0

        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            if should_filter:
                chunk = chunk[chunk[origin_column].isin(geo_units)]
            if chunk.empty:
                continue

            # Register every origin appearing (post geo_units-filter, PRE
            # exclude_destinations) so it gets a dense code even if all its
            # destinations end up excluded below — a known origin with zero
            # destinations must resolve to [], not a missing key.
            origin_codes = origin_code_table.encode_series(chunk[origin_column])

            if exclude_destinations:
                keep_mask = (~chunk[destination_column].isin(exclude_destinations)).to_numpy()
                chunk = chunk[keep_mask]
                origin_codes = origin_codes[keep_mask]
            if chunk.empty:
                continue

            dest_codes = dest_code_table.encode_series(chunk[destination_column])
            likelihoods = chunk[likelihood_column].to_numpy(dtype=np.float64)

            origin_code_chunks.append(origin_codes)
            dest_code_chunks.append(dest_codes)
            likelihood_chunks.append(likelihoods)

            for meta_key, meta_column in metadata_columns.items():
                if meta_column not in chunk.columns:
                    raise ValueError(
                        f"metadata column '{meta_column}' (declared as "
                        f"'{meta_key}') is not in the file. Declared metadata "
                        "must exist: storage is one dense array per key, so a "
                        "missing column has no representable value."
                    )
                series = chunk[meta_column]
                is_numeric = pd.api.types.is_numeric_dtype(series)
                if meta_key not in metadata_is_numeric:
                    metadata_is_numeric[meta_key] = is_numeric
                    if not is_numeric:
                        metadata_code_tables[meta_key] = CodeTable()
                elif metadata_is_numeric[meta_key] != is_numeric:
                    # pandas infers dtype per chunk, so a column that's numeric
                    # in one chunk and stringy in another must fail loud rather
                    # than have the wrong-typed chunk silently skip encoding.
                    raise ValueError(
                        f"metadata column '{meta_column}' has inconsistent dtype "
                        f"across chunks or files (numeric vs non-numeric)"
                    )
                metadata_chunks.setdefault(meta_key, [])
                if metadata_is_numeric[meta_key]:
                    metadata_chunks[meta_key].append(series.to_numpy())
                else:
                    metadata_chunks[meta_key].append(metadata_code_tables[meta_key].encode_series(series))

            n_rows += len(chunk)

        if origin_code_chunks:
            origin_codes = np.concatenate(origin_code_chunks)
            dest_codes = np.concatenate(dest_code_chunks)
            likelihoods = np.concatenate(likelihood_chunks)
        else:
            origin_codes = np.empty(0, dtype=np.int32)
            dest_codes = np.empty(0, dtype=np.int32)
            likelihoods = np.empty(0, dtype=np.float64)
        metadata_arrays = {
            meta_key: (np.concatenate(chunks) if chunks else np.empty(0))
            for meta_key, chunks in metadata_chunks.items()
        }

        logger.info(f"  ✓ Loaded {n_rows} row(s) from {file_path.name}")
        return origin_codes, dest_codes, likelihoods, metadata_arrays

    def _build_groups(self, origin_code_chunks: List[np.ndarray],
                       dest_code_chunks: List[np.ndarray],
                       likelihood_chunks: List[np.ndarray],
                       metadata_chunks: Dict[str, List[np.ndarray]]) -> None:
        """Concatenate the files of this source into one grouped, normalised
        store. Grouping and normalisation happen once, over all files, so an
        origin is normalised against its own rows wherever they came from."""
        if origin_code_chunks:
            origin_codes = np.concatenate(origin_code_chunks)
            dest_codes = np.concatenate(dest_code_chunks)
            likelihoods = np.concatenate(likelihood_chunks)
        else:
            origin_codes = np.empty(0, dtype=np.int32)
            dest_codes = np.empty(0, dtype=np.int32)
            likelihoods = np.empty(0, dtype=np.float64)
        metadata_arrays = {
            meta_key: np.concatenate(chunks)
            for meta_key, chunks in metadata_chunks.items()
        }

        # Sort by origin code (stable, to preserve within-origin file order —
        # matches the old groupby+iterrows behaviour), then group densely:
        # origin codes are 0..V-1 by construction, so bincount/cumsum gives
        # O(1) direct-index group lookup — simpler than a sorted-key binary
        # search, which RouteLegsIndex needs only because its key is a
        # 3-field composite (sparse keyspace), unlike this single-field one.
        order = np.argsort(origin_codes, kind='stable')
        origin_codes = origin_codes[order]
        dest_codes = dest_codes[order]
        likelihoods = likelihoods[order]
        metadata_arrays = {k: v[order] for k, v in metadata_arrays.items()}

        vocab_size = len(self._origin_code_table)
        counts = (np.bincount(origin_codes, minlength=vocab_size) if len(origin_codes)
                  else np.zeros(vocab_size, dtype=np.int64))
        group_stops = np.cumsum(counts)
        group_starts = group_stops - counts

        # Normalise likelihoods to sum to 1.0 per origin. A numpy-slice loop
        # over origins (~thousands to low hundred-thousands), not a
        # Python-tuple loop over rows — the row count was the memory/time
        # problem, not the origin count.
        massless_origins: List[str] = []
        for code in range(vocab_size):
            start, stop = int(group_starts[code]), int(group_stops[code])
            if stop <= start:
                continue
            total = likelihoods[start:stop].sum()
            if total > 0:
                likelihoods[start:stop] /= total
            else:
                # A massless origin: every row carries zero likelihood, so
                # there is nothing to divide by. The destinations and their
                # metadata are still real observations, so the origin is
                # weighted uniformly rather than left as a zero vector no
                # downstream draw can sample from. Settled here, once, so no
                # later stage (the boundary policy above all) has to re-handle
                # a sum that cannot be normalised.
                likelihoods[start:stop] = 1.0 / (stop - start)
                massless_origins.append(self._origin_code_table[code])

        if massless_origins:
            logger.warning(
                f"  [massless] {len(massless_origins)} origin(s) in source "
                f"'{self.name}' carry zero likelihood mass on every row "
                f"(e.g. {massless_origins[:15]}). Their weights are INVENTED: "
                "mass spread uniformly over the destinations observed."
            )

        self._dest_codes = dest_codes
        self._likelihoods = likelihoods
        self._metadata_arrays = metadata_arrays
        self._group_starts = group_starts
        self._group_stops = group_stops

        logger.info(f"  ✓ Loaded {vocab_size} origins for source '{self.name}'")

    def _collapse_onto_sentinel(
            self, outside_metadata: Dict[str, np.ndarray], outside_likelihoods: np.ndarray,
            outside_code: int, dest_dtype: np.dtype,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Merge out-of-boundary rows onto sentinel rows, one per distinct
        combination of the `sentinel_preserves` columns.

        Each out-of-boundary row takes the sentinel as its destination but
        keeps the metadata named by `sentinel_preserves`; rows agreeing on
        those merge. Collapsing them all onto one row would erase the
        distinctions a downstream draw may condition on, and a draw restricted
        to travelling work modes would then find no sentinel row at all and
        silently renormalise the out-of-area mass onto in-world destinations.
        Grouping on EVERY column is the opposite failure: destination-unique
        columns make the collapse a no-op and the sentinel count tracks the
        out-of-boundary row count.

        The same collapse serves both routes to a sentinel — the `outside`
        policy, and a `redistribute` origin with no valid destinations. The
        latter is just the case where every row of the origin is out, so there
        is no reason for it to invent anything the former does not.

        Returns the sentinel rows' (destination codes, masses, metadata).
        """
        group_metadata, group_masses = _group_by_metadata(
            {k: outside_metadata[k] for k in self._sentinel_preserves},
            outside_likelihoods,
        )
        # Non-preserved columns keep their (dense) slot but carry no meaning on
        # a sentinel row; lookup() omits them rather than reporting a
        # placeholder as if it were data.
        sentinel_metadata = {
            k: (group_metadata[k] if k in group_metadata
                else np.zeros(len(group_masses), dtype=v.dtype))
            for k, v in outside_metadata.items()
        }
        sentinel_dest_codes = np.full(len(group_masses), outside_code, dtype=dest_dtype)
        return sentinel_dest_codes, group_masses, sentinel_metadata

    def _boundary_units(self, geo_units: set) -> set:
        """Narrow the loaded geo units to the level destinations are declared at.

        Every file entry must agree on `destination_level`; a source mixing
        levels in one destination column has no single answer here. When the
        caller supplied no level breakdown (a direct programmatic load rather
        than a World run) the flat set stands in.
        """
        levels = {fc.get('destination_level') for fc in self._file_configs}
        levels.discard(None)
        if not levels:
            return geo_units
        if len(levels) > 1:
            raise ValueError(
                f"O-D source '{self.name}': files disagree on 'destination_level' "
                f"({sorted(levels)}). All destinations in one source must live at "
                "the same hierarchy level."
            )
        level = levels.pop()
        if not self.geo_units_by_level:
            return geo_units
        if level not in self.geo_units_by_level:
            raise ValueError(
                f"O-D source '{self.name}': destination_level='{level}' is not a "
                f"level in the loaded geography ({sorted(self.geo_units_by_level)})."
            )
        return self.geo_units_by_level[level]

    def _apply_boundary_policy(self, geo_units: Optional[set]):
        """
        Resolve destinations that fall outside the loaded world boundary
        according to the configured `out_of_boundary` policy.

        Boundary membership is "destination value is among the loaded geo units
        AT `destination_level`". Scoping to the declared level matters because
        names repeat across levels: a census O-D table lists whole-country
        catch-alls ("Wales", "England") next to real LADs, and an unscoped check
        would call "Wales" in-boundary whenever any Welsh unit is loaded, then
        hand a non-existent LAD to the steps that sample within it.

        With no geo_units (an unbounded / whole-world run) there is no boundary,
        so the policy is a no-op.

        Operates on the columnar arrays: a single vectorised membership check
        over all rows, then a per-origin-group loop over numpy slices
        (mirrors the original per-origin Python-list loop structure, just
        without a Python tuple/dict per row).
        """
        if not geo_units:
            return

        # Narrow to the level destinations are declared at BEFORE the membership
        # test: names repeat across levels, so an unscoped set calls a
        # whole-country catch-all in-boundary.
        geo_units = self._boundary_units(geo_units)

        vocab_size = len(self._origin_code_table)
        dest_codes = self._dest_codes
        likelihoods = self._likelihoods
        metadata_arrays = self._metadata_arrays
        group_starts = self._group_starts
        group_stops = self._group_stops

        dest_in_boundary_by_code = np.array(
            [d in geo_units for d in self._dest_code_table.decode], dtype=bool
        )
        row_in_boundary = (dest_in_boundary_by_code[dest_codes]
                            if len(dest_codes) else np.empty(0, dtype=bool))

        # Sentinel destination code, allocated lazily (only if the 'outside'
        # outcome can actually occur) — never fed back into row_in_boundary,
        # so it's safe to add to dest_code_table mid-pass.
        outside_code: Optional[int] = None

        def get_outside_code() -> int:
            nonlocal outside_code
            if outside_code is None:
                outside_code = int(self._dest_code_table.encode_series(
                    pd.Series([self._outside_value]))[0])
                self._outside_dest_code = outside_code
            return outside_code

        error_offenders: Dict[str, List[str]] = {}
        origins_with_no_valid_destinations: List[str] = []
        origins_with_out = 0
        dropped_options = 0
        out_mass_total = 0.0
        outside_origins = 0

        new_dest_chunks: List[np.ndarray] = []
        new_likelihood_chunks: List[np.ndarray] = []
        new_metadata_chunks: Dict[str, List[np.ndarray]] = {k: [] for k in metadata_arrays}
        new_counts = np.zeros(vocab_size, dtype=np.int64)

        def append_rows(code: int, dest_slice: np.ndarray, lik_slice: np.ndarray,
                        meta_slices: Dict[str, np.ndarray]) -> None:
            new_dest_chunks.append(dest_slice)
            new_likelihood_chunks.append(lik_slice)
            for k in metadata_arrays:
                new_metadata_chunks[k].append(meta_slices[k])
            new_counts[code] = len(dest_slice)

        for code in range(vocab_size):
            start, stop = int(group_starts[code]), int(group_stops[code])
            if stop <= start:
                continue
            origin_str = self._origin_code_table[code]
            in_b = row_in_boundary[start:stop]

            if in_b.all():
                append_rows(code, dest_codes[start:stop], likelihoods[start:stop],
                            {k: v[start:stop] for k, v in metadata_arrays.items()})
                continue

            origins_with_out += 1
            out_mass = float(likelihoods[start:stop][~in_b].sum())
            out_mass_total += out_mass

            if self._out_of_boundary == 'error':
                error_offenders[origin_str] = [
                    self._dest_code_table[c] for c in dest_codes[start:stop][~in_b]
                ]
                append_rows(code, dest_codes[start:stop], likelihoods[start:stop],
                            {k: v[start:stop] for k, v in metadata_arrays.items()})
            elif self._out_of_boundary == 'redistribute':
                in_total = float(likelihoods[start:stop][in_b].sum())
                if in_total <= 0:
                    # Every destination of this origin is out-of-boundary, so
                    # nothing was dropped in favour of an in-world alternative
                    # — there was none. Counting these rows as dropped would
                    # overstate the discarded options.
                    origins_with_no_valid_destinations.append(origin_str)
                    if self._on_empty == 'outside':
                        # Every row is out-of-boundary, so this is the 'outside'
                        # policy's case with the in-boundary side empty: take
                        # the same collapse, carrying the same real metadata.
                        sentinel_dest_codes, sentinel_masses, sentinel_metadata = (
                            self._collapse_onto_sentinel(
                                {k: v[start:stop] for k, v in metadata_arrays.items()},
                                likelihoods[start:stop],
                                get_outside_code(), dest_codes.dtype,
                            )
                        )
                        append_rows(code, sentinel_dest_codes, sentinel_masses,
                                    sentinel_metadata)
                    # else: on_empty='error' — will raise below, leave empty.
                else:
                    dropped_options += int((~in_b).sum())
                    append_rows(
                        code, dest_codes[start:stop][in_b],
                        likelihoods[start:stop][in_b] / in_total,
                        {k: v[start:stop][in_b] for k, v in metadata_arrays.items()},
                    )
                    # Probability a worker from this origin was bounced back in
                    # (= the out-of-boundary mass that got redistributed). Drives
                    # the per-person Bernoulli mark in the strategy.
                    self._redistributed_fraction[origin_str] = out_mass
            else:  # 'outside'
                outside_origins += 1
                sentinel_dest_codes, sentinel_masses, sentinel_metadata = (
                    self._collapse_onto_sentinel(
                        {k: v[start:stop][~in_b] for k, v in metadata_arrays.items()},
                        likelihoods[start:stop][~in_b],
                        get_outside_code(), dest_codes.dtype,
                    )
                )
                append_rows(
                    code,
                    np.concatenate([dest_codes[start:stop][in_b], sentinel_dest_codes]),
                    np.concatenate([likelihoods[start:stop][in_b], sentinel_masses]),
                    {k: np.concatenate([v[start:stop][in_b], sentinel_metadata[k]])
                     for k, v in metadata_arrays.items()},
                )

        if self._out_of_boundary == 'error' and error_offenders:
            sample = sorted({d for ds in error_offenders.values() for d in ds})[:15]
            raise ValueError(
                f"O-D source '{self.name}': out_of_boundary='error' but "
                f"{len(error_offenders)} origin(s) have out-of-boundary destinations "
                f"(e.g. {sample}). Set out_of_boundary to 'redistribute' or 'outside', "
                "or load those destinations into the world."
            )

        if self._out_of_boundary == 'redistribute':
            if origins_with_no_valid_destinations:
                destination_of_their_flow = (
                    f"their whole flow went to the sentinel '{self._outside_value}'"
                    if self._on_empty == 'outside' else "no flow survives for them")
                logger.warning(
                    f"  [out_of_boundary] {len(origins_with_no_valid_destinations)} "
                    f"origin(s) have NO in-boundary destination, so "
                    f"{destination_of_their_flow} (e.g. "
                    f"{origins_with_no_valid_destinations[:15]})."
                )
                if self._on_empty == 'error':
                    raise ValueError(
                        f"O-D source '{self.name}': "
                        f"{len(origins_with_no_valid_destinations)} origin(s) have "
                        f"no in-boundary destination and on_empty='error' "
                        f"(e.g. {origins_with_no_valid_destinations[:15]}). Switch "
                        "on_empty to 'outside', or shrink/extend the world."
                    )
            mean_pct = 100.0 * out_mass_total / origins_with_out if origins_with_out else 0.0
            logger.info(
                f"  [out_of_boundary=redistribute] dropped {dropped_options} out-of-boundary "
                f"destination option(s) across {origins_with_out} origin(s); mean "
                f"{mean_pct:.1f}% of flow redistributed inward per affected origin."
            )
        elif self._out_of_boundary == 'outside':
            logger.info(
                f"  [out_of_boundary=outside] {outside_origins} origin(s) route "
                f"out-of-boundary flow to sentinel '{self._outside_value}'."
            )

        self._dest_codes = (np.concatenate(new_dest_chunks) if new_dest_chunks
                             else np.empty(0, dtype=np.int32))
        self._likelihoods = (np.concatenate(new_likelihood_chunks) if new_likelihood_chunks
                              else np.empty(0, dtype=np.float64))
        self._metadata_arrays = {
            k: (np.concatenate(v) if v else np.empty(0, dtype=metadata_arrays[k].dtype))
            for k, v in new_metadata_chunks.items()
        }
        self._group_stops = np.cumsum(new_counts)
        self._group_starts = self._group_stops - new_counts

    def lookup(self, origin: str) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Look up possible destinations for a given origin.

        Args:
            origin: Origin code (e.g., SGU code)

        Returns:
            List of (destination, metadata, likelihood) tuples
        """
        if not self._data_loaded:
            raise RuntimeError(
                f"Data not loaded for source '{self.name}'. No fallbacks."
            )
        code = self._origin_code_table.code_of(origin)
        if code is None:
            raise KeyError(
                f"Source '{self.name}' has no destinations for origin '{origin}'. "
                "No fallbacks. The O-D matrix must cover every origin."
            )
        start, stop = int(self._group_starts[code]), int(self._group_stops[code])
        result = []
        for i in range(start, stop):
            dest = self._dest_code_table[int(self._dest_codes[i])]
            likelihood = float(self._likelihoods[i])
            # Every sentinel row absorbed real out-of-boundary flow, so it
            # carries that flow's preserved metadata. Only the preserved
            # columns mean anything there; the rest are omitted, not filled, so
            # nothing downstream can mistake a placeholder for a value —
            # outside_value names a destination and never a metadata value.
            is_sentinel = (self._outside_dest_code is not None
                           and int(self._dest_codes[i]) == self._outside_dest_code)
            metadata = {}
            for meta_key, arr in self._metadata_arrays.items():
                if is_sentinel and meta_key not in self._sentinel_preserves:
                    continue
                value = arr[i]
                if meta_key in self._metadata_code_tables:
                    value = self._metadata_code_tables[meta_key][int(value)]
                elif hasattr(value, 'item'):
                    value = value.item()
                metadata[meta_key] = value
            result.append((dest, metadata, likelihood))
        return result

class GUSamplerSource(DataSource):
    """
    Data source for sampling geographical units within a parent GU weighted by distribution.
    Generic source that works with any geographical hierarchy level.

    Returns GU codes as categorical values with distribution-based weights.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize geographical unit sampler source."""
        super().__init__(name, config)
        # Lookup: parent_gu_name -> {child_gu_code: weight}
        self._lookup: Dict[str, Dict[str, float]] = {}
        self._file_configs = config.get('files', [])
        # Person attribute that supplies the parent GU to sample within, read
        # from config (key_columns value), so the sampler is generic over any
        # parent attribute / hierarchy level.
        self._parent_attribute: Optional[str] = None

    def load_data(self, geo_units: Optional[set] = None):
        """Load GU distribution by parent GU."""
        logger.info(f"Loading data for source '{self.name}'...")

        for file_config in self._file_configs:
            file_path = Path(pr.resolve(file_config['path']))

            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)

                    # Parent-GU lookup key: canonical one-entry key_columns mapping.
                    # Its value names the person attribute that supplies the parent GU,
                    # which keeps this generic over any parent attribute.
                    parent_column = _ordered_key_columns(file_config, self.name, expected=1)[0]
                    key_resolution = file_config['key_columns'][parent_column]
                    if not isinstance(key_resolution, dict) or not key_resolution.get('attribute'):
                        raise ValueError(
                            f"GU sampler source '{self.name}': key column '{parent_column}' "
                            "must map to a resolution with an 'attribute' naming the person "
                            "attribute that holds the parent GU, e.g. "
                            f"{{{parent_column}: {{attribute: workplace_location}}}}."
                        )
                    self._parent_attribute = key_resolution['attribute']
                    weight_column = file_config.get('weight_column', 'Total')

                    # The sampled child-GU output column (distinct from the lookup
                    # key above), format: {name: ..., level: ...}. `level` is the
                    # user-facing label, used only for logging.
                    geo_unit_config = file_config.get('geographical_unit_column')
                    if geo_unit_config:
                        geo_unit_column = geo_unit_config.get('name')
                        geo_unit_level = geo_unit_config.get('level')

                    # Filter to only relevant geographical units
                    if geo_units and geo_unit_column and geo_unit_column in df.columns:
                        original_len = len(df)
                        df = df[df[geo_unit_column].isin(geo_units)]
                        logger.info(f"  Filtered CSV from {original_len} to {len(df)} rows based on {len(geo_units)} geographical units")

                    # Handle exclude_rows (list format)
                    exclude_rows_config = file_config.get('exclude_rows', [])
                    if isinstance(exclude_rows_config, list):
                        # format: [{column: "col", values: [vals]}]
                        for exclude_rule in exclude_rows_config:
                            col = exclude_rule.get('column')
                            exclude_values = exclude_rule.get('values', [])
                            if col and col in df.columns:
                                df = df[~df[col].isin(exclude_values)]

                    # Group by parent GU and build child GU distribution
                    file_lookup = {}
                    for parent_name, group in df.groupby(parent_column):
                        geo_dist = {}
                        for _, row in group.iterrows():
                            geo_code = row[geo_unit_column]
                            weight = float(row[weight_column])
                            if weight > 0:  # Only include GUs with workers
                                geo_dist[geo_code] = weight

                        # Normalize to probabilities
                        if geo_dist:
                            file_lookup[parent_name] = self._normalize_probabilities(geo_dist)

                    logger.info(f"  ✓ Loaded {geo_unit_level} distributions for {len(file_lookup)} parent GUs from {file_path.name}")

                except Exception as e:
                    # Fail loud on a load/parse error.
                    raise RuntimeError(
                        f"failed to load data source file {file_path}: {e}"
                    ) from e
                _merge_disjoint(self._lookup, file_lookup, self.name, file_path)
            else:
                raise FileNotFoundError(f"data source file not found: {file_path}")

        self._data_loaded = True

    def lookup(self, person, household=None, context=None) -> Dict[str, float]:
        """
        Look up the child-GU distribution for the person's parent GU.

        Resolves the key itself: the parent GU is the person's already
        assigned `workplace_location`.
        """
        if not self._data_loaded:
            raise RuntimeError(
                f"Data not loaded for source '{self.name}'. No fallbacks."
            )
        parent_gu_name = get_person_attribute(person, self._parent_attribute)
        if not parent_gu_name:
            raise KeyError(
                f"Source '{self.name}': person {person.id} has no '{self._parent_attribute}' "
                "to sample a child GU within. No fallbacks."
            )
        if parent_gu_name not in self._lookup:
            raise KeyError(
                f"Source '{self.name}' has no child-GU distribution for parent "
                f"'{parent_gu_name}'. No fallbacks."
            )
        return self._lookup[parent_gu_name]


class DataSourceManager:
    """
    Manager for all data sources.

    Coordinates loading and access to multiple data sources.
    """

    def __init__(self, config):
        """
        Initialize data source manager.

        Args:
            config: AttributeAssignmentConfig instance
        """
        self.config = config
        self.sources: Dict[str, DataSource] = {}
        self._initialize_sources()

    # csv_lookup `format` → source class. Chosen explicitly in config.
    _CSV_FORMATS = {
        'geo_distribution': GeoDistributionSource,
        'diversity': DiversitySource,
        'pair': PairProbabilitySource,
        'multi_key': MultiKeyLookupSource,
        'origin_destination_matrix': OriginDestinationMatrixSource,
        'gu_sampler': GUSamplerSource,
    }

    def _initialize_sources(self):
        """Initialize data sources from config (explicit type/format dispatch)."""
        for source_name, source_config in self.config.data_sources.items():
            source_type = source_config.type

            if source_type == 'constant':
                logger.debug(f"Skipping constant source: {source_name}")
                continue
            if source_type != 'csv_lookup':
                raise ValueError(
                    f"Data source '{source_name}': unknown type '{source_type}' "
                    "(expected 'csv_lookup')."
                )

            fmt = source_config.config.get('format')
            cls = self._CSV_FORMATS.get(fmt)
            if cls is None:
                raise ValueError(
                    f"Data source '{source_name}' needs an explicit 'format' "
                    f"(one of {sorted(self._CSV_FORMATS)}), got {fmt!r}."
                )

            # MultiKeyLookupSource needs the assignment config for key/category resolution.
            if cls is MultiKeyLookupSource:
                self.sources[source_name] = cls(source_name, source_config.config, self.config)
            else:
                self.sources[source_name] = cls(source_name, source_config.config)

    def load_all(self, geo_units: Optional[set] = None,
                 geo_units_by_level: Optional[Dict[str, set]] = None):
        """
        Load all data sources.

        Args:
            geo_units: Optional set of geographical unit codes to preload
            geo_units_by_level: The same names keyed by hierarchy level, for
                sources that declare which level their values live at.
        """
        logger.info("Loading all data sources...")
        for source_name, source in self.sources.items():
            source.geo_units_by_level = geo_units_by_level
            source.load_data(geo_units)
        logger.info("✓ All data sources loaded")

    def get_source(self, source_name: str) -> Optional[DataSource]:
        """Get a data source by name."""
        return self.sources.get(source_name)

    def lookup(self, source_name: str, *args, **kwargs) -> Dict[str, float]:
        """
        Look up probabilities from a data source.

        Args:
            source_name: Name of data source
            *args, **kwargs: Arguments to pass to source's lookup method

        Returns:
            Probability distribution
        """
        source = self.get_source(source_name)
        if not source:
            raise KeyError(
                f"Data source '{source_name}' is not registered. No fallbacks. "
                "Fix the source name in the config."
            )
        return source.lookup(*args, **kwargs)
